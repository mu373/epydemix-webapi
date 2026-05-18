"""Epydemix model wrappers and simulation orchestration.

This module provides functions for creating and configuring epydemix models,
and orchestrating the simulation workflow from request to response.
"""

import uuid

import numpy as np
from epydemix.model.epimodel import EpiModel
from epydemix.population.population import Population, load_epydemix_population
from epydemix.utils.utils import compute_simulation_dates

from ..api.v1.schemas.simulation import (
    CustomPopulationConfig,
    InitialConditionsConfig,
    InterventionConfig,
    ModelConfig,
    ModelMetadata,
    ParameterResults,
    ParameterTransformConfig,
    PopulationConfig,
    PopulationMetadata,
    SimulationConfig,
    SimulationMetadata,
    SimulationRequest,
    SimulationResponse,
    SimulationRunMetadata,
)
from ..presets import PRESETS
from ..utils.calculated_parameters import (
    RESERVED_NAMES,
    compute_reserved_params,
    evaluate_expressions,
)
from ..utils.parameter_conversions import resolve_parameter_conversions
from ..utils.parameter_transforms import (
    apply_transform_to_parameter,
    broadcast_to_time_and_age,
    compute_transform_array,
    window_mask_for_dates,
)
from .population_service import _resolve_contacts_source
from .results_processing import process_results
from .vaccination_service import apply_vaccinations

DEFAULT_LAYERS = ["home", "work", "school", "community"]


def _build_population_metadata(
    request_population: PopulationConfig,
    model_population,
) -> PopulationMetadata:
    """Build PopulationMetadata from the request and the loaded model population."""
    age_groups = {
        str(name): int(count) for name, count in zip(model_population.Nk_names, model_population.Nk)
    }
    contact_matrices = {
        str(layer): np.asarray(matrix, dtype=float).tolist()
        for layer, matrix in (model_population.contact_matrices or {}).items()
    }
    if isinstance(request_population, CustomPopulationConfig):
        return PopulationMetadata(
            source="custom",
            name=request_population.name,
            contacts_source=None,
            layers=list(request_population.contact_matrices.keys()),
            age_group_mapping=None,
            total=int(model_population.total_population),
            age_groups=age_groups,
            contact_matrices=contact_matrices,
        )
    return PopulationMetadata(
        source="builtin",
        name=request_population.name,
        contacts_source=_resolve_contacts_source(
            request_population.name, request_population.contacts_source
        ),
        layers=request_population.layers or DEFAULT_LAYERS,
        age_group_mapping=request_population.age_group_mapping,
        total=int(model_population.total_population),
        age_groups=age_groups,
        contact_matrices=contact_matrices,
    )


def _build_run_metadata(request: SimulationRequest) -> SimulationRunMetadata:
    """Build SimulationRunMetadata from the request."""
    return SimulationRunMetadata(
        start_date=request.simulation.start_date,
        end_date=request.simulation.end_date,
        Nsim=request.simulation.Nsim,
        dt=request.simulation.dt,
        seed=request.simulation.seed,
        resample_frequency=request.simulation.resample_frequency,
    )


def create_model(
    config: ModelConfig,
) -> tuple[EpiModel, dict[str, list[float]], dict[str, str]]:
    """Create an EpiModel from configuration.

    Splits parameters into three groups by value type:

    - Scalars (``float`` / ``int``) are wired in immediately. For presets, the
      preset builder seeds defaults and applies user scalars on top; for
      custom models, scalars are wired straight into ``EpiModel`` at
      construction.
    - List values (age-varying) are returned for application after
      ``setup_population`` has run, since epydemix needs
      ``population.num_groups`` for shape checks.
    - String values are expressions over other parameters and are returned
      for evaluation after ``parameter_transforms`` so transformed source
      shapes propagate through.

    For presets, any preset-specific calculated parameters (e.g. V-SEIHR's
    VE twins) are merged into ``expr_params``; user-supplied calc-params win
    on collision.
    """
    raw = config.parameters or {}
    scalar_params: dict[str, float] = {
        k: float(v)
        for k, v in raw.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    list_params: dict[str, list[float]] = {k: v for k, v in raw.items() if isinstance(v, list)}
    expr_params: dict[str, str] = {k: v for k, v in raw.items() if isinstance(v, str)}

    if config.preset:
        preset_def = PRESETS[config.preset]
        model, preset_calc_params = preset_def.build_model(scalar_params)
        # User calc-params win on collision (sensitivity scans, custom calibration).
        expr_params = {**preset_calc_params, **expr_params}
        return model, list_params, expr_params

    model = EpiModel(compartments=config.compartments, parameters=scalar_params)

    if config.transitions:
        for trans in config.transitions:
            params = trans.params
            # Spontaneous: single parameter name; mediated: tuple of (rate, compartment)
            if len(params) == 1:
                params = params[0]
            else:
                params = tuple(params)
            model.add_transition(
                source=trans.source,
                target=trans.target,
                kind=trans.kind,
                params=params,
            )

    return model, list_params, expr_params


def apply_age_varying_parameters(
    model: EpiModel,
    list_params: dict[str, list[float]],
) -> None:
    """Add length-N age-varying parameters after the population is set.

    Validates list length against `model.population.num_groups` and stores a
    fresh `np.array` so the model does not retain a reference to the
    request-supplied list. Mutates `model` in place.
    """
    if not list_params:
        return

    n_groups = model.population.num_groups
    for name, value in list_params.items():
        if len(value) != n_groups:
            raise ValueError(
                f"Parameter '{name}' has length {len(value)} but population has {n_groups} age groups"
            )
        # Epydemix represents age-varying parameters as 2D arrays of shape (1, N).
        # A 1D shape (N,) is interpreted as a (too-short) time series.
        model.add_parameter(parameter_name=name, value=np.array(value).reshape(1, n_groups))


def setup_population(model: EpiModel, config: PopulationConfig) -> None:
    """Load and set population for the model.

    For builtin populations, loads from the epydemix data repository.
    For custom populations, builds a Population in-memory from the inline
    `age_groups` dict and `contact_matrices` dict.

    Parameters
    ----------
    model : EpiModel
        EpiModel to configure with population data.
    config : BuiltinPopulationConfig or CustomPopulationConfig
        Population configuration. The discriminator selects the branch.
    """
    if isinstance(config, CustomPopulationConfig):
        # Insertion order of `age_groups` defines the contact-matrix row/col order.
        names = list(config.age_groups.keys())
        sizes = [float(config.age_groups[k]) for k in names]
        population = Population(name=config.name)
        population.add_population(Nk=sizes, Nk_names=names)
        for layer_name, matrix in config.contact_matrices.items():
            population.add_contact_matrix(
                contact_matrix=np.array(matrix, dtype=float),
                layer_name=layer_name,
            )
        model.set_population(population)
        return

    population = load_epydemix_population(
        population_name=config.name,
        contacts_source=config.contacts_source,
        layers=config.layers or DEFAULT_LAYERS,
        age_group_mapping=config.age_group_mapping,
    )
    model.set_population(population)


def create_initial_conditions(
    model: EpiModel, config: InitialConditionsConfig | None
) -> dict[str, np.ndarray] | None:
    """Create initial conditions dictionary from configuration.

    Builds initial conditions either from absolute counts per compartment
    or from percentages of the total population.

    Parameters
    ----------
    model : EpiModel
        EpiModel with population already set, needed for population counts.
    config : InitialConditionsConfig or None
        Initial conditions configuration. If None, epydemix defaults are used.

    Returns
    -------
    dict of {str: np.ndarray} or None
        Dictionary mapping compartment names to arrays of counts per age group,
        or None to use epydemix default initial conditions.
    """
    if config is None:
        return None

    if config.method == "absolute" and config.compartments:
        return {k: np.array(v) for k, v in config.compartments.items()}

    if config.method == "percentage" and config.initial_percentages:
        # Build initial conditions from percentages
        # Get population per age group
        pop_per_group = np.array(model.population.Nk)

        initial_conditions: dict[str, np.ndarray] = {}

        # Calculate counts for each specified compartment
        remaining_pop = pop_per_group.copy().astype(float)
        for comp_name, percentage in config.initial_percentages.items():
            # Distribute percentage proportionally across age groups
            comp_count = pop_per_group * (percentage / 100.0)
            initial_conditions[comp_name] = comp_count
            remaining_pop -= comp_count

        # Assign remaining population to first compartment (typically Susceptible)
        first_compartment = model.compartments[0]
        if first_compartment not in initial_conditions:
            initial_conditions[first_compartment] = remaining_pop
        else:
            initial_conditions[first_compartment] += remaining_pop

        return initial_conditions

    return None


def apply_interventions(model: EpiModel, interventions: list[InterventionConfig] | None) -> None:
    """Apply contact reduction interventions to the model.

    Adds interventions that modify contact rates in specific layers
    during specified time periods.

    Parameters
    ----------
    model : EpiModel
        EpiModel to configure with interventions.
    interventions : list of InterventionConfig or None
        List of intervention configurations. If None or empty, no
        interventions are applied.
    """
    if not interventions:
        return

    for intervention in interventions:
        model.add_intervention(
            layer_name=intervention.layer_name,
            start_date=intervention.start_date,
            end_date=intervention.end_date,
            reduction_factor=intervention.reduction_factor,
            name=intervention.name or "",
        )


def _apply_transforms_to_pass(
    model: EpiModel,
    transforms: list[ParameterTransformConfig],
    simulation_config: SimulationConfig,
) -> None:
    """Apply a list of transforms (already filtered to one pass) in place.

    Used by both the source-pass and the calc-pass. Multiplicative transforms
    (``balcan`` / ``scale``) compose in user-supplied order; ``override``
    transforms are applied last so they always win for their window. Each
    transform writes back to ``model.parameters`` via ``add_parameter``.

    Assumes every ``target_parameter`` is already validated to exist in
    ``model.parameters``.
    """
    if not transforms:
        return

    multiplicative = [t for t in transforms if t.method in ("balcan", "scale")]
    overrides = [t for t in transforms if t.method == "override"]

    # Multiplicative transforms compose in user-supplied order.
    # apply_transform_to_parameter always returns a fresh array, so writing
    # new_value back via add_parameter does not alias the previous value.
    for transform in multiplicative:
        existing = model.get_parameter(transform.target_parameter)
        transform_array = compute_transform_array(
            transform,
            simulation_config.start_date,
            simulation_config.end_date,
            simulation_config.dt,
        )
        new_value = apply_transform_to_parameter(existing, transform_array)
        model.add_parameter(parameter_name=transform.target_parameter, value=new_value)

    if not overrides:
        return

    # Overrides write into model.parameters as (T, N) arrays so calculated
    # parameters that reference the target pick up the override automatically
    # (the same way balcan/scale propagate through expressions).
    dates = compute_simulation_dates(
        simulation_config.start_date,
        simulation_config.end_date,
        dt=simulation_config.dt,
    )
    T = len(dates)
    n_groups = model.population.num_groups

    for transform in overrides:
        if isinstance(transform.value, list):
            if len(transform.value) != n_groups:
                raise ValueError(
                    f"parameter_transforms[*].value for '{transform.target_parameter}' has length "
                    f"{len(transform.value)} but population has {n_groups} age groups"
                )
            window_value: np.ndarray | float = np.asarray(transform.value, dtype=np.float64)
        else:
            window_value = float(transform.value)

        existing = model.get_parameter(transform.target_parameter)
        arr = broadcast_to_time_and_age(existing, T, n_groups)
        mask = window_mask_for_dates(transform.start_date, transform.end_date, dates)
        # Scalar broadcasts to (N,); 1D length-N broadcasts across the window's time slice.
        arr[mask, :] = window_value
        model.add_parameter(parameter_name=transform.target_parameter, value=arr)


def apply_parameter_transforms_sources(
    model: EpiModel,
    transforms: list[ParameterTransformConfig] | None,
    simulation_config: SimulationConfig,
    calculated_names: set[str] | None = None,
) -> None:
    """Apply transforms targeting **source** (non-calc-param) parameters.

    Validates target-parameter names against ``model.parameters`` (a typo
    surfaces as a clean error). Skips any transform whose target is a
    calculated parameter; those are deferred to
    ``apply_parameter_transforms_calc`` so they see post-eval values.
    """
    if not transforms:
        return

    calc_names = calculated_names or set()
    pending: list[ParameterTransformConfig] = []
    for transform in transforms:
        if transform.target_parameter in calc_names:
            continue  # deferred to the calc-pass
        if transform.target_parameter not in model.parameters:
            raise ValueError(
                f"parameter_transforms[*].target_parameter '{transform.target_parameter}' is not defined in model.parameters"
            )
        pending.append(transform)

    _apply_transforms_to_pass(model, pending, simulation_config)


def apply_parameter_transforms_calc(
    model: EpiModel,
    transforms: list[ParameterTransformConfig] | None,
    simulation_config: SimulationConfig,
    calculated_names: set[str] | None = None,
) -> None:
    """Apply transforms targeting **calculated** parameters.

    Runs after ``apply_calculated_parameters`` so each calc-param has its
    evaluated array stored on the model. Multiplicative transforms layer on
    top of the evaluated value; overrides replace it within the window. A
    transform on a source still propagates through any expression that
    references it via ``apply_calculated_parameters``; this pass enables an
    *additional* transform on the calc-param itself (e.g. a flat scale on
    ``transmission_rate_vax`` while ``balcan`` modulates ``transmission_rate``).
    """
    if not transforms:
        return

    calc_names = calculated_names or set()
    calc_targeting = [t for t in transforms if t.target_parameter in calc_names]
    if not calc_targeting:
        return

    for transform in calc_targeting:
        if transform.target_parameter not in model.parameters:
            raise ValueError(
                f"parameter_transforms[*].target_parameter '{transform.target_parameter}' is a calculated "
                f"parameter that was not evaluated; check the `parameters` block for the matching expression."
            )

    _apply_transforms_to_pass(model, calc_targeting, simulation_config)


def apply_calculated_parameters(
    model: EpiModel,
    expr_params: dict[str, str],
) -> None:
    """Evaluate expression-valued parameters and add the results to the model.

    Runs after scalars, age-varying values, and `parameter_transforms`, so
    each expression sees the post-transform shapes of its sources and numpy
    broadcasting carries time- and age-variation through naturally.

    Reserved names (e.g. ``CONTACT_MATRIX_EIGENVALUE_ALL``) are computed from
    the model state and injected into the eval namespace alongside
    ``model.parameters``. They are NOT stored on the model and so do not
    appear in ``results.parameters``.

    Mutates `model` in place. Raises `ValueError` (forwarded as 422) on
    syntax errors, disallowed AST nodes, undefined name references, or
    circular dependencies among expressions.
    """
    if not expr_params:
        return

    namespace: dict[str, object] = dict(model.parameters)
    namespace.update(compute_reserved_params(model))
    results = evaluate_expressions(expr_params, namespace)
    for name, value in results.items():
        model.add_parameter(parameter_name=name, value=value)


def extract_parameter_results(
    model: EpiModel, simulation_config: SimulationConfig
) -> ParameterResults:
    """Build the per-step effective parameter arrays for the response.

    Walks the same date grid the simulator uses (``compute_simulation_dates``)
    and broadcasts each parameter in ``model.parameters`` to a ``(T, N)``
    array. Transforms (including overrides) have already been baked into
    ``model.parameters`` upstream, so the returned arrays match exactly what
    the simulator runs with.
    """
    dates = compute_simulation_dates(
        simulation_config.start_date,
        simulation_config.end_date,
        dt=simulation_config.dt,
    )
    T = len(dates)
    age_groups = [str(name) for name in model.population.Nk_names]
    N = len(age_groups)

    date_strs = [str(np.datetime_as_string(d, unit="D")) for d in dates]

    data: dict[str, dict[str, list[float]]] = {}
    for name, value in model.parameters.items():
        try:
            arr = broadcast_to_time_and_age(value, T, N)
        except ValueError:
            # Unknown shape (e.g. wrapped object array). Skip rather than fail.
            continue
        data[name] = {age_groups[i]: arr[:, i].tolist() for i in range(N)}

    return ParameterResults(dates=date_strs, data=data)


def run_simulation(request: SimulationRequest) -> SimulationResponse:
    """Run an epidemic simulation based on the request configuration.

    This is the main orchestration function that executes the full
    simulation workflow:

    1. Creates and configures the epidemic model
    2. Loads and sets the population
    3. Applies interventions and parameter overrides
    4. Runs the stochastic simulations
    5. Processes and returns results

    Parameters
    ----------
    request : SimulationRequest
        Complete simulation request containing model, population,
        simulation parameters, and output configuration.

    Returns
    -------
    SimulationResponse
        Response containing simulation results with compartment and
        transition trajectories, metadata, and optional summary statistics.
        If an error occurs, returns a failed status with error message.
    """
    simulation_id = f"sim_{uuid.uuid4().hex[:12]}"

    try:
        # Reject any user parameter name that collides with a reserved
        # SCREAMING_SNAKE_CASE name (e.g. CONTACT_MATRIX_EIGENVALUE_ALL).
        # These are computed from the model state and injected into the
        # eval namespace by `apply_calculated_parameters`; allowing a
        # user override would silently mask the model-derived value.
        user_param_names = set((request.model.parameters or {}).keys())
        reserved_collisions = user_param_names & RESERVED_NAMES
        if reserved_collisions:
            raise ValueError(
                f"Parameter name(s) {sorted(reserved_collisions)} collide with "
                f"reserved name(s) injected by the calculated-parameter evaluator. "
                f"Reserved names are SCREAMING_SNAKE_CASE constants derived from "
                f"the model state and cannot be overridden."
            )

        # Create model (scalar params applied; age-varying and expression
        # params are deferred: the former until population is set, the latter
        # until after transforms so source shapes propagate through).
        model, list_params, expr_params = create_model(request.model)

        # Setup population
        setup_population(model, request.population)

        # Add age-varying base parameters now that the population is resolved.
        apply_age_varying_parameters(model, list_params)

        # Inject preset-scoped parameter conversions (period→rate, R0→β).
        # Custom models (no preset) opt out: pass {} so nothing is injected.
        preset_def = PRESETS[request.model.preset] if request.model.preset else None
        conversions = preset_def.parameter_conversions if preset_def else {}
        user_scalar_names = {
            k for k, v in (request.model.parameters or {}).items() if not isinstance(v, str)
        }
        converted = resolve_parameter_conversions(
            model.parameters, user_scalar_names, conversions
        )
        # User calc-params still win over registry-injected conversions on collision.
        expr_params = {**converted, **expr_params}

        # Apply interventions
        apply_interventions(model, request.interventions)

        # Source-pass transforms first (balcan/scale/override on non-calc names).
        calc_names = set(expr_params)
        apply_parameter_transforms_sources(
            model,
            request.parameter_transforms,
            request.simulation,
            calculated_names=calc_names,
        )

        # Evaluate calculated (expression) parameters now that all source
        # values have their final shapes.
        apply_calculated_parameters(model, expr_params)

        # Calc-pass transforms on top of evaluated calc-params.
        apply_parameter_transforms_calc(
            model,
            request.parameter_transforms,
            request.simulation,
            calculated_names=calc_names,
        )

        # Vaccination flow (source to vaccinated target). Mutates model in place;
        # no-op when the request has no `vaccination` block.
        apply_vaccinations(
            model,
            request.vaccination,
            request.simulation,
            request.model.preset,
        )

        # Create initial conditions
        initial_conditions = create_initial_conditions(model, request.initial_conditions)

        # Create random number generator from seed if provided
        rng = None
        if request.simulation.seed is not None:
            rng = np.random.default_rng(request.simulation.seed)

        # Run simulations
        results = model.run_simulations(
            start_date=request.simulation.start_date,
            end_date=request.simulation.end_date,
            Nsim=request.simulation.Nsim,
            dt=request.simulation.dt,
            initial_conditions_dict=initial_conditions,
            resample_frequency=request.simulation.resample_frequency,
            rng=rng,
        )

        # Process results
        results_data = process_results(results, request.output, model)

        # Optionally attach effective per-step parameter arrays for plotting.
        if request.output is not None and request.output.include_parameters:
            results_data.parameters = extract_parameter_results(model, request.simulation)

        # Build metadata
        metadata = SimulationMetadata(
            model=ModelMetadata(
                preset=request.model.preset,
                compartments=model.compartments,
            ),
            population=_build_population_metadata(request.population, model.population),
            simulation=_build_run_metadata(request),
            interventions=request.interventions,
            parameter_transforms=request.parameter_transforms,
            vaccination=request.vaccination,
        )

        return SimulationResponse(
            simulation_id=simulation_id,
            status="completed",
            metadata=metadata,
            results=results_data,
        )

    except ValueError:
        # Config validation errors propagate to the route as 422.
        raise
    except Exception as e:
        if isinstance(request.population, CustomPopulationConfig):
            pop_meta = PopulationMetadata(
                source="custom",
                name=request.population.name,
                contacts_source=None,
                layers=list(request.population.contact_matrices.keys()),
                age_group_mapping=None,
                total=0,
                age_groups={},
                contact_matrices=request.population.contact_matrices,
            )
        else:
            pop_meta = PopulationMetadata(
                source="builtin",
                name=request.population.name,
                contacts_source=request.population.contacts_source,
                layers=request.population.layers,
                age_group_mapping=request.population.age_group_mapping,
                total=0,
                age_groups={},
                contact_matrices={},
            )
        return SimulationResponse(
            simulation_id=simulation_id,
            status="failed",
            metadata=SimulationMetadata(
                model=ModelMetadata(
                    preset=request.model.preset,
                    compartments=[],
                ),
                population=pop_meta,
                simulation=_build_run_metadata(request),
            ),
            error=str(e),
        )
