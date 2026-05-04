"""Epydemix model wrappers and simulation orchestration.

This module provides functions for creating and configuring epydemix models,
and orchestrating the simulation workflow from request to response.
"""

import uuid

import numpy as np
import pandas as pd
from epydemix.model.epimodel import EpiModel
from epydemix.model.predefined_models import load_predefined_model
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
from ..utils.parameter_transforms import (
    apply_transform_to_parameter,
    compute_transform_array,
)
from .population_service import _resolve_contacts_source
from .results_processing import process_results

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


def create_model(config: ModelConfig) -> tuple[EpiModel, dict[str, list[float]]]:
    """Create an EpiModel from configuration.

    Splits parameters into scalar and list-typed groups. Scalars are wired in
    immediately (presets seed defaults via the preset constructor, then user
    scalars upsert via `add_parameter`); list (age-varying) values are returned
    to the caller so they can be added after `setup_population` has run, since
    epydemix needs `population.num_groups` for shape checks.

    Parameters
    ----------
    config : ModelConfig
        Model configuration containing either a preset name or custom
        compartments and transitions.

    Returns
    -------
    tuple of (EpiModel, dict)
        The model with scalar parameters applied, plus a dictionary of any
        age-varying parameters that still need to be added once the
        population is set.
    """
    raw = config.parameters or {}
    scalar_params: dict[str, float] = {
        k: float(v) for k, v in raw.items() if not isinstance(v, list)
    }
    list_params: dict[str, list[float]] = {k: v for k, v in raw.items() if isinstance(v, list)}

    if config.preset:
        # Load with the preset's own defaults; user scalars then override them.
        # Whichever scalars the preset doesn't reference are still added to
        # model.parameters but stay unused, matching prior behavior.
        model: EpiModel = load_predefined_model(config.preset)  # type: ignore[assignment]
        for name, value in scalar_params.items():
            model.add_parameter(parameter_name=name, value=value)
        return model, list_params

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

    return model, list_params


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


def apply_parameter_transforms(
    model: EpiModel,
    transforms: list[ParameterTransformConfig] | None,
    simulation_config: SimulationConfig,
) -> None:
    """Apply parameter transforms (`balcan` / `scale` / `override`) to the model.

    Multiplicative transforms (`balcan`, `scale`) are composed in the order
    the user supplied them and written back to `model.parameters`. Override
    transforms are applied last via `model.override_parameter` and live in
    `model.overrides`, so they always win for their date window. All target
    parameter names are validated up front so a typo surfaces as a clean error.

    Mutates `model` in place.
    """
    if not transforms:
        return

    for transform in transforms:
        if transform.target_parameter not in model.parameters:
            raise ValueError(
                f"parameter_transforms[*].target_parameter '{transform.target_parameter}' is not defined in model.parameters"
            )

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

    # Overrides last; epydemix stores these in model.overrides separately.
    # Defensive copy of list values so the model does not retain a reference
    # to the request-body list. Per-age-group lists are reshaped to (1, N) so
    # epydemix interprets them as age-varying rather than as a (too-short)
    # time series.
    n_groups = model.population.num_groups
    for transform in overrides:
        if isinstance(transform.value, list):
            if len(transform.value) != n_groups:
                raise ValueError(
                    f"parameter_transforms[*].value for '{transform.target_parameter}' has length "
                    f"{len(transform.value)} but population has {n_groups} age groups"
                )
            value = np.array(transform.value).reshape(1, n_groups)
        else:
            value = transform.value
        model.override_parameter(
            start_date=transform.start_date,
            end_date=transform.end_date,
            parameter_name=transform.target_parameter,
            value=value,
        )


def extract_parameter_results(
    model: EpiModel, simulation_config: SimulationConfig
) -> ParameterResults:
    """Build the per-step effective parameter arrays for the response.

    Walks the same date grid the simulator uses (`compute_simulation_dates`),
    broadcasts each parameter in `model.parameters` to a `(T, N)` array, then
    bakes in any `model.overrides` windows (so the array reflects what actually
    drove the simulation, not just `model.parameters`).
    """
    dates = compute_simulation_dates(
        simulation_config.start_date,
        simulation_config.end_date,
        dt=simulation_config.dt,
    )
    T = len(dates)
    age_groups = [str(name) for name in model.population.Nk_names]
    N = len(age_groups)

    # Convert numpy datetime64 grid to ISO strings for the response.
    date_strs = [str(np.datetime_as_string(d, unit="D")) for d in dates]

    def _broadcast(value) -> np.ndarray:
        """Coerce a parameter's stored value (any of scalar / (T,) / (1,N) / (T,N)) to (T, N)."""
        if not hasattr(value, "__len__"):
            return np.full((T, N), float(value))
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1 and arr.shape[0] == T:
            return np.broadcast_to(arr[:, None], (T, N)).copy()
        if arr.ndim == 1 and arr.shape[0] == N:
            return np.broadcast_to(arr[None, :], (T, N)).copy()
        if arr.ndim == 2 and arr.shape == (1, N):
            return np.broadcast_to(arr, (T, N)).copy()
        if arr.ndim == 2 and arr.shape == (T, N):
            return arr.copy()
        raise ValueError(f"Cannot broadcast parameter array of shape {arr.shape} to (T={T}, N={N})")

    # Pre-compute per-date pandas timestamps for override-window comparison.
    date_ts = pd.to_datetime(date_strs)

    data: dict[str, dict[str, list[float]]] = {}
    for name, value in model.parameters.items():
        try:
            arr = _broadcast(value)
        except ValueError:
            # Unknown shape (e.g., some prior wrapped in np.array of dtype=object). Skip.
            continue

        # Apply overrides into the array. epydemix stores them in model.overrides
        # as a dict[name, list[{start_date, end_date, value}]].
        for override in model.overrides.get(name, []):
            start = pd.Timestamp(override["start_date"])
            end = pd.Timestamp(override["end_date"])
            mask = (date_ts >= start) & (date_ts <= end)
            ov_value = override["value"]
            if hasattr(ov_value, "__len__"):
                ov_arr = np.asarray(ov_value, dtype=float).reshape(-1)
                if ov_arr.shape[0] == N:
                    arr[mask, :] = ov_arr[None, :]
                else:
                    # Length doesn't match age groups; skip rather than guess.
                    continue
            else:
                arr[mask, :] = float(ov_value)

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
        # Create model (scalar params applied; age-varying params deferred until
        # population is set so num_groups is available for shape validation).
        model, list_params = create_model(request.model)

        # Setup population
        setup_population(model, request.population)

        # Add age-varying base parameters now that the population is resolved.
        apply_age_varying_parameters(model, list_params)

        # Apply interventions
        apply_interventions(model, request.interventions)

        # Apply parameter transforms (balcan / scale / override).
        apply_parameter_transforms(model, request.parameter_transforms, request.simulation)

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
