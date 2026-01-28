"""Epydemix model wrappers and simulation orchestration.

This module provides functions for creating and configuring epydemix models,
and orchestrating the simulation workflow from request to response.
"""

import uuid

import numpy as np
from epydemix.model.epimodel import EpiModel
from epydemix.model.predefined_models import load_predefined_model
from epydemix.population.population import load_epydemix_population

from ..api.v1.schemas.simulation import (
    InitialConditionsConfig,
    InterventionConfig,
    ModelConfig,
    ParameterOverrideConfig,
    PopulationConfig,
    SimulationMetadata,
    SimulationRequest,
    SimulationResponse,
)
from .results_processing import process_results


def create_model(config: ModelConfig) -> EpiModel:
    """Create an EpiModel from configuration.

    Creates either a predefined model (SIR, SEIR, SIS) with optional parameter
    overrides, or a fully custom model with user-defined compartments and
    transitions.

    Parameters
    ----------
    config : ModelConfig
        Model configuration containing either a preset name or custom
        compartments and transitions.

    Returns
    -------
    EpiModel
        Configured epydemix EpiModel instance ready for population setup.
    """
    if config.preset:
        # Use predefined model with parameter overrides
        params = config.parameters or {}
        model = load_predefined_model(
            config.preset,
            transmission_rate=params.get("transmission_rate", 0.3),
            recovery_rate=params.get("recovery_rate", 0.1),
            incubation_rate=params.get("incubation_rate", 0.2),
        )
        # Override any additional parameters
        for param_name, value in params.items():
            if param_name not in ["transmission_rate", "recovery_rate", "incubation_rate"]:
                model.add_parameter(parameter_name=param_name, value=value)
        return model

    # Create custom model
    model = EpiModel(compartments=config.compartments, parameters=config.parameters)

    # Add transitions
    if config.transitions:
        for trans in config.transitions:
            params = trans.params
            # Convert list to tuple if needed (for mediated transitions)
            if isinstance(params, list):
                params = tuple(params)
            model.add_transition(
                source=trans.source,
                target=trans.target,
                kind=trans.kind,
                params=params,
            )

    return model


def setup_population(model: EpiModel, config: PopulationConfig) -> None:
    """Load and set population for the model.

    Loads a population from the epydemix data repository and attaches it
    to the model.

    Parameters
    ----------
    model : EpiModel
        EpiModel to configure with population data.
    config : PopulationConfig
        Population configuration specifying location, contact source,
        and optional layer filtering.
    """
    population = load_epydemix_population(
        population_name=config.name,
        contacts_source=config.contacts_source,
        layers=config.layers or ["home", "work", "school", "community"],
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


def apply_parameter_overrides(
    model: EpiModel, overrides: list[ParameterOverrideConfig] | None
) -> None:
    """Apply time-varying parameter overrides to the model.

    Modifies parameter values during specified time periods, allowing
    for scenarios like changing transmission rates.

    Parameters
    ----------
    model : EpiModel
        EpiModel to configure with parameter overrides.
    overrides : list of ParameterOverrideConfig or None
        List of parameter override configurations. If None or empty,
        no overrides are applied.
    """
    if not overrides:
        return

    for override in overrides:
        model.override_parameter(
            start_date=override.start_date,
            end_date=override.end_date,
            parameter_name=override.parameter_name,
            value=override.value,
        )


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
        # Create model
        model = create_model(request.model)

        # Setup population
        setup_population(model, request.population)

        # Apply interventions
        apply_interventions(model, request.interventions)

        # Apply parameter overrides
        apply_parameter_overrides(model, request.parameter_overrides)

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

        # Build metadata
        metadata = SimulationMetadata(
            model_preset=request.model.preset,
            compartments=model.compartments,
            population_name=request.population.name,
            population_size=int(model.population.total_population),
            n_age_groups=model.population.num_groups,
            start_date=request.simulation.start_date,
            end_date=request.simulation.end_date,
            n_simulations=request.simulation.Nsim,
            dt=request.simulation.dt,
            seed=request.simulation.seed,
        )

        return SimulationResponse(
            simulation_id=simulation_id,
            status="completed",
            metadata=metadata,
            results=results_data,
        )

    except Exception as e:
        return SimulationResponse(
            simulation_id=simulation_id,
            status="failed",
            metadata=SimulationMetadata(
                compartments=[],
                population_name=request.population.name,
                population_size=0,
                n_age_groups=0,
                start_date=request.simulation.start_date,
                end_date=request.simulation.end_date,
                n_simulations=request.simulation.Nsim,
                dt=request.simulation.dt,
                seed=request.simulation.seed,
            ),
            error=str(e),
        )
