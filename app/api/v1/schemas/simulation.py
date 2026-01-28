"""Simulation-related schema definitions.

This module defines Pydantic models for simulation requests and responses,
including model configuration, population settings, interventions, and results.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TransitionConfig(BaseModel):
    """Configuration for a single transition between compartments.

    Attributes
    ----------
    source : str
        Source compartment name.
    target : str
        Target compartment name.
    kind : {'spontaneous', 'mediated'}
        Type of transition.
    params : str or list of str
        Parameter name for spontaneous transitions, or [rate_param, agent_compartment]
        for mediated transitions.
    """

    source: str = Field(..., description="Source compartment name")
    target: str = Field(..., description="Target compartment name")
    kind: Literal["spontaneous", "mediated"] = Field(
        ..., description="Type of transition: 'spontaneous' or 'mediated'"
    )
    params: str | list[str] = Field(
        ...,
        description="Parameter name (spontaneous) or [rate_param, agent_compartment] (mediated)",
    )


class ModelConfig(BaseModel):
    """Configuration for the epidemic model.

    Either ``preset`` must be specified, or both ``compartments`` and
    ``transitions`` must be provided for a custom model.

    Attributes
    ----------
    preset : {'SIR', 'SEIR', 'SIS'} or None
        Predefined model preset. If set, compartments and transitions are
        auto-configured.
    compartments : list of str or None
        List of compartment names (required if no preset).
    parameters : dict of {str: float}
        Model parameters (e.g., transmission_rate, recovery_rate).
    transitions : list of TransitionConfig or None
        Transition definitions (required if no preset).
    """

    preset: Literal["SIR", "SEIR", "SIS"] | None = Field(
        default=None,
        description="Predefined model preset. If set, compartments and transitions are auto-configured.",
    )
    compartments: list[str] | None = Field(
        default=None, description="List of compartment names (required if no preset)"
    )
    parameters: dict[str, float] = Field(
        default_factory=dict, description="Model parameters (e.g., transmission_rate, recovery_rate)"
    )
    transitions: list[TransitionConfig] | None = Field(
        default=None, description="Transition definitions (required if no preset)"
    )

    @model_validator(mode="after")
    def validate_model_config(self) -> "ModelConfig":
        """Validate that either preset or custom config is provided."""
        if self.preset is None and (self.compartments is None or self.transitions is None):
            raise ValueError(
                "Either 'preset' must be specified, or both 'compartments' and 'transitions' must be provided"
            )
        return self


class PopulationConfig(BaseModel):
    """Configuration for the population.

    Attributes
    ----------
    name : str
        Population name (e.g., 'United_States').
    contacts_source : str or None
        Contact matrix source: 'prem_2017', 'prem_2021', or 'mistry_2021'.
    layers : list of str or None
        Contact layers to include (e.g., ['home', 'work', 'school', 'community']).
    age_group_mapping : dict of {str: list of str} or None
        Custom age group aggregation mapping.
    """

    name: str = Field(..., description="Population name (e.g., 'United_States')")
    contacts_source: str | None = Field(
        default=None,
        description="Contact matrix source: 'prem_2017', 'prem_2021', or 'mistry_2021'",
    )
    layers: list[str] | None = Field(
        default=None,
        description="Contact layers to include (e.g., ['home', 'work', 'school', 'community'])",
    )
    age_group_mapping: dict[str, list[str]] | None = Field(
        default=None,
        description=(
            "Custom age group aggregation mapping. Keys are new group names, "
            "values are lists of source age groups to aggregate. "
            "For prem contacts (prem_2017, prem_2021), use 5-year groups: '0-4', '5-9', '10-14', etc. "
            "Example: {'0-19': ['0-4', '5-9', '10-14', '15-19'], '20-64': ['20-24', ...], '65+': ['65-69', '70-74', '75+']}"
        ),
    )


class SimulationConfig(BaseModel):
    """Configuration for simulation execution.

    Attributes
    ----------
    start_date : str
        Simulation start date (YYYY-MM-DD format).
    end_date : str
        Simulation end date (YYYY-MM-DD format).
    Nsim : int
        Number of simulations to run (1-1000).
    dt : float
        Time step in days.
    seed : int or None
        Random seed for reproducibility.
    resample_frequency : str
        Pandas frequency for resampling (D=daily, W=weekly, M=monthly).
    """

    start_date: str = Field(..., description="Simulation start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Simulation end date (YYYY-MM-DD)")
    Nsim: int = Field(default=100, ge=1, le=1000, description="Number of simulations to run")
    dt: float = Field(default=1.0, gt=0, description="Time step in days")
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility. If None, results will vary between runs."
    )
    resample_frequency: str = Field(
        default="D", description="Pandas frequency for resampling (D=daily, W=weekly, M=monthly)"
    )


class InitialConditionsConfig(BaseModel):
    """Configuration for initial conditions.

    Attributes
    ----------
    method : {'percentage', 'absolute'}
        Method for setting initial conditions.
    initial_percentages : dict of {str: float} or None
        Percentage of population in each compartment (for 'percentage' method).
        Keys are compartment names, values are percentages (0-100).
    compartments : dict of {str: list of float} or None
        Absolute counts per compartment per age group (for 'absolute' method).
    """

    method: Literal["percentage", "absolute"] = Field(
        default="percentage", description="Method for setting initial conditions"
    )
    initial_percentages: dict[str, float] | None = Field(
        default=None,
        description=(
            "Percentage of population in each compartment (for 'percentage' method). "
            "Keys are compartment names, values are percentages (0-100). "
            "Example: {'I': 0.01, 'R': 10.0}. Remaining population goes to first compartment."
        ),
    )
    compartments: dict[str, list[float]] | None = Field(
        default=None,
        description="Absolute counts per compartment per age group (for 'absolute' method)",
    )

    @model_validator(mode="after")
    def validate_initial_conditions(self) -> "InitialConditionsConfig":
        """Validate that absolute method has compartments specified."""
        if self.method == "absolute" and self.compartments is None:
            raise ValueError(
                "'compartments' must be provided when method is 'absolute'"
            )
        return self


class InterventionConfig(BaseModel):
    """Configuration for a contact reduction intervention.

    Attributes
    ----------
    layer_name : str
        Contact layer to modify (e.g., 'school', 'work').
    start_date : str
        Intervention start date (YYYY-MM-DD).
    end_date : str
        Intervention end date (YYYY-MM-DD).
    reduction_factor : float
        Factor to multiply contacts by (0.2 = reduce to 20% of normal).
    name : str or None
        Optional intervention name.
    """

    layer_name: str = Field(..., description="Contact layer to modify (e.g., 'school', 'work')")
    start_date: str = Field(..., description="Intervention start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Intervention end date (YYYY-MM-DD)")
    reduction_factor: float = Field(
        ...,
        ge=0,
        le=1,
        description="Factor to multiply contacts by (0.2 = reduce to 20% of normal)",
    )
    name: str | None = Field(default=None, description="Optional intervention name")


class ParameterOverrideConfig(BaseModel):
    """Configuration for a parameter override during a time period.

    Attributes
    ----------
    parameter_name : str
        Name of parameter to override.
    start_date : str
        Override start date (YYYY-MM-DD).
    end_date : str
        Override end date (YYYY-MM-DD).
    value : float
        New parameter value during this period.
    """

    parameter_name: str = Field(..., description="Name of parameter to override")
    start_date: str = Field(..., description="Override start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Override end date (YYYY-MM-DD)")
    value: float = Field(..., description="New parameter value during this period")


class SummaryConfig(BaseModel):
    """Configuration for summary statistics computation.

    Attributes
    ----------
    peak_compartments : list of str or None
        Compartments to compute peak statistics for.
    total_transitions : list of str or None
        Transitions to compute total counts for.
    """

    peak_compartments: list[str] | None = Field(
        default=None,
        description=(
            "Compartments to compute peak statistics for (e.g., ['I', 'Infected']). "
            "Returns peak value, CI, and peak date for each."
        ),
    )
    total_transitions: list[str] | None = Field(
        default=None,
        description=(
            "Transitions to compute total counts for (e.g., ['S_to_I', 'Susceptible_to_Infected']). "
            "Returns cumulative sum with median and CI for each."
        ),
    )


class OutputConfig(BaseModel):
    """Configuration for simulation output.

    Attributes
    ----------
    quantiles : list of float or None
        Quantiles to compute. Default uses epydemix default.
    include_trajectories : bool
        Whether to include raw trajectory data.
    compartments : list of str or None
        Compartments to include in output. Default: all.
    transitions : list of str or None
        Transitions to include in output. Default: all.
    age_groups : list of str or None
        Age groups to include. Default: all. Use 'total' for aggregated.
    summary : SummaryConfig or None
        Configuration for summary statistics.
    """

    quantiles: list[float] | None = Field(
        default=None,
        description="Quantiles to compute. Default: [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975] (from epydemix)",
    )
    include_trajectories: bool = Field(
        default=False, description="Include raw trajectory data (can be large)"
    )
    compartments: list[str] | None = Field(
        default=None,
        description="Compartments to include in output (e.g., ['Susceptible', 'Infected']). Default: all",
    )
    transitions: list[str] | None = Field(
        default=None,
        description="Transitions to include (e.g., ['Susceptible_to_Infected']). Default: all",
    )
    age_groups: list[str] | None = Field(
        default=None,
        description="Age groups to include (e.g., ['0-4', '5-19', 'total']). Default: all. Use 'total' for aggregated.",
    )
    summary: SummaryConfig | None = Field(
        default=None,
        description="Configuration for summary statistics. If None, no summary is computed.",
    )


class SimulationRequest(BaseModel):
    """Complete simulation request.

    Attributes
    ----------
    model : ModelConfig
        Epidemic model configuration.
    population : PopulationConfig
        Population configuration.
    simulation : SimulationConfig
        Simulation execution parameters.
    initial_conditions : InitialConditionsConfig or None
        Initial conditions configuration.
    interventions : list of InterventionConfig or None
        List of interventions to apply.
    parameter_overrides : list of ParameterOverrideConfig or None
        List of parameter overrides.
    output : OutputConfig or None
        Output configuration.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": {"preset": "SIR"},
                    "population": {"name": "United_States"},
                    "simulation": {
                        "start_date": "2024-01-01",
                        "end_date": "2024-03-01",
                        "Nsim": 100,
                    },
                }
            ]
        }
    }

    model: ModelConfig
    population: PopulationConfig
    simulation: SimulationConfig
    initial_conditions: InitialConditionsConfig | None = Field(default=None)
    interventions: list[InterventionConfig] | None = Field(default=None)
    parameter_overrides: list[ParameterOverrideConfig] | None = Field(default=None)
    output: OutputConfig | None = Field(default=None)


class CompartmentResults(BaseModel):
    """Results for compartment trajectories.

    Hierarchical structure organized by compartment name and age group.

    Attributes
    ----------
    dates : list of str
        Dates corresponding to values.
    data : dict
        Hierarchical data: ``{compartment: {age_group: {quantile: [values]}}}``.
    """

    dates: list[str] = Field(..., description="Dates corresponding to values")
    data: dict[str, dict[str, dict[str, list[float]]]] = Field(
        ...,
        description=(
            "Hierarchical data: {compartment: {age_group: {quantile: [values]}}}. "
            "Example: {'Infected': {'total': {'0.5': [...]}, '0-4': {'0.5': [...]}}}"
        ),
    )


class TransitionResults(BaseModel):
    """Results for transition counts.

    Hierarchical structure organized by transition name and age group.

    Attributes
    ----------
    dates : list of str
        Dates corresponding to values.
    data : dict
        Hierarchical data: ``{transition: {age_group: {quantile: [values]}}}``.
    """

    dates: list[str] = Field(..., description="Dates corresponding to values")
    data: dict[str, dict[str, dict[str, list[float]]]] = Field(
        ...,
        description=(
            "Hierarchical data: {transition: {age_group: {quantile: [values]}}}. "
            "Example: {'Susceptible_to_Infected': {'total': {'0.5': [...]}}}"
        ),
    )


class SummaryStatistic(BaseModel):
    """A summary statistic with median and confidence interval.

    Attributes
    ----------
    median : float
        Median value.
    ci_95 : list of float
        95% confidence interval [lower, upper].
    """

    median: float
    ci_95: list[float] = Field(..., min_length=2, max_length=2)


class PeakStatistic(BaseModel):
    """Peak statistic with date.

    Attributes
    ----------
    median : float
        Median peak value.
    ci_95 : list of float
        95% confidence interval [lower, upper].
    peak_date : str or None
        Date of the peak (from median trajectory).
    """

    median: float
    ci_95: list[float] = Field(..., min_length=2, max_length=2)
    peak_date: str | None = None


class SummaryResults(BaseModel):
    """Summary statistics of the simulation.

    Attributes
    ----------
    peaks : dict of {str: PeakStatistic} or None
        Peak statistics per compartment.
    totals : dict of {str: SummaryStatistic} or None
        Total transition counts.
    """

    peaks: dict[str, PeakStatistic] | None = Field(
        default=None,
        description="Peak statistics per compartment: {compartment_name: {median, ci_95, peak_date}}",
    )
    totals: dict[str, SummaryStatistic] | None = Field(
        default=None,
        description="Total transition counts: {transition_name: {median, ci_95}}",
    )


class TrajectoryData(BaseModel):
    """Raw trajectory data for a single simulation run.

    Hierarchical structure organized by name and age group.

    Attributes
    ----------
    compartments : dict
        Compartment values: ``{compartment: {age_group: [values]}}``.
    transitions : dict
        Transition counts: ``{transition: {age_group: [values]}}``.
    """

    compartments: dict[str, dict[str, list[float]]] = Field(
        ...,
        description="Compartment values: {compartment: {age_group: [values]}}",
    )
    transitions: dict[str, dict[str, list[float]]] = Field(
        ...,
        description="Transition counts: {transition: {age_group: [values]}}",
    )


class TrajectoriesResults(BaseModel):
    """Raw trajectories from all simulation runs.

    Attributes
    ----------
    dates : list of str
        Dates corresponding to values.
    runs : list of TrajectoryData
        Data for each simulation run.
    """

    dates: list[str] = Field(..., description="Dates corresponding to values")
    runs: list[TrajectoryData] = Field(..., description="Data for each simulation run")


class SimulationResultsData(BaseModel):
    """Container for all simulation results.

    Attributes
    ----------
    compartments : CompartmentResults
        Compartment trajectory quantiles.
    transitions : TransitionResults
        Transition count quantiles.
    summary : SummaryResults or None
        Summary statistics (if requested).
    trajectories : TrajectoriesResults or None
        Raw trajectory data (if requested).
    """

    compartments: CompartmentResults
    transitions: TransitionResults
    summary: SummaryResults | None = None
    trajectories: TrajectoriesResults | None = Field(
        default=None, description="Raw trajectory data (only if include_trajectories=true)"
    )


class SimulationMetadata(BaseModel):
    """Metadata about the simulation run.

    Attributes
    ----------
    model_preset : str or None
        Model preset used, if any.
    compartments : list of str
        Compartment names in the model.
    population_name : str
        Population used.
    population_size : int
        Total population size.
    n_age_groups : int
        Number of age groups.
    start_date : str
        Simulation start date.
    end_date : str
        Simulation end date.
    n_simulations : int
        Number of simulations run.
    dt : float
        Time step used.
    seed : int or None
        Random seed used.
    """

    model_preset: str | None = None
    compartments: list[str]
    population_name: str
    population_size: int
    n_age_groups: int
    start_date: str
    end_date: str
    n_simulations: int
    dt: float
    seed: int | None = None


class SimulationResponse(BaseModel):
    """Complete simulation response.

    Attributes
    ----------
    simulation_id : str
        Unique identifier for this simulation run.
    status : {'completed', 'failed'}
        Simulation status.
    metadata : SimulationMetadata
        Metadata about the simulation.
    results : SimulationResultsData or None
        Simulation results (None if failed).
    error : str or None
        Error message (if failed).
    """

    simulation_id: str
    status: Literal["completed", "failed"]
    metadata: SimulationMetadata
    results: SimulationResultsData | None = None
    error: str | None = None
