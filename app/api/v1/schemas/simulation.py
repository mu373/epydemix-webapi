"""Simulation-related schema definitions.

This module defines Pydantic models for simulation requests and responses,
including model configuration, population settings, interventions, and results.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, model_validator

from .population import AgeGroupInfo


def _ensure_list(v: str | list[str]) -> list[str]:
    """Accept a single string or a list of strings, always return a list."""
    if isinstance(v, str):
        return [v]
    return v


class TransitionConfig(BaseModel):
    """A single transition between compartments. Transitions define how individuals move between compartments, either spontaneously at a fixed rate or mediated by contact with another compartment."""

    source: str = Field(..., description="Source compartment name.")
    target: str = Field(..., description="Target compartment name.")
    kind: Literal["spontaneous", "mediated"] = Field(
        ...,
        description=(
            "Type of transition.\n"
            "- `spontaneous`: rate is a fixed parameter (e.g. recovery, incubation).\n"
            "- `mediated`: rate depends on the proportion of a mediating compartment "
            "(e.g. transmission driven by contact with infectious individuals)."
        ),
    )
    params: Annotated[list[str], BeforeValidator(_ensure_list)] = Field(
        ...,
        description=(
            "Parameter name(s) governing this transition.\n"
            "- Spontaneous: a single parameter name (the rate), e.g. `[\"recovery_rate\"]`.\n"
            "- Mediated: a two-element list `[rate_param, mediating_compartment]`, "
            "e.g. `[\"transmission_rate\", \"I\"]`.\n"
            "A single string is also accepted and will be wrapped into a list."
        ),
    )


class ModelConfig(BaseModel):
    """Epidemic model configuration. Use `preset` for a built-in model (SIR, SEIR, SIS), or provide `compartments`, `parameters`, and `transitions` for a custom model."""

    preset: Literal["SIR", "SEIR", "SIS"] | None = Field(
        default=None,
        description=(
            "Predefined model preset. Auto-configures compartments and transitions.\n"
            "- `SIR`: Susceptible-Infected-Recovered\n"
            "- `SEIR`: Susceptible-Exposed-Infected-Recovered\n"
            "- `SIS`: Susceptible-Infected-Susceptible\n"
            "When using a preset, you can still override default parameter values via `parameters`."
        ),
    )
    compartments: list[str] | None = Field(
        default=None,
        description=(
            "List of compartment names for a custom model. Required if no preset.\n"
            "Example: `[\"S\", \"E\", \"I\", \"R\", \"H\"]`."
        ),
    )
    parameters: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Model parameters as key-value pairs. Each key is a parameter name "
            "referenced by transitions, and the value is its rate.\n"
            "For presets, these override the default values. "
            "For custom models, all parameters used in transitions must be defined here.\n"
            "Example: `{\"transmission_rate\": 0.3, \"recovery_rate\": 0.1}`."
        ),
    )
    transitions: list[TransitionConfig] | None = Field(
        default=None,
        description=(
            "List of transitions between compartments. Required if no preset.\n"
            "Each transition defines how individuals move from one compartment to another."
        ),
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
    """Population configuration."""

    name: str = Field(..., description="Population name, e.g. `United_States`.")
    contacts_source: str | None = Field(
        default=None,
        description="Contact matrix source.\nOptions: `prem_2017`, `prem_2021`, `mistry_2021`.",
    )
    layers: list[str] | None = Field(
        default=None,
        description="Contact layers to include.\nOptions: `home`, `work`, `school`, `community`.",
    )
    age_group_mapping: dict[str, list[str]] | None = Field(
        default=None,
        description=(
            "Custom age group aggregation. Keys are new group names, values are lists of source age groups to merge.\n"
            "Example: `{\"0-19\": [\"0-4\", \"5-9\", \"10-14\", \"15-19\"], \"65+\": [\"65-69\", \"70-74\", \"75+\"]}`."
        ),
    )


class SimulationConfig(BaseModel):
    """Simulation execution parameters."""

    start_date: str = Field(..., description="Start date in `YYYY-MM-DD` format.")
    end_date: str = Field(..., description="End date in `YYYY-MM-DD` format.")
    Nsim: int = Field(default=10, ge=1, le=1000, description="Number of simulation runs.")
    dt: float = Field(default=1.0, gt=0, description="Time step in days.")
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility."
    )
    resample_frequency: str = Field(
        default="D",
        description="Resampling frequency. `D` = daily, `W` = weekly, `M` = monthly.",
    )


class InitialConditionsConfig(BaseModel):
    """Initial conditions for compartments."""

    method: Literal["percentage", "absolute"] = Field(
        default="percentage", description="Method: `percentage` or `absolute`."
    )
    initial_percentages: dict[str, float] | None = Field(
        default=None,
        description=(
            "Percentage of population in each compartment. "
            "Remainder goes to the first compartment.\n"
            "Example: `{\"I\": 0.01, \"R\": 10.0}`."
        ),
    )
    compartments: dict[str, list[float]] | None = Field(
        default=None,
        description="Absolute counts per compartment per age group. Required when method is `absolute`.",
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
    """A contact reduction intervention applied to a specific layer during a time period."""

    layer_name: str = Field(..., description="Contact layer to modify, e.g. `school`, `work`.")
    start_date: str = Field(..., description="Start date in `YYYY-MM-DD` format.")
    end_date: str = Field(..., description="End date in `YYYY-MM-DD` format.")
    reduction_factor: float = Field(
        ...,
        ge=0,
        le=1,
        description="Multiplier for contacts. `0.2` = reduce to 20% of normal.",
    )
    name: str | None = Field(default=None, description="Optional name for this intervention.")


class ParameterOverrideConfig(BaseModel):
    """Override a model parameter during a time period."""

    parameter_name: str = Field(..., description="Parameter to override.")
    start_date: str = Field(..., description="Start date in `YYYY-MM-DD` format.")
    end_date: str = Field(..., description="End date in `YYYY-MM-DD` format.")
    value: float = Field(..., description="Parameter value during this period.")


class SummaryConfig(BaseModel):
    """Configuration for summary statistics."""

    peak_compartments: list[str] | None = Field(
        default=None,
        description="Compartments to compute peak statistics for. Returns peak value, CI, and peak date.",
    )
    total_transitions: list[str] | None = Field(
        default=None,
        description="Transitions to compute cumulative totals for. Returns median and CI.",
    )


class OutputConfig(BaseModel):
    """Output configuration."""

    quantiles: list[float] | None = Field(
        default=None,
        description="Quantiles to compute. Default: `[0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]`.",
    )
    include_trajectories: bool = Field(
        default=False, description="Include raw trajectory data (can be large)."
    )
    compartments: list[str] | None = Field(
        default=None,
        description="Compartments to include in output. Default: all.",
    )
    transitions: list[str] | None = Field(
        default=None,
        description="Transitions to include in output. Default: all.",
    )
    age_groups: list[str] | None = Field(
        default=None,
        description="Age groups to include, e.g. `[\"0-4\", \"5-19\", \"total\"]`. Default: all.",
    )
    summary: SummaryConfig | None = Field(
        default=None,
        description="Summary statistics configuration.",
    )


class SimulationRequest(BaseModel):
    """Complete simulation request."""

    model: ModelConfig = Field(..., description="Epidemic model configuration.")
    population: PopulationConfig = Field(..., description="Population configuration.")
    simulation: SimulationConfig = Field(..., description="Simulation execution parameters.")
    initial_conditions: InitialConditionsConfig | None = Field(
        default=None, description="Initial conditions. Defaults to a small infected fraction."
    )
    interventions: list[InterventionConfig] | None = Field(
        default=None, description="Contact reduction interventions to apply."
    )
    parameter_overrides: list[ParameterOverrideConfig] | None = Field(
        default=None, description="Parameter overrides during specific time periods."
    )
    output: OutputConfig | None = Field(
        default=None, description="Output configuration. Defaults to all compartments/transitions with standard quantiles."
    )


class CompartmentResults(BaseModel):
    """Compartment trajectory quantiles."""

    dates: list[str] = Field(..., description="Dates corresponding to values.")
    data: dict[str, dict[str, dict[str, list[float]]]] = Field(
        ...,
        description="Nested structure: `compartment -> age_group -> quantile -> [values]`.",
    )


class TransitionResults(BaseModel):
    """Transition count quantiles."""

    dates: list[str] = Field(..., description="Dates corresponding to values.")
    data: dict[str, dict[str, dict[str, list[float]]]] = Field(
        ...,
        description="Nested structure: `transition -> age_group -> quantile -> [values]`.",
    )


class StatisticQuantiles(BaseModel):
    """Quantile values for a summary statistic, keyed by quantile string (e.g. `0.5`)."""

    quantiles: dict[str, float] = Field(
        ...,
        description="Quantile to value mapping, e.g. `{\"0.025\": ..., \"0.5\": ..., \"0.975\": ...}`.",
    )


class PeakStatistic(BaseModel):
    """Peak statistic for a compartment in one age group."""

    quantiles: dict[str, float] = Field(
        ...,
        description="Peak value per quantile, e.g. `{\"0.025\": ..., \"0.5\": ..., \"0.975\": ...}`.",
    )
    peak_date: str | None = Field(
        default=None,
        description="Date of the peak from the median trajectory.",
    )


class SummaryResults(BaseModel):
    """Summary statistics of the simulation."""

    peaks: dict[str, dict[str, PeakStatistic]] | None = Field(
        default=None,
        description="Peak statistics: `compartment -> age_group -> {quantiles, peak_date}`.",
    )
    totals: dict[str, dict[str, StatisticQuantiles]] | None = Field(
        default=None,
        description="Cumulative transition totals: `transition -> age_group -> {quantiles}`.",
    )


class TrajectoryData(BaseModel):
    """Raw trajectory data for a single simulation run."""

    compartments: dict[str, dict[str, list[float]]] = Field(
        ...,
        description="Compartment values: `{compartment: {age_group: [values]}}`.",
    )
    transitions: dict[str, dict[str, list[float]]] = Field(
        ...,
        description="Transition counts: `{transition: {age_group: [values]}}`.",
    )


class TrajectoriesResults(BaseModel):
    """Raw trajectories from all simulation runs."""

    dates: list[str] = Field(..., description="Dates corresponding to values.")
    runs: list[TrajectoryData] = Field(..., description="Data for each simulation run.")


class SimulationResultsData(BaseModel):
    """All simulation results."""

    compartments: CompartmentResults
    transitions: TransitionResults
    summary: SummaryResults | None = None
    trajectories: TrajectoriesResults | None = Field(
        default=None, description="Raw trajectory data. Only present if `include_trajectories` was true."
    )


class ModelMetadata(BaseModel):
    """Model section of simulation metadata. Mirrors the `model` section of the request."""

    preset: str | None = Field(default=None, description="Preset name if a preset was used.")
    compartments: list[str] = Field(..., description="Compartment names in the model.")


class PopulationMetadata(BaseModel):
    """Population section of simulation metadata. Mirrors the `population` section of the request and adds resolved/derived values."""

    name: str = Field(..., description="Population identifier.")
    contacts_source: str | None = Field(
        default=None,
        description="Resolved contact matrix source actually used.",
    )
    layers: list[str] | None = Field(
        default=None,
        description="Resolved contact layers actually used.",
    )
    age_group_mapping: dict[str, list[str]] | None = Field(
        default=None,
        description="Custom age group aggregation, echoed back if the request supplied one.",
    )
    total: int = Field(..., description="Total population size.")
    age_groups: list[AgeGroupInfo] = Field(
        ...,
        description="Population count per age group, in model order.",
    )


class SimulationRunMetadata(BaseModel):
    """Simulation section of metadata. Mirrors the `simulation` section of the request."""

    start_date: str = Field(..., description="Simulation start date.")
    end_date: str = Field(..., description="Simulation end date.")
    Nsim: int = Field(..., description="Number of simulation runs.")
    dt: float = Field(..., description="Time step in days.")
    seed: int | None = Field(default=None, description="Random seed used.")
    resample_frequency: str = Field(..., description="Resampling frequency.")


class SimulationMetadata(BaseModel):
    """Metadata about the simulation run, grouped to mirror the request shape."""

    model: ModelMetadata = Field(..., description="Model configuration used for the run.")
    population: PopulationMetadata = Field(..., description="Resolved population configuration and derived counts.")
    simulation: SimulationRunMetadata = Field(..., description="Simulation execution parameters used for the run.")


class SimulationResponse(BaseModel):
    """Complete simulation response."""

    simulation_id: str = Field(..., description="Unique identifier for this simulation run.")
    status: Literal["completed", "failed"] = Field(..., description="Whether the simulation completed successfully.")
    metadata: SimulationMetadata = Field(..., description="Metadata about the simulation run.")
    results: SimulationResultsData | None = Field(default=None, description="Simulation results. Null if status is `failed`.")
    error: str | None = Field(default=None, description="Error message if status is `failed`.")
