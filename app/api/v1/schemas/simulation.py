"""Simulation-related schema definitions.

This module defines Pydantic models for simulation requests and responses,
including model configuration, population settings, interventions, and results.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TransitionConfig(BaseModel):
    """A single transition between compartments."""

    source: str = Field(..., description="Source compartment name.")
    target: str = Field(..., description="Target compartment name.")
    kind: Literal["spontaneous", "mediated"] = Field(
        ..., description="Type of transition: `spontaneous` or `mediated`."
    )
    params: str | list[str] = Field(
        ...,
        description=(
            "Parameter name(s) for this transition.\n\n"
            "- **Spontaneous**: a single parameter name, e.g. `\"gamma\"`\n"
            "- **Mediated**: `[rate_param, agent_compartment]`, e.g. `[\"beta\", \"I\"]`"
        ),
    )


class ModelConfig(BaseModel):
    """Epidemic model configuration.

    Specify `preset` for a built-in model, or provide both `compartments` and `transitions` for a custom model.
    """

    preset: Literal["SIR", "SEIR", "SIS"] | None = Field(
        default=None,
        description="Predefined model preset (`SIR`, `SEIR`, `SIS`). Auto-configures compartments and transitions.",
    )
    compartments: list[str] | None = Field(
        default=None, description="Compartment names. Required if no preset."
    )
    parameters: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Model parameters as key-value pairs.\n\n"
            "Example:\n"
            "```json\n"
            "{\"transmission_rate\": 0.3, \"recovery_rate\": 0.1}\n"
            "```"
        ),
    )
    transitions: list[TransitionConfig] | None = Field(
        default=None, description="Transition definitions. Required if no preset."
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
        description=(
            "Contact matrix source.\n\n"
            "Options: `prem_2017`, `prem_2021`, `mistry_2021`."
        ),
    )
    layers: list[str] | None = Field(
        default=None,
        description=(
            "Contact layers to include.\n\n"
            "Options: `home`, `work`, `school`, `community`."
        ),
    )
    age_group_mapping: dict[str, list[str]] | None = Field(
        default=None,
        description=(
            "Custom age group aggregation.\n\n"
            "Keys are new group names, values are lists of source age groups to merge.\n\n"
            "Example:\n"
            "```json\n"
            "{\n"
            "  \"0-19\": [\"0-4\", \"5-9\", \"10-14\", \"15-19\"],\n"
            "  \"20-64\": [\"20-24\", \"25-29\", ...],\n"
            "  \"65+\": [\"65-69\", \"70-74\", \"75+\"]\n"
            "}\n"
            "```"
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
        description=(
            "Resampling frequency.\n\n"
            "- `D` - daily\n"
            "- `W` - weekly\n"
            "- `M` - monthly"
        ),
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
            "Remainder goes to the first compartment.\n\n"
            "Example:\n"
            "```json\n"
            "{\"I\": 0.01, \"R\": 10.0}\n"
            "```"
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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": {
                        "preset": "SIR",
                        "parameters": {"transmission_rate": 0.3, "recovery_rate": 0.1},
                    },
                    "population": {"name": "United_States"},
                    "simulation": {
                        "start_date": "2024-01-01",
                        "end_date": "2024-03-01",
                        "Nsim": 10,
                    },
                }
            ]
        }
    }

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
        description=(
            "Nested structure:\n\n"
            "```\n"
            "compartment -> age_group -> quantile -> [values]\n"
            "```"
        ),
    )


class TransitionResults(BaseModel):
    """Transition count quantiles."""

    dates: list[str] = Field(..., description="Dates corresponding to values.")
    data: dict[str, dict[str, dict[str, list[float]]]] = Field(
        ...,
        description=(
            "Nested structure:\n\n"
            "```\n"
            "transition -> age_group -> quantile -> [values]\n"
            "```"
        ),
    )


class SummaryStatistic(BaseModel):
    """A summary statistic with median and confidence interval."""

    median: float
    ci_95: list[float] = Field(..., min_length=2, max_length=2)


class PeakStatistic(BaseModel):
    """Peak statistic with date."""

    median: float
    ci_95: list[float] = Field(..., min_length=2, max_length=2)
    peak_date: str | None = None


class SummaryResults(BaseModel):
    """Summary statistics of the simulation."""

    peaks: dict[str, PeakStatistic] | None = Field(
        default=None,
        description="Peak statistics per compartment.",
    )
    totals: dict[str, SummaryStatistic] | None = Field(
        default=None,
        description="Total transition counts.",
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


class SimulationMetadata(BaseModel):
    """Metadata about the simulation run."""

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
    """Complete simulation response."""

    simulation_id: str
    status: Literal["completed", "failed"]
    metadata: SimulationMetadata
    results: SimulationResultsData | None = None
    error: str | None = None
