"""Simulation request schemas: model, population, run, output, and the top-level request."""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, model_validator

from .transforms import ParameterTransformConfig


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
    parameters: dict[str, float | list[float]] = Field(
        default_factory=dict,
        description=(
            "Model parameters as key-value pairs. Each key is a parameter name "
            "referenced by transitions; the value is either a scalar rate or a "
            "list of one value per age group (age-varying). For age-varying "
            "values, the list length must match the resolved population's age groups.\n"
            "For presets, these override the default values. "
            "For custom models, all parameters used in transitions must be defined here.\n"
            "Example scalar: `{\"transmission_rate\": 0.3, \"recovery_rate\": 0.1}`.\n"
            "Example age-varying: `{\"transmission_rate\": [0.35, 0.30, 0.25]}`."
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


class SummaryConfig(BaseModel):
    """Configuration for summary statistics.

    Summary is returned by default for every compartment and transition, broken
    down by age group and by the quantiles requested in `output.quantiles`.
    Use the fields below to narrow that down."""

    peak_compartments: list[str] | None = Field(
        default=None,
        description=(
            "By default, peak statistics are returned for every compartment. "
            "Pass a list to narrow the response, e.g. `[\"Infected\"]`, "
            "or pass `[]` to explicitly skip returning this summary. "
            "Per-quantile peak values and the median-trajectory peak date are returned for each included age group."
        ),
    )
    total_transitions: list[str] | None = Field(
        default=None,
        description=(
            "By default, cumulative totals are returned for every transition. "
            "Pass a list to narrow the response, e.g. `[\"Susceptible_to_Infected\"]`, "
            "or pass `[]` to explicitly skip returning this summary. "
            "Per-quantile total event counts are returned for each included age group."
        ),
    )


class OutputConfig(BaseModel):
    """Output configuration. Everything is optional; defaults return all compartments, all transitions, all age groups (including `total`), all standard quantiles, and a populated `summary`."""

    quantiles: list[float] | None = Field(
        default=None,
        description=(
            "Quantiles to compute for trajectories and summary. "
            "Default: `[0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]`."
        ),
    )
    include_trajectories: bool = Field(
        default=False, description="Include raw per-run trajectory data in the response. Can be large."
    )
    compartments: list[str] | None = Field(
        default=None,
        description="Compartments to include in the trajectory section. Default: all. Does not affect `summary`.",
    )
    transitions: list[str] | None = Field(
        default=None,
        description="Transitions to include in the trajectory section. Default: all. Does not affect `summary`.",
    )
    age_groups: list[str] | None = Field(
        default=None,
        description=(
            "Age groups to include in both trajectories and summary, "
            "e.g. `[\"0-4\", \"5-19\", \"total\"]`. Default: all age groups plus `total`."
        ),
    )
    summary: SummaryConfig | None = Field(
        default=None,
        description="Summary statistics configuration. Omit to return the default summary.",
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
    parameter_transforms: list[ParameterTransformConfig] | None = Field(
        default=None,
        description=(
            "Parameter transforms applied to model parameters. Three methods are supported:\n"
            "- `balcan`: sinusoidal seasonality across the simulation timeline (multiplicative).\n"
            "- `scale`: multiplicative factor over a date window.\n"
            "- `override`: absolute replacement over a date window (scalar or per-age-group).\n"
            "Multiple transforms on the same parameter compose: `balcan` and `scale` "
            "stack multiplicatively in the order listed; `override` always wins for its "
            "date window, regardless of position in the list."
        ),
    )
    output: OutputConfig | None = Field(
        default=None, description="Output configuration. Defaults to all compartments/transitions with standard quantiles."
    )
