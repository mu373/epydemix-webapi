"""Simulation request schemas: model, population, run, output, and the top-level request."""

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, BeforeValidator, Field, model_validator

from .....presets import preset_names
from .transforms import ParameterTransformConfig

# Preset literal sourced from the registry so adding a preset doesn't require
# editing this file.
PresetName: TypeAlias = Literal[preset_names()]  # type: ignore[valid-type]


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
            '- Spontaneous: a single parameter name (the rate), e.g. `["recovery_rate"]`.\n'
            "- Mediated: a two-element list `[rate_param, mediating_compartment]`, "
            'e.g. `["transmission_rate", "I"]`.\n'
            "A single string is also accepted and will be wrapped into a list."
        ),
    )


class ModelConfig(BaseModel):
    """Epidemic model configuration. Use `preset` for a built-in model (SIR, SEIR, SIS), or provide `compartments`, `parameters`, and `transitions` for a custom model."""

    preset: PresetName | None = Field(
        default=None,
        description=(
            "Predefined model preset. Auto-configures compartments and transitions.\n"
            "- `SIR`: Susceptible-Infected-Recovered\n"
            "- `SEIR`: Susceptible-Exposed-Infected-Recovered\n"
            "- `SIS`: Susceptible-Infected-Susceptible\n"
            "- `V-SEIHR`: Vaccinated SEIHR with parallel unvaccinated/vaccinated "
            "compartments. Use together with the `vaccination` block.\n"
            "When using a preset, you can still override default parameter values via `parameters`."
        ),
    )
    compartments: list[str] | None = Field(
        default=None,
        description=(
            "List of compartment names for a custom model. Required if no preset.\n"
            '- Example: `["S", "E", "I", "R", "H"]`.'
        ),
    )
    parameters: dict[str, float | list[float] | str] = Field(
        default_factory=dict,
        description=(
            "Model parameters as key-value pairs. Each value is one of:\n"
            "- scalar `float`: uniform constant rate.\n"
            "- `list[float]`: age-varying (length must match resolved population age groups).\n"
            "- `str`: arithmetic expression over other parameter names, "
            'e.g. `"(1 - p_h) * gamma"`. Evaluated after scalars, age-varying '
            "values, and `parameter_transforms`, in dependency order. Only "
            "arithmetic operators are supported (`+`, `-`, `*`, `/`, `//`, "
            "`**`, `%`, unary `+`/`-`); no function calls, attribute access, "
            "subscripts, or comparisons. Source-parameter shapes propagate "
            "via numpy broadcasting, so a calculated parameter inherits "
            "time- or age-variation from its sources automatically. "
            "Calculated parameters cannot be the target of "
            "`parameter_transforms` (apply transforms to the source instead). "
            "Expressions can also reference reserved SCREAMING_SNAKE_CASE "
            "names derived from the model state, currently:\n"
            "- `CONTACT_MATRIX_EIGENVALUE_ALL`: dominant eigenvalue of the "
            "sum of all contact-matrix layers. Useful for R0 calibration, "
            'e.g. `{"transmission_rate": "R0 * gamma / CONTACT_MATRIX_EIGENVALUE_ALL"}`. '
            "User parameter names cannot collide with these reserved names.\n"
            "For presets, scalar/list values override the preset defaults. "
            "For custom models, all parameter names used in transitions must "
            "be defined here. Examples:\n"
            '- Scalar parameter: `{"transmission_rate": 0.3, "recovery_rate": 0.1}`.\n'
            '- Age-varying parameter: `{"transmission_rate": [0.35, 0.30, 0.25]}`.\n'
            '- Calculated parameter: `{"p_h": 0.05, "gamma": 0.1, '
            '"recovery_rate": "(1 - p_h) * gamma"}`.'
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


class BuiltinPopulationConfig(BaseModel):
    """Population loaded from the epydemix data repository (countries, regions, etc.)."""

    source: Literal["builtin"] = Field(
        default="builtin",
        description="Loads a published population by name from the epydemix data repo.",
    )
    name: str = Field(
        ...,
        description="Population name in epydemix, e.g. `United_States`. It should be one of the population available in `GET /populations` endpoint.",
    )
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
            '- Example: `{"0-19": ["0-4", "5-9", "10-14", "15-19"], "65+": ["65-69", "70-74", "75+"]}`.'
        ),
    )


class CustomPopulationConfig(BaseModel):
    """Fully custom population."""

    source: Literal["custom"] = Field(
        ...,
        description="Defines age groups and contact matrices inline.",
    )
    name: str = Field(
        default="Custom Population",
        description="Display label for this population.",
    )
    age_groups: dict[str, int] = Field(
        ...,
        description=(
            "Age group label to population count. Insertion order defines the row/column "
            "order used by every entry in `contact_matrices`.\n"
            '- Homogeneous (single group): `{"A": 100000}`.\n'
            '- Two groups: `{"A": 100, "B": 100}`.'
        ),
    )
    contact_matrices: dict[str, list[list[float]]] = Field(
        ...,
        description=(
            "Contact matrices keyed by layer name. The keys define the layer set; each "
            "matrix must be square with one row/column per `age_groups` entry, in the same "
            'order. Layer name `"overall"` is reserved by epydemix and rejected.\n'
            '- Homogeneous (1x1): `{"all": [[1.0]]}`.\n'
            '- Two groups (2x2): `{"all": [[0.2, 0.3], [0.3, 0.2]]}`.'
        ),
    )

    @model_validator(mode="after")
    def _validate_shapes(self) -> "CustomPopulationConfig":
        if not self.age_groups:
            raise ValueError("'age_groups' must contain at least one entry")
        if not self.contact_matrices:
            raise ValueError("'contact_matrices' must contain at least one layer")
        n = len(self.age_groups)
        for layer, matrix in self.contact_matrices.items():
            if layer == "overall":
                raise ValueError(
                    "'overall' is a reserved layer name in epydemix; use a different "
                    "name (e.g. 'all')"
                )
            if len(matrix) != n:
                raise ValueError(
                    f"contact_matrices['{layer}'] has {len(matrix)} rows but "
                    f"age_groups has {n} entries"
                )
            for i, row in enumerate(matrix):
                if len(row) != n:
                    raise ValueError(
                        f"contact_matrices['{layer}'] row {i} has length {len(row)} "
                        f"but expected {n} (square matrix)"
                    )
        return self


PopulationConfig: TypeAlias = Annotated[
    BuiltinPopulationConfig | CustomPopulationConfig,
    Field(discriminator="source"),
]


class SimulationConfig(BaseModel):
    """Simulation execution parameters."""

    start_date: str = Field(..., description="Start date in `YYYY-MM-DD` format.")
    end_date: str = Field(..., description="End date in `YYYY-MM-DD` format.")
    Nsim: int = Field(default=10, ge=1, le=1000, description="Number of simulation runs.")
    dt: float = Field(default=1.0, gt=0, description="Time step in days.")
    seed: int | None = Field(default=None, description="Random seed for reproducibility.")
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
            '- Example: `{"I": 0.01, "R": 10.0}`.'
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
            raise ValueError("'compartments' must be provided when method is 'absolute'")
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
            'Pass a list to narrow the response, e.g. `["Infected"]`, '
            "or pass `[]` to explicitly skip returning this summary. "
            "Per-quantile peak values and the median-trajectory peak date are returned for each included age group."
        ),
    )
    total_transitions: list[str] | None = Field(
        default=None,
        description=(
            "By default, cumulative totals are returned for every transition. "
            'Pass a list to narrow the response, e.g. `["Susceptible_to_Infected"]`, '
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
        default=False,
        description="Include raw per-run trajectory data in the response. Can be large.",
    )
    include_parameters: bool = Field(
        default=False,
        description=(
            "Include the effective per-step parameter arrays under `results.parameters`. "
            "Useful for plotting parameters such as `transmission_rate` after balcan/scale/override "
            "transforms have been applied. Off by default."
        ),
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
            'e.g. `["0-4", "5-19", "total"]`. Default: all age groups plus `total`.'
        ),
    )
    summary: SummaryConfig | None = Field(
        default=None,
        description="Summary statistics configuration. Omit to return the default summary.",
    )


class SimulationRequest(BaseModel):
    """Complete simulation request."""

    model: ModelConfig = Field(..., description="Epidemic model configuration.")
    population: PopulationConfig = Field(
        ...,
        description="Population configuration. It can load preset population from epydemix, or a custom population defined inline.",
    )
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
        default=None,
        description="Output configuration. Defaults to all compartments/transitions with standard quantiles.",
    )

    @model_validator(mode="before")
    @classmethod
    def _default_population_source(cls, data: Any) -> Any:
        # Pydantic discriminated unions require the discriminator field to be
        # present. Pre-existing payloads use `{"population": {"name": "..."}}`
        # without a `source` field; treat those as the builtin branch so the
        # schema change is non-breaking.
        if isinstance(data, dict):
            pop = data.get("population")
            if isinstance(pop, dict) and "source" not in pop:
                data = {**data, "population": {**pop, "source": "builtin"}}
        return data
