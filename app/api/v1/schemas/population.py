"""Population-related schema definitions.

This module defines Pydantic models for population data, contact matrices,
and model presets.
"""

from pydantic import BaseModel, Field


class PopulationSummary(BaseModel):
    """Summary information for a population."""

    name: str = Field(..., description="Population identifier (e.g., 'United_States')")
    display_name: str = Field(..., description="Human-readable name")
    total_population: int | None = Field(default=None, description="Total population size")
    n_age_groups: int | None = Field(default=None, description="Number of age groups")
    available_contact_sources: list[str] = Field(
        default_factory=list, description="Available contact matrix sources"
    )


class PopulationListResponse(BaseModel):
    """Response for listing all available populations."""

    populations: list[PopulationSummary]
    total: int


class ContactMatrixInfo(BaseModel):
    """Information about a contact matrix."""

    layer: str = Field(..., description="Contact layer name")
    shape: list[int] = Field(..., description="Matrix dimensions [rows, cols]")
    mean_contacts: float = Field(..., description="Mean number of contacts")


class PopulationDetail(BaseModel):
    """Detailed information about a population."""

    name: str = Field(..., description="Population identifier.")
    display_name: str = Field(..., description="Human-readable name.")
    total_population: int = Field(..., description="Total population size.")
    age_groups: dict[str, int] = Field(
        ...,
        description="Age group label to population count, e.g. `{\"0-4\": 18608139}`. Keys are in model (age-ascending) order.",
        examples=[{"0-4": 18608139, "5-19": 63540783, "20-49": 132780169, "50-64": 63172279, "65+": 60019216}],
    )
    contact_sources: list[str] = Field(..., description="Available contact matrix sources.")
    default_contact_source: str | None = Field(default=None, description="Default contact source for this population.")
    available_layers: list[str] = Field(..., description="Available contact layers (e.g. home, work, school, community).")


class ContactMatrixResponse(BaseModel):
    """Response containing contact matrices for a population."""

    population_name: str = Field(..., description="Population identifier.")
    contact_source: str = Field(..., description="Contact matrix source used.")
    layers: dict[str, list[list[float]]] = Field(
        ..., description="Contact matrices by layer name"
    )
    overall: list[list[float]] | None = Field(
        default=None, description="Combined contact matrix across all layers"
    )
    age_groups: list[str] = Field(..., description="Age group labels for matrix indices")
    spectral_radius: dict[str, float] = Field(
        default_factory=dict,
        description="Spectral radius (largest eigenvalue) for each layer and overall",
    )


class PresetInfo(BaseModel):
    """Information about a predefined epidemic model."""

    name: str = Field(..., description="Preset name (e.g., 'SIR', 'SEIR')")
    description: str = Field(..., description="Description of the model")
    compartments: list[str] = Field(..., description="Compartment names")
    parameters: dict[str, float] = Field(..., description="Default parameter values")
    transitions: list[dict] = Field(..., description="Transition definitions")


class PresetsListResponse(BaseModel):
    """Response listing all available model presets."""

    presets: list[PresetInfo]
