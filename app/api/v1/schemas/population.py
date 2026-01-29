"""Population-related schema definitions.

This module defines Pydantic models for population data, contact matrices,
and model presets.
"""

from pydantic import BaseModel, Field


class PopulationSummary(BaseModel):
    """Summary information for a population.

    Attributes
    ----------
    name : str
        Population identifier (e.g., 'United_States').
    display_name : str
        Human-readable name (e.g., 'United States').
    total_population : int or None
        Total population size, if available.
    n_age_groups : int or None
        Number of age groups, if available.
    available_contact_sources : list of str
        Available contact matrix sources for this population.
    """

    name: str = Field(..., description="Population identifier (e.g., 'United_States')")
    display_name: str = Field(..., description="Human-readable name")
    total_population: int | None = Field(default=None, description="Total population size")
    n_age_groups: int | None = Field(default=None, description="Number of age groups")
    available_contact_sources: list[str] = Field(
        default_factory=list, description="Available contact matrix sources"
    )


class PopulationListResponse(BaseModel):
    """Response for listing all available populations.

    Attributes
    ----------
    populations : list of PopulationSummary
        List of population summaries.
    total : int
        Total number of populations.
    """

    populations: list[PopulationSummary]
    total: int


class AgeGroupInfo(BaseModel):
    """Information about a single age group.

    Attributes
    ----------
    name : str
        Age group name (e.g., '0-4', '5-9').
    population : int
        Population count in this age group.
    """

    name: str = Field(..., description="Age group name (e.g., '0-4', '5-9')")
    population: int = Field(..., description="Population count in this age group")


class ContactMatrixInfo(BaseModel):
    """Information about a contact matrix.

    Attributes
    ----------
    layer : str
        Contact layer name (e.g., 'home', 'work').
    shape : list of int
        Matrix dimensions [rows, cols].
    mean_contacts : float
        Mean number of contacts per person.
    """

    layer: str = Field(..., description="Contact layer name")
    shape: list[int] = Field(..., description="Matrix dimensions [rows, cols]")
    mean_contacts: float = Field(..., description="Mean number of contacts")


class PopulationDetail(BaseModel):
    """Detailed information about a population.

    Attributes
    ----------
    name : str
        Population identifier.
    display_name : str
        Human-readable name.
    total_population : int
        Total population size.
    age_groups : list of AgeGroupInfo
        Information about each age group.
    contact_sources : list of str
        Available contact matrix sources.
    default_contact_source : str or None
        Default contact source for this population.
    available_layers : list of str
        Available contact layers (e.g., home, work, school, community).
    """

    name: str
    display_name: str
    total_population: int
    age_groups: list[AgeGroupInfo]
    contact_sources: list[str]
    default_contact_source: str | None = None
    available_layers: list[str]


class ContactMatrixResponse(BaseModel):
    """Response containing contact matrices for a population.

    Attributes
    ----------
    population_name : str
        Population identifier.
    contact_source : str
        Contact matrix source used.
    layers : dict of {str: list of list of float}
        Contact matrices by layer name, each as a 2D list.
    overall : list of list of float or None
        Combined contact matrix across all layers.
    age_groups : list of str
        Age group labels corresponding to matrix indices.
    spectral_radius : dict of {str: float}
        Spectral radius (largest eigenvalue) for each layer and overall matrix.
        The spectral radius is related to the basic reproduction number (R0).
    """

    population_name: str
    contact_source: str
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
    """Information about a predefined epidemic model.

    Attributes
    ----------
    name : str
        Preset name (e.g., 'SIR', 'SEIR', 'SIS').
    description : str
        Human-readable description of the model.
    compartments : list of str
        Names of compartments in the model.
    parameters : dict of {str: float}
        Default parameter values.
    transitions : list of dict
        Transition definitions between compartments.
    """

    name: str = Field(..., description="Preset name (e.g., 'SIR', 'SEIR')")
    description: str = Field(..., description="Description of the model")
    compartments: list[str] = Field(..., description="Compartment names")
    parameters: dict[str, float] = Field(..., description="Default parameter values")
    transitions: list[dict] = Field(..., description="Transition definitions")


class PresetsListResponse(BaseModel):
    """Response listing all available model presets.

    Attributes
    ----------
    presets : list of PresetInfo
        List of available model presets.
    """

    presets: list[PresetInfo]
