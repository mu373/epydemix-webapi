"""Population data service for accessing epydemix population information.

This module provides functions for listing available populations,
retrieving population details, and accessing contact matrices.
"""

import functools

import numpy as np
from epydemix.population.population import (
    Population,
    get_available_locations,
    load_epydemix_population,
)

from ..api.v1.schemas.population import (
    AgeGroupInfo,
    ContactMatrixResponse,
    PopulationDetail,
    PopulationListResponse,
    PopulationSummary,
)


@functools.lru_cache(maxsize=1)
def get_locations_df():
    """Get the locations dataframe from epydemix.

    Results are cached to avoid repeated network/disk fetches.

    Returns
    -------
    pd.DataFrame
        DataFrame containing available locations and their metadata.
    """
    return get_available_locations()


def list_populations() -> PopulationListResponse:
    """List all available populations.

    Retrieves summary information for all populations available in the
    epydemix data repository.

    Returns
    -------
    PopulationListResponse
        Response containing list of population summaries and total count.
    """
    df = get_locations_df()

    populations = []
    for _, row in df.iterrows():
        name = row["location"]
        # Parse available contact sources from the row if available
        available_sources = []
        for source in ["prem_2017", "prem_2021", "mistry_2021"]:
            if source in df.columns and row.get(source, False):
                available_sources.append(source)
        # If no sources found in columns, use default
        if not available_sources:
            # Check primary_contact_source column
            if "primary_contact_source" in df.columns:
                available_sources = [row["primary_contact_source"]]
            else:
                available_sources = ["mistry_2021"]

        populations.append(
            PopulationSummary(
                name=name,
                display_name=name.replace("_", " "),
                total_population=None,  # Would require loading full population
                n_age_groups=None,
                available_contact_sources=available_sources,
            )
        )

    return PopulationListResponse(populations=populations, total=len(populations))


@functools.lru_cache(maxsize=50)
def _load_population_cached(
    name: str, contacts_source: str | None = None
) -> Population:
    """Load a population with caching.

    Parameters
    ----------
    name : str
        Population name (e.g., 'United_States').
    contacts_source : str or None, optional
        Contact matrix source. If None, uses the default for the population.

    Returns
    -------
    Population
        Loaded epydemix Population object.
    """
    return load_epydemix_population(
        population_name=name,
        contacts_source=contacts_source,
    )


def get_population_detail(name: str, contacts_source: str | None = None) -> PopulationDetail:
    """Get detailed information about a population.

    Loads the population and returns comprehensive information including
    demographics, age groups, and available contact sources.

    Parameters
    ----------
    name : str
        Population name (e.g., 'United_States').
    contacts_source : str or None, optional
        Contact matrix source to use when loading. If None, uses default.

    Returns
    -------
    PopulationDetail
        Detailed population information including age groups and contact sources.

    Raises
    ------
    ValueError
        If the population name is not found.
    """
    pop = _load_population_cached(name, contacts_source)

    age_groups = [
        AgeGroupInfo(name=str(ag_name), population=int(pop_count))
        for ag_name, pop_count in zip(pop.Nk_names, pop.Nk)
    ]

    # Get available contact sources from locations df
    df = get_locations_df()
    row = df[df["location"] == name].iloc[0] if len(df[df["location"] == name]) > 0 else None

    available_sources = []
    default_source = None
    if row is not None:
        for source in ["prem_2017", "prem_2021", "mistry_2021"]:
            if source in df.columns and row.get(source, False):
                available_sources.append(source)
        if "primary_contact_source" in df.columns:
            default_source = row["primary_contact_source"]

    if not available_sources:
        available_sources = ["mistry_2021"]

    return PopulationDetail(
        name=name,
        display_name=name.replace("_", " "),
        total_population=int(pop.total_population),
        age_groups=age_groups,
        contact_sources=available_sources,
        default_contact_source=default_source,
        available_layers=pop.layers,
    )


def get_contact_matrices(
    name: str,
    contacts_source: str | None = None,
    layers: list[str] | None = None,
) -> ContactMatrixResponse:
    """Get contact matrices for a population.

    Returns contact matrices for specified layers as well as the combined
    overall contact matrix.

    Parameters
    ----------
    name : str
        Population name (e.g., 'United_States').
    contacts_source : str or None, optional
        Contact matrix source. If None, uses the default for the population.
    layers : list of str or None, optional
        Contact layers to include. If None, includes all available layers.

    Returns
    -------
    ContactMatrixResponse
        Contact matrices by layer and the combined overall matrix.

    Raises
    ------
    ValueError
        If the population name is not found.
    """
    pop = _load_population_cached(name, contacts_source)

    # Filter layers if specified
    layer_names = layers if layers else pop.layers
    matrices = {}
    for layer in layer_names:
        if layer in pop.contact_matrices:
            matrices[layer] = pop.contact_matrices[layer].tolist()

    # Compute overall matrix
    overall = None
    if matrices:
        overall_matrix = np.zeros_like(list(pop.contact_matrices.values())[0])
        for layer in layer_names:
            if layer in pop.contact_matrices:
                overall_matrix += pop.contact_matrices[layer]
        overall = overall_matrix.tolist()

    return ContactMatrixResponse(
        population_name=name,
        contact_source=contacts_source or "default",
        layers=matrices,
        overall=overall,
        age_groups=[str(ag) for ag in pop.Nk_names],
    )
