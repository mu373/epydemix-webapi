"""Population data service for accessing epydemix population information.

This module provides functions for listing available populations,
retrieving population details, and accessing contact matrices.
"""

import functools
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed

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
from ..config import settings

logger = logging.getLogger(__name__)


class PopulationLoadTimeoutError(Exception):
    """Raised when population loading exceeds the configured timeout."""

    def __init__(self, population_name: str, timeout: float):
        self.population_name = population_name
        self.timeout = timeout
        super().__init__(
            f"Loading population '{population_name}' timed out after {timeout}s"
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


# Cache for population metadata (total_population, n_age_groups)
# Populated by warm_cache() or on individual population loads
_population_metadata_cache: dict[str, dict] = {}


def list_populations() -> PopulationListResponse:
    """List all available populations.

    Retrieves summary information for all populations available in the
    epydemix data repository. If a population has been cached (e.g., via
    warm_cache), includes total_population and n_age_groups.

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

        # Get cached metadata if available
        metadata = _population_metadata_cache.get(name, {})
        total_pop = metadata.get("total_population")
        n_groups = metadata.get("n_age_groups")

        populations.append(
            PopulationSummary(
                name=name,
                display_name=str(name).replace("_", " "),
                total_population=total_pop,
                n_age_groups=n_groups,
                available_contact_sources=available_sources,
            )
        )

    return PopulationListResponse(populations=populations, total=len(populations))


@functools.lru_cache(maxsize=50)
def _load_population_cached_inner(name: str, contacts_source: str) -> Population:
    """Load a population with caching (internal).

    Parameters
    ----------
    name : str
        Population name (e.g., 'United_States').
    contacts_source : str
        Contact matrix source (must be resolved, not None).

    Returns
    -------
    Population
        Loaded epydemix Population object.
    """
    return load_epydemix_population(
        population_name=name,
        contacts_source=contacts_source,
    )


def _resolve_contacts_source(name: str, contacts_source: str | None) -> str:
    """Resolve contacts_source to actual value.

    Parameters
    ----------
    name : str
        Population name.
    contacts_source : str or None
        Contact source, or None to use default.

    Returns
    -------
    str
        Resolved contact source name.
    """
    if contacts_source is not None:
        return contacts_source
    # Get the default from locations.csv
    df = get_locations_df()
    row = df[df["location"] == name]
    if len(row) > 0 and "primary_contact_source" in df.columns:
        return row.iloc[0]["primary_contact_source"]
    return "mistry_2021"  # fallback default


def _load_population_cached(
    name: str,
    contacts_source: str | None = None,
    timeout: float | None = None,
) -> Population:
    """Load a population with caching and optional timeout.

    Normalizes contacts_source before caching to avoid duplicate cache entries
    for None vs explicit default value. Also populates the metadata cache.

    Parameters
    ----------
    name : str
        Population name (e.g., 'United_States').
    contacts_source : str or None, optional
        Contact matrix source. If None, uses the default for the population.
    timeout : float or None, optional
        Maximum time in seconds to wait for loading. If None, uses config default.
        Only applies to cache misses (cached loads are instant).

    Returns
    -------
    Population
        Loaded epydemix Population object.

    Raises
    ------
    PopulationLoadTimeoutError
        If loading exceeds the timeout.
    """
    resolved_source = _resolve_contacts_source(name, contacts_source)

    if timeout is None:
        timeout = settings.population_load_timeout

    # Use ThreadPoolExecutor to enforce timeout on the load
    # Note: If already cached, the load is instant so timeout won't trigger
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_load_population_cached_inner, name, resolved_source)
        try:
            pop = future.result(timeout=timeout)
        except FuturesTimeoutError:
            future.cancel()
            raise PopulationLoadTimeoutError(name, timeout)

    # Update metadata cache
    _population_metadata_cache[name] = {
        "total_population": int(pop.total_population),
        "n_age_groups": len(pop.Nk),
    }

    return pop


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


def _compute_spectral_radius(matrix: np.ndarray) -> float:
    """Compute the spectral radius of a matrix.

    The spectral radius is the largest absolute eigenvalue of the matrix.
    For contact matrices, this is related to the basic reproduction number (R0).

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to compute spectral radius for.

    Returns
    -------
    float
        The spectral radius (largest absolute eigenvalue).
    """
    eigenvalues = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigenvalues.real)))


def get_contact_matrices(
    name: str,
    contacts_source: str | None = None,
    layers: list[str] | None = None,
) -> ContactMatrixResponse:
    """Get contact matrices for a population.

    Returns contact matrices for specified layers as well as the combined
    overall contact matrix, along with spectral radii.

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
        Contact matrices by layer, combined overall matrix, and spectral radii.

    Raises
    ------
    ValueError
        If the population name is not found.
    """
    pop = _load_population_cached(name, contacts_source)

    # Filter layers if specified
    layer_names = layers if layers else pop.layers
    matrices = {}
    spectral_radii = {}

    for layer in layer_names:
        if layer in pop.contact_matrices:
            matrix = pop.contact_matrices[layer]
            matrices[layer] = matrix.tolist()
            spectral_radii[layer] = _compute_spectral_radius(matrix)

    # Compute overall matrix and its spectral radius
    overall = None
    if matrices:
        overall_matrix = np.zeros_like(list(pop.contact_matrices.values())[0])
        for layer in layer_names:
            if layer in pop.contact_matrices:
                overall_matrix += pop.contact_matrices[layer]
        overall = overall_matrix.tolist()
        spectral_radii["overall"] = _compute_spectral_radius(overall_matrix)

    return ContactMatrixResponse(
        population_name=name,
        contact_source=contacts_source or "default",
        layers=matrices,
        overall=overall,
        age_groups=[str(ag) for ag in pop.Nk_names],
        spectral_radius=spectral_radii,
    )


# Default populations to pre-warm on startup
DEFAULT_WARM_POPULATIONS = [
    "United_States",
    "Italy",
    "United_Kingdom",
    "Germany",
    "France",
    "Spain",
    "Canada",
    "Australia",
    "Japan",
    "Brazil",
]


def warm_cache(
    populations: list[str] | None = None,
    max_workers: int = 4,
) -> dict[str, bool]:
    """Pre-warm the population cache for faster subsequent requests.

    Loads specified populations in parallel to populate both the lru_cache
    and the metadata cache.

    Parameters
    ----------
    populations : list of str or None, optional
        Population names to warm. If None, uses DEFAULT_WARM_POPULATIONS.
    max_workers : int, optional
        Maximum number of concurrent threads for loading. Default is 4.

    Returns
    -------
    dict[str, bool]
        Dictionary mapping population names to success status.
    """
    if populations is None:
        populations = DEFAULT_WARM_POPULATIONS

    results: dict[str, bool] = {}

    def load_one(name: str) -> tuple[str, bool]:
        try:
            _load_population_cached(name, None)
            return (name, True)
        except Exception as e:
            logger.warning(f"Failed to warm cache for {name}: {e}")
            return (name, False)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_one, name): name for name in populations}
        for future in as_completed(futures):
            name, success = future.result()
            results[name] = success
            if success:
                logger.info(f"Warmed cache for {name}")

    return results


def get_cache_info() -> dict:
    """Get information about the current cache state.

    Returns
    -------
    dict
        Cache statistics including hits, misses, and cached populations.
    """
    cache_info = _load_population_cached_inner.cache_info()
    return {
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "maxsize": cache_info.maxsize,
        "currsize": cache_info.currsize,
        "cached_populations": list(_population_metadata_cache.keys()),
    }
