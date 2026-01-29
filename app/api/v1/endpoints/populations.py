"""Population API endpoints.

This module provides endpoints for listing populations, retrieving
population details, and accessing contact matrices.
"""

from fastapi import APIRouter, HTTPException, Query

from ....services import population_service
from ..schemas.population import (
    ContactMatrixResponse,
    PopulationDetail,
    PopulationListResponse,
)
from ..schemas.common import CacheInfoResponse

router = APIRouter()


@router.get(
    "",
    response_model=PopulationListResponse,
    summary="List available populations",
    description="Get a list of all available populations that can be used in simulations.",
)
async def get_populations() -> PopulationListResponse:
    """List all available populations.

    Returns a summary of each population including name and available
    contact sources from the epydemix data repository.

    Returns
    -------
    PopulationListResponse
        List of population summaries with total count.

    Raises
    ------
    HTTPException
        500 if population listing fails.
    """
    try:
        return population_service.list_populations()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list populations: {str(e)}")


@router.get(
    "/cache",
    response_model=CacheInfoResponse,
    summary="Get cache status",
    description="Get information about the population cache.",
)
async def get_cache_status() -> CacheInfoResponse:
    """Get population cache status.

    Returns cache statistics including hits, misses, and list of
    cached populations.

    Returns
    -------
    CacheInfoResponse
        Cache statistics and list of cached populations.
    """
    info = population_service.get_cache_info()
    return CacheInfoResponse(**info)


@router.get(
    "/{name}",
    response_model=PopulationDetail,
    summary="Get population details",
    description="Get detailed information about a specific population.",
)
async def get_population(
    name: str,
    contacts_source: str | None = Query(
        default=None,
        description="Contact matrix source (prem_2017, prem_2021, or mistry_2021)",
    ),
) -> PopulationDetail:
    """Get detailed information about a population.

    Returns demographic data including total population, age groups,
    available contact sources, and contact layers.

    Parameters
    ----------
    name : str
        Population identifier (e.g., 'United_States').
    contacts_source : str or None, optional
        Contact matrix source to use. If None, uses the default.

    Returns
    -------
    PopulationDetail
        Detailed population information.

    Raises
    ------
    HTTPException
        404 if population not found, 500 if retrieval fails.
    """
    try:
        return population_service.get_population_detail(name, contacts_source)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get population: {str(e)}")


@router.get(
    "/{name}/contacts",
    response_model=ContactMatrixResponse,
    summary="Get contact matrices",
    description="Get contact matrices for a specific population.",
)
async def get_contact_matrices(
    name: str,
    contacts_source: str | None = Query(
        default=None,
        description="Contact matrix source (prem_2017, prem_2021, or mistry_2021)",
    ),
    layers: list[str] | None = Query(
        default=None,
        description="Contact layers to include (e.g., home, work, school, community)",
    ),
) -> ContactMatrixResponse:
    """Get contact matrices for a population.

    Returns contact matrices for each specified layer (home, work, school,
    community) as well as the overall combined matrix.

    Parameters
    ----------
    name : str
        Population identifier (e.g., 'United_States').
    contacts_source : str or None, optional
        Contact matrix source. If None, uses the default.
    layers : list of str or None, optional
        Contact layers to include. If None, includes all layers.

    Returns
    -------
    ContactMatrixResponse
        Contact matrices by layer and combined overall matrix.

    Raises
    ------
    HTTPException
        404 if population not found, 500 if retrieval fails.
    """
    try:
        return population_service.get_contact_matrices(name, contacts_source, layers)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get contacts: {str(e)}")
