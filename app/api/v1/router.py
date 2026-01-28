"""API v1 router configuration.

This module aggregates all API v1 endpoint routers into a single router
that is mounted at the /api/v1 prefix.
"""

from fastapi import APIRouter

from .endpoints import populations, presets, simulations

router = APIRouter()

router.include_router(
    simulations.router,
    prefix="/simulations",
    tags=["Simulations"],
)

router.include_router(
    populations.router,
    prefix="/populations",
    tags=["Populations"],
)

router.include_router(
    presets.router,
    prefix="/models/presets",
    tags=["Model Presets"],
)
