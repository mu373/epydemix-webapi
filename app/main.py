"""FastAPI application entry point for epydemix WebAPI.

This module configures and creates the FastAPI application instance,
sets up CORS middleware, and defines the root and health check endpoints.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.v1.router import router as api_v1_router
from .api.v1.schemas.common import HealthResponse
from .config import settings
from .services.population_service import warm_cache

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup: warm the population cache
    if settings.warm_cache_on_startup:
        logger.info("Warming population cache...")
        results = warm_cache(populations=settings.warm_cache_populations)
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Warmed {success_count}/{len(results)} populations")
    yield
    # Shutdown: nothing to clean up


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="REST API for running epidemic simulations with epydemix",
    docs_url=f"{settings.api_v1_prefix}/docs",
    redoc_url=f"{settings.api_v1_prefix}/redoc",
    openapi_url=f"{settings.api_v1_prefix}/openapi.json",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_v1_router, prefix=settings.api_v1_prefix)


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Check API health status.

    Returns the API version and epydemix library version to verify
    the service is running correctly.

    Returns
    -------
    HealthResponse
        Health status including API and epydemix versions.
    """
    try:
        import epydemix

        epydemix_version = getattr(epydemix, "__version__", "unknown")
    except ImportError:
        epydemix_version = None

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        epydemix_version=epydemix_version,
    )


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint providing API information.

    Returns
    -------
    dict
        Dictionary with API name and links to documentation and health check.
    """
    return {
        "message": "epydemix WebAPI",
        "docs": f"{settings.api_v1_prefix}/docs",
        "health": f"{settings.api_v1_prefix}/health",
    }
