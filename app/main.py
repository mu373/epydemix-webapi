"""FastAPI application entry point for epydemix WebAPI.

This module configures and creates the FastAPI application instance,
sets up CORS middleware, and defines the root and health check endpoints.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import PlainTextResponse

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
    servers=[
        {"url": "https://epyscenario-api.isi.it", "description": "Production"},
        {"url": "http://localhost:8000", "description": "Local"},
    ],
    openapi_tags=[
        {
            "name": "Simulations",
            "description": "Run stochastic epidemic simulations using preset or custom compartmental models.",
        },
        {
            "name": "Populations",
            "description": "Browse available populations, age group demographics, and contact matrices.",
        },
        {
            "name": "Model Presets",
            "description": "List built-in epidemic models (SIR, SEIR, SIS) with their compartments, parameters, and transitions.",
        },
    ],
)

# Return gzip when the response is over minimum_size.
app.add_middleware(GZipMiddleware, minimum_size=1000)

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
    from importlib.metadata import PackageNotFoundError, version

    try:
        epydemix_version = version("epydemix")
    except PackageNotFoundError:
        epydemix_version = None

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        epydemix_version=epydemix_version,
    )


_API_BASE = settings.api_v1_prefix


def _link_header() -> str:
    """RFC 8288 ``Link`` header advertising agent-discoverable resources."""
    return ", ".join([
        '</.well-known/api-catalog>; rel="api-catalog"; type="application/linkset+json"',
        f'<{_API_BASE}/openapi.json>; rel="service-desc"; type="application/json"',
        f'<{_API_BASE}/docs>; rel="service-doc"; type="text/html"',
        f'<{_API_BASE}/health>; rel="status"; type="application/json"',
    ])


@app.get("/", include_in_schema=False)
async def root(response: Response):
    """Root endpoint providing API information.

    Sets ``Link`` headers (RFC 8288) advertising the OpenAPI spec, interactive
    docs, health endpoint, and API catalog so agents can discover them
    programmatically.
    """
    response.headers["Link"] = _link_header()
    return {
        "message": "epydemix WebAPI",
        "docs": f"{_API_BASE}/docs",
        "health": f"{_API_BASE}/health",
        "api_catalog": "/.well-known/api-catalog",
    }


_ROBOTS_TXT = (Path(__file__).parent / "static" / "robots.txt").read_text(encoding="utf-8")


@app.get("/robots.txt", include_in_schema=False, response_class=PlainTextResponse)
async def robots_txt() -> str:
    """Robots policy (RFC 9309) plus Content Signals (contentsignals.org).

    Served verbatim from ``app/static/robots.txt``. Open policy: this is a
    public REST API for an open-source epidemic simulator, so all crawlers
    and AI agents are allowed and the Content-Signal directive opts in to
    AI training, search indexing, and live AI grounding (RAG) usage.
    """
    return _ROBOTS_TXT


@app.get(
    "/.well-known/api-catalog",
    include_in_schema=False,
)
async def api_catalog() -> Response:
    """API catalog (RFC 9727) returning ``application/linkset+json``.

    Each entry advertises the API root with a ``service-desc`` (OpenAPI spec),
    ``service-doc`` (interactive Scalar docs), and ``status`` (health check)
    so agents can discover the schema, documentation, and liveness probe
    from a single well-known location.
    """
    catalog = {
        "linkset": [
            {
                "anchor": _API_BASE,
                "service-desc": [
                    {
                        "href": f"{_API_BASE}/openapi.json",
                        "type": "application/json",
                    }
                ],
                "service-doc": [
                    {
                        "href": f"{_API_BASE}/docs",
                        "type": "text/html",
                    }
                ],
                "status": [
                    {
                        "href": f"{_API_BASE}/health",
                        "type": "application/json",
                    }
                ],
            }
        ]
    }
    return Response(
        content=json.dumps(catalog),
        media_type="application/linkset+json",
    )
