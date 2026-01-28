"""Common schema definitions used across the API.

This module defines shared response schemas used by multiple endpoints.
"""

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard error response schema.

    Attributes
    ----------
    detail : str
        Human-readable error message.
    """

    detail: str


class HealthResponse(BaseModel):
    """Health check response schema.

    Attributes
    ----------
    status : str
        Health status (e.g., 'healthy').
    version : str
        API version string.
    epydemix_version : str or None
        Version of the epydemix library, or None if not available.
    """

    status: str
    version: str
    epydemix_version: str | None = None
