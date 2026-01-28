"""Application configuration settings.

This module defines the Settings class which loads configuration from
environment variables with the EPYDEMIX_ prefix.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden by environment variables with the
    EPYDEMIX_ prefix (e.g., EPYDEMIX_DEBUG=true).

    Attributes
    ----------
    app_name : str
        Application display name.
    app_version : str
        Application version string.
    debug : bool
        Enable debug mode.
    api_v1_prefix : str
        URL prefix for API v1 endpoints.
    default_nsim : int
        Default number of simulations if not specified.
    max_nsim : int
        Maximum allowed number of simulations.
    default_dt : float
        Default time step in days.
    """

    model_config = SettingsConfigDict(env_prefix="EPYDEMIX_")

    app_name: str = "epydemix WebAPI"
    app_version: str = "0.1.0"
    debug: bool = False

    # API settings
    api_v1_prefix: str = "/api/v1"

    # Simulation defaults
    default_nsim: int = 100
    max_nsim: int = 1000
    default_dt: float = 1.0


settings = Settings()
