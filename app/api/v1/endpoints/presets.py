"""Model presets API endpoints.

This module provides the endpoint for listing available predefined
epidemic models (SIR, SEIR, SIS).
"""

from fastapi import APIRouter

from ..schemas.population import PresetInfo, PresetsListResponse

router = APIRouter()


# Predefined model information
PRESETS = [
    PresetInfo(
        name="SIR",
        description="Basic Susceptible-Infected-Recovered model. "
        "Suitable for diseases with permanent immunity after recovery.",
        compartments=["Susceptible", "Infected", "Recovered"],
        parameters={
            "transmission_rate": 0.3,
            "recovery_rate": 0.1,
        },
        transitions=[
            {
                "source": "Susceptible",
                "target": "Infected",
                "kind": "mediated",
                "params": ["transmission_rate", "Infected"],
            },
            {
                "source": "Infected",
                "target": "Recovered",
                "kind": "spontaneous",
                "params": "recovery_rate",
            },
        ],
    ),
    PresetInfo(
        name="SEIR",
        description="Susceptible-Exposed-Infected-Recovered model. "
        "Includes an exposed/latent period before becoming infectious.",
        compartments=["Susceptible", "Exposed", "Infected", "Recovered"],
        parameters={
            "transmission_rate": 0.3,
            "incubation_rate": 0.2,
            "recovery_rate": 0.1,
        },
        transitions=[
            {
                "source": "Susceptible",
                "target": "Exposed",
                "kind": "mediated",
                "params": ["transmission_rate", "Infected"],
            },
            {
                "source": "Exposed",
                "target": "Infected",
                "kind": "spontaneous",
                "params": "incubation_rate",
            },
            {
                "source": "Infected",
                "target": "Recovered",
                "kind": "spontaneous",
                "params": "recovery_rate",
            },
        ],
    ),
    PresetInfo(
        name="SIS",
        description="Susceptible-Infected-Susceptible model. "
        "No lasting immunity - individuals return to susceptible after recovery.",
        compartments=["Susceptible", "Infected"],
        parameters={
            "transmission_rate": 0.3,
            "recovery_rate": 0.1,
        },
        transitions=[
            {
                "source": "Susceptible",
                "target": "Infected",
                "kind": "mediated",
                "params": ["transmission_rate", "Infected"],
            },
            {
                "source": "Infected",
                "target": "Susceptible",
                "kind": "spontaneous",
                "params": "recovery_rate",
            },
        ],
    ),
]


@router.get(
    "",
    response_model=PresetsListResponse,
    summary="List model presets",
    description="Get information about available predefined epidemic models.",
)
async def get_presets() -> PresetsListResponse:
    """List all available model presets.

    Returns information about predefined epidemic models including their
    compartments, default parameters, and transition definitions.

    Returns
    -------
    PresetsListResponse
        List of available model presets with their configurations.
    """
    return PresetsListResponse(presets=PRESETS)
