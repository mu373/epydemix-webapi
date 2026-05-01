"""Simulation API endpoints.

This module provides the endpoint for running epidemic simulations.
"""

from fastapi import APIRouter, Body, HTTPException
from fastapi.openapi.models import Example

from ....services.simulation_service import run_simulation
from ..schemas.simulation import SimulationRequest, SimulationResponse

router = APIRouter()

SIMULATION_REQUEST_EXAMPLES: dict[str, Example] = {
    "SIR": Example(
        summary="Basic SIR simulation",
        description="Run a Susceptible-Infected-Recovered model on the US population over two months.",
        value={
            "model": {
                "preset": "SIR",
                "parameters": {"transmission_rate": 0.3, "recovery_rate": 0.1},
            },
            "population": {"name": "United_States"},
            "simulation": {
                "start_date": "2024-01-01",
                "end_date": "2024-03-01",
                "Nsim": 10,
            },
        },
    ),
}


@router.post(
    "",
    response_model=SimulationResponse,
    summary="Run epidemic simulation",
    description="Execute an epidemic simulation with the specified configuration.",
)
async def create_simulation(
    request: SimulationRequest = Body(..., openapi_examples=SIMULATION_REQUEST_EXAMPLES),
) -> SimulationResponse:
    """Run an epidemic simulation.

    Accepts a simulation configuration and returns results including
    compartment trajectories, transitions, and optional summary statistics.

    The simulation can use predefined model presets (SIR, SEIR, SIS) or
    custom compartmental models defined by the user.

    Parameters
    ----------
    request : SimulationRequest
        Complete simulation configuration including model, population,
        simulation parameters, and output options.

    Returns
    -------
    SimulationResponse
        Simulation results with compartment and transition data.

    Raises
    ------
    HTTPException
        400 if validation fails, 500 if simulation fails.
    """
    try:
        return run_simulation(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
