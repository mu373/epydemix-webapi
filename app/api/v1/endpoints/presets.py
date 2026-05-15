"""Model presets API endpoints.

Lists available predefined epidemic models. Data comes from the single-source
registry in ``app.presets.registry`` so adding a preset doesn't require edits
here.
"""

from fastapi import APIRouter

from ....presets import PRESETS
from ..schemas.population import PresetInfo, PresetsListResponse

router = APIRouter()


def _preset_info(definition) -> PresetInfo:
    return PresetInfo(
        name=definition.name,
        description=definition.description,
        compartments=list(definition.compartments),
        parameters=dict(definition.default_parameters),
        transitions=list(definition.transitions),
    )


@router.get(
    "",
    response_model=PresetsListResponse,
    summary="List model presets",
    description="Get information about available predefined epidemic models.",
    operation_id="list_model_presets",
)
async def get_presets() -> PresetsListResponse:
    """List all available model presets.

    Returns information about predefined epidemic models including their
    compartments, default parameters, and transition definitions.
    """
    return PresetsListResponse(
        presets=[_preset_info(d) for d in PRESETS.values()],
    )
