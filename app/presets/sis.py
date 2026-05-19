"""SIS preset: model builder + registry definition."""

from __future__ import annotations

from epydemix.model.epimodel import EpiModel

from ..utils.parameter_conversions import ParameterConversion
from ._build import add_transitions


COMPARTMENTS: list[str] = ["Susceptible", "Infected"]

DEFAULT_PARAMETERS: dict[str, float] = {
    "transmission_rate": 0.3,
    "recovery_rate": 0.1,
}

DESCRIPTION: str = (
    "Susceptible-Infected-Susceptible model. "
    "No lasting immunity - individuals return to susceptible after recovery."
)

TRANSITIONS: list[dict] = [
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
        "params": ["recovery_rate"],
    },
]

# Friendlier source inputs the user can supply instead of the rate-form
# defaults above. Resolved by ``app.utils.parameter_conversions``.
PARAMETER_CONVERSIONS: dict[str, ParameterConversion] = {
    "recovery_rate": ParameterConversion("infectious_period", "1 / infectious_period"),
    "transmission_rate": ParameterConversion(
        "R0", "R0 * recovery_rate / CONTACT_MATRIX_EIGENVALUE_ALL"
    ),
}


def build_sis_model(
    user_scalars: dict[str, float],
) -> tuple[EpiModel, dict[str, str]]:
    """Construct an SIS model with user scalar overrides.

    Returns ``(model, preset_calc_params)``. Period→rate and R0→β are handled
    by ``app.utils.parameter_conversions``.
    """
    merged = {**DEFAULT_PARAMETERS, **user_scalars}
    model = EpiModel(compartments=COMPARTMENTS)
    for name, value in merged.items():
        model.add_parameter(parameter_name=name, value=float(value))

    add_transitions(model, TRANSITIONS)
    return model, {}
