"""SIS preset: model builder + registry definition."""

from __future__ import annotations

from epydemix.model.epimodel import EpiModel


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

PARAMETER_CONVERSIONS: list[str] = ["recovery_rate", "transmission_rate"]


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

    model.add_transition(
        source="Susceptible",
        target="Infected",
        kind="mediated",
        params=("transmission_rate", "Infected"),
    )
    model.add_transition(
        source="Infected",
        target="Susceptible",
        kind="spontaneous",
        params="recovery_rate",
    )
    return model, {}
