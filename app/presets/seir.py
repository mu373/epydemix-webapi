"""SEIR preset: model builder + registry definition."""

from __future__ import annotations

from epydemix.model.epimodel import EpiModel


COMPARTMENTS: list[str] = ["Susceptible", "Exposed", "Infected", "Recovered"]

DEFAULT_PARAMETERS: dict[str, float] = {
    "transmission_rate": 0.3,
    "incubation_rate": 0.2,
    "recovery_rate": 0.1,
}

DESCRIPTION: str = (
    "Susceptible-Exposed-Infected-Recovered model. "
    "Includes an exposed/latent period before becoming infectious."
)

TRANSITIONS: list[dict] = [
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
        "params": ["incubation_rate"],
    },
    {
        "source": "Infected",
        "target": "Recovered",
        "kind": "spontaneous",
        "params": ["recovery_rate"],
    },
]

PARAMETER_CONVERSIONS: list[str] = [
    "incubation_rate",
    "recovery_rate",
    "transmission_rate",
]


def build_seir_model(
    user_scalars: dict[str, float],
) -> tuple[EpiModel, dict[str, str]]:
    """Construct an SEIR model with user scalar overrides.

    Returns ``(model, preset_calc_params)``. Period→rate and R0→β are handled
    by ``app.utils.parameter_conversions``.
    """
    merged = {**DEFAULT_PARAMETERS, **user_scalars}
    model = EpiModel(compartments=COMPARTMENTS)
    for name, value in merged.items():
        model.add_parameter(parameter_name=name, value=float(value))

    model.add_transition(
        source="Susceptible",
        target="Exposed",
        kind="mediated",
        params=("transmission_rate", "Infected"),
    )
    model.add_transition(
        source="Exposed",
        target="Infected",
        kind="spontaneous",
        params="incubation_rate",
    )
    model.add_transition(
        source="Infected",
        target="Recovered",
        kind="spontaneous",
        params="recovery_rate",
    )
    return model, {}
