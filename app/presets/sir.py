"""SIR preset: model builder + registry definition."""

from __future__ import annotations

from epydemix.model.epimodel import EpiModel


COMPARTMENTS: list[str] = ["Susceptible", "Infected", "Recovered"]

DEFAULT_PARAMETERS: dict[str, float] = {
    "transmission_rate": 0.3,
    "recovery_rate": 0.1,
}

DESCRIPTION: str = (
    "Basic Susceptible-Infected-Recovered model. "
    "Suitable for diseases with permanent immunity after recovery."
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
        "target": "Recovered",
        "kind": "spontaneous",
        "params": ["recovery_rate"],
    },
]

# DERIVED parameter names this preset opts into for the period→rate / R0→β
# resolver (see ``app.utils.parameter_conversions``). Order is irrelevant.
PARAMETER_CONVERSIONS: list[str] = ["recovery_rate", "transmission_rate"]


def build_sir_model(
    user_scalars: dict[str, float],
) -> tuple[EpiModel, dict[str, str]]:
    """Construct an SIR model with user scalar overrides.

    Returns ``(model, preset_calc_params)``. The preset itself defines no
    calculated parameters; period→rate and R0→β conversions are handled by
    the preset-scoped resolver in ``app.utils.parameter_conversions``.

    Age-varying (list) and expression-valued parameters are applied by the
    pipeline in ``simulation_service.run_simulation`` after the population is
    resolved; this builder only handles scalars.
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
        target="Recovered",
        kind="spontaneous",
        params="recovery_rate",
    )
    return model, {}
