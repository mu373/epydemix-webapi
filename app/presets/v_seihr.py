"""V-SEIHR preset: vaccinated/unvaccinated parallel-compartment SEIHR.

Every compartment has a vaccinated twin (``X_vax``). Vaccine efficacy reduces
susceptibility (``VE_S``, applied to the force-of-infection on the vaccinated
layer) and severity (``VE_H``, applied to the hospitalization split for
vaccinated infected individuals). Vaccinated infectious individuals carry the
*same* per-contact transmissibility as unvaccinated.

The ``vaccination`` request block drives ``Susceptible → Susceptible_vax``.
Both layers share the rate-form parameters (``incubation_rate``,
``recovery_rate``, ``hosp_recovery_rate``, ``waning_rate``); the VE twins
``transmission_rate_vax`` and ``hosp_proportion_vax`` are calculated from the
unvaccinated values via ``(1 - VE)``.
"""

from __future__ import annotations

from epydemix.model.epimodel import EpiModel


DESCRIPTION: str = (
    "Vaccinated SEIHR model with parallel unvaccinated/vaccinated compartments. "
    "Vaccine efficacy reduces susceptibility (VE_S) and severity (VE_H). Use "
    "with the `vaccination` block to drive doses from Susceptible to "
    "Susceptible_vax."
)


COMPARTMENTS: list[str] = [
    "Susceptible",
    "Susceptible_vax",
    "Exposed",
    "Exposed_vax",
    "Infected",
    "Infected_vax",
    "Hospitalized",
    "Hospitalized_vax",
    "Recovered",
    "Recovered_vax",
]


# Scalar defaults. Periods are in days; ``hosp_proportion`` / VE are unitless
# in [0, 1]; ``R0`` is unitless. The dashboard SEIHR (COVID-19) preset is the
# reference for the disease-history values; VE_S / VE_H are placeholders.
#
# Waning is OFF by default (``waning_rate: 0.0``). Pass ``immunity_duration``
# in the request to enable; the parameter-conversion resolver then injects
# ``waning_rate = 1 / immunity_duration`` as a calculated parameter and the
# scalar default is dropped.
DEFAULT_PARAMETERS: dict[str, float] = {
    "R0": 2.5,
    "incubation_period": 3.0,
    "infectious_period": 2.5,
    "hospitalization_duration": 5.0,
    "waning_rate": 0.0,
    "hosp_proportion": 0.05,
    "VE_S": 0.7,
    "VE_H": 0.85,
}


# V-SEIHR-specific calculated parameters. Period→rate and R0→β are universal
# and live in ``app.utils.parameter_conversions``; this dict carries only what
# is genuinely V-SEIHR-only.
#
# Users can override these by passing a matching name in ``parameters``;
# user-supplied calc-params win on collision (see ``create_model`` merge order).
TRANSITIONS: list[dict] = [
    {"source": "Susceptible", "target": "Exposed", "kind": "mediated",
     "params": ["transmission_rate", "Infected"]},
    {"source": "Susceptible", "target": "Exposed", "kind": "mediated",
     "params": ["transmission_rate", "Infected_vax"]},
    {"source": "Exposed", "target": "Infected", "kind": "spontaneous",
     "params": ["incubation_rate"]},
    {"source": "Infected", "target": "Recovered", "kind": "spontaneous",
     "params": ["I_to_R_rate"]},
    {"source": "Infected", "target": "Hospitalized", "kind": "spontaneous",
     "params": ["I_to_H_rate"]},
    {"source": "Hospitalized", "target": "Recovered", "kind": "spontaneous",
     "params": ["hosp_recovery_rate"]},
    {"source": "Recovered", "target": "Susceptible", "kind": "spontaneous",
     "params": ["waning_rate"]},
    {"source": "Susceptible_vax", "target": "Exposed_vax", "kind": "mediated",
     "params": ["transmission_rate_vax", "Infected"]},
    {"source": "Susceptible_vax", "target": "Exposed_vax", "kind": "mediated",
     "params": ["transmission_rate_vax", "Infected_vax"]},
    {"source": "Exposed_vax", "target": "Infected_vax", "kind": "spontaneous",
     "params": ["incubation_rate"]},
    {"source": "Infected_vax", "target": "Recovered_vax", "kind": "spontaneous",
     "params": ["Ivax_to_R_rate"]},
    {"source": "Infected_vax", "target": "Hospitalized_vax", "kind": "spontaneous",
     "params": ["Ivax_to_H_rate"]},
    {"source": "Hospitalized_vax", "target": "Recovered_vax", "kind": "spontaneous",
     "params": ["hosp_recovery_rate"]},
    {"source": "Recovered_vax", "target": "Susceptible_vax", "kind": "spontaneous",
     "params": ["waning_rate"]},
]


PARAMETER_CONVERSIONS: list[str] = [
    "incubation_rate",
    "recovery_rate",
    "hosp_recovery_rate",
    "waning_rate",
    "transmission_rate",
]


PRESET_CALC_PARAMETERS: dict[str, str] = {
    # Vaccine-efficacy twins.
    "transmission_rate_vax": "(1 - VE_S) * transmission_rate",
    "hosp_proportion_vax": "(1 - VE_H) * hosp_proportion",
    # Competing-exit composites: I → R or H split by ``hosp_proportion``,
    # both exit at rate ``recovery_rate`` (no separate "time to hospitalization").
    "I_to_R_rate": "(1 - hosp_proportion) * recovery_rate",
    "I_to_H_rate": "hosp_proportion * recovery_rate",
    "Ivax_to_R_rate": "(1 - hosp_proportion_vax) * recovery_rate",
    "Ivax_to_H_rate": "hosp_proportion_vax * recovery_rate",
}


def build_v_seihr_model(
    user_scalars: dict[str, float],
) -> tuple[EpiModel, dict[str, str]]:
    """Construct the V-SEIHR model.

    Returns ``(model, preset_calc_params)``. The preset writes its scalar
    defaults (including ``R0``, periods, and ``waning_rate: 0.0``) directly into
    ``model.parameters``; user scalars override on top. Period→rate and R0→β
    conversions are applied after this returns, by
    ``app.utils.parameter_conversions.resolve_parameter_conversions``.

    The vaccinated layer mirrors the unvaccinated transitions exactly, with two
    differences:

      - the force-of-infection uses ``transmission_rate_vax`` (= (1 - VE_S) *
        transmission_rate);
      - the hospitalization split uses ``hosp_proportion_vax`` (= (1 - VE_H) *
        hosp_proportion).

    The ``Susceptible → Susceptible_vax`` flow is *not* declared here; it is
    added later by ``apply_vaccinations`` when the request supplies a
    ``vaccination`` block. Without a vaccination block the vaccinated layer
    stays at zero (default initial conditions).
    """
    model = EpiModel(compartments=COMPARTMENTS)
    merged = {**DEFAULT_PARAMETERS, **user_scalars}
    for name, value in merged.items():
        model.add_parameter(parameter_name=name, value=float(value))

    # Unvaccinated layer.
    model.add_transition(
        source="Susceptible", target="Exposed", kind="mediated",
        params=("transmission_rate", "Infected"),
    )
    model.add_transition(
        source="Susceptible", target="Exposed", kind="mediated",
        params=("transmission_rate", "Infected_vax"),
    )
    model.add_transition(
        source="Exposed", target="Infected", kind="spontaneous",
        params="incubation_rate",
    )
    model.add_transition(
        source="Infected", target="Recovered", kind="spontaneous",
        params="I_to_R_rate",
    )
    model.add_transition(
        source="Infected", target="Hospitalized", kind="spontaneous",
        params="I_to_H_rate",
    )
    model.add_transition(
        source="Hospitalized", target="Recovered", kind="spontaneous",
        params="hosp_recovery_rate",
    )
    model.add_transition(
        source="Recovered", target="Susceptible", kind="spontaneous",
        params="waning_rate",
    )

    # Vaccinated layer (twin transitions; force-of-infection mediated by both
    # I and I_vax with the VE-attenuated transmission rate).
    model.add_transition(
        source="Susceptible_vax", target="Exposed_vax", kind="mediated",
        params=("transmission_rate_vax", "Infected"),
    )
    model.add_transition(
        source="Susceptible_vax", target="Exposed_vax", kind="mediated",
        params=("transmission_rate_vax", "Infected_vax"),
    )
    model.add_transition(
        source="Exposed_vax", target="Infected_vax", kind="spontaneous",
        params="incubation_rate",
    )
    model.add_transition(
        source="Infected_vax", target="Recovered_vax", kind="spontaneous",
        params="Ivax_to_R_rate",
    )
    model.add_transition(
        source="Infected_vax", target="Hospitalized_vax", kind="spontaneous",
        params="Ivax_to_H_rate",
    )
    model.add_transition(
        source="Hospitalized_vax", target="Recovered_vax", kind="spontaneous",
        params="hosp_recovery_rate",
    )
    model.add_transition(
        source="Recovered_vax", target="Susceptible_vax", kind="spontaneous",
        params="waning_rate",
    )

    return model, dict(PRESET_CALC_PARAMETERS)
