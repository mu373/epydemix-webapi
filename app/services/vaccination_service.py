"""Apply a request-level vaccination block to an epydemix model.

Validates the config, resolves the source/target compartments (preset
defaults vs. user overrides), pre-computes one ``daily_doses_at_t`` schedule
per campaign, registers the ``vaccination_count`` transition kind, and adds
a single ``vaccination_count`` transition. The simulator then drives doses
during each step according to the resolved schedules.

Mutates ``model`` in place; no-op when the request supplies no
``vaccination`` block or an empty campaign list.
"""

from __future__ import annotations

import numpy as np
from epydemix.model.epimodel import EpiModel
from epydemix.utils.utils import compute_simulation_dates

from ..api.v1.schemas.simulation import (
    FlatCountRollout,
    SimulationConfig,
    VaccinationConfig,
)
from ..utils.vaccination import (
    ResolvedCampaign,
    build_flat_count_schedule,
    make_vaccination_rate_fn,
    register_vaccination_kind,
)


_V_SEIHR_DEFAULT_SOURCE = "Susceptible"
_V_SEIHR_DEFAULT_TARGET = "Susceptible_vax"


def apply_vaccinations(
    model: EpiModel,
    config: VaccinationConfig | None,
    simulation: SimulationConfig,
    preset: str | None,
) -> None:
    """Wire the vaccination flow into ``model``.

    Must be called after ``setup_population`` so the age-group names and the
    population vector are available. Order vs. calculated parameters and
    parameter transforms doesn't matter: vaccination uses a custom transition
    kind that reads the source population at each step, not a regular
    parameter.

    Raises ``ValueError`` (forwarded as 422) on any validation problem:

      - empty ``campaigns`` list when the block is present;
      - missing ``source_compartment`` / ``target_compartment`` on a custom model;
      - source or target not in ``model.compartments``;
      - ``source_compartment == target_compartment``;
      - ``target_age_groups`` referencing an unknown age-group label.
    """
    if config is None or not config.campaigns:
        if config is not None and not config.campaigns:
            raise ValueError("'vaccination.campaigns' must contain at least one entry")
        return

    source, target = _resolve_source_target(config, preset, list(model.compartments))

    age_names = [str(name) for name in model.population.Nk_names]
    n_groups = model.population.num_groups
    dates = compute_simulation_dates(
        simulation.start_date, simulation.end_date, dt=simulation.dt
    )

    resolved: list[ResolvedCampaign] = []
    for camp in config.campaigns:
        age_idx = _resolve_age_indices(camp.target_age_groups, age_names)
        schedule = _build_schedule(
            camp.rollout, dates, simulation.dt, camp.start_date, camp.end_date
        )
        resolved.append(
            ResolvedCampaign(daily_doses_at_t=schedule, target_age_indices=age_idx)
        )

    rate_fn = make_vaccination_rate_fn(resolved, n_groups)
    register_vaccination_kind(model, rate_fn)
    model.add_transition(
        source=source,
        target=target,
        kind="vaccination_count",
        params={"source": source},
    )


def _resolve_source_target(
    config: VaccinationConfig,
    preset: str | None,
    compartments: list[str],
) -> tuple[str, str]:
    """Pick the source/target compartments, defaulting for V-SEIHR.

    - V-SEIHR: defaults to ``Susceptible`` / ``Susceptible_vax``; the user
      may override (e.g. to drive a booster flow on a custom V-SEIHR-derived
      model).
    - Other presets / custom models: both must be supplied.
    - Both names must exist in ``model.compartments`` and be distinct.
    """
    source = config.source_compartment
    target = config.target_compartment
    if preset == "V-SEIHR":
        if source is None:
            source = _V_SEIHR_DEFAULT_SOURCE
        if target is None:
            target = _V_SEIHR_DEFAULT_TARGET
    if source is None or target is None:
        missing = [
            name
            for name, value in (
                ("source_compartment", source),
                ("target_compartment", target),
            )
            if value is None
        ]
        raise ValueError(
            "'vaccination' requires "
            + " and ".join(repr(m) for m in missing)
            + " for models without the V-SEIHR preset"
        )
    if source not in compartments:
        raise ValueError(
            f"'vaccination.source_compartment' = {source!r} is not in "
            f"model.compartments; available: {compartments}"
        )
    if target not in compartments:
        raise ValueError(
            f"'vaccination.target_compartment' = {target!r} is not in "
            f"model.compartments; available: {compartments}"
        )
    if source == target:
        raise ValueError(
            "'vaccination.source_compartment' and 'vaccination.target_compartment' must be distinct"
        )
    return source, target


def _resolve_age_indices(
    target_age_groups: list[str] | None,
    age_names: list[str],
) -> np.ndarray:
    """Map requested labels to integer indices into ``population.Nk``.

    ``None`` means "all groups"; returns ``[0, 1, ..., N-1]`` in that case.
    Raises ``ValueError`` if any requested label is unknown.
    """
    if target_age_groups is None:
        return np.arange(len(age_names), dtype=np.int64)
    label_to_idx = {name: i for i, name in enumerate(age_names)}
    unknown = [g for g in target_age_groups if g not in label_to_idx]
    if unknown:
        raise ValueError(
            f"'vaccination.target_age_groups' contains unknown label(s) {unknown}; "
            f"valid age groups: {age_names}"
        )
    return np.asarray([label_to_idx[g] for g in target_age_groups], dtype=np.int64)


def _build_schedule(
    rollout,
    dates: np.ndarray,
    dt: float,
    c_start: str,
    c_end: str,
) -> np.ndarray:
    """Dispatch on the rollout discriminator and return a ``(T,)`` schedule.

    Adding a new rollout strategy = a new branch here plus a new
    ``build_*_schedule`` function in ``app.utils.vaccination``.
    """
    if isinstance(rollout, FlatCountRollout):
        return build_flat_count_schedule(dates, dt, c_start, c_end, rollout.daily_doses)
    raise ValueError(f"Unknown rollout type: {type(rollout).__name__}")
