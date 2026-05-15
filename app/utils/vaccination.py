"""Vaccination custom transition kind and per-strategy schedule builders.

Adds a ``vaccination_count`` transition kind to an epydemix model. The kind's
rate function reads the current source-compartment population at each step
and returns a per-age-group rate that delivers ``daily_doses(t) * dt`` doses,
split across the target age groups proportional to current susceptibility.

Each rollout strategy reduces to producing a length-``T`` ``daily_doses_at_t``
schedule; the rate function is strategy-agnostic. For v1 only ``flat_count``
is supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from epydemix.model.epimodel import EpiModel


@dataclass(frozen=True)
class ResolvedCampaign:
    """A single campaign reduced to a dose schedule and a target age subset.

    Attributes
    ----------
    daily_doses_at_t : np.ndarray
        Shape ``(T,)``. ``daily_doses_at_t[t]`` is the *per-day* dose count
        the simulator should aim to deliver at step ``t``; multiply by ``dt``
        for the per-step count. Zero outside the campaign window.
    target_age_indices : np.ndarray
        Shape ``(k,)``, ``int``. Indices into ``population.Nk``. ``None`` /
        ``[]`` is not a valid value; callers must resolve "all groups" to the
        full index range upstream.
    """

    daily_doses_at_t: np.ndarray
    target_age_indices: np.ndarray


def make_vaccination_rate_fn(
    campaigns: list[ResolvedCampaign],
    n_groups: int,
) -> Callable:
    """Build the rate function for the ``vaccination_count`` transition kind.

    The returned function is closed over the precomputed schedules. The
    simulator's per-step cost is one array index + one slice + one division
    per active campaign. Inactive steps (schedule value zero) short-circuit.

    The rate function signature matches what epydemix calls at each step:
    ``rate_fn(params, data)``. ``params`` carries ``{"source": <compartment>}``
    so the function can pull the current source-compartment population from
    ``data["pop"]`` without a global lookup.
    """

    def rate_fn(params, data):
        t = data["t"]
        source = params["source"]
        pop_source = data["pop"][data["comp_indices"][source]]  # (N,)
        rate = np.zeros(n_groups, dtype=np.float64)
        for camp in campaigns:
            doses_t = camp.daily_doses_at_t[t]
            if doses_t <= 0:
                continue
            tgt = camp.target_age_indices
            s_sum = float(pop_source[tgt].sum())
            if s_sum > 0:
                rate[tgt] += doses_t / s_sum
        return rate

    return rate_fn


def register_vaccination_kind(model: EpiModel, rate_fn: Callable) -> None:
    """Register ``vaccination_count`` on the model.

    Safe to call multiple times: the latest registration wins for a given
    model. Each request constructs a fresh model, so cross-request leakage
    is impossible.
    """
    model.register_transition_kind("vaccination_count", rate_fn)


def build_flat_count_schedule(
    sim_dates: np.ndarray,
    dt: float,
    c_start: str,
    c_end: str,
    daily_doses: float,
) -> np.ndarray:
    """Return a length-``T`` schedule for a constant ``daily_doses`` strategy.

    Active inside ``[c_start, c_end]`` (inclusive on both ends, aligned to the
    daily grid), zero outside. ``dt`` is accepted for parity with other
    builders but does not affect the per-day count: the rate function scales
    by ``dt`` when computing per-step transitions, so callers should not
    pre-scale here.

    Parameters
    ----------
    sim_dates : np.ndarray
        The simulation date grid from ``compute_simulation_dates``.
    dt : float
        Time step in days. Unused; retained so all strategy builders share a
        signature.
    c_start, c_end : str
        Campaign window dates (``YYYY-MM-DD``). Inclusive.
    daily_doses : float
        Constant per-day dose count during the window. Must be ``> 0``;
        upstream schema validation enforces this.
    """
    del dt  # unused for flat_count; kept for signature parity
    date_ts = pd.to_datetime([np.datetime_as_string(d, unit="D") for d in sim_dates])
    active = (date_ts >= pd.Timestamp(c_start)) & (date_ts <= pd.Timestamp(c_end))
    return np.where(np.asarray(active), float(daily_doses), 0.0)
