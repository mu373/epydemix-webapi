"""Unit tests for the vaccination machinery in isolation.

Exercises ``build_flat_count_schedule``, ``make_vaccination_rate_fn``, and the
end-to-end flow against a minimal X / X_vax model with no infection dynamics,
so transition counts directly reflect the dose schedule.
"""

import numpy as np
from epydemix.model.epimodel import EpiModel
from epydemix.population.population import Population
from epydemix.utils.utils import compute_simulation_dates

from app.utils.vaccination import (
    ResolvedCampaign,
    build_flat_count_schedule,
    make_vaccination_rate_fn,
    register_vaccination_kind,
)


def _minimal_model(initial_X: float = 1_000_000.0) -> tuple[EpiModel, dict]:
    """Two compartments X / X_vax, one age group, no infection dynamics."""
    model = EpiModel(compartments=["X", "X_vax"])
    pop = Population(name="test")
    pop.add_population(Nk=[initial_X], Nk_names=["all"])
    pop.add_contact_matrix(contact_matrix=np.array([[1.0]]), layer_name="all")
    model.set_population(pop)
    return model, {"X": np.array([initial_X]), "X_vax": np.array([0.0])}


def test_schedule_zero_outside_window():
    """`build_flat_count_schedule` puts zeros outside the campaign window."""
    dates = compute_simulation_dates("2025-01-01", "2025-01-31", dt=1.0)
    schedule = build_flat_count_schedule(dates, 1.0, "2025-01-10", "2025-01-20", 500.0)
    assert schedule.shape == (len(dates),)
    # Days before 2025-01-10 and after 2025-01-20 should be 0.
    for i, d in enumerate(dates):
        date_str = np.datetime_as_string(d, unit="D")
        if "2025-01-10" <= date_str <= "2025-01-20":
            assert schedule[i] == 500.0, f"in-window step {date_str} should be 500"
        else:
            assert schedule[i] == 0.0, f"out-of-window step {date_str} should be 0"


def test_rate_fn_inactive_campaign_returns_zero():
    """When the schedule is zero at step `t`, the rate function returns all zeros."""
    schedule = np.zeros(10, dtype=float)
    campaign = ResolvedCampaign(daily_doses_at_t=schedule, target_age_indices=np.array([0]))
    rate_fn = make_vaccination_rate_fn([campaign], n_groups=1)
    rate = rate_fn(
        {"source": "X"},
        {"t": 0, "pop": np.array([[100.0], [0.0]]), "comp_indices": {"X": 0, "X_vax": 1}},
    )
    assert np.allclose(rate, 0.0)


def test_rate_fn_with_empty_source_returns_zero():
    """When the source compartment is empty, the rate function returns zero (no NaN)."""
    schedule = np.array([100.0, 100.0])
    campaign = ResolvedCampaign(daily_doses_at_t=schedule, target_age_indices=np.array([0]))
    rate_fn = make_vaccination_rate_fn([campaign], n_groups=1)
    rate = rate_fn(
        {"source": "X"},
        {"t": 0, "pop": np.array([[0.0], [0.0]]), "comp_indices": {"X": 0, "X_vax": 1}},
    )
    assert np.all(np.isfinite(rate))
    assert np.allclose(rate, 0.0)


def test_flat_count_delivers_expected_doses_per_day():
    """Each in-window day delivers approximately `daily_doses * dt` transitions."""
    model, initial = _minimal_model(initial_X=1_000_000.0)
    sim_start, sim_end, dt = "2025-01-01", "2025-02-28", 1.0
    c_start, c_end = "2025-01-10", "2025-01-20"
    daily_doses = 1000.0

    sim_dates = compute_simulation_dates(sim_start, sim_end, dt=dt)
    schedule = build_flat_count_schedule(sim_dates, dt, c_start, c_end, daily_doses)
    campaign = ResolvedCampaign(
        daily_doses_at_t=schedule, target_age_indices=np.array([0])
    )
    rate_fn = make_vaccination_rate_fn([campaign], n_groups=1)
    register_vaccination_kind(model, rate_fn)
    model.add_transition(
        source="X",
        target="X_vax",
        kind="vaccination_count",
        params={"source": "X"},
    )

    results = model.run_simulations(
        start_date=sim_start,
        end_date=sim_end,
        Nsim=50,
        dt=dt,
        initial_conditions_dict=initial,
        rng=np.random.default_rng(42),
    )

    # Stack per-trajectory total-age transition counts into a (Nsim, T) array.
    per_step = np.stack(
        [traj.transitions["X_to_X_vax_total"] for traj in results.trajectories]
    )
    mean_per_step = per_step.mean(axis=0)
    # Inside the window the mean should land near `daily_doses`. With Nsim=50
    # and per-step std ~ sqrt(1000) ~ 32, the std of the mean is ~ 4.5,
    # so a 15% tolerance gives plenty of headroom.
    in_window = (sim_dates >= np.datetime64(c_start)) & (sim_dates <= np.datetime64(c_end))
    assert np.allclose(mean_per_step[in_window], daily_doses, rtol=0.15)
    # Outside the window: hard zero (schedule is exactly 0, no stochasticity).
    assert np.all(per_step[:, ~in_window] == 0)


def test_dose_cap_when_source_depleted():
    """When `daily_doses` >> available source, total transitions cap at the source pool."""
    initial_X = 1000.0
    model, initial = _minimal_model(initial_X=initial_X)
    sim_dates = compute_simulation_dates("2025-01-01", "2025-01-31", dt=1.0)
    schedule = build_flat_count_schedule(sim_dates, 1.0, "2025-01-01", "2025-01-31", 100_000)
    campaign = ResolvedCampaign(
        daily_doses_at_t=schedule, target_age_indices=np.array([0])
    )
    rate_fn = make_vaccination_rate_fn([campaign], n_groups=1)
    register_vaccination_kind(model, rate_fn)
    model.add_transition(
        source="X", target="X_vax", kind="vaccination_count", params={"source": "X"}
    )

    results = model.run_simulations(
        start_date="2025-01-01",
        end_date="2025-01-31",
        Nsim=20,
        dt=1.0,
        initial_conditions_dict=initial,
        rng=np.random.default_rng(0),
    )
    totals = np.array(
        [traj.transitions["X_to_X_vax_total"].sum() for traj in results.trajectories]
    )
    # Total vaccinations must equal the initial source pool (1000) within
    # rounding; cannot exceed it because the source compartment is finite.
    assert np.all(totals <= initial_X + 1e-6)
    # With 100k/day on a 31-day window over 1000 individuals, draining is
    # essentially deterministic: nearly all 1000 get vaccinated.
    assert totals.mean() > 0.99 * initial_X
