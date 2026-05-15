"""Unit tests for the vaccination machinery in isolation.

Exercises ``build_flat_count_schedule``, ``make_vaccination_rate_fn``, and the
end-to-end flow against a minimal X / X_vax model with no infection dynamics,
so transition counts directly reflect the dose schedule.
"""

import numpy as np
import pytest
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
    assert rate == pytest.approx(0.0, abs=1e-12)


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
    assert rate == pytest.approx(0.0, abs=1e-12)


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
    assert mean_per_step[in_window] == pytest.approx(daily_doses, rel=0.15)
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
    # Total vaccinations cannot exceed the finite source pool. One-sided hard
    # invariant, not an approximate equality.
    assert np.all(totals <= initial_X)
    # With 100k/day on a 31-day window over 1000 individuals, draining is
    # essentially deterministic: the mean lands on initial_X within 1% noise.
    assert totals.mean() == pytest.approx(initial_X, rel=0.01)


def _two_group_model(nk: list[float]) -> tuple[EpiModel, dict]:
    """Two compartments X / X_vax, two age groups, no infection dynamics."""
    model = EpiModel(compartments=["X", "X_vax"])
    pop = Population(name="two_group")
    pop.add_population(Nk=nk, Nk_names=["A", "B"])
    pop.add_contact_matrix(
        contact_matrix=np.array([[1.0, 1.0], [1.0, 1.0]]),
        layer_name="all",
    )
    model.set_population(pop)
    return model, {
        "X": np.array(nk, dtype=float),
        "X_vax": np.zeros(len(nk), dtype=float),
    }


def _wire_vaccination(
    model: EpiModel,
    schedules_and_targets: list[tuple[np.ndarray, np.ndarray]],
    n_groups: int,
) -> None:
    campaigns = [
        ResolvedCampaign(daily_doses_at_t=schedule, target_age_indices=targets)
        for schedule, targets in schedules_and_targets
    ]
    rate_fn = make_vaccination_rate_fn(campaigns, n_groups=n_groups)
    register_vaccination_kind(model, rate_fn)
    model.add_transition(
        source="X",
        target="X_vax",
        kind="vaccination_count",
        params={"source": "X"},
    )


def test_two_groups_proportional_split():
    """Doses split across target groups proportional to current S_i(t).

    With a 60/40 source split and a large source pool relative to total doses
    (so depletion is negligible during the window), per-group cumulative
    transitions should land on a 60/40 ratio within stochastic noise.
    """
    nk = [60_000.0, 40_000.0]
    model, initial = _two_group_model(nk)
    sim_dates = compute_simulation_dates("2025-01-01", "2025-01-31", dt=1.0)
    schedule = build_flat_count_schedule(sim_dates, 1.0, "2025-01-05", "2025-01-15", 100.0)
    _wire_vaccination(model, [(schedule, np.array([0, 1]))], n_groups=2)

    results = model.run_simulations(
        start_date="2025-01-01",
        end_date="2025-01-31",
        Nsim=200,
        dt=1.0,
        initial_conditions_dict=initial,
        rng=np.random.default_rng(0),
    )
    per_group_totals = {
        "A": np.array([traj.transitions["X_to_X_vax_A"].sum() for traj in results.trajectories]),
        "B": np.array([traj.transitions["X_to_X_vax_B"].sum() for traj in results.trajectories]),
    }
    total_mean = per_group_totals["A"].mean() + per_group_totals["B"].mean()
    a_share = per_group_totals["A"].mean() / total_mean
    # Expected 0.6 / 0.4 with shrinkage of <1% (source ratio drift across the
    # 11-day window: ~1100 doses out of 60000 in group A vs 40000 in group B).
    assert a_share == pytest.approx(0.6, abs=0.02)


def test_target_subset_age_groups():
    """A campaign targeting only one group leaves the other group's source untouched."""
    nk = [50_000.0, 50_000.0]
    model, initial = _two_group_model(nk)
    sim_dates = compute_simulation_dates("2025-01-01", "2025-01-31", dt=1.0)
    schedule = build_flat_count_schedule(sim_dates, 1.0, "2025-01-05", "2025-01-15", 500.0)
    # Target only group B (index 1).
    _wire_vaccination(model, [(schedule, np.array([1]))], n_groups=2)

    results = model.run_simulations(
        start_date="2025-01-01",
        end_date="2025-01-31",
        Nsim=50,
        dt=1.0,
        initial_conditions_dict=initial,
        rng=np.random.default_rng(1),
    )
    # Group A receives no doses; Group B receives ~500/day for 11 days.
    a_totals = np.array([traj.transitions["X_to_X_vax_A"].sum() for traj in results.trajectories])
    b_totals = np.array([traj.transitions["X_to_X_vax_B"].sum() for traj in results.trajectories])
    assert np.all(a_totals == 0)
    expected = 500.0 * 11
    assert b_totals.mean() == pytest.approx(expected, rel=0.05)


def test_two_overlapping_campaigns_rates_add():
    """Two campaigns on the same source/target in overlapping windows: rates add per step."""
    model, initial = _minimal_model(initial_X=1_000_000.0)
    sim_dates = compute_simulation_dates("2025-01-01", "2025-01-31", dt=1.0)
    schedule_a = build_flat_count_schedule(sim_dates, 1.0, "2025-01-05", "2025-01-15", 300.0)
    schedule_b = build_flat_count_schedule(sim_dates, 1.0, "2025-01-10", "2025-01-20", 200.0)
    _wire_vaccination(
        model,
        [(schedule_a, np.array([0])), (schedule_b, np.array([0]))],
        n_groups=1,
    )

    results = model.run_simulations(
        start_date="2025-01-01",
        end_date="2025-01-31",
        Nsim=200,
        dt=1.0,
        initial_conditions_dict=initial,
        rng=np.random.default_rng(2),
    )
    per_step = np.stack(
        [traj.transitions["X_to_X_vax_total"] for traj in results.trajectories]
    )
    mean_per_step = per_step.mean(axis=0)

    # Per-step expected: 0 before 01-05, 300 in [01-05, 01-09], 500 in [01-10, 01-15],
    # 200 in [01-16, 01-20], 0 after.
    for i, d in enumerate(sim_dates):
        date_str = np.datetime_as_string(d, unit="D")
        if date_str < "2025-01-05" or date_str > "2025-01-20":
            assert per_step[:, i].sum() == 0, f"step {date_str} should be zero"
        elif "2025-01-05" <= date_str <= "2025-01-09":
            assert mean_per_step[i] == pytest.approx(300.0, rel=0.1), f"step {date_str}"
        elif "2025-01-10" <= date_str <= "2025-01-15":
            assert mean_per_step[i] == pytest.approx(500.0, rel=0.1), f"step {date_str}"
        else:  # 2025-01-16 .. 2025-01-20
            assert mean_per_step[i] == pytest.approx(200.0, rel=0.15), f"step {date_str}"


def test_depletion_shoulder_matches_discrete_theory():
    """Per-step transitions follow `S(t) * (1 - exp(-c/S(t)))` through the depletion shoulder.

    Single group, small enough source that depletion bites partway through.
    With Nsim=300 the per-step mean tracks the iterated deterministic update
    tightly enough to discriminate it from the naive `daily_doses * dt` cap.
    """
    N = 10_000.0
    daily_doses = 5_000.0
    dt = 1.0
    horizon_days = 8

    model, initial = _minimal_model(initial_X=N)
    sim_dates = compute_simulation_dates("2025-01-01", "2025-01-08", dt=dt)
    schedule = build_flat_count_schedule(sim_dates, dt, "2025-01-01", "2025-01-08", daily_doses)
    _wire_vaccination(model, [(schedule, np.array([0]))], n_groups=1)

    results = model.run_simulations(
        start_date="2025-01-01",
        end_date="2025-01-08",
        Nsim=300,
        dt=dt,
        initial_conditions_dict=initial,
        rng=np.random.default_rng(3),
    )
    per_step = np.stack(
        [traj.transitions["X_to_X_vax_total"] for traj in results.trajectories]
    )
    mean_per_step = per_step.mean(axis=0)

    # Theoretical deterministic iteration of S(t+1) = S(t) * exp(-c/S(t)).
    c = daily_doses * dt
    s = N
    expected = np.empty(horizon_days, dtype=np.float64)
    for k in range(horizon_days):
        if s <= 0:
            expected[k] = 0.0
            continue
        p = 1.0 - np.exp(-c / s)
        delta = s * p
        expected[k] = delta
        s -= delta

    # Compare each step. Per-step std ~ sqrt(N * p * (1-p)) <= ~50; SE on mean
    # over 300 sims is ~3. Pick 5% relative tolerance and skip the deep tail
    # where the expected mean drops below 1 and stochastic noise dominates.
    for k in range(horizon_days):
        if expected[k] < 1:
            continue
        assert mean_per_step[k] == pytest.approx(expected[k], rel=0.05), f"step {k}"
