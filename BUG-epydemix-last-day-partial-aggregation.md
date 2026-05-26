# Bug: last-day daily-resampled transitions are scaled down by `dt` when `dt < 1.0`

**Project:** epydemix (verified against `epistorm/epydemix` v1.0.2)

**Files:** `epydemix/utils/utils.py` (`compute_simulation_dates`), `epydemix/model/epimodel.py` (`simulate` -> resampling), `epydemix/model/simulation_results.py` (`Trajectory.resample`)

**Severity:** Medium. Affects any daily-resampled transition output when `dt < 1.0`. Compartment trajectories are unaffected.

---

## Summary

When the simulation step `dt` is sub-daily (e.g., `dt=0.5`, `0.25`, `0.1`) and the output is resampled to daily (`resample_frequency="D"`, the default), the **last calendar day** of the output reports a transition sum equal to roughly `dt x true_daily_value` instead of the full daily total.

The cause is grid alignment: `compute_simulation_dates` builds an inclusive `[start_date, end_date]` grid with spacing `dt`, so every interior day gets `1/dt` sub-steps but the final day gets only **one** sub-step (the `end_date 00:00:00` mark). Daily resampling with `sum` aggregation then under-counts the last day.

Compartments use `last` aggregation and only read the single sub-step value at the day boundary, so they look correct (but represent the value at `end_date 00:00`, not "end of day").

## Reproduction

```python
from fastapi.testclient import TestClient
from app.main import app
import numpy as np
import pandas as pd
from epydemix.utils.utils import compute_simulation_dates

# Confirm grid: interior days have 1/dt sub-steps, last day has 1
for dt in [1.0, 0.5, 0.25, 0.1]:
    d = compute_simulation_dates("2025-01-01", "2025-04-15", dt=dt)
    pdt = pd.to_datetime([str(x) for x in d])
    last = (pdt.normalize() == pd.Timestamp("2025-04-15")).sum()
    prev = (pdt.normalize() == pd.Timestamp("2025-04-14")).sum()
    print(f"dt={dt}: Apr-14 has {prev} sub-step(s), Apr-15 has {last}")
# dt=1.0:  Apr-14 has 1, Apr-15 has 1
# dt=0.5:  Apr-14 has 2, Apr-15 has 1
# dt=0.25: Apr-14 has 4, Apr-15 has 1
# dt=0.1:  Apr-14 has 10, Apr-15 has 1

# Observed E -> I medians (no vaccination, flu params, 200 sims, seed 7):
# dt=1.0 ending Apr-15:
#   ..., Apr-13: 1908, Apr-14: 1749, Apr-15: 1627  <- smooth decay
# dt=0.5 ending Apr-15:
#   ..., Apr-13: 1931, Apr-14: 1790, Apr-15:  843  <- last day is ~half
```

The last-day ratio matches `dt`:
- `dt=1.0`: last/prev ~ 0.93 (natural epidemic decay)
- `dt=0.5`: last/prev ~ 0.47 (half)
- `dt=0.25`: last/prev ~ 0.25 (quarter)
- `dt=0.1`: last/prev ~ 0.10 (tenth)

## Root cause

`compute_simulation_dates` (epydemix/utils/utils.py) builds the grid with `pd.date_range(start_date, end_date, freq=f"{dt}D")` (or equivalent), which is **inclusive on both ends**. With `dt=0.5` and `[2025-01-01, 2025-04-15]`:

```
2025-01-01 00:00, 2025-01-01 12:00,
2025-01-02 00:00, 2025-01-02 12:00,
...
2025-04-14 00:00, 2025-04-14 12:00,
2025-04-15 00:00         <- only one stamp on Apr-15, no 12:00 generated
```

After running `stochastic_simulation`, `transitions_evolution` has shape `(T, n_trans, N)` where `T = len(simulation_dates)`. Each entry is the count delivered during one sub-step.

`Trajectory.resample("D", resample_aggregation_transitions="sum", ...)` then groups by calendar day and sums. Interior days have `1/dt` rows so they sum the full day; the last day has 1 row so its sum is scaled by `dt`.

Compartments resample with `last` (the snapshot at the final sub-step within each calendar day), so they show the value at `Apr-15 00:00` for the last day, which is correct as a snapshot but does not represent "end of day Apr-15".

## Why this is a real-world pattern, not a misuse

Sub-daily `dt` is the standard recommendation for any model where per-step probabilities `1 - exp(-r*dt)` approach saturation at `dt=1` (e.g., fast incubation, fast recovery, high force of infection). Flu-like params (`incubation_period=1.5`, `infectious_period=4`) give `dt=1` exit probabilities of ~0.49 and ~0.22, large enough that the discrete-time bias is visible (peak incidence differs by ~25% between `dt=1.0` and `dt=0.1`). Users who reduce `dt` to fix that bias then hit this last-day artifact.

The artifact also breaks downstream tooling that reads `transitions[-1]` as "yesterday's incidence" or feeds the last day into a forecast cumulator, since that value is silently scaled by `dt`.

## Impact

| Output | Correct? |
|---|---|
| Compartments at last day | Snapshot at `end_date 00:00`, correct as snapshot but not "end of day" |
| Transitions at interior days | Correct |
| Transitions at last day (dt < 1) | Scaled by `dt` (under-counted) |
| Cumulative totals over the simulation | Under-count by `(1 - dt) x last_day_true_value` |

## Suggested fix

Option A: make the simulation grid right-open. Change `compute_simulation_dates` so the grid is `[start_date, end_date)`. The last sub-step then ends *at* `end_date` rather than starting at it. Every calendar day in the output has the same number of sub-steps. Breaks API compat for users who expect `simulation_dates[-1] == end_date`.

Option B: drop the partial last day during resampling. In `Trajectory.resample`, detect when the last calendar day has fewer than `1/dt` sub-steps and exclude it from the resampled output (returning a series whose last index is `end_date - 1 day`).

Option C: scale the partial last day by `1 / fraction_covered`. Preserves the index but reintroduces sampling noise on the last point. Worst of the three for downstream analysis.

We prefer A for upstream and B as a local mitigation in epydemix-api's `process_results` until A lands.

## Workarounds (for downstream users until fix lands)

1. Set `end_date = desired_end + 1 day` and slice off the trailing day yourself.
2. Drop the last row of `results.transitions[*]` whenever `dt < 1.0`.
3. Use `dt=1.0` and tolerate the discrete-time bias.

## Notes

- Bug verified against `epydemix==1.0.2`. Not yet checked against `mobs-lab/epydemix`.
- Related but distinct from the K-factor duplication in `BUG-epydemix-transition-count-duplication.md`. Both can be present simultaneously: a V-SEIHR run with `dt=0.5` will have `Susceptible_to_Exposed` doubled on every interior day **and** scaled to roughly `dt` of double on the last day.
- The artifact is a function of `(end_date - start_date) mod dt`. When the span is an exact multiple of `dt` and `end_date` is generated as an interior sub-step boundary, the last calendar day still has fewer-than-full sub-steps in the same way; the issue is inherent to the inclusive grid, not to specific spans.
