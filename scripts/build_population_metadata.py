"""Generate the precomputed population metadata CSV.

Loads every population from epydemix-data and writes its default age-group
breakdown to ``app/data/population_metadata.csv``. The CSV seeds the runtime
metadata cache at startup so ``GET /v1/populations`` can return full summaries
without live-loading hundreds of countries on boot.

Re-run this script whenever the upstream epydemix-data release changes, and
commit the regenerated CSV alongside the dependency bump.

Usage:
    uv run python scripts/build_population_metadata.py
"""

from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from epydemix.population.population import (
    get_available_locations,
    load_epydemix_population,
)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "app" / "data" / "population_metadata.csv"
MAX_WORKERS = 8


def compute_entry(name: str) -> tuple[str, list[tuple[str, int]]] | tuple[str, None]:
    """Load a population and return (name, [(age_group, count), ...]).

    Returns the name with None if the load fails so the caller can log the
    failure and skip the population. Order of age groups follows the
    population's own ordering (age-ascending).
    """
    try:
        pop = load_epydemix_population(population_name=name)
        pairs = [
            (str(label), int(count)) for label, count in zip(pop.Nk_names, pop.Nk)
        ]
        return name, pairs
    except Exception as exc:
        print(f"  failed: {name}: {exc}", file=sys.stderr)
        return name, None


def main() -> int:
    locations = get_available_locations()["location"].tolist()
    print(f"Loading {len(locations)} populations with {MAX_WORKERS} workers...")
    t0 = time.time()

    entries: dict[str, list[tuple[str, int]]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(compute_entry, name): name for name in locations}
        for i, future in enumerate(as_completed(futures), 1):
            name, pairs = future.result()
            if pairs is None:
                continue
            entries[name] = pairs
            if i % 25 == 0:
                print(f"  {i}/{len(locations)}")

    # Sort populations alphabetically, keep age-group order within each.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "age_group", "population"])
        for name in sorted(entries):
            for age_group, count in entries[name]:
                writer.writerow([name, age_group, count])

    elapsed = time.time() - t0
    total_rows = sum(len(v) for v in entries.values())
    print(f"Wrote {total_rows} rows ({len(entries)} populations) to {OUTPUT_PATH} in {elapsed:.1f}s")
    skipped = len(locations) - len(entries)
    if skipped:
        print(f"Skipped {skipped} populations that failed to load.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
