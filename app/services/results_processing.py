"""Post-processing utilities for simulation results.

This module provides functions for transforming epydemix simulation results
into the API response format, including hierarchical data structures,
trajectory extraction, and summary statistics computation.
"""

import numpy as np
import pandas as pd
from epydemix.model.epimodel import EpiModel
from epydemix.model.simulation_results import SimulationResults

from ..api.v1.schemas.simulation import (
    CompartmentResults,
    OutputConfig,
    PeakStatistic,
    SummaryConfig,
    SummaryResults,
    SummaryStatistic,
    TrajectoriesResults,
    TrajectoryData,
    TransitionResults,
)
from ..utils.column_utils import parse_column_name


def build_quantile_hierarchy(
    df: pd.DataFrame,
    known_bases: list[str],
    quantiles_used: list[float],
    filter_bases: list[str] | None,
    filter_age_groups: list[str] | None,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Build hierarchical data structure from quantiles dataframe.

    Transforms a flat dataframe with columns like 'Infected_0-4', 'Infected_total'
    into a nested dictionary structure organized by base name, age group, and quantile.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'date', 'quantile' columns and data columns.
    known_bases : list of str
        Known base names for column parsing (e.g., compartment or transition names).
    quantiles_used : list of float
        List of quantile values present in the data.
    filter_bases : list of str or None
        If provided, only include these base names in output.
    filter_age_groups : list of str or None
        If provided, only include these age groups in output.

    Returns
    -------
    dict
        Nested dictionary with structure:
        ``{base_name: {age_group: {quantile_str: [values]}}}``.
    """
    cols = [c for c in df.columns if c not in ["date", "quantile"]]

    hierarchical: dict[str, dict[str, dict[str, list[float]]]] = {}

    for col in cols:
        base_name, age_group = parse_column_name(col, known_bases)

        # Apply filters
        if filter_bases is not None and base_name not in filter_bases:
            continue
        if filter_age_groups is not None and age_group not in filter_age_groups:
            continue

        if base_name not in hierarchical:
            hierarchical[base_name] = {}
        if age_group not in hierarchical[base_name]:
            hierarchical[base_name][age_group] = {}

        for q in quantiles_used:
            q_data = df[df["quantile"] == q][col].values.tolist()
            hierarchical[base_name][age_group][str(q)] = q_data

    return hierarchical


def extract_trajectories(
    results: SimulationResults,
    output_config: OutputConfig,
    compartment_names: list[str],
    transition_names: list[str],
) -> TrajectoriesResults:
    """Extract raw trajectory data from simulation results.

    Converts the raw simulation trajectories into a hierarchical structure
    organized by compartment/transition name and age group.

    Parameters
    ----------
    results : SimulationResults
        Simulation results from epydemix.
    output_config : OutputConfig
        Output configuration specifying filters (e.g., age_groups).
    compartment_names : list of str
        List of compartment names from the model.
    transition_names : list of str
        List of transition names (e.g., ['S_to_I', 'I_to_R']).

    Returns
    -------
    TrajectoriesResults
        Trajectory data with hierarchical structure for each simulation run.
    """
    dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in results.dates]

    age_groups_filter = output_config.age_groups

    runs = []
    for trajectory in results.trajectories:
        # Build hierarchical compartments: {compartment: {age_group: [values]}}
        compartments: dict[str, dict[str, list[float]]] = {}
        for col_name, values in trajectory.compartments.items():
            base_name, age_group = parse_column_name(col_name, compartment_names)
            if age_groups_filter is not None and age_group not in age_groups_filter:
                continue
            if base_name not in compartments:
                compartments[base_name] = {}
            compartments[base_name][age_group] = values.tolist()

        # Build hierarchical transitions: {transition: {age_group: [values]}}
        transitions: dict[str, dict[str, list[float]]] = {}
        for col_name, values in trajectory.transitions.items():
            base_name, age_group = parse_column_name(col_name, transition_names)
            if age_groups_filter is not None and age_group not in age_groups_filter:
                continue
            if base_name not in transitions:
                transitions[base_name] = {}
            transitions[base_name][age_group] = values.tolist()

        runs.append(TrajectoryData(compartments=compartments, transitions=transitions))

    return TrajectoriesResults(dates=dates, runs=runs)


def compute_summary(
    results: SimulationResults,
    summary_config: SummaryConfig | None,
) -> SummaryResults | None:
    """Compute summary statistics from simulation results.

    Calculates peak statistics for specified compartments and total counts
    for specified transitions across all simulation runs.

    Parameters
    ----------
    results : SimulationResults
        Simulation results from epydemix.
    summary_config : SummaryConfig or None
        Configuration specifying which compartments and transitions to
        compute statistics for. If None, returns None.

    Returns
    -------
    SummaryResults or None
        Summary statistics with peaks and totals, or None if no config
        provided or no statistics could be computed.
    """
    if summary_config is None:
        return None

    peaks: dict[str, PeakStatistic] = {}
    totals: dict[str, SummaryStatistic] = {}

    # Compute peak statistics for requested compartments
    if summary_config.peak_compartments:
        stacked = results.get_stacked_compartments()
        for comp_name in summary_config.peak_compartments:
            # Try with _total suffix first, then without
            comp_key = f"{comp_name}_total"
            if comp_key not in stacked:
                comp_key = comp_name
                if comp_key not in stacked:
                    continue

            comp_data = stacked[comp_key]  # Shape: (Nsim, timesteps)

            peak_per_sim = np.max(comp_data, axis=1)
            peak_median = float(np.median(peak_per_sim))
            peak_ci = [
                float(np.percentile(peak_per_sim, 2.5)),
                float(np.percentile(peak_per_sim, 97.5)),
            ]

            # Find peak date (use median trajectory)
            median_traj = np.median(comp_data, axis=0)
            peak_idx = int(np.argmax(median_traj))
            peak_date = None
            if len(results.dates) > 0:
                date_val = results.dates[peak_idx]
                peak_date = pd.Timestamp(date_val).strftime("%Y-%m-%d")

            peaks[comp_name] = PeakStatistic(
                median=peak_median, ci_95=peak_ci, peak_date=peak_date
            )

    # Compute total counts for requested transitions
    if summary_config.total_transitions:
        trans_stacked = results.get_stacked_transitions()
        for trans_name in summary_config.total_transitions:
            # Try with _total suffix first, then without
            trans_key = f"{trans_name}_total"
            if trans_key not in trans_stacked:
                trans_key = trans_name
                if trans_key not in trans_stacked:
                    continue

            trans_data = trans_stacked[trans_key]
            total_per_sim = np.sum(trans_data, axis=1)
            totals[trans_name] = SummaryStatistic(
                median=float(np.median(total_per_sim)),
                ci_95=[
                    float(np.percentile(total_per_sim, 2.5)),
                    float(np.percentile(total_per_sim, 97.5)),
                ],
            )

    if not peaks and not totals:
        return None

    return SummaryResults(
        peaks=peaks if peaks else None,
        totals=totals if totals else None,
    )


def process_results(
    results: SimulationResults,
    output_config: OutputConfig | None,
    model: EpiModel,
) -> "SimulationResultsData":
    """Process simulation results into API response format.

    Transforms raw epydemix simulation results into the structured response
    format, including compartment quantiles, transition quantiles, optional
    summary statistics, and optional raw trajectories.

    Parameters
    ----------
    results : SimulationResults
        Simulation results from epydemix.
    output_config : OutputConfig or None
        Output configuration specifying quantiles, filters, and options.
        Uses defaults if None.
    model : EpiModel
        The EpiModel used for the simulation, needed for compartment and
        transition names.

    Returns
    -------
    SimulationResultsData
        Processed results containing compartments, transitions, and
        optionally summary statistics and raw trajectories.
    """
    # Import here to avoid circular imports
    from ..api.v1.schemas.simulation import SimulationResultsData

    if output_config is None:
        output_config = OutputConfig()

    # Get compartment quantiles (None lets epydemix use its default)
    comp_df = results.get_quantiles_compartments(quantiles=output_config.quantiles)
    dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in comp_df["date"].unique()]

    # Get the actual quantiles used (from the dataframe)
    quantiles_used = sorted(comp_df["quantile"].unique().tolist())

    # Known compartment names from model
    compartment_names = model.compartments

    # Build hierarchical compartment data
    comp_data = build_quantile_hierarchy(
        comp_df,
        compartment_names,
        quantiles_used,
        output_config.compartments,
        output_config.age_groups,
    )
    compartment_results = CompartmentResults(dates=dates, data=comp_data)

    # Get transition quantiles
    trans_df = results.get_quantiles_transitions(quantiles=output_config.quantiles)

    # Known transition names from model
    transition_names = [f"{t.source}_to_{t.target}" for t in model.transitions_list]

    # Build hierarchical transition data
    trans_data = build_quantile_hierarchy(
        trans_df,
        transition_names,
        quantiles_used,
        output_config.transitions,
        output_config.age_groups,
    )
    transition_results = TransitionResults(dates=dates, data=trans_data)

    # Compute summary statistics
    summary = compute_summary(results, output_config.summary)

    # Include raw trajectories if requested
    trajectories = None
    if output_config.include_trajectories:
        trajectories = extract_trajectories(
            results, output_config, compartment_names, transition_names
        )

    return SimulationResultsData(
        compartments=compartment_results,
        transitions=transition_results,
        summary=summary,
        trajectories=trajectories,
    )
