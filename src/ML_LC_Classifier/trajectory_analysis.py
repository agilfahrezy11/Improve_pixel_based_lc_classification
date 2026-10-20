"""Multi-temporal trajectory analysis, annualized change rates, and eco-functional change mapping.

Designed for multi-epoch land cover time series (e.g. 2005, 2015, 2020, 2025)
in Timor-Leste and tropical smallholder landscapes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import rasterio.windows

from .lc_post_process import DEFAULT_TIMOR_17_CLASSES

RasterPath = str | PathLike[str]

#: Extended 18-class dictionary including Dry_riverbed (Class 18)
DEFAULT_TIMOR_18_CLASSES: dict[int, str] = {
    1: "Primary_forest",
    2: "Secondary_forest",
    3: "Mangrove_forest",
    4: "Coastal_forest",
    5: "Teak_plantation",
    6: "Eucalyptus_plantation",
    7: "Coffe_agroforestry",
    8: "Cocoa_agroforestry",
    9: "Coconut_agroforestry",
    10: "Mixed_garden",
    11: "Paddy_field",
    12: "Other_cropland",
    13: "Grassland",
    14: "Shrubland",
    15: "Cleared_land",
    16: "Built-up",
    17: "Waterbody",
    18: "Dry_riverbed",
}

#: Trajectory code definitions for multi-year landscape dynamics
TRAJECTORY_CLASSES: dict[int, str] = {
    1: "Persistent / Stable Matrix",
    2: "Unidirectional Forest Loss",
    3: "Forest Regrowth / Succession",
    4: "Rotational Swidden / Fallow Cycle",
    5: "Agroforestry / Plantation Adoption",
    6: "Urban / Built-up Expansion",
    7: "Water / Riverbed Dynamics",
    8: "Other Multi-Directional Changes",
    9: "Potential Spurious Noise (QA Flag)",
}


@dataclass
class IntervalTransitionSummary:
    """Summary of changes for a single temporal interval (t1 -> t2)."""

    t1: int
    t2: int
    interval_years: float
    transition_matrix_ha: pd.DataFrame
    class_summary_ha: pd.DataFrame


@dataclass
class MultiTemporalTrajectoryResult:
    """Complete result container for multi-epoch trajectory analysis."""

    epochs: list[int]
    interval_summaries: list[IntervalTransitionSummary]
    net_summary_ha: IntervalTransitionSummary
    trajectory_counts: pd.DataFrame
    trajectory_raster_path: Path | None = None


def _calculate_interval_metrics(
    t1_arr: np.ndarray,
    t2_arr: np.ndarray,
    classes: Mapping[int, str],
    pixel_area_ha: float,
    t1_year: int,
    t2_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute cross-tabulation, Net/Swap components, and annual rates."""
    all_ids = sorted(classes.keys())
    max_id = max(max(all_ids, default=0), 255)
    labels = [classes[cid] for cid in all_ids]
    dt = max(float(t2_year - t1_year), 0.1)

    # 2D cross-tabulation
    encoded = t1_arr * (max_id + 1) + t2_arr
    flat_counts = np.bincount(encoded, minlength=(max_id + 1) ** 2)
    counts_2d = flat_counts.reshape((max_id + 1, max_id + 1))

    sub_counts = counts_2d[np.ix_(all_ids, all_ids)]
    sub_areas = np.round(sub_counts * pixel_area_ha, 4)

    matrix_df = pd.DataFrame(sub_areas, index=labels, columns=labels)
    matrix_df.index.name = f"From ({t1_year})"
    matrix_df.columns.name = f"To ({t2_year})"

    # Pontius & Millones Net vs. Swap analysis per class
    row_totals = sub_areas.sum(axis=1)  # Area at t1
    col_totals = sub_areas.sum(axis=0)  # Area at t2
    diagonal = np.diag(sub_areas)  # Persistence

    losses = row_totals - diagonal
    gains = col_totals - diagonal
    net_change = col_totals - row_totals
    swap = 2.0 * np.minimum(losses, gains)
    total_change = losses + gains

    # Puyravaud Compound Annual Rate of Change r = (1/dt) * ln(A2 / A1) * 100%
    # Protect against zero division or log(0)
    annual_rates = np.zeros(len(all_ids), dtype=np.float64)
    for idx, (a1, a2) in enumerate(zip(row_totals, col_totals)):
        if a1 > 0 and a2 > 0:
            annual_rates[idx] = np.round((1.0 / dt) * np.log(a2 / a1) * 100.0, 3)
        else:
            annual_rates[idx] = 0.0

    summary_df = pd.DataFrame(
        {
            f"Area_{t1_year}_ha": np.round(row_totals, 2),
            f"Area_{t2_year}_ha": np.round(col_totals, 2),
            "Persistence_ha": np.round(diagonal, 2),
            "Loss_ha": np.round(losses, 2),
            "Gain_ha": np.round(gains, 2),
            "Net_Change_ha": np.round(net_change, 2),
            "Swap_ha": np.round(swap, 2),
            "Total_Change_ha": np.round(total_change, 2),
            "Annual_Rate_pct_yr": annual_rates,
        },
        index=labels,
    )
    summary_df.index.name = "Class"

    return matrix_df, summary_df


def _classify_pixel_trajectory_vector(
    traj: np.ndarray,
) -> np.ndarray:
    """Classify 1D pixel histories into eco-functional trajectory categories.

    Parameters
    ----------
    traj : np.ndarray of shape (N, T)
        Class IDs across T chronological epochs for N valid pixels.

    Returns
    -------
    np.ndarray of shape (N,)
        Trajectory category IDs (1-9).
    """
    n_pixels, n_epochs = traj.shape
    out = np.full(n_pixels, 8, dtype=np.uint8)  # default: Other Multi-Directional

    # Set definitions based on Timor-Leste LULC taxonomy
    forest_set = {1, 2, 3, 4}  # Primary, Secondary, Mangrove, Coastal
    tree_crop_set = {5, 6, 7, 8, 9}  # Plantations & Agroforestry
    swidden_set = {10, 11, 12, 13, 14, 15}  # Gardens, Crops, Grass, Shrub, Cleared
    water_river_set = {17, 18}  # Waterbody & Dry riverbed

    c_first = traj[:, 0]
    c_last = traj[:, -1]

    # 1. Persistent / Stable Matrix (all epochs identical)
    stable_mask = np.all(traj == traj[:, [0]], axis=1)
    out[stable_mask] = 1

    # Unstable pixels evaluate rules
    unstable = ~stable_mask
    if not np.any(unstable):
        return out

    u_traj = traj[unstable]
    u_first = c_first[unstable]
    u_last = c_last[unstable]
    u_out = np.full(len(u_first), 8, dtype=np.uint8)

    in_forest_start = np.isin(u_first, list(forest_set))
    in_forest_end = np.isin(u_last, list(forest_set))

    in_swidden_start = np.isin(u_first, list(swidden_set))
    in_swidden_end = np.isin(u_last, list(swidden_set))

    in_treecrop_end = np.isin(u_last, list(tree_crop_set))

    # 2. Unidirectional Forest Loss: Started in Forest -> Ended in Cropland, Cleared, or Built-up
    loss_mask = in_forest_start & (in_swidden_end | (u_last == 16))
    u_out[loss_mask] = 2

    # 3. Forest Regrowth / Succession: Started in Swidden/Cleared/Grass -> Ended in Forest
    regrowth_mask = in_swidden_start & in_forest_end
    u_out[regrowth_mask] = 3

    # 4. Rotational Swidden / Fallow Cycle: All steps remain within swidden/fallow matrix
    # e.g., Cropland <-> Cleared <-> Shrubland <-> Mixed Garden
    swidden_all = np.all(np.isin(u_traj, list(swidden_set)), axis=1)
    u_out[swidden_all] = 4

    # 5. Agroforestry / Tree Plantation Adoption: Transitioned into Tree Crops
    agro_mask = (~in_treecrop_end) & in_treecrop_end  # started non-agro, ended agro
    agro_mask = (~np.isin(u_first, list(tree_crop_set))) & in_treecrop_end
    u_out[agro_mask] = 5

    # 6. Urban / Built-up Expansion: Non-built-up at start -> Built-up at end
    urban_mask = (u_first != 16) & (u_last == 16)
    u_out[urban_mask] = 6

    # 7. Water / Riverbed Dynamics: Transitions strictly between Water & Riverbeds
    water_dyn = np.all(np.isin(u_traj, list(water_river_set)), axis=1)
    u_out[water_dyn] = 7

    # 9. Potential Spurious Noise / Reversal QA Flag:
    # Example: Built-up -> Forest -> Built-up OR Forest -> Built-up -> Forest
    if n_epochs >= 3:
        # Check for rapid flip-flop back to initial state
        flip_back = (u_first == u_last) & (u_traj[:, 1] != u_first)
        # Built-up abandonment is physically rare in expanding towns
        spurious_urban = np.any((u_traj[:, :-1] == 16) & (u_traj[:, 1:] == 1), axis=1)
        u_out[flip_back | spurious_urban] = 9

    out[unstable] = u_out
    return out


def analyze_multi_temporal_transitions(
    epoch_rasters: Mapping[int, RasterPath],
    class_names: Mapping[int, str] | None = None,
    *,
    output_trajectory_raster: RasterPath | None = None,
    output_csv_dir: RasterPath | None = None,
    block_size: int = 2048,
) -> MultiTemporalTrajectoryResult:
    """Perform multi-epoch land-cover transition and trajectory analysis.

    Evaluates sequential periods (e.g., 2005->2015, 2015->2020, 2020->2025)
    and full net span (2005->2025). Calculates:
      1. Interval transition matrices (ha).
      2. Pontius Net vs. Swap components and Puyravaud annual change rates (%/yr).
      3. Pixel-level 20-year eco-functional trajectory categorization.
      4. Optional single-band trajectory map GeoTIFF (streamed block-by-block).

    Parameters
    ----------
    epoch_rasters : Mapping[int, RasterPath]
        Dictionary mapping year to GeoTIFF path, sorted chronologically:
        e.g. {2005: "lc_2005.tif", 2015: "lc_2015.tif", 2020: "lc_2020.tif", 2025: "lc_2025.tif"}
    class_names : Mapping[int, str] or None
        Class definitions. Defaults to DEFAULT_TIMOR_17_CLASSES or 18 classes.
    output_trajectory_raster : RasterPath or None
        Destination path to save the 9-class trajectory map GeoTIFF.
    output_csv_dir : RasterPath or None
        Folder path to export transition matrix CSVs.
    block_size : int
        Window block size for streaming large GeoTIFFs (default: 2048).
    """
    sorted_years = sorted(epoch_rasters.keys())
    if len(sorted_years) < 2:
        raise ValueError("At least two temporal epochs are required for transition analysis.")

    classes = (
        DEFAULT_TIMOR_18_CLASSES
        if (class_names is None and any(18 in c for c in [DEFAULT_TIMOR_18_CLASSES]))
        else (DEFAULT_TIMOR_17_CLASSES if class_names is None else dict(class_names))
    )

    raster_paths = [Path(epoch_rasters[yr]) for yr in sorted_years]

    # Verify matching spatial profiles
    with rasterio.open(raster_paths[0]) as first_src:
        meta = first_src.meta.copy()
        res_x = abs(first_src.transform.a)
        res_y = abs(first_src.transform.e)
        pixel_area_ha = (res_x * res_y) / 10000.0
        height, width = first_src.height, first_src.width

    for p in raster_paths[1:]:
        with rasterio.open(p) as s:
            if s.shape != (height, width):
                raise ValueError(
                    f"Dimension mismatch: {p.name} {s.shape} != base {height, width}."
                )

    # Accumulators for sequential interval transitions
    n_intervals = len(sorted_years) - 1
    # Store flattened pairs per interval to aggregate
    all_ids = sorted(classes.keys())
    max_id = max(max(all_ids, default=0), 255)

    interval_counts = [
        np.zeros((max_id + 1, max_id + 1), dtype=np.int64) for _ in range(n_intervals)
    ]
    net_counts = np.zeros((max_id + 1, max_id + 1), dtype=np.int64)
    trajectory_hist = np.zeros(len(TRAJECTORY_CLASSES) + 1, dtype=np.int64)

    # Setup trajectory writer if requested
    out_traj_path = Path(output_trajectory_raster) if output_trajectory_raster else None
    traj_writer = None
    if out_traj_path is not None:
        out_traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_meta = meta.copy()
        traj_meta.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=0,
            compress="deflate",
        )
        traj_writer = rasterio.open(out_traj_path, "w", **traj_meta)

    # Block-by-block streaming across all epochs simultaneously
    try:
        readers = [rasterio.open(p) for p in raster_paths]

        for y0 in range(0, height, block_size):
            y1 = min(y0 + block_size, height)
            for x0 in range(0, width, block_size):
                x1 = min(x0 + block_size, width)
                win = rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0)

                # Read window from all epoch rasters
                blocks = [r.read(1, window=win, masked=True) for r in readers]

                # Valid in ALL epochs
                valid = np.ones((y1 - y0, x1 - x0), dtype=bool)
                for b in blocks:
                    valid &= ~b.mask

                if not np.any(valid):
                    if traj_writer is not None:
                        empty = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                        traj_writer.write(empty, 1, window=win)
                    continue

                # Extract valid pixel vectors (N_pixels, N_epochs)
                epoch_data = []
                for b in blocks:
                    d = np.asarray(b.data)[valid].astype(np.int64)
                    epoch_data.append(d)

                traj_matrix = np.column_stack(epoch_data)

                # Filter positive class IDs
                pos_mask = np.all(traj_matrix > 0, axis=1)
                traj_matrix = traj_matrix[pos_mask]

                if len(traj_matrix) == 0:
                    if traj_writer is not None:
                        empty = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                        traj_writer.write(empty, 1, window=win)
                    continue

                # 1. Update sequential interval transition counts
                for i in range(n_intervals):
                    p1 = traj_matrix[:, i]
                    p2 = traj_matrix[:, i + 1]
                    enc = p1 * (max_id + 1) + p2
                    b_counts = np.bincount(enc, minlength=(max_id + 1) ** 2)
                    interval_counts[i] += b_counts.reshape((max_id + 1, max_id + 1))

                # 2. Update overall net span (first -> last)
                first_p = traj_matrix[:, 0]
                last_p = traj_matrix[:, -1]
                enc_net = first_p * (max_id + 1) + last_p
                net_b = np.bincount(enc_net, minlength=(max_id + 1) ** 2)
                net_counts += net_b.reshape((max_id + 1, max_id + 1))

                # 3. Classify trajectories
                t_cats = _classify_pixel_trajectory_vector(traj_matrix)
                cat_counts = np.bincount(t_cats, minlength=len(trajectory_hist))
                trajectory_hist += cat_counts

                # Write block to output raster if enabled
                if traj_writer is not None:
                    block_out = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                    # Map valid & positive back to spatial window
                    sub_coords = np.where(valid)
                    valid_y = sub_coords[0][pos_mask]
                    valid_x = sub_coords[1][pos_mask]
                    block_out[valid_y, valid_x] = t_cats
                    traj_writer.write(block_out, 1, window=win)

    finally:
        for r in readers:
            r.close()
        if traj_writer is not None:
            traj_writer.close()

    # Build Interval Transition Summaries
    labels = [classes[cid] for cid in all_ids]
    interval_summaries = []

    out_csv = Path(output_csv_dir) if output_csv_dir else None
    if out_csv:
        out_csv.mkdir(parents=True, exist_ok=True)

    for i in range(n_intervals):
        y_start = sorted_years[i]
        y_end = sorted_years[i + 1]
        dt = y_end - y_start

        sub_counts = interval_counts[i][np.ix_(all_ids, all_ids)]
        sub_areas = np.round(sub_counts * pixel_area_ha, 4)

        mat_df = pd.DataFrame(sub_areas, index=labels, columns=labels)
        mat_df.index.name = f"From ({y_start})"
        mat_df.columns.name = f"To ({y_end})"

        row_t = sub_areas.sum(axis=1)
        col_t = sub_areas.sum(axis=0)
        diag = np.diag(sub_areas)
        loss = row_t - diag
        gain = col_t - diag
        net = col_t - row_t
        swap = 2.0 * np.minimum(loss, gain)
        tot = loss + gain

        rates = np.zeros(len(all_ids), dtype=np.float64)
        for idx, (a1, a2) in enumerate(zip(row_t, col_t)):
            rates[idx] = (
                np.round((1.0 / dt) * np.log(a2 / a1) * 100.0, 3)
                if (a1 > 0 and a2 > 0)
                else 0.0
            )

        sum_df = pd.DataFrame(
            {
                f"Area_{y_start}_ha": np.round(row_t, 2),
                f"Area_{y_end}_ha": np.round(col_t, 2),
                "Persistence_ha": np.round(diag, 2),
                "Loss_ha": np.round(loss, 2),
                "Gain_ha": np.round(gain, 2),
                "Net_Change_ha": np.round(net, 2),
                "Swap_ha": np.round(swap, 2),
                "Total_Change_ha": np.round(tot, 2),
                "Annual_Rate_pct_yr": rates,
            },
            index=labels,
        )
        sum_df.index.name = "Class"

        if out_csv:
            mat_df.to_csv(out_csv / f"transitions_{y_start}_{y_end}_matrix.csv")
            sum_df.to_csv(out_csv / f"transitions_{y_start}_{y_end}_summary.csv")

        interval_summaries.append(
            IntervalTransitionSummary(
                t1=y_start,
                t2=y_end,
                interval_years=float(dt),
                transition_matrix_ha=mat_df,
                class_summary_ha=sum_df,
            )
        )

    # Net overall (first -> last)
    y_first = sorted_years[0]
    y_last = sorted_years[-1]
    dt_net = y_last - y_first
    net_sub_counts = net_counts[np.ix_(all_ids, all_ids)]
    net_sub_areas = np.round(net_sub_counts * pixel_area_ha, 4)

    net_mat_df = pd.DataFrame(net_sub_areas, index=labels, columns=labels)
    net_mat_df.index.name = f"From ({y_first})"
    net_mat_df.columns.name = f"To ({y_last})"

    row_n = net_sub_areas.sum(axis=1)
    col_n = net_sub_areas.sum(axis=0)
    diag_n = np.diag(net_sub_areas)
    loss_n = row_n - diag_n
    gain_n = col_n - diag_n
    net_diff = col_n - row_n
    swap_n = 2.0 * np.minimum(loss_n, gain_n)
    tot_n = loss_n + gain_n

    net_rates = np.zeros(len(all_ids), dtype=np.float64)
    for idx, (a1, a2) in enumerate(zip(row_n, col_n)):
        net_rates[idx] = (
            np.round((1.0 / dt_net) * np.log(a2 / a1) * 100.0, 3)
            if (a1 > 0 and a2 > 0)
            else 0.0
        )

    net_sum_df = pd.DataFrame(
        {
            f"Area_{y_first}_ha": np.round(row_n, 2),
            f"Area_{y_last}_ha": np.round(col_n, 2),
            "Persistence_ha": np.round(diag_n, 2),
            "Loss_ha": np.round(loss_n, 2),
            "Gain_ha": np.round(gain_n, 2),
            "Net_Change_ha": np.round(net_diff, 2),
            "Swap_ha": np.round(swap_n, 2),
            "Total_Change_ha": np.round(tot_n, 2),
            "Annual_Rate_pct_yr": net_rates,
        },
        index=labels,
    )
    net_sum_df.index.name = "Class"

    if out_csv:
        net_mat_df.to_csv(out_csv / f"transitions_{y_first}_{y_last}_net_matrix.csv")
        net_sum_df.to_csv(out_csv / f"transitions_{y_first}_{y_last}_net_summary.csv")

    net_summary = IntervalTransitionSummary(
        t1=y_first,
        t2=y_last,
        interval_years=float(dt_net),
        transition_matrix_ha=net_mat_df,
        class_summary_ha=net_sum_df,
    )

    # Trajectory Summary DataFrame
    traj_labels = [TRAJECTORY_CLASSES[k] for k in sorted(TRAJECTORY_CLASSES.keys())]
    traj_areas = [
        np.round(trajectory_hist[k] * pixel_area_ha, 2)
        for k in sorted(TRAJECTORY_CLASSES.keys())
    ]
    total_traj_area = max(sum(traj_areas), 1e-6)
    traj_pcts = [np.round((a / total_traj_area) * 100.0, 2) for a in traj_areas]

    df_trajectories = pd.DataFrame(
        {
            "Trajectory_Category": traj_labels,
            "Area_ha": traj_areas,
            "Share_pct": traj_pcts,
        },
        index=sorted(TRAJECTORY_CLASSES.keys()),
    )
    df_trajectories.index.name = "Trajectory_ID"

    if out_csv:
        df_trajectories.to_csv(out_csv / "temporal_trajectories_summary.csv")

    return MultiTemporalTrajectoryResult(
        epochs=sorted_years,
        interval_summaries=interval_summaries,
        net_summary_ha=net_summary,
        trajectory_counts=df_trajectories,
        trajectory_raster_path=out_traj_path,
    )


def export_multi_epoch_sankey(
    result: MultiTemporalTrajectoryResult,
    output_html_path: RasterPath,
    top_transitions_per_step: int = 15,
    min_area_ha: float = 100.0,
) -> Path:
    """Generate an interactive HTML multi-stage Sankey diagram across all epochs.

    Visualizes alluvial flows between 2005 -> 2015 -> 2020 -> 2025.
    Requires ``plotly``.

    Parameters
    ----------
    result : MultiTemporalTrajectoryResult
        Output returned by ``analyze_multi_temporal_transitions``.
    output_html_path : RasterPath
        Destination path for interactive HTML file.
    top_transitions_per_step : int
        Number of leading class flows to display per step (prevents clutter).
    min_area_ha : float
        Minimum transition area threshold in hectares to include in flow.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as err:
        raise ImportError(
            "Plotly is required for Sankey generation. Install with: pip install plotly"
        ) from err

    epochs = result.epochs
    nodes = []
    node_map = {}  # (epoch_idx, class_name) -> global_node_id

    # Create nodes per epoch
    global_id = 0
    for e_idx, yr in enumerate(epochs):
        # Retrieve unique classes active in this epoch
        classes_in_epoch = result.net_summary_ha.class_summary_ha.index.tolist()
        for cname in classes_in_epoch:
            node_map[(e_idx, cname)] = global_id
            nodes.append(f"{cname} ({yr})")
            global_id += 1

    sources = []
    targets = []
    values = []

    for e_idx, interval in enumerate(result.interval_summaries):
        mat = interval.transition_matrix_ha
        for from_cls in mat.index:
            for to_cls in mat.columns:
                val = mat.loc[from_cls, to_cls]
                if val >= min_area_ha:
                    s_id = node_map[(e_idx, from_cls)]
                    t_id = node_map[(e_idx + 1, to_cls)]
                    sources.append(s_id)
                    targets.append(t_id)
                    values.append(val)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                ),
            )
        ]
    )

    epoch_str = " \u2192 ".join(map(str, epochs))
    fig.update_layout(
        title_text=f"Timor-Leste Multi-Epoch Land Cover Transitions ({epoch_str})",
        font_size=11,
    )

    out_file = Path(output_html_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_file))
    return out_file
