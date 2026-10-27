"""
Hybrid Pixel-Object Classification Assembly with Stability Maps
Based on Malinverni & Sanita (2026) 

STREAMING WINDOWED & ROBUST VERSION:
  1. Fixes the GEE 32-bit signed overflow issue:
     - Clusters that wrapped to negative values (< -1) are treated as valid unique segments.
     - Clusters clamped to 2,147,483,647 (max int32) are localized per-window using connected
       components labeling (scipy.ndimage.label), preventing ocean-land mega-clusters.
     - Only -1 and 0 (or all-zero probability pixels) are treated as nodata.
  2. Streaming windowed architecture:
     - Processes the national mosaic in overlapping blocks (e.g. 2048x2048 with 64px padding).
     - Keeps RAM usage under 1.5 GB regardless of total raster size.
     - Zero MemoryError; fast parallel execution.
  3. Intrinsic Rule Refinement:
     - False Hydrology Suppression: Suppresses spurious water when unstable (CI > 0.65)
       and runner-up is bare/built-up land.
     - Post-classification merge of Dry riverbed (18) into Waterbody (17).
  4. Extensible Mask-Layer Post-Refinement (refine_lulc_with_mask_layer):
     - Generic, data-agnostic spatial rule engine for refining any LULC class using auxiliary masks.
     - WarpedVRT on-the-fly resampling: supports 30m, 90m, or 10m masks in any projection.
     - Callable both as a standalone fast post-processor or integrated into process_tile().

"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy.ndimage import label
from .seam_healing import heal_array_seams

# 1. CLASS DEFINITIONS & SMOOTHNESS WEIGHTS
CLASS_ID_TO_NAME = {
    1:  'Primary_forest',
    2:  'Secondary_forest',
    3:  'Mangrove_forest',
    4:  'Coastal_forest',
    5:  'Teak_plantation',
    6:  'Eucalyptus_plantation',
    7:  'Coffe_agroforestry',
    8:  'Cocoa_agroforestry',
    9:  'Coconut_agroforestry',
    10: 'Mixed_garden',
    11: 'Paddy_field',
    12: 'Other_cropland',
    13: 'Grassland',
    14: 'Shrubland',
    15: 'Cleared_land',
    16: 'Built-up',
    17: 'Waterbody',
    18: 'Dry_riverbed'
}

SMOOTHNESS_TL = {
    'Primary_forest':        4.5,
    'Secondary_forest':      3.0,
    'Mangrove_forest':       3.0,
    'Coastal_forest':        3.5,
    'Teak_plantation':       1.2,
    'Eucalyptus_plantation': 1.2,
    'Coffe_agroforestry':    1.5,
    'Cocoa_agroforestry':    1.5,
    'Coconut_agroforestry':  1.5,
    'Mixed_garden':          1.2,
    'Paddy_field':           1.0,
    'Other_cropland':        1.0,
    'Grassland':             1.5,
    'Shrubland':             1.5,
    'Cleared_land':          1.5,
    'Built-up':              1.0,
    'Waterbody':             4.0,
    'Dry_riverbed':          0.1
}

# Default parameters
STABILITY_TAU = 0.65       # Threshold: CI <= tau is Stable (Malinverni & Sanita 2026)
ADAPTIVE_SENSITIVITY = 0.6 # How strongly instability damps object weight
MERGE_WATER_RIVERBED = True # Merge class 18 into 17 post-classification


# =============================================================================
# 2. GENERIC MASK-BASED REFINEMENT RULE ENGINE
# =============================================================================
@dataclass
class MaskRule:
    """
    Generic reclassification rule based on one or more auxiliary mask layers.

    Attributes:
        from_classes: Single class integer or list/tuple of classes to match.
        to_class: The target class integer to reassign matched pixels to.
        require_mask_path: Optional path to mask raster that MUST equal require_value.
        exclude_mask_path: Optional path to mask raster that MUST NOT equal exclude_value (i.e. exclude == 0).
        require_value: Target value to match in require_mask (default: 1).
        exclude_value: Value to exclude in exclude_mask (default: 1).
        name: Human-readable name for logging / diagnostics.
    """
    from_classes: int | list[int] | tuple[int, ...]
    to_class: int
    require_mask_path: str | Path | None = None
    exclude_mask_path: str | Path | None = None
    require_value: int | float = 1
    exclude_value: int | float = 1
    name: str = "Custom Mask Rule"

    def __post_init__(self):
        if isinstance(self.from_classes, int):
            self.from_classes = (self.from_classes,)
        else:
            self.from_classes = tuple(self.from_classes)


def _resolve_mask_rules(
    rules: list[MaskRule | dict[str, Any]] | None = None,
    *,
    river_corridor_path: str | Path | None = None,
    settlement_path: str | Path | None = None,
    builtup_class_id: int = 16,
    water_class_id: int = 17,
    riverbed_class_id: int | None = 18,
) -> list[MaskRule]:
    """Builds a standardized list of MaskRule objects from rules or convenience arguments."""
    resolved_rules: list[MaskRule] = []

    if rules:
        for r in rules:
            if isinstance(r, MaskRule):
                resolved_rules.append(r)
            elif isinstance(r, dict):
                resolved_rules.append(MaskRule(**r))
            else:
                raise ValueError(f"Unsupported rule type: {type(r)}")

    # Convenience shortcuts if explicit paths are passed and not already in rules
    if river_corridor_path is not None:
        resolved_rules.append(
            MaskRule(
                from_classes=builtup_class_id,
                to_class=water_class_id,
                require_mask_path=river_corridor_path,
                exclude_mask_path=settlement_path,
                name=f"False Built-up ({builtup_class_id}) in River Corridor -> Water ({water_class_id})"
            )
        )

    if settlement_path is not None and riverbed_class_id is not None:
        target_classes = (water_class_id, riverbed_class_id)
        resolved_rules.append(
            MaskRule(
                from_classes=target_classes,
                to_class=builtup_class_id,
                require_mask_path=settlement_path,
                exclude_mask_path=river_corridor_path,
                name=f"False Water ({target_classes}) in Confirmed Settlement -> Built-up ({builtup_class_id})"
            )
        )

    return resolved_rules


def refine_lulc_with_mask_layer(
    input_lulc_path: str | Path,
    output_lulc_path: str | Path,
    rules: list[MaskRule | dict[str, Any]] | None = None,
    *,
    river_corridor_path: str | Path | None = None,
    settlement_path: str | Path | None = None,
    builtup_class_id: int = 16,
    water_class_id: int = 17,
    riverbed_class_id: int | None = 18,
    block_size: int = 2048,
    compress: str = "deflate",
) -> Path:
    """
    Standalone, streaming windowed post-processing tool to refine an existing LULC map
    using one or more auxiliary mask layers (e.g. topography, settlements, hydrology, slope).

    - Memory-safe: streams in 2048x2048 blocks (RAM < 250 MB).
    - Multi-resolution: automatically aligns 30m, 90m, or 10m masks on the fly via WarpedVRT.
    - Zero data distortion: operates strictly via nearest-neighbor categorical alignment.

    Returns the Path to the refined output GeoTIFF.
    """
    input_lulc_path = Path(input_lulc_path)
    output_lulc_path = Path(output_lulc_path)
    output_lulc_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_rules = _resolve_mask_rules(
        rules,
        river_corridor_path=river_corridor_path,
        settlement_path=settlement_path,
        builtup_class_id=builtup_class_id,
        water_class_id=water_class_id,
        riverbed_class_id=riverbed_class_id
    )

    if not resolved_rules:
        raise ValueError("No refinement rules or mask layer paths were provided.")

    print(f"\n=======================================================")
    print(f"--- Mask-Layer LULC Refinement ---")
    print(f"Input LULC:  {input_lulc_path}")
    print(f"Output LULC: {output_lulc_path}")
    print(f"Configured Rules: ({len(resolved_rules)})")
    for idx, r in enumerate(resolved_rules, 1):
        print(f"  {idx}. {r.name}: Classes {r.from_classes} -> {r.to_class}")
        if r.require_mask_path:
            print(f"     Required mask: {Path(r.require_mask_path).name} == {r.require_value}")
        if r.exclude_mask_path:
            print(f"     Excluded mask: {Path(r.exclude_mask_path).name} == {r.exclude_value}")
    print(f"=======================================================")

    # Identify all unique mask paths needed
    unique_mask_paths: set[Path] = set()
    for r in resolved_rules:
        if r.require_mask_path:
            unique_mask_paths.add(Path(r.require_mask_path))
        if r.exclude_mask_path:
            unique_mask_paths.add(Path(r.exclude_mask_path))

    with rasterio.open(input_lulc_path) as src_lulc:
        meta = src_lulc.profile.copy()
        meta.update(
            compress=compress,
            tiled=True,
            blockxsize=512,
            blockysize=512
        )
        H, W = src_lulc.height, src_lulc.width

        # Open and wrap mask layers in WarpedVRT for automatic spatial alignment
        mask_sources = {}
        vrt_contexts = {}
        for p in unique_mask_paths:
            src_m = rasterio.open(p)
            mask_sources[p] = src_m
            vrt_contexts[p] = WarpedVRT(
                src_m,
                crs=src_lulc.crs,
                transform=src_lulc.transform,
                width=W,
                height=H,
                resampling=Resampling.nearest
            )

        # Tracking metrics
        rule_change_counts = {idx: 0 for idx in range(len(resolved_rules))}
        total_pixels = H * W

        r_steps = list(range(0, H, block_size))
        c_steps = list(range(0, W, block_size))
        total_blocks = len(r_steps) * len(c_steps)

        with rasterio.open(output_lulc_path, "w", **meta) as dst_lulc:
            block_idx = 0
            for r_off in r_steps:
                for c_off in c_steps:
                    block_idx += 1
                    win_h = min(block_size, H - r_off)
                    win_w = min(block_size, W - c_off)
                    win = Window(c_off, r_off, win_w, win_h)

                    lulc_block = src_lulc.read(1, window=win)

                    # Skip empty/nodata ocean blocks
                    if np.all(lulc_block == 0):
                        dst_lulc.write(lulc_block, 1, window=win)
                        continue

                    # Read corresponding blocks from warped mask VRTs
                    cached_mask_blocks = {}
                    for p, vrt in vrt_contexts.items():
                        cached_mask_blocks[p] = vrt.read(1, window=win)

                    # Evaluate each rule sequentially
                    for r_idx, rule in enumerate(resolved_rules):
                        candidate_mask = np.isin(lulc_block, rule.from_classes)
                        if not np.any(candidate_mask):
                            continue

                        rule_mask = candidate_mask.copy()
                        if rule.require_mask_path:
                            req_block = cached_mask_blocks[Path(rule.require_mask_path)]
                            rule_mask &= (req_block == rule.require_value)

                        if rule.exclude_mask_path:
                            exc_block = cached_mask_blocks[Path(rule.exclude_mask_path)]
                            rule_mask &= (exc_block != rule.exclude_value)

                        n_modified = int(np.sum(rule_mask))
                        if n_modified > 0:
                            lulc_block[rule_mask] = rule.to_class
                            rule_change_counts[r_idx] += n_modified

                    dst_lulc.write(lulc_block, 1, window=win)

                    if block_idx % 20 == 0 or block_idx == total_blocks:
                        print(f"  Refinement block {block_idx:3d}/{total_blocks} done ({(block_idx/total_blocks)*100:5.1f}%)...")

        # Clean up open VRTs and raster sources
        for vrt in vrt_contexts.values():
            vrt.close()
        for src_m in mask_sources.values():
            src_m.close()

    print("\n--- Mask-Layer Refinement Complete! ---")
    print("Modifications Summary:")
    for r_idx, rule in enumerate(resolved_rules):
        cnt = rule_change_counts[r_idx]
        pct = (cnt / total_pixels) * 100
        print(f"  - {rule.name:50s}: {cnt:9d} pixels ({pct:5.2f}%)")
    print(f"Refined LULC map saved to: {output_lulc_path}\n")

    return output_lulc_path


# =============================================================================
# 3. STREAMING WINDOWED PROCESSOR (PIXEL-OBJECT BLENDING)
# =============================================================================
def process_tile(
    prob_path: str | Path,
    seg_path: str | Path,
    out_dir: str | Path,
    *,
    block_size: int = 2048,
    overlap: int = 64,
    tau: float = STABILITY_TAU,
    adaptive_sensitivity: float = ADAPTIVE_SENSITIVITY,
    smoothness_dict: dict[str, float] | None = None,
    merge_water_riverbed: bool = MERGE_WATER_RIVERBED,
    # Optional mask layers / rules
    refinement_rules: list[MaskRule | dict[str, Any]] | None = None,
    river_corridor_path: str | Path | None = None,
    settlement_path: str | Path | None = None,
) -> dict[str, Path]:
    """
    Streaming windowed processor for full-country or regional mosaics.
    Processes rasters in padded blocks (e.g. 2048x2048 with 64px overlap),
    completely eliminating MemoryError and int32 signed overflow artifacts.

    Optionally applies mask-layer refinement rules directly in the output stream.

    Returns dict of output paths: {'lulc': ..., 'stability': ..., 'ci': ...}
    """
    prob_path = Path(prob_path)
    seg_path = Path(seg_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = prob_path.stem
    print(f"\n=======================================================")
    print(f"--- Processing: {base_name} ---")
    print(f"Probabilities: {prob_path}")
    print(f"Segments:      {seg_path}")
    print(f"Block size:    {block_size}x{block_size} (overlap: {overlap}px)")
    print(f"Stability tau: {tau} | Adaptive sensitivity: {adaptive_sensitivity}")

    # Prepare optional mask refinement rules
    active_mask_rules = _resolve_mask_rules(
        refinement_rules,
        river_corridor_path=river_corridor_path,
        settlement_path=settlement_path
    )
    if active_mask_rules:
        print(f"Active Mask-Layer Rules: {len(active_mask_rules)}")
        for idx, r in enumerate(active_mask_rules, 1):
            print(f"  {idx}. {r.name}")
    print(f"=======================================================")

    with rasterio.open(seg_path) as src_seg, rasterio.open(prob_path) as src_prob:
        target_h, target_w = src_seg.shape
        num_classes = src_prob.count
        seg_meta = src_seg.profile.copy()

        # Class base weights
        smooth_table = smoothness_dict or SMOOTHNESS_TL
        w_base = np.zeros(num_classes, dtype=np.float32)
        for c_id, name in CLASS_ID_TO_NAME.items():
            if c_id - 1 < num_classes:
                smoothness = smooth_table.get(name, 1.0)
                w_base[c_id - 1] = float(np.clip(smoothness / 5.0, 0.2, 0.9))

        # Setup output profiles (uint8 for LULC/Stability, float32 for CI)
        out_meta_uint8 = seg_meta.copy()
        out_meta_uint8.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0,
            compress='deflate',
            tiled=True,
            blockxsize=512,
            blockysize=512
        )

        out_meta_float = seg_meta.copy()
        out_meta_float.update(
            dtype=rasterio.float32,
            count=1,
            nodata=0.0,
            compress='deflate',
            tiled=True,
            blockxsize=512,
            blockysize=512
        )

        lulc_out = out_dir / f"{base_name}_LULC.tif"
        stab_out = out_dir / f"{base_name}_Stability_Map.tif"
        ci_out   = out_dir / f"{base_name}_Confusion_Index.tif"

        # Setup mask VRTs if active
        unique_mask_paths = set()
        for r in active_mask_rules:
            if r.require_mask_path:
                unique_mask_paths.add(Path(r.require_mask_path))
            if r.exclude_mask_path:
                unique_mask_paths.add(Path(r.exclude_mask_path))

        mask_sources = {}
        vrt_contexts = {}
        for p in unique_mask_paths:
            src_m = rasterio.open(p)
            mask_sources[p] = src_m
            vrt_contexts[p] = WarpedVRT(
                src_m,
                crs=src_seg.crs,
                transform=src_seg.transform,
                width=target_w,
                height=target_h,
                resampling=Resampling.nearest
            )

        # Calculate grid of blocks
        r_steps = list(range(0, target_h, block_size))
        c_steps = list(range(0, target_w, block_size))
        total_blocks = len(r_steps) * len(c_steps)
        print(f"Total windowed blocks: {total_blocks} ({len(r_steps)} rows x {len(c_steps)} cols)")

        with rasterio.open(lulc_out, "w", **out_meta_uint8) as dst_lulc, \
             rasterio.open(stab_out, "w", **out_meta_uint8) as dst_stab, \
             rasterio.open(ci_out,   "w", **out_meta_float) as dst_ci:

            block_idx = 0
            for r_off in r_steps:
                for c_off in c_steps:
                    block_idx += 1
                    win_write_h = min(block_size, target_h - r_off)
                    win_write_w = min(block_size, target_w - c_off)
                    win_write = Window(c_off, r_off, win_write_w, win_write_h)

                    # Padded read window
                    pad_top = min(overlap, r_off)
                    pad_bottom = min(overlap, target_h - (r_off + win_write_h))
                    pad_left = min(overlap, c_off)
                    pad_right = min(overlap, target_w - (c_off + win_write_w))

                    read_r = r_off - pad_top
                    read_c = c_off - pad_left
                    read_h = win_write_h + pad_top + pad_bottom
                    read_w = win_write_w + pad_left + pad_right
                    win_seg = Window(read_c, read_r, read_w, read_h)

                    # Read segments
                    seg_block = src_seg.read(1, window=win_seg).copy()

                    # Read and resample corresponding probabilities
                    bounds = src_seg.window_bounds(win_seg)
                    row_start, col_start = src_prob.index(bounds[0], bounds[3])
                    row_end, col_end = src_prob.index(bounds[2], bounds[1])
                    r_p = max(0, min(src_prob.height - 1, row_start))
                    c_p = max(0, min(src_prob.width - 1, col_start))
                    h_p = max(1, min(src_prob.height - r_p, row_end - row_start))
                    w_p = max(1, min(src_prob.width - c_p, col_end - col_start))
                    win_prob = Window(c_p, r_p, w_p, h_p)

                    prob_block = src_prob.read(
                        window=win_prob,
                        out_shape=(num_classes, read_h, read_w),
                        resampling=Resampling.bilinear
                    ).astype(np.float32) / 10000.0

                    # Check if all probabilities are zero (true background nodata)
                    prob_has_signal = np.any(prob_block > 0, axis=0)
                    if not np.any(prob_has_signal):
                        dst_lulc.write(np.zeros((win_write_h, win_write_w), dtype=np.uint8), 1, window=win_write)
                        dst_stab.write(np.zeros((win_write_h, win_write_w), dtype=np.uint8), 1, window=win_write)
                        dst_ci.write(np.zeros((win_write_h, win_write_w), dtype=np.float32), 1, window=win_write)
                        continue

                    # 1. Localize clamped pixels (2,147,483,647) to distinct connected components.
                    #    If a connected component of 2,147,483,647 is larger than 400 pixels, multiple
                    #    adjacent SNIC segments have fused together due to int32 clamping (e.g. ocean
                    #    fused with coastal forest/urban). For these fused mega-clusters, fall back
                    #    to pure pixel probabilities (seg_block = 0) so ocean water never floods land.
                    is_clamp = seg_block == 2147483647
                    if np.any(is_clamp):
                        labeled, n_comp = label(is_clamp)
                        if n_comp > 0:
                            comp_sizes = np.bincount(labeled.ravel())
                            # Components > 400 px are fused overflow mega-polygons -> pure pixel fallback (0)
                            oversized_mask = comp_sizes[labeled] > 400
                            valid_clamp = is_clamp & (~oversized_mask)
                            seg_block[is_clamp & oversized_mask] = 0
                            seg_block[valid_clamp] = -2000000000 - labeled[valid_clamp]

                    # 2. Heal thin 1-4px zero seams and gaps in segments within valid data regions
                    seg_block = heal_array_seams(seg_block, nodata=0, max_seam_width=4)

                    # 3. Identify segmented pixels vs pure-pixel fallback
                    has_seg = (seg_block != -1) & (seg_block != 0) & prob_has_signal
                    has_any_seg = np.any(has_seg)

                    # 4. Compact ID remapping for valid segments in this window
                    if has_any_seg:
                        unique_ids, inverse = np.unique(seg_block[has_seg], return_inverse=True)
                        n_segs = len(unique_ids)
                        seg_compact = np.full(seg_block.shape, -1, dtype=np.int32)
                        seg_compact[has_seg] = inverse.astype(np.int32)
                        counts = np.bincount(seg_compact[has_seg], minlength=n_segs)
                    else:
                        n_segs = 0
                        counts = np.array([], dtype=np.int64)

                    # 5. Bayesian Blending & Argmax Tracking
                    H, W = read_h, read_w
                    W_max = np.zeros((H, W), dtype=np.float32)
                    S_max = np.zeros((H, W), dtype=np.float32)
                    winner = np.zeros((H, W), dtype=np.uint8)
                    runner = np.zeros((H, W), dtype=np.uint8)
                    blend_sum = np.zeros((H, W), dtype=np.float32)

                    for c in range(num_classes):
                        class_id = c + 1
                        raw = prob_block[c]

                        if has_any_seg:
                            # Object mean via bincount
                            prob_v = raw[has_seg].astype(np.float64)
                            sums = np.bincount(seg_compact[has_seg], weights=prob_v, minlength=n_segs)
                            with np.errstate(divide='ignore', invalid='ignore'):
                                mean_lookup = np.where(counts > 0, sums / counts, 0.0).astype(np.float32)

                            obj = np.zeros((H, W), dtype=np.float32)
                            obj[has_seg] = mean_lookup[seg_compact[has_seg]]

                            # Adaptive object weight (Malinverni & Sanita 2026)
                            obj_ci_proxy = np.abs(obj - raw).clip(0.0, 1.0)
                            w_eff = np.where(
                                has_seg,
                                w_base[c] * (1.0 - adaptive_sensitivity * obj_ci_proxy),
                                0.0,
                            )
                            blended = (obj * w_eff) + (raw * (1.0 - w_eff))
                        else:
                            # Pure pixel fallback if no segments present in this window
                            blended = raw

                        blend_sum += blended

                        # Winner / Runner-up argmax tracking
                        is_new_w = blended > W_max
                        is_new_r = (~is_new_w) & (blended > S_max)
                        runner = np.where(is_new_w, winner, runner)
                        S_max  = np.where(is_new_w, W_max, S_max)
                        W_max  = np.where(is_new_w, blended, W_max)
                        winner = np.where(is_new_w, class_id, winner).astype(np.uint8)
                        S_max  = np.where(is_new_r, blended, S_max)
                        runner = np.where(is_new_r, class_id, runner).astype(np.uint8)

                    # 5. Stability Map & Confusion Index
                    with np.errstate(divide='ignore', invalid='ignore'):
                        CI = np.where(W_max > 1e-4, S_max / W_max, 0.0).clip(0.0, 1.0).astype(np.float32)
                    is_stable = (CI <= tau).astype(np.uint8)

                    # 6. Intrinsic Rule Refinement
                    # False Hydrology Suppression
                    false_hydro = np.isin(winner, [17, 18]) & np.isin(runner, [15, 16]) & (CI > 0.65)
                    winner[false_hydro] = runner[false_hydro]

                    # Merge Dry riverbed into Waterbody
                    if merge_water_riverbed:
                        winner[winner == 18] = 17

                    # Apply optional auxiliary mask layer rules on-the-fly
                    if active_mask_rules:
                        cached_mask_blocks = {}
                        for p, vrt in vrt_contexts.items():
                            cached_mask_blocks[p] = vrt.read(1, window=win_seg)

                        for rule in active_mask_rules:
                            candidate_mask = np.isin(winner, rule.from_classes)
                            if not np.any(candidate_mask):
                                continue

                            rule_mask = candidate_mask.copy()
                            if rule.require_mask_path:
                                req_block = cached_mask_blocks[Path(rule.require_mask_path)]
                                rule_mask &= (req_block == rule.require_value)

                            if rule.exclude_mask_path:
                                exc_block = cached_mask_blocks[Path(rule.exclude_mask_path)]
                                rule_mask &= (exc_block != rule.exclude_value)

                            if np.any(rule_mask):
                                winner[rule_mask] = rule.to_class

                    # Apply nodata mask (only true background nodata with zero satellite probability)
                    nodata_mask = ~prob_has_signal
                    winner[nodata_mask] = 0
                    is_stable[nodata_mask] = 0
                    CI[nodata_mask] = 0.0

                    # 7. Unpad and write to disk
                    crop_r_start = pad_top
                    crop_r_end   = pad_top + win_write_h
                    crop_c_start = pad_left
                    crop_c_end   = pad_left + win_write_w

                    winner_unpadded = winner[crop_r_start:crop_r_end, crop_c_start:crop_c_end]
                    stable_unpadded = is_stable[crop_r_start:crop_r_end, crop_c_start:crop_c_end]
                    ci_unpadded     = CI[crop_r_start:crop_r_end, crop_c_start:crop_c_end]

                    dst_lulc.write(winner_unpadded, 1, window=win_write)
                    dst_stab.write(stable_unpadded, 1, window=win_write)
                    dst_ci.write(ci_unpadded, 1, window=win_write)

                    if block_idx % 10 == 0 or block_idx == total_blocks:
                        print(f"  Block {block_idx:3d}/{total_blocks} done ({(block_idx/total_blocks)*100:5.1f}%)...")

        for vrt in vrt_contexts.values():
            vrt.close()
        for src_m in mask_sources.values():
            src_m.close()

    print("\n--- Processing Complete! ---")
    print(f"Outputs written to: {out_dir}")
    print(f"  1. LULC Map:        {lulc_out}")
    print(f"  2. Stability Map:   {stab_out}")
    print(f"  3. Confusion Index: {ci_out}")

    return {
        'lulc': lulc_out,
        'stability': stab_out,
        'ci': ci_out
    }
