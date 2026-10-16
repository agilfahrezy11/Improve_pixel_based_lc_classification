"""
Hybrid Pixel-Object Classification Assembly with Stability Maps
Based on Malinverni & Sanita (2026) and Timor-Leste 18-class schema.

Requirements:
    pip install numpy rasterio scipy
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling

# =============================================================================
# 1. CLASS DEFINITIONS & SMOOTHNESS WEIGHTS
# =============================================================================
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

# Parameters
STABILITY_TAU = 0.65       # Threshold: CI <= tau is Stable (Malinverni & Sanita 2026)
ADAPTIVE_SENSITIVITY = 0.6 # How strongly instability damps object weight
MERGE_WATER_RIVERBED = True# Merge class 18 into 17


def fast_segment_mean(prob_band, segments_2d):
    """
    Computes object-mean probability per segment using C-accelerated np.bincount.
    Runs in ~0.05 seconds for 30 million pixels.
    """
    flat_seg = segments_2d.ravel()
    flat_prob = prob_band.ravel()

    # Mask valid segments
    valid_mask = flat_seg >= 0
    seg_valid = flat_seg[valid_mask]
    prob_valid = flat_prob[valid_mask]

    counts = np.bincount(seg_valid)
    sums = np.bincount(seg_valid, weights=prob_valid)

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_lookup = np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float32), where=counts > 0)

    # Broadcast back to 2D
    obj_mean = mean_lookup[flat_seg].reshape(segments_2d.shape)
    return obj_mean


def compute_stability(probs_3d):
    """
    Computes Winner (W), Runner-up (S), Confusion Index (CI = S / W),
    and binary Stability Map.
    probs_3d shape: (num_classes, H, W)
    """
    sorted_probs = np.sort(probs_3d, axis=0)
    W = sorted_probs[-1]
    S = sorted_probs[-2]

    # CI = S / W (clamped between 0 and 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        CI = np.divide(S, np.maximum(W, 1e-4), dtype=np.float32)
        CI = np.clip(CI, 0.0, 1.0)

    is_stable = (CI <= STABILITY_TAU).astype(np.uint8)
    return W, S, CI, is_stable


def process_tile(prob_path, seg_path, out_dir):
    """
    Processes one regional tile:
    - Reads 18-band probabilities and 10m segments.
    - Bilinear-upsamples probabilities to 10m if needed.
    - Computes object means, stability, adaptive blending, and rule refinement.
    - Writes GeoTIFFs.
    """
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(prob_path).replace('.tif', '')

    print(f"\n--- Processing: {base_name} ---")

    # 1. Read Segment Raster (10m)
    with rasterio.open(seg_path) as src_seg:
        seg_meta = src_seg.profile.copy()
        segments = src_seg.read(1).astype(np.int32)
        target_h, target_w = segments.shape

    # 2. Read Probability Raster (Resampling to 10m if needed)
    with rasterio.open(prob_path) as src_prob:
        if (src_prob.height, src_prob.width) != (target_h, target_w):
            print(f"Resampling probabilities from ({src_prob.height}, {src_prob.width}) to ({target_h}, {target_w}) using Bilinear...")
            probs = src_prob.read(
                out_shape=(src_prob.count, target_h, target_w),
                resampling=Resampling.bilinear
            ).astype(np.float32) / 10000.0
        else:
            probs = src_prob.read().astype(np.float32) / 10000.0

    num_classes = probs.shape[0]
    print(f"Loaded {num_classes} probability bands. Computing segment means...")

    # 3. Object-Level Mean Probabilities
    obj_probs = np.empty_like(probs)
    for c in range(num_classes):
        obj_probs[c] = fast_segment_mean(probs[c], segments)

    # 4. Object Stability & Adaptive Weights
    _, _, obj_CI, _ = compute_stability(obj_probs)

    # Base class weights (w)
    w_base = np.zeros(num_classes, dtype=np.float32)
    for c_id, name in CLASS_ID_TO_NAME.items():
        if c_id - 1 < num_classes:
            smoothness = SMOOTHNESS_TL.get(name, 1.0)
            w_base[c_id - 1] = np.clip(smoothness / 5.0, 0.2, 0.9)

    # 5. Adaptive Convex Blending
    # w_eff = w * (1 - 0.6 * CI)
    stability_factor = 1.0 - (obj_CI * ADAPTIVE_SENSITIVITY)
    blended = np.empty_like(probs)
    for c in range(num_classes):
        w_eff = w_base[c] * stability_factor
        blended[c] = (obj_probs[c] * w_eff) + (probs[c] * (1.0 - w_eff))

    # Normalize blended probabilities to sum to 1.0
    sum_blend = np.sum(blended, axis=0, keepdims=True)
    blended = np.divide(blended, np.maximum(sum_blend, 1e-6))

    # 6. Winning and Runner-up Class IDs
    W, S, CI, is_stable = compute_stability(blended)
    winner_idx = np.argmax(blended, axis=0) + 1  # 1-indexed (1 to 18)

    # Fast runner-up calculation
    masked_blend = blended.copy()
    for c in range(num_classes):
        masked_blend[c][winner_idx == (c + 1)] = -1.0
    runner_up_idx = np.argmax(masked_blend, axis=0) + 1

    # 7. Intrinsic Rule Refinement
    print("Applying Intrinsic Rule Refinement...")
    # Rule 1: False Hydrology Suppression (Waterbody 17 or Dry riverbed 18 competing with Built-up 16 or Cleared 15)
    false_hydrology = (
        np.isin(winner_idx, [17, 18]) &
        np.isin(runner_up_idx, [15, 16]) &
        (CI > 0.65)
    )
    winner_idx[false_hydrology] = runner_up_idx[false_hydrology]

    # Rule 2: Post-classification merge of Dry riverbed (18) into Waterbody (17)
    if MERGE_WATER_RIVERBED:
        winner_idx[winner_idx == 18] = 17

    # 8. Export Output GeoTIFFs
    out_meta = seg_meta.copy()
    out_meta.update(dtype=rasterio.uint8, count=1, nodata=0, compress='deflate')

    lulc_out = os.path.join(out_dir, f"{base_name}_LULC.tif")
    with rasterio.open(lulc_out, "w", **out_meta) as dst:
        dst.write(winner_idx.astype(np.uint8), 1)

    stab_out = os.path.join(out_dir, f"{base_name}_Stability_Map.tif")
    with rasterio.open(stab_out, "w", **out_meta) as dst:
        dst.write(is_stable.astype(np.uint8), 1)

    ci_out = os.path.join(out_dir, f"{base_name}_Confusion_Index.tif")
    with rasterio.open(ci_out, "w", **out_meta) as dst:
        dst.write((CI * 100).round().astype(np.uint8), 1)

    print(f" Saved: {lulc_out}")
    print(f" Saved: {stab_out}")
    print(f" Saved: {ci_out}")


if __name__ == "__main__":
    # Example local directories:
    PROBS_DIR = "./data/probabilities"
    SEGS_DIR = "./data/segments"
    OUTPUT_DIR = "./outputs/hybrid_lulc"

    prob_files = sorted(glob.glob(os.path.join(PROBS_DIR, "*probs*.tif")))
    for p_file in prob_files:
        # Match segment file by tile name (e.g. central_timor, west_box)
        tile_token = os.path.basename(p_file).split("_")[3] # Adjust token index to your naming
        matching_segs = glob.glob(os.path.join(SEGS_DIR, f"*{tile_token}*.tif"))
        if matching_segs:
            process_tile(p_file, matching_segs[0], OUTPUT_DIR)
        else:
            print(f"Warning: No matching segment file found for {p_file}")
