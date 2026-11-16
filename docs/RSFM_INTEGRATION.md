# Remote Sensing Foundation Model (RSFM) Integration Guide

This document describes the architectural role, API contracts, and operational workflows of the **`ML_LC_Classifier.rsfm`** subpackage. It complements [METHODOLOGY.md](./METHODOLOGY.md) by replacing the exhaustive upstream CCDC harmonic reconstruction and manual spectral/texture engineering stages with cross-sensor normalized foundation models—while keeping the downstream `hybrid_assembly`, Bayesian spatial smoothing, and multi-epoch `trajectory_analysis` pipeline intact.

---

## 1. Architectural Role in the Repository

```
+-----------------------------------------------------------------------------+
| 1. Lightweight Ingestion (GEE / Planetary Computer)                         |
|    Wet-Season Median + Dry-Season Median (Optical + optional SAR VV/VH)     |
|    + Static Ancillary Rasters (NASADEM, Slope, TPI, TWI, dist_road, urban)  |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| 2. ML_LC_Classifier.rsfm (Pluggable OpenGeoAI / CPU Backbone + LoRA)        |
|    - Absorbs: spectral indices, seasonal deltas, CCDC harmonics, GLCM       |
|    - Preserves via Late Fusion: Topography & Anthropogenic Distance layers  |
+-----------------------------------------------------------------------------+
            |                                 |                         |
            v                                 v                         v
 [Mode A: Pixel Inference]       [Mode B: GEOBIA Pooling]   [Mode C: Bi-Temporal]
  In-memory Late Fusion           Mean/Std per SNIC ID       Cosine Distance d_cos
  -> 18-band probs.tif            -> Compact segment table   -> TL_stable_mask.tif
  (0 intermediate 38GB I/O)       -> project_segment_probs   -> Sample filter
            |                                 |                         |
            +----------------+----------------+                         |
                             |                                          |
                             v                                          v
+-----------------------------------------------------------------------------+
| 3. Existing Downstream Pipeline (Unchanged)                                 |
|    hybrid_assembly.process_tile() -> classification_assembly.bayesian_smooth|
|    -> lc_post_process.apply_mmu() -> trajectory_analysis & accuracy         |
+-----------------------------------------------------------------------------+
```

### Why Replace CCDC with an RSFM?
1. **Eliminates Heavy GEE CCDC Compute:** Instead of fitting per-pixel harmonic regressions over hundreds of historical scenes, the RSFM only requires simple seasonal median composites (`Wet` and `Dry`).
2. **Cross-Sensor Normalization (2005–2025+):** Pre-computed embedding catalogs (such as AlphaEarth) are locked to 2017–2024. Wavelength-aware and multimodal open foundation models (e.g., Clay/DOFA, OlmoEarth, Prithvi via OpenGeoAI) project 30 m Landsat 5/7/8 (2005, 2015) and 10 m Sentinel-2 (2020, 2025+) into a unified embedding space.
3. **Avoids Intermediate Disk Bottlenecks:** At 10 m across national extents (~150 million pixels), writing a 64-band `float32` pixel embedding GeoTIFF takes ~38 GB per epoch. `ML_LC_Classifier.rsfm` prevents this via **in-memory late-fused probability streaming** (`infer_rsfm_probabilities`) and **GEOBIA SNIC segment pooling** (`pool_rsfm_by_segments`).

---

## 2. Division of Predictor Labor

| Predictor Group | Legacy CCDC Role | RSFM Role |
| :--- | :--- | :--- |
| **Optical Bands & Indices** (`NDVI`, `EVI`, `MNDWI`, `DBSI`, `BLFEI`, ...) | Manual index engineering + RFECV | **Replaced by RSFM** (feed raw seasonal reflectance bands) |
| **Phenological Deltas & Harmonics** (`NIR_amp`, `phase`, `Index_delta`) | CCDC harmonic coefficients | **Replaced by RSFM** (multi-seasonal / temporal inputs) |
| **GLCM Texture** (`contrast`, `entropy`, `ASM`, `IDM`) | 3×3 dry-season NIR GLCM | **Replaced by RSFM** (patch self-attention captures multi-band texture) |
| **Sentinel-1 SAR** (`VV_wet`, `VH_wet`, `VV_dry`, `VH_dry`) | Stacked manually | **Ingested directly** by multimodal RSFMs (or appended in Late Fusion) |
| **Topography** (`NASADEM_HGT`, `Slope`, `TPI`, `TWI`) | Manual raster predictors | **Kept Manual via Late Fusion** (preserves hard altitude/hydrology rules) |
| **Anthropogenic Proximity** (`dist_road`, `dist_urban`) | Manual raster predictors | **Kept Manual via Late Fusion** (captures multi-km vector infrastructure context) |
| **Change Break Mask** (`TL_stable_mask`, sample $\Delta t_{\text{break}}$) | CCDC break detection | **Replaced by Embedding Cosine Distance** ($d_{\text{cos}}(z_{t_1}, z_{t_2})$) |

---

## 3. Installation

The base package remains lightweight and pure CPU:

```bash
pip install -e .
```

To enable live foundation model loading via **OpenGeoAI** (`geoai-py`), PyTorch, and PEFT on Google Colab or a GPU workstation, install the optional `rsfm` extra:

```bash
pip install -e ".[rsfm]"
```

For offline unit testing or CPU prototyping without `torch`/`geoai` installed, use `SimpleCPUBackbone`, which implements the exact `RSFMBackbone` protocol deterministically in NumPy/SciPy.

---

## 4. Core Workflows & Public API

### Workflow A: OpenGeoAI Backbone & Custom Embedding Stack
```python
from ML_LC_Classifier import (
    SimpleCPUBackbone,
    extract_rsfm_embeddings,
    list_opengeoai_models,
    load_and_split_training_data,
    load_opengeoai_backbone,
    tune_model,
)

# On Colab / GPU with .[rsfm] installed:
# print(list_opengeoai_models())
# backbone = load_opengeoai_backbone("prithvi_eo_v2_300")

# On CPU / offline testing:
backbone = SimpleCPUBackbone(embed_dim=64, seed=42)

# Writes blended GeoTIFF + JSON manifest as FeatureStackResult
emb_stack = extract_rsfm_embeddings(
    "input_data/sentinel2_wet_dry.tif",
    "output/rsfm_embeddings.tif",
    backbone=backbone,
    chip_size=224,
    stride=112,
    wavelengths_um=[0.49, 0.56, 0.66, 0.84],
)

X_train, X_test, y_train, y_test = load_and_split_training_data(
    emb_stack.stack_path,
    "input_data/GT_timor_v3_samples.geojson",
    class_field="class_id",
)
tuned = tune_model("RandomForest", {"n_estimators": [300]}, X_train, y_train)
```

### Workflow B: Sparse-Polygon Chip Extraction & LoRA Fine-Tuning
Unlabeled pixels outside your sparse polygons/points receive `ignore_index=0`, so gradients only backpropagate on annotated pixels while the full chip provides spatial context:

```python
from ML_LC_Classifier import finetune_rsfm, load_rsfm_checkpoint, prepare_rsfm_chips

dataset = prepare_rsfm_chips(
    "input_data/sentinel2_wet_dry.tif",
    "input_data/GT_timor_v3_samples.geojson",
    class_field="class_id",
    chip_size=224,
)

ft = finetune_rsfm(dataset, backbone=backbone, lora_rank=8, lora_alpha=16.0, epochs=25)
ft.bundle.save("output/lora_timor_ckpt")

# Reload portable checkpoint (~50 MB) later for streaming inference
bundle = load_rsfm_checkpoint("output/lora_timor_ckpt", backbone=backbone)
```

### Workflow C: In-Memory Late Ancillary Fusion & Probability Streaming
Aligns topographic and distance rasters (even in different CRS/resolutions) on the fly via `WarpedVRT`, fuses `[RSFM embeddings + ancillary bands]` in RAM, and outputs the `uint16` scaled probability stack expected by `hybrid_assembly.process_tile()`:

```python
from ML_LC_Classifier import infer_rsfm_probabilities, process_tile

prob_path = infer_rsfm_probabilities(
    "input_data/sentinel2_wet_dry.tif",
    "output/rsfm_probs_2025.tif",
    bundle=bundle,                  # or backbone=backbone + classifier=tuned.model
    classifier=None,                # set to fitted RF/LightGBM for late-fused tabular head
    ancillary_paths=["input_data/nasadem.tif", "input_data/dist_road.tif"],
    output_dtype="uint16",          # 0..10000 scaled, ready for hybrid_assembly
)

process_tile(
    prob_path=prob_path,
    seg_path="input_data/snic_segments.tif",
    out_dir="output/hybrid_2025",
)
```

### Workflow D: GEOBIA SNIC Segment Pooling & Object Probability Projection
Compresses national pixel rasters into a compact object feature table (preserving negative wrapped 32-bit GEE segment IDs `< -1`):

```python
from ML_LC_Classifier import (
    assign_segment_labels,
    pool_rsfm_by_segments,
    project_segment_probabilities,
)

seg_df = pool_rsfm_by_segments(
    "input_data/sentinel2_wet_dry.tif",
    "input_data/snic_segments.tif",
    backbone=backbone,
    ancillary_paths=["input_data/nasadem.tif", "input_data/dist_road.tif"],
)
labels_df = assign_segment_labels("input_data/snic_segments.tif", "input_data/GT_timor_v3_samples.geojson")
```

### Workflow E: Bi-Temporal Embedding Cosine Stability Mask
Replaces CCDC break detection (`TL_stable_mask` and sample stability filtering):

```python
from ML_LC_Classifier import (
    apply_transition_consistency,
    compute_embedding_stability,
    filter_disturbed_samples,
)

dist_path, mask_path = compute_embedding_stability(
    "input_data/epoch_2020.tif",
    "input_data/epoch_2025.tif",
    "output/cosine_dist_2020_2025.tif",
    "output/stable_mask_2020_2025.tif",
    backbone=backbone,
    tau_stable=0.15,
)

clean_samples, flagged_samples = filter_disturbed_samples(
    "input_data/GT_timor_v3_samples.geojson",
    dist_path,
    max_change_distance=0.15,
)
```

---

## 5. Live Testing Notebook

See [`notebooks/rsfm_live_test.ipynb`](../notebooks/rsfm_live_test.ipynb) for an executable walkthrough that tests all five workflows end-to-end—either locally on CPU (`USE_OPENGEOAI = False`) or on Google Colab/GPU with OpenGeoAI models (`USE_OPENGEOAI = True`).
