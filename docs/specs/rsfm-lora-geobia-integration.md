# Spec: Modular RSFM Subpackage (LoRA Fine-Tuning, Custom Cross-Sensor Embeddings, Late Ancillary Fusion, & GEOBIA Integration)

**Triage Label:** `ready-for-agent`

## Problem Statement

Multi-epoch, national-scale land use / land cover (LULC) classification workflows (such as the 18-class hierarchical Timor-Leste mapping across 2005, 2015, 2020, and 2025+) currently rely on Continuous Change Detection and Classification (CCDC) harmonic segmentation and manual spectral/texture feature engineering (36+ seasonal reflectance bands, spectral indices, phenological deltas, harmonics, and GLCM texture layers).

While the downstream pipeline—multi-probability classification, spatial Bayesian smoothing, Malinverni & Sanità (2026) hybrid pixel-object stability assembly over Unified SNIC Superpixels, intrinsic mask refinement, and vectorized multi-epoch temporal consistency—performs well, the upstream CCDC and manual feature engineering stage presents four severe operational bottlenecks:

1. **Exhaustive Cloud Compute Overhead**: Fitting pixel-level harmonic regressions over 25 years of cloudy optical imagery across national extents consumes excessive Google Earth Engine EECU and storage budgets.
2. **Lack of Cross-Sensor Normalization**: CCDC models fitted on 30 m historical Landsat 5/7/8 archives (2005, 2015) and 10–20 m Sentinel-2 archives (2020, 2025+) exhibit radiometric, spatial, and harmonic-break discrepancies, hindering cross-sensor classifier transfer and multi-epoch consistency.
3. **Static Embedding Catalog Temporal Limits**: Pre-computed global foundation model embedding catalogs (such as AlphaEarth or pre-computed Zarr stores) are restricted to 2017–2024 due to Sentinel-1/2 constellation availability and annual batch lag, leaving historical baseline epochs (2005, 2015) and current/future epochs (2025+) uncovered.
4. **Hardware Constraints at National Scale**: Running naive deep learning inference or saving uncompressed 64-band floating-point pixel embeddings across national 10 m mosaics (~150 million pixels per epoch) creates prohibitive GPU memory and disk I/O bottlenecks (~38 GB per epoch) on standard workstations and free/pro cloud notebooks (Google Colab).

## Solution

Introduce an isolated, modular Remote Sensing Foundation Model (RSFM) subpackage gated behind an optional dependency group (`rsfm`) that replaces the CCDC harmonic reconstruction, manual spectral indices, GLCM textures, and CCDC break-detection stages while preserving 100% compatibility with the existing downstream classification, GEOBIA hybrid pixel-object stability assembly, and multi-epoch trajectory consistency pipeline.

The solution provides two high-level, hardware-aware capabilities:

1. **Chip Extraction & LoRA Fine-Tuning Engine**: Extracts compact multi-seasonal optical (`Wet` and `Dry` seasonal medians) and radar (`VV`, `VH`) image chips around sparse ground-truth vector samples, optionally filters out disturbed training samples via bi-temporal embedding cosine distance (replacing CCDC break distance filtering), and fine-tunes a pluggable pre-trained multimodal/wavelength-aware RSFM backbone (e.g., Clay/DOFA, OlmoEarth, or Prithvi) using Low-Rank Adaptation (LoRA) and sparse-label loss masking (`ignore_index=0`) into a portable checkpoint bundle.
2. **Memory-Safe Streaming Inference, Late Ancillary Fusion, & GEOBIA Pooling Engine**: Streams arbitrary regional or national rasters in padded windows with Gaussian overlap blending and supports three seamless output contracts:
   - **Custom Embedding Stack & GEOBIA Object Pooling**: Generates custom cross-sensor normalized embedding stacks (or pools embeddings by mean/std inside Unified SNIC Superpixels to compress national datasets from ~150M pixels to ~2M objects) concatenated with manual topographic (`Elevation`, `Slope`, `TPI`, `TWI`) and proximity (`dist_road`, `dist_urban`) layers for rapid shallow classifier tuning (`RandomForest`, `LightGBM`, `XGBoost`).
   - **In-Memory Late-Fused Probability Cube Generation**: Extracts RSFM embeddings in memory per streaming window, concatenates on-the-fly resampled manual ancillary layers, evaluates the classifier in RAM (eliminating 38 GB intermediate disk writes), and writes the multi-class probability raster consumed directly by the existing Bayesian smoother and hybrid pixel-object stability processor.
   - **Bi-Temporal Embedding Stability Mask Generation**: Computes pixel- or segment-level cosine similarity between two epochs' RSFM embeddings to output the binary stability mask and continuous change magnitude required by the multi-epoch temporal consistency engine (replacing the CCDC stable mask).

## User Stories

1. As a remote sensing practitioner, I want to install the core classification package without heavy deep learning dependencies by default, and optionally install the RSFM dependencies only on GPU/Colab environments, so that my local CPU post-processing environment stays fast and lightweight.
2. As a remote sensing practitioner, I want to pass simple wet-season and dry-season surface reflectance median composites (and optional SAR backscatter medians) instead of CCDC harmonic assets, so that I can eliminate exhaustive CCDC EECU computation in Google Earth Engine.
3. As a multi-epoch analyst mapping 2005, 2015, 2020, and 2025+, I want to configure sensor band mappings with physical central wavelengths (in micrometers) or modality stream keys (`landsat`, `sentinel2`, `sentinel1`), so that the foundation model normalizes historical Landsat 5/7/8 and modern Sentinel-2/Landsat 9 inputs into a unified feature space.
4. As a model developer, I want to extract fixed-size spatial training chips (e.g., `224x224` pixels) centered on my labeled vector polygons or points with unlabeled background pixels assigned `ignore_index=0`, so that I can fine-tune a foundation model on sparse ground-truth samples without needing wall-to-wall labeled training tiles.
5. As a model developer, I want to pre-filter training samples using bi-temporal embedding cosine distance (or seasonal spectral change magnitude) before training, so that samples that underwent abrupt land-cover disturbance near the target epoch are excluded just like CCDC break-distance filtering.
6. As a model developer working on a single Google Colab GPU (T4/L4), I want to fine-tune the pre-trained RSFM backbone using Low-Rank Adaptation (LoRA) adapters paired with a lightweight segmentation or classification head, so that the model adapts its self-attention to my regional land-cover taxonomy using under 12 GB of VRAM and without catastrophic forgetting.
7. As a model developer, I want the fine-tuning workflow to export a self-contained checkpoint bundle containing the LoRA adapter weights, task head weights, band/wavelength order, and normalization statistics, so that I can transfer a ~50 MB artifact from Colab to my local workstation or batch inference environment.
8. As a remote sensing practitioner, I want to keep my manually engineered topographic layers (`Elevation`, `Slope`, `TPI`, `TWI`) and anthropogenic proximity layers (`dist_road`, `dist_urban`) and fuse them with the RSFM's 64-D or 128-D embeddings ("Late Fusion"), so that ecological niche constraints (such as high-altitude coffee agroforestry, coastal mangroves, and road-adjacent built-up areas) are strictly preserved.
9. As a remote sensing practitioner, I want auxiliary topographic and proximity rasters with different resolutions or projections (e.g., 30 m DEM vs. 10 m Sentinel-2) to be aligned on the fly during windowed inference, so that I do not have to manually reproject and duplicate multi-gigabyte ancillary rasters on disk.
10. As a practitioner working with GEOBIA, I want to pool the RSFM embeddings and manual ancillary features (e.g., mean and standard deviation) within my Unified SNIC Superpixel segments, so that a national 150-million-pixel mosaic is compressed into a compact segment feature table that trains `RandomForest` or `LightGBM` in seconds on a laptop CPU.
11. As a practitioner running national-scale pixel-level inference on limited hardware, I want an in-memory late-fusion streaming mode that extracts window embeddings, appends the 6 manual ancillary bands in RAM, predicts class probabilities in RAM, and writes only the final 18-band probability GeoTIFF to disk, so that I never have to store 38 GB intermediate embedding rasters per epoch.
12. As a practitioner running sliding-window RSFM inference across large rasters, I want overlapping chip predictions within each streaming block to be blended using a 2D Gaussian / Hann spatial weight window, so that rigid Vision Transformer patch boundaries do not create visible grid seams in the output probability cube.
13. As a practitioner using the Malinverni & Sanità (2026) hybrid pixel-object workflow, I want the RSFM inference engine's output probability GeoTIFF to match the exact band count, class ordering, and scaling contract expected by the existing hybrid assembly processor and Bayesian smoother, so that intra-object stability ($CI = S/W$), adaptive convex blending, false hydrology suppression, and mask rules work without modification.
14. As a multi-temporal analyst, I want to compute a bi-temporal Embedding Stability Mask between any two epochs (e.g., 2005 vs. 2015 or 2020 vs. 2025) using pixel-wise or SNIC-segment-wise cosine distance ($d_{\text{cos}}(z_{t_1}, z_{t_2}) \le \tau_{\text{stable}}$), so that I can replace the CCDC stable mask in the vectorized temporal consistency engine and prevent classifier churn on undisturbed parcels.
15. As a developer running automated tests in CI or on a CPU-only machine, I want the RSFM subpackage to accept a pluggable backbone interface with a deterministic lightweight CPU backbone implementation, so that the entire chip extraction, LoRA adaptation, late fusion, GEOBIA pooling, and streaming inference pipeline can be tested end-to-end without downloading multi-gigabyte model weights or requiring a CUDA GPU.

## Implementation Decisions

- **Modular Subpackage & Optional Dependency Isolation**:
  - All foundation-model functionality will live in a dedicated subpackage under the main package namespace, with its public API re-exported cleanly when optional dependencies are available.
  - Deep learning and foundation model dependencies (`torch`, `peft`, `geoai-py`) will be declared under an optional `rsfm` dependency group in the package configuration so the base installation remains pure CPU tabular/geospatial Python.
- **Pluggable RSFM Backbone Protocol with OpenGeoAI (`geoai`) Support**:
  - Define a minimal, clean backbone protocol that accepts multi-seasonal or multi-band input tensors and returns spatial feature embeddings `(batch, embed_dim, height, width)`, plus `apply_lora(rank, alpha)` for parameter-efficient fine-tuning.
  - Provide a built-in OpenGeoAI backbone adapter (`geoai.list_foundation_models()`, `geoai.load_foundation_model()`) so users can load TerraTorch/Hugging Face foundation models from the `opengeoai.org` catalog in one call, alongside a deterministic lightweight CPU backbone for fast automated testing without network or GPU requirements.
- **Sparse-Polygon Chip Dataset Extraction**:
  - Build a chip extraction function that reads vector training samples (polygons or points) and extracts fixed-size (`chip_size`, default `224`) multi-seasonal raster chips centered on sample geometries.
  - Rasterize all training geometries intersecting each chip into a 2D target mask `(chip_size, chip_size)` where labeled pixels carry integer class IDs `1..K` and unlabeled background pixels carry `0` (`ignore_index=0`).
  - Include an optional sample stability filter that computes the cosine distance between the target epoch's chip embedding and a reference epoch's chip embedding at labeled pixels, dropping samples whose change distance exceeds `max_change_distance` (replacing CCDC $\Delta t_{\text{break}}$ filtering).
- **LoRA Fine-Tuning & Checkpoint Bundle Contract**:
  - Fine-tuning pairs the LoRA-enabled backbone (`freeze_backbone=True` on base weights, trainable low-rank matrices $A, B$ of configurable `lora_rank`, default `8`) with a lightweight pixel projection / UperNet-style multi-scale neck and linear classification head (optionally concatenating static ancillary channels at the pixel level before the final linear layer).
  - Loss computation uses Cross-Entropy or Focal Loss with `ignore_index=0` so gradients backpropagate strictly from labeled polygon/point pixels while 100% of the chip provides spatial self-attention context.
  - Saving a checkpoint writes a portable directory or archive containing the adapter state dict, head state dict, and a JSON manifest recording `embed_dim`, `class_values`, `band_names`, `wavelengths_um`, per-band normalization statistics (`mean`, `std`), and `ancillary_names`.
- **Division of Predictor Labor (RSFM vs. Manual Ancillary Late Fusion)**:
  - Optical reflectance, SAR backscatter (`VV`, `VH`), spectral transformations, seasonal deltas, and spatial textures are handled inside the RSFM backbone to produce the $D$-dimensional embedding ($D \in \{64, 128, \dots\}$).
  - Topographic derivatives (`Elevation`, `Slope`, `TPI`, `TWI`) and anthropogenic proximity layers (`dist_road`, `dist_urban`) remain explicit manual raster layers that are aligned on the fly using virtual warping (`WarpedVRT`) and concatenated with the $D$-dimensional RSFM embedding ("Late Fusion") prior to final class probability prediction.
- **Three Composable Inference Modes Sharing One Windowed Streaming Engine**:
  - The inference engine processes rasters in memory-safe padded outer blocks (e.g., `2048x2048` with `64px` overlap) and slides overlapping inner chips (`chip_size=224`, `stride=112`) weighted by a 2D Hann/Gaussian window to eliminate ViT patch-edge seams.
  - **Mode 1 — Custom Embedding Stack (`FeatureStackResult` Contract)**: Writes the $D$-band custom embedding (plus optional aligned ancillary bands) to a GeoTIFF and companion JSON manifest matching the existing `FeatureStackResult` return type so `load_and_split_training_data` and `tune_model` work out of the box.
  - **Mode 2 — GEOBIA Segment-Pooled Embedding Table**: Given a Unified SNIC segment raster (`seg_path`), aggregates the $D$-dimensional RSFM embeddings and manual ancillary bands per segment ID (mean and optional standard deviation), returning a segment-indexed DataFrame/Parquet table and mapping fitted segment probabilities back to raster tiles.
  - **Mode 3 — In-Memory Late-Fused Probability Cube (`prob_path` Contract)**: Concatenates the window's $D$-dimensional RSFM embeddings with the aligned ancillary bands in RAM, evaluates either the neural classification head or a fitted tabular estimator (`RandomForest` / `LightGBM` from `tune_model`) in RAM, and writes the $K$-band probability GeoTIFF (`float32` or scaled `uint16`) expected by `bayesian_smooth` and `hybrid_assembly.process_tile`.
- **Bi-Temporal Embedding Stability Mask (`TL_stable_mask` Replacement)**:
  - Provide a dedicated function that streams two epochs' embeddings (or computes them on the fly per block), calculates the pixel-wise or SNIC-segment-wise cosine distance $d_{\text{cos}}(z_{t_1}, z_{t_2}) = 1 - \frac{z_{t_1} \cdot z_{t_2}}{\|z_{t_1}\|\|z_{t_2}\|}$, and writes both the continuous change distance raster and the binary stability mask ($d_{\text{cos}} \le \tau_{\text{stable}}$) consumed by the multi-epoch consistency engine.

## Testing Decisions

- **Philosophy**: Tests must verify observable external behavior and data contracts across the two public seams without testing internal neural network layer details, and must run fast and deterministically on CPU without network access or multi-gigabyte model downloads.
- **Modules to be Tested**:
  - Chip extraction & sparse-polygon mask generation (`ignore_index=0` preservation, CRS reprojection, nodata handling, and bi-temporal sample stability filtering).
  - LoRA fine-tuning & checkpoint bundle round-tripping (verifying that base backbone weights remain frozen, LoRA adapter + head weights update and reduce loss on synthetic multi-class chips, and checkpoint save/load reproduces identical embeddings and probabilities).
  - Windowed streaming inference across all three output modes:
    1. Custom embedding `FeatureStackResult` compatibility with `load_and_split_training_data` and `tune_model`.
    2. GEOBIA SNIC segment pooling (verifying segment-level mean/std aggregation including negative wrapped int32 GEE segment IDs and nodata exclusion).
    3. In-memory late fusion of RSFM embeddings + `WarpedVRT`-aligned ancillary rasters (`Elevation`, `dist_road`) producing an 18-band probability GeoTIFF that passes directly into `hybrid_assembly.process_tile` and `bayesian_smooth`, plus Gaussian overlap blending continuity across chip boundaries.
  - Bi-temporal embedding cosine distance & binary stability mask generation, verifying compatibility with `trajectory_analysis`.
- **Prior Art in the Codebase**:
  - Follow the synthetic temporary GeoTIFF and vector fixture patterns used in the existing test suite (creating temporary multi-band rasters with `rasterio` and polygon GeoDataFrames in `tmp_path` fixtures, testing windowed consistency, nodata propagation, and manifest serialization).

## Out of Scope

- Pre-training a new foundation model from scratch on global unlabeled satellite archives (we always start from existing pre-trained foundation model weights).
- Modifying the core mathematical formulas of the downstream Malinverni & Sanità (2026) hybrid pixel-object stability blender, empirical-Bayes spatial smoother, or multi-epoch transition rule tables.
- Non-LULC computer vision tasks such as bounding-box object detection or text-prompted visual question answering.

## Further Notes

- For Google Earth Engine data ingestion across 2005, 2015, 2020, and 2025+, companion GEE export helper scripts should export simple cloud-masked Wet Season (`Dec–Apr`) and Dry Season (`May–Nov`) median composites of the 6 common optical/SWIR bands (`Blue, Green, Red, NIR, SWIR1, SWIR2`) plus Sentinel-1 `VV/VH` (where available), replacing the heavy CCDC harmonic reconstruction step.
- Keeping the Unified SNIC Superpixels (`seg_*`) fixed to the reference epoch geometry (and only splitting segments in historical epochs where the bi-temporal embedding cosine distance exceeds $\tau_{\text{stable}}$) prevents sliver polygons across multi-decade trajectory analyses.
