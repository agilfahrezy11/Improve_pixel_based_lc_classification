# Hybrid Pixel-Object Land Cover Classification & Post-Processing

A specialized geospatial Python toolkit for **Hybrid Pixel-Object Land Cover Assembly**, post-processing, and multi-temporal trajectory analysis, alongside a complete end-to-end machine learning classification pipeline. 

---

##  Companion Repositories & Upstream Pipeline

This repository acts as the local post-processing, spatial assembly, and validation engine for land cover workflows. 

For continuous time series analysis, satellite image preprocessing, and superpixel segmentation in Google Earth Engine, see the companion repository:

 **[agilfahrezy11/Temporal_segment_gee](https://github.com/agilfahrezy11/Temporal_segment_gee)**  
*Implementation and source code for continuous change detection and classification (CCDC) and SNIC segmentation in Google Earth Engine.*

---

## Methodology Overview

For a comprehensive explanation of the algorithms, statistical foundations, and mathematical formulations, please see our dedicated methodology documentation:

-  **[Methodology (`docs/METHODOLOGY.md`)](docs/METHODOLOGY.md)**: Detailed breakdown of the Bayesian blending, spatial stability metrics, morphological MMU filtering, multi-epoch trajectory analysis, and area-adjusted accuracy assessment.

---

## Key Capabilities

### 1. Primary Workflow: Hybrid Pixel-Object Post-Processing & Refinement
Designed to ingest multi-band class probability rasters and superpixel segmentation masks (e.g. SNIC from Google Earth Engine or local SLIC):
- **Object-Pixel Bayesian Blending (`hybrid_assembly.py`)**: Blends pixel probabilities with superpixel segment-mean probabilities, tracking stability and Confusion Index ($CI$).
- **Multi-Layer Mask Refinement (`MaskRule`)**: Enforces rule-based hydrology/corridor constraints, urban boundaries, or exclusion layers.
- **Minimum Mapping Unit Filtering (`apply_mmu`)**: Tiled morphological filtering eliminating spurious isolated pixels below minimum spatial thresholds.
- **Seam & Boundary Smoothing (`seam_healing.py`)**: Smooths tile borders and suppresses probability spikes along sliding-window boundaries.
- **Multi-Epoch Trajectory & Consistency Analysis (`trajectory_analysis.py`)**: Enforces temporal consistency (anchor-based backwards/forwards propagation), detects ecological flip-back noise, and exports Sankey transition diagrams.
- **Thematic Accuracy Assessment (`accuracy_assessment.py`)**: Full Olofsson et al. (2014) area-adjusted error matrix, user's/producer's accuracy, and confidence intervals.
- **Streamed Class Area Calculation (`calculate_class_area`)**: Bounded-memory, window-streamed class area accounting with support for both projected (UTM) and WGS-84 geographic CRS geodesic math.

### 2. Secondary Workflow: End-to-End Machine Learning Pipeline
- **Data Intake (`load_extract.py`)**: Extracts labeled pixels from multi-band rasters, handles automatic CRS re-projection, and generates stratified train/test splits safely with guaranteed file closure.
- **Feature Selection (`feature_elimination.py`)**: RFECV band selection optimizing cross-validation accuracy.
- **Model Tuning (`tune_model.py`)**: Hyperparameter search (Random Forest, Extra Trees, LightGBM, XGBoost) with lazy optional dependency loading.
- **Tiled Raster Inference (`classify_raster.py`)**: Windowed raster classification writing both **hard class predictions** and **multi-band uint16/float32 probability rasters** (`mode="proba"`), with feature manifest verification.

### 3. Experimental & Standalone Tools (`ML_LC_Classifier.experimental`)
- Microsoft Planetary Computer STAC downloaders (`planetary_computer.py`)
- Euclidean distance raster generators (`distance_features.py`)
- Raster mosaic & stacking utilities (`raster_mosaic_stack.py`)
- Training sample point outlier detection (`outlier_detection.py`)
- Archival GEE JavaScript scripts (`experimental/javascript/`)

---

## Installation

```bash
git clone https://github.com/AgilAkbarFahrezi/Improve_pixel_based_lc_classification.git
cd Improve_pixel_based_lc_classification
python -m pip install -e .
```

### Optional Dependencies
```bash
# Gradient boosting support (LightGBM, XGBoost)
python -m pip install -e ".[boosting]"

# Plotting and visualization (matplotlib)
python -m pip install -e ".[viz]"

# Planetary Computer & GIS prep utilities
python -m pip install -e ".[experimental]"

# Complete suite
python -m pip install -e ".[all]"
```

---

## Quick Start Examples

### Hybrid Pixel-Object Processing
```python
from ML_LC_Classifier import process_tile, apply_mmu

result = process_tile(
    prob_path="probabilities.tif",
    segment_path="snic_segments.tif",
    out_dir="output/",
    prob_scale=10000.0,
)
print("Generated LULC:", result.lulc)
print("Stability map:", result.stability)
print("Confusion Index:", result.ci)

# Apply Minimum Mapping Unit (MMU) spatial filter
cleaned_lulc = apply_mmu(
    raster_path=result.lulc,
    output_path="output/lulc_mmu.tif",
)
```

### Soft Probability Raster Inference
```python
from ML_LC_Classifier import classify_raster_proba

# Generate multi-band probability raster scaled to [0, 10000] uint16
classify_raster_proba(
    raster_path="feature_stack.tif",
    model=fitted_classifier,
    output_path="probabilities.tif",
    manifest_path="feature_stack.tif.json",
    block_size=512,
)
```

### Multi-Epoch Trajectory Analysis
```python
from ML_LC_Classifier import analyze_multi_temporal_transitions, export_multi_epoch_sankey

epochs = {
    2005: "lulc_2005.tif",
    2015: "lulc_2015.tif",
    2020: "lulc_2020.tif",
    2025: "lulc_2025.tif",
}

results = analyze_multi_temporal_transitions(
    epoch_rasters=epochs,
    output_trajectory_raster="trajectories.tif",
    output_csv_dir="tables/",
)

export_multi_epoch_sankey(results, "sankey.html")
```

---

## Running Tests

```bash
pytest
```
All 37 unit tests run in <3 seconds.
