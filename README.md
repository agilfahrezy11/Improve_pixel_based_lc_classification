# Pixel Based Supervised Classification for Land Cover Mapping

A Python toolkit for land-cover classification utilizing machine learning classiifer. It extracts labeled pixels from a multi-stack raster and training samples, optionally selects features, tunes a various machine learning classifier, evaluates the model, and writes a classified GeoTIFF.

## Features

- Extracts raster-band values under labeled training polygons.
- Reprojects training data to the raster CRS when necessary and removes masked/non-finite pixels.
- Creates reproducible, stratified train/test splits.
- Selects bands with recursive feature elimination and cross-validation (RFECV).
- Tunes Random Forest, Extra Trees, LightGBM, XGBoost, or another compatible estimator.
- Reports accuracy, balanced accuracy, macro/weighted F1, a classification report, and a confusion matrix.
- Classifies large rasters in blocks while retaining their spatial metadata.

## Requirements and installation

- Python 3.10 or later
- A multiband raster readable by Rasterio (for example, GeoTIFF)
- Polygon training data readable by GeoPandas, with a column containing class labels

From the repository root, create an environment and install the package:

```bash
python -m venv .venv
.venv\\Scripts\\activate        # Windows PowerShell
# source .venv/bin/activate       # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -e .
```

Install XGBoost support only when you need it:

```bash
python -m pip install -e ".[boosting]"
```

Install remote-imagery support only when you need Microsoft Planetary Computer:

```bash
python -m pip install -e ".[planetary-computer]"
```

## Quick start

Place your raster and its training-vector files somewhere accessible (the data itself is not included in the repository). The vector layer must contain polygon geometries and a label field; substitute the paths and `class_id` below for your data.

```python
from ML_LC_Classifier import (
    classify_raster,
    evaluate_model,
    load_and_split_training_data,
    select_features,
    tune_model,
)

raster_path = "input_data/feature_stack.tif"
training_path = "input_data/training_samples.shp"

# Each raster band becomes a feature. Pixel labels come from `class_id`.
X_train, X_test, y_train, y_test = load_and_split_training_data(
    raster_path,
    training_path,
    class_field="class_id",
    test_size=0.30,
    random_state=42,
)

# Optional: fit RFECV on training data only, then retain the selector.
selection = select_features(X_train, y_train, X_test, cv=5)

result = tune_model(
    classifier="RandomForest",
    parameter_space={
        "n_estimators": [300, 500],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2],
    },
    X_train=selection.X_train,
    y_train=y_train,
    search_method="random",
    n_iter=4,
    cv=5,
)

evaluation = evaluate_model(result.model, selection.X_test, y_test)
print(evaluation.metrics)

classify_raster(
    raster_path,
    result.model,
    "output/landcover_prediction.tif",
    selector=selection.selector,
)
```

`classify_raster` processes the source raster in windows, so it does not need to load the entire raster into memory. Invalid source pixels are written with the output's `nodata` value (default: `0`). The model's predicted classes must be numeric for GeoTIFF output.

Use `apply_mmu` to remove undersized 8-connected class patches and fill them
with the iterated 3x3 neighborhood mode. Thresholds are pixel counts, and
source nodata pixels are retained:

```python
from ML_LC_Classifier import apply_mmu

apply_mmu(
    "output/landcover_prediction.tif",
    "output/landcover_prediction_mmu.tif",
    iterations=6,
)
```

The default thresholds are available as `MMU_TABLE`; pass a custom mapping
with `table={class_id: min_pixels, ...}` to override them. Output is a
single-band uint8 GeoTIFF that keeps the input raster's spatial metadata.

## Project layout

```text
src/ML_LC_Classifier/
  load_extract.py        Load data, extract labeled pixels, and split samples
  feature_elimination.py RFECV feature selection
  tune_model.py          Model construction, hyperparameter tuning, and evaluation
  classify_raster.py     Block-wise GeoTIFF classification
  lc_post_process.py     Minimum mapping unit filtering for classified rasters
javascript/              Google Earth Engine scripts (parallel GEE workflow)
  extract_split_data.js  Pixel sampling and train/test splitting in GEE
  tune_model.js          Random Forest grid-search tuning in GEE
  classify_model.js      Hard / soft / multi-probability classification in GEE
  Post-process.js        Bayesian spatial smoothing for probability maps
notebooks/               Exploratory notebooks and implementation examples
input_data/              Local input-data location (ignored by Git)
output/                  Local predictions (ignored by Git)
```


## Main API

| Function | Purpose |
| --- | --- |
| `load_and_split_training_data` | Load raster/vector data, extract valid pixels, and return a stratified split. |
| `select_features` | Fit RFECV using training data and transform train/test features. |
| `build_classifier` | Create `RandomForest`, `ExtraTrees`, `LightGBM`, or `XGBoost` estimators. |
| `tune_model` | Run grid or randomized cross-validation search and return the best fitted model. |
| `evaluate_model` | Calculate held-out classification metrics and a confusion matrix. |
| `classify_raster` | Predict a single-band classified GeoTIFF from an input feature stack. |
| `apply_mmu` | Remove undersized class patches and fill them with neighborhood mode. |
| `save_flagged_points` | Write the original training points as a shapefile with outlier flags and scores. |
| `mosaic_rasters` | Merge adjacent or overlapping Earth Engine tile downloads into one GeoTIFF. |
| `stack_rasters` | Align raster layers and write them as ordered bands in one GeoTIFF. |

For tiled downloads, pass a glob pattern to `mosaic_rasters`. For separate
bands, pass an ordered list to `stack_rasters`; the list order becomes the
GeoTIFF band order:

```python
from ML_LC_Classifier import mosaic_rasters, stack_rasters

mosaic_rasters(
    "input_data/sentinel2_tiles/*.tif",
    "input_data/sentinel2_mosaic.tif",
    method="max",  #ideal for binary masks
    nodata=0,
)
stack_rasters(
    ["input_data/B04.tif", "input_data/B08.tif"],
    "input_data/feature_stack.tif",
    band_names=["B04", "B08"],
)
```

## Optional remote imagery and feature stacks

The optional Planetary Computer helpers find scenes, return a lightweight preview
URL, download selected Cloud Optimized GeoTIFF assets, and create a local feature
stack for the existing training and classification workflow. Supported collections
include Sentinel-2 L2A optical imagery, Sentinel-1 RTC radar, Landsat Collection
2 Level-2 (Landsat 4--9), and STAC DEM collections such as `cop-dem-glo-30`.

```python
from ML_LC_Classifier import (
    create_feature_stack,
    download_imagery,
    preview_imagery,
    search_imagery,
)

scenes = search_imagery(
    aoi="input_data/study_area.geojson",  # GeoJSON/vector file, geometry, or [W, S, E, N]
    collection="sentinel-2-l2a",
    start_date="2024-01-01",
    end_date="2024-06-30",
    max_cloud_cover=20,
)
print(preview_imagery(scenes[0]))

assets = download_imagery(
    scenes,
    assets=["B02", "B03", "B04", "B08", "B11"],
    output_dir="input_data/sentinel2",
)
stack = create_feature_stack(
    assets,
    "input_data/feature_stack.tif",
    composite="median",  # or "first_valid"
    indices=["NDVI", "MNDWI"],
)
print(stack.feature_names)
```

`create_feature_stack` writes `feature_stack.tif.json` beside the GeoTIFF. Keep
this manifest with the fitted model: it records the exact feature-band order,
source scenes, compositing method, and grid settings. To add a STAC DEM, download
its height asset, map its asset key to a clear name, and request terrain features:

```python
stack = create_feature_stack(
    imagery=[*optical_assets, *dem_assets],
    output_path="input_data/feature_stack.tif",
    asset_aliases={"data": "dem"},
    terrain_from=["dem"],  # adds dem_slope and dem_aspect
)
```

For Landsat, use the STAC asset names supplied by `scenes[0].available_assets`.
The index calculator recognises the common Landsat names (`blue`, `green`, `red`,
`nir08`, `swir16`, and `swir22`) and Sentinel-2 names (`B02`, `B03`, `B04`,
`B08`, `B11`, and `B12`). Sentinel-1 RTC assets are normally `vv` and `vh`.
Sentinel-1 RTC may require a configured Planetary Computer account for SAS access.

## Google Earth Engine workflow (JavaScript)

The `javascript/` folder contains a parallel pipeline for running the same conceptual workflow entirely inside [Google Earth Engine](https://earthengine.google.com/) (GEE). Each file is a self-contained GEE module that you `require()` in your own GEE script.

### Modules

| File | Purpose |
| --- | --- |
| `extract_split_data.js` | Sample pixel values from a GEE feature stack and split them into training / testing sets. |
| `tune_model.js` | Grid-search hyperparameter tuning for GEE Random Forest. |
| `classify_model.js` | Hard, soft (OVR probability), and multi-probability classification, plus feature importance and accuracy evaluation. |
| `Post-process.js` | Empirical-Bayes spatial smoothing (Bayesian smoothing) on multi-class probability images based on Camara et al. (2024). |

### Usage

Import a module at the top of your GEE script using its repository asset path:

```javascript
var extract = require('users/<username>/<repo>:javascript/extract_split_data');
var tuning  = require('users/<username>/<repo>:javascript/tune_model');
var clf     = require('users/<username>/<repo>:javascript/classify_model');
var post    = require('users/<username>/<repo>:javascript/Post-process');
```

#### 1 — Extract and split training data

```javascript
// Stratified split (recommended — preserves class distribution)
var split = extract.stratifiedSplit(roi, featureStack, 'classId', 10, 0.7, 0, 16);
// split.trainingPixels — pixel table for model training
// split.testingPixels  — pixel table for evaluation
// split.trainFC / split.testFC — vector features for map display

// Simple random split
var split = extract.randomSplit(featureStack, roi, 'classId', 0.6, 10, 16, 0);
```

#### 2 — Tune Random Forest hyperparameters

```javascript
var results = tuning.tuneRandomForest(
  split.trainingPixels,
  split.testingPixels,
  'classId',
  featureStack.bandNames(),
  [50, 100, 200],   // numberOfTrees
  [0],              // variablesPerSplit (0 = GEE default √N)
  [1, 5]            // minLeafPopulation
);

var best = tuning.getBestParams(results, 'kappa');
print('Best params:', best);
```

#### 3 — Classify

The `classify` function supports three modes controlled by the `mode` argument:

```javascript
// Hard classification (single-band label map)
var hardResult = clf.classify('hard', split.trainingPixels, 'classId', featureStack, {
  nTrees: 200, minLeaf: 1, seed: 0
});
Map.addLayer(hardResult.classificationMap, {}, 'Hard classification');

// Soft / OVR probability stack + argmax map
var softResult = clf.classify('soft', split.trainingPixels, 'classId', featureStack, {
  nTrees: 200, probabilityScale: 100
});

// Multi-probability mode (required for Bayesian smoothing)
var classIdToNameMap = {1: 'Forest', 2: 'Cropland', 3: 'Water', 4: 'Built-up', 5: 'Bare'};
var multiResult = clf.classify('multi', split.trainingPixels, 'classId', featureStack, {
  classIdToNameMap: classIdToNameMap, nTrees: 200
});
```

Retrieve feature importance and evaluate on the held-out set:

```javascript
var importance = clf.getFeatureImportance(hardResult.trainedModel);
print('Feature importance:', importance);

var metrics = clf.evaluateModel(hardResult.trainedModel, split.testingPixels, 'classId');
print('OA:', metrics.overallAccuracy, 'Kappa:', metrics.kappa);
print('Error matrix:', metrics.errorMatrix);
```

#### 4 — Post-process with Bayesian spatial smoothing

Bayesian smoothing is only available for the `'multi'` probability output.

```javascript
var classBands  = multiResult.classBands;   // ['Forest', 'Cropland', ...]
var classValues = multiResult.classValues;  // [1, 2, ...]

// Inspect local variance to choose a smoothness value
var localVar = post.localVariance(multiResult.probsImage, classBands, 7, 0.5);

// Apply smoothing (smoothness ~ prior variance in logit space; higher = more smoothing)
var smoothedProbs = post.bayesianSmooth(
  multiResult.probsImage,
  classBands,
  0.5,   // smoothness — scalar or {Forest: 0.3, Water: 0.8, ...}
  7,     // window size (pixels)
  0.5    // neighbourhood fraction used for local statistics
);

// Convert smoothed probabilities to a final label map
var smoothedMap = post.classifyFromProbs(smoothedProbs, classBands, classValues);
Map.addLayer(smoothedMap, {}, 'Smoothed classification');
```

### GEE workflow vs Python workflow

| | Python (`ML_LC_Classifier`) | GEE JavaScript |
| --- | --- | --- |
| Compute | Local CPU/GPU | Google cloud |
| Imagery source | Local GeoTIFF / Planetary Computer | GEE image catalog |
| Classifiers | RF, Extra Trees, LightGBM, XGBoost | Random Forest (GEE built-in) |
| Spatial smoothing | — | Bayesian (Camara et al. 2024) |
| Output | Local GeoTIFF | GEE asset / Drive export |

## Notes

- The order of bands used for prediction must match the order used to train the model.
- Keep the fitted RFECV selector and pass it to `classify_raster` whenever feature selection was used during training.
- For reproducible splits and searches, the defaults use `random_state=42`.
- In the GEE scripts, `seed=0` is used throughout for reproducibility; pass a different seed to any function to change this.
