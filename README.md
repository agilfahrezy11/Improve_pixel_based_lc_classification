# ML Land-Cover Classifier

A Python toolkit for pixel-based land-cover classification utilizing machine learning classiifer. It extracts labeled pixels from a multi-stack raster and training samples, optionally selects features, tunes a various machine learning classifier, evaluates the model, and writes a classified GeoTIFF.

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

## Project layout

```text
src/ML_LC_Classifier/
  load_extract.py        Load data, extract labeled pixels, and split samples
  feature_elimination.py RFECV feature selection
  tune_model.py          Model construction, hyperparameter tuning, and evaluation
  classify_raster.py     Block-wise GeoTIFF classification
notebooks/               Exploratory notebooks and implementation examples
input_data/              Local input-data location (ignored by Git)
output/                  Local predictions (ignored by Git)
```

`src/end-to-end-ml.py` is an earlier exploratory script with machine-specific paths. Prefer the package API above or adapt it to your environment before running it.

## Main API

| Function | Purpose |
| --- | --- |
| `load_and_split_training_data` | Load raster/vector data, extract valid pixels, and return a stratified split. |
| `select_features` | Fit RFECV using training data and transform train/test features. |
| `build_classifier` | Create `RandomForest`, `ExtraTrees`, `LightGBM`, or `XGBoost` estimators. |
| `tune_model` | Run grid or randomized cross-validation search and return the best fitted model. |
| `evaluate_model` | Calculate held-out classification metrics and a confusion matrix. |
| `classify_raster` | Predict a single-band classified GeoTIFF from an input feature stack. |

## Notes

- The order of bands used for prediction must match the order used to train the model.
- Keep the fitted RFECV selector and pass it to `classify_raster` whenever feature selection was used during training.
- For reproducible splits and searches, the defaults use `random_state=42`.
