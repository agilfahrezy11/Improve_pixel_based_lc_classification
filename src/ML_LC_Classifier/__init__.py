"""Reusable pixel based classification pipeline for land cover mapping tasks"""

from .classify_raster import classify_raster
from .feature_stack import FeatureStackResult, create_feature_stack
from .feature_elimination import FeatureSelectionResult, select_features
from .load_extract import (
    class_distribution,
    drop_nan_samples,
    extract_pixels_from_shapefile,
    load_and_extract,
    load_and_split_training_data,
    load_training_samples,
    open_raster,
    train_test_split_data,
)
from .tune_model import (
    ModelEvaluation,
    ModelTuningResult,
    build_classifier,
    evaluate_model,
    tune_model,
)
from .planetary_computer import (
    DownloadedAsset,
    ImagerySearchResult,
    download_imagery,
    preview_imagery,
    search_imagery,
)

__all__ = [
    "FeatureSelectionResult",
    "FeatureStackResult",
    "DownloadedAsset",
    "ImagerySearchResult",
    "ModelEvaluation",
    "ModelTuningResult",
    "build_classifier",
    "class_distribution",
    "classify_raster",
    "create_feature_stack",
    "download_imagery",
    "drop_nan_samples",
    "extract_pixels_from_shapefile",
    "load_and_extract",
    "load_and_split_training_data",
    "load_training_samples",
    "open_raster",
    "preview_imagery",
    "search_imagery",
    "select_features",
    "train_test_split_data",
    "evaluate_model",
    "tune_model",
]
