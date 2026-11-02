"""Reusable pixel-based classification & hybrid post-processing pipeline."""

import warnings

# Core ML and Post-Processing modules
from .classify_raster import classify_raster
from .feature_stack import FeatureStackResult, create_feature_stack
from .core_types import DownloadedAsset
from .lc_post_process import (
    DEFAULT_HYBRID_MMU_TABLE,
    DEFAULT_TIMOR_17_CLASSES,
    MMU_TABLE,
    TransitionRule,
    apply_mmu,
    apply_transition_consistency,
    calculate_class_area,
    calculate_transition_matrix,
)
from .feature_elimination import FeatureSelectionResult, select_features
from .load_extract import (
    class_distribution,
    drop_nan_samples,
    extract_pixels_from_shapefile,
    extract_point_features,
    load_and_extract,
    load_and_extract_points,
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
from .hybrid_assembly import MaskRule, process_tile, refine_lulc_with_mask_layer
from .classification_assembly import (
    bayesian_smooth,
    classify_from_probs,
    local_mean_variance,
    local_variance,
    logits_to_probs,
    probs_to_logits,
)
from .trajectory_analysis import (
    BACKWARD_CONSISTENCY_RULES,
    DEFAULT_IMPOSSIBLE_TRANSITIONS,
    DEFAULT_TIMOR_18_CLASSES,
    FORWARD_CONSISTENCY_RULES,
    IntervalTransitionSummary,
    MultiTemporalTrajectoryResult,
    TRAJECTORY_CLASSES,
    analyze_multi_temporal_transitions,
    apply_bidirectional_multi_epoch_consistency,
    apply_multi_epoch_transition_consistency,
    export_multi_epoch_sankey,
)
from .seam_healing import (
    heal_array_seams,
    heal_probability_array_spikes,
    heal_probability_seams,
    heal_raster_seams,
)
from .accuracy_assessment import (
    ThematicAccuracyResult,
    assess_map_accuracy,
    compute_thematic_accuracy,
    count_raster_pixels_streaming,
)


# Backward-compatible lazy loaders for modules moved to experimental/
def __getattr__(name: str):
    experimental_modules = {
        "create_distance_features": "distance_features",
        "mosaic_rasters": "raster_mosaic_stack",
        "stack_rasters": "raster_mosaic_stack",
        "detect_outliers_per_class": "outlier_detection",
        "detect_point_outliers": "outlier_detection",
        "flag_outliers_in_vector": "outlier_detection",
        "load_points": "outlier_detection",
        "remove_outliers": "outlier_detection",
        "remove_outliers_from_vector": "outlier_detection",
        "save_cleaned_points": "outlier_detection",
        "save_flagged_points": "outlier_detection",
        "ImagerySearchResult": "planetary_computer",
        "download_imagery": "planetary_computer",
        "preview_imagery": "planetary_computer",
        "search_imagery": "planetary_computer",
    }
    if name in experimental_modules:
        mod_name = experimental_modules[name]
        warnings.warn(
            f"'{name}' has been moved to the 'experimental' package and will be "
            f"deprecated from top-level 'ML_LC_Classifier' in a future release. "
            f"Import it via 'ML_LC_Classifier.experimental.{mod_name}' instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        import importlib
        mod = importlib.import_module(f"{__name__}.experimental.{mod_name}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ThematicAccuracyResult",
    "assess_map_accuracy",
    "compute_thematic_accuracy",
    "count_raster_pixels_streaming",
    "heal_raster_seams",
    "heal_array_seams",
    "heal_probability_seams",
    "heal_probability_array_spikes",
    "apply_bidirectional_multi_epoch_consistency",
    "FORWARD_CONSISTENCY_RULES",
    "BACKWARD_CONSISTENCY_RULES",
    "analyze_multi_temporal_transitions",
    "apply_multi_epoch_transition_consistency",
    "export_multi_epoch_sankey",
    "MultiTemporalTrajectoryResult",
    "IntervalTransitionSummary",
    "TRAJECTORY_CLASSES",
    "DEFAULT_TIMOR_18_CLASSES",
    "DEFAULT_IMPOSSIBLE_TRANSITIONS",
    "FeatureSelectionResult",
    "FeatureStackResult",
    "DownloadedAsset",
    "ImagerySearchResult",
    "ModelEvaluation",
    "ModelTuningResult",
    "build_classifier",
    "bayesian_smooth",
    "apply_mmu",
    "apply_transition_consistency",
    "calculate_class_area",
    "calculate_transition_matrix",
    "classify_from_probs",
    "DEFAULT_TIMOR_17_CLASSES",
    "DEFAULT_HYBRID_MMU_TABLE",
    "MMU_TABLE",
    "TransitionRule",
    "class_distribution",
    "classify_raster",
    "create_distance_features",
    "create_feature_stack",
    "mosaic_rasters",
    "stack_rasters",
    "detect_outliers_per_class",
    "detect_point_outliers",
    "flag_outliers_in_vector",
    "download_imagery",
    "drop_nan_samples",
    "extract_pixels_from_shapefile",
    "extract_point_features",
    "load_and_extract",
    "load_and_extract_points",
    "load_and_split_training_data",
    "load_points",
    "remove_outliers",
    "remove_outliers_from_vector",
    "save_cleaned_points",
    "save_flagged_points",
    "load_training_samples",
    "local_mean_variance",
    "local_variance",
    "logits_to_probs",
    "open_raster",
    "preview_imagery",
    "probs_to_logits",
    "search_imagery",
    "select_features",
    "train_test_split_data",
    "evaluate_model",
    "tune_model",
    "MaskRule",
    "process_tile",
    "refine_lulc_with_mask_layer",
]
