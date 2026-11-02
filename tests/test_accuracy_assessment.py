import json
import numpy as np
import pandas as pd
import pytest

from ML_LC_Classifier.accuracy_assessment import (
    ThematicAccuracyResult,
    compute_thematic_accuracy,
)


def test_compute_thematic_accuracy_sample_metrics():
    # 3 classes: 1, 2, 3
    # Ground truth (reference) vs Predicted (map)
    y_true = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    y_pred = [1, 1, 1, 2, 2, 2, 1, 3, 3, 2]

    class_names = {1: "Forest", 2: "Agriculture", 3: "Water"}
    res = compute_thematic_accuracy(
        y_true,
        y_pred,
        class_names=class_names,
        labels=[1, 2, 3],
    )

    metrics = res.overall_metrics
    # 7 correct out of 10 -> OA = 0.70
    assert np.isclose(metrics["overall_accuracy_sample"], 0.70)
    assert metrics["total_samples"] == 10
    assert metrics["num_classes_evaluated"] == 3

    # Check confusion matrix counts
    # Rows = Map (pred), Cols = Ref (true) with formatted labels "1: Forest", etc.
    cm = res.confusion_matrix_counts
    assert cm.loc["1: Forest", "1: Forest"] == 3
    assert cm.loc["1: Forest", "2: Agriculture"] == 1
    assert cm.loc["1: Forest", "3: Water"] == 0

    # User's accuracy for Forest: 3 / (3 + 1) = 0.75
    class_df = res.class_metrics
    forest_row = class_df[class_df["class_id"] == 1].iloc[0]
    assert np.isclose(forest_row["user_accuracy"], 0.75)

    # Producer's accuracy for Forest: 3 / 4 = 0.75
    assert np.isclose(forest_row["producer_accuracy_sample"], 0.75)


def test_compute_thematic_accuracy_olofsson_area_weighted(tmp_path):
    # When mapped_pixel_counts is supplied, Olofsson et al. 2014 area weighting applies
    y_true = [1, 1, 2, 2]
    y_pred = [1, 2, 1, 2]

    # Map counts: Class 1 has 10,000 pixels, Class 2 has 90,000 pixels (Total = 100,000)
    mapped_counts = {1: 10000, 2: 90000}
    # 10m x 10m pixel = 100 m²
    pixel_area_m2 = 100.0

    res = compute_thematic_accuracy(
        y_true,
        y_pred,
        labels=[1, 2],
        mapped_pixel_counts=mapped_counts,
        pixel_area_m2=pixel_area_m2,
    )

    metrics = res.overall_metrics
    assert "overall_accuracy_area_weighted" in metrics
    assert res.confusion_matrix_area_prop is not None

    # Class metrics must have error-adjusted area columns
    class_df = res.class_metrics
    assert "adjusted_area_ha" in class_df.columns
    assert "adjusted_area_se_ha" in class_df.columns
    assert "mapped_area_ha" in class_df.columns

    # Verify report export
    saved = res.save_reports(tmp_path, prefix="test_run")
    assert saved["overall_metrics_json"].exists()
    assert saved["class_metrics_csv"].exists()
    assert saved["confusion_matrix_counts_csv"].exists()
    assert saved["confusion_matrix_area_prop_csv"].exists()

    with open(saved["overall_metrics_json"], "r", encoding="utf-8") as f:
        meta = json.load(f)
        assert "overall_accuracy_area_weighted" in meta
