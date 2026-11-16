"""Tests for GEOBIA SNIC segment-pooled RSFM embeddings and probability projection."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from ML_LC_Classifier import process_tile, tune_model
from ML_LC_Classifier.rsfm import (
    SimpleCPUBackbone,
    assign_segment_labels,
    pool_rsfm_by_segments,
    project_segment_probabilities,
)


def _create_geobia_fixtures(tmp_path):
    """Create a 16x16 raster, coarse DEM, SNIC segments with negative int32 IDs, and training polys."""
    utm_transform = from_origin(780000.0, 9050000.0, 10.0, 10.0)
    s2_path = tmp_path / "s2.tif"
    data = np.zeros((4, 16, 16), dtype=np.float32)
    data[:, :8, :] = np.array([0.04, 0.06, 0.04, 0.14], dtype=np.float32)[:, None, None]
    data[:, 8:, :] = np.array([0.05, 0.09, 0.03, 0.48], dtype=np.float32)[:, None, None]
    data[:, 0, 0] = np.nan

    with rasterio.open(
        s2_path,
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=4,
        dtype="float32",
        crs="EPSG:32751",
        transform=utm_transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data)

    dem_path = tmp_path / "dem.tif"
    dem = np.zeros((1, 16, 16), dtype=np.float32)
    dem[0, :8, :] = 100.0
    dem[0, 8:, :] = 900.0
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=1,
        dtype="float32",
        crs="EPSG:32751",
        transform=utm_transform,
        nodata=np.nan,
    ) as dst:
        dst.write(dem)

    # SNIC segments:
    # - Segment 101 (top-left), Segment 102 (top-right)
    # - Segment -2000000000 (wrapped negative int32 GEE ID, bottom-left)
    # - Segment -2000000001 (bottom-right)
    # - Pixel (0, 1) set to 0 (nodata), pixel (0, 2) set to -1 (GEE nodata)
    seg_path = tmp_path / "snic_seg.tif"
    seg = np.zeros((1, 16, 16), dtype=np.int32)
    seg[0, :8, :8] = 101
    seg[0, :8, 8:] = 102
    seg[0, 8:, :8] = -2000000000
    seg[0, 8:, 8:] = -2000000001
    seg[0, 0, 1] = 0
    seg[0, 0, 2] = -1

    with rasterio.open(
        seg_path,
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=1,
        dtype="int32",
        crs="EPSG:32751",
        transform=utm_transform,
        nodata=0,
    ) as dst:
        dst.write(seg)

    # Training polygons covering Segment 101 (Class 1) and Segment -2000000000 (Class 2)
    training_path = tmp_path / "samples.geojson"
    gpd.GeoDataFrame(
        {"class_id": [1, 2]},
        geometry=[
            box(780010.0, 9049930.0, 780070.0, 9049980.0),
            box(780010.0, 9049850.0, 780070.0, 9049900.0),
        ],
        crs="EPSG:32751",
    ).to_file(training_path, driver="GeoJSON")

    return s2_path, dem_path, seg_path, training_path


def test_geobia_segment_pooling_training_and_projection(tmp_path):
    """Pool RSFM + ancillary features per SNIC segment, train RF, and project probabilities."""
    s2_path, dem_path, seg_path, training_path = _create_geobia_fixtures(tmp_path)
    backbone = SimpleCPUBackbone(embed_dim=4, seed=13)

    seg_df = pool_rsfm_by_segments(
        s2_path,
        seg_path,
        backbone=backbone,
        ancillary_paths=[dem_path],
        include_std=True,
        chip_size=8,
        stride=4,
        block_size=10,
    )

    # Only the 4 valid segments (nodata 0 and -1 excluded; negative int32 IDs preserved)
    assert set(seg_df["segment_id"]) == {101, 102, -2000000000, -2000000001}
    # 5 features (4 embed + 1 DEM) x 2 (mean, std) = 10 feature columns + segment_id + pixel_count
    feature_cols = [c for c in seg_df.columns if c.endswith("_mean") or c.endswith("_std")]
    assert len(feature_cols) == 10
    # Segment 101 had (0,0)=NaN, (0,1)=seg0, (0,2)=seg-1 -> 64 - 3 = 61 valid pixels
    row_101 = seg_df.loc[seg_df["segment_id"] == 101].iloc[0]
    assert row_101["pixel_count"] == 61
    assert np.isclose(row_101["anc_000_mean"], 100.0)

    labels_df = assign_segment_labels(seg_path, training_path, class_field="class_id", block_size=10)
    merged = seg_df.merge(labels_df, on="segment_id", how="inner")
    assert set(merged["segment_id"]) == {101, -2000000000}

    # Add synthetic samples or duplicate so CV=2 in tune_model has >=2 samples per class
    X_train = np.vstack([merged[feature_cols].to_numpy(), merged[feature_cols].to_numpy()])
    y_train = np.concatenate([merged["class_id"].to_numpy(), merged["class_id"].to_numpy()])
    tuned = tune_model(
        classifier="RandomForest",
        parameter_space={"n_estimators": [10]},
        X_train=X_train,
        y_train=y_train,
        cv=2,
        random_state=42,
    )

    seg_probs = tuned.model.predict_proba(seg_df[feature_cols].to_numpy())
    prob_path = tmp_path / "obj_probs.tif"
    project_segment_probabilities(
        seg_path,
        seg_df["segment_id"].to_numpy(),
        seg_probs,
        prob_path,
        class_values=[int(c) for c in tuned.model.classes_],
        block_size=10,
    )

    with rasterio.open(prob_path) as src:
        assert src.count == 2
        assert src.shape == (16, 16)
        arr = src.read()
        # Nodata segments 0 and -1 at (0, 1) and (0, 2) must be NaN
        assert np.isnan(arr[:, 0, 1]).all()
        assert np.isnan(arr[:, 0, 2]).all()
        # Segment -2000000000 (bottom-left) must favor Class 2
        assert arr[1, 12, 4] > arr[0, 12, 4]
        # Segment 101 (top-left) must favor Class 1
        assert arr[0, 4, 4] > arr[1, 4, 4]

    # Verify projected object probability raster feeds directly into hybrid_assembly.process_tile
    hybrid_out = process_tile(
        prob_path=prob_path,
        seg_path=seg_path,
        out_dir=tmp_path / "hybrid_geobia",
        block_size=16,
        overlap=0,
    )
    assert hybrid_out["lulc"].exists()
