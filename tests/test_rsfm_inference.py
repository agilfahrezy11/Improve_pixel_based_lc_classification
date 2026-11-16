"""Tests for in-memory late ancillary fusion and streaming probability inference."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from sklearn.ensemble import RandomForestClassifier

from ML_LC_Classifier import (
    bayesian_smooth,
    classify_from_probs,
    process_tile,
)
from ML_LC_Classifier.rsfm import (
    SimpleCPUBackbone,
    finetune_rsfm,
    infer_rsfm_probabilities,
    prepare_rsfm_chips,
)
import geopandas as gpd
from shapely.geometry import box


def _create_utm_s2_and_wgs84_dem(tmp_path):
    """Create a 20x20 10m UTM raster and a coarser EPSG:4326 DEM raster."""
    s2_path = tmp_path / "s2_utm.tif"
    height, width, bands = 20, 20, 4
    utm_transform = from_origin(780000.0, 9050000.0, 10.0, 10.0)

    data = np.zeros((bands, height, width), dtype=np.float32)
    # Identical optical signature in top and bottom, separated by DEM elevation!
    # Plus left/right spectral difference for classes 1 & 2 vs 3
    data[:, :10, :] = np.array([0.04, 0.07, 0.04, 0.35], dtype=np.float32)[:, None, None]
    data[:, 10:, :] = np.array([0.04, 0.07, 0.04, 0.35], dtype=np.float32)[:, None, None]
    data[:, :2, :2] = np.nan

    with rasterio.open(
        s2_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
        crs="EPSG:32751",
        transform=utm_transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data)

    # Create a coarser EPSG:4326 DEM covering the UTM extent (with northern low=100m, southern high=1100m)
    # UTM (780000..780200, 9049800..9050000) in Zone 51S is roughly lon ~125.53, lat ~-8.58
    dem_path = tmp_path / "dem_wgs84.tif"
    dem_arr = np.zeros((1, 4, 4), dtype=np.float32)
    dem_arr[0, :2, :] = 100.0
    dem_arr[0, 2:, :] = 1100.0
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(125.50, -8.55, 0.02, 0.02),
        nodata=np.nan,
    ) as dst:
        dst.write(dem_arr)

    return s2_path, dem_path


def test_infer_rsfm_probabilities_late_fuses_ancillary_and_feeds_hybrid_assembly(tmp_path):
    """Streaming inference aligns ancillary rasters via WarpedVRT and feeds hybrid_assembly."""
    s2_path, dem_path = _create_utm_s2_and_wgs84_dem(tmp_path)
    backbone = SimpleCPUBackbone(embed_dim=6, seed=5)

    # Fit a tabular classifier on [6 embedding dims + 1 DEM elevation]
    rng = np.random.default_rng(42)
    X_low = np.hstack([rng.normal(0.0, 0.1, size=(20, 6)), np.full((20, 1), 100.0)])
    X_high = np.hstack([rng.normal(0.0, 0.1, size=(20, 6)), np.full((20, 1), 1100.0)])
    X_train = np.vstack([X_low, X_high]).astype(np.float32)
    y_train = np.array([1] * 20 + [2] * 20)
    clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)

    prob_path = tmp_path / "rsfm_probs.tif"
    out = infer_rsfm_probabilities(
        s2_path,
        prob_path,
        backbone=backbone,
        classifier=clf,
        ancillary_paths=[dem_path],
        chip_size=8,
        stride=4,
        block_size=12,
    )

    assert out == prob_path
    with rasterio.open(prob_path) as src:
        assert src.count == 2
        assert src.shape == (20, 20)
        probs = src.read()
        assert np.isnan(probs[:, :2, :2]).all()
        valid_probs = probs[:, 2:, :]
        assert np.isfinite(valid_probs).all()
        np.testing.assert_allclose(valid_probs.sum(axis=0), 1.0, atol=1e-5)

        # Feed directly into bayesian_smooth
        smoothed = bayesian_smooth(np.nan_to_num(probs, nan=0.0), smoothness=1.0, window_size=3)
        lulc_bayes = classify_from_probs(smoothed, class_values=[1, 2])
        assert lulc_bayes.shape == (20, 20)

    # Feed directly into hybrid_assembly.process_tile with a SNIC segment raster
    seg_path = tmp_path / "snic_segments.tif"
    seg_data = np.ones((1, 20, 20), dtype=np.int32)
    seg_data[0, 10:, :] = -2  # Test negative wrapped int32 GEE segment ID
    with rasterio.open(
        seg_path,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=1,
        dtype="int32",
        crs="EPSG:32751",
        transform=from_origin(780000.0, 9050000.0, 10.0, 10.0),
        nodata=0,
    ) as dst:
        dst.write(seg_data)

    hybrid_out = process_tile(
        prob_path=prob_path,
        seg_path=seg_path,
        out_dir=tmp_path / "hybrid_out",
        block_size=20,
        overlap=0,
    )
    assert hybrid_out["lulc"].exists()
    assert hybrid_out["stability"].exists()
    assert hybrid_out["ci"].exists()


def test_infer_rsfm_probabilities_from_lora_bundle_uint16(tmp_path):
    """Fine-tuned LoRA checkpoint bundle streams uint16 scaled probabilities."""
    s2_path = tmp_path / "s2_bundle.tif"
    utm_transform = from_origin(780000.0, 9050000.0, 10.0, 10.0)
    data = np.zeros((4, 16, 16), dtype=np.float32)
    data[:, :8, :] = np.array([0.04, 0.06, 0.04, 0.14], dtype=np.float32)[:, None, None]
    data[:, 8:, :] = np.array([0.05, 0.09, 0.03, 0.48], dtype=np.float32)[:, None, None]
    data[:, :2, :2] = np.nan

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

    training_path = tmp_path / "poly.geojson"
    gpd.GeoDataFrame(
        {"class_id": [1, 2]},
        geometry=[
            box(780040.0, 9049940.0, 780100.0, 9049980.0),
            box(780040.0, 9049860.0, 780100.0, 9049900.0),
        ],
        crs="EPSG:32751",
    ).to_file(training_path, driver="GeoJSON")

    backbone = SimpleCPUBackbone(embed_dim=8, seed=9)
    ds = prepare_rsfm_chips(s2_path, training_path, chip_size=8)
    ft = finetune_rsfm(ds, backbone=backbone, lora_rank=2, epochs=25, lr=0.1)

    prob_path = tmp_path / "bundle_probs_u16.tif"
    infer_rsfm_probabilities(
        s2_path,
        prob_path,
        bundle=ft.bundle,
        chip_size=8,
        stride=4,
        block_size=10,
        output_dtype="uint16",
    )

    with rasterio.open(prob_path) as src:
        assert src.dtypes == ("uint16", "uint16")
        assert src.nodata == 0
        arr = src.read()
        assert (arr[:, :2, :2] == 0).all()
        valid_sums = arr[:, 2:, :].sum(axis=0)
        assert np.all((valid_sums >= 9998) & (valid_sums <= 10002))
