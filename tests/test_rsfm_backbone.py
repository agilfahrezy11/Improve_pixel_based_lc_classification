"""Tests for the RSFM optional dependency boundary, OpenGeoAI loader, and windowed embeddings."""

import json
from pathlib import Path
import sys
import types
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from ML_LC_Classifier import (
    FeatureStackResult,
    load_and_split_training_data,
    tune_model,
)
from ML_LC_Classifier.rsfm import (
    RSFMBackbone,
    SimpleCPUBackbone,
    extract_rsfm_embeddings,
    list_opengeoai_models,
    load_opengeoai_backbone,
)


def test_rsfm_optional_import_boundary(monkeypatch):
    """Base package imports without geoai/torch/peft, and loader gives clear guidance."""
    monkeypatch.setitem(sys.modules, "geoai", None)
    with pytest.raises(ImportError, match=r"pip install -e \"\.\[rsfm\]\""):
        load_opengeoai_backbone("clay-v1.5")

    with pytest.raises(ImportError, match=r"pip install -e \"\.\[rsfm\]\""):
        list_opengeoai_models()


def test_opengeoai_loader_delegates_to_geoai_catalog(monkeypatch):
    """OpenGeoAI loader delegates listing and model loading to the geoai boundary."""
    fake_geoai = types.ModuleType("geoai")
    fake_geoai.list_foundation_models = lambda **kwargs: ["prithvi_eo_v2_300", "clay_v1_5"]
    fake_geoai.load_foundation_model = lambda name, **kwargs: types.SimpleNamespace(
        model_name=name,
        embed_dim=64,
    )
    monkeypatch.setitem(sys.modules, "geoai", fake_geoai)

    assert list_opengeoai_models() == ["prithvi_eo_v2_300", "clay_v1_5"]
    adapter = load_opengeoai_backbone("prithvi_eo_v2_300")
    assert adapter.embed_dim == 64


def test_simple_cpu_backbone_produces_deterministic_spatial_embeddings():
    """Lightweight CPU backbone satisfies RSFMBackbone and returns deterministic spatial maps."""
    backbone = SimpleCPUBackbone(embed_dim=8, seed=42)
    assert isinstance(backbone, RSFMBackbone)
    assert backbone.embed_dim == 8

    rng = np.random.default_rng(0)
    chips = rng.normal(size=(2, 4, 16, 16)).astype(np.float32)
    out1 = backbone.extract_features(chips, wavelengths_um=[0.49, 0.56, 0.66, 0.84])
    out2 = backbone.extract_features(chips, wavelengths_um=[0.49, 0.56, 0.66, 0.84])

    assert out1.shape == (2, 8, 16, 16)
    assert out1.dtype == np.float32
    np.testing.assert_array_equal(out1, out2)
    assert np.isfinite(out1).all()


def test_extract_rsfm_embeddings_streams_blended_stack_into_tabular_pipeline(tmp_path):
    """Windowed extraction blends overlapping chips, preserves nodata, and feeds tune_model."""
    raster_path = tmp_path / "sentinel2.tif"
    height, width, bands = 20, 20, 4
    transform = from_origin(0, 20, 1, 1)

    data = np.zeros((bands, height, width), dtype=np.float32)
    # Class 1 region (top half, low NIR) vs Class 2 region (bottom half, high NIR)
    data[:, :10, :] = np.array([0.05, 0.06, 0.04, 0.15], dtype=np.float32)[:, None, None]
    data[:, 10:, :] = np.array([0.04, 0.08, 0.03, 0.45], dtype=np.float32)[:, None, None]
    # Nodata block in top-left 2x2
    data[:, :2, :2] = np.nan

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data)

    backbone = SimpleCPUBackbone(embed_dim=6, seed=7)
    output_path = tmp_path / "embeddings.tif"
    result = extract_rsfm_embeddings(
        raster_path,
        output_path,
        backbone=backbone,
        chip_size=8,
        stride=4,
        wavelengths_um=[0.49, 0.56, 0.66, 0.84],
    )

    assert isinstance(result, FeatureStackResult)
    assert result.feature_names == tuple(f"embed_{i:03d}" for i in range(6))

    with rasterio.open(result.stack_path) as src:
        assert src.count == 6
        assert src.shape == (height, width)
        assert src.descriptions == result.feature_names
        emb = src.read()
        assert np.isnan(emb[:, :2, :2]).all()
        assert np.isfinite(emb[:, 2:, :]).all()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["feature_names"] == list(result.feature_names)
    assert manifest["chip_size"] == 8
    assert manifest["stride"] == 4

    # Verify compatibility with existing load_and_split_training_data + tune_model
    training_path = tmp_path / "samples.geojson"
    gdf = gpd.GeoDataFrame(
        {"class_id": [1, 2]},
        geometry=[box(3, 12, 18, 18), box(3, 2, 18, 8)],
        crs="EPSG:4326",
    )
    gdf.to_file(training_path, driver="GeoJSON")

    X_train, X_test, y_train, y_test = load_and_split_training_data(
        result.stack_path,
        training_path,
        class_field="class_id",
        test_size=0.3,
        random_state=42,
    )
    assert X_train.shape[1] == 6
    tuned = tune_model(
        classifier="RandomForest",
        parameter_space={"n_estimators": [10]},
        X_train=X_train,
        y_train=y_train,
        cv=2,
        random_state=42,
    )
    preds = tuned.model.predict(X_test)
    assert len(preds) == len(y_test)
    assert (preds == y_test).mean() > 0.9
