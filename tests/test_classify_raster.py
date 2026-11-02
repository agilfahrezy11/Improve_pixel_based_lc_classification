import numpy as np
import rasterio
from rasterio.transform import from_origin
import pytest

from ML_LC_Classifier.classify_raster import (
    classify_raster,
    classify_raster_proba,
    validate_feature_stack_manifest,
)


class MockClassifier:
    """Mock classifier implementing predict and predict_proba."""
    def __init__(self):
        self.classes_ = np.array([1, 2], dtype=np.uint8)
        self.n_features_in_ = 2

    def predict(self, X):
        # Class 1 if feature_0 > feature_1 else 2
        return np.where(X[:, 0] > X[:, 1], 1, 2).astype(np.uint8)

    def predict_proba(self, X):
        p1 = np.where(X[:, 0] > X[:, 1], 0.8, 0.2)
        p2 = 1.0 - p1
        return np.column_stack([p1, p2])


def test_classify_raster_hard_prediction(tmp_path):
    transform = from_origin(500000, 9000000, 10, 10)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": 2,
        "height": 2,
        "count": 2,
        "crs": "EPSG:32751",
        "transform": transform,
    }

    b1 = np.array([[10.0, 2.0], [5.0, 1.0]], dtype=np.float32)
    b2 = np.array([[5.0, 8.0], [5.0, -9999.0]], dtype=np.float32)

    in_tif = tmp_path / "features.tif"
    with rasterio.open(in_tif, "w", **profile) as dst:
        dst.write(b1, 1)
        dst.write(b2, 2)

    out_tif = tmp_path / "classified.tif"
    model = MockClassifier()

    classify_raster(in_tif, model, out_tif, mode="hard", block_size=1)

    assert out_tif.exists()
    with rasterio.open(out_tif) as src:
        assert src.count == 1
        data = src.read(1)
        # [0, 0]: 10 > 5 -> 1
        assert data[0, 0] == 1
        # [0, 1]: 2 < 8 -> 2
        assert data[0, 1] == 2
        # [1, 1]: has nodata -9999 -> nodata (0)
        assert data[1, 1] == 0


def test_classify_raster_proba_prediction(tmp_path):
    transform = from_origin(500000, 9000000, 10, 10)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": 2,
        "height": 1,
        "count": 2,
        "crs": "EPSG:32751",
        "transform": transform,
    }

    b1 = np.array([[10.0, 2.0]], dtype=np.float32)
    b2 = np.array([[5.0, 8.0]], dtype=np.float32)

    in_tif = tmp_path / "features.tif"
    with rasterio.open(in_tif, "w", **profile) as dst:
        dst.write(b1, 1)
        dst.write(b2, 2)

    out_tif = tmp_path / "proba.tif"
    model = MockClassifier()

    classify_raster_proba(in_tif, model, out_tif, proba_scale=10000.0)

    assert out_tif.exists()
    with rasterio.open(out_tif) as src:
        assert src.count == 2
        assert src.dtypes[0] == "uint16"
        p1 = src.read(1)
        p2 = src.read(2)
        # pixel 0: p1 = 0.8 * 10000 = 8000, p2 = 2000
        assert p1[0, 0] == 8000
        assert p2[0, 0] == 2000
        # pixel 1: p1 = 0.2 * 10000 = 2000, p2 = 8000
        assert p1[0, 1] == 2000
        assert p2[0, 1] == 8000
