import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from ML_LC_Classifier import extract_point_features, remove_outliers


def test_remove_outliers_returns_clean_copy_without_flagged_points():
    samples = gpd.GeoDataFrame(
        {"point_id": ["p1", "p2", "p3"], "class_id": [1, 1, 2]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
    )
    flags = gpd.GeoDataFrame(
        {"point_id": ["p1", "p2"], "outlier": [False, True]},
    )

    cleaned = remove_outliers(samples, flags)

    assert cleaned["point_id"].tolist() == ["p1", "p3"]
    assert samples["point_id"].tolist() == ["p1", "p2", "p3"]


def test_extract_point_features_preserves_ids_labels_and_band_names(tmp_path):
    raster_path = tmp_path / "features.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
        nodata=-9999,
    ) as raster:
        raster.write(np.array([[1, 2], [3, 4]], dtype="float32"), 1)
        raster.write(np.array([[10, 20], [30, 40]], dtype="float32"), 2)
        raster.set_band_description(1, "B2")
        raster.set_band_description(2, "NDVI")

    points = gpd.GeoDataFrame(
        {"point_id": ["p1", "p2"], "class_id": [1, 2]},
        geometry=[Point(0.5, 1.5), Point(1.5, 0.5)],
        crs="EPSG:4326",
    )

    with rasterio.open(raster_path) as raster:
        result = extract_point_features(points, raster, "class_id")

    assert list(result.columns) == ["point_id", "class_id", "B2", "NDVI"]
    assert result.to_dict("records") == [
        {"point_id": "p1", "class_id": 1, "B2": 1.0, "NDVI": 10.0},
        {"point_id": "p2", "class_id": 2, "B2": 4.0, "NDVI": 40.0},
    ]
