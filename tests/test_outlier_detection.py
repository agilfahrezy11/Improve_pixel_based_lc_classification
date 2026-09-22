import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from ML_LC_Classifier import (
    extract_point_features,
    remove_outliers,
    remove_outliers_from_vector,
)


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


def test_remove_outliers_from_vector_preserves_geometry_and_crs():
    samples = gpd.GeoDataFrame(
        {"point_id": ["p1", "p2"], "class_id": [1, 1]},
        geometry=[Point(100, 0), Point(101, 1)],
        crs="EPSG:4326",
    )
    flags = gpd.GeoDataFrame(
        {"point_id": ["p1", "p2"], "outlier": [False, True]},
    )

    cleaned = remove_outliers_from_vector(samples, flags)

    assert cleaned["point_id"].tolist() == ["p1"]
    assert cleaned.geometry.iloc[0].equals(Point(100, 0))
    assert cleaned.crs == samples.crs


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
        {
            "point_id": ["p1", "p2"],
            "class_id": [1, 2],
            "LULC_type": ["Forest", "Water"],
        },
        geometry=[Point(0.5, 1.5), Point(1.5, 0.5)],
        crs="EPSG:4326",
    )

    with rasterio.open(raster_path) as raster:
        result = extract_point_features(
            points,
            raster,
            "class_id",
            class_name_field="LULC_type",
        )

    assert list(result.columns) == [
        "point_id",
        "class_id",
        "LULC_type",
        "B2",
        "NDVI",
    ]
    assert result.to_dict("records") == [
        {
            "point_id": "p1",
            "class_id": 1,
            "LULC_type": "Forest",
            "B2": 1.0,
            "NDVI": 10.0,
        },
        {
            "point_id": "p2",
            "class_id": 2,
            "LULC_type": "Water",
            "B2": 4.0,
            "NDVI": 40.0,
        },
    ]
