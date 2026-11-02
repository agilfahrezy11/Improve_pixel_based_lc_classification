import numpy as np
import rasterio
from rasterio.transform import from_origin
import pytest

from ML_LC_Classifier.area_calculation import calculate_class_area, plot_class_area


def test_calculate_class_area_projected_streaming(tmp_path):
    data = np.array([
        [1, 1, 2],
        [2, 0, 3],
        [1, 3, 3]
    ], dtype=np.uint8)
    # 10m x 10m pixel = 100 m² = 0.01 ha per pixel
    transform = from_origin(500000, 9000000, 10, 10)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0,
        "width": 3,
        "height": 3,
        "count": 1,
        "crs": "EPSG:32751",  # Projected UTM
        "transform": transform,
    }
    raster_file = tmp_path / "test_area.tif"
    with rasterio.open(raster_file, "w", **profile) as dst:
        dst.write(data, 1)

    result = calculate_class_area(raster_file, block_size=2)
    df = result.area_ha

    assert len(df) == 3
    # Class 1: 3 pixels -> 0.03 ha
    assert df.loc[df["class_id"] == 1, "pixel_count"].values[0] == 3
    assert np.isclose(df.loc[df["class_id"] == 1, "area_ha"].values[0], 0.03)

    # Class 2: 2 pixels -> 0.02 ha
    assert df.loc[df["class_id"] == 2, "pixel_count"].values[0] == 2
    assert np.isclose(df.loc[df["class_id"] == 2, "area_ha"].values[0], 0.02)

    # Class 3: 3 pixels -> 0.03 ha
    assert df.loc[df["class_id"] == 3, "pixel_count"].values[0] == 3
    assert np.isclose(df.loc[df["class_id"] == 3, "area_ha"].values[0], 0.03)

    # Figure should be None by default
    assert result.figure is None


def test_calculate_class_area_csv_export(tmp_path):
    data = np.array([[1, 2]], dtype=np.uint8)
    transform = from_origin(125.0, -8.5, 0.0001, 0.0001)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0,
        "width": 2,
        "height": 1,
        "count": 1,
        "crs": "EPSG:4326",  # Geographic WGS-84
        "transform": transform,
    }
    raster_file = tmp_path / "test_geo.tif"
    with rasterio.open(raster_file, "w", **profile) as dst:
        dst.write(data, 1)

    csv_out = tmp_path / "out" / "area.csv"
    result = calculate_class_area(raster_file, csv_path=csv_out)
    assert csv_out.exists()
    assert result.csv_path == csv_out.resolve()
