import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from ML_LC_Classifier import create_distance_features


def _write_raster(path, values, *, crs="EPSG:32651", transform=None):
    values = np.asarray(values, dtype="uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs=crs,
        transform=transform or from_origin(500000, 100, 10, 10),
        nodata=255,
    ) as destination:
        destination.write(values, 1)


def test_distance_from_binary_raster_uses_reference_pixel_units(tmp_path):
    transform = from_origin(500000, 100, 10, 10)
    source_path = tmp_path / "binary.tif"
    reference_path = tmp_path / "reference.tif"
    _write_raster(source_path, [[1, 0, 0], [0, 0, 0], [0, 0, 0]], transform=transform)
    _write_raster(reference_path, np.zeros((3, 3)), transform=transform)

    output_path = create_distance_features(source_path, reference_path, tmp_path / "distance.tif")

    with rasterio.open(output_path) as result:
        assert result.crs.to_string() == "EPSG:32651"
        assert result.descriptions == ("distance",)
        np.testing.assert_allclose(
            result.read(1),
            [[0, 10, 20], [10, np.sqrt(200), np.sqrt(500)], [20, np.sqrt(500), np.sqrt(800)]],
        )


def test_distance_from_shapefile_burns_feature_cells(tmp_path):
    transform = from_origin(500000, 100, 10, 10)
    reference_path = tmp_path / "reference.tif"
    _write_raster(reference_path, np.zeros((3, 3)), transform=transform)
    vector_path = tmp_path / "feature.shp"
    gpd.GeoDataFrame(
        geometry=[Point(500005, 95)],
        crs="EPSG:32651",
    ).to_file(vector_path)

    output_path = create_distance_features(vector_path, reference_path, tmp_path / "distance.tif")

    with rasterio.open(output_path) as result:
        assert result.read(1)[0, 0] == 0
        assert result.read(1)[0, 1] == 10