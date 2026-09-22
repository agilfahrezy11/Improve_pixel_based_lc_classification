import numpy as np
import rasterio
from rasterio.transform import from_origin

from ML_LC_Classifier import mosaic_rasters, stack_rasters


def _write_raster(path, values, transform=from_origin(0, 2, 1, 1), description=None):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as destination:
        destination.write(np.asarray(values, dtype="uint8"), 1)
        if description:
            destination.set_band_description(1, description)


def test_mosaic_rasters_merges_tiles_with_max(tmp_path):
    _write_raster(tmp_path / "tile_b.tif", [[0, 1], [0, 0]])
    _write_raster(tmp_path / "tile_a.tif", [[1, 0], [0, 0]], from_origin(2, 2, 1, 1))

    output = mosaic_rasters(tmp_path / "tile_*.tif", tmp_path / "mosaic.tif")

    with rasterio.open(output) as raster:
        assert (raster.width, raster.height) == (4, 2)
        np.testing.assert_array_equal(raster.read(1), [[0, 1, 1, 0], [0, 0, 0, 0]])


def test_stack_rasters_writes_named_aligned_bands(tmp_path):
    red = tmp_path / "red.tif"
    nir = tmp_path / "nir.tif"
    _write_raster(red, [[1, 2], [3, 4]], description="red")
    _write_raster(nir, [[5, 6], [7, 8]], description="nir")

    output = stack_rasters([nir, red], tmp_path / "stack.tif", band_names=["NIR", "RED"])

    with rasterio.open(output) as raster:
        assert raster.count == 2
        assert raster.descriptions == ("NIR", "RED")
        np.testing.assert_array_equal(raster.read(), [[[5, 6], [7, 8]], [[1, 2], [3, 4]]])