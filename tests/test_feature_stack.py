import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from ML_LC_Classifier import create_feature_stack


def _write_raster(path, values):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
        nodata=np.nan,
    ) as destination:
        destination.write(np.asarray(values, dtype="float32"), 1)


def test_creates_composite_indices_terrain_and_manifest(tmp_path):
    values = {
        "B02": [[1, 1], [1, 1]],
        "B03": [[2, 2], [2, 2]],
        "B04": [[3, 3], [3, 3]],
        "B08": [[5, 5], [5, 5]],
        "B11": [[4, 4], [4, 4]],
        "data": [[0, 1], [2, 3]],
    }
    inputs = []
    for name, array in values.items():
        path = tmp_path / f"{name}.tif"
        _write_raster(path, array)
        inputs.append(path)

    result = create_feature_stack(
        inputs,
        tmp_path / "stack.tif",
        indices=["NDVI", "MNDWI"],
        asset_aliases={"data": "dem"},
        terrain_from=["dem"],
    )

    assert result.feature_names == (
        "B02", "B03", "B04", "B08", "B11", "dem", "NDVI", "MNDWI",
        "dem_slope", "dem_aspect",
    )
    with rasterio.open(result.stack_path) as stack:
        assert stack.count == 10
        assert stack.descriptions == result.feature_names
        np.testing.assert_allclose(stack.read(7), 0.25)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["feature_names"] == list(result.feature_names)
