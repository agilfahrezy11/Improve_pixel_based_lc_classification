import numpy as np
import rasterio
from rasterio.transform import from_origin

from ML_LC_Classifier import MMU_TABLE, apply_mmu


def _write_classification(path, values, nodata=255):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:32632",
        transform=from_origin(500000, 1000, 10, 10),
        nodata=nodata,
    ) as destination:
        destination.write(values, 1)


def test_apply_mmu_fills_small_patches_and_preserves_metadata_and_nodata(tmp_path):
    values = np.ones((5, 5), dtype=np.uint8)
    values[2, 2] = 2
    values[0, 0] = 255
    source_path = tmp_path / "classification.tif"
    output_path = tmp_path / "filtered.tif"
    _write_classification(source_path, values)

    result_path = apply_mmu(
        source_path,
        output_path,
        table={1: 0, 2: 2},
        iterations=1,
    )

    assert result_path == output_path
    with rasterio.open(source_path) as source, rasterio.open(output_path) as result:
        assert result.count == 1
        assert result.dtypes == ("uint8",)
        assert result.descriptions == ("classification",)
        assert result.crs == source.crs
        assert result.transform == source.transform
        assert result.nodata == 255
        assert result.read(1)[2, 2] == 1
        assert result.read(1, masked=True).mask[0, 0]


def test_apply_mmu_uses_eight_connected_components(tmp_path):
    values = np.ones((5, 5), dtype=np.uint8)
    values[2, 2] = 2
    values[3, 3] = 2
    source_path = tmp_path / "diagonal.tif"
    output_path = tmp_path / "diagonal_filtered.tif"
    _write_classification(source_path, values)

    apply_mmu(source_path, output_path, table={1: 0, 2: 2}, iterations=1)

    with rasterio.open(output_path) as result:
        np.testing.assert_array_equal(result.read(1)[2:4, 2:4], [[2, 1], [1, 2]])


def test_default_mmu_table_matches_requested_thresholds():
    assert MMU_TABLE == {
        1: 50,
        2: 50,
        3: 50,
        4: 50,
        5: 50,
        6: 0,
        7: 50,
        8: 0,
        9: 20,
        10: 0,
        11: 20,
        12: 20,
        13: 50,
        14: 50,
        15: 20,
        16: 10,
        17: 0,
    }
