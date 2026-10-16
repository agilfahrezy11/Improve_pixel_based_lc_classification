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


def test_default_hybrid_mmu_table():
    from ML_LC_Classifier import DEFAULT_HYBRID_MMU_TABLE
    assert DEFAULT_HYBRID_MMU_TABLE[1] == 50
    assert DEFAULT_HYBRID_MMU_TABLE[2] == 30
    assert DEFAULT_HYBRID_MMU_TABLE[16] == 20
    assert DEFAULT_HYBRID_MMU_TABLE[17] == 20


def test_apply_mmu_windowed_streaming(tmp_path):
    # Test streaming path by setting block_size smaller than image dimensions
    values = np.ones((100, 100), dtype=np.uint8)
    # Put a small patch of class 2 of size 3 in the middle
    values[49:52, 49] = 2
    source_path = tmp_path / "large_mock.tif"
    output_path = tmp_path / "filtered_mock.tif"
    _write_classification(source_path, values)

    apply_mmu(
        source_path,
        output_path,
        table={1: 0, 2: 5},
        iterations=1,
        block_size=64,
        overlap=16,
    )

    with rasterio.open(output_path) as res:
        out_arr = res.read(1)
        # The small patch should have been filled by focal mode to 1
        assert (out_arr == 1).all()


def test_calculate_transition_matrix(tmp_path):
    from ML_LC_Classifier import calculate_transition_matrix

    t1_data = np.array([[1, 1], [2, 2]], dtype=np.uint8)
    t2_data = np.array([[1, 2], [2, 1]], dtype=np.uint8)

    t1_p = tmp_path / "t1.tif"
    t2_p = tmp_path / "t2.tif"
    _write_classification(t1_p, t1_data)
    _write_classification(t2_p, t2_data)

    custom_classes = {1: "Forest", 2: "NonForest"}
    df_counts, df_areas = calculate_transition_matrix(
        t1_p, t2_p, class_names=custom_classes, block_size=64
    )

    assert df_counts.loc["Forest", "Forest"] == 1
    assert df_counts.loc["Forest", "NonForest"] == 1
    assert df_counts.loc["NonForest", "Forest"] == 1
    assert df_counts.loc["NonForest", "NonForest"] == 1


def test_apply_transition_consistency_t2_correction(tmp_path):
    from ML_LC_Classifier import apply_transition_consistency

    # t1: Secondary Forest (2)
    # t2: Primary Forest (1) -> Impossible transition (2 -> 1)
    t1_data = np.full((10, 10), 2, dtype=np.uint8)
    t2_data = np.full((10, 10), 1, dtype=np.uint8)

    t1_p = tmp_path / "t1.tif"
    t2_p = tmp_path / "t2.tif"
    out_p = tmp_path / "t2_corrected.tif"
    flag_p = tmp_path / "flag.tif"

    _write_classification(t1_p, t1_data)
    _write_classification(t2_p, t2_data)

    # Correct t2 (demote 2025 back to Secondary Forest)
    apply_transition_consistency(
        t1_p,
        t2_p,
        out_p,
        forbidden=[(2, 1)],
        correction_target="t2",
        flag_path=flag_p,
        block_size=64,
    )

    with rasterio.open(out_p) as res:
        corrected = res.read(1)
        # All pixels should have been reverted back to class 2
        assert (corrected == 2).all()

    with rasterio.open(flag_p) as f_res:
        flag = f_res.read(1)
        # All pixels were corrected
        assert (flag == 1).all()
