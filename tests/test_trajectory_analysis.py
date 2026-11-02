import numpy as np
import rasterio
from rasterio.transform import from_origin
import pytest

from ML_LC_Classifier.trajectory_analysis import (
    _classify_pixel_trajectory_vector,
    analyze_multi_temporal_transitions,
)


def test_classify_pixel_trajectory_flip_back_multi_epoch():
    # 4 epochs: 2005, 2015, 2020, 2025
    # Pixel A: [1, 1, 16, 1] -> forest, forest, built-up (flip), forest
    # Pixel B: [1, 1, 1, 1]   -> persistent forest (1)
    # Pixel C: [1, 14, 14, 14] -> forest loss (2)
    traj = np.array([
        [1, 1, 16, 1],
        [1, 1, 1, 1],
        [1, 14, 14, 14]
    ], dtype=np.uint8)

    cats = _classify_pixel_trajectory_vector(traj)
    # Pixel A must be categorized as 9 (spurious noise / flip-back)
    assert cats[0] == 9
    # Pixel B is persistent stable forest (1)
    assert cats[1] == 1
    # Pixel C is forest loss (2)
    assert cats[2] == 2


def test_analyze_multi_temporal_transitions(tmp_path):
    transform = from_origin(500000, 9000000, 10, 10)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0,
        "width": 2,
        "height": 2,
        "count": 1,
        "crs": "EPSG:32751",
        "transform": transform,
    }

    p2005 = tmp_path / "lc_2005.tif"
    p2015 = tmp_path / "lc_2015.tif"
    p2025 = tmp_path / "lc_2025.tif"

    d2005 = np.array([[1, 1], [1, 16]], dtype=np.uint8)
    d2015 = np.array([[1, 14], [1, 16]], dtype=np.uint8)
    d2025 = np.array([[1, 1], [14, 16]], dtype=np.uint8)

    for path, data in [(p2005, d2005), (p2015, d2015), (p2025, d2025)]:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)

    epoch_rasters = {2005: p2005, 2015: p2015, 2025: p2025}
    traj_out = tmp_path / "traj.tif"

    res = analyze_multi_temporal_transitions(
        epoch_rasters,
        output_trajectory_raster=traj_out,
        block_size=2,
    )

    assert len(res.interval_summaries) == 2
    assert traj_out.exists()

    with rasterio.open(traj_out) as src:
        out_data = src.read(1)
        # [0, 1] was [1, 14, 1] -> flip-back, should be 9
        assert out_data[0, 1] == 9
        # [0, 0] was [1, 1, 1] -> stable, should be 1
        assert out_data[0, 0] == 1
