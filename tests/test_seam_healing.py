import numpy as np
import rasterio
from rasterio.transform import from_origin
import pytest

from ML_LC_Classifier.seam_healing import (
    heal_array_seams,
    heal_raster_seams,
    heal_probability_array_spikes,
    heal_probability_seams,
)


def test_heal_array_seams_fills_nodata_gap():
    # 5x5 array with a 1-pixel column seam of nodata (0) down the middle
    arr = np.array([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
    ], dtype=np.uint8)

    healed = heal_array_seams(arr, nodata=0, max_seam_width=2)
    # The seam column (col 2) should be filled with neighbor values (1 or 2)
    assert not (healed[:, 2] == 0).any()
    assert set(np.unique(healed)).issubset({1, 2})


def test_heal_raster_seams(tmp_path):
    transform = from_origin(500000, 9000000, 10, 10)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0,
        "width": 6,
        "height": 6,
        "count": 1,
        "crs": "EPSG:32751",
        "transform": transform,
    }

    data = np.full((6, 6), 1, dtype=np.uint8)
    data[:, 3] = 0  # 1-pixel vertical seam

    in_tif = tmp_path / "seam_input.tif"
    out_tif = tmp_path / "seam_healed.tif"

    with rasterio.open(in_tif, "w", **profile) as dst:
        dst.write(data, 1)

    res = heal_raster_seams(in_tif, out_tif, nodata=0, max_seam_width=2)
    assert res.exists()

    with rasterio.open(out_tif) as src:
        healed_data = src.read(1)
        assert (healed_data == 1).all()


def test_heal_probability_array_spikes():
    # 3 classes: uniform prob 0.33, with one pixel having a spurious spike of 0.99 in band 0
    probs = np.full((3, 5, 5), 0.33, dtype=np.float32)
    probs[0, 2, 2] = 0.99  # Isolated probability spike
    probs[1, 2, 2] = 0.005
    probs[2, 2, 2] = 0.005

    healed = heal_probability_array_spikes(probs, spike_threshold=0.90)
    # Spike should be dampened toward neighbor mean
    assert healed[0, 2, 2] < 0.90
    assert np.isclose(np.sum(healed[:, 2, 2]), 1.0, atol=0.05)
