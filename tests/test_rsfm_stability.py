"""Tests for bi-temporal RSFM embedding stability masks and sample disturbance filtering."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from ML_LC_Classifier import apply_transition_consistency
from ML_LC_Classifier.rsfm import (
    SimpleCPUBackbone,
    compute_embedding_stability,
    filter_disturbed_samples,
)


def _create_bitemporal_rasters(tmp_path):
    """Create t1 and t2 4-band rasters: top half unchanged, bottom half changed."""
    transform = from_origin(780000.0, 9050000.0, 10.0, 10.0)
    t1_path = tmp_path / "epoch_2020.tif"
    t2_path = tmp_path / "epoch_2025.tif"

    arr_t1 = np.zeros((4, 16, 16), dtype=np.float32)
    arr_t1[:, :, :] = np.array([0.04, 0.06, 0.04, 0.45], dtype=np.float32)[:, None, None]
    arr_t1[:, 0, 0] = np.nan

    arr_t2 = arr_t1.copy()
    # Bottom half undergoes major spectral change (forest -> bare/cleared)
    arr_t2[:, 8:, :] = np.array([0.22, 0.25, 0.28, 0.14], dtype=np.float32)[:, None, None]

    for path, arr in [(t1_path, arr_t1), (t2_path, arr_t2)]:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=16,
            width=16,
            count=4,
            dtype="float32",
            crs="EPSG:32751",
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(arr)

    return t1_path, t2_path


def test_compute_embedding_stability_mask_and_transition_integration(tmp_path):
    """Bi-temporal cosine distance generates stability masks compatible with consistency rules."""
    t1_path, t2_path = _create_bitemporal_rasters(tmp_path)
    backbone = SimpleCPUBackbone(embed_dim=8, seed=17)

    dist_path = tmp_path / "cosine_dist.tif"
    mask_path = tmp_path / "stable_mask.tif"
    compute_embedding_stability(
        t1_path,
        t2_path,
        dist_path,
        mask_path,
        backbone=backbone,
        tau_stable=0.15,
        chip_size=8,
        stride=4,
        block_size=10,
    )

    with rasterio.open(dist_path) as d_src, rasterio.open(mask_path) as m_src:
        dist = d_src.read(1)
        mask = m_src.read(1)
        assert d_src.dtypes == ("float32",)
        assert m_src.dtypes == ("uint8",)
        assert np.isnan(dist[0, 0])
        assert mask[0, 0] == 0
        # Top undisturbed region (rows 2..5) should have low cosine distance and mask=1
        assert np.all(dist[2:6, 2:14] <= 0.15)
        assert np.all(mask[2:6, 2:14] == 1)
        # Bottom disturbed region (rows 11..14) should exceed threshold and have mask=0
        assert np.all(dist[11:15, 2:14] > 0.15)
        assert np.all(mask[11:15, 2:14] == 0)

    # Filter training samples: one stable point in top half, one disturbed point in bottom half
    samples_path = tmp_path / "points.geojson"
    gpd.GeoDataFrame(
        {"class_id": [1, 1]},
        geometry=[
            Point(780045.0, 9049965.0),  # row 3, col 4 (stable)
            Point(780045.0, 9049865.0),  # row 13, col 4 (disturbed)
        ],
        crs="EPSG:32751",
    ).to_file(samples_path, driver="GeoJSON")

    clean_gdf, flagged_gdf = filter_disturbed_samples(
        samples_path,
        dist_path,
        max_change_distance=0.15,
    )
    assert len(flagged_gdf) == 2
    assert list(flagged_gdf["is_disturbed"]) == [False, True]
    assert len(clean_gdf) == 1

    # Verify stable_mask integrates directly with apply_transition_consistency
    lc_t1_path = tmp_path / "lc_2020.tif"
    lc_t2_path = tmp_path / "lc_2025.tif"
    lc_out_path = tmp_path / "lc_2025_consistent.tif"
    transform = from_origin(780000.0, 9050000.0, 10.0, 10.0)
    lc_t1 = np.full((1, 16, 16), 2, dtype=np.uint8)
    lc_t2 = np.full((1, 16, 16), 13, dtype=np.uint8)
    for p, arr in [(lc_t1_path, lc_t1), (lc_t2_path, lc_t2)]:
        with rasterio.open(
            p,
            "w",
            driver="GTiff",
            height=16,
            width=16,
            count=1,
            dtype="uint8",
            crs="EPSG:32751",
            transform=transform,
            nodata=0,
        ) as dst:
            dst.write(arr)

    apply_transition_consistency(
        lc_t1_path,
        lc_t2_path,
        lc_out_path,
        forbidden=[(2, 13)],
        correction_target="t2",
        stable_mask_path=mask_path,
        block_size=16,
    )
    with rasterio.open(lc_out_path) as out_src:
        corrected = out_src.read(1)
        # Top stable region (mask=1) was corrected back to class 2
        assert np.all(corrected[2:6, 2:14] == 2)
        # Bottom disturbed region (mask=0) was allowed to transition to class 13
        assert np.all(corrected[11:15, 2:14] == 13)
