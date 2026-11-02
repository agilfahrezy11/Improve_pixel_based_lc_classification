import numpy as np
import rasterio
from rasterio.transform import from_origin
import pytest

from ML_LC_Classifier.hybrid_assembly import (
    MaskRule,
    ProcessTileResult,
    process_tile,
    refine_lulc_with_mask_layer,
)


def test_process_tile_synthetic(tmp_path):
    transform = from_origin(500000, 9000000, 10, 10)
    w, h = 8, 8

    # 1. Multi-band probability raster (2 classes: 1 and 2), scaled uint16 [0, 10000]
    prob_tif = tmp_path / "probs.tif"
    p_b1 = np.full((h, w), 8000, dtype=np.uint16)
    p_b2 = np.full((h, w), 2000, dtype=np.uint16)

    prob_profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "nodata": 0,
        "width": w,
        "height": h,
        "count": 2,
        "crs": "EPSG:32751",
        "transform": transform,
    }
    with rasterio.open(prob_tif, "w", **prob_profile) as dst:
        dst.write(p_b1, 1)
        dst.write(p_b2, 2)
        dst.set_band_description(1, "Class_1_prob")
        dst.set_band_description(2, "Class_2_prob")

    # 2. Superpixel segmentation raster (1 segment ID across all pixels)
    seg_tif = tmp_path / "segments.tif"
    seg_data = np.ones((h, w), dtype=np.uint32)
    seg_profile = {
        "driver": "GTiff",
        "dtype": "uint32",
        "nodata": 0,
        "width": w,
        "height": h,
        "count": 1,
        "crs": "EPSG:32751",
        "transform": transform,
    }
    with rasterio.open(seg_tif, "w", **seg_profile) as dst:
        dst.write(seg_data, 1)

    out_dir = tmp_path / "output"
    res = process_tile(
        prob_path=prob_tif,
        seg_path=seg_tif,
        out_dir=out_dir,
        block_size=4,
        overlap=1,
    )

    assert isinstance(res, ProcessTileResult)
    assert res.lulc.exists()
    assert res.stability.exists()
    assert res.ci.exists()

    # Dictionary access compatibility
    assert res["lulc"] == res.lulc

    with rasterio.open(res.lulc) as src:
        lulc_data = src.read(1)
        # Class 1 probability (8000) > Class 2 probability (2000), so Winner should be Class 1
        assert (lulc_data == 1).all()


def test_refine_lulc_with_mask_layer(tmp_path):
    transform = from_origin(500000, 9000000, 10, 10)
    w, h = 4, 4

    lulc_tif = tmp_path / "lulc.tif"
    # All pixels classified as Class 1
    data = np.full((h, w), 1, dtype=np.uint8)

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0,
        "width": w,
        "height": h,
        "count": 1,
        "crs": "EPSG:32751",
        "transform": transform,
    }
    with rasterio.open(lulc_tif, "w", **profile) as dst:
        dst.write(data, 1)

    # Auxiliary mask where (row 0) == 1
    mask_tif = tmp_path / "corridor_mask.tif"
    mask_data = np.zeros((h, w), dtype=np.uint8)
    mask_data[0, :] = 1
    with rasterio.open(mask_tif, "w", **profile) as dst:
        dst.write(mask_data, 1)

    # Rule: If from Class 1 and corridor_mask == 1, reassign to Class 9
    rule = MaskRule(
        from_classes=[1],
        to_class=9,
        require_mask_path=mask_tif,
        require_value=1,
        name="Corridor Reclassification",
    )

    out_tif = tmp_path / "lulc_refined.tif"
    refine_lulc_with_mask_layer(
        input_lulc_path=lulc_tif,
        output_lulc_path=out_tif,
        rules=[rule],
        block_size=2,
    )

    assert out_tif.exists()
    with rasterio.open(out_tif) as src:
        refined = src.read(1)
        # Row 0 should be reassigned to Class 9
        assert (refined[0, :] == 9).all()
        # Other rows remain Class 1
        assert (refined[1:, :] == 1).all()
