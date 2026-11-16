"""Tests for sparse-polygon chip extraction, LoRA fine-tuning, and checkpoint bundles."""

import json
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from ML_LC_Classifier.rsfm import (
    RSFMChipDataset,
    SimpleCPUBackbone,
    finetune_rsfm,
    load_rsfm_checkpoint,
    prepare_rsfm_chips,
)


def _write_utm_raster_and_wgs84_samples(tmp_path):
    raster_path = tmp_path / "timor_utm.tif"
    height, width, bands = 24, 24, 4
    # UTM Zone 51S origin (meters)
    transform = from_origin(780000.0, 9050000.0, 10.0, 10.0)
    data = np.zeros((bands, height, width), dtype=np.float32)
    # Top half: Class 1 spectral signature; bottom half: Class 2 spectral signature
    data[:, :12, :] = np.array([0.04, 0.06, 0.04, 0.14], dtype=np.float32)[:, None, None]
    data[:, 12:, :] = np.array([0.05, 0.09, 0.03, 0.48], dtype=np.float32)[:, None, None]

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
        crs="EPSG:32751",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data)
        dst.descriptions = ("B02", "B03", "B04", "B08")

    # Polygons in EPSG:32751, then reprojected to EPSG:4326 to test CRS reprojection
    poly1 = box(780030.0, 9049920.0, 780070.0, 9049960.0)  # rows 4..8, cols 3..7 (Class 1)
    poly2 = box(780150.0, 9049800.0, 780190.0, 9049840.0)  # rows 16..20, cols 15..19 (Class 2)
    gdf_utm = gpd.GeoDataFrame({"class_id": [1, 2]}, geometry=[poly1, poly2], crs="EPSG:32751")
    gdf_wgs84 = gdf_utm.to_crs("EPSG:4326")

    training_path = tmp_path / "samples_wgs84.geojson"
    gdf_wgs84.to_file(training_path, driver="GeoJSON")
    return raster_path, training_path


def test_prepare_rsfm_chips_reprojects_and_preserves_ignore_index(tmp_path):
    """Chip extractor reprojects vectors and marks unlabeled pixels with ignore_index=0."""
    raster_path, training_path = _write_utm_raster_and_wgs84_samples(tmp_path)

    dataset = prepare_rsfm_chips(
        raster_path,
        training_path,
        class_field="class_id",
        chip_size=8,
    )

    assert isinstance(dataset, RSFMChipDataset)
    assert dataset.chips.shape == (2, 4, 8, 8)
    assert dataset.masks.shape == (2, 8, 8)
    assert dataset.class_values == (1, 2)
    assert dataset.band_names == ("B02", "B03", "B04", "B08")
    # Unlabeled background pixels in each 8x8 chip must be 0 (ignore_index)
    assert (dataset.masks == 0).any()
    assert set(np.unique(dataset.masks)).issubset({0, 1, 2})
    assert 1 in dataset.masks and 2 in dataset.masks
    assert dataset.mean.shape == (4,)
    assert dataset.std.shape == (4,)


def test_finetune_rsfm_with_lora_updates_adapters_and_roundtrips_checkpoint(tmp_path):
    """LoRA fine-tuning reduces masked loss, keeps base weights frozen, and round-trips."""
    raster_path, training_path = _write_utm_raster_and_wgs84_samples(tmp_path)
    dataset = prepare_rsfm_chips(
        raster_path,
        training_path,
        class_field="class_id",
        chip_size=8,
        wavelengths_um=[0.49, 0.56, 0.66, 0.84],
    )

    backbone = SimpleCPUBackbone(embed_dim=8, seed=11)
    base_before = backbone.extract_features(dataset.chips, wavelengths_um=dataset.wavelengths_um)

    result = finetune_rsfm(
        dataset,
        backbone=backbone,
        lora_rank=2,
        lora_alpha=4.0,
        epochs=30,
        lr=0.1,
    )

    # Loss decreased and labeled pixels are classified accurately
    assert result.loss_history[-1] < result.loss_history[0]
    assert result.bundle.lora_rank == 2
    assert result.bundle.class_values == (1, 2)

    # Base backbone remains frozen / unmodified
    base_after = backbone.extract_features(dataset.chips, wavelengths_um=dataset.wavelengths_um)
    np.testing.assert_array_equal(base_before, base_after)

    # Probabilities shape (N, K, H, W) and sum to 1 across classes
    probs = result.bundle.predict_proba(dataset.chips)
    assert probs.shape == (2, 2, 8, 8)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Check labeled pixel accuracy > 90%
    preds_idx = probs.argmax(axis=1)  # 0 or 1
    labeled = dataset.masks != 0
    expected_idx = dataset.masks[labeled] - 1
    assert (preds_idx[labeled] == expected_idx).mean() > 0.9

    # Save and reload checkpoint bundle
    ckpt_dir = tmp_path / "lora_ckpt"
    result.bundle.save(ckpt_dir)
    assert (ckpt_dir / "manifest.json").exists()
    manifest = json.loads((ckpt_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["class_values"] == [1, 2]
    assert manifest["band_names"] == ["B02", "B03", "B04", "B08"]

    reloaded = load_rsfm_checkpoint(ckpt_dir, backbone=backbone)
    np.testing.assert_allclose(
        reloaded.predict_embeddings(dataset.chips),
        result.bundle.predict_embeddings(dataset.chips),
        atol=1e-6,
    )
    np.testing.assert_allclose(reloaded.predict_proba(dataset.chips), probs, atol=1e-6)
