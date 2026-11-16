"""Sparse-polygon chip extraction for RSFM fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Sequence
import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window


PathLikeInput = str | PathLike[str]


@dataclass(frozen=True)
class RSFMChipDataset:
    """Extracted multi-band chips and sparse-label masks for RSFM fine-tuning.

    Attributes
    ----------
    chips : numpy.ndarray
        Float32 array of shape ``(N, C, chip_size, chip_size)``.
    masks : numpy.ndarray
        Int64 array of shape ``(N, chip_size, chip_size)`` where unlabeled
        pixels are ``0`` (``ignore_index=0``) and labeled pixels hold integer
        class IDs.
    class_values : tuple of int
        Sorted unique non-zero class IDs present in ``masks``.
    band_names : tuple of str
        Band names in channel order.
    wavelengths_um : tuple of float or None
        Optional central wavelengths in micrometers per channel.
    mean : numpy.ndarray
        Per-channel mean over valid chip pixels, shape ``(C,)``.
    std : numpy.ndarray
        Per-channel standard deviation (clamped to >= 1e-6), shape ``(C,)``.
    """

    chips: NDArray[np.float32]
    masks: NDArray[np.int64]
    class_values: tuple[int, ...]
    band_names: tuple[str, ...]
    wavelengths_um: tuple[float, ...] | None
    mean: NDArray[np.float32]
    std: NDArray[np.float32]


def prepare_rsfm_chips(
    raster_path: PathLikeInput,
    training_path: PathLikeInput,
    *,
    class_field: str = "class_id",
    chip_size: int = 224,
    ignore_index: int = 0,
    wavelengths_um: Sequence[float] | None = None,
) -> RSFMChipDataset:
    """Extract fixed-size chips centered on vector samples with sparse masks.

    Polygons or points from ``training_path`` are reprojected to the raster CRS
    when necessary. Each sample geometry generates a ``chip_size x chip_size``
    window, and all geometries intersecting that window are rasterized into a
    2D mask with ``ignore_index`` (default ``0``) for unlabeled background.
    """
    if chip_size < 1:
        raise ValueError("chip_size must be at least 1")

    gdf = gpd.read_file(training_path)
    if class_field not in gdf.columns:
        raise ValueError(f"Column '{class_field}' not found in {training_path}")
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError("Training vector file contains no valid geometries")

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        if gdf.crs is not None and gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        if wavelengths_um is not None and len(wavelengths_um) != src.count:
            raise ValueError(
                f"wavelengths_um length ({len(wavelengths_um)}) must match raster bands ({src.count})"
            )

        band_names = tuple(
            desc if desc else f"band_{idx + 1}"
            for idx, desc in enumerate(src.descriptions or [None] * src.count)
        )

        # Normalize class IDs to positive integers matching source labels
        gdf[class_field] = gdf[class_field].astype(int)
        if (gdf[class_field] == ignore_index).any():
            raise ValueError(
                f"Training labels must not equal ignore_index ({ignore_index})"
            )

        half = chip_size // 2
        chips_list: list[NDArray[np.float32]] = []
        masks_list: list[NDArray[np.int64]] = []
        valid_pixels_per_band: list[NDArray[np.float32]] = []

        for _, row in gdf.iterrows():
            centroid = row.geometry.centroid
            center_r, center_c = src.index(centroid.x, centroid.y)

            r0 = max(0, min(src.height - chip_size, center_r - half))
            c0 = max(0, min(src.width - chip_size, center_c - half))
            read_h = min(chip_size, src.height - r0)
            read_w = min(chip_size, src.width - c0)

            win = Window(c0, r0, read_w, read_h)
            win_transform = src.window_transform(win)
            block = src.read(window=win, masked=True)

            raw = np.ma.getdata(block).astype(np.float32, copy=True)
            masked = np.ma.getmaskarray(block)
            invalid = masked.any(axis=0) | (~np.isfinite(raw).all(axis=0))

            if (~invalid).any():
                valid_pixels_per_band.append(raw[:, ~invalid])
            if invalid.any():
                raw[:, invalid] = 0.0

            shapes = [
                (geom, int(cls_val))
                for geom, cls_val in zip(gdf.geometry, gdf[class_field])
            ]
            mask = rasterize(
                shapes=shapes,
                out_shape=(read_h, read_w),
                transform=win_transform,
                fill=ignore_index,
                dtype="int64",
            )
            if invalid.any():
                mask[invalid] = ignore_index

            if read_h < chip_size or read_w < chip_size:
                raw = np.pad(
                    raw,
                    ((0, 0), (0, chip_size - read_h), (0, chip_size - read_w)),
                    mode="edge",
                )
                mask = np.pad(
                    mask,
                    ((0, chip_size - read_h), (0, chip_size - read_w)),
                    mode="constant",
                    constant_values=ignore_index,
                )

            if (mask != ignore_index).any():
                chips_list.append(raw)
                masks_list.append(mask)

    if not chips_list:
        raise ValueError("No valid labeled pixels fell inside extracted chips")

    chips_arr = np.stack(chips_list, axis=0).astype(np.float32, copy=False)
    masks_arr = np.stack(masks_list, axis=0).astype(np.int64, copy=False)

    all_valid = np.concatenate(valid_pixels_per_band, axis=1)
    mean = np.mean(all_valid, axis=1).astype(np.float32)
    std = np.maximum(np.std(all_valid, axis=1), np.float32(1e-6)).astype(np.float32)

    unique_classes = tuple(
        sorted(int(v) for v in np.unique(masks_arr) if int(v) != ignore_index)
    )

    return RSFMChipDataset(
        chips=chips_arr,
        masks=masks_arr,
        class_values=unique_classes,
        band_names=band_names,
        wavelengths_um=tuple(float(w) for w in wavelengths_um) if wavelengths_um is not None else None,
        mean=mean,
        std=std,
    )
