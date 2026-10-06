"""Raster-based minimum mapping unit filtering for land-cover classifications."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from os import PathLike
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

RasterPath = str | PathLike[str]

MMU_TABLE: dict[int, int] = {
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

_EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)
_MODE_NODATA = 256


def _find_small_patches(
    labels: np.ndarray,
    valid: np.ndarray,
    table: Mapping[int, int],
) -> np.ndarray:
    """Return valid pixels in same-class 8-connected patches below threshold."""
    small = np.zeros(labels.shape, dtype=bool)
    for class_id, min_size in table.items():
        if min_size == 0:
            continue

        components, component_count = ndimage.label(
            valid & (labels == class_id), structure=_EIGHT_CONNECTED
        )
        if component_count == 0:
            continue

        sizes = np.bincount(components.ravel(), minlength=component_count + 1)
        small |= (components != 0) & (sizes[components] < min_size)

    return small


def _focal_mode_fill(
    labels: np.ndarray,
    valid: np.ndarray,
    small: np.ndarray,
    iterations: int,
) -> np.ndarray:
    """Fill small-patch pixels by iterated 3x3 square focal mode."""
    active = valid & ~small
    current = np.full(labels.shape, _MODE_NODATA, dtype=np.uint16)
    current[active] = labels[active]

    for _ in range(iterations):
        modes = np.full(labels.shape, _MODE_NODATA, dtype=np.uint16)
        best_counts = np.zeros(labels.shape, dtype=np.uint8)
        for class_id in np.unique(current[active]):
            class_pixels = (current == class_id) & active
            counts = ndimage.convolve(
                class_pixels.astype(np.uint8),
                _EIGHT_CONNECTED,
                mode="constant",
                cval=0,
            )
            wins = counts > best_counts
            modes[wins] = class_id
            best_counts[wins] = counts[wins]

        can_fill = valid & small & (best_counts > 0)
        update = active | can_fill
        current[update] = modes[update]
        active |= can_fill

    result = labels.copy()
    filled = valid & small & active
    result[filled] = current[filled].astype(np.uint8)
    return result


def apply_mmu(
    raster_path: RasterPath,
    output_path: RasterPath,
    table: Mapping[int, int] | None = None,
    iterations: int = 6,
) -> Path:
    """Filter undersized classification patches and fill them with focal mode.

    ``table`` maps class IDs to minimum patch sizes in pixels; a threshold of
    zero preserves that class. Components use 8-connectivity. Small patches
    are filled from a 3x3 square neighborhood for ``iterations`` passes. The
    output is a single-band uint8 GeoTIFF with the source spatial metadata.
    Source nodata pixels remain nodata.

    The full single-band raster is processed in memory so connected patches
    are measured consistently across the image.
    """
    if (
        not isinstance(iterations, Integral)
        or isinstance(iterations, bool)
        or iterations < 1
    ):
        raise ValueError("iterations must be at least 1")
    iterations = int(iterations)

    mmu_table = MMU_TABLE if table is None else table
    for class_id, min_size in mmu_table.items():
        if (
            not isinstance(class_id, Integral)
            or isinstance(class_id, bool)
            or not 0 <= class_id <= 255
        ):
            raise ValueError("table class IDs must be integers from 0 to 255")
        if not isinstance(min_size, Integral) or isinstance(min_size, bool) or min_size < 0:
            raise ValueError("table minimum sizes must be non-negative integers")

    raster_path = Path(raster_path)
    output_path = Path(output_path)
    if raster_path.resolve() == output_path.resolve():
        raise ValueError("output_path must differ from raster_path")

    with rasterio.open(raster_path) as source:
        if source.count != 1:
            raise ValueError("The classification raster must have exactly one band")

        source_data = source.read(1, masked=True)
        raw_labels = np.asarray(np.ma.getdata(source_data))
        valid = ~np.ma.getmaskarray(source_data)
        if np.issubdtype(raw_labels.dtype, np.floating):
            valid &= np.isfinite(raw_labels)

        valid_labels = raw_labels[valid]
        if not np.equal(valid_labels, np.floor(valid_labels)).all():
            raise ValueError("Valid classification pixels must contain integer class IDs")
        if ((valid_labels < 0) | (valid_labels > 255)).any():
            raise ValueError("Valid classification class IDs must be from 0 to 255")

        labels = np.zeros(raw_labels.shape, dtype=np.uint8)
        labels[valid] = valid_labels.astype(np.uint8)
        small = _find_small_patches(labels, valid, mmu_table)
        result = _focal_mode_fill(labels, valid, small, iterations)

        profile = source.profile.copy()
        profile.update(count=1, dtype="uint8", compress="deflate")
        if (
            source.nodata is not None
            and np.isfinite(source.nodata)
            and 0 <= source.nodata <= 255
            and float(source.nodata).is_integer()
        ):
            profile.update(nodata=source.nodata)
        else:
            profile.update(nodata=None)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write(result, 1)
            destination.set_band_description(1, "classification")
            destination.write_mask(valid.astype(np.uint8) * 255)

    return output_path
