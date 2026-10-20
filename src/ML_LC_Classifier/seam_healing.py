"""Post-processing utility to heal unclassified mosaic seam lines and sub-pixel tile gaps.

Removes 1- to 4-pixel vertical or horizontal nodata/zero lines caused by
bounding-box snapping, tile boundary misalignment, or clipping gaps during
mosaicking. Works on both single-band categorical maps and multi-band
probability rasters without altering external background nodata.
"""

from __future__ import annotations
from os import PathLike
from pathlib import Path
from typing import Sequence
import numpy as np
import rasterio
import rasterio.windows
from scipy.ndimage import distance_transform_edt

RasterPath = str | PathLike[str]


def heal_array_seams(
    data: np.ndarray,
    nodata: float | int = 0,
    max_seam_width: int = 4,
) -> np.ndarray:
    """Heal thin internal nodata seams in a 2D or 3D NumPy array.

    Uses Euclidean distance transform to identify pixels within ``max_seam_width``
    of valid data and copies the values from the nearest valid pixel.
    For 3D arrays (bands, rows, cols), the full feature/probability vector is
    copied from the nearest valid coordinate to preserve cross-band consistency
    and sum-to-one probability distributions.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (H, W) or (C, H, W).
    nodata : float or int
        The value representing invalid/unclassified pixels (default: 0).
    max_seam_width : int
        Maximum gap width in pixels to fill (default: 4). Gaps wider than this
        (e.g., broad ocean, external background) remain untouched.

    Returns
    -------
    np.ndarray
        Array of the same shape and dtype with thin seams filled.
    """
    is_3d = data.ndim == 3
    if is_3d:
        # Valid if any band has non-nodata and finite values
        if np.issubdtype(data.dtype, np.floating):
            valid_2d = np.any((data != nodata) & np.isfinite(data), axis=0)
        else:
            valid_2d = np.any(data != nodata, axis=0)
    else:
        if np.issubdtype(data.dtype, np.floating):
            valid_2d = (data != nodata) & np.isfinite(data)
        else:
            valid_2d = data != nodata

    if not np.any(valid_2d) or np.all(valid_2d):
        return data.copy()

    # Calculate distance to nearest valid pixel and coordinate indices
    distances, indices = distance_transform_edt(
        ~valid_2d,
        return_distances=True,
        return_indices=True,
    )

    # Only fill invalid pixels within max_seam_width of valid data
    fill_mask = (~valid_2d) & (distances <= max_seam_width)
    if not np.any(fill_mask):
        return data.copy()

    nearest_row = indices[0]
    nearest_col = indices[1]

    out = data.copy()
    if is_3d:
        # Vectorized copy across all bands using identical nearest coordinates
        for b in range(out.shape[0]):
            band_data = out[b]
            band_data[fill_mask] = band_data[nearest_row[fill_mask], nearest_col[fill_mask]]
            out[b] = band_data
    else:
        out[fill_mask] = out[nearest_row[fill_mask], nearest_col[fill_mask]]

    return out


def heal_raster_seams(
    input_path: RasterPath,
    output_path: RasterPath,
    *,
    nodata: float | int | None = 0,
    max_seam_width: int = 4,
    block_size: int = 2048,
    overlap: int = 32,
) -> Path:
    """Heal thin vertical/horizontal tile seams in a GeoTIFF.

    Supports both single-band discrete classification maps and multi-band
    probability GeoTIFFs. Streams large rasters in overlapping spatial windows
    to run safely within bounded memory.

    Parameters
    ----------
    input_path : str or PathLike
        Source GeoTIFF containing mosaic seams or tile gaps.
    output_path : str or PathLike
        Destination path for the healed GeoTIFF.
    nodata : float or int or None
        Nodata / unclassified value to fill along seams (default: 0).
    max_seam_width : int
        Maximum gap width in pixels to infill (default: 4).
    block_size : int
        Spatial window tile size in pixels (default: 2048).
    overlap : int
        Halo buffer around windows to eliminate boundary seams (default: 32).

    Returns
    -------
    Path
        Destination output Path object.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(in_path) as src:
        profile = src.profile.copy()
        target_nodata = nodata if nodata is not None else src.nodata
        if target_nodata is None:
            target_nodata = 0

        width, height = src.width, src.height
        band_count = src.count

        # For moderately sized rasters (<= 8000x8000), whole-array processing
        # is faster and completely avoids boundary edge conditions
        is_moderate_size = (width * height) <= (8000 * 8000)

        if is_moderate_size:
            data = src.read()  # (count, H, W)
            if band_count == 1:
                healed = heal_array_seams(data[0], nodata=target_nodata, max_seam_width=max_seam_width)
                healed = healed[np.newaxis, :, :]
            else:
                healed = heal_array_seams(data, nodata=target_nodata, max_seam_width=max_seam_width)

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(healed)
                for b in range(1, band_count + 1):
                    desc = src.descriptions[b - 1]
                    if desc:
                        dst.set_band_description(b, desc)
            return out_path

        # Streamed block processing for very large rasters
        with rasterio.open(out_path, "w", **profile) as dst:
            for b in range(1, band_count + 1):
                desc = src.descriptions[b - 1]
                if desc:
                    dst.set_band_description(b, desc)

            for y0 in range(0, height, block_size):
                y1 = min(y0 + block_size, height)
                for x0 in range(0, width, block_size):
                    x1 = min(x0 + block_size, width)

                    # Compute padded reading window
                    pad_top = min(y0, overlap)
                    pad_bottom = min(height - y1, overlap)
                    pad_left = min(x0, overlap)
                    pad_right = min(width - x1, overlap)

                    read_y0 = y0 - pad_top
                    read_y1 = y1 + pad_bottom
                    read_x0 = x0 - pad_left
                    read_x1 = x1 + pad_right

                    read_win = rasterio.windows.Window(
                        read_x0, read_y0, read_x1 - read_x0, read_y1 - read_y0
                    )
                    target_win = rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0)

                    block_data = src.read(window=read_win)

                    if band_count == 1:
                        healed_block = heal_array_seams(
                            block_data[0], nodata=target_nodata, max_seam_width=max_seam_width
                        )
                        # Crop interior
                        crop_y1 = (read_y1 - read_y0) - pad_bottom
                        crop_x1 = (read_x1 - read_x0) - pad_right
                        interior = healed_block[pad_top:crop_y1, pad_left:crop_x1]
                        dst.write(interior[np.newaxis, :, :], window=target_win)
                    else:
                        healed_block = heal_array_seams(
                            block_data, nodata=target_nodata, max_seam_width=max_seam_width
                        )
                        crop_y1 = (read_y1 - read_y0) - pad_bottom
                        crop_x1 = (read_x1 - read_x0) - pad_right
                        interior = healed_block[:, pad_top:crop_y1, pad_left:crop_x1]
                        dst.write(interior, window=target_win)

    return out_path


def heal_probability_array_spikes(
    data: np.ndarray,
    spike_threshold: float | None = None,
    major_spike_threshold: float | None = None,
    min_bands_agreement: int = 3,
) -> np.ndarray:
    """Detect and heal isolated 1-pixel computation tile seams in probability arrays.

    In Google Earth Engine, tile boundaries (typically every 256 pixels) can develop
    a 1-pixel computation seam where neighborhood operators evaluate near edge boundaries.
    This function detects isolated 1-pixel spikes or drops along the horizontal (and vertical)
    axes and replaces them with the linear average of their immediate neighbors,
    preserving probability vector sums and cross-band consistency.

    Parameters
    ----------
    data : np.ndarray
        Multi-band array of shape (C, H, W) or 2D array of shape (H, W).
    spike_threshold : float, optional
        Minimum probability difference per band (default: 100 for [0-10000] scale, 0.01 for [0-1]).
    major_spike_threshold : float, optional
        Threshold where a single band jump alone identifies a seam (default: 600 or 0.06).
    min_bands_agreement : int
        Number of bands that must simultaneously show a peak/valley to trigger (default: 3).

    Returns
    -------
    np.ndarray
        Array with isolated 1-pixel seams healed.
    """
    is_2d = data.ndim == 2
    data_3d = data[np.newaxis, :, :] if is_2d else data

    count, height, width = data_3d.shape
    if width < 3 and height < 3:
        return data.copy()

    # Adapt thresholds based on data range (0-1 float vs 0-10000 uint16)
    is_float_scale = np.issubdtype(data.dtype, np.floating) and (np.nanmax(data) <= 1.05)
    if spike_threshold is None:
        spike_threshold = 0.01 if is_float_scale else 100.0
    if major_spike_threshold is None:
        major_spike_threshold = 0.06 if is_float_scale else 600.0

    out = data_3d.copy().astype(np.float32)

    # 1. Horizontal pass (heals vertical column seam lines)
    if width >= 3:
        diff_left = out[:, :, 1:-1] - out[:, :, :-2]
        diff_right = out[:, :, 1:-1] - out[:, :, 2:]

        is_peak = (diff_left > spike_threshold) & (diff_right > spike_threshold)
        is_valley = (diff_left < -spike_threshold) & (diff_right < -spike_threshold)

        has_big_spike = np.any((diff_left > major_spike_threshold) & (diff_right > major_spike_threshold), axis=0)
        has_multi_spike = (np.sum(is_peak, axis=0) + np.sum(is_valley, axis=0)) >= min_bands_agreement
        seam_cols = has_big_spike | has_multi_spike

        if np.any(seam_cols):
            for b in range(count):
                band = out[b]
                band[:, 1:-1] = np.where(
                    seam_cols,
                    0.5 * (band[:, :-2] + band[:, 2:]),
                    band[:, 1:-1],
                )
                out[b] = band

    # 2. Vertical pass (heals horizontal row seam lines)
    if height >= 3:
        diff_top = out[:, 1:-1, :] - out[:, :-2, :]
        diff_bottom = out[:, 1:-1, :] - out[:, 2:, :]

        is_peak_v = (diff_top > spike_threshold) & (diff_bottom > spike_threshold)
        is_valley_v = (diff_top < -spike_threshold) & (diff_bottom < -spike_threshold)

        has_big_spike_v = np.any((diff_top > major_spike_threshold) & (diff_bottom > major_spike_threshold), axis=0)
        has_multi_spike_v = (np.sum(is_peak_v, axis=0) + np.sum(is_valley_v, axis=0)) >= min_bands_agreement
        seam_rows = has_big_spike_v | has_multi_spike_v

        if np.any(seam_rows):
            for b in range(count):
                band = out[b]
                band[1:-1, :] = np.where(
                    seam_rows,
                    0.5 * (band[:-2, :] + band[2:, :]),
                    band[1:-1, :],
                )
                out[b] = band

    result = np.round(out).astype(data.dtype) if np.issubdtype(data.dtype, np.integer) else out.astype(data.dtype)
    return result[0] if is_2d else result


def heal_probability_seams(
    input_path: RasterPath,
    output_path: RasterPath,
    *,
    spike_threshold: float | None = None,
    major_spike_threshold: float | None = None,
    min_bands_agreement: int = 3,
    block_size: int = 2048,
    overlap: int = 16,
) -> Path:
    """Heal 1-pixel computation tile lines in multi-band probability GeoTIFFs.

    Parameters
    ----------
    input_path : str or PathLike
        Source probability GeoTIFF containing GEE 256-pixel computation seams.
    output_path : str or PathLike
        Destination path for the seam-healed probability GeoTIFF.
    spike_threshold : float, optional
        Sensitivity threshold to detect isolated 1-pixel jumps (default 100 for uint16, 0.01 for float).
    major_spike_threshold : float, optional
        Threshold where a single band jump alone identifies a seam (default: 600 or 0.06).
    min_bands_agreement : int
        Number of bands that must simultaneously show a peak/valley to trigger (default: 3).
    block_size : int
        Window size for block streaming (default 2048).
    overlap : int
        Padding overlap around windows (default 16).
    """
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(in_path) as src:
        profile = src.profile.copy()
        width, height = src.width, src.height
        band_count = src.count

        is_moderate = (width * height) <= (8000 * 8000)
        if is_moderate:
            data = src.read()
            healed = heal_probability_array_spikes(
                data,
                spike_threshold=spike_threshold,
                major_spike_threshold=major_spike_threshold,
                min_bands_agreement=min_bands_agreement,
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(healed)
                for b in range(1, band_count + 1):
                    desc = src.descriptions[b - 1]
                    if desc:
                        dst.set_band_description(b, desc)
            return out_path

        # Streamed block processing
        with rasterio.open(out_path, "w", **profile) as dst:
            for b in range(1, band_count + 1):
                desc = src.descriptions[b - 1]
                if desc:
                    dst.set_band_description(b, desc)

            for y0 in range(0, height, block_size):
                y1 = min(y0 + block_size, height)
                pad_top = min(y0, overlap)
                pad_bottom = min(height - y1, overlap)
                read_y0 = y0 - pad_top
                read_y1 = y1 + pad_bottom

                for x0 in range(0, width, block_size):
                    x1 = min(x0 + block_size, width)
                    pad_left = min(x0, overlap)
                    pad_right = min(width - x1, overlap)
                    read_x0 = x0 - pad_left
                    read_x1 = x1 + pad_right

                    read_win = rasterio.windows.Window(
                        read_x0, read_y0, read_x1 - read_x0, read_y1 - read_y0
                    )
                    target_win = rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0)

                    block_data = src.read(window=read_win)
                    healed_block = heal_probability_array_spikes(
                        block_data,
                        spike_threshold=spike_threshold,
                        major_spike_threshold=major_spike_threshold,
                        min_bands_agreement=min_bands_agreement,
                    )

                    crop_y1 = (read_y1 - read_y0) - pad_bottom
                    crop_x1 = (read_x1 - read_x0) - pad_right
                    interior = healed_block[:, pad_top:crop_y1, pad_left:crop_x1]
                    dst.write(interior, window=target_win)

    return out_path

