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
        ) # type: ignore
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


# ---------------------------------------------------------------------------
# Transition consistency
# ---------------------------------------------------------------------------

TransitionRule = tuple[int, int]  # (from_class, to_class)


def _load_single_band_labels(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a single-band classification raster.

    Returns
    -------
    labels : np.ndarray of uint8, shape (rows, cols)
        Class IDs; nodata pixels are set to 0.
    valid : np.ndarray of bool, shape (rows, cols)
        True where the pixel has a real class value.
    profile : dict
        Rasterio profile of the source file.
    """
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(
                f"{path}: classification raster must have exactly one band"
            )
        data = src.read(1, masked=True)
        raw = np.asarray(np.ma.getdata(data))
        valid = ~np.ma.getmaskarray(data)
        if np.issubdtype(raw.dtype, np.floating):
            valid &= np.isfinite(raw)

        valid_vals = raw[valid]
        if not np.equal(valid_vals, np.floor(valid_vals)).all():
            raise ValueError(
                f"{path}: valid pixels must contain integer class IDs"
            )
        if ((valid_vals < 0) | (valid_vals > 255)).any():
            raise ValueError(
                f"{path}: class IDs must be in the range 0–255"
            )

        labels = np.zeros(raw.shape, dtype=np.uint8)
        labels[valid] = valid_vals.astype(np.uint8)
        profile = src.profile.copy()

    return labels, valid, profile


def apply_transition_consistency(
    t1_path: RasterPath,
    t2_path: RasterPath,
    output_path: RasterPath,
    forbidden: list[TransitionRule],
    *,
    flag_path: RasterPath | None = None,
    stable_mask_path: RasterPath | None = None,
) -> Path:
    """Correct impossible land-cover transitions between two classification dates.

    For every pixel where the (t1 class → t2 class) pair appears in
    ``forbidden``, the t1 label is replaced with the t2 label in the output
    (i.e. the later date wins, same logic as the GEE version).  All other
    pixels are written unchanged from t1.

    Parameters
    ----------
    t1_path : str or PathLike
        Single-band uint8 GeoTIFF — earlier date classification.
    t2_path : str or PathLike
        Single-band uint8 GeoTIFF — later date classification.
        Must share the same grid (rows, cols, CRS, transform) as *t1_path*.
    output_path : str or PathLike
        Destination GeoTIFF (corrected labels, uint8, deflate).
    forbidden : list of (int, int)
        Transition pairs ``(from_class, to_class)`` that are considered
        impossible.  Any pixel matching one of these pairs will be overridden.
    flag_path : str or PathLike or None
        If given, write a second single-band uint8 GeoTIFF marking corrected
        pixels with 1 and unchanged pixels with 0.
    stable_mask_path : str or PathLike or None
        Optional single-band raster.  When provided, only pixels where the
        stable mask is non-zero (and non-nodata) are eligible for correction.
        Pixels outside the stable mask are written from t1 unchanged.

    Returns
    -------
    Path
        Resolved path of the written *output_path*.

    Raises
    ------
    ValueError
        If either raster has more than one band, contains non-integer valid
        pixels, class IDs outside 0–255, or if the grids do not match.
    """
    if not forbidden:
        raise ValueError("forbidden must contain at least one transition rule")

    for rule in forbidden:
        if (
            not isinstance(rule, (tuple, list))
            or len(rule) != 2
            or not all(isinstance(v, Integral) and not isinstance(v, bool) for v in rule)
            or not all(0 <= v <= 255 for v in rule)
        ):
            raise ValueError(
                "Each forbidden rule must be a (int, int) pair with values 0–255"
            )

    t1_path = Path(t1_path)
    t2_path = Path(t2_path)
    output_path = Path(output_path)

    t1_labels, t1_valid, profile = _load_single_band_labels(t1_path)
    t2_labels, t2_valid, profile2 = _load_single_band_labels(t2_path)

    if t1_labels.shape != t2_labels.shape:
        raise ValueError(
            f"Grid mismatch: t1 {t1_labels.shape} vs t2 {t2_labels.shape}. "
            "Both rasters must share the same rows × cols."
        )

    # Pixels valid in both dates are the only ones we can evaluate.
    both_valid = t1_valid & t2_valid

    # Optional stable mask — restrict corrections to stable areas.
    if stable_mask_path is not None:
        sm_labels, sm_valid, _ = _load_single_band_labels(Path(stable_mask_path))
        stable = sm_valid & (sm_labels != 0)
    else:
        stable = np.ones(t1_labels.shape, dtype=bool)

    # Build a flag array: True where the transition is forbidden.
    # Encode pairs as  from_class * 256 + to_class  (fits in uint16).
    pair = t1_labels.astype(np.uint32) * 256 + t2_labels.astype(np.uint32)

    flag = np.zeros(t1_labels.shape, dtype=bool)
    for from_cls, to_cls in forbidden:
        code = int(from_cls) * 256 + int(to_cls)
        flag |= pair == code

    # Only flag pixels that are valid in both dates AND inside stable mask.
    flag &= both_valid & stable

    # Apply correction: replace t1 with t2 where flagged.
    result = t1_labels.copy()
    result[flag] = t2_labels[flag]

    # Output valid mask: any pixel valid in t1 (we preserve nodata from t1).
    out_valid = t1_valid

    # Write corrected classification raster.
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", compress="deflate")
    if (
        profile.get("nodata") is not None
        and np.isfinite(profile["nodata"])
        and 0 <= profile["nodata"] <= 255
        and float(profile["nodata"]).is_integer()
    ):
        out_profile.update(nodata=profile["nodata"])
    else:
        out_profile.update(nodata=None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(result, 1)
        dst.set_band_description(1, "classification_corrected")
        dst.write_mask(out_valid.astype(np.uint8) * 255)

    # Optionally write the flag raster.
    if flag_path is not None:
        flag_out = Path(flag_path)
        flag_profile = profile.copy()
        flag_profile.update(count=1, dtype="uint8", compress="deflate", nodata=None)
        flag_out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(flag_out, "w", **flag_profile) as dst:
            dst.write(flag.astype(np.uint8), 1)
            dst.set_band_description(1, "transition_flag")
            dst.write_mask(out_valid.astype(np.uint8) * 255)

    return output_path.resolve()
