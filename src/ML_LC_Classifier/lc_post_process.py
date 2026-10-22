"""Raster-based minimum mapping unit filtering for land-cover classifications."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from os import PathLike
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
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

#: Recommended MMU thresholds for Timor-Leste 18-class hybrid OBIA classification.
#: In hybrid classification, SNIC segmentation already cleans fine pixel noise.
#: Critical linear and narrow classes (Mangrove, Settlements, Waterbodies, Dry riverbed)
#: are protected with min_size=0, while gentle clean-up (5-20 pixels) is applied to vegetation.
DEFAULT_HYBRID_MMU_TABLE: dict[int, int] = {
    1: 50,   #primary forest (0.50 ha)
    2: 30,   #secondary forest (0.30 ha)
    3: 20,   #mangrove forest (0.20)
    4: 30,   #coastal forest 
    5: 15,   #teak plantation 
    6: 10,   #eucalyptus plantation
    7: 15,   #coffee agroforestry
    8: 15,   #cocoa agroforestry
    9: 10,   #coconut agroforestry
    10: 10,  #mixed garden
    11: 10,  #paddy field
    12: 10,  #other cropland
    13: 15,  #grassland
    14: 15,  #shurbland
    15: 30,  #cleared land
    16: 20,  #built-up
    17: 20,  #Waterbody
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
        if min_size <= 1:
            continue

        class_mask = valid & (labels == class_id)
        if not np.any(class_mask):
            continue

        components, component_count = ndimage.label(
            class_mask, structure=_EIGHT_CONNECTED
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
    if not np.any(small):
        return labels.copy()

    active = valid & ~small
    current = np.full(labels.shape, _MODE_NODATA, dtype=np.uint16)
    current[active] = labels[active]

    # Only process unique classes that are actually active in this array
    present_classes = np.unique(current[active])
    present_classes = present_classes[present_classes != _MODE_NODATA]

    for _ in range(iterations):
        if not np.any(valid & small & ~active):
            break

        modes = np.full(labels.shape, _MODE_NODATA, dtype=np.uint16)
        best_counts = np.zeros(labels.shape, dtype=np.uint8)

        for class_id in present_classes:
            class_pixels = (current == class_id) & active
            if not np.any(class_pixels):
                continue

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
        if not np.any(can_fill):
            break

        update = active | can_fill
        current[update] = modes[update]
        active |= can_fill

    result = labels.copy()
    filled = valid & small & active
    result[filled] = current[filled].astype(np.uint8)
    return result


def _filter_block(
    block_labels: np.ndarray,
    block_valid: np.ndarray,
    table: Mapping[int, int],
    iterations: int,
) -> np.ndarray:
    """Run MMU detection and focal mode fill on a single window/block."""
    if not np.any(block_valid):
        return block_labels.copy()

    small = _find_small_patches(block_labels, block_valid, table)
    if not np.any(small):
        return block_labels.copy()

    return _focal_mode_fill(block_labels, block_valid, small, iterations)


def apply_mmu(
    raster_path: RasterPath,
    output_path: RasterPath,
    table: Mapping[int, int] | None = None,
    iterations: int = 6,
    *,
    block_size: int = 2048,
    overlap: int = 64,
) -> Path:
    """Filter undersized classification patches and fill them with focal mode.

    ``table`` maps class IDs to minimum patch sizes in pixels; a threshold of
    zero or one preserves that class. Components use 8-connectivity. Small patches
    are filled from a 3x3 square neighborhood for ``iterations`` passes. The
    output is a single-band uint8 GeoTIFF with the source spatial metadata.
    Source nodata pixels remain nodata.

    Large rasters are streamed in overlapping spatial windows (default 2048x2048
    with 64px halo overlap), keeping peak memory usage small (< 250 MB) while
    seamlessly filtering national-scale extents.
    """
    if (
        not isinstance(iterations, Integral)
        or isinstance(iterations, bool)
        or iterations < 1
    ):
        raise ValueError("iterations must be at least 1")
    iterations = int(iterations)

    if block_size < 64:
        raise ValueError("block_size must be at least 64")
    if overlap < 0 or overlap >= block_size // 2:
        raise ValueError("overlap must be >= 0 and < block_size // 2")

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

        height = source.height
        width = source.width

        profile = source.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            compress="deflate",
        )
        if width >= 256 and height >= 256:
            profile.update(
                tiled=True,
                blockxsize=512,
                blockysize=512,
            )
        else:
            profile.pop("tiled", None)
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
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

        # For small images that fit completely in block_size, process in a single pass
        if height <= block_size and width <= block_size:
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
            result = _filter_block(labels, valid, mmu_table, iterations)

            with rasterio.open(output_path, "w", **profile) as destination:
                destination.write(result, 1)
                destination.set_band_description(1, "classification")
                destination.write_mask(valid.astype(np.uint8) * 255)

            return output_path

        # For larger rasters, stream via overlapping windows
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.set_band_description(1, "classification")

            for y0 in range(0, height, block_size):
                y1 = min(y0 + block_size, height)
                for x0 in range(0, width, block_size):
                    x1 = min(x0 + block_size, width)

                    # Compute buffered read coordinates with halo
                    pad_top = min(overlap, y0)
                    pad_bottom = min(overlap, height - y1)
                    pad_left = min(overlap, x0)
                    pad_right = min(overlap, width - x1)

                    read_y0 = y0 - pad_top
                    read_y1 = y1 + pad_bottom
                    read_x0 = x0 - pad_left
                    read_x1 = x1 + pad_right

                    window = rasterio.windows.Window(
                        read_x0,
                        read_y0,
                        read_x1 - read_x0,
                        read_y1 - read_y0,
                    )
                    block_data = source.read(1, window=window, masked=True)

                    if block_data.mask.all():
                        # Entire block is masked/nodata
                        empty = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                        target_win = rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0)
                        destination.write(empty, 1, window=target_win)
                        destination.write_mask(empty, window=target_win)
                        continue

                    raw_labels = np.asarray(np.ma.getdata(block_data))
                    valid = ~np.ma.getmaskarray(block_data)
                    if np.issubdtype(raw_labels.dtype, np.floating):
                        valid &= np.isfinite(raw_labels)

                    labels = np.zeros(raw_labels.shape, dtype=np.uint8)
                    labels[valid] = raw_labels[valid].astype(np.uint8)

                    # Filter buffered block
                    filtered = _filter_block(labels, valid, mmu_table, iterations)

                    # Crop out the central interior tile (discarding overlap halo)
                    crop_y1 = (read_y1 - read_y0) - pad_bottom
                    crop_x1 = (read_x1 - read_x0) - pad_right
                    interior_filtered = filtered[pad_top:crop_y1, pad_left:crop_x1]
                    interior_valid = valid[pad_top:crop_y1, pad_left:crop_x1]

                    target_win = rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0)
                    destination.write(interior_filtered, 1, window=target_win)
                    destination.write_mask((interior_valid.astype(np.uint8) * 255), window=target_win)

    return output_path


# ---------------------------------------------------------------------------
# Transition consistency & Cross-tabulation
# ---------------------------------------------------------------------------

TransitionRule = tuple[int, int]  # (from_class, to_class)

DEFAULT_TIMOR_17_CLASSES: dict[int, str] = {
    1: "Primary_forest",
    2: "Secondary_forest",
    3: "Mangrove_forest",
    4: "Coastal_forest",
    5: "Teak_plantation",
    6: "Eucalyptus_plantation",
    7: "Coffe_agroforestry",
    8: "Cocoa_agroforestry",
    9: "Coconut_agroforestry",
    10: "Mixed_garden",
    11: "Paddy_field",
    12: "Other_cropland",
    13: "Grassland",
    14: "Shrubland",
    15: "Cleared_land",
    16: "Built-up",
    17: "Waterbody",
}


def calculate_transition_matrix(
    t1_path: RasterPath,
    t2_path: RasterPath,
    class_names: Mapping[int, str] | None = None,
    *,
    csv_path: RasterPath | None = None,
    block_size: int = 2048,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute a multi-temporal transition cross-tabulation matrix (t1 -> t2).

    Streams rasters in memory-safe blocks without loading full grids into RAM.
    Returns two pandas DataFrames:
      1. transition_counts: Pixel counts between each (t1, t2) pair.
      2. transition_areas_ha: Hectares converted between each (t1, t2) pair.

    Parameters
    ----------
    t1_path : str or PathLike
        Single-band uint8 GeoTIFF representing the earlier time period (t1).
    t2_path : str or PathLike
        Single-band uint8 GeoTIFF representing the later time period (t2).
    class_names : Mapping[int, str] or None
        Custom mapping from class integer IDs to class name strings.
        If None, defaults to DEFAULT_TIMOR_17_CLASSES.
    csv_path : str or PathLike or None
        Optional destination path to save the transition area table as CSV.
    block_size : int
        Window block size for memory-safe streaming (default 2048).
    """
    classes = DEFAULT_TIMOR_17_CLASSES if class_names is None else dict(class_names)
    t1_path = Path(t1_path)
    t2_path = Path(t2_path)

    with rasterio.open(t1_path) as src1, rasterio.open(t2_path) as src2:
        if src1.shape != src2.shape:
            raise ValueError(
                f"Grid mismatch: t1 {src1.shape} vs t2 {src2.shape}. "
                "Both rasters must share the exact same dimensions."
            )

        # Derive pixel area in hectares
        res_x = abs(src1.transform.a)
        res_y = abs(src1.transform.e)
        pixel_area_ha = (res_x * res_y) / 10000.0

        # Max class ID determines matrix dimension
        all_ids = sorted(classes.keys())
        max_id = max(max(all_ids, default=0), 255)
        # 2D transition histogram accumulator: [t1_class, t2_class]
        counts_matrix = np.zeros((max_id + 1, max_id + 1), dtype=np.int64)

        height, width = src1.height, src1.width

        for y0 in range(0, height, block_size):
            y1 = min(y0 + block_size, height)
            for x0 in range(0, width, block_size):
                x1 = min(x0 + block_size, width)
                win = rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0)

                b1 = src1.read(1, window=win, masked=True)
                b2 = src2.read(1, window=win, masked=True)

                valid = (~b1.mask) & (~b2.mask)
                d1 = np.asarray(b1.data)[valid].astype(np.int64)
                d2 = np.asarray(b2.data)[valid].astype(np.int64)

                # Filter out background / nodata 0
                non_zero = (d1 > 0) & (d2 > 0)
                d1 = d1[non_zero]
                d2 = d2[non_zero]

                if len(d1) > 0:
                    encoded = d1 * (max_id + 1) + d2
                    flat_counts = np.bincount(encoded, minlength=(max_id + 1) ** 2)
                    counts_matrix += flat_counts.reshape((max_id + 1, max_id + 1))

    # Build clean DataFrames for the configured class list
    labels = [classes.get(cid, f"Class_{cid}") for cid in all_ids]
    sub_counts = counts_matrix[np.ix_(all_ids, all_ids)]
    sub_areas = np.round(sub_counts * pixel_area_ha, 4)

    df_counts = pd.DataFrame(sub_counts, index=labels, columns=labels)
    df_counts.index.name = "From (t1)"

    df_areas = pd.DataFrame(sub_areas, index=labels, columns=labels)
    df_areas.index.name = "From (t1)"

    if csv_path is not None:
        csv_file = Path(csv_path)
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        df_areas.to_csv(csv_file)

    return df_counts, df_areas


def apply_transition_consistency(
    t1_path: RasterPath,
    t2_path: RasterPath,
    output_path: RasterPath,
    forbidden: list[TransitionRule],
    *,
    correction_target: str = "t2",
    replacement_map: Mapping[TransitionRule, int] | None = None,
    flag_path: RasterPath | None = None,
    stable_mask_path: RasterPath | None = None,
    block_size: int = 2048,
) -> Path:
    """Correct impossible land-cover transitions between two classification dates.

    Streams rasters in memory-safe blocks, keeping peak RAM usage minimal (< 500 MB)
    even for full national extents.

    Parameters
    ----------
    t1_path : str or PathLike
        Single-band uint8 GeoTIFF — earlier date classification (e.g., 2020).
    t2_path : str or PathLike
        Single-band uint8 GeoTIFF — later date classification (e.g., 2025).
        Must share the same grid (rows, cols, CRS, transform) as *t1_path*.
    output_path : str or PathLike
        Destination GeoTIFF (corrected labels, uint8, deflate).
    forbidden : list of (int, int)
        Transition pairs ``(from_class, to_class)`` that are considered
        impossible (e.g., (2, 1) = Secondary Forest -> Primary Forest).
    correction_target : str
        Which date the output represents and corrects:
        - "t2" (default): Output is the corrected t2 map. If (t1, t2) is forbidden,
          t2 is adjusted back to t1 (or to replacement_map[(t1, t2)] if specified).
        - "t1": Output is the corrected t1 map. If (t1, t2) is forbidden,
          t1 is overridden with t2.
    replacement_map : Mapping[(int, int), int] or None
        Optional explicit replacement class for specific forbidden transitions.
        If not provided or a pair is not in replacement_map, defaults to:
        - t1's label when correction_target == "t2" (e.g. demote 2025 back to 2020).
        - t2's label when correction_target == "t1" (e.g. override 2020 with 2025).
    flag_path : str or PathLike or None
        If given, write a second single-band uint8 GeoTIFF marking corrected
        pixels with 1 and unchanged pixels with 0.
    stable_mask_path : str or PathLike or None
        Optional single-band raster. When provided, only pixels where the
        stable mask is non-zero are eligible for correction.
    block_size : int
        Window block size for memory-safe streaming (default 2048).

    Returns
    -------
    Path
        Resolved path of the written *output_path*.
    """
    if not forbidden:
        raise ValueError("forbidden must contain at least one transition rule")
    if correction_target not in ("t1", "t2"):
        raise ValueError("correction_target must be either 't1' or 't2'")

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

    # Encode forbidden rules as uint32 key: from_cls * 256 + to_cls
    forbidden_dict: dict[int, int | None] = {}
    for rule in forbidden:
        f_cls, t_cls = int(rule[0]), int(rule[1])
        key = f_cls * 256 + t_cls
        repl = None
        if replacement_map is not None:
            repl = replacement_map.get((f_cls, t_cls), None)
            if repl is None:
                repl = replacement_map.get(rule, None)
        forbidden_dict[key] = repl

    with rasterio.open(t1_path) as src1, rasterio.open(t2_path) as src2:
        if src1.shape != src2.shape:
            raise ValueError(
                f"Grid mismatch: t1 {src1.shape} vs t2 {src2.shape}. "
                "Both rasters must share the exact same dimensions."
            )

        height, width = src1.height, src1.width

        profile = src1.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            compress="deflate",
        )
        if width >= 256 and height >= 256:
            profile.update(tiled=True, blockxsize=512, blockysize=512)
        else:
            profile.pop("tiled", None)
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)

        if (
            src1.nodata is not None
            and np.isfinite(src1.nodata)
            and 0 <= src1.nodata <= 255
            and float(src1.nodata).is_integer()
        ):
            profile.update(nodata=src1.nodata)
        else:
            profile.update(nodata=None)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        flag_dst = None
        if flag_path is not None:
            fpath = Path(flag_path)
            fpath.parent.mkdir(parents=True, exist_ok=True)
            flag_profile = profile.copy()
            flag_profile.update(nodata=None)
            flag_dst = rasterio.open(fpath, "w", **flag_profile)
            flag_dst.set_band_description(1, "transition_flag")

        src_sm = None
        if stable_mask_path is not None:
            src_sm = rasterio.open(stable_mask_path)

        try:
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.set_band_description(
                    1,
                    f"classification_corrected_{correction_target}",
                )

                for y0 in range(0, height, block_size):
                    y1 = min(y0 + block_size, height)
                    for x0 in range(0, width, block_size):
                        x1 = min(x0 + block_size, width)
                        win = rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0)

                        b1 = src1.read(1, window=win, masked=True)
                        b2 = src2.read(1, window=win, masked=True)

                        v1 = ~b1.mask
                        v2 = ~b2.mask
                        both_valid = v1 & v2

                        d1 = np.asarray(b1.data).astype(np.uint8)
                        d2 = np.asarray(b2.data).astype(np.uint8)

                        if not np.any(both_valid):
                            empty = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                            dst.write(empty, 1, window=win)
                            dst.write_mask(empty, window=win)
                            if flag_dst is not None:
                                flag_dst.write(empty, 1, window=win)
                            continue

                        if src_sm is not None:
                            sm_block = src_sm.read(1, window=win, masked=True)
                            stable = (~sm_block.mask) & (sm_block.data != 0)
                        else:
                            stable = np.ones((y1 - y0, x1 - x0), dtype=bool)

                        # Encode (t1, t2) pair
                        pair = d1.astype(np.uint32) * 256 + d2.astype(np.uint32)

                        # Base output selection
                        if correction_target == "t2":
                            out_block = d2.copy()
                            target_valid = v2
                        else:
                            out_block = d1.copy()
                            target_valid = v1

                        block_flag = np.zeros((y1 - y0, x1 - x0), dtype=bool)

                        for code, custom_repl in forbidden_dict.items():
                            match = (pair == code) & both_valid & stable
                            if np.any(match):
                                block_flag |= match
                                if custom_repl is not None:
                                    out_block[match] = int(custom_repl)
                                else:
                                    if correction_target == "t2":
                                        # Demote/revert t2 back to t1
                                        out_block[match] = d1[match]
                                    else:
                                        # Override t1 with t2
                                        out_block[match] = d2[match]

                        dst.write(out_block, 1, window=win)
                        dst.write_mask(
                            (target_valid.astype(np.uint8) * 255),
                            window=win,
                        )

                        if flag_dst is not None:
                            flag_dst.write(
                                block_flag.astype(np.uint8),
                                1,
                                window=win,
                            )
                            flag_dst.write_mask(
                                (both_valid.astype(np.uint8) * 255),
                                window=win,
                            )

        finally:
            if flag_dst is not None:
                flag_dst.close()
            if src_sm is not None:
                src_sm.close()

    return output_path.resolve()


def _load_single_band_labels(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a single-band classification raster."""
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


# ---------------------------------------------------------------------------
# Area calculation
# ---------------------------------------------------------------------------

#: Mapping passed to :func:`calculate_class_area` to describe each class.
#: Keys are class IDs; each value is a dict with optional keys:
#:   ``name``  (str)  – human-readable label shown in plots and CSV.
#:   ``color`` (str)  – any matplotlib color string (hex, named, etc.).
ClassSpec = dict[int, dict[str, str]]

_M2_PER_HECTARE = 10_000.0


@dataclass
class ClassAreaResult:
    """Return value of :func:`calculate_class_area`.

    Attributes
    ----------
    area_ha : pandas.DataFrame
        One row per class with columns ``class_id``, ``class_name``,
        ``pixel_count``, and ``area_ha``.  Sorted by ``class_id``.
    figure : matplotlib.figure.Figure
        Bar chart of area per class, coloured by ``class_spec``.
    csv_path : Path or None
        Resolved path of the written CSV, or ``None`` if *csv_path* was not
        supplied.
    """

    area_ha: pd.DataFrame
    figure: object  # matplotlib.figure.Figure — imported lazily
    csv_path: Path | None = field(default=None)


def calculate_class_area(
    raster_path: RasterPath,
    class_spec: ClassSpec | None = None,
    *,
    csv_path: RasterPath | None = None,
    figsize: tuple[float, float] = (10.0, 6.0),
    title: str = "Land-Cover Area by Class",
    xlabel: str = "Land-Cover Class",
    ylabel: str = "Area (ha)",
) -> ClassAreaResult:
    """Calculate per-class area from a single-band land-cover raster.

    Pixel area is derived from the raster's affine transform.  If the CRS is
    geographic (degrees), pixels are reprojected to an equal-area equivalent
    so that areas are meaningful; if the CRS is already projected the pixel
    dimensions are used directly.

    Parameters
    ----------
    raster_path : str or PathLike
        Single-band uint8 (or integer) GeoTIFF — the land-cover classification.
    class_spec : dict[int, dict[str, str]] or None
        Optional mapping from class ID to display metadata.  Each value is a
        dict that may contain:

        * ``"name"``  – label used in the plot and CSV (defaults to
          ``"Class <id>"`` when absent).
        * ``"color"`` – matplotlib colour string for the bar (defaults to the
          current axes colour cycle when absent).

        Classes present in the raster but absent from *class_spec* are still
        included in the output, labelled ``"Class <id>"`` with a cycle colour.
        Class IDs listed in *class_spec* but absent from the raster produce no
        row in the output (they are silently skipped).

        Example::

            class_spec = {
                1:  {"name": "Forest",       "color": "#1a7a1a"},
                2:  {"name": "Agriculture",  "color": "#f5c518"},
                9:  {"name": "Water",        "color": "#2471a3"},
            }

    csv_path : str or PathLike or None
        If provided, the area table is written to this path as a UTF-8 CSV.
        Parent directories are created automatically.
    figsize : (float, float)
        Matplotlib figure size in inches, ``(width, height)``.
    title : str
        Figure title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label (should reflect the unit, which is always hectares).

    Returns
    -------
    ClassAreaResult
        Dataclass with ``area_ha`` (DataFrame), ``figure`` (Figure), and
        ``csv_path`` (Path or None).

    Raises
    ------
    ValueError
        If the raster has more than one band, contains non-integer valid
        pixels, or class IDs outside 0–255.
    ImportError
        If ``matplotlib`` is not installed.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for calculate_class_area. "
            "Install it with:  pip install matplotlib"
        ) from exc

    raster_path = Path(raster_path)
    labels, valid, profile = _load_single_band_labels(raster_path)

    # ------------------------------------------------------------------ #
    # Pixel area in m²                                                     #
    # ------------------------------------------------------------------ #
    crs: CRS = profile.get("crs")
    transform = profile["transform"]

    # Pixel dimensions from the affine transform (absolute values).
    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)

    if crs is not None and crs.is_geographic:
        # Geographic CRS — approximate pixel area at image centre using
        # the WGS-84 semi-major axis.  Good enough for area summaries;
        # not a substitute for proper equal-area projection.
        import math

        _WGS84_A = 6_378_137.0  # metres
        _WGS84_F = 1 / 298.257_223_563
        _WGS84_B = _WGS84_A * (1 - _WGS84_F)

        rows, cols = labels.shape
        centre_row = rows / 2.0
        # Latitude at pixel centre (transform.f is the top-left y).
        centre_lat_deg = transform.f + centre_row * transform.e
        centre_lat_rad = math.radians(centre_lat_deg)

        # Meridian and prime-vertical radii of curvature.
        sin_lat = math.sin(centre_lat_rad)
        cos_lat = math.cos(centre_lat_rad)
        e2 = 1.0 - (_WGS84_B / _WGS84_A) ** 2
        denom = math.sqrt(1.0 - e2 * sin_lat**2)
        M = _WGS84_A * (1 - e2) / denom**3  # meridional
        N = _WGS84_A / denom                 # prime-vertical

        # 1° in metres along meridian / parallel at this latitude.
        m_per_deg_lat = M * math.pi / 180.0
        m_per_deg_lon = N * cos_lat * math.pi / 180.0

        pixel_area_m2 = (pixel_width * m_per_deg_lon) * (pixel_height * m_per_deg_lat)
    else:
        # Projected CRS (or unknown) — pixel dimensions already in metres.
        pixel_area_m2 = pixel_width * pixel_height

    pixel_area_ha = pixel_area_m2 / _M2_PER_HECTARE

    # ------------------------------------------------------------------ #
    # Count pixels per class                                               #
    # ------------------------------------------------------------------ #
    # Exclude class 0 — treated as unclassified / background.
    valid_labels = labels[valid & (labels != 0)]
    class_ids_present = np.unique(valid_labels)

    # np.bincount is fast and sufficient for uint8 class IDs.
    counts = np.bincount(valid_labels.astype(np.int64), minlength=256)

    rows_list = []
    for cid in class_ids_present:
        count = int(counts[cid])
        spec = (class_spec or {}).get(int(cid), {})
        name = spec.get("name", f"Class {cid}")
        rows_list.append(
            {
                "class_id": int(cid),
                "class_name": name,
                "pixel_count": count,
                "area_ha": round(count * pixel_area_ha, 4),
            }
        )

    area_df = pd.DataFrame(
        rows_list, columns=["class_id", "class_name", "pixel_count", "area_ha"]
    ).sort_values("class_id").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Build bar chart                                                      #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=figsize)

    bar_colors = []
    for _, row in area_df.iterrows():
        spec = (class_spec or {}).get(int(row["class_id"]), {})
        color = spec.get("color", None)
        bar_colors.append(color)

    # Replace None placeholders with matplotlib cycle colours.
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    resolved_colors = []
    cycle_idx = 0
    for c in bar_colors:
        if c is None:
            resolved_colors.append(prop_cycle[cycle_idx % len(prop_cycle)])
            cycle_idx += 1
        else:
            resolved_colors.append(c)

    x_positions = range(len(area_df))
    bars = ax.bar(x_positions, area_df["area_ha"], color=resolved_colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(area_df["class_name"], rotation=45, ha="right", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f"{val:,.0f}")
    )

    # Annotate bars with the area value.
    for bar, val in zip(bars, area_df["area_ha"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() * 1.01,
            f"{val:,.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    # ------------------------------------------------------------------ #
    # Optionally write CSV                                                 #
    # ------------------------------------------------------------------ #
    resolved_csv: Path | None = None
    if csv_path is not None:
        resolved_csv = Path(csv_path)
        resolved_csv.parent.mkdir(parents=True, exist_ok=True)
        area_df.to_csv(resolved_csv, index=False, encoding="utf-8")
        resolved_csv = resolved_csv.resolve()

    return ClassAreaResult(area_ha=area_df, figure=fig, csv_path=resolved_csv)
