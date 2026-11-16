"""Bi-temporal RSFM embedding cosine stability masks and sample disturbance filtering."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Sequence
import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window

from .backbone import RSFMBackbone
from .finetune import RSFMCheckpointBundle
from .geobia import _valid_segment_mask
from .inference import _extract_block_embeddings


PathLikeInput = str | PathLike[str]


def _cosine_distance(
    z1: NDArray[np.float32],
    z2: NDArray[np.float32],
    axis: int = 0,
) -> NDArray[np.float32]:
    """Compute cosine distance ``1 - cos_sim(z1, z2)`` clipped to ``[0, 2]``."""
    dot = np.sum(z1 * z2, axis=axis)
    norm1 = np.linalg.norm(z1, axis=axis)
    norm2 = np.linalg.norm(z2, axis=axis)
    denom = np.maximum(norm1 * norm2, np.float32(1e-8))
    dist = 1.0 - (dot / denom)
    return np.clip(dist, 0.0, 2.0).astype(np.float32, copy=False)


def compute_embedding_stability(
    raster_t1_path: PathLikeInput,
    raster_t2_path: PathLikeInput,
    distance_output_path: PathLikeInput,
    stable_mask_output_path: PathLikeInput,
    *,
    backbone: RSFMBackbone | None = None,
    bundle: RSFMCheckpointBundle | None = None,
    backbone_t2: RSFMBackbone | None = None,
    bundle_t2: RSFMCheckpointBundle | None = None,
    seg_path: PathLikeInput | None = None,
    tau_stable: float = 0.15,
    chip_size: int = 224,
    stride: int | None = None,
    block_size: int = 2048,
    wavelengths_t1_um: Sequence[float] | None = None,
    wavelengths_t2_um: Sequence[float] | None = None,
    batch_size: int = 8,
) -> tuple[Path, Path]:
    """Compute bi-temporal embedding cosine distance and binary stability mask.

    Writes a ``float32`` cosine distance GeoTIFF (``1 - cos_sim`` in ``[0, 2]``,
    ``NaN`` for nodata) and a ``uint8`` stability mask GeoTIFF (``1`` where
    ``distance <= tau_stable``, ``0`` where disturbed or nodata). When ``seg_path``
    is supplied, embeddings are averaged per segment before computing distance.
    """
    resolved_stride = chip_size // 2 if stride is None else int(stride)
    ext1: RSFMBackbone | RSFMCheckpointBundle | None = (
        bundle if bundle is not None else backbone
    )
    ext2: RSFMBackbone | RSFMCheckpointBundle | None = (
        bundle_t2 if bundle_t2 is not None else (backbone_t2 if backbone_t2 is not None else ext1)
    )
    overlap = max(0, chip_size // 2) if (ext1 is not None or ext2 is not None) else 0

    dist_path = Path(distance_output_path)
    mask_path = Path(stable_mask_output_path)
    dist_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_t1_path) as src1, rasterio.open(raster_t2_path) as src2:
        if src1.shape != src2.shape:
            raise ValueError(
                f"Grid mismatch: t1 {src1.shape} vs t2 {src2.shape}"
            )
        height, width = src1.height, src1.width

        dist_profile = src1.profile.copy()
        dist_profile.pop("blockxsize", None)
        dist_profile.pop("blockysize", None)
        dist_profile.pop("tiled", None)
        dist_profile.update(count=1, dtype="float32", nodata=float("nan"), compress="deflate")

        mask_profile = dist_profile.copy()
        mask_profile.update(dtype="uint8", nodata=0)
        if height >= 256 and width >= 256:
            dist_profile.update(tiled=True, blockxsize=256, blockysize=256)
            mask_profile.update(tiled=True, blockxsize=256, blockysize=256)

        src_seg = rasterio.open(seg_path) if seg_path is not None else None
        try:
            if src_seg is not None and src_seg.shape != (height, width):
                raise ValueError(f"seg_path shape {src_seg.shape} must match {(height, width)}")

            with rasterio.open(dist_path, "w", **dist_profile) as dst_dist, \
                 rasterio.open(mask_path, "w", **mask_profile) as dst_mask:

                for r_off in range(0, height, block_size):
                    win_h = min(block_size, height - r_off)
                    for c_off in range(0, width, block_size):
                        win_w = min(block_size, width - c_off)
                        write_win = Window(c_off, r_off, win_w, win_h)

                        pad_top = min(overlap, r_off)
                        pad_bottom = min(overlap, height - (r_off + win_h))
                        pad_left = min(overlap, c_off)
                        pad_right = min(overlap, width - (c_off + win_w))
                        read_win = Window(
                            c_off - pad_left,
                            r_off - pad_top,
                            win_w + pad_left + pad_right,
                            win_h + pad_top + pad_bottom,
                        )

                        if ext1 is not None:
                            e1_pad, v1_pad = _extract_block_embeddings(
                                src1,
                                read_win,
                                extractor=ext1,
                                chip_size=chip_size,
                                stride=resolved_stride,
                                wavelengths_um=wavelengths_t1_um,
                                batch_size=batch_size,
                            )
                            z1 = e1_pad[:, pad_top : pad_top + win_h, pad_left : pad_left + win_w]
                            v1 = v1_pad[pad_top : pad_top + win_h, pad_left : pad_left + win_w]
                        else:
                            b1 = src1.read(window=write_win, masked=True)
                            z1 = np.ma.getdata(b1).astype(np.float32, copy=False)
                            v1 = (~np.ma.getmaskarray(b1).any(axis=0)) & np.isfinite(z1).all(axis=0)

                        if ext2 is not None:
                            e2_pad, v2_pad = _extract_block_embeddings(
                                src2,
                                read_win,
                                extractor=ext2,
                                chip_size=chip_size,
                                stride=resolved_stride,
                                wavelengths_um=wavelengths_t2_um,
                                batch_size=batch_size,
                            )
                            z2 = e2_pad[:, pad_top : pad_top + win_h, pad_left : pad_left + win_w]
                            v2 = v2_pad[pad_top : pad_top + win_h, pad_left : pad_left + win_w]
                        else:
                            b2 = src2.read(window=write_win, masked=True)
                            z2 = np.ma.getdata(b2).astype(np.float32, copy=False)
                            v2 = (~np.ma.getmaskarray(b2).any(axis=0)) & np.isfinite(z2).all(axis=0)

                        valid = v1 & v2
                        dist_blk = np.full((win_h, win_w), np.nan, dtype=np.float32)
                        mask_blk = np.zeros((win_h, win_w), dtype=np.uint8)

                        if src_seg is not None:
                            seg_blk = src_seg.read(1, window=write_win)
                            valid &= _valid_segment_mask(seg_blk, src_seg.nodata)
                            if valid.any():
                                sids = seg_blk[valid].astype(np.int64, copy=False)
                                uniq_ids, inv = np.unique(sids, return_inverse=True)
                                counts = np.bincount(inv).astype(np.float32)
                                n_dim = z1.shape[0]
                                mean1 = np.empty((n_dim, len(uniq_ids)), dtype=np.float32)
                                mean2 = np.empty((n_dim, len(uniq_ids)), dtype=np.float32)
                                z1_v = z1[:, valid]
                                z2_v = z2[:, valid]
                                for d in range(n_dim):
                                    mean1[d] = np.bincount(inv, weights=z1_v[d]) / counts
                                    mean2[d] = np.bincount(inv, weights=z2_v[d]) / counts
                                seg_dist = _cosine_distance(mean1, mean2, axis=0)
                                dist_vals = seg_dist[inv]
                                dist_blk[valid] = dist_vals
                                mask_blk[valid] = (dist_vals <= tau_stable).astype(np.uint8)
                        elif valid.any():
                            dist_vals = _cosine_distance(z1[:, valid], z2[:, valid], axis=0)
                            dist_blk[valid] = dist_vals
                            mask_blk[valid] = (dist_vals <= tau_stable).astype(np.uint8)

                        dst_dist.write(dist_blk, 1, window=write_win)
                        dst_mask.write(mask_blk, 1, window=write_win)
        finally:
            if src_seg is not None:
                src_seg.close()

    return dist_path, mask_path


def filter_disturbed_samples(
    training_path: PathLikeInput,
    distance_raster_path: PathLikeInput,
    *,
    max_change_distance: float = 0.15,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Flag and filter training samples whose bi-temporal cosine distance exceeds threshold.

    Returns ``(clean_gdf, flagged_gdf)`` where ``flagged_gdf`` includes
    ``change_distance`` and ``is_disturbed`` columns, and ``clean_gdf`` keeps
    only undisturbed samples with finite distance ``<= max_change_distance``.
    """
    gdf = gpd.read_file(training_path)
    if gdf.empty:
        flagged = gdf.copy()
        flagged["change_distance"] = np.nan
        flagged["is_disturbed"] = False
        return flagged.copy(), flagged

    with rasterio.open(distance_raster_path) as src:
        if src.crs is not None and gdf.crs is not None and gdf.crs != src.crs:
            work_gdf = gdf.to_crs(src.crs)
        else:
            work_gdf = gdf

        dist_arr = src.read(1, masked=True)
        raw_dist = np.ma.getdata(dist_arr).astype(np.float32, copy=False)
        invalid = np.ma.getmaskarray(dist_arr) | (~np.isfinite(raw_dist))

        distances: list[float] = []
        for geom in work_gdf.geometry:
            if geom is None or geom.is_empty:
                distances.append(float("nan"))
                continue
            if geom.geom_type == "Point":
                r, c = src.index(geom.x, geom.y)
                if 0 <= r < src.height and 0 <= c < src.width and not invalid[r, c]:
                    distances.append(float(raw_dist[r, c]))
                else:
                    distances.append(float("nan"))
            else:
                inside = ~geometry_mask(
                    [geom],
                    out_shape=(src.height, src.width),
                    transform=src.transform,
                    invert=False,
                    all_touched=True,
                )
                valid_geom = inside & (~invalid)
                if valid_geom.any():
                    distances.append(float(np.mean(raw_dist[valid_geom])))
                else:
                    distances.append(float("nan"))

    flagged = gdf.copy()
    dist_series = np.array(distances, dtype=np.float32)
    flagged["change_distance"] = dist_series
    flagged["is_disturbed"] = (~np.isfinite(dist_series)) | (dist_series > max_change_distance)
    clean = flagged[~flagged["is_disturbed"]].copy()
    return clean, flagged
