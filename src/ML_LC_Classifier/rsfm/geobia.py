"""GEOBIA SNIC segment pooling, label matching, and probability projection for RSFMs."""

from __future__ import annotations

from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Any, Sequence
import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from .backbone import RSFMBackbone
from .finetune import RSFMCheckpointBundle
from .inference import _extract_block_embeddings


PathLikeInput = str | PathLike[str]


def _valid_segment_mask(
    seg_block: NDArray[np.integer[Any]],
    seg_nodata: float | int | None,
) -> NDArray[np.bool_]:
    """Identify valid SNIC segments, treating 0/-1 as nodata while keeping < -1."""
    valid = (seg_block != 0) & (seg_block != -1)
    if seg_nodata is not None and np.isfinite(seg_nodata) and seg_nodata not in (0, -1):
        valid &= seg_block != seg_nodata
    return valid


def pool_rsfm_by_segments(
    raster_path: PathLikeInput,
    seg_path: PathLikeInput,
    *,
    backbone: RSFMBackbone | None = None,
    bundle: RSFMCheckpointBundle | None = None,
    ancillary_paths: Sequence[PathLikeInput] | None = None,
    include_std: bool = True,
    chip_size: int = 224,
    stride: int | None = None,
    block_size: int = 2048,
    wavelengths_um: Sequence[float] | None = None,
    batch_size: int = 8,
) -> pd.DataFrame:
    """Pool RSFM embeddings and manual ancillary layers per SNIC segment in streaming blocks.

    When ``backbone`` or ``bundle`` is provided, RSFM embeddings are extracted
    in memory per padded block from ``raster_path``. When both are ``None``,
    ``raster_path`` is read directly as a pre-computed embedding/feature stack.
    Negative wrapped GEE 32-bit segment IDs (``< -1``) are preserved, while
    ``0``, ``-1``, and ``seg_nodata`` pixels are excluded.
    """
    resolved_stride = chip_size // 2 if stride is None else int(stride)
    extractor: RSFMBackbone | RSFMCheckpointBundle | None = (
        bundle if bundle is not None else backbone
    )
    overlap = max(0, chip_size // 2) if extractor is not None else 0

    seg_counts: dict[int, int] = defaultdict(int)
    seg_sums: dict[int, NDArray[np.float64]] = {}
    seg_sum_sqs: dict[int, NDArray[np.float64]] = {}

    with rasterio.open(seg_path) as src_seg, rasterio.open(raster_path) as src_ras:
        height, width = src_seg.height, src_seg.width
        if (src_ras.height, src_ras.width) != (height, width):
            raise ValueError(
                f"raster_path shape {(src_ras.height, src_ras.width)} must match "
                f"seg_path shape {(height, width)}"
            )

        if extractor is not None:
            rsfm_names = [f"embed_{i:03d}" for i in range(extractor.embed_dim)]
        else:
            descs = src_ras.descriptions or [None] * src_ras.count
            rsfm_names = [
                desc if desc else f"feat_{i:03d}" for i, desc in enumerate(descs)
            ]

        ancillary_srcs: list[rasterio.DatasetReader] = []
        ancillary_vrts: list[WarpedVRT] = []
        anc_names: list[str] = []
        try:
            anc_idx = 0
            for anc_path in ancillary_paths or ():
                anc_src = rasterio.open(anc_path)
                ancillary_srcs.append(anc_src)
                ancillary_vrts.append(
                    WarpedVRT(
                        anc_src,
                        crs=src_seg.crs,
                        transform=src_seg.transform,
                        width=width,
                        height=height,
                        resampling=Resampling.bilinear,
                    )
                )
                for b_i in range(anc_src.count):
                    desc = anc_src.descriptions[b_i] if anc_src.descriptions else None
                    anc_names.append(desc if desc else f"anc_{anc_idx:03d}")
                    anc_idx += 1

            all_names = rsfm_names + anc_names
            n_feats = len(all_names)

            for r_off in range(0, height, block_size):
                win_h = min(block_size, height - r_off)
                for c_off in range(0, width, block_size):
                    win_w = min(block_size, width - c_off)
                    write_win = Window(c_off, r_off, win_w, win_h)

                    seg_blk = src_seg.read(1, window=write_win)
                    valid_seg = _valid_segment_mask(seg_blk, src_seg.nodata)
                    if not valid_seg.any():
                        continue

                    if extractor is not None:
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
                        emb_pad, valid_pad = _extract_block_embeddings(
                            src_ras,
                            read_win,
                            extractor=extractor,
                            chip_size=chip_size,
                            stride=resolved_stride,
                            wavelengths_um=wavelengths_um,
                            batch_size=batch_size,
                        )
                        feat_blk = emb_pad[
                            :,
                            pad_top : pad_top + win_h,
                            pad_left : pad_left + win_w,
                        ]
                        valid_pix = valid_pad[
                            pad_top : pad_top + win_h,
                            pad_left : pad_left + win_w,
                        ]
                    else:
                        blk = src_ras.read(window=write_win, masked=True)
                        feat_blk = np.ma.getdata(blk).astype(np.float32, copy=False)
                        masked = np.ma.getmaskarray(blk)
                        valid_pix = (~masked.any(axis=0)) & np.isfinite(feat_blk).all(axis=0)

                    valid_all = valid_seg & valid_pix

                    if ancillary_vrts:
                        anc_blocks: list[NDArray[np.float32]] = []
                        for vrt in ancillary_vrts:
                            anc_blk = vrt.read(window=write_win, masked=True)
                            anc_raw = np.ma.getdata(anc_blk).astype(np.float32, copy=False)
                            anc_mask = np.ma.getmaskarray(anc_blk)
                            valid_all &= (~anc_mask.any(axis=0)) & np.isfinite(anc_raw).all(axis=0)
                            anc_blocks.append(anc_raw)
                        feat_blk = np.concatenate([feat_blk, *anc_blocks], axis=0)

                    if not valid_all.any():
                        continue

                    sids = seg_blk[valid_all].astype(np.int64, copy=False)
                    vals = np.moveaxis(feat_blk, 0, -1)[valid_all].astype(np.float64, copy=False)

                    uniq_ids, inv = np.unique(sids, return_inverse=True)
                    counts = np.bincount(inv)

                    sums_blk = np.empty((len(uniq_ids), n_feats), dtype=np.float64)
                    sum_sqs_blk = np.empty((len(uniq_ids), n_feats), dtype=np.float64)
                    for f_idx in range(n_feats):
                        col = vals[:, f_idx]
                        sums_blk[:, f_idx] = np.bincount(inv, weights=col)
                        sum_sqs_blk[:, f_idx] = np.bincount(inv, weights=col * col)

                    for u_idx, raw_sid in enumerate(uniq_ids):
                        sid = int(raw_sid)
                        seg_counts[sid] += int(counts[u_idx])
                        if sid in seg_sums:
                            seg_sums[sid] += sums_blk[u_idx]
                            seg_sum_sqs[sid] += sum_sqs_blk[u_idx]
                        else:
                            seg_sums[sid] = sums_blk[u_idx].copy()
                            seg_sum_sqs[sid] = sum_sqs_blk[u_idx].copy()
        finally:
            for vrt in ancillary_vrts:
                vrt.close()
            for anc_src in ancillary_srcs:
                anc_src.close()

    if not seg_counts:
        cols = ["segment_id", "pixel_count"] + [f"{n}_mean" for n in all_names]
        if include_std:
            cols += [f"{n}_std" for n in all_names]
        return pd.DataFrame(columns=cols)

    ordered_ids = sorted(seg_counts.keys())
    counts_arr = np.array([seg_counts[sid] for sid in ordered_ids], dtype=np.int64)
    sums_mat = np.stack([seg_sums[sid] for sid in ordered_ids], axis=0)
    means_mat = sums_mat / counts_arr[:, None]

    out_data: dict[str, Any] = {
        "segment_id": np.array(ordered_ids, dtype=np.int64),
        "pixel_count": counts_arr,
    }
    for f_idx, name in enumerate(all_names):
        out_data[f"{name}_mean"] = means_mat[:, f_idx].astype(np.float32)

    if include_std:
        sq_mat = np.stack([seg_sum_sqs[sid] for sid in ordered_ids], axis=0)
        var_mat = np.maximum(0.0, sq_mat / counts_arr[:, None] - means_mat * means_mat)
        stds_mat = np.sqrt(var_mat)
        for f_idx, name in enumerate(all_names):
            out_data[f"{name}_std"] = stds_mat[:, f_idx].astype(np.float32)

    return pd.DataFrame(out_data)


def assign_segment_labels(
    seg_path: PathLikeInput,
    training_path: PathLikeInput,
    *,
    class_field: str = "class_id",
    block_size: int = 2048,
) -> pd.DataFrame:
    """Assign majority training class labels to SNIC segments intersecting vectors."""
    gdf = gpd.read_file(training_path)
    if class_field not in gdf.columns:
        raise ValueError(f"Column '{class_field}' not found in {training_path}")
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return pd.DataFrame(columns=["segment_id", class_field])

    votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    with rasterio.open(seg_path) as src_seg:
        if src_seg.crs is None:
            raise ValueError(f"Segment raster has no CRS: {seg_path}")
        if gdf.crs is not None and gdf.crs != src_seg.crs:
            gdf = gdf.to_crs(src_seg.crs)

        gdf[class_field] = gdf[class_field].astype(int)
        shapes = [
            (geom, int(cls_val))
            for geom, cls_val in zip(gdf.geometry, gdf[class_field])
        ]

        height, width = src_seg.height, src_seg.width
        for r_off in range(0, height, block_size):
            win_h = min(block_size, height - r_off)
            for c_off in range(0, width, block_size):
                win_w = min(block_size, width - c_off)
                win = Window(c_off, r_off, win_w, win_h)
                win_transform = src_seg.window_transform(win)

                seg_blk = src_seg.read(1, window=win)
                valid_seg = _valid_segment_mask(seg_blk, src_seg.nodata)
                if not valid_seg.any():
                    continue

                lbl_blk = rasterize(
                    shapes=shapes,
                    out_shape=(win_h, win_w),
                    transform=win_transform,
                    fill=0,
                    dtype="int64",
                )
                labeled = valid_seg & (lbl_blk != 0)
                if not labeled.any():
                    continue

                sids = seg_blk[labeled].astype(np.int64, copy=False)
                lbls = lbl_blk[labeled].astype(np.int64, copy=False)
                pairs, pair_counts = np.unique(
                    np.column_stack([sids, lbls]), axis=0, return_counts=True
                )
                for (sid, cls_val), cnt in zip(pairs, pair_counts):
                    votes[int(sid)][int(cls_val)] += int(cnt)

    rows = []
    for sid in sorted(votes.keys()):
        best_cls = max(votes[sid].items(), key=lambda item: (item[1], -item[0]))[0]
        rows.append({"segment_id": sid, class_field: best_cls})
    return pd.DataFrame(rows, columns=["segment_id", class_field])


def project_segment_probabilities(
    seg_path: PathLikeInput,
    segment_ids: Sequence[int] | NDArray[np.integer[Any]],
    segment_probs: NDArray[np.floating[Any]],
    output_path: PathLikeInput,
    *,
    class_values: Sequence[int] | None = None,
    block_size: int = 2048,
    output_dtype: str = "float32",
) -> Path:
    """Project per-segment class probabilities onto the SNIC segment raster grid."""
    if output_dtype not in ("float32", "uint16"):
        raise ValueError("output_dtype must be 'float32' or 'uint16'")

    seg_id_arr = np.asarray(segment_ids, dtype=np.int64)
    probs_arr = np.asarray(segment_probs, dtype=np.float32)
    if probs_arr.ndim != 2 or probs_arr.shape[0] != len(seg_id_arr):
        raise ValueError("segment_probs must have shape (len(segment_ids), num_classes)")

    order = np.argsort(seg_id_arr)
    sorted_ids = seg_id_arr[order]
    sorted_probs = probs_arr[order]
    num_classes = probs_arr.shape[1]

    resolved_classes = (
        list(range(1, num_classes + 1)) if class_values is None else [int(c) for c in class_values]
    )
    band_names = tuple(f"class_{c}" for c in resolved_classes)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(seg_path) as src_seg:
        height, width = src_seg.height, src_seg.width
        profile = src_seg.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("tiled", None)
        nodata_val: float | int = float("nan") if output_dtype == "float32" else 0
        profile.update(
            count=num_classes,
            dtype=output_dtype,
            nodata=nodata_val,
            compress="deflate",
        )
        if height >= 256 and width >= 256:
            profile.update(tiled=True, blockxsize=256, blockysize=256)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.descriptions = band_names

            for r_off in range(0, height, block_size):
                win_h = min(block_size, height - r_off)
                for c_off in range(0, width, block_size):
                    win_w = min(block_size, width - c_off)
                    win = Window(c_off, r_off, win_w, win_h)

                    seg_blk = src_seg.read(1, window=win)
                    valid_seg = _valid_segment_mask(seg_blk, src_seg.nodata)

                    if output_dtype == "uint16":
                        out_blk = np.zeros((num_classes, win_h, win_w), dtype=np.uint16)
                    else:
                        out_blk = np.full((num_classes, win_h, win_w), np.nan, dtype=np.float32)

                    if valid_seg.any() and len(sorted_ids) > 0:
                        sids = seg_blk[valid_seg].astype(np.int64, copy=False)
                        pos = np.searchsorted(sorted_ids, sids)
                        matched = (pos < len(sorted_ids)) & (
                            sorted_ids[np.minimum(pos, len(sorted_ids) - 1)] == sids
                        )
                        if matched.any():
                            matched_probs = sorted_probs[pos[matched]]  # (M_matched, K)
                            if output_dtype == "uint16":
                                matched_vals = np.rint(
                                    np.clip(matched_probs * 10000.0, 0.0, 10000.0)
                                ).astype(np.uint16)
                            else:
                                matched_vals = matched_probs

                            valid_rc = np.where(valid_seg)
                            r_m = valid_rc[0][matched]
                            c_m = valid_rc[1][matched]
                            out_blk[:, r_m, c_m] = matched_vals.T

                    dst.write(out_blk, window=win)

    return out_path
