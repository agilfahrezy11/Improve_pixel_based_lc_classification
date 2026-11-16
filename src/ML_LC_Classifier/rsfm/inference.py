"""Streaming windowed RSFM probability inference with in-memory late ancillary fusion."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from ..core_types import ClassifierModelWithProba
from .backbone import RSFMBackbone
from .embedding_stack import _blend_window_2d, _tile_starts
from .finetune import RSFMCheckpointBundle


PathLikeInput = str | PathLike[str]


def _extract_block_embeddings(
    src: rasterio.DatasetReader,
    read_win: Window,
    *,
    extractor: RSFMBackbone | RSFMCheckpointBundle,
    chip_size: int,
    stride: int,
    wavelengths_um: Sequence[float] | None,
    batch_size: int,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Compute blended ``(D, H_blk, W_blk)`` embeddings and valid mask for one block."""
    blk_h, blk_w = int(read_win.height), int(read_win.width)
    embed_dim = extractor.embed_dim
    blend_2d = _blend_window_2d(chip_size)

    accum = np.zeros((embed_dim, blk_h, blk_w), dtype=np.float32)
    weights = np.zeros((blk_h, blk_w), dtype=np.float32)
    valid_mask = np.ones((blk_h, blk_w), dtype=bool)

    row_starts = _tile_starts(blk_h, chip_size, stride)
    col_starts = _tile_starts(blk_w, chip_size, stride)
    coords = [(r, c) for r in row_starts for c in col_starts]

    for idx in range(0, len(coords), batch_size):
        batch_coords = coords[idx : idx + batch_size]
        batch_chips: list[NDArray[np.float32]] = []
        batch_spans: list[tuple[int, int, int, int]] = []

        for r0, c0 in batch_coords:
            read_h = min(chip_size, blk_h - r0)
            read_w = min(chip_size, blk_w - c0)
            win = Window(int(read_win.col_off) + c0, int(read_win.row_off) + r0, read_w, read_h)
            block = src.read(window=win, masked=True)

            raw = np.ma.getdata(block).astype(np.float32, copy=True)
            masked = np.ma.getmaskarray(block)
            invalid = masked.any(axis=0) | (~np.isfinite(raw).all(axis=0))
            valid_mask[r0 : r0 + read_h, c0 : c0 + read_w] &= ~invalid

            if invalid.any():
                raw[:, invalid] = 0.0
            if read_h < chip_size or read_w < chip_size:
                raw = np.pad(
                    raw,
                    ((0, 0), (0, chip_size - read_h), (0, chip_size - read_w)),
                    mode="edge",
                )

            batch_chips.append(raw)
            batch_spans.append((r0, c0, read_h, read_w))

        chip_tensor = np.stack(batch_chips, axis=0)
        if isinstance(extractor, RSFMCheckpointBundle):
            feats = extractor.predict_embeddings(chip_tensor)
        else:
            feats = extractor.extract_features(chip_tensor, wavelengths_um=wavelengths_um)

        for b_idx, (r0, c0, read_h, read_w) in enumerate(batch_spans):
            w_slice = blend_2d[:read_h, :read_w]
            accum[:, r0 : r0 + read_h, c0 : c0 + read_w] += (
                feats[b_idx, :, :read_h, :read_w] * w_slice[None, :, :]
            )
            weights[r0 : r0 + read_h, c0 : c0 + read_w] += w_slice

    with np.errstate(divide="ignore", invalid="ignore"):
        embeddings = accum / np.maximum(weights[None, :, :], np.float32(1e-8))
    return embeddings, valid_mask


def _extract_block_bundle_probs(
    src: rasterio.DatasetReader,
    read_win: Window,
    *,
    bundle: RSFMCheckpointBundle,
    chip_size: int,
    stride: int,
    batch_size: int,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Compute blended ``(K, H_blk, W_blk)`` probabilities directly from a LoRA bundle."""
    blk_h, blk_w = int(read_win.height), int(read_win.width)
    num_classes = len(bundle.class_values)
    blend_2d = _blend_window_2d(chip_size)

    accum = np.zeros((num_classes, blk_h, blk_w), dtype=np.float32)
    weights = np.zeros((blk_h, blk_w), dtype=np.float32)
    valid_mask = np.ones((blk_h, blk_w), dtype=bool)

    row_starts = _tile_starts(blk_h, chip_size, stride)
    col_starts = _tile_starts(blk_w, chip_size, stride)
    coords = [(r, c) for r in row_starts for c in col_starts]

    for idx in range(0, len(coords), batch_size):
        batch_coords = coords[idx : idx + batch_size]
        batch_chips: list[NDArray[np.float32]] = []
        batch_spans: list[tuple[int, int, int, int]] = []

        for r0, c0 in batch_coords:
            read_h = min(chip_size, blk_h - r0)
            read_w = min(chip_size, blk_w - c0)
            win = Window(int(read_win.col_off) + c0, int(read_win.row_off) + r0, read_w, read_h)
            block = src.read(window=win, masked=True)

            raw = np.ma.getdata(block).astype(np.float32, copy=True)
            masked = np.ma.getmaskarray(block)
            invalid = masked.any(axis=0) | (~np.isfinite(raw).all(axis=0))
            valid_mask[r0 : r0 + read_h, c0 : c0 + read_w] &= ~invalid

            if invalid.any():
                raw[:, invalid] = 0.0
            if read_h < chip_size or read_w < chip_size:
                raw = np.pad(
                    raw,
                    ((0, 0), (0, chip_size - read_h), (0, chip_size - read_w)),
                    mode="edge",
                )

            batch_chips.append(raw)
            batch_spans.append((r0, c0, read_h, read_w))

        chip_tensor = np.stack(batch_chips, axis=0)
        probs = bundle.predict_proba(chip_tensor)

        for b_idx, (r0, c0, read_h, read_w) in enumerate(batch_spans):
            w_slice = blend_2d[:read_h, :read_w]
            accum[:, r0 : r0 + read_h, c0 : c0 + read_w] += (
                probs[b_idx, :, :read_h, :read_w] * w_slice[None, :, :]
            )
            weights[r0 : r0 + read_h, c0 : c0 + read_w] += w_slice

    with np.errstate(divide="ignore", invalid="ignore"):
        blended = accum / np.maximum(weights[None, :, :], np.float32(1e-8))
    return blended, valid_mask


def infer_rsfm_probabilities(
    raster_path: PathLikeInput,
    output_path: PathLikeInput,
    *,
    backbone: RSFMBackbone | None = None,
    bundle: RSFMCheckpointBundle | None = None,
    classifier: ClassifierModelWithProba | None = None,
    ancillary_paths: Sequence[PathLikeInput] | None = None,
    chip_size: int = 224,
    stride: int | None = None,
    block_size: int = 2048,
    wavelengths_um: Sequence[float] | None = None,
    batch_size: int = 8,
    output_dtype: str = "float32",
) -> Path:
    """Stream a raster through an RSFM (and optional ancillary layers) to write class probabilities.

    When ``classifier`` is provided, ``D``-dimensional RSFM embeddings and
    ``WarpedVRT``-aligned manual ancillary rasters (e.g., Elevation, Slope,
    TPI, TWI, dist_road, dist_urban) are concatenated in memory per block and
    evaluated via ``classifier.predict_proba`` without writing intermediate
    embedding rasters to disk.

    When ``classifier`` is ``None`` and ``bundle`` is provided, the fine-tuned
    LoRA neural head predicts class probabilities directly.
    """
    if bundle is None and backbone is None:
        raise ValueError("Provide either a fine-tuned 'bundle' or a 'backbone'")
    if classifier is None and bundle is None:
        raise ValueError("'classifier' is required when inferring from a raw backbone")
    if output_dtype not in ("float32", "uint16"):
        raise ValueError("output_dtype must be 'float32' or 'uint16'")

    resolved_stride = chip_size // 2 if stride is None else int(stride)
    if resolved_stride < 1 or resolved_stride > chip_size:
        raise ValueError("stride must be between 1 and chip_size")

    extractor: RSFMBackbone | RSFMCheckpointBundle = bundle if bundle is not None else backbone  # type: ignore[assignment]
    if classifier is not None and hasattr(classifier, "classes_"):
        class_values = [int(c) for c in classifier.classes_]
    elif bundle is not None:
        class_values = list(bundle.class_values)
    else:
        raise ValueError("Unable to determine class_values from classifier or bundle")

    num_classes = len(class_values)
    band_names = tuple(f"class_{c}" for c in class_values)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlap = max(0, chip_size // 2)

    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width
        profile = src.profile.copy()
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

        ancillary_srcs: list[rasterio.DatasetReader] = []
        ancillary_vrts: list[WarpedVRT] = []
        try:
            for anc_path in ancillary_paths or ():
                anc_src = rasterio.open(anc_path)
                ancillary_srcs.append(anc_src)
                ancillary_vrts.append(
                    WarpedVRT(
                        anc_src,
                        crs=src.crs,
                        transform=src.transform,
                        width=width,
                        height=height,
                        resampling=Resampling.bilinear,
                    )
                )

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.descriptions = band_names

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

                        if classifier is None and bundle is not None:
                            probs_pad, valid_pad = _extract_block_bundle_probs(
                                src,
                                read_win,
                                bundle=bundle,
                                chip_size=chip_size,
                                stride=resolved_stride,
                                batch_size=batch_size,
                            )
                        else:
                            emb_pad, valid_pad = _extract_block_embeddings(
                                src,
                                read_win,
                                extractor=extractor,
                                chip_size=chip_size,
                                stride=resolved_stride,
                                wavelengths_um=wavelengths_um,
                                batch_size=batch_size,
                            )

                            # Concatenate aligned ancillary layers in memory
                            if ancillary_vrts:
                                anc_bands: list[NDArray[np.float32]] = []
                                for vrt in ancillary_vrts:
                                    anc_blk = vrt.read(window=read_win, masked=True)
                                    anc_raw = np.ma.getdata(anc_blk).astype(np.float32, copy=False)
                                    anc_mask = np.ma.getmaskarray(anc_blk)
                                    valid_pad &= ~anc_mask.any(axis=0)
                                    valid_pad &= np.isfinite(anc_raw).all(axis=0)
                                    anc_bands.append(anc_raw)
                                fused_pad = np.concatenate([emb_pad, *anc_bands], axis=0)
                            else:
                                fused_pad = emb_pad

                            blk_h, blk_w = fused_pad.shape[1], fused_pad.shape[2]
                            probs_pad = np.full(
                                (num_classes, blk_h, blk_w),
                                np.nan,
                                dtype=np.float32,
                            )
                            if valid_pad.any():
                                flat_feats = np.moveaxis(fused_pad, 0, -1)[valid_pad]
                                pred_probs = np.asarray(
                                    classifier.predict_proba(flat_feats),  # type: ignore[union-attr]
                                    dtype=np.float32,
                                )
                                probs_hwc = np.moveaxis(probs_pad, 0, -1)
                                probs_hwc[valid_pad] = pred_probs
                                probs_pad = np.moveaxis(probs_hwc, -1, 0)

                        # Crop padding back to write window
                        probs_crop = probs_pad[
                            :,
                            pad_top : pad_top + win_h,
                            pad_left : pad_left + win_w,
                        ]
                        valid_crop = valid_pad[
                            pad_top : pad_top + win_h,
                            pad_left : pad_left + win_w,
                        ]

                        if output_dtype == "uint16":
                            out_arr = np.zeros((num_classes, win_h, win_w), dtype=np.uint16)
                            if valid_crop.any():
                                scaled = np.clip(probs_crop * 10000.0, 0.0, 10000.0)
                                out_arr[:, valid_crop] = np.rint(
                                    scaled[:, valid_crop]
                                ).astype(np.uint16)
                            dst.write(out_arr, window=write_win)
                        else:
                            probs_crop[:, ~valid_crop] = np.nan
                            dst.write(probs_crop.astype(np.float32, copy=False), window=write_win)
        finally:
            for vrt in ancillary_vrts:
                vrt.close()
            for anc_src in ancillary_srcs:
                anc_src.close()

    return out_path
