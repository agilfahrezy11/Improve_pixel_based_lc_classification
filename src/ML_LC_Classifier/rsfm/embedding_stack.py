"""Windowed RSFM embedding extraction with 2D Hann/Gaussian overlap blending."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.windows import Window

from ..feature_stack import FeatureStackResult
from .backbone import RSFMBackbone


PathLikeInput = str | PathLike[str]


def _blend_window_2d(size: int) -> NDArray[np.float32]:
    """Return a strictly positive 2D Hann spatial blending window."""
    if size < 1:
        raise ValueError("chip_size must be at least 1")
    if size == 1:
        return np.ones((1, 1), dtype=np.float32)
    win_1d = np.hanning(size + 2)[1:-1].astype(np.float32)
    win_2d = np.outer(win_1d, win_1d)
    return np.maximum(win_2d, np.float32(1e-3))


def _tile_starts(length: int, chip_size: int, stride: int) -> list[int]:
    """Compute sliding-window start offsets covering ``[0, length)``."""
    if length <= chip_size:
        return [0]
    starts = list(range(0, length - chip_size + 1, stride))
    if starts[-1] + chip_size < length:
        starts.append(length - chip_size)
    return starts


def extract_rsfm_embeddings(
    raster_path: PathLikeInput,
    output_path: PathLikeInput,
    *,
    backbone: RSFMBackbone,
    chip_size: int = 224,
    stride: int | None = None,
    wavelengths_um: Sequence[float] | None = None,
    batch_size: int = 8,
    band_prefix: str = "embed",
) -> FeatureStackResult:
    """Extract overlapping RSFM embeddings from a raster into a ``FeatureStackResult``.

    Chips of size ``chip_size x chip_size`` are extracted across the source
    raster with step ``stride``, passed through ``backbone.extract_features``,
    and blended with a 2D Hann window to prevent patch-boundary seams.
    Invalid/nodata pixels in the source raster are preserved as ``NaN``.
    """
    if chip_size < 1:
        raise ValueError("chip_size must be at least 1")
    resolved_stride = chip_size // 2 if stride is None else int(stride)
    if resolved_stride < 1 or resolved_stride > chip_size:
        raise ValueError("stride must be between 1 and chip_size")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    source_path = Path(raster_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(f"{out_path}.json")

    embed_dim = backbone.embed_dim
    feature_names = tuple(f"{band_prefix}_{idx:03d}" for idx in range(embed_dim))
    blend_2d = _blend_window_2d(chip_size)

    with rasterio.open(source_path) as src:
        height, width, bands = src.height, src.width, src.count
        profile = src.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("tiled", None)
        profile.update(
            count=embed_dim,
            dtype="float32",
            nodata=float("nan"),
            compress="deflate",
        )
        if height >= 256 and width >= 256:
            profile.update(tiled=True, blockxsize=256, blockysize=256)

        row_starts = _tile_starts(height, chip_size, resolved_stride)
        col_starts = _tile_starts(width, chip_size, resolved_stride)

        accum = np.zeros((embed_dim, height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        valid_mask = np.ones((height, width), dtype=bool)

        coords = [(r, c) for r in row_starts for c in col_starts]
        for idx in range(0, len(coords), batch_size):
            batch_coords = coords[idx : idx + batch_size]
            batch_chips: list[NDArray[np.float32]] = []
            batch_spans: list[tuple[int, int, int, int]] = []

            for r0, c0 in batch_coords:
                read_h = min(chip_size, height - r0)
                read_w = min(chip_size, width - c0)
                win = Window(c0, r0, read_w, read_h)
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
            feats = backbone.extract_features(chip_tensor, wavelengths_um=wavelengths_um)
            if feats.shape != (len(batch_coords), embed_dim, chip_size, chip_size):
                raise ValueError(
                    f"Backbone returned shape {feats.shape}, expected "
                    f"{(len(batch_coords), embed_dim, chip_size, chip_size)}"
                )

            for b_idx, (r0, c0, read_h, read_w) in enumerate(batch_spans):
                w_slice = blend_2d[:read_h, :read_w]
                accum[:, r0 : r0 + read_h, c0 : c0 + read_w] += (
                    feats[b_idx, :, :read_h, :read_w] * w_slice[None, :, :]
                )
                weights[r0 : r0 + read_h, c0 : c0 + read_w] += w_slice

        with np.errstate(divide="ignore", invalid="ignore"):
            embeddings = accum / np.maximum(weights[None, :, :], np.float32(1e-8))
        embeddings[:, ~valid_mask] = np.nan

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(embeddings.astype(np.float32, copy=False))
        dst.descriptions = feature_names

    manifest: dict[str, Any] = {
        "source_raster": str(source_path),
        "feature_names": list(feature_names),
        "embed_dim": embed_dim,
        "chip_size": chip_size,
        "stride": resolved_stride,
        "wavelengths_um": list(wavelengths_um) if wavelengths_um is not None else None,
        "source_bands": bands,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return FeatureStackResult(
        stack_path=out_path,
        manifest_path=manifest_path,
        feature_names=feature_names,
    )
