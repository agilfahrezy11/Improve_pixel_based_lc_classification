"""Apply a fitted classifier to a raster and write hard class predictions or probability rasters."""

import json
from os import PathLike
from pathlib import Path
from typing import Any, Literal
import numpy as np
import rasterio
from rasterio.windows import Window

from .core_types import ClassifierModel, SelectorProtocol

PathLikeInput = str | PathLike[str]


def validate_feature_stack_manifest(
    manifest_path: PathLikeInput,
    model: Any,
    raster_path: PathLikeInput | None = None,
) -> list[str]:
    """Validate that a feature stack manifest aligns with the model and raster.

    Parameters
    ----------
    manifest_path : PathLikeInput
        JSON manifest produced alongside a feature stack.
    model : Any
        Fitted model expected to have ``n_features_in_`` or ``feature_names_in_``.
    raster_path : PathLikeInput, optional
        Feature stack GeoTIFF to verify band count against manifest.

    Returns
    -------
    list of str
        The ordered list of feature names recorded in the manifest.

    Raises
    ------
    ValueError
        If feature counts or names do not match the fitted model or raster.
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_names = meta.get("features", [])
    n_manifest_features = len(feature_names)

    if raster_path is not None:
        with rasterio.open(raster_path) as src:
            if src.count != n_manifest_features:
                raise ValueError(
                    f"Raster band count ({src.count}) does not match manifest "
                    f"feature count ({n_manifest_features}) in {manifest_path}"
                )

    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != n_manifest_features:
            raise ValueError(
                f"Model expects {model.n_features_in_} features, but manifest "
                f"records {n_manifest_features} features"
            )

    return feature_names


def classify_raster(
    raster_path: PathLikeInput,
    model: ClassifierModel | Any,
    output_path: PathLikeInput,
    *,
    mode: Literal["hard", "proba"] = "hard",
    selector: SelectorProtocol | Any | None = None,
    manifest_path: PathLikeInput | None = None,
    block_size: int = 512,
    nodata: int | float = 0,
    proba_scale: float | None = 10000.0,
    output_dtype: str | np.dtype[Any] | None = None,
) -> Path:
    """Classify every valid raster pixel with a fitted model.

    The raster bands are passed to ``model.predict`` (when mode="hard") or
    ``model.predict_proba`` (when mode="proba") in their original order.
    When ``selector`` is supplied, it must be the fitted selector used during
    training. Invalid or non-finite pixels receive ``nodata``.

    Parameters
    ----------
    raster_path : PathLikeInput
        Input multiband feature raster.
    model : ClassifierModel or Any
        Fitted scikit-learn compatible classifier.
    output_path : PathLikeInput
        Destination GeoTIFF path.
    mode : {"hard", "proba"}
        - ``"hard"``: Writes a single-band GeoTIFF with predicted class IDs.
        - ``"proba"``: Writes a multi-band GeoTIFF with class probabilities.
    selector : SelectorProtocol or Any, optional
        Fitted feature selector with a ``.transform()`` method.
    manifest_path : PathLikeInput, optional
        Optional path to the feature stack JSON manifest for validation.
    block_size : int
        Window block size for tiled streaming (default: 512).
    nodata : int or float
        Nodata fill value for unclassified/invalid pixels (default: 0).
    proba_scale : float or None
        Scale factor for probabilities when mode="proba". Defaults to 10000.0
        writing uint16 ``[0, 10000]`` rasters (matching hybrid_assembly),
        or None to write float32 ``[0.0, 1.0]`` rasters.
    output_dtype : str or np.dtype, optional
        Explicit output data type. Defaults to model class dtype for hard
        classification, or uint16 (if proba_scale is set) / float32 for proba.

    Returns
    -------
    Path
        Resolved output GeoTIFF path.
    """
    if block_size < 1:
        raise ValueError("block_size must be at least 1")

    if mode not in ("hard", "proba"):
        raise ValueError(f"mode must be 'hard' or 'proba', got {mode!r}")

    if manifest_path is not None:
        validate_feature_stack_manifest(manifest_path, model, raster_path=raster_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_path) as source:
        if mode == "hard":
            n_bands = 1
            if output_dtype is None:
                if not hasattr(model, "classes_"):
                    raise ValueError(
                        "output_dtype is required when the model has no classes_ attribute"
                    )
                output_dtype = np.asarray(model.classes_).dtype
            output_dtype = np.dtype(output_dtype)
            if output_dtype.kind not in "biuf":
                raise ValueError("output_dtype must be a numeric raster data type")
        else:
            # mode == "proba"
            if not hasattr(model, "classes_"):
                raise ValueError("model must have classes_ attribute for mode='proba'")
            classes = list(model.classes_)
            n_bands = len(classes)
            if output_dtype is None:
                output_dtype = np.dtype(np.uint16) if proba_scale is not None else np.dtype(np.float32)
            else:
                output_dtype = np.dtype(output_dtype)

        profile = source.profile.copy()
        profile.update(
            count=n_bands,
            dtype=output_dtype.name,
            nodata=nodata,
            compress="deflate",
        )

        with rasterio.open(output_path, "w", **profile) as destination:
            if mode == "proba":
                # Set band descriptions to class names/IDs
                for b_idx, cls_val in enumerate(model.classes_, start=1):
                    destination.set_band_description(b_idx, f"Class_{cls_val}_prob")

            for row_start in range(0, source.height, block_size):
                height = min(block_size, source.height - row_start)
                for column_start in range(0, source.width, block_size):
                    width = min(block_size, source.width - column_start)
                    window = Window(column_start, row_start, width, height)  # type: ignore
                    block = source.read(window=window, masked=True)

                    pixels = np.ma.getdata(block).reshape(source.count, -1).T
                    masked = np.ma.getmaskarray(block).reshape(source.count, -1).T
                    valid = ~masked.any(axis=1)
                    valid &= np.isfinite(pixels).all(axis=1)

                    if mode == "hard":
                        classified = np.full((height * width), nodata, dtype=output_dtype)
                        if valid.any():
                            features = pixels[valid]
                            if selector is not None:
                                features = selector.transform(features)
                                if hasattr(features, "toarray"):
                                    features = features.toarray()
                            classified[valid] = np.asarray(
                                model.predict(features), dtype=output_dtype
                            )
                        destination.write(
                            classified.reshape(height, width), 1, window=window
                        )
                    else:
                        # mode == "proba"
                        proba_block = np.full((height * width, n_bands), nodata, dtype=output_dtype)
                        if valid.any():
                            features = pixels[valid]
                            if selector is not None:
                                features = selector.transform(features)
                                if hasattr(features, "toarray"):
                                    features = features.toarray()
                            raw_probs = model.predict_proba(features)
                            if proba_scale is not None:
                                raw_probs = np.clip(np.round(raw_probs * proba_scale), 0, proba_scale)
                            proba_block[valid] = np.asarray(raw_probs, dtype=output_dtype)

                        # Write each band
                        for b_idx in range(n_bands):
                            band_data = proba_block[:, b_idx].reshape(height, width)
                            destination.write(band_data, b_idx + 1, window=window)

    return output_path


def classify_raster_proba(
    raster_path: PathLikeInput,
    model: Any,
    output_path: PathLikeInput,
    *,
    selector: SelectorProtocol | Any | None = None,
    manifest_path: PathLikeInput | None = None,
    block_size: int = 512,
    nodata: int | float = 0,
    proba_scale: float | None = 10000.0,
    output_dtype: str | np.dtype[Any] | None = None,
) -> Path:
    """Convenience alias for :func:`classify_raster` with ``mode='proba'``."""
    return classify_raster(
        raster_path=raster_path,
        model=model,
        output_path=output_path,
        mode="proba",
        selector=selector,
        manifest_path=manifest_path,
        block_size=block_size,
        nodata=nodata,
        proba_scale=proba_scale,
        output_dtype=output_dtype,
    )
