"""Array-based Bayesian smoothing for land-cover probability rasters."""

from collections.abc import Mapping, Sequence
from math import floor
from operator import index
from typing import TypeAlias
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from scipy.special import expit

FloatArray: TypeAlias = NDArray[np.float64]

_EPSILON = 1e-3


def _class_first(
    values: ArrayLike,
    class_axis: int,
) -> tuple[FloatArray, int, bool]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim not in (2, 3):
        raise ValueError("probabilities must be a 2D or 3D array")
    class_axis = index(class_axis)

    was_2d = array.ndim == 2
    if was_2d:
        if class_axis not in (0, -1):
            raise ValueError("class_axis must be 0 or -1 for a 2D array")
        return array[np.newaxis, ...], 0, True

    if class_axis < -array.ndim or class_axis >= array.ndim:
        raise ValueError(f"class_axis {class_axis} is out of bounds for a 3D array")
    normalized_axis = class_axis % array.ndim
    return np.moveaxis(array, normalized_axis, 0), normalized_axis, False


def _restore_class_axis(
    values: FloatArray,
    original_axis: int,
    was_2d: bool,
) -> FloatArray:
    if was_2d:
        return values[0]
    return np.moveaxis(values, 0, original_axis)


def _logits_from_class_first(classes: FloatArray) -> FloatArray:
    if np.any(np.isinf(classes)):
        raise ValueError("probabilities must not contain infinite values")
    clipped = np.clip(classes, _EPSILON, 1 - _EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def _validate_window(window_size: int, neigh_fraction: float) -> tuple[int, int]:
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise TypeError("window_size must be an integer")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if not np.isfinite(neigh_fraction) or not 0 <= neigh_fraction <= 1:
        raise ValueError("neigh_fraction must be between 0 and 1")
    percentile = floor((1 - neigh_fraction) * 100 + 0.5)
    return window_size, percentile


def probs_to_logits(probabilities: ArrayLike, class_axis: int = 0) -> FloatArray:
    """Convert probabilities to logits, clipping endpoints for numerical stability."""
    classes, original_axis, was_2d = _class_first(probabilities, class_axis)
    logits = _logits_from_class_first(classes)
    return _restore_class_axis(logits, original_axis, was_2d)


def logits_to_probs(
    logits: ArrayLike,
    class_axis: int = 0,
    normalize: bool = True,
) -> FloatArray:
    """Convert logits to probabilities and optionally normalize across classes."""
    classes, original_axis, was_2d = _class_first(logits, class_axis)
    probabilities = expit(classes)
    if normalize:
        totals = np.sum(probabilities, axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            probabilities = probabilities / totals
    return _restore_class_axis(probabilities, original_axis, was_2d)


def _smoothness_by_class(
    smoothness: float | Sequence[float] | Mapping[str, float],
    class_count: int,
    class_bands: Sequence[str] | None,
) -> FloatArray:
    if isinstance(smoothness, Mapping):
        if class_bands is None or len(class_bands) != class_count:
            raise ValueError(
                "class_bands matching the class dimension are required when "
                "smoothness is a mapping"
            )
        values = [smoothness.get(name, 1.0) for name in class_bands]
    elif np.isscalar(smoothness):
        values = [float(smoothness)] * class_count
    else:
        values = list(smoothness)
        if len(values) != class_count:
            raise ValueError("smoothness sequence must match the class dimension")

    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("smoothness values must be finite and non-negative")
    return result[:, np.newaxis, np.newaxis]


def local_mean_variance(
    logits: ArrayLike,
    window_size: int = 7,
    neigh_fraction: float = 0.5,
    class_axis: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Return local mean and variance of logits retained above a local percentile.

    Neighborhoods are square, and pixels outside the array or with non-finite
    values are ignored. A missing local mean falls back to the input logit;
    missing or zero variances are set to 1, matching the Earth Engine version.
    """
    size, percentile = _validate_window(window_size, neigh_fraction)
    classes, original_axis, was_2d = _class_first(logits, class_axis)
    kernel = np.ones((size, size), dtype=np.float64)
    means = np.empty_like(classes)
    variances = np.empty_like(classes)

    for index, band in enumerate(classes):
        threshold = ndimage.generic_filter(
            band,
            _finite_percentile,
            size=size,
            mode="constant",
            cval=np.nan,
            extra_arguments=(percentile,),
        )
        selected = np.isfinite(band) & (band >= threshold)
        selected_values = np.where(selected, band, 0.0)
        counts = ndimage.convolve(
            selected.astype(np.float64), kernel, mode="constant", cval=0.0
        )
        totals = ndimage.convolve(
            selected_values, kernel, mode="constant", cval=0.0
        )
        squared_totals = ndimage.convolve(
            selected_values * selected_values, kernel, mode="constant", cval=0.0
        )

        mean = np.divide(
            totals,
            counts,
            out=band.copy(),
            where=counts > 0,
        )
        variance = np.divide(
            squared_totals,
            counts,
            out=np.ones_like(band),
            where=counts > 0,
        ) - np.square(mean)
        variance = np.maximum(variance, 0.0)
        variance[(counts == 0) | (variance == 0)] = 1.0
        mean[~np.isfinite(band)] = np.nan
        variance[~np.isfinite(band)] = 1.0
        means[index] = mean
        variances[index] = variance

    return (
        _restore_class_axis(means, original_axis, was_2d),
        _restore_class_axis(variances, original_axis, was_2d),
    )


def _finite_percentile(values: FloatArray, percentile: int) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan
    return float(np.percentile(finite_values, percentile))


def local_variance(
    probabilities: ArrayLike,
    window_size: int = 7,
    neigh_fraction: float = 0.5,
    class_axis: int = 0,
) -> FloatArray:
    """Calculate per-class local logit variance from probability values."""
    logits = probs_to_logits(probabilities, class_axis=class_axis)
    _, variance = local_mean_variance(
        logits,
        window_size=window_size,
        neigh_fraction=neigh_fraction,
        class_axis=class_axis,
    )
    return variance


def bayesian_smooth(
    probabilities: ArrayLike,
    smoothness: float | Sequence[float] | Mapping[str, float] = 1.0,
    window_size: int = 7,
    neigh_fraction: float = 0.5,
    class_axis: int = 0,
    class_bands: Sequence[str] | None = None,
) -> FloatArray:
    """Apply empirical-Bayes spatial smoothing to class probabilities.

    2D input is treated as a single band. For 3D input, ``class_axis=0`` by
    default, matching arrays read by Rasterio. Non-finite input pixels remain
    non-finite in the smoothed result.
    """
    logits, original_axis, was_2d = _class_first(probabilities, class_axis)
    logits = _logits_from_class_first(logits)
    mean, variance = local_mean_variance(
        logits,
        window_size=window_size,
        neigh_fraction=neigh_fraction,
        class_axis=0,
    )
    sigma2 = _smoothness_by_class(smoothness, logits.shape[0], class_bands)
    denominator = sigma2 + variance
    posterior_logits = (
        logits * (variance / denominator)
        + mean * (sigma2 / denominator)
    )
    posterior = expit(posterior_logits)
    totals = np.sum(posterior, axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        posterior = posterior / totals
    posterior[:, ~np.all(np.isfinite(logits), axis=0)] = np.nan
    return _restore_class_axis(posterior, original_axis, was_2d)


def classify_from_probs(
    probabilities: ArrayLike,
    class_values: Sequence[int] | None = None,
    class_axis: int = 0,
) -> NDArray[np.int64]:
    """Return the highest-probability class value; all-invalid pixels are -1."""
    classes, _, _ = _class_first(probabilities, class_axis)
    valid_pixels = np.any(np.isfinite(classes), axis=0)
    safe_classes = np.where(np.isfinite(classes), classes, -np.inf)
    indices = np.argmax(safe_classes, axis=0)
    if class_values is not None:
        values = np.asarray(class_values)
        if values.ndim != 1 or values.size != classes.shape[0]:
            raise ValueError("class_values must match the class dimension")
        result = values[indices].astype(np.int64)
    else:
        result = indices.astype(np.int64)
    result[~valid_pixels] = -1
    return result