"""Build local feature-stack GeoTIFFs for the land-cover classifier.

The public function aligns optical, radar, DEM, and other local raster inputs
onto one grid, optionally composites repeated scenes, derives supported spectral
or terrain features, and writes a float32 multiband GeoTIFF. Its JSON manifest
records the resulting feature order for use during model training and prediction.
"""

from dataclasses import dataclass
import json
from os import PathLike
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import array_bounds, from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from .planetary_computer import DownloadedAsset


@dataclass(frozen=True)
class FeatureStackResult:
    """Paths and ordered feature names produced by :func:`create_feature_stack`.

    Attributes
    ----------
    stack_path : pathlib.Path
        Local multiband GeoTIFF compatible with the current classification
        pipeline.
    manifest_path : pathlib.Path
        JSON manifest recording feature order, processing settings, and source
        files.
    feature_names : tuple of str
        Feature names in exact GeoTIFF band order.
    """

    stack_path: Path
    manifest_path: Path
    feature_names: tuple[str, ...]


def _normalise_inputs(
    imagery: Iterable[DownloadedAsset | str | PathLike[str]],
    additional_layers: Iterable[str | PathLike[str]],
    aliases: Mapping[str, str],
) -> tuple[dict[str, list[Path]], list[dict[str, str]]]:
    """Group input rasters by output feature name and collect provenance.

    Parameters
    ----------
    imagery : iterable
        Downloaded STAC assets or local raster paths.
    additional_layers : iterable
        Extra local raster paths.
    aliases : mapping of str to str
        Mapping from source asset or file-stem names to output feature names.

    Returns
    -------
    tuple
        Feature-name-to-path groups and serializable source metadata.

    Raises
    ------
    ValueError
        If no raster inputs are supplied.
    """
    grouped: dict[str, list[Path]] = {}
    sources: list[dict[str, str]] = []
    for entry in imagery:
        if isinstance(entry, DownloadedAsset):
            name = aliases.get(entry.asset_key, entry.asset_key)
            path = entry.path
            sources.append({"path": str(path), "collection": entry.collection, "item_id": entry.item_id, "asset": entry.asset_key})
        else:
            path = Path(entry)
            name = aliases.get(path.stem, path.stem)
            sources.append({"path": str(path), "source": "local"})
        grouped.setdefault(name, []).append(path)
    for layer in additional_layers:
        path = Path(layer)
        name = aliases.get(path.stem, path.stem)
        grouped.setdefault(name, []).append(path)
        sources.append({"path": str(path), "source": "local"})
    if not grouped:
        raise ValueError("Provide downloaded imagery and/or additional_layers")
    return grouped, sources


def _grid(path: Path, target_crs: str | None, resolution: float | None) -> tuple[object, object, int, int]:
    """Select the output grid from the first input raster and user overrides.

    Parameters
    ----------
    path : pathlib.Path
        First input raster, which defines the default extent and grid.
    target_crs : str or None
        Requested output CRS, or ``None`` to retain the source CRS.
    resolution : float or None
        Requested output pixel size, or ``None`` to retain source resolution.

    Returns
    -------
    tuple
        Output CRS, affine transform, width, and height.

    Raises
    ------
    ValueError
        If the reference raster has no CRS.
    """
    with rasterio.open(path) as source:
        if source.crs is None:
            raise ValueError(f"Raster has no CRS: {path}")
        crs = target_crs or source.crs
        if resolution is not None and crs.is_geographic:
            raise ValueError(
                "resolution is expressed in target CRS units. The target CRS "
                f"{crs} uses degrees; provide a projected target_crs for a "
                "meter resolution, or omit resolution to preserve the source grid."
            )
        if str(crs) == str(source.crs) and resolution is None:
            return crs, source.transform, source.width, source.height
        west, south, east, north = array_bounds(source.height, source.width, source.transform)
        if str(crs) != str(source.crs):
            west, south, east, north = transform_bounds(source.crs, crs, west, south, east, north)
        pixel_size = resolution or max(abs(source.transform.a), abs(source.transform.e))
        width = max(1, int(np.ceil((east - west) / pixel_size)))
        height = max(1, int(np.ceil((north - south) / pixel_size)))
        return crs, from_bounds(west, south, east, north, width, height), width, height


def _composite(paths: Sequence[Path], window: Window, *, crs: object, transform: object, width: int, height: int, method: str) -> np.ndarray:
    """Read, align, and composite one source feature for a raster window.

    Parameters
    ----------
    paths : sequence of pathlib.Path
        One or more raster files representing the same source feature.
    window : rasterio.windows.Window
        Output-grid window to calculate.
    crs, transform, width, height
        Definition of the common output grid.
    method : {"median", "first_valid"}
        Temporal compositing method.

    Returns
    -------
    numpy.ndarray
        Float32 array with NaN for pixels invalid in every input.
    """
    values: list[np.ndarray] = []
    for path in paths:
        with rasterio.open(path) as source:
            with WarpedVRT(source, crs=crs, transform=transform, width=width, height=height, resampling=Resampling.bilinear) as warped:
                data = warped.read(1, window=window, masked=True).astype("float32")
                values.append(np.ma.filled(data, np.nan))
    data = np.stack(values)
    if method == "median":
        return np.nanmedian(data, axis=0)
    if method == "first_valid":
        valid = ~np.isnan(data)
        index = valid.argmax(axis=0)
        result = np.take_along_axis(data, index[None, ...], axis=0)[0]
        return np.where(valid.any(axis=0), result, np.nan)
    raise ValueError("composite must be 'median' or 'first_valid'")


def _index(name: str, layers: Mapping[str, np.ndarray]) -> np.ndarray:
    """Calculate one named spectral index from aligned source-band arrays.

    Parameters
    ----------
    name : str
        Uppercase index name supported by :func:`create_feature_stack`.
    layers : mapping of str to numpy.ndarray
        Feature arrays keyed by Sentinel-2 or Landsat canonical band names.

    Returns
    -------
    numpy.ndarray
        Derived index values. Division-by-zero pixels are NaN.

    Raises
    ------
    ValueError
        If the index is unsupported or required source bands are absent.
    """
    aliases = {"blue": ("B02", "blue"), "green": ("B03", "green"), "red": ("B04", "red"), "nir": ("B08", "nir08"), "swir1": ("B11", "swir16"), "swir2": ("B12", "swir22")}
    def band(key: str) -> np.ndarray:
        for candidate in aliases[key]:
            if candidate in layers:
                return layers[candidate]
        raise ValueError(f"{name} requires {key}; supply one of {aliases[key]}")
    with np.errstate(divide="ignore", invalid="ignore"):
        if name == "NDVI":
            nir, red = band("nir"), band("red")
            return (nir - red) / (nir + red)
        if name == "NDWI":
            green, nir = band("green"), band("nir")
            return (green - nir) / (green + nir)
        if name == "MNDWI":
            green, swir1 = band("green"), band("swir1")
            return (green - swir1) / (green + swir1)
        if name == "NDBI":
            swir1, nir = band("swir1"), band("nir")
            return (swir1 - nir) / (swir1 + nir)
        if name == "EVI":
            nir, red, blue = band("nir"), band("red"), band("blue")
            return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
        if name == "SAVI":
            nir, red = band("nir"), band("red")
            return 1.5 * (nir - red) / (nir + red + 0.5)
    raise ValueError("indices must use: NDVI, NDWI, MNDWI, NDBI, EVI, or SAVI")


def _terrain(dem: np.ndarray, xres: float, yres: float) -> tuple[np.ndarray, np.ndarray]:
    """Calculate slope and aspect from one aligned DEM window.

    Parameters
    ----------
    dem : numpy.ndarray
        Elevation values for a raster window.
    xres, yres : float
        Pixel dimensions in map units.

    Returns
    -------
    tuple of numpy.ndarray
        Slope in degrees and clockwise aspect in degrees from north.
    """
    # Windows on a raster edge may be one pixel wide/high. A zero derivative is
    # the only meaningful finite estimate available in that direction.
    dy = np.gradient(dem, yres, axis=0) if dem.shape[0] > 1 else np.zeros_like(dem)
    dx = np.gradient(dem, xres, axis=1) if dem.shape[1] > 1 else np.zeros_like(dem)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    aspect = (np.degrees(np.arctan2(-dx, dy)) + 360) % 360
    return slope, aspect


def create_feature_stack(
    imagery: Iterable[DownloadedAsset | str | PathLike[str]],
    output_path: str | PathLike[str],
    *,
    composite: str = "median",
    indices: Sequence[str] = (),
    additional_layers: Iterable[str | PathLike[str]] = (),
    asset_aliases: Mapping[str, str] | None = None,
    terrain_from: Sequence[str] = (),
    target_crs: str | None = None,
    resolution: float | None = None,
    block_size: int = 512,
) -> FeatureStackResult:
    """Create a multiband GeoTIFF from imagery, DEMs, and derived features.

    Parameters
    ----------
    imagery : iterable of DownloadedAsset or path-like
        Downloaded STAC assets from :func:`download_imagery`, or local raster
        paths. Repeated ``DownloadedAsset`` values with the same asset key are
        temporally composited into one output feature.
    output_path : path-like or str
        Output GeoTIFF path. A JSON manifest is written beside it using the
        name ``<output_path>.json``.
    composite : {"median", "first_valid"}, default="median"
        Method for combining repeated observations of the same asset.
    indices : sequence of str, optional
        Named spectral indices to add. Supported values are ``"NDVI"``,
        ``"NDWI"``, ``"MNDWI"``, ``"NDBI"``, ``"EVI"``, and ``"SAVI"``.
        Required Sentinel-2 or Landsat source bands must also be inputs.
    additional_layers : iterable of path-like, optional
        Extra local single-band rasters, for example precomputed slope, a soil
        map, or a DEM. They are aligned and added as feature bands.
    asset_aliases : mapping of str to str, optional
        Output names for input asset keys. For example, use
        ``{"data": "dem"}`` for a STAC DEM asset named ``"data"``.
    terrain_from : sequence of str, optional
        Names of DEM features from which to calculate and append slope and
        aspect. Names refer to the aliases after ``asset_aliases`` is applied.
    target_crs : str, optional
        Output CRS. When omitted, the first input raster CRS is used.
    resolution : float, optional
        Output pixel size in target-CRS units. When omitted, the first input
        raster resolution is used.
    block_size : int, default=512
        Width and height of processing windows in pixels.

    Returns
    -------
    FeatureStackResult
        GeoTIFF path, JSON manifest path, and the exact ordered feature names.

    Raises
    ------
    ValueError
        If no inputs are supplied, feature names collide, a requested index
        lacks its required bands, a terrain source is missing, or processing
        settings are invalid.

    Notes
    -----
    All output bands are float32 with NaN nodata. Preserve the JSON manifest
    with the fitted model because its feature order must match the raster used
    later by :func:`ML_LC_Classifier.classify_raster`.
    """
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    method = composite.lower()
    if method not in {"median", "first_valid"}:
        raise ValueError("composite must be 'median' or 'first_valid'")
    aliases = dict(asset_aliases or {})
    grouped, sources = _normalise_inputs(imagery, additional_layers, aliases)
    first_path = next(iter(grouped.values()))[0]
    crs, transform, width, height = _grid(first_path, target_crs, resolution)
    feature_names = list(grouped)
    feature_names.extend(index.upper() for index in indices)
    for dem_name in terrain_from:
        if dem_name not in grouped:
            raise ValueError(f"terrain source {dem_name!r} is not an input feature")
        feature_names.extend([f"{dem_name}_slope", f"{dem_name}_aspect"])
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Feature names must be unique; adjust asset_aliases or indices")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "height": height, "width": width, "count": len(feature_names), "dtype": "float32", "crs": crs, "transform": transform, "nodata": np.nan, "compress": "deflate"}
    if width >= 16 and height >= 16:
        profile["tiled"] = True
    with rasterio.open(output, "w", **profile) as destination:
        for number, name in enumerate(feature_names, start=1):
            destination.set_band_description(number, name)
        for row in range(0, height, block_size):
            for col in range(0, width, block_size):
                window = Window(col, row, min(block_size, width - col), min(block_size, height - row)) # type: ignore
                # Read a one-pixel halo so terrain derivatives are continuous at
                # block boundaries; crop it before writing the requested window.
                halo_col = max(0, col - 1)
                halo_row = max(0, row - 1)
                halo = Window(
                    halo_col, # type: ignore
                    halo_row,
                    min(width - halo_col, int(window.width) + (col - halo_col) + 1),
                    min(height - halo_row, int(window.height) + (row - halo_row) + 1),
                )
                row_offset = row - halo_row
                col_offset = col - halo_col
                crop = (slice(row_offset, row_offset + int(window.height)), slice(col_offset, col_offset + int(window.width)))
                layer_values = {name: _composite(paths, halo, crs=crs, transform=transform, width=width, height=height, method=method) for name, paths in grouped.items()}
                derived = [_index(index.upper(), layer_values) for index in indices]
                terrain_values: list[np.ndarray] = []
                xres, yres = abs(transform.a), abs(transform.e)  # type: ignore[attr-defined]
                for dem_name in terrain_from:
                    terrain_values.extend(_terrain(layer_values[dem_name], xres, yres))
                for band, values in enumerate([*layer_values.values(), *derived, *terrain_values], start=1):
                    destination.write(values[crop].astype("float32"), band, window=window)

    manifest = output.with_suffix(output.suffix + ".json")
    manifest.write_text(json.dumps({"stack_path": str(output), "feature_names": feature_names, "composite": method, "crs": str(crs), "resolution": resolution, "sources": sources}, indent=2), encoding="utf-8")
    return FeatureStackResult(output, manifest, tuple(feature_names))
