"""
Mosaic tiled rasters and stack aligned rasters into GeoTIFFs.
Peform only mosaicking and raster stacking to generate multiband raster.
"""

from contextlib import ExitStack
from os import PathLike
from pathlib import Path
from typing import Iterable, Sequence
import glob
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
RasterPath = str | PathLike[str]

#resolve paths issues 
#so that glob patterns and iterables of paths are handled consistently
def _resolve_paths(rasters: Iterable[RasterPath] | str | PathLike[str]) -> list[Path]:
    """Expand a glob or validate an iterable of raster paths."""
    if isinstance(rasters, (str, PathLike)):
        paths = sorted(Path(path) for path in glob.glob(str(rasters)))
    else:
        paths = [Path(path) for path in rasters]
    if not paths:
        raise ValueError("At least one input raster is required")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Raster does not exist: {missing[0]}")
    return paths
#perform mosaic of rasters
def mosaic_rasters(
    rasters: Iterable[RasterPath] | str | PathLike[str],
    output_path: RasterPath,
    *,
    method: str = "max",
    nodata: float | int | None = 0,
    compress: str = "lzw",
) -> Path:
    """Merge spatially overlapping or adjacent raster tiles into one GeoTIFF.

    ``rasters`` may be an iterable of paths or a glob pattern. Inputs are
    sorted before merging, and ``method='max'`` is useful for binary masks
    because a valid value overrides an overlapping zero.
    """
    paths = _resolve_paths(rasters)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(path)) for path in paths]
        mosaic, transform = merge(sources, method=method, nodata=nodata)
        profile = sources[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            count=mosaic.shape[0],
            transform=transform,
            nodata=nodata,
            compress=compress,
        )
        with rasterio.open(output, "w", **profile) as destination:
            destination.write(mosaic)
            for band in range(1, destination.count + 1):
                destination.set_band_description(band, sources[0].descriptions[band - 1] or f"band_{band}")
    return output
#perform stacking of raster data
def stack_rasters(
    rasters: Iterable[RasterPath] | str | PathLike[str],
    output_path: RasterPath,
    *,
    target_crs: str | None = None,
    resolution: float | None = None,
    resampling: Resampling = Resampling.bilinear,
    band_names: Sequence[str] | None = None,
    nodata: float | int | None = None,
    compress: str = "deflate",
) -> Path:
    """Align rasters to one grid and write their bands to a single GeoTIFF.

    Each input contributes all of its bands. The first raster supplies the
    default grid; ``target_crs`` and ``resolution`` can override it. Inputs
    may therefore differ in extent, resolution, CRS, and pixel alignment.
    """
    paths = _resolve_paths(rasters)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptions = []
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(path)) for path in paths]
        reference = sources[0]
        if reference.crs is None:
            raise ValueError(f"Raster has no CRS: {paths[0]}")
        crs = CRS.from_user_input(target_crs or reference.crs)
        if resolution is not None and crs.is_geographic:
            raise ValueError(
                "resolution is expressed in target CRS units. The target CRS "
                f"{crs} uses degrees; provide a projected target_crs for a "
                "meter resolution, or omit resolution to preserve the source grid."
            )
        pixel_size = resolution or max(abs(reference.transform.a), abs(reference.transform.e))
        if target_crs is None and resolution is None:
            transform, width, height = reference.transform, reference.width, reference.height
        else:
            from rasterio.warp import calculate_default_transform

            transform, width, height = calculate_default_transform(
                reference.crs, crs, reference.width, reference.height, *reference.bounds, resolution=pixel_size
            )
        band_count = sum(source.count for source in sources)
        for source in sources:
            descriptions.extend(description or f"band_{index}" for index, description in enumerate(source.descriptions, start=1))
        if band_names is not None:
            if len(band_names) != band_count or len(set(band_names)) != len(band_names):
                raise ValueError("band_names must contain one unique name per output band")
            descriptions = list(band_names)
        profile = sources[0].profile.copy()
        profile.update(driver="GTiff", height=height, width=width, count=band_count, crs=crs, transform=transform, nodata=nodata, compress=compress)
        with rasterio.open(output, "w", **profile) as destination:
            output_band = 1
            for source in sources:
                with WarpedVRT(source, crs=crs, transform=transform, width=width, height=height, resampling=resampling) as warped:
                    for source_band in range(1, source.count + 1):
                        destination.write(warped.read(source_band), output_band)
                        destination.set_band_description(output_band, descriptions[output_band - 1])
                        output_band += 1
    return output