"""Create Euclidean distance rasters from binary rasters or vector features."""

from os import PathLike
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from scipy.ndimage import distance_transform_edt


def create_distance_features(
    source_path: str | PathLike[str],
    reference_raster: str | PathLike[str],
    output_path: str | PathLike[str],
    *,
    source_type: str = "auto",
    band: int = 1,
    target_value: float | None = None,
) -> Path:
    """Write distance to the nearest feature pixel as a single-band GeoTIFF.

    Parameters
    ----------
    source_path : path-like or str
        Binary raster or vector file containing the features. Raster cells are
        features when nonzero, unless ``target_value`` is supplied. Vector
        geometries are rasterized with all touched cells included.
    reference_raster : path-like or str
        Raster defining the output CRS, extent, resolution, and alignment.
    output_path : path-like or str
        Destination GeoTIFF path.
    source_type : {"auto", "raster", "vector"}, default="auto"
        Source format. Automatic detection treats ``.shp`` as vector and other
        extensions as raster.
    band : int, default=1
        One-based source band to use when ``source_type="raster"``.
    target_value : float, optional
        Exact raster value identifying feature cells. By default, all
        nonzero, finite, unmasked cells are features.

    Returns
    -------
    pathlib.Path
        Path to the output distance raster. Distances are measured between
        pixel centers in the reference CRS units.

    Raises
    ------
    ValueError
        If the source type, grid, or source data are invalid, or no feature
        cells occur within the reference extent.
    """
    source = Path(source_path)
    reference_path = Path(reference_raster)
    output = Path(output_path)
    if source_type == "auto":
        source_type = "vector" if source.suffix.lower() == ".shp" else "raster"
    if source_type not in {"raster", "vector"}:
        raise ValueError("source_type must be 'auto', 'raster', or 'vector'")
    if band < 1:
        raise ValueError("band must be at least 1")
    if output.resolve() in {source.resolve(), reference_path.resolve()}:
        raise ValueError("output_path must differ from source_path and reference_raster")

    with rasterio.open(reference_path) as reference:
        if reference.crs is None:
            raise ValueError(f"Reference raster has no CRS: {reference_path}")
        if not reference.crs.is_projected:
            raise ValueError("Reference raster must use a projected CRS for Euclidean distances")
        transform = reference.transform
        if not np.isclose(transform.b, 0) or not np.isclose(transform.d, 0):
            raise ValueError("Rotated reference rasters are not supported")
        xres, yres = abs(transform.a), abs(transform.e)
        if xres == 0 or yres == 0:
            raise ValueError("Reference raster pixel dimensions must be nonzero")
        width, height = reference.width, reference.height
        crs = reference.crs

        if source_type == "raster":
            with rasterio.open(source) as raster:
                if raster.crs is None:
                    raise ValueError(f"Source raster has no CRS: {source}")
                if band > raster.count:
                    raise ValueError(f"Source raster has no band {band}: {source}")
                with WarpedVRT(
                    raster,
                    crs=crs,
                    transform=transform,
                    width=width,
                    height=height,
                    resampling=Resampling.nearest,
                ) as aligned:
                    values = aligned.read(band, masked=True)
                    valid = ~np.ma.getmaskarray(values) & np.isfinite(values.data)
                    features = values.data == target_value if target_value is not None else values.data != 0
                    feature_mask = valid & features
        else:
            vector = gpd.read_file(source)
            if vector.crs is None:
                raise ValueError(f"Source vector file has no CRS: {source}")
            vector = vector.to_crs(crs)
            geometries = [geometry for geometry in vector.geometry if geometry is not None and not geometry.is_empty]
            feature_mask = np.asarray(
                rasterize(
                    ((geometry, 1) for geometry in geometries),
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    all_touched=True,
                    dtype="uint8",
                ),
                dtype=bool,
            )

    if not feature_mask.any():
        raise ValueError("No feature cells found within the reference raster extent")

    distances = np.asarray(
        distance_transform_edt(~feature_mask, sampling=(yres, xres)),
        dtype=np.float32,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
    }
    if width >= 16 and height >= 16:
        profile["tiled"] = True
    with rasterio.open(output, "w", **profile) as destination:
        destination.write(distances, 1)
        destination.set_band_description(1, "distance")
    return output