"""Spatial geometry, raster grid alignment, and geodesic area calculations."""

import math
from os import PathLike
from pathlib import Path
from typing import Sequence, Tuple
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

PathType = str | PathLike[str]


def compute_pixel_area_ha(
    crs: CRS | None,
    transform: Affine,
    height: int,
    row_indices: np.ndarray | None = None,
) -> float | np.ndarray:
    """Calculate pixel area in hectares, handling both projected and geographic CRSs.

    Parameters
    ----------
    crs : rasterio.crs.CRS or None
        Coordinate reference system of the raster.
    transform : rasterio.transform.Affine
        Affine geotransform.
    height : int
        Raster height in pixels.
    row_indices : numpy.ndarray of int, optional
        Specific row indices to compute area for. If omitted and the CRS is
        geographic, returns an array of shape ``(height,)`` with per-row areas.
        If projected, returns a scalar float.

    Returns
    -------
    float or numpy.ndarray
        Area per pixel in hectares.
    """
    res_x = abs(transform.a)
    res_y = abs(transform.e)

    if crs is not None and crs.is_geographic:
        # WGS-84 ellipsoid dimensions (meters)
        semi_major = 6378137.0
        eccentricity = 0.081819190842622

        if row_indices is None:
            rows = np.arange(height, dtype=np.float64)
        else:
            rows = np.asarray(row_indices, dtype=np.float64)

        # Center latitude for each row
        lat_deg = transform.f + (rows + 0.5) * transform.e
        lat_rad = np.radians(lat_deg)

        sin_lat = np.sin(lat_rad)
        denom = 1.0 - (eccentricity ** 2) * (sin_lat ** 2)
        sqrt_denom = np.sqrt(denom)

        # Meridional radius of curvature (North-South)
        m = (semi_major * (1.0 - eccentricity ** 2)) / (denom * sqrt_denom)
        # Prime vertical radius of curvature (East-West)
        n = semi_major / sqrt_denom

        delta_phi = np.radians(res_y)
        delta_lambda = np.radians(res_x)

        dy = m * delta_phi
        dx = n * np.cos(lat_rad) * delta_lambda

        # m² to hectares (1 ha = 10,000 m²)
        area_ha = (dx * dy) / 10000.0
        return area_ha if row_indices is not None or height > 1 else float(area_ha[0])

    # Projected CRS: resolution is already in linear units (typically meters)
    return float((res_x * res_y) / 10000.0)


def validate_raster_alignment(
    *raster_paths: PathType,
    tolerance: float = 1e-4,
) -> Tuple[int, int, CRS | None, Affine]:
    """Verify that multiple rasters share identical dimensions, CRS, and transform.

    Parameters
    ----------
    *raster_paths : PathType
        Paths to rasters to compare.
    tolerance : float
        Tolerance for comparing affine transform coefficients.

    Returns
    -------
    tuple of (height, width, crs, transform)
        The shared raster metadata.

    Raises
    ------
    ValueError
        If dimensions, CRS, or transform differ between rasters.
    """
    if not raster_paths:
        raise ValueError("At least one raster path must be provided")

    with rasterio.open(raster_paths[0]) as first:
        base_h = first.height
        base_w = first.width
        base_crs = first.crs
        base_transform = first.transform

    for path in raster_paths[1:]:
        with rasterio.open(path) as src:
            if src.height != base_h or src.width != base_w:
                raise ValueError(
                    f"Dimension mismatch for {path}: expected ({base_h}, {base_w}), "
                    f"got ({src.height}, {src.width})"
                )
            if src.crs != base_crs:
                raise ValueError(
                    f"CRS mismatch for {path}: expected {base_crs}, got {src.crs}"
                )
            for i, (a, b) in enumerate(zip(base_transform, src.transform)):
                if not math.isclose(a, b, abs_tol=tolerance):
                    raise ValueError(
                        f"Transform mismatch for {path}: coeff {i} expected {a}, got {b}"
                    )

    return base_h, base_w, base_crs, base_transform
