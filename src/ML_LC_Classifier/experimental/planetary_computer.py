"""Find, preview, and download imagery from Microsoft Planetary Computer.

This module is optional. Install its dependencies with
``pip install -e \".[planetary-computer]\"``. It does not change the local
raster-based training and classification workflow in the rest of the package.

Functions in this module return lightweight scene records first, so users can
inspect available imagery before downloading potentially large files.
"""

from dataclasses import dataclass
from datetime import date
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import urlretrieve

import geopandas as gpd


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


@dataclass(frozen=True)
class ImagerySearchResult:
    """A Planetary Computer scene selected by :func:`search_imagery`.

    Attributes
    ----------
    collection : str
        STAC collection containing the scene.
    item_id : str
        Unique STAC item identifier.
    datetime : str or None
        Scene acquisition date and time in ISO 8601 format, when supplied by
        the collection.
    bbox : tuple of float or None
        WGS84 scene extent as ``(west, south, east, north)``.
    cloud_cover : float or None
        Reported percentage cloud cover, when the collection supplies it.
    available_assets : tuple of str
        Asset keys that can be previewed or downloaded for this scene.
    item : object
        Signed PySTAC item used internally by :func:`download_imagery`.
    """

    collection: str
    item_id: str
    datetime: str | None
    bbox: tuple[float, float, float, float] | None
    cloud_cover: float | None
    available_assets: tuple[str, ...]
    item: Any


@dataclass(frozen=True)
class DownloadedAsset:
    """One locally downloaded STAC asset and its source identifiers.

    Attributes
    ----------
    path : pathlib.Path
        Local path to the downloaded raster asset.
    collection : str
        Source STAC collection.
    item_id : str
        Source STAC item identifier.
    asset_key : str
        Asset key used by the source collection, such as ``"B04"`` or
        ``"vv"``.
    """

    path: Path
    collection: str
    item_id: str
    asset_key: str


AOI = Mapping[str, Any] | Sequence[float] | str | PathLike[str]


def _optional_clients() -> tuple[Any, Any]:
    """Import optional Planetary Computer client libraries on demand.

    Returns
    -------
    tuple
        Imported ``planetary_computer`` module and PySTAC ``Client`` class.

    Raises
    ------
    ImportError
        If optional remote-imagery dependencies are unavailable.
    """
    try:
        import planetary_computer
        from pystac_client import Client
    except ImportError as error:
        raise ImportError(
            "Planetary Computer support requires optional dependencies. Install with "
            "pip install -e \".[planetary-computer]\""
        ) from error
    return planetary_computer, Client


def _aoi_search_arguments(aoi: AOI) -> dict[str, Any]:
    """Convert a user area of interest to STAC search arguments.

    Parameters
    ----------
    aoi : mapping, sequence of float, path-like, or str
        GeoJSON geometry or Feature, vector-file path, or WGS84 bounding box.

    Returns
    -------
    dict
        A dictionary containing either the STAC ``intersects`` geometry or
        ``bbox`` parameter.

    Raises
    ------
    ValueError
        If a file has no CRS or geometries, or an AOI has an invalid shape.
    """
    if isinstance(aoi, (str, PathLike)):
        data = gpd.read_file(aoi)
        if data.empty:
            raise ValueError("AOI file contains no geometries")
        if data.crs is None:
            raise ValueError("AOI file must define a CRS")
        geometries = data.to_crs("EPSG:4326").geometry
        union = geometries.union_all() if hasattr(geometries, "union_all") else geometries.unary_union
        geometry = union.__geo_interface__
        return {"intersects": geometry}
    if isinstance(aoi, Mapping):
        geometry = aoi.get("geometry", aoi)
        if geometry.get("type") == "Feature":
            geometry = geometry.get("geometry")
        if not isinstance(geometry, Mapping) or "type" not in geometry:
            raise ValueError("AOI mapping must be a GeoJSON geometry or Feature")
        return {"intersects": dict(geometry)}
    if len(aoi) != 4:
        raise ValueError("AOI bounding box must contain west, south, east, north")
    return {"bbox": list(aoi)}


def search_imagery(
    aoi: AOI,
    collection: str,
    start_date: str | date,
    end_date: str | date,
    *,
    max_cloud_cover: float | None = None,
    max_items: int = 20,
    query: Mapping[str, Any] | None = None,
) -> list[ImagerySearchResult]:
    """Find Planetary Computer scenes intersecting an area and date range.

    Parameters
    ----------
    aoi : mapping, sequence of float, path-like, or str
        Area of interest. Supply a GeoJSON geometry or Feature, a vector-file
        path, or a WGS84 bounding box ``[west, south, east, north]``. Vector
        files are reprojected to WGS84 before searching.
    collection : str
        Planetary Computer collection identifier, for example
        ``"sentinel-2-l2a"``, ``"sentinel-1-rtc"``, ``"landsat-c2-l2"``,
        or ``"cop-dem-glo-30"``.
    start_date, end_date : str or datetime.date
        Inclusive search date range in ISO 8601 format or as ``date`` objects.
    max_cloud_cover : float, optional
        Maximum ``eo:cloud_cover`` percentage. Use this for optical imagery;
        omit it for radar and DEM collections.
    max_items : int, default=20
        Maximum number of matching scenes to return.
    query : mapping, optional
        Additional collection-specific STAC query filters.

    Returns
    -------
    list of ImagerySearchResult
        Matching scenes with their source metadata and available asset keys.

    Raises
    ------
    ImportError
        If optional Planetary Computer dependencies are not installed.
    ValueError
        If the AOI, cloud-cover limit, or item limit is invalid.
    """
    if max_items < 1:
        raise ValueError("max_items must be at least 1")
    if max_cloud_cover is not None and not 0 <= max_cloud_cover <= 100:
        raise ValueError("max_cloud_cover must be between 0 and 100")

    planetary_computer, Client = _optional_clients()
    filters = dict(query or {})
    if max_cloud_cover is not None:
        filters["eo:cloud_cover"] = {"lte": max_cloud_cover}
    catalog = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[collection],
        datetime=f"{start_date}/{end_date}",
        query=filters or None,
        max_items=max_items,
        **_aoi_search_arguments(aoi),
    )
    results: list[ImagerySearchResult] = []
    for item in search.item_collection():
        bbox = tuple(item.bbox) if item.bbox is not None else None
        results.append(
            ImagerySearchResult(
                collection=item.collection_id or collection,
                item_id=item.id,
                datetime=item.datetime.isoformat() if item.datetime else None,
                bbox=bbox,  # type: ignore[arg-type]
                cloud_cover=item.properties.get("eo:cloud_cover"),
                available_assets=tuple(sorted(item.assets)),
                item=item,
            )
        )
    return results


def preview_imagery(
    scene: ImagerySearchResult,
    *,
    asset: str = "visual",
) -> str:
    """Return a PNG preview URL for one asset in a selected scene.

    Parameters
    ----------
    scene : ImagerySearchResult
        Scene returned by :func:`search_imagery`.
    asset : str, default="visual"
        Scene asset to render. Optical scenes commonly provide ``"visual"``;
        inspect ``scene.available_assets`` for other collections.

    Returns
    -------
    str
        Planetary Computer Data API URL that can be displayed in a notebook or
        opened in a browser without downloading the original GeoTIFF.

    Raises
    ------
    ValueError
        If the requested asset is absent from the scene.
    """
    if asset not in scene.available_assets:
        raise ValueError(f"Asset {asset!r} is not available on scene {scene.item_id!r}")
    parameters = urlencode({"collection": scene.collection, "item": scene.item_id, "assets": asset})
    return f"https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png?{parameters}"


def download_imagery(
    scenes: Sequence[ImagerySearchResult],
    assets: Sequence[str],
    output_dir: str | PathLike[str],
    *,
    overwrite: bool = False,
) -> list[DownloadedAsset]:
    """Download selected signed STAC assets to a local directory.

    Parameters
    ----------
    scenes : sequence of ImagerySearchResult
        Scenes returned by :func:`search_imagery`.
    assets : sequence of str
        Asset keys to download from every scene. Examples include ``"B04"``
        for Sentinel-2, ``"vv"`` for Sentinel-1 RTC, and ``"red"`` for
        Landsat.
    output_dir : path-like or str
        Directory where downloaded GeoTIFFs will be written.
    overwrite : bool, default=False
        If ``True``, download again when a matching local file already exists.

    Returns
    -------
    list of DownloadedAsset
        Local files with the STAC identifiers required for provenance records.

    Raises
    ------
    ValueError
        If no scenes or assets are supplied, or a requested asset is missing.

    Notes
    -----
    Signed URLs are temporary credentials. They are used for the transfer but
    are never stored in the returned records or feature-stack manifest.
    """
    if not scenes:
        raise ValueError("At least one scene is required")
    if not assets:
        raise ValueError("At least one asset is required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    downloads: list[DownloadedAsset] = []
    for scene in scenes:
        for asset_key in assets:
            try:
                href = scene.item.assets[asset_key].href
            except KeyError as error:
                raise ValueError(
                    f"Asset {asset_key!r} is not available on scene {scene.item_id!r}"
                ) from error
            path = destination / f"{scene.item_id}_{asset_key}.tif"
            if overwrite or not path.exists():
                urlretrieve(href, path)
            downloads.append(DownloadedAsset(path, scene.collection, scene.item_id, asset_key))
    return downloads
