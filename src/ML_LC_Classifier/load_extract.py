"""Load raster training data, extract labeled pixels, and split the samples."""

from os import PathLike
from typing import Any, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from sklearn.model_selection import train_test_split


Path = str | PathLike[str]

def open_raster(raster_path: Path):
    """Open a raster with rasterio and return the dataset handle (caller keeps it open)."""
    return rasterio.open(raster_path)


def load_training_samples(shapefile_path: Path) -> "gpd.GeoDataFrame":
    """Load a training-sample shapefile (or any OGR vector format) as a GeoDataFrame."""
    return gpd.read_file(shapefile_path)


def extract_pixels_from_shapefile(
    shapefile: "gpd.GeoDataFrame",
    raster: Any,
    class_field: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract per-pixel band values and class labels for every polygon in
    `shapefile`, masked against `raster`.

    Parameters
    ----------
    shapefile : GeoDataFrame with polygon geometries and a class-label column.
    raster : an open rasterio dataset.
    class_field : name of the column in `shapefile` holding the class id/label
        (e.g. 'LUCID', 'New_ID', 'CLASS_NAME').

    Returns
    -------
    features : ndarray of shape (n_pixels, n_bands)
    labels   : ndarray of shape (n_pixels,)
    """
    if class_field not in shapefile.columns:
        raise ValueError(f"Class field {class_field!r} was not found in the training data")

    training_samples = []
    for _, row in shapefile.iterrows():
        geometry = row["geometry"]
        if geometry is None or geometry.is_empty:
            continue

        id_class = row[class_field]

        out_image, _ = rio_mask(raster, [geometry], crop=True, filled=False)
        pixels = np.ma.asarray(out_image).reshape(raster.count, -1).T
        pixel_values = pixels.data
        invalid_pixels = np.ma.getmaskarray(pixels).any(axis=1)
        invalid_pixels |= ~np.isfinite(pixel_values).all(axis=1)

        for pixel in pixel_values[~invalid_pixels]:
            training_samples.append((pixel, id_class))

    if not training_samples:
        return np.empty((0, raster.count)), np.empty((0,))

    features = np.asarray([sample[0] for sample in training_samples])
    labels = np.asarray([sample[1] for sample in training_samples])
    return features, labels


def drop_nan_samples(features: np.ndarray, labels: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Remove samples with non-finite features or missing labels."""
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features must be 2-D and labels 1-D with matching lengths")

    valid_features = np.isfinite(features).all(axis=1)
    valid_labels = ~pd.isna(labels)
    valid_samples = valid_features & valid_labels
    return features[valid_samples], labels[valid_samples]


def class_distribution(labels: np.ndarray, name: str = "class") -> pd.DataFrame:
    """Simple value_counts table, handy for sanity-checking sample balance."""
    return pd.DataFrame({name: labels})[name].value_counts().to_frame()


def load_and_extract(
    raster_path: Path,
    shapefile_path: Path,
    class_field: str,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: open raster + shapefile, extract pixels, drop NaNs.
    Returns (open_raster_dataset, features, labels). Caller is responsible for
    closing the raster dataset when done (or use it as a context manager).
    """
    dataset = open_raster(raster_path)
    samples = load_training_samples(shapefile_path)
    if samples.crs is not None and dataset.crs is not None and samples.crs != dataset.crs:
        samples = samples.to_crs(dataset.crs)

    features, labels = extract_pixels_from_shapefile(samples, dataset, class_field)
    features, labels = drop_nan_samples(features, labels)
    return dataset, features, labels

def train_test_split_data(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split (mirrors the single-model notebook)."""
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    return x_train, x_test, y_train, y_test


def load_and_split_training_data(
    raster_path: Path,
    shapefile_path: Path,
    class_field: str,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, extract, clean, and stratify-split polygon-labeled raster pixels.

    The returned tuple is ``(X_train, X_test, y_train, y_test)``. The raster
    file is closed before this function returns, so it is safe to call from a
    notebook without managing a rasterio dataset handle.
    """
    dataset, features, labels = load_and_extract(
        raster_path=raster_path,
        shapefile_path=shapefile_path,
        class_field=class_field,
    )
    try:
        return train_test_split_data(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
        )
    finally:
        dataset.close()