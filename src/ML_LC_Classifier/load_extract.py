"""
Feature Extraction Module
This module provides functions to load raster and vector data, extract pixel values from polygons, clean the data, 
and split it into training and testing sets. It is designed for use in land cover classification tasks.

"""

#required Library
from os import PathLike
from typing import Any, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from sklearn.model_selection import StratifiedShuffleSplit


Path = str | PathLike[str]
#load raster file, just define the path
def open_raster(raster_path: Path):
    """Open a raster with rasterio and return the dataset handle (caller keeps it open)."""
    return rasterio.open(raster_path)

#load the trining samples, define the path of the shapefile
def load_training_samples(shapefile_path: Path) -> "gpd.GeoDataFrame":
    """Load a training-sample shapefile (or any OGR vector format) as a GeoDataFrame."""
    return gpd.read_file(shapefile_path)


def _raster_feature_names(raster: Any) -> list[str]:
    """Return stable column names for the bands in an open raster."""
    names = []
    for index, description in enumerate(raster.descriptions, start=1):
        names.append(description or f"band_{index}")
    if len(set(names)) != len(names):
        raise ValueError("Raster band descriptions must be unique")
    return names


def extract_point_features(
    points: "gpd.GeoDataFrame",
    raster: Any,
    class_field: str,
    id_field: str = "point_id",
    class_name_field: str | None = None,
) -> pd.DataFrame:
    """Extract one raster feature row for each labelled point.

    The returned DataFrame contains ``id_field``, ``class_field``, and one
    column per raster band. Points outside the raster or containing invalid
    raster values are omitted.
    """
    required_fields = [class_field, id_field]
    if class_name_field is not None:
        required_fields.append(class_name_field)
    for field in required_fields:
        if field not in points.columns:
            raise ValueError(f"Field {field!r} was not found in the training data")

    feature_names = _raster_feature_names(raster)
    output_columns = [id_field, class_field]
    if class_name_field is not None:
        output_columns.append(class_name_field)
    output_columns.extend(feature_names)
    rows = []
    for _, row in points.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        if geometry.geom_type != "Point":
            raise ValueError("Point-level extraction requires Point geometries")
        if pd.isna(row[class_field]) or (
            class_name_field is not None and pd.isna(row[class_name_field])
        ):
            continue

        values = next(raster.sample([(geometry.x, geometry.y)], masked=True))
        values = np.ma.asarray(values)
        if np.ma.getmaskarray(values).any() or not np.isfinite(values.data).all():
            continue

        point_values = {
            id_field: row[id_field],
            class_field: row[class_field],
            **dict(zip(feature_names, values.data.tolist())),
        }
        if class_name_field is not None:
            point_values[class_name_field] = row[class_name_field]
        rows.append(point_values)

    return pd.DataFrame(rows, columns=output_columns)

#xtract the raster feature from shapefile
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

#drop invalid samples
def drop_nan_samples(features: np.ndarray, labels: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Remove samples with non-finite features or missing labels."""
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features must be 2-D and labels 1-D with matching lengths")

    valid_features = np.isfinite(features).all(axis=1)
    valid_labels = ~pd.isna(labels)
    valid_samples = valid_features & valid_labels
    return features[valid_samples], labels[valid_samples]

#sample distrnbution of the samples
def class_distribution(labels: np.ndarray, name: str = "class") -> pd.DataFrame:
    """Simple value_counts table, handy for sanity-checking sample balance."""
    return pd.DataFrame({name: labels})[name].value_counts().to_frame()

#wrapper function to execute the aforementioned functions
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


def load_and_extract_points(
    raster_path: Path,
    points_path: Path,
    class_field: str,
    id_field: str = "point_id",
    class_name_field: str | None = None,
) -> pd.DataFrame:
    """Load point labels and raster values as a tidy outlier-detection table."""
    with open_raster(raster_path) as dataset:
        points = load_training_samples(points_path)
        if points.crs is not None and dataset.crs is not None and points.crs != dataset.crs:
            points = points.to_crs(dataset.crs)
        return extract_point_features(
            points,
            dataset,
            class_field=class_field,
            id_field=id_field,
            class_name_field=class_name_field,
        )

#perform train and test split on the extracted samples
#use stratified shuffle split to ensure class distribution is preserved
def train_test_split_data(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create one reproducible stratified shuffle train/test split."""
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features must be 2-D and labels 1-D with matching lengths")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices, test_indices = next(splitter.split(features, labels))
    x_train = features[train_indices]
    x_test = features[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
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