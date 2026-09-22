"""
Outlier detection for labeled LULC training points.

Idea: for each land-cover class, fit an outlier/anomaly detector on the
spectral (+ index/topo) features of points belonging to that class only,
then flag points that look anomalous relative to their own class.
Output is a table of point_id + outlier flag + score, so flagged points
can be sent back for manual QA.

Expected input: a table (CSV/GeoJSON/DataFrame) with one row per point,
containing at least:
    - point_id      : unique identifier for the point
    - class_label   : the labeled land-cover class
    - band/index columns : e.g. B2, B3, B4, B8, NDVI, ...

Requires: pandas, numpy, scikit-learn (and geopandas if reading a
vector file with a geometry column).
"""

from os import PathLike

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
def load_points(path: str | PathLike[str]) -> pd.DataFrame:
    """Load training points into a DataFrame. Adjust for your file type."""
    path = str(path)
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        import geopandas as gpd
        df = gpd.read_file(path)
        df = df.drop(columns="geometry", errors="ignore")
    return df


# ---------------------------------------------------------------------
# 2. Per-class outlier detection
# ---------------------------------------------------------------------
def detect_outliers_per_class(
    df: pd.DataFrame,
    feature_cols: list[str],
    class_col: str = "class_label",
    id_col: str = "point_id",
    class_name_col: str | None = None,
    method: str = "isolation_forest",   # or "elliptic_envelope"
    contamination: float = 0.05,        # expected outlier fraction per class
    min_points_for_ee: int = 30,        # EllipticEnvelope needs a reasonable sample
) -> pd.DataFrame:
    """
    Runs outlier detection independently within each class.
    Returns a DataFrame with identifiers, class label/name, outlier (bool),
    and score.
    Higher score = more anomalous.
    """
    required_cols = [id_col, class_col, *feature_cols]
    if class_name_col is not None:
        required_cols.append(class_name_col)
    missing_cols = [column for column in required_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one feature")
    if method not in {"isolation_forest", "elliptic_envelope"}:
        raise ValueError(f"Unsupported outlier method: {method!r}")
    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be greater than 0 and at most 0.5")

    results = []

    for cls, group in df.groupby(class_col):
        X = group[feature_cols].to_numpy(dtype=float)
        n = len(group)

        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite feature values found in class {cls!r}")

        if n < 2:
            pred = np.ones(n, dtype=int)
            score = np.full(n, np.nan)
        else:
            X_scaled = StandardScaler().fit_transform(X)

            if method == "elliptic_envelope" and n >= min_points_for_ee:
                model = EllipticEnvelope(contamination=contamination, random_state=0)
                pred = model.fit_predict(X_scaled)
                score = model.mahalanobis(X_scaled)
            else:
                model = IsolationForest(
                    contamination=contamination, random_state=0, n_estimators=300
                )
                pred = model.fit_predict(X_scaled)
                score = -model.score_samples(X_scaled)

        out = pd.DataFrame({
            id_col: group[id_col].values,
            class_col: cls,
            "outlier": pred == -1,
            "score": score,
            "n_in_class": n,
        })
        if class_name_col is not None:
            out.insert(2, class_name_col, group[class_name_col].tolist())
        results.append(out)

    return pd.concat(results, ignore_index=True).sort_values(
        ["outlier", "score"], ascending=[False, False]
    )


def remove_outliers(
    df: pd.DataFrame,
    flags: pd.DataFrame,
    id_col: str = "point_id",
    outlier_col: str = "outlier",
) -> pd.DataFrame:
    """Return a copy of ``df`` with flagged samples removed.

    The original DataFrame and the detector result are left unchanged. Use the
    returned table as the cleaned training data after reviewing ``flags``.
    """
    for name, table, required in (
        ("training data", df, [id_col]),
        ("outlier flags", flags, [id_col, outlier_col]),
    ):
        missing = [column for column in required if column not in table.columns]
        if missing:
            raise ValueError(f"Missing columns in {name}: {missing}")

    flagged_ids = flags.loc[flags[outlier_col].astype(bool), id_col]
    return df.loc[~df[id_col].isin(flagged_ids)].copy().reset_index(drop=True)


def remove_outliers_from_vector(
    points: "object",
    flags: pd.DataFrame,
    id_col: str = "point_id",
    outlier_col: str = "outlier",
) -> "object":
    """Return a GeoDataFrame with flagged point geometries removed."""
    import geopandas as gpd

    if not isinstance(points, gpd.GeoDataFrame):
        raise TypeError("points must be a GeoDataFrame")
    return remove_outliers(
        points,
        flags,
        id_col=id_col,
        outlier_col=outlier_col,
    )


def save_cleaned_points(
    points_path: str | PathLike[str],
    flags: pd.DataFrame,
    output_path: str | PathLike[str],
    id_col: str = "point_id",
    outlier_col: str = "outlier",
    driver: str = "ESRI Shapefile",
) -> None:
    """Remove flagged points from a vector file and write the cleaned file."""
    import geopandas as gpd

    points = gpd.read_file(points_path)
    cleaned_points = remove_outliers_from_vector(
        points,
        flags,
        id_col=id_col,
        outlier_col=outlier_col,
    )
    cleaned_points.to_file(output_path, driver=driver)


def detect_point_outliers(
    raster_path: str | PathLike[str],
    points_path: str | PathLike[str],
    class_field: str,
    id_field: str = "point_id",
    class_name_field: str | None = "LULC_type",
    method: str = "isolation_forest",
    contamination: float = 0.05,
    min_points_for_ee: int = 30,
) -> pd.DataFrame:
    """Extract point features from a raster and flag class-wise outliers."""
    from .load_extract import load_and_extract_points

    samples = load_and_extract_points(
        raster_path=raster_path,
        points_path=points_path,
        class_field=class_field,
        id_field=id_field,
        class_name_field=class_name_field,
    )
    feature_cols = [
        column
        for column in samples.columns
        if column not in {id_field, class_field, class_name_field}
    ]
    return detect_outliers_per_class(
        samples,
        feature_cols=feature_cols,
        class_col=class_field,
        id_col=id_field,
        class_name_col=class_name_field,
        method=method,
        contamination=contamination,
        min_points_for_ee=min_points_for_ee,
    )


# ---------------------------------------------------------------------
# 3. Run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    INPUT_PATH = "training_points.csv"       # <- point this at your file
    FEATURE_COLS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI"]  # adjust to your bands/indices
    OUTPUT_PATH = "training_points_outlier_flags.csv"

    df = load_points(INPUT_PATH)

    flags = detect_outliers_per_class(
        df,
        feature_cols=FEATURE_COLS,
        class_col="class_label",
        id_col="point_id",
        method="isolation_forest",   # switch to "elliptic_envelope" if you prefer Mahalanobis distance
        contamination=0.05,          # flag ~5% of each class as candidate outliers
    )

    flags.to_csv(OUTPUT_PATH, index=False)

    n_flagged = flags["outlier"].sum()
    print(f"Flagged {n_flagged} of {len(flags)} points as candidate outliers.")
    print(flags[flags["outlier"]].groupby("class_label").size())