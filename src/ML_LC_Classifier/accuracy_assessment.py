"""Thematic accuracy assessment and error adjusted area estimation.

Implements:
1. Sample-based and area-weighted confusion matrices 
2. Overall Accuracy (OA), User's Accuracy (UA / Precision), Producer's Accuracy
   (PA / Recall), Specificity, Omission & Commission Errors, with Standard Errors (SE)
   and 95% Confidence Intervals.
3. Class-imbalance metrics:
   - Per-class & Macro/Weighted/Micro F1-Score
   - Per-class & Macro/Weighted G-Mean (sqrt(Sensitivity * Specificity))
   - Multi-class Geometric Mean of Recalls ((prod PA_j)^(1/K))
   - Fowlkes-Mallows Index (sqrt(UA * PA))
   - Balanced Accuracy, Cohen's Kappa, and Matthews Correlation Coefficient (MCC)
4. Stratified / Post-Stratified Error-Adjusted Area Estimation (Olofsson et al., 2014;
   Stehman, 2014; QGIS AcATaMa plugin):
   - Area proportions matrix (p_ij = W_i * n_ij / n_i.)
   - Error-adjusted area (ha and km2) per class
   - Standard Error (SE), 95% Confidence Interval (CI), and Coefficient of Variation (CV)
5. Multi-temporal CCDC stability filtering support (e.g. stable_2020 == 1,
   stable_2015 == 1, stable_2005 == 1) 
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from os import PathLike
from pathlib import Path
from typing import Any
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask as rio_mask
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

from .trajectory_analysis import DEFAULT_TIMOR_18_CLASSES

PathType = str | PathLike[str]

DEFAULT_TIMOR_17_SCHEMA: dict[int, str] = {
    k: v for k, v in DEFAULT_TIMOR_18_CLASSES.items() if k != 18
}


@dataclass
class ThematicAccuracyResult:
    """Container for thematic accuracy assessment and Olofsson/AcATaMa area outputs.

    Attributes
    ----------
    overall_metrics : dict[str, float]
        Dictionary of global accuracy & imbalance metrics:
        - ``overall_accuracy_sample`` and ``overall_accuracy_sample_se`` / ``ci95``
        - ``overall_accuracy_area_weighted`` and ``overall_accuracy_area_se`` / ``ci95``
          (Olofsson et al. 2014 Eq. 4 & 5, when raster / mapped pixel counts are provided)
        - ``balanced_accuracy``
        - ``f1_macro``, ``f1_weighted``, ``f1_micro``
        - ``g_mean_macro``, ``g_mean_weighted``, ``g_mean_multiclass``
        - ``fowlkes_mallows_macro``, ``fowlkes_mallows_weighted``
        - ``kappa``, ``mcc``
        - ``total_samples``, ``num_classes``
    class_metrics : pd.DataFrame
        Per-class accuracy, imbalance, and (when mapped counts are available)
        AcATaMa / Olofsson error-adjusted area table with SE, 95% CI, and CV.
    confusion_matrix_counts : pd.DataFrame
        Sample error matrix (Rows = Map Class i, Columns = Reference Class j,
        plus row totals ``n_i.`` and column totals ``n_.j``).
    confusion_matrix_area_prop : pd.DataFrame or None
        Estimated area proportion matrix ``p_ij = W_i * (n_ij / n_i.)`` following
        AcATaMa / Olofsson et al. (2014) Eq. 1, or ``None`` if mapped pixel counts
        were not supplied.
    sampled_points : pd.DataFrame or gpd.GeoDataFrame or None
        Point/feature-level table of reference vs. predicted labels used in the
        assessment (useful for downstream QA or spatial logistic regression).
    """

    overall_metrics: dict[str, float]
    class_metrics: pd.DataFrame
    confusion_matrix_counts: pd.DataFrame
    confusion_matrix_area_prop: pd.DataFrame | None = field(default=None)
    sampled_points: pd.DataFrame | gpd.GeoDataFrame | None = field(default=None)

    def save_reports(
        self,
        output_dir: PathType,
        prefix: str = "accuracy",
    ) -> dict[str, Path]:
        """Write summary metrics JSON, class table CSV, and confusion matrices to disk."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}

        metrics_path = out_dir / f"{prefix}_overall_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.overall_metrics, f, indent=2)
        paths["overall_metrics_json"] = metrics_path

        class_csv = out_dir / f"{prefix}_class_accuracy_area.csv"
        self.class_metrics.to_csv(class_csv, index=False, encoding="utf-8")
        paths["class_metrics_csv"] = class_csv

        cm_csv = out_dir / f"{prefix}_confusion_matrix_counts.csv"
        self.confusion_matrix_counts.to_csv(cm_csv, encoding="utf-8")
        paths["confusion_matrix_counts_csv"] = cm_csv

        if self.confusion_matrix_area_prop is not None:
            cm_prop_csv = out_dir / f"{prefix}_confusion_matrix_area_proportions.csv"
            self.confusion_matrix_area_prop.to_csv(cm_prop_csv, encoding="utf-8")
            paths["confusion_matrix_area_prop_csv"] = cm_prop_csv

        if self.sampled_points is not None:
            pts_csv = out_dir / f"{prefix}_evaluated_samples.csv"
            df_to_save = pd.DataFrame(
                self.sampled_points.drop(columns="geometry", errors="ignore")
            )
            df_to_save.to_csv(pts_csv, index=False, encoding="utf-8")
            paths["evaluated_samples_csv"] = pts_csv

        return paths


def _compute_pixel_area_m2(crs: CRS | None, transform: Any, height: int) -> float:
    """Compute single-pixel area in square meters from raster CRS and Affine transform."""
    pixel_width = abs(float(transform.a))
    pixel_height = abs(float(transform.e))

    if crs is not None and crs.is_geographic:
        wgs84_a = 6_378_137.0
        wgs84_f = 1.0 / 298.257_223_563
        wgs84_b = wgs84_a * (1.0 - wgs84_f)

        centre_lat_deg = float(transform.f) + (height / 2.0) * float(transform.e)
        centre_lat_rad = math.radians(centre_lat_deg)

        sin_lat = math.sin(centre_lat_rad)
        cos_lat = math.cos(centre_lat_rad)
        e2 = 1.0 - (wgs84_b / wgs84_a) ** 2
        denom = math.sqrt(1.0 - e2 * sin_lat**2)
        meridional = wgs84_a * (1.0 - e2) / (denom**3)
        prime_vertical = wgs84_a / denom

        m_per_deg_lat = meridional * math.pi / 180.0
        m_per_deg_lon = prime_vertical * cos_lat * math.pi / 180.0
        return (pixel_width * m_per_deg_lon) * (pixel_height * m_per_deg_lat)

    return pixel_width * pixel_height


def count_raster_pixels_streaming(
    raster_path: PathType,
    *,
    band: int = 1,
    nodata_values: Sequence[int | float] = (0, -1, 255),
    class_remap: Mapping[int, int] | None = None,
    block_size: int = 2048,
) -> tuple[dict[int, int], float]:
    """Count mapped pixels per class using windowed streaming and return (counts_dict, pixel_area_m2).

    Parameters
    ----------
    raster_path : str or PathLike
        Path to the classified thematic GeoTIFF.
    band : int, default 1
        Band index (1-based) containing discrete class labels.
    nodata_values : sequence of numbers, default (0, -1, 255)
        Pixel values to exclude as background / nodata (in addition to ``src.nodata``).
    class_remap : Mapping[int, int] or None
        Optional dictionary mapping raw pixel codes to target class IDs (e.g., ``{18: 17}``).
    block_size : int, default 2048
        Window size for out-of-core reading.

    Returns
    -------
    counts_dict : dict[int, int]
        Mapping of ``{class_id: pixel_count}``.
    pixel_area_m2 : float
        Area of a single pixel in square meters.
    """
    totals: dict[int, int] = {}
    nodata_set = set(nodata_values)

    with rasterio.open(raster_path) as src:
        if src.nodata is not None and np.isfinite(src.nodata):
            nodata_set.add(int(src.nodata))

        pixel_area_m2 = _compute_pixel_area_m2(src.crs, src.transform, src.height)

        for r_off in range(0, src.height, block_size):
            win_h = min(block_size, src.height - r_off)
            for c_off in range(0, src.width, block_size):
                win_w = min(block_size, src.width - c_off)
                win = rasterio.windows.Window(c_off, r_off, win_w, win_h)
                arr = src.read(band, window=win, masked=True)

                valid_mask = ~np.ma.getmaskarray(arr)
                data = np.asarray(arr.data)
                valid_mask &= np.isfinite(data)
                for nd in nodata_set:
                    valid_mask &= data != nd

                if not np.any(valid_mask):
                    continue

                valid_vals = data[valid_mask].astype(np.int64)
                if class_remap:
                    for src_c, dst_c in class_remap.items():
                        valid_vals[valid_vals == int(src_c)] = int(dst_c)

                uniq, cnts = np.unique(valid_vals, return_counts=True)
                for u, c in zip(uniq.tolist(), cnts.tolist()):
                    totals[int(u)] = totals.get(int(u), 0) + int(c)

    return dict(sorted(totals.items())), pixel_area_m2


def compute_thematic_accuracy(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    *,
    class_names: Mapping[int, str] | None = None,
    labels: Sequence[int] | None = None,
    mapped_pixel_counts: Mapping[int, int] | None = None,
    pixel_area_m2: float | None = None,
    confidence_level: float = 0.95,
    merge_water_riverbed: bool = True,
    class_remap: Mapping[int, int] | None = None,
) -> ThematicAccuracyResult:
    """Compute thematic accuracy, class-imbalance metrics, and AcATaMa/Olofsson area estimates.

    Follows the international standard convention of **Olofsson et al. (2014)**,
    **Stehman (2014)**, and **QGIS AcATaMa**:
    - Confusion matrix rows ``i`` represent **Map (Predicted)** classes.
    - Confusion matrix columns ``j`` represent **Reference (Ground-Truth)** classes.

    Parameters
    ----------
    y_true : array-like of int
        Reference (ground-truth) class IDs.
    y_pred : array-like of int
        Predicted (map) class IDs at the reference locations.
    class_names : Mapping[int, str] or None
        Dictionary mapping integer class IDs to human-readable class names.
        Defaults to Timor-Leste 17/18-class schema when omitted.
    labels : sequence of int or None
        Explicit ordered list of class IDs to include in the evaluation.
    mapped_pixel_counts : Mapping[int, int] or None
        Total mapped pixel count ``N_i`` for each map class ``i`` across the entire
        classified raster. Required for AcATaMa / Olofsson et al. (2014) area-weighted
        accuracies and error-adjusted area estimation.
    pixel_area_m2 : float or None
        Area of a single raster pixel in square meters (e.g. ``900.0`` for 30 m Landsat,
        ``100.0`` for 10 m Sentinel-2). Used to convert pixel counts into hectares (ha)
        and square kilometers (km²).
    confidence_level : float, default 0.95
        Confidence level for standard-error confidence intervals (e.g. 0.95 -> z = 1.96).
    merge_water_riverbed : bool, default True
        If True, automatically merges Class 18 (``Dry_riverbed``) into Class 17
        (``Waterbody``) in both ``y_true``, ``y_pred``, and ``mapped_pixel_counts``.
    class_remap : Mapping[int, int] or None
        Additional custom class ID remapping applied before computing metrics.

    Returns
    -------
    ThematicAccuracyResult
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=np.int64).ravel()

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length (got {len(y_true_arr)} vs {len(y_pred_arr)})"
        )
    if len(y_true_arr) == 0:
        raise ValueError("Cannot compute accuracy on empty sample arrays.")

    remap: dict[int, int] = {}
    if merge_water_riverbed:
        remap[18] = 17
    if class_remap:
        for k, v in class_remap.items():
            remap[int(k)] = int(v)

    if remap:
        y_true_arr = y_true_arr.copy()
        y_pred_arr = y_pred_arr.copy()
        for src_id, dst_id in remap.items():
            y_true_arr[y_true_arr == src_id] = dst_id
            y_pred_arr[y_pred_arr == src_id] = dst_id

    # Harmonize mapped_pixel_counts if provided
    mapped_counts_clean: dict[int, int] | None = None
    if mapped_pixel_counts is not None:
        mapped_counts_clean = {}
        for cid, cnt in mapped_pixel_counts.items():
            cid_int = remap.get(int(cid), int(cid))
            if cid_int <= 0:
                continue
            mapped_counts_clean[cid_int] = mapped_counts_clean.get(cid_int, 0) + int(cnt)

    # Resolve class schema & ordered label list
    if class_names is None:
        schema = (
            DEFAULT_TIMOR_17_SCHEMA
            if merge_water_riverbed
            else DEFAULT_TIMOR_18_CLASSES
        )
    else:
        schema = {int(k): str(v) for k, v in class_names.items()}
        if merge_water_riverbed and 18 in schema and 18 in remap:
            schema = {k: v for k, v in schema.items() if k != 18}

    if labels is not None:
        class_list = sorted({remap.get(int(c), int(c)) for c in labels})
    else:
        observed = set(np.unique(y_true_arr).tolist()) | set(np.unique(y_pred_arr).tolist())
        if mapped_counts_clean is not None:
            observed |= set(mapped_counts_clean.keys())
        if class_names is not None:
            observed |= set(schema.keys())
        class_list = sorted(c for c in observed if c > 0)

    q = len(class_list)
    n_total = len(y_true_arr)
    z_score = float(norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))

    # -------------------------------------------------------------------------
    # 1. Sample Confusion Matrix: Rows = Map (Predicted i), Cols = Reference (j)
    #    Note: sklearn.metrics.confusion_matrix(y_true, y_pred) has Rows=True, Cols=Pred.
    #    Transposing (.T) yields Rows = Map (i), Cols = Reference (j) per Olofsson/AcATaMa.
    # -------------------------------------------------------------------------
    cm_true_pred = confusion_matrix(y_true_arr, y_pred_arr, labels=class_list)
    n_ij = cm_true_pred.T.astype(np.float64)  # shape (q, q): row i = map, col j = ref

    n_i_dot = n_ij.sum(axis=1)  # row totals (predicted sample counts per map class i)
    n_dot_j = n_ij.sum(axis=0)  # col totals (reference sample counts per ref class j)
    diag_ii = np.diag(n_ij)     # correctly classified counts n_ii

    label_names = [schema.get(c, f"Class_{c}") for c in class_list]
    row_col_labels = [f"{c}: {name}" for c, name in zip(class_list, label_names)]

    # Build readable Sample Count Confusion Matrix DataFrame with Totals
    cm_counts_df = pd.DataFrame(
        n_ij.astype(np.int64),
        index=row_col_labels,
        columns=row_col_labels,
    )
    cm_counts_df.index.name = "Map_Predicted (i) \\ Ref (j)"
    cm_counts_df["Total_Map (n_i.)"] = n_i_dot.astype(np.int64)
    col_totals_row = pd.Series(
        list(n_dot_j.astype(np.int64)) + [int(n_total)],
        index=cm_counts_df.columns,
        name="Total_Ref (n_.j)",
    )
    cm_counts_df = pd.concat([cm_counts_df, col_totals_row.to_frame().T])

    # -------------------------------------------------------------------------
    # 2. Sample-Based Class-Level Accuracy & Imbalance Metrics
    # -------------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        # User's Accuracy (Precision): n_ii / n_i.
        ua_sample = np.where(n_i_dot > 0, diag_ii / n_i_dot, np.nan)
        ua_sample_var = np.where(
            n_i_dot > 1,
            ua_sample * (1.0 - ua_sample) / (n_i_dot - 1.0),
            np.nan,
        )
        ua_sample_se = np.sqrt(ua_sample_var)

        # Producer's Accuracy (Recall / Sensitivity): n_jj / n_.j
        pa_sample = np.where(n_dot_j > 0, diag_ii / n_dot_j, np.nan)
        pa_sample_var = np.where(
            n_dot_j > 1,
            pa_sample * (1.0 - pa_sample) / (n_dot_j - 1.0),
            np.nan,
        )
        pa_sample_se = np.sqrt(pa_sample_var)

        # Specificity (True Negative Rate): TN_c / (TN_c + FP_c)
        # For class c: TP = diag_ii[c], FP = n_i_dot[c] - TP, FN = n_dot_j[c] - TP
        fp_c = n_i_dot - diag_ii
        fn_c = n_dot_j - diag_ii
        tn_c = n_total - (diag_ii + fp_c + fn_c)
        neg_support = tn_c + fp_c
        specificity = np.where(neg_support > 0, tn_c / neg_support, np.nan)

        # F1-Score: 2 * UA * PA / (UA + PA)
        f1_per_class = np.where(
            (ua_sample + pa_sample) > 0,
            2.0 * ua_sample * pa_sample / (ua_sample + pa_sample),
            0.0,
        )
        f1_per_class = np.where(
            np.isnan(ua_sample) & np.isnan(pa_sample), np.nan, np.nan_to_num(f1_per_class, nan=0.0)
        )

        # G-Mean (Sensitivity * Specificity)^0.5 — standard binary-per-class imbalance metric
        g_mean_per_class = np.sqrt(np.clip(pa_sample, 0.0, 1.0) * np.clip(specificity, 0.0, 1.0))

        # Fowlkes-Mallows Index: sqrt(UA * PA) — geometric mean of Precision and Recall
        fm_per_class = np.sqrt(
            np.clip(np.nan_to_num(ua_sample, nan=0.0), 0.0, 1.0)
            * np.clip(np.nan_to_num(pa_sample, nan=0.0), 0.0, 1.0)
        )
        fm_per_class = np.where(np.isnan(ua_sample) & np.isnan(pa_sample), np.nan, fm_per_class)

    # -------------------------------------------------------------------------
    # 3. Global Sample-Based Metrics
    # -------------------------------------------------------------------------
    oa_sample = float(accuracy_score(y_true_arr, y_pred_arr))
    oa_sample_se = (
        math.sqrt(oa_sample * (1.0 - oa_sample) / (n_total - 1.0))
        if n_total > 1
        else 0.0
    )

    active_ref_mask = n_dot_j > 0
    ref_weights = np.where(active_ref_mask, n_dot_j / np.sum(n_dot_j[active_ref_mask]), 0.0)

    valid_gmean = g_mean_per_class[active_ref_mask]
    g_mean_macro = float(np.nanmean(valid_gmean)) if len(valid_gmean) > 0 else float("nan")
    g_mean_weighted = (
        float(np.nansum(g_mean_per_class[active_ref_mask] * ref_weights[active_ref_mask]))
        if np.any(active_ref_mask)
        else float("nan")
    )

    # Multi-class geometric mean of recalls: (prod_{c in active} PA_c)^(1/K_active)
    active_recalls = pa_sample[active_ref_mask]
    if len(active_recalls) > 0 and np.all(active_recalls > 0):
        g_mean_multiclass = float(np.exp(np.mean(np.log(active_recalls))))
    elif len(active_recalls) > 0:
        g_mean_multiclass = 0.0
    else:
        g_mean_multiclass = float("nan")

    fm_macro = float(np.nanmean(fm_per_class[active_ref_mask])) if np.any(active_ref_mask) else float("nan")
    fm_weighted = (
        float(np.nansum(fm_per_class[active_ref_mask] * ref_weights[active_ref_mask]))
        if np.any(active_ref_mask)
        else float("nan")
    )

    overall_metrics: dict[str, float] = {
        "overall_accuracy_sample": oa_sample,
        "overall_accuracy_sample_se": oa_sample_se,
        "overall_accuracy_sample_ci95": z_score * oa_sample_se,
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "f1_macro": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "f1_micro": float(f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)),
        "g_mean_macro": g_mean_macro,
        "g_mean_weighted": g_mean_weighted,
        "g_mean_multiclass_recall": g_mean_multiclass,
        "fowlkes_mallows_macro": fm_macro,
        "fowlkes_mallows_weighted": fm_weighted,
        "kappa": float(cohen_kappa_score(y_true_arr, y_pred_arr)),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred_arr)),
        "confidence_level": float(confidence_level),
        "z_score": z_score,
        "total_samples": int(n_total),
        "num_classes_evaluated": int(np.sum(active_ref_mask | (n_i_dot > 0))),
    }

    # -------------------------------------------------------------------------
    # 4. AcATaMa / Olofsson et al. (2014) Stratified Area & Accuracy Estimators
    # -------------------------------------------------------------------------
    cm_area_prop_df: pd.DataFrame | None = None

    class_rows: list[dict[str, Any]] = []
    for idx, cid in enumerate(class_list):
        class_rows.append(
            {
                "class_id": int(cid),
                "class_name": label_names[idx],
                "ref_samples": int(n_dot_j[idx]),
                "map_samples": int(n_i_dot[idx]),
                "correct_samples": int(diag_ii[idx]),
                "user_accuracy": float(ua_sample[idx]) if np.isfinite(ua_sample[idx]) else np.nan,
                "user_accuracy_se": float(ua_sample_se[idx]) if np.isfinite(ua_sample_se[idx]) else np.nan,
                "user_accuracy_ci95": (
                    float(z_score * ua_sample_se[idx]) if np.isfinite(ua_sample_se[idx]) else np.nan
                ),
                "commission_error": (
                    float(1.0 - ua_sample[idx]) if np.isfinite(ua_sample[idx]) else np.nan
                ),
                "producer_accuracy_sample": (
                    float(pa_sample[idx]) if np.isfinite(pa_sample[idx]) else np.nan
                ),
                "producer_accuracy_sample_se": (
                    float(pa_sample_se[idx]) if np.isfinite(pa_sample_se[idx]) else np.nan
                ),
                "omission_error_sample": (
                    float(1.0 - pa_sample[idx]) if np.isfinite(pa_sample[idx]) else np.nan
                ),
                "specificity": float(specificity[idx]) if np.isfinite(specificity[idx]) else np.nan,
                "f1_score": float(f1_per_class[idx]) if np.isfinite(f1_per_class[idx]) else np.nan,
                "g_mean": float(g_mean_per_class[idx]) if np.isfinite(g_mean_per_class[idx]) else np.nan,
                "fowlkes_mallows": float(fm_per_class[idx]) if np.isfinite(fm_per_class[idx]) else np.nan,
            }
        )

    if mapped_counts_clean is not None and sum(mapped_counts_clean.values()) > 0:
        N_i = np.array([float(mapped_counts_clean.get(c, 0)) for c in class_list], dtype=np.float64)
        N_total = float(np.sum(N_i))
        W_i = N_i / N_total  # Map stratum weights (sum W_i = 1)

        # Conditional sample proportions q_ij = n_ij / n_i. (0 where n_i. == 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            q_ij = np.where(n_i_dot[:, None] > 0, n_ij / n_i_dot[:, None], 0.0)

        # Estimated area proportions matrix: p_ij = W_i * (n_ij / n_i.) (Olofsson Eq. 1)
        p_ij = W_i[:, None] * q_ij
        p_i_dot = p_ij.sum(axis=1)  # Row totals (= W_i where n_i. > 0)
        p_dot_j = p_ij.sum(axis=0)  # Column totals = Error-adjusted area proportion per class j (Eq. 8)
        p_diag = np.diag(p_ij)

        # Within-stratum binomial variance term: W_i^2 * q_ij * (1 - q_ij) / (n_i. - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom_i = np.where(n_i_dot[:, None] > 1, n_i_dot[:, None] - 1.0, np.nan)
            cell_var_ij = (W_i[:, None] ** 2) * q_ij * (1.0 - q_ij) / denom_i
            cell_var_ij = np.nan_to_num(cell_var_ij, nan=0.0)

        # 1. Area-weighted Overall Accuracy (Olofsson Eq. 4 & 5)
        oa_area = float(np.sum(p_diag))
        oa_area_var = float(np.sum(np.diag(cell_var_ij)))
        oa_area_se = math.sqrt(max(oa_area_var, 0.0))

        overall_metrics["overall_accuracy_area_weighted"] = oa_area
        overall_metrics["overall_accuracy_area_se"] = oa_area_se
        overall_metrics["overall_accuracy_area_ci95"] = z_score * oa_area_se
        overall_metrics["total_mapped_pixels"] = int(N_total)

        # 2. Area-weighted Producer's Accuracy (Olofsson Eq. 3 & 7)
        with np.errstate(divide="ignore", invalid="ignore"):
            pa_area = np.where(p_dot_j > 0, p_diag / p_dot_j, np.nan)

        pa_area_se = np.full(q, np.nan, dtype=np.float64)
        for j in range(q):
            if p_dot_j[j] <= 0 or not np.isfinite(pa_area[j]):
                continue
            # Olofsson et al. (2014) Eq. 7:
            # V(P_j) = (1 / p_.j^2) * sum_i [ W_i^2 * q_ij * (1 - q_ij) / (n_i. - 1) * (delta_ij - P_j)^2 ]
            delta_j = np.zeros(q, dtype=np.float64)
            delta_j[j] = 1.0
            var_pj = np.sum(cell_var_ij[:, j] * ((delta_j - pa_area[j]) ** 2)) / (p_dot_j[j] ** 2)
            pa_area_se[j] = math.sqrt(max(float(var_pj), 0.0))

        # 3. Error-Adjusted Area Proportions & Standard Error (Olofsson Eq. 8, 9, 10, 11)
        p_dot_j_var = np.sum(cell_var_ij, axis=0)
        p_dot_j_se = np.sqrt(np.maximum(p_dot_j_var, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_area = np.where(p_dot_j > 0, p_dot_j_se / p_dot_j, np.nan)

        # Physical area units (ha and km²)
        pix_m2 = float(pixel_area_m2) if pixel_area_m2 is not None else 1.0
        pix_ha = pix_m2 / 10_000.0
        pix_km2 = pix_m2 / 1_000_000.0

        total_area_ha = N_total * pix_ha
        total_area_km2 = N_total * pix_km2
        if pixel_area_m2 is not None:
            overall_metrics["pixel_area_m2"] = pix_m2
            overall_metrics["total_mapped_area_ha"] = total_area_ha
            overall_metrics["total_mapped_area_km2"] = total_area_km2

        for idx in range(q):
            row = class_rows[idx]
            row["mapped_pixels"] = int(N_i[idx])
            row["map_weight_Wi"] = float(W_i[idx])
            row["producer_accuracy_area"] = (
                float(pa_area[idx]) if np.isfinite(pa_area[idx]) else np.nan
            )
            row["producer_accuracy_area_se"] = (
                float(pa_area_se[idx]) if np.isfinite(pa_area_se[idx]) else np.nan
            )
            row["producer_accuracy_area_ci95"] = (
                float(z_score * pa_area_se[idx]) if np.isfinite(pa_area_se[idx]) else np.nan
            )
            row["omission_error_area"] = (
                float(1.0 - pa_area[idx]) if np.isfinite(pa_area[idx]) else np.nan
            )
            row["adjusted_area_prop"] = float(p_dot_j[idx])
            row["adjusted_area_prop_se"] = float(p_dot_j_se[idx])
            row["adjusted_area_cv"] = (
                float(cv_area[idx]) if np.isfinite(cv_area[idx]) else np.nan
            )

            # Adjusted pixel count (matches Table 5 in Gatis et al. 2022)
            row["adjusted_pixels"] = float(N_total * p_dot_j[idx])
            row["adjusted_pixels_se"] = float(N_total * p_dot_j_se[idx])
            row["adjusted_pixels_ci95"] = float(z_score * N_total * p_dot_j_se[idx])

            if pixel_area_m2 is not None:
                mapped_ha = float(N_i[idx] * pix_ha)
                adj_ha = float(total_area_ha * p_dot_j[idx])
                se_ha = float(total_area_ha * p_dot_j_se[idx])
                ci95_ha = float(z_score * se_ha)

                mapped_km2 = float(N_i[idx] * pix_km2)
                adj_km2 = float(total_area_km2 * p_dot_j[idx])
                se_km2 = float(total_area_km2 * p_dot_j_se[idx])
                ci95_km2 = float(z_score * se_km2)

                row["mapped_area_ha"] = mapped_ha
                row["adjusted_area_ha"] = adj_ha
                row["adjusted_area_se_ha"] = se_ha
                row["adjusted_area_ci95_ha"] = ci95_ha
                row["adjusted_area_lower_ha"] = max(0.0, adj_ha - ci95_ha)
                row["adjusted_area_upper_ha"] = adj_ha + ci95_ha

                row["mapped_area_km2"] = mapped_km2
                row["adjusted_area_km2"] = adj_km2
                row["adjusted_area_se_km2"] = se_km2
                row["adjusted_area_ci95_km2"] = ci95_km2
                row["adjusted_area_lower_km2"] = max(0.0, adj_km2 - ci95_km2)
                row["adjusted_area_upper_km2"] = adj_km2 + ci95_km2

        # Build Area Proportions Confusion Matrix DataFrame
        cm_area_prop_df = pd.DataFrame(
            p_ij,
            index=row_col_labels,
            columns=row_col_labels,
        )
        cm_area_prop_df.index.name = "Map_Predicted (i) \\ Ref (j)"
        cm_area_prop_df["W_i (Map_Prop)"] = W_i
        cm_area_prop_df["Total_Map_Prop (p_i.)"] = p_i_dot
        prop_totals_row = pd.Series(
            list(p_dot_j) + [float(np.sum(W_i)), float(np.sum(p_dot_j))],
            index=cm_area_prop_df.columns,
            name="Adjusted_Ref_Prop (p_.j)",
        )
        cm_area_prop_df = pd.concat([cm_area_prop_df, prop_totals_row.to_frame().T])

    class_metrics_df = pd.DataFrame(class_rows)

    return ThematicAccuracyResult(
        overall_metrics=overall_metrics,
        class_metrics=class_metrics_df,
        confusion_matrix_counts=cm_counts_df,
        confusion_matrix_area_prop=cm_area_prop_df,
    )


def _extract_reference_and_map_labels(
    raster_path: PathType,
    reference_data: PathType | gpd.GeoDataFrame,
    class_field: str = "new_id",
    *,
    band: int = 1,
    stability_col: str | None = None,
    stability_val: int | float = 1,
    polygon_mode: str = "majority",
    nodata_values: Sequence[int | float] = (0, -1, 255),
) -> gpd.GeoDataFrame:
    """Extract map predictions at reference vector geometries (Points or Polygons).

    Parameters
    ----------
    raster_path : str or PathLike
        Path to classified single-band thematic GeoTIFF.
    reference_data : str, PathLike, or GeoDataFrame
        Path to GeoJSON/Shapefile/GeoPackage or loaded GeoDataFrame.
    class_field : str, default 'new_id'
        Column containing reference ground-truth integer class IDs.
    band : int, default 1
        Band index in classified raster.
    stability_col : str or None
        Optional CCDC stability flag column (e.g., ``'stable_2020'``, ``'stable_2015'``,
        ``'stable_2005'``). When specified, only rows where ``row[stability_col] == stability_val``
        are retained.
    stability_val : int or float, default 1
        Target value for ``stability_col``.
    polygon_mode : {'majority', 'all_pixels', 'centroid'}, default 'majority'
        How polygon reference features are evaluated against the raster:
        - ``'majority'``: modal valid predicted class inside the polygon (1 sample unit per polygon).
        - ``'centroid'``: samples the pixel at each polygon's representative point / centroid.
        - ``'all_pixels'``: expands each polygon into individual valid pixels.
    nodata_values : sequence of numbers
        Invalid raster values to ignore.
    """
    if isinstance(reference_data, gpd.GeoDataFrame):
        gdf = reference_data.copy()
    else:
        gdf = gpd.read_file(reference_data)

    if class_field not in gdf.columns:
        raise ValueError(
            f"Reference class field {class_field!r} not found. Available columns: {list(gdf.columns)}"
        )

    if stability_col is not None:
        if stability_col not in gdf.columns:
            raise ValueError(
                f"Stability column {stability_col!r} not found in reference data. "
                f"Available columns: {list(gdf.columns)}"
            )
        gdf = gdf[gdf[stability_col] == stability_val].copy()

    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty & gdf[class_field].notna()].copy()
    if len(gdf) == 0:
        raise ValueError("No valid reference features remaining after filtering.")

    nodata_set = set(nodata_values)

    with rasterio.open(raster_path) as src:
        if src.nodata is not None and np.isfinite(src.nodata):
            nodata_set.add(int(src.nodata))

        if gdf.crs is not None and src.crs is not None and gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        geom_types = set(gdf.geometry.geom_type.unique())
        is_all_points = geom_types.issubset({"Point"})

        if is_all_points or polygon_mode == "centroid":
            sample_geoms = (
                gdf.geometry
                if is_all_points
                else gdf.geometry.representative_point()
            )
            coords = [(pt.x, pt.y) for pt in sample_geoms]
            sampled_iter = src.sample(coords, indexes=band, masked=True)

            preds: list[int] = []
            valid_rows: list[bool] = []
            for val_arr in sampled_iter:
                val_ma = np.ma.asarray(val_arr)
                if np.ma.getmaskarray(val_ma).any():
                    preds.append(-1)
                    valid_rows.append(False)
                    continue
                v = float(val_ma.data[0])
                if not np.isfinite(v) or int(v) in nodata_set:
                    preds.append(-1)
                    valid_rows.append(False)
                else:
                    preds.append(int(v))
                    valid_rows.append(True)

            gdf["ref_class"] = gdf[class_field].astype(int)
            gdf["map_class"] = preds
            gdf = gdf[valid_rows].copy()
            return gdf.reset_index(drop=True)

        # Polygon handling ('majority' or 'all_pixels')
        records: list[dict[str, Any]] = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            ref_c = int(row[class_field])

            if geom.geom_type == "Point":
                val_ma = np.ma.asarray(next(src.sample([(geom.x, geom.y)], indexes=band, masked=True)))
                if np.ma.getmaskarray(val_ma).any():
                    continue
                v = float(val_ma.data[0])
                if not np.isfinite(v) or int(v) in nodata_set:
                    continue
                rec = row.to_dict()
                rec["ref_class"] = ref_c
                rec["map_class"] = int(v)
                records.append(rec)
                continue

            try:
                out_img, _ = rio_mask(src, [geom], indexes=band, crop=True, filled=False)
            except ValueError:
                continue

            vals = np.ma.asarray(out_img).compressed()
            vals = vals[np.isfinite(vals)]
            for nd in nodata_set:
                vals = vals[vals != nd]

            if len(vals) == 0:
                continue

            int_vals = vals.astype(np.int64)
            if polygon_mode == "majority":
                uniq, cnts = np.unique(int_vals, return_counts=True)
                modal_class = int(uniq[np.argmax(cnts)])
                rec = row.to_dict()
                rec["ref_class"] = ref_c
                rec["map_class"] = modal_class
                records.append(rec)
            elif polygon_mode == "all_pixels":
                base_dict = row.to_dict()
                for px_val in int_vals.tolist():
                    rec = base_dict.copy()
                    rec["ref_class"] = ref_c
                    rec["map_class"] = int(px_val)
                    records.append(rec)
            else:
                raise ValueError(
                    f"Unsupported polygon_mode={polygon_mode!r}. Choose 'majority', 'centroid', or 'all_pixels'."
                )

        if not records:
            raise ValueError("No valid raster pixels overlapped the reference geometries.")

        return gpd.GeoDataFrame(records, crs=gdf.crs).reset_index(drop=True)


def assess_map_accuracy(
    raster_path: PathType,
    reference_data: PathType | gpd.GeoDataFrame,
    class_field: str = "new_id",
    *,
    class_names: Mapping[int, str] | None = None,
    labels: Sequence[int] | None = None,
    stability_col: str | None = None,
    stability_val: int | float = 1,
    merge_water_riverbed: bool = True,
    class_remap: Mapping[int, int] | None = None,
    compute_area_adjustment: bool = True,
    polygon_mode: str = "majority",
    confidence_level: float = 0.95,
    nodata_values: Sequence[int | float] = (0, -1, 255),
    block_size: int = 2048,
    output_dir: PathType | None = None,
    output_prefix: str = "accuracy",
) -> ThematicAccuracyResult:
    """End-to-end thematic accuracy assessment & AcATaMa/Olofsson area estimation for a LULC GeoTIFF.

    Parameters
    ----------
    raster_path : str or PathLike
        Path to the classified single-band thematic GeoTIFF (Landsat 30m or Sentinel-2 10m).
    reference_data : str, PathLike, or GeoDataFrame
        Path to the testing/validation GeoJSON, Shapefile, or GeoDataFrame.
    class_field : str, default 'new_id'
        Column name in ``reference_data`` containing the ground-truth class IDs.
    class_names : Mapping[int, str] or None
        Optional mapping of ``{class_id: class_name}``. Defaults to the Timor-Leste
        17/18-class schema when omitted.
    labels : sequence of int or None
        Explicit list of class IDs to include.
    stability_col : str or None
        Optional CCDC stability flag attribute for multi-temporal validation
        (e.g., ``'stable_2020'``, ``'stable_2015'``, ``'stable_2005'``, ``'stable_2025'``).
        When provided, filters the GeoJSON to rows where ``stability_col == stability_val``.
    stability_val : int or float, default 1
        Target value for ``stability_col``.
    merge_water_riverbed : bool, default True
        If True, automatically merges Class 18 (``Dry_riverbed``) into Class 17 (``Waterbody``)
        in both the reference data and raster counts.
    class_remap : Mapping[int, int] or None
        Additional custom class ID remapping.
    compute_area_adjustment : bool, default True
        If True, streams the classified raster in blocks to count mapped pixels per class
        and computes AcATaMa / Olofsson et al. (2014) area-weighted accuracies, error-adjusted
        area (ha and km²), standard errors (SE), 95% confidence intervals (CI), and CV.
    polygon_mode : {'majority', 'all_pixels', 'centroid'}, default 'majority'
        How polygon geometries in ``reference_data`` are sampled (ignored for Point geometries).
    confidence_level : float, default 0.95
        Confidence level for standard-error intervals (default 95%, z = 1.96).
    nodata_values : sequence of numbers, default (0, -1, 255)
        Pixel values treated as background / nodata.
    block_size : int, default 2048
        Window block size for streaming pixel counts from ``raster_path``.
    output_dir : str, PathLike, or None
        If provided, automatically exports the JSON summary, class accuracy & area CSV,
        confusion matrix CSVs, and evaluated sample points to this folder.
    output_prefix : str, default 'accuracy'
        Filename prefix when ``output_dir`` is specified.

    Returns
    -------
    ThematicAccuracyResult
    """
    sampled_gdf = _extract_reference_and_map_labels(
        raster_path=raster_path,
        reference_data=reference_data,
        class_field=class_field,
        stability_col=stability_col,
        stability_val=stability_val,
        polygon_mode=polygon_mode,
        nodata_values=nodata_values,
    )

    remap: dict[int, int] = {}
    if merge_water_riverbed:
        remap[18] = 17
    if class_remap:
        for k, v in class_remap.items():
            remap[int(k)] = int(v)

    if remap:
        sampled_gdf["ref_class"] = sampled_gdf["ref_class"].replace(remap)
        sampled_gdf["map_class"] = sampled_gdf["map_class"].replace(remap)

    sampled_gdf["is_correct"] = (sampled_gdf["ref_class"] == sampled_gdf["map_class"]).astype(int)

    mapped_counts: dict[int, int] | None = None
    pixel_area_m2: float | None = None
    if compute_area_adjustment:
        mapped_counts, pixel_area_m2 = count_raster_pixels_streaming(
            raster_path,
            nodata_values=nodata_values,
            class_remap=remap if remap else None,
            block_size=block_size,
        )

    result = compute_thematic_accuracy(
        y_true=sampled_gdf["ref_class"].to_numpy(),
        y_pred=sampled_gdf["map_class"].to_numpy(),
        class_names=class_names,
        labels=labels,
        mapped_pixel_counts=mapped_counts,
        pixel_area_m2=pixel_area_m2,
        confidence_level=confidence_level,
        merge_water_riverbed=merge_water_riverbed,
        class_remap=class_remap,
    )
    result.sampled_points = sampled_gdf

    if output_dir is not None:
        result.save_reports(output_dir, prefix=output_prefix)

    return result
