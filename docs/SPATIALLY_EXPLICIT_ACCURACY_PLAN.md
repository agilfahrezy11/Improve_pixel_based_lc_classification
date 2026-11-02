# Implementation Plan: Spatially Explicit Accuracy Assessment (Per-Pixel & Multi-Temporal Change Confidence)

## 1. Objective & Theoretical Background
Standard confusion matrices (Olofsson et al., 2014; AcATaMa) summarize thematic map accuracy globally and per class, assuming error is uniform within each mapped stratum. However, classification error varies spatially across landscapes due to topographic gradients, spectral ambiguity at habitat boundaries, and sensor geometry.

Following **Gatis et al. (2022)** (*Ecological Solutions and Evidence*, Section 2.2.5 & 2.3.2) and **Khatami et al. (2017)** (*Remote Sensing of Environment*, 191, 156–167), this plan outlines the implementation of a **Spatially Explicit Accuracy Assessment** module (`spatial_accuracy.py`) inside `src/ML_LC_Classifier/` for both **Landsat (30 m)** and **Sentinel-2 (10 m)** products across multiple epochs (2005, 2015, 2020, 2025).

---

## 2. Mathematical Formulation

### 2.1 Binary Correctness Response Variable
Using the `sampled_points` GeoDataFrame already produced by `assess_map_accuracy()` in [`accuracy_assessment.py`](src/ML_LC_Classifier/accuracy_assessment.py), each validation point $i \in \{1, \dots, n\}$ has a binary correctness indicator:
$$z_i = \mathbb{I}(\hat{y}_i = y_i) = \begin{cases} 1 & \text{if Map Class } \hat{y}_i = \text{Reference Class } y_i \\ 0 & \text{if Map Class } \hat{y}_i \neq \text{Reference Class } y_i \end{cases}$$

### 2.2 Candidate Spatial Predictor Layers ($\mathbf{x}_i$)
To bridge **Gatis et al. (2022)** (which uses environmental/spectral layers like spring blue reflectance and topographic slope) with **Malinverni & Sanità (2026)** (which outputs internal classifier uncertainty rasters in [`hybrid_assembly.py`](src/ML_LC_Classifier/hybrid_assembly.py)), the logistic regression model will accept two groups of aligned raster predictors:

1. **Classifier Uncertainty & Class Prior Layers (Always Available Locally):**
   - `confusion_index.tif` ($CI = S_{\max} / W_{\max} \in [0, 1]$): Local competition between runner-up and winning probability.
   - `stability_map.tif` ($\mathbb{I}(CI \le 0.65)$): Binary parcel stability indicator.
   - `class_user_accuracy`: Empirical User's Accuracy ($\hat{U}_{\hat{y}}$) of the predicted class at pixel $(r, c)$ mapped from the confusion matrix.
   - *(Optional)* Winning probability $W_{\max}$ and Shannon Entropy $H = -\sum_c P_c \ln(P_c)$ from the 18-band probability stack.

2. **Environmental, Topographic & Spectral Covariate Rasters (Gatis et al., 2022 Mode):**
   - Any dictionary of single-band or multi-band GeoTIFFs (e.g., `slope.tif`, `elevation.tif`, `TWI.tif`, seasonal synthetic bands `B2_wet`, `B4_dry`, `NDVI_delta`).
   - Reprojected and resampled on-the-fly to match the target classification grid via `rasterio.vrt.WarpedVRT`.
   - Standardized ($z$-score: $\tilde{x}_k = \frac{x_k - \mu_k}{\sigma_k}$) using the mean $\mu_k$ and standard deviation $\sigma_k$ fitted on the validation points.

### 2.3 Stepwise AIC Logistic Regression Selection
Following Gatis et al. (2022) Table 7:
1. Fit a multiple logistic regression model:
   $$\hat{P}(z = 1 \mid \tilde{\mathbf{x}}) = \frac{1}{1 + \exp\left(-(\beta_0 + \sum_{k \in \mathcal{S}} \beta_k \tilde{x}_k)\right)}$$
2. Implement **Forward, Backward, and Bidirectional Stepwise Selection** minimizing the **Akaike Information Criterion (AIC)**:
   $$\text{AIC} = 2(|\mathcal{S}| + 1) - 2 \sum_{i=1}^{n} \left[ z_i \ln(\hat{p}_i) + (1 - z_i) \ln(1 - \hat{p}_i) \right]$$
3. Report model comparison table (`Forward`, `Backward`, `Stepwise`, `Full`, and `Uncertainty-Only`) with AIC, BIC, ROC-AUC, Brier Score, and per-predictor $\beta_k$, standard error, $z$-statistic, and $p$-value.

### 2.4 Multi-Epoch Change Accuracy Product (Gatis et al., 2022 Section 2.3.2)
For any pair of epochs $(t_1, t_2)$ (e.g., `2020 -> 2025` or `2005 -> 2025`):
- Assuming independent per-epoch classification errors conditioned on covariates, the joint probability that a mapped transition (or persistence) at pixel $(r, c)$ is accurate in both epochs is:
  $$\hat{P}_{\text{change}}(r, c) = \hat{P}_{t_1}(r, c) \times \hat{P}_{t_2}(r, c)$$
- Outputs a wall-to-wall `change_accuracy_t1_t2.tif` raster (`Float32` in `[0, 1]` or `UInt8` in `0–100%`) to filter or prioritize field verification of mapped habitat changes.

---

## 3. Proposed Architecture & Public API

### Target File
`src/ML_LC_Classifier/spatial_accuracy.py` (exported in `src/ML_LC_Classifier/__init__.py`)

### Dataclasses & Functions

```python
@dataclass
class SpatialAccuracyModelResult:
    selected_features: list[str]
    coefficients_table: pd.DataFrame       # feature, beta, std_err, z_stat, p_value, odds_ratio
    candidate_models_aic: pd.DataFrame     # step/direction, formula, aic, bic, roc_auc, brier_score
    scaler_mean: dict[str, float]
    scaler_std: dict[str, float]
    intercept: float
    roc_auc: float
    brier_score: float
    accuracy_raster_path: Path | None = None

def fit_spatial_accuracy_model(
    accuracy_result: ThematicAccuracyResult,
    covariate_rasters: Mapping[str, PathType] | None = None,
    *,
    confusion_index_raster: PathType | None = None,
    stability_raster: PathType | None = None,
    include_class_ua_prior: bool = True,
    selection_method: str = "stepwise_aic",  # "stepwise_aic", "forward_aic", "backward_aic", "none"
) -> SpatialAccuracyModelResult:
    """Extract covariate values at validation points, standardize, and fit AIC-selected logistic regression."""

def predict_spatial_accuracy_raster(
    model_result: SpatialAccuracyModelResult,
    lulc_raster_path: PathType,
    output_raster_path: PathType,
    covariate_rasters: Mapping[str, PathType] | None = None,
    *,
    confusion_index_raster: PathType | None = None,
    stability_raster: PathType | None = None,
    class_ua_map: Mapping[int, float] | None = None,
    block_size: int = 2048,
    dtype: str = "float32",  # or "uint8" (0-100%)
) -> Path:
    """Stream across the national LULC grid in 2048x2048 windows and write per-pixel predicted accuracy."""

def compute_change_accuracy_raster(
    accuracy_raster_t1: PathType,
    accuracy_raster_t2: PathType,
    output_raster_path: PathType,
    *,
    block_size: int = 2048,
) -> Path:
    """Multiply two epoch spatial accuracy rasters (P_t1 * P_t2) per Gatis et al. (2022) Section 2.3.2."""
```

---

## 4. Implementation Steps (When Ready to Execute)

1. **Step 1 — Covariate Point Sampler with `WarpedVRT`:**
   - Sample `confusion_index.tif`, `stability_map.tif`, and any user-supplied environmental rasters (`slope.tif`, `elevation.tif`, seasonal spectral rasters) at the `sampled_points` coordinates from `ThematicAccuracyResult`.
2. **Step 2 — Stepwise AIC Logistic Regression Solver:**
   - Implement exact maximum-likelihood logistic regression (via `scipy.optimize` / Newton-Raphson IRLS so no extra `statsmodels` dependency is required) with Hessian-derived standard errors, $p$-values, and forward/backward/stepwise AIC variable selection matching Table 7 of Gatis et al. (2022).
3. **Step 3 — Out-of-Core Windowed Raster Prediction:**
   - Use `rasterio` windowed streaming (`2048 x 2048` blocks) and `WarpedVRT` to evaluate $\hat{P}(z = 1 \mid \tilde{\mathbf{x}})$ wall-to-wall across Timor-Leste in `< 1 GB` RAM.
4. **Step 4 — Multi-Temporal Change Confidence Raster:**
   - Implement `compute_change_accuracy_raster()` to compute $\hat{P}_{t_1} \times \hat{P}_{t_2}$ across epoch pairs (`2005–2015`, `2015–2020`, `2020–2025`).
