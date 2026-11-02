# ml-lc-classifier — Codebase Design Review

## Overview

The package has grown from its original 8-module ML pipeline into a 14-module system covering the full LULC production chain. The core pipeline (load → select features → tune → classify) is well-shaped. The post-processing layer (seam healing, hybrid assembly, trajectory analysis, MMU) has accumulated more complexity and more design debt. This review maps both layers against the deep-module principles of small interface / large implementation, clean seam placement, and testability through the interface.

---

## Module-by-module depth assessment

### `classify_raster.py` — Shallow on one dimension (missing probability path)

The existing hard-label path is clean and appropriately deep. The gap is a **missing soft-classification path**: the entire post-processing layer (`process_tile`, `bayesian_smooth`, `classification_assembly`) operates on probability rasters, but there is no public function that takes a fitted model and writes a multi-band probability GeoTIFF. Callers today have to construct their own block loop to call `model.predict_proba`. That logic belongs inside `classify_raster`.

**Recommendation:** Add a `classify_raster_proba(raster_path, model, output_path, *, selector, ...)` function, or extend `classify_raster` with a `mode: Literal["hard", "proba"]` parameter.

---

### `process_tile` in `hybrid_assembly.py` — One function doing the work of four

`process_tile` is ~300 lines. Inside it, it:
1. Reads and spatially aligns segment IDs and probability rasters
2. Computes object-mean probabilities via `np.bincount`
3. Does Bayesian blending (adaptive weight, argmax tracking, confusion index)
4. Applies mask-layer refinement rules
5. Writes three output GeoTIFFs (LULC, stability, confusion index)

Steps 2–3 overlap significantly with `classification_assembly.py` — the `bayesian_smooth` and `classify_from_probs` functions exist precisely to encapsulate this logic, but `process_tile` reimplements argmax + confusion index tracking inline rather than calling them. The result is duplicated logic at the cost of locality.

The return type is `dict[str, Path]` — the only public function in the package that doesn't return a `@dataclass`. The keys `"lulc"`, `"stability"`, `"ci"` are string-typed, so typos are silent until runtime.

**Recommendations:**
- Define a `ProcessTileResult` dataclass with `lulc`, `stability`, `ci` attributes.
- Extract the inner pixel-blending loop into a helper that delegates to `classification_assembly.classify_from_probs` and `bayesian_smooth` so the two paths can't drift apart.
- Move the `print()` progress reporting behind a `verbose: bool = True` flag, consistent with `tune_model`'s `verbose: int` parameter.

---

### `refine_lulc_with_mask_layer` and `process_tile` — Duplicated mask-evaluation logic

Both functions independently:
1. Build a set of unique mask paths
2. Open each mask in a `WarpedVRT` aligned to the target raster
3. Apply each `MaskRule` (candidate mask → require-mask AND → exclude-mask AND → write)

The VRT setup and rule evaluation loop is copy-pasted between the two. If a rule's evaluation logic ever needs to change, it must be changed in two places.

**Recommendation:** Extract `_evaluate_mask_rules(lulc_block, rules, cached_mask_blocks)` as a shared private helper, and extract the VRT lifecycle into a context manager or helper that both functions share.

---

### `lc_post_process.calculate_class_area` — Computation + visualisation mixed in one call

The function accepts `figsize`, `title`, and `xlabel` as parameters, builds a matplotlib figure, and returns it embedded in `ClassAreaResult.figure`. This couples the area computation (always needed) to the figure (sometimes not — e.g., headless CI, downstream reporting). A caller who only wants the DataFrame has no way to skip matplotlib.

The function also raises `ImportError` on matplotlib absence, which is different from every other optional dependency pattern in the package (all others guard with `try/except ImportError` inside and raise with a clear message only when the relevant code path is actually hit — which this does, but then makes matplotlib *required* any time you call the function regardless of whether you want a figure).

**Recommendation:** Split into two: `calculate_class_area(raster_path, class_spec, *, csv_path)` returns just the DataFrame, and a separate `plot_class_area(area_df, *, figsize, title, ...)` returns the figure. The result type stays `ClassAreaResult` but `figure` becomes optional (`None` when not requested).

---

### `apply_mmu` in `lc_post_process.py` — Two execution paths hidden in one function

The function has a small-image path (loads the whole raster at once) and a large-image path (tiled windowed processing). The branching point is buried in the middle. The label-validation logic runs in both branches. The function is ~150 lines of body.

**Recommendation:** Extract `_apply_mmu_full(...)` and `_apply_mmu_tiled(...)` as private helpers. The public `apply_mmu` becomes a thin dispatcher, and the validation logic moves out of the branches.

---

### `tune_model` — Interface width from backwards compatibility

The function has 15 parameters. Two of them are backwards-compatibility aliases: `estimator` ↔ `classifier` and `param_distributions` ↔ `parameter_space`. The aliases are documented, but a 15-parameter interface is wide. A caller first using the function must read the entire signature to know which name to use.

**Recommendation:** Keep the aliases for now (they're actively used), but deprecate the old names explicitly with a `warnings.warn(..., DeprecationWarning)` so callers know which names are canonical. This signals that the interface will shrink over time.

---

### `load_extract.py` — Two shallow pass-through functions leak open handles

`open_raster` and `load_and_extract` both return an open `rasterio.DatasetReader`. The caller must remember to `.close()`. `load_and_extract_points` uses the handle as a context manager internally (closes it before returning), but `load_and_extract` does not. The inconsistency is a resource-leak footgun.

**Recommendation:** Deprecate `open_raster` and `load_and_extract` as public functions or add a context-manager wrapper. The high-level `load_and_split_training_data` already handles cleanup correctly — that's the pattern to promote.

---

### `load_extract` + `outlier_detection` — Duplicated vector-loading logic

`outlier_detection.load_points` and `load_extract.load_training_samples` both read a vector file and return the same result. `load_points` also handles CSV (drops geometry column after reading). The existence of two near-identical functions in different modules with slightly different APIs forces callers to choose the right one by context.

**Recommendation:** Unify under `load_training_samples` in `load_extract` (with an optional `drop_geometry: bool` flag), and make `outlier_detection.load_points` a thin alias calling it. Single place for the read logic.

---

### `feature_stack.py` → `planetary_computer.py` — Hidden eager coupling

`feature_stack.py` imports `DownloadedAsset` from `.planetary_computer` at module level. `planetary_computer.py` imports `geopandas` at module level. The result: `import ML_LC_Classifier` always loads `geopandas`, even in environments where planetary computer support is not installed. This contradicts the "no side effects on import" architectural rule.

**Recommendation:** Move the `DownloadedAsset` dataclass to a thin `_types.py` module (or directly into `feature_stack.py` as a local type alias), and make `planetary_computer.py` import it from there. `planetary_computer.py` can then guard its own `geopandas` import inside the functions that need it.

---

### `classify_raster` — Untyped `selector` and `model` parameters

Both `model` and `selector` are typed `Any`. The interface communicates nothing about what methods are required. `tune_model.py` already defines a `ClassifierModel` Protocol for `.predict()` — `classify_raster` could use it.

**Recommendation:** Define (or reuse) a `SelectorProtocol` with `.transform(X)` and expose it in `__init__.py`, then type `selector` and `model` with these Protocols.

---

## Missing modules / capabilities

### `predict_proba` raster path

As noted above: no public function writes a probability raster. This is the single largest functional gap given how central probability rasters are to the post-processing pipeline.

### Pipeline state / validation

There is no way to verify that a feature stack raster is compatible with a fitted model before running `classify_raster`. Band order is the contract (per the architecture rules), but nothing validates it at call time. The JSON manifest exists but `classify_raster` doesn't read it.

**Recommendation:** Add a `validate_feature_stack(manifest_path, model, feature_names)` utility that confirms feature count and name alignment before a long classification run.

---

## Test coverage gaps

| Module | Risk | Missing |
|---|---|---|
| `feature_elimination.py` | Medium | No tests — RFECV behaviour, edge cases (all features selected, n_features_to_select > n_features) |
| `tune_model.py` | Medium | No tests — grid vs random search, build_classifier for each algorithm, evaluate_model metrics |
| `classify_raster.py` | High | No tests — block boundary handling, nodata propagation, selector path |
| `hybrid_assembly.py` | High | No tests — most complex module, ~300-line function |
| `trajectory_analysis.py` | Medium | No tests — Sankey output, impossible-transition enforcement |
| `seam_healing.py` | Medium | No tests — edge cases for 1px-wide seams, probability spike suppression |
| `planetary_computer.py` | Low | Not testable without credentials, but AOI parsing (`_aoi_search_arguments`) is testable offline |

The biggest risk is `classify_raster` having no tests. A block-boundary off-by-one or nodata leak would silently corrupt output rasters.

---

## Interface surface (`__all__`) — Too flat

`__all__` exports ~70 symbols. It mixes:
- High-level entry points (`load_and_split_training_data`, `create_feature_stack`, `tune_model`, `classify_raster`)
- Low-level building blocks (`open_raster`, `drop_nan_samples`, `extract_pixels_from_shapefile`)
- Intermediate helpers (`local_mean_variance`, `local_variance`, `probs_to_logits`)
- Domain constants (`DEFAULT_TIMOR_17_CLASSES`, `DEFAULT_TIMOR_18_CLASSES`, `TRAJECTORY_CLASSES`, etc.)

The flat namespace makes the intended entry points for a new user invisible. Domain constants from a specific use case (Timor-Leste) are exported as top-level package symbols.

**Recommendation:**
1. Move Timor-specific constants to a `timor_defaults` sub-namespace or a dedicated `defaults.py` module — let users access them explicitly when needed but not encounter them while scanning the API.
2. Consider grouping intermediate helpers (`local_mean_variance`, `probs_to_logits`, etc.) under a `ML_LC_Classifier.internals` or just removing them from `__all__` if they aren't used outside `process_tile`.

---

## Compression and nodata inconsistencies

| Module | Default compress | Default nodata |
|---|---|---|
| `classify_raster` | `deflate` | `0` |
| `feature_stack` | `deflate` | `NaN` |
| `mosaic_rasters` | **`lzw`** | `0` |
| `stack_rasters` | `deflate` | preserves source |
| `refine_lulc_with_mask_layer` | `deflate` | source |
| `distance_features` | `deflate` | `NaN` |

`mosaic_rasters` defaulting to `lzw` while everything else uses `deflate` is a noted inconsistency. It isn't wrong, but a caller building a pipeline needs to know this or end up with mixed compression in their output stack.

**Recommendation:** Standardise on `deflate` across all write paths, or document the mismatch explicitly in both `mosaic_rasters` and `stack_rasters` docstrings.

---

## `print()` side effects in production code

`refine_lulc_with_mask_layer` and `process_tile` both emit multi-line progress banners unconditionally via `print()`. All other modules are silent. This breaks the "no side effects on import" principle conceptually and makes these functions awkward to use in notebook or service contexts where output is captured.

**Recommendation:** Replace `print()` calls with `logging.getLogger(__name__).info(...)` throughout, or add a `verbose: bool = True` parameter that gates all console output.

---

## Summary of priority actions

| Priority | Change | Benefit |
|---|---|---|
| 1 | Add `classify_raster_proba` | Fills the functional gap for the entire probability-based pipeline |
| 2 | Add tests for `classify_raster`, `tune_model`, `feature_elimination` | Protects the most-used modules from regressions |
| 3 | Extract `ProcessTileResult` dataclass from `process_tile` | Aligns with the package's own conventions; unbreaks the `dict` return |
| 4 | Move `DownloadedAsset` out of `planetary_computer.py` | Removes the hidden eager `geopandas` import on any import |
| 5 | Extract shared mask-evaluation helper | Fixes code duplication between `refine_lulc_with_mask_layer` and `process_tile` |
| 6 | `calculate_class_area` → separate compute + plot | Removes mandatory `matplotlib` for headless use |
| 7 | Replace `print()` with `logging` | Makes the two verbose functions consistent with the rest |
| 8 | Deprecate `open_raster` / `load_and_extract` as public functions | Closes the resource-leak footgun |
| 9 | Thin `__all__` — group or remove Timor constants and intermediate helpers | Makes the API surface intelligible to new users |
| 10 | Add `validate_feature_stack` utility | Catches band-order mismatches before long classify runs |
