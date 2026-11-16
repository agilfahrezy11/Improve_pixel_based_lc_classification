# 03: In-Memory Late Ancillary Fusion and Streaming Probability Cube Inference

**What to build:** Allow users to stream large regional or national rasters through a fine-tuned RSFM checkpoint (or a frozen RSFM backbone paired with a fitted tabular classifier), align manual topographic (`Elevation`, `Slope`, `TPI`, `TWI`) and proximity (`dist_road`, `dist_urban`) rasters on the fly via virtual warping, concatenate the RSFM embeddings with the manual ancillary bands in RAM (avoiding multi-gigabyte intermediate disk writes), and write a multi-band class probability GeoTIFF that feeds directly into the existing Bayesian smoother and hybrid pixel-object stability assembly.

**Blocked by:** 01: OpenGeoAI Backbone Loader and Windowed Custom Embedding Stack, 02: Sparse-Polygon Chip Extractor and LoRA Fine-Tuning with Checkpoint Bundle

**Status:** ready-for-agent

- [ ] Ancillary rasters with differing resolutions, extents, or CRS are aligned on the fly to the target streaming window using `WarpedVRT` without requiring pre-warped copies on disk.
- [ ] During windowed inference, the $D$-dimensional RSFM embeddings and $M$ manual ancillary bands are concatenated in memory per block and passed to either the fine-tuned neural head or a fitted tabular classifier (`RandomForest` / `LightGBM` supporting `predict_proba`) to produce per-class probabilities without writing the intermediate $(D+M)$-band raster to disk.
- [ ] The output multi-band probability GeoTIFF (`float32` or scaled `uint16`) preserves spatial metadata, class band order, and nodata masks, and passes directly into both `bayesian_smooth` and `hybrid_assembly.process_tile` in an end-to-end integration test.
