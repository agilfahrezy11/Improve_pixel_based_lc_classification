# 04: GEOBIA SNIC Segment-Pooled RSFM Embeddings and Object Probability Projection

**What to build:** Allow users to pool custom RSFM embeddings and manual ancillary features (mean and optional standard deviation) within Unified SNIC Superpixel segments (`seg_path`) to create a compact segment-level feature table for rapid CPU classifier training, and project fitted segment-level class probabilities back onto raster tiles for hybrid pixel-object assembly.

**Blocked by:** 01: OpenGeoAI Backbone Loader and Windowed Custom Embedding Stack

**Status:** done

- [x] Given an input raster (or RSFM embedding stack), optional manual ancillary layers, and a SNIC segment raster, the segment pooling function computes per-segment feature statistics (mean and optional std) in streaming windows while excluding nodata pixels and properly preserving negative wrapped 32-bit GEE segment IDs (`< -1`).
- [x] Training polygons/points intersecting segments can assign segment-level training labels (via majority/point-in-segment matching) so `tune_model` (`RandomForest`, `LightGBM`, `XGBoost`) trains directly on the compact segment feature table.
- [x] Predicted per-segment class probabilities map cleanly back to a multi-band probability raster aligned with the segment grid for downstream hybrid assembly or thematic output.
