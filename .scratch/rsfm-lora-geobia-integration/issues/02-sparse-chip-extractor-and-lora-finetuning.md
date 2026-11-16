# 02: Sparse-Polygon Chip Extractor and LoRA Fine-Tuning with Checkpoint Bundle

**What to build:** Allow users to extract fixed-size multi-band/multi-seasonal training chips centered on sparse ground-truth vector polygons or points (with unlabeled background pixels masked as `ignore_index=0`), fine-tune an OpenGeoAI/RSFM backbone using Low-Rank Adaptation (LoRA) and a lightweight pixel classification head, and save/reload a portable checkpoint bundle containing the learned LoRA adapters, head weights, and band/normalization metadata.

**Blocked by:** 01: OpenGeoAI Backbone Loader and Windowed Custom Embedding Stack

**Status:** ready-for-agent

- [ ] Passing a raster stack and vector training layer extracts fixed-size (`chip_size`, e.g. `224x224`) image chips and rasterized target masks where labeled geometries carry integer class IDs `1..K` and unlabeled pixels carry `0` (`ignore_index=0`), reprojecting vectors to the raster CRS when needed.
- [ ] Running LoRA fine-tuning freezes the base foundation model weights, injects trainable low-rank adapter matrices (`rank`, `alpha`) plus a task classification head, and updates only the adapter and head parameters while masking loss on `ignore_index=0` pixels.
- [ ] Saving a fine-tuned model writes a self-contained checkpoint directory/archive (adapter weights, head weights, class IDs, band/wavelength order, and per-band mean/std normalization statistics).
- [ ] Reloading the saved checkpoint bundle reproduces identical adapted embeddings and class probability predictions on held-out test chips.
