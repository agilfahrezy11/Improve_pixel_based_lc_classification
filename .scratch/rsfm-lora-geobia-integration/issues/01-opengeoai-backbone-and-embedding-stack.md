# 01: OpenGeoAI Backbone Loader and Windowed Custom Embedding Stack

**What to build:** Allow users to install an optional foundation-model dependency group, load a pre-trained foundation model from the OpenGeoAI catalog (or a lightweight deterministic CPU backbone for offline testing), and stream any multi-band or multi-seasonal raster through the backbone with 2D Gaussian overlap blending to produce a custom embedding GeoTIFF and JSON manifest that plug directly into the existing training data extraction and shallow model tuning functions.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Installing the package with the optional `rsfm` extra pulls in `geoai-py`, `torch`, and `peft`, while the default installation requires none of them and continues to import cleanly on CPU-only environments.
- [ ] Users can list and load foundation model backbones through an OpenGeoAI-backed loader or supply a lightweight deterministic CPU backbone implementing the same feature-extraction protocol.
- [ ] Running windowed embedding extraction on a source raster slides overlapping chips weighted by a 2D Gaussian/Hann window to prevent patch-boundary seams, preserves nodata masks, and returns a `FeatureStackResult` (GeoTIFF + JSON manifest) with ordered embedding band names.
- [ ] The resulting `FeatureStackResult` passes directly into `load_and_split_training_data` and `tune_model` in an end-to-end CPU test.
