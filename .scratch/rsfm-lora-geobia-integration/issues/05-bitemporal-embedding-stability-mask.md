# 05: Bi-Temporal Embedding Cosine Stability Mask and Sample Disturbance Filter

**What to build:** Replace CCDC harmonic break detection by streaming two epochs' rasters through the RSFM backbone to compute pixel-wise or SNIC-segment-wise Embedding Cosine Distance, outputting both a continuous change-magnitude raster and a binary stability mask (`TL_stable_mask`) for multi-epoch trajectory consistency, as well as filtering out disturbed training samples between epochs.

**Blocked by:** 01: OpenGeoAI Backbone Loader and Windowed Custom Embedding Stack

**Status:** done

- [x] Given two epochs' input rasters (or embedding stacks), the stability engine computes $d_{\text{cos}}(z_{t_1}, z_{t_2}) = 1 - \frac{z_{t_1} \cdot z_{t_2}}{\|z_{t_1}\| \, \|z_{t_2}\|}$ in memory-safe streaming windows (or per SNIC segment) and writes both the continuous cosine distance GeoTIFF (`float32` in `[0, 2]`) and the binary stability mask (`uint8`, `1` where $d_{\text{cos}} \le \tau_{\text{stable}}$, `0` where disturbed or nodata).
- [x] Passing a training vector layer and two epochs' embeddings filters out sample points/polygons whose cosine distance exceeds a user-defined threshold (`max_change_distance`), returning the purified training samples and a flag column.
- [x] The generated binary stability mask integrates directly with the multi-epoch trajectory consistency functions in `trajectory_analysis` to gate unlikely land-cover transitions.
