# Methodology Guide: Hybrid Pixel-Object Land Cover Post-Processing & Validation

This document describes the methodological framework, mathematical formulations, and spatial decision logic implemented across the `ml-lc-classifier` toolkit, namely the hybrid-pixel object classification assembly.

---

## 1. Architectural Overview & Workflow Chain

The system operates on an asymmetric hybrid model where **pixel-based machine learning classifiers** capture subtle spectral signatures across feature stacks, while **object segmentations** constrain high-frequency spatial noise (salt-and-pepper artifacts) and reinforce contextual boundaries.

```
+-------------------------------------------------------------------------+
| Upstream: Earth Engine / CCDC (agilfahrezy11/Temporal_segment_gee)      |
| or Local ML Classifier (classify_raster_proba)                          |
+-------------------------------------------------------------------------+
                                    |
                  +-----------------+-----------------+
                  |                                   |
                  v                                   v
       [Probability GeoTIFF]                  [Superpixel Mask]
       Multi-band uint16 (0-10000)            SNIC / SLIC segment IDs
                  |                                   |
                  +-----------------+-----------------+
                                    |
                                    v
                     +-----------------------------+
                     | 1. Seam Healing             |
                     | - Border smoothing          |
                     | - Probability spike dampening|
                     +-----------------------------+
                                    |
                                    v
                     +-----------------------------+
                     | 2. Hybrid Object Blending   |
                     | - Bayesian blending         |
                     | - Adaptive stability        |
                     | - Confusion Index (CI)      |
                     +-----------------------------+
                                    |
                                    v
                     +-----------------------------+
                     | 3. Auxiliary Mask Rules     |
                     | - River corridors & water   |
                     | - Urban / road buffers      |
                     +-----------------------------+
                                    |
                                    v
                     +-----------------------------+
                     | 4. MMU Spatial Filtering    |
                     | - Morphological sieve       |
                     | - Class-specific thresholds |
                     +-----------------------------+
                                    |
                                    v
                     +-----------------------------+
                     | 5. Multi-Epoch Trajectory   |
                     | - Bidirectional consistency |
                     | - Flip-back QA flag         |
                     | - Pontius Net/Swap & Sankey |
                     +-----------------------------+
                                    |
                                    v
                     +-----------------------------+
                     | 6. Accuracy Assessment      |
                     | - Olofsson area adjustment  |
                     | - Error-adjusted ha & 95% CI|
                     +-----------------------------+
```

---

## 2. Mathematical Formulations & Steps

### 2.1. Object-Pixel Bayesian Blending (`hybrid_assembly.py`)

For each superpixel segment $S_k$ comprising $N_k$ pixels, the mean segment probability for class $c \in \{1, \dots, C\}$ is computed as:

$$\bar{P}_{S_k}(c) = \frac{1}{N_k} \sum_{i \in S_k} P_{pixel, i}(c)$$

Rather than using a fixed static blending weight across all landscape features, the module calculates an **adaptive spatial stability metric** $\lambda_i$:

$$\Delta P_i = P_{max, 1}(i) - P_{max, 2}(i)$$

$$\lambda_i = \frac{1}{1 + \exp\left(-\gamma \cdot (\Delta P_i - \tau)\right)}$$

where:
- $\Delta P_i$ is the margin between the winning class probability and the runner-up class probability.
- $\tau$ is the stability threshold (default: $0.25$).
- $\gamma$ is the adaptive sensitivity scaling factor (default: $10.0$).

The hybrid probability $P_{hybrid, i}(c)$ combines the pixel prior and the segment likelihood:

$$P_{hybrid, i}(c) = \lambda_i \cdot P_{pixel, i}(c) + (1 - \lambda_i) \cdot \bar{P}_{S_k}(c)$$

#### Confusion Index ($CI$)
To provide an explicit spatial uncertainty layer, the Confusion Index is computed at each pixel:

$$CI_i = \frac{S_{max, i}}{W_{max, i}}$$

where $W_{max, i}$ is the highest probability (winner) and $S_{max, i}$ is the second highest probability (runner-up). 
- A $CI \approx 0.0$ signifies strong confidence (unambiguous winner).
- A $CI \approx 1.0$ indicates maximum classification ambiguity.

---

### 2.2. Minimum Mapping Unit (MMU) Morphological Filtering (`lc_post_process.py` / `mmu.py`)

Single-pixel misclassifications are resolved using a class-specific **connected-component sieve filter**:
1. Binary connected components are extracted for each class $c$.
2. Patches with an area smaller than $A_{min}(c)$ (specified in `MMU_TABLE`, e.g., $0.5\text{ ha}$ for forest, $0.1\text{ ha}$ for urban structures) are flagged.
3. Flagged patches are iteratively infilled using a focal majority filter restricted to the surrounding valid neighborhood, preserving genuine landscape boundaries.

---

### 2.3. Multi-Epoch Trajectory & Temporal Consistency (`trajectory_analysis.py`)

Land cover transitions across chronological epochs $(t_1, t_2, \dots, t_T)$ are validated against ecological transition matrices:

1. **Bidirectional Cascade Consistency:**
   - **Backward Cascade (`anchor="last"`):** Starting from the highest-confidence recent epoch (e.g., ground-truthed 2025), impossible predecessor states (e.g., Cropland $\rightarrow$ Primary Forest within 5 years) are corrected retroactively.
   - **Forward Cascade (`anchor="first"`):** Propagates forward from a baseline epoch.
2. **Multi-Epoch Flip-Back Detection (QA Flag):**
   Pixels that fluctuate temporarily before reverting to their original state:
   
   $$\text{FlipBack} = (y_{t_1} = y_{t_T}) \land \exists \, k \in (1, \dots, T-1) : y_{t_k} \neq y_{t_1}$$
   
   are tagged as spurious noise (Class 9) rather than genuine rotational land-use change.
3. **Transition Dynamics Decomposition (Pontius et al., 2004):**
   Decomposes total change between periods into:
   - **Quantity Difference (Net Component):** Overall expansion or contraction of class totals.
   - **Allocation Difference (Swap Component):** Simultaneous loss and gain in different spatial locations with zero net area change.

---

### 2.4. Error-Adjusted Area Estimation & Accuracy (`accuracy_assessment.py`)

Traditional pixel-counting from classified maps is inherently biased by omission and commission errors. This module strictly adheres to the **Olofsson et al. (2013, 2014)** and **Stehman (2014)** standards:

#### 1. Estimated Area Proportion Matrix ($p_{ij}$)
Given mapped stratum weight $W_i = \frac{A_{map, i}}{A_{total}}$ and sample error matrix counts $n_{ij}$ (where row $i$ is map class, column $j$ is reference class):

$$p_{ij} = W_i \cdot \frac{n_{ij}}{n_{i.}}$$

#### 2. User's and Producer's Accuracies
$$UA_i = \frac{p_{ii}}{p_{i.}}, \qquad PA_j = \frac{p_{jj}}{p_{.j}}$$

#### 3. Error-Adjusted Class Area ($\hat{A}_j$)
The true area of class $j$ is estimated from the marginal column proportions:

$$\hat{A}_j = A_{total} \cdot \sum_{i=1}^{K} p_{ij}$$

#### 4. Standard Error & 95% Confidence Intervals
$$S(\hat{A}_j) = A_{total} \cdot \sqrt{\sum_{i=1}^{K} W_i^2 \cdot \frac{n_{ij}}{n_{i.}} \cdot \left(1 - \frac{n_{ij}}{n_{i.}}\right) \cdot \frac{1}{n_{i.} - 1}}$$

$$\text{CI}_{95\%} = \hat{A}_j \pm 1.96 \cdot S(\hat{A}_j)$$

---

## 3. References

1. **Olofsson, P., Foody, G. M., Stehman, S. V., & Woodcock, C. E. (2013).** *Making better use of accuracy data in land change studies: Estimating accuracy and area and quantifying uncertainty.* Remote Sensing of Environment, 129, 122-131.
2. **Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., & Wulder, M. A. (2014).** *Good practices for estimating area and assessing accuracy of land change.* Remote Sensing of Environment, 148, 42-57.
3. **Pontius Jr, R. G., Shusas, E., & McEachern, M. (2004).** *Detecting important categorical land changes while accounting for persistence.* Agriculture, Ecosystems & Environment, 101(2-3), 251-268.
4. **Stehman, S. V. (2014).** *Estimating area and map accuracy for stratified random sampling when the strata are different from the map classes.* International Journal of Remote Sensing, 35(13), 4923-4939.
5. **Malinverni, Eva S., Sanità, M. (2026).** *A Hybrid Pixel–Object–Rule-Based Classification Framework with Stability Maps for Large-Scale LULC Mapping.*
ISPRS International Journal of Geo-Information, 15(6), 241.