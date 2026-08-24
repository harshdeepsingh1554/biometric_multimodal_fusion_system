# Biometric Evaluation System Architecture & Methodology Report

## Executive Summary

This report provides a detailed breakdown of how image embeddings are extracted, stored, and evaluated across the multimodal biometric framework (`face`, `finger`, `iris`). It clarifies the exact role of **averaged templates** versus **individual image embeddings (live sets)**, how genuine/impostor matrices are generated for both unimodal and fused evaluations, how Equal Error Rate (EER) is computed, and what the current benchmark results reveal about system performance.

---

## 1. Image Embedding Extraction Pipeline

### 1.1 Multi-Image Extraction (`main.py`)
For each enrolled person in a standardized multi-image dataset layout (e.g. `data/all_images/setA_std`):
- Every person directory contains three modality subfolders with multiple images:
  - `face/`: e.g. 14 images (`face_01.jpg` ... `face_14.jpg`)
  - `finger/`: e.g. 8 images (`finger_01.jpg` ... `finger_08.jpg`)
  - `iris/`: e.g. 10 images (`iris_01.jpg` ... `iris_10.jpg`)

The pipeline loads extractors once and processes all images:
1. **Face Extractor** (`ArcFace` + `SCRFD` detector): outputs 512-dimensional unit vector.
2. **Fingerprint Extractor** (`DeepPrint` PyTorch model): outputs 512-dimensional unit vector.
3. **Iris Extractor** (`ArcIris ResNet100`): outputs 512-dimensional unit vector.

### 1.2 Dataset Partition Strategy
A subset of images per modality (e.g., first 3) is averaged to form the **enrolled template**. The remaining images are stored as a **live set** (individual probe embeddings per image). If a modality has fewer images than needed for a live set iteration, earlier images are cycled/repeated to fill the gap.

---

## 2. Averaging vs. Single Image Storage

The system maintains **two complementary storage levels** in SQLite (`database/biometric.db` or `database/biometric_final.db`):

```
                          ┌─────────────────────────────────────────┐
                          │    Person Subfolder (e.g. Person_001)   │
                          └────────────────────┬────────────────────┘
                                               │
                       ┌───────────────────────┴──────────────────────┐
                       ▼                                              ▼
          ┌──────────────────────────┐                  ┌──────────────────────────┐
          │   Averaged Template      │                  │    Live Set Embeddings    │
          │   (EMBEDDINGS table)     │                  │  (live_embeddings table) │
          └────────────┬─────────────┘                  └────────────┬─────────────┘
                       │                                             │
         Mean-pool N images → L2-normalize             Per-image float32 BLOBs
         v_avg = normalize(mean(v_1..v_N))              stored as (live_index, embedding)
```

### 2.1 The `EMBEDDINGS` / `templates` Table (Averaged Template)
- **Calculation**: Mean-pooling across all $N$ enrollment images for a person's modality, followed by L2-normalization:
  $$\mathbf{v}_{\text{template}} = \frac{\mathbf{v}_{\text{mean}}}{\|\mathbf{v}_{\text{mean}}\|_2}, \quad \mathbf{v}_{\text{mean}} = \frac{1}{N} \sum_{i=1}^N \mathbf{v}_i$$
- **Purpose**: Represents the core **Enrolled Template** used in 1:1 verification and 1:N identification.

### 2.2 The `live_embeddings` Table (Live Set Probe Embeddings)
- **Calculation**: Every single image embedding $\mathbf{v}_i$ in the live set is saved as a `float32` BLOB with an associated `live_index`.
- **Purpose**: Provides probe vectors for evaluation. During evaluation, all live set embeddings for a person are compared against enrolled templates to compute genuine scores (using multiple probes per genuine pair for richer statistical estimates).

---

## 3. Evaluation Methodology (`biometric_eval_matrix.py`)

When `biometric_eval_matrix.py` executes, it strictly separates **Enrollment** and **Probe** to avoid data leakage:

### 3.1 Enrollment Vector vs. Probe Vectors per Person
- **Enrollment Vector ($\mathbf{e}_i$)**: Retrieved from the `EMBEDDINGS` / `templates` table (the pre-computed averaged template for Person $i$).
- **Probe Vectors ($\mathbf{p}_{i,k}$)**: Retrieved from `live_embeddings` as **all live-set embeddings** for Person $i$. If no live set exists, the enrollment vector itself is used as a single probe fallback.
- Multiple probes per genuine pair are supported: `gen_cos` accumulates scores from **all** `k` live set samples vs. the enrolled template.

### 3.2 Score Computation
For each probe $\mathbf{p}_{i,k}$ of Person $i$ compared against enrollment template $\mathbf{e}_j$ of Person $j$:
$$S_{i,j,k} = \text{CosineSimilarity}(\mathbf{p}_{i,k}, \mathbf{e}_j) = \hat{\mathbf{p}}_{i,k} \cdot \hat{\mathbf{e}}_j$$

The similarity matrix entry $\mathbf{S}_{i,j}$ is the **mean** across all $k$ probe samples.

---

## 4. Matrix Generation & Genuine / Impostor Partitioning

For $N$ enrolled persons, an $N \times N$ similarity matrix $\mathbf{S}$ is constructed per modality:

```
                                    Enrollment Templates (j = 1..N)
                              Person_001    Person_002   ...   Person_N
                            ┌─────────────┬─────────────┬─────┬─────────────┐
                Person_001  │   GENUINE   │  Impostor   │ ... │  Impostor   │
                            ├─────────────┼─────────────┼─────┼─────────────┤
    Probe       Person_002  │  Impostor   │   GENUINE   │ ... │  Impostor   │
    Samples     
    (i = 1..N)      :       │     :       │      :      │  \  │      :      │
                            ├─────────────┼─────────────┼─────┼─────────────┤
                Person_N    │  Impostor   │  Impostor   │ ... │   GENUINE   │
                            └─────────────┴─────────────┴─────┴─────────────┘
```

1. **Diagonal Cells ($i = j$) → Genuine Scores**: Compares all of Person $i$'s probe(s) against Person $i$'s enrolled template. Total genuine pairs = $N \times K$ (where $K$ = number of live set images per person).
2. **Off-Diagonal Cells ($i \neq j$) → Impostor Scores**: Compares Person $i$'s probe(s) against Person $j$'s template. Total impostor pairs = $N(N-1) \times K$.

Both **cosine similarity matrices** and **Euclidean distance matrices** are computed and saved separately.

---

## 5. Threshold Sweep, EER, and Journey Matrix

### 5.1 Equal Error Rate (EER) Calculation
1. Sweeps 1,000 threshold points $t \in [\min(\mathbf{S}), \max(\mathbf{S})]$.
2. For each threshold $t$:
   $$\text{FAR}(t) = \frac{\sum \mathbb{I}(\text{ImpostorScores} \ge t)}{|\text{ImpostorPairs}|}, \quad \text{FRR}(t) = \frac{\sum \mathbb{I}(\text{GenuineScores} < t)}{|\text{GenuinePairs}|}$$
3. **EER Threshold ($t_{\text{EER}}$)** selected where $|\text{FAR}(t) - \text{FRR}(t)|$ is minimized.
4. **EER** = $(\text{FAR}(t_{\text{EER}}) + \text{FRR}(t_{\text{EER}})) / 2$.

Both cosine (higher = match) and Euclidean distance (lower = match, `higher_is_match=False`) sweeps are run.

### 5.2 Per-Person Metrics
For each person $i$, a local threshold sweep is run across their individual row in the cosine matrix:
- `genuine_score`: cosine similarity of Person $i$'s probe vs. their own enrolled template.
- `mean_impostor_score`: mean cosine similarity vs. all other persons' templates.
- `local_threshold`, `local_FAR`, `local_FRR`: person-specific EER operating point.

Saved as `<modality>_per_person_metrics.csv`.

### 5.3 Threshold Journey Matrix (`<modality>_threshold_journey_matrix.csv`)
A $1000 \times (N+4)$ dataset table tracks the step-by-step impact of threshold changes on every person:
- **Columns**: `threshold`, `is_EER_thr`, `Person_001` ... `Person_N`, `n_accepted`, `n_rejected`.
- **Row Journey**:
  - **Permissive Phase** (low threshold): FAR ≈ 1.0, FRR ≈ 0.0. All persons `ACCEPT`.
  - **Optimal EER Phase** (EER threshold): Crossover point — optimal decision boundary.
  - **Strict Phase** (high threshold): FAR ≈ 0.0, FRR ≈ 1.0. Almost all persons `REJECT`.

---

## 6. Fused Evaluation Methodology (`fused_eval_matrix.py`)

Fusion evaluation operates on concatenated or CBP-fused embeddings stored in `fused_templates` / `fused_live_embeddings` tables. The pipeline is identical in structure to the unimodal evaluation but operates on higher-dimensional fused vectors:

- **Concat-L2**: 1536-dimensional L2-normalized vector (face + finger + iris concatenated).
- **CBP Fused**: Compact bilinear product of all three modalities projected to `output_dim`.

Both `raw` template embeddings and `biohash`-protected templates can be evaluated by specifying `--template_type`.

---

## 7. BioHashing Evaluation Layer

After raw embeddings are enrolled, `biohashing.py` applies Traditional BioHashing to produce **cancelable templates**:
- **`biohash_embeddings`**: BioHashed unimodal enrolled templates.
- **`biohash_live_embeddings`**: BioHashed live set probe embeddings.
- **`biohash_fused`**: BioHashed fused (CBP/Concat) enrolled templates.

The `biometric_eval_matrix.py` and `fused_eval_matrix.py` both support `--template_type biohash` to evaluate on the protected template space, allowing direct EER comparison between raw and biohashed modalities.

---

## 8. Summary of Current Benchmark Results

### 8.1 Unimodal Results — from `biometric_eval_matrix.py` (80 persons, setA)

| Modality | Genuine Pairs | Impostor Pairs | Genuine Mean | Impostor Mean | Cosine EER | EER Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Face** | 80 | 6,320 | 0.8916 | 0.0304 | **0.02%** | 0.3481 |
| **Iris** | 80 | 6,320 | 0.5654 | 0.2159 | **5.94%** | 0.3591 |
| **Finger** | 80 | 6,320 | 0.9616 | 0.2871 | **2.73%** | 0.8598 |
| **Overall Avg** | — | — | — | — | **2.90%** | 0.5223 |

### 8.2 Extended Unimodal Results — from `results.json` (multi-probe evaluation)

| Modality | Genuine Pairs | Impostor Pairs | Genuine Mean | Impostor Mean | EER | EER Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Face** | 5,408 | 1,770 | 0.8254 | 0.1086 | **0.00%** | 0.4535 |
| **Finger** | 1,680 | 1,770 | 0.8893 | 0.6000 | **0.018%** | 0.8343 |
| **Iris** | 2,700 | 1,770 | 0.2683 | 0.3415 | **0.00%** | 0.3101 |
| **Fused (CBP)** | 1,680 | 1,770 | 0.6630 | 0.3621 | **0.029%** | 0.5051 |

> **Note on Iris EER Anomaly**: The iris modality shows `genuine_mean (0.268) < impostor_mean (0.341)` in the extended evaluation, indicating the iris ResNet100 embeddings exhibit **inverted similarity ordering** in this run. This is consistent with iris being the weakest modality (EER=5.94% in the 80-person eval) and suggests the ArcIris model weights may not be fully optimized for this specific dataset distribution or image quality. The `gabor` backend's classical Masked Hamming Distance would be more reliable for this modality.

### 8.3 Fused Performance Summary

| Metric | CBP Fused (Raw) | Notes |
| :--- | :--- | :--- |
| **Genuine Mean** | 0.6630 | Intermediate (between face 0.89 and iris 0.27) |
| **Impostor Mean** | 0.3621 | Higher than face (0.03) due to iris noise |
| **EER** | **0.029%** | Better than finger (0.018%? — see multi-probe note) |
| **EER Threshold** | 0.5051 | |
| **FMR@1%** | FNMR=4.3% | at threshold 0.5231 |
| **FMR@0.1%** | FNMR=11.4% | at threshold 0.5716 |

---

## 9. Key Evaluation Design Decisions

### 9.1 Why Live Set, Not a Fixed Last Image
The current system stores a **live set** of multiple probe images per person rather than a single held-out sample. Each probe image in the live set contributes an independent genuine score when compared against the enrolled template. This produces:
- Richer genuine score distributions (multiple data points per person).
- More statistically robust EER estimates.
- Realistic simulation of a deployed verification system receiving fresh captures.

### 9.2 Why Mean-Pool for Enrollment, Not Concatenation
Mean-pooled enrollment templates preserve the same 512-dimensional embedding space as individual probe embeddings, enabling direct cosine similarity comparison. Concatenation would require a fixed number of images at enrollment and would produce embeddings of different dimensionality than single-probe queries.

### 9.3 Handling Modality Imbalance in Live Sets
If a modality has fewer images available than a given live set index requires, earlier images are repeated (cycled). This ensures every person has the same number of live set entries across all modalities, maintaining balanced evaluation matrices.

---

*Note: Evaluation was performed on dataset `setA` using SQLite databases `database/biometric.db` and `database/biometric_final.db`. The `biohash` template type evaluations require running `biohashing.py` first to populate `biohash_embeddings` and `biohash_live_embeddings` tables.*
