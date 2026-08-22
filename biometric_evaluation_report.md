# Biometric Evaluation System Architecture & Methodology Report

## Executive Summary

This report provides a detailed breakdown of how image embeddings are extracted, stored, and evaluated across our multimodal biometric framework (`face`, `finger`, `iris`). It clarifies the exact role of **averaged templates** versus **single individual image embeddings**, how genuine/impostor matrices are generated, and how Equal Error Rate (EER) and threshold journeys are computed.

---

## 1. Image Embedding Extraction Pipeline

### 1.1 Multi-Image Extraction (`main.py`)
For each person (`Person_001` through `Person_080`) in a standardized multi-image dataset layout (e.g. `data/all_images/setA_std`):
- Every person directory contains three modality subfolders:
  - `face/`: 14 images (`face_01.jpg` ... `face_14.jpg`)
  - `finger/`: 8 images (`finger_01.jpg` ... `finger_08.jpg`)
  - `iris/`: 10 images (`iris_01.jpg` ... `iris_10.jpg`)

The pipeline loads extractors once:
1. **Face Extractor** (`ArcFace` + `SCRFD` detector): outputs a 512-dimensional vector.
2. **Fingerprint Extractor** (`DeepPrint` PyTorch model): outputs a 512-dimensional vector.
3. **Iris Extractor** (`ArcIris ResNet100` architecture): outputs a 512-dimensional vector.

For every image found in a person's subfolder, the extractor outputs an un-normalized or L2-normalized 512-dim embedding array.

---

## 2. Averaging vs. Single Image Storage

Our system maintains **two complementary storage levels** in SQLite (`database/biometric.db`), ensuring **zero data/precision loss**:

```
                              ┌─────────────────────────────────────────┐
                              │    Person Subfolder (e.g. Person_001)   │
                              └────────────────────┬────────────────────┘
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
            ┌─────────────────────────┐                         ┌─────────────────────────┐
            │  Individual Embeddings  │                         │    Averaged Template    │
            │  (image_embeddings tab) │                         │    (templates table)    │
            └────────────┬────────────┘                         └────────────┬────────────┘
                         │                                                   │
           Stores ALL 14 face, 8 finger,                      Calculates mean pool across all N
           & 10 iris vectors independently                    embeddings and L2-normalizes:
           as exact float32 BLOBs.                            v_avg = normalize(mean(v_1..v_N))
```

### 2.1 The `templates` Table (Averaged Template)
- **Calculation**: Mean-pooling across all $N$ image vectors for a person's modality, followed by L2-normalization:
  $$\mathbf{v}_{\text{mean}} = \frac{1}{N} \sum_{i=1}^N \mathbf{v}_i, \quad \mathbf{v}_{\text{template}} = \frac{\mathbf{v}_{\text{mean}}}{\|\mathbf{v}_{\text{mean}}\|_2}$$
- **Purpose**: Represents the core **Enrolled Template** in 1:1 verification and 1:N identification.

### 2.2 The `image_embeddings` Table (Individual Raw Embeddings)
- **Calculation**: Every single image embedding $\mathbf{v}_i$ ($i = 0 \dots N-1$) is saved untouched as a `float32` BLOB.
- **Purpose**: Preserves full un-averaged data for held-out probe testing, detailed pair-wise score distributions, and dataset audits.

---

## 3. Evaluation Methodology (`biometric_eval_matrix.py`)

When `biometric_eval_matrix.py` executes, it strictly separates **Enrollment** and **Probe** to avoid data leakage:

### 3.1 Enrollment Vector vs. Probe Vector per Person
- **Enrollment Vector ($\mathbf{e}_i$)**: Retrieved from the `templates` table (the pre-computed averaged template for Person $i$).
- **Probe Vector ($\mathbf{p}_i$)**: Retrieved from the `image_embeddings` table as the **LAST image embedding** ($\mathbf{v}_{N-1}$) for Person $i$.
  - *Why the last image?* The probe is treated as a held-out sample representing a live verification query.

---

## 4. Matrix Generation & Genuine / Impostor Partitioning

For 80 enrolled persons, an $80 \times 80$ similarity matrix $\mathbf{S}$ is constructed for each modality:

$$\mathbf{S}_{i,j} = \text{CosineSimilarity}(\mathbf{p}_i, \mathbf{e}_j) = \frac{\mathbf{p}_i \cdot \mathbf{e}_j}{\|\mathbf{p}_i\|_2 \|\mathbf{e}_j\|_2}$$

```
                                    Enrollment Templates (j = 1..80)
                              Person_001    Person_002   ...   Person_080
                            ┌─────────────┬─────────────┬─────┬─────────────┐
                Person_001  │   GENUINE   │  Impostor   │ ... │  Impostor   │
                            ├─────────────┼─────────────┼─────┼─────────────┤
    Probe       Person_002  │  Impostor   │   GENUINE   │ ... │  Impostor   │
    Samples     
    (i = 1..80)     :       │     :       │      :      │  \  │      :      │
                            ├─────────────┼─────────────┼─────┼─────────────┤
                Person_080  │  Impostor   │  Impostor   │ ... │   GENUINE   │
                            └─────────────┴─────────────┴─────┴─────────────┘
```

1. **Diagonal Cells ($i = j$) -> Genuine Scores**:
   - Compares Person $i$'s probe against Person $i$'s averaged enrolled template.
   - Total Genuine Pairs = 80.
2. **Off-Diagonal Cells ($i \neq j$) -> Impostor Scores**:
   - Compares Person $i$'s probe against Person $j$'s averaged enrolled template ($j \neq i$).
   - Total Impostor Pairs = $80 \times 79 = 6,320$.

---

## 5. Threshold Sweep, EER, and Journey Matrix

### 5.1 Equal Error Rate (EER) Calculation
1. Sweeps 1,000 threshold points $t \in [\min(\mathbf{S}), \max(\mathbf{S})]$.
2. For each threshold $t$:
   $$\text{FAR}(t) = \frac{\sum \mathbb{I}(\text{ImpostorScores} \ge t)}{\text{Total Impostors (6,320)}}, \quad \text{FRR}(t) = \frac{\sum \mathbb{I}(\text{GenuineScores} < t)}{\text{Total Genuines (80)}}$$
3. **$\text{EER}$ Threshold ($t_{\text{EER}}$)** is selected where $|\text{FAR}(t) - \text{FRR}(t)|$ is minimized.

### 5.2 Threshold Journey Matrix (`<modality>_threshold_journey_matrix.csv`)
A $1000 \times 87$ dataset table tracks the step-by-step impact of threshold changes on every person:
- **Columns**: `threshold`, `FAR`, `FRR`, `phase` (`HIGH_FAR` / `EER` / `HIGH_FRR`), `is_EER_thr`, `Person_001` ... `Person_080`, `n_accepted`, `n_rejected`.
- **Row Journey**:
  - **Permissive Phase (`threshold` ~ -0.20)**: `FAR` = 1.0, `FRR` = 0.0. All 80 persons are `ACCEPT`.
  - **Optimal EER Phase (`threshold` ~ 0.35)**: Crossover point, optimal decision for system operating threshold.
  - **Strict Phase (`threshold` ~ 0.95)**: `FAR` = 0.0, `FRR` = 0.98. Almost all persons become `REJECT`.

---

## 6. Summary of Current Benchmark Results

| Modality | Genuine Pairs | Impostor Pairs | Genuine Mean | Impostor Mean | Cosine EER | EER Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Face** | 80 | 6,320 | 0.8916 | 0.0304 | **0.02%** | 0.3481 |
| **Iris** | 80 | 6,320 | 0.5654 | 0.2159 | **5.94%** | 0.3591 |
| **Finger** | 80 | 6,320 | 0.9616 | 0.2871 | **2.73%** | 0.8598 |
| **Overall Avg**| - | - | - | - | **2.90%** | **0.5223** |

*Note: Evaluation was performed on dataset `setA` (80 persons) using SQLite database `database/biometric.db`.*
