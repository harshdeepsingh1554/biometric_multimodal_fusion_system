# Deep-Dive Technical Research Report: Unimodal Biometric Extractors — Architecture, Preprocessing, Post-Processing & Recommendations

---

## 1. Executive Summary

This report provides a comprehensive, research-grade technical analysis of the unimodal biometric feature extractors (**Face**, **Fingerprint**, and **Iris**) implemented within the `biometrics_fusion` framework. The analysis is based on a full source-code audit of every extractor, model, pipeline, and fusion module. For each modality this document details:

1. **System Architecture & Model Selection**: Core neural network architectures, backbone designs, and algorithmic frameworks.
2. **Model Weights & Pretrained Checkpoints**: Exact weight file paths, sizes (verified), and loading conventions.
3. **Preprocessing & Normalization**: Data loading pipeline, resolution resizing, aspect-ratio preservation, color-space transformations, and statistical normalization.
4. **Post-Processing & Embedding Formats**: Projection layers, dimensionalities, unit-sphere L2-normalization, quality-gating metrics, and template protection integration.
5. **Dual Classical-Pipeline Architecture**: How the iris extractor integrates a fully self-contained Hough + Gabor pipeline alongside ArcIris deep learning.
6. **Fusion & Template Protection**: CBP fusion, Concat-L2 fusion, and BioHashing.
7. **Architectural Gaps & Prioritized Recommendations**: Specific, concrete recommendations for improving accuracy, efficiency, and operational robustness.

---

## 2. Unimodal Modality 1: Face Extractor

### 2.1 Overview & Workflow
The face processing pipeline comprises a two-stage ONNX-based computer vision architecture:
- **Detection & Landmark Estimation**: Single-shot Scale-Controlled Face Detector (SCRFD).
- **Alignment & Feature Extraction**: 5-point canonical similarity transform alignment followed by an ArcFace Deep Convolutional Neural Network.

```
Raw Face Image ──► SCRFD (det_10g.onnx) [1×3×640×640]
                       ├── Stride-8 head  (score, bbox, kps)
                       ├── Stride-16 head (score, bbox, kps)
                       └── Stride-32 head (score, bbox, kps)
                                │
                                ▼ Best-scoring anchor (score > 0.3)
                       5-Point Landmark Coordinates (scaled to original res)
                                │
                                ▼
                   5-Point Affine Alignment → 112 × 112 crop
                                │
                                ▼
                   ArcFace ONNX (w600k_r50.onnx) [1×3×112×112]
                                │
                                ▼
                  Raw 512-d Float32 → L2-Normalize → Unit 512-d Vector
```

### 2.2 Model Architecture & Weights
* **Face Detector**:
  * **Model**: SCRFD (Scale-Controlled Real-Time Face Detector).
  * **Weights File**: [`weights/face/det_10g.onnx`](file:///d:/biometrics_intern/biometrics_fusion/weights/face/det_10g.onnx) — **16.14 MB**.
  * **Input Shape**: `1 × 3 × 640 × 640`.
  * **Anchors & Strides**: Multi-stride feature maps with strides `[8, 16, 32]`. 9 output tensors total (score, bbox, kps per stride).
  * **Anchor Grid**: Reconstructed in Python via `np.meshgrid` per stride. Each anchor repeated twice (`np.repeat(..., 2, axis=0)`) for SCRFD's two-anchors-per-location design.
  * **Detection Score Threshold**: `0.3` (configurable at init via `det_score_threshold`).
* **Face Embedder**:
  * **Model**: ArcFace (Additive Angular Margin Loss DCNN) with ResNet-50 backbone (`r50`).
  * **Weights File**: [`weights/face/w600k_r50.onnx`](file:///d:/biometrics_intern/biometrics_fusion/weights/face/w600k_r50.onnx) — **166.31 MB**, pretrained on the WebFace600K dataset.
  * **Alternative Weights**: [`weights/face/face_extractor_best.pth`](file:///d:/biometrics_intern/biometrics_fusion/weights/face/face_extractor_best.pth) — **16.02 MB** (alternative/fine-tuned checkpoint, not used in production path).
  * **GPU Support**: `ArcFaceONNXModel` auto-selects `CUDAExecutionProvider` if available; SCRFD detector is CPU-only by default (`use_gpu_detector=False`).

### 2.3 Preprocessing & Normalization
1. **Landmark Detection Preprocessing**:
   * Image resized to `640 × 640`.
   * BGR to RGB color conversion.
   * Pixel normalization: $\text{Tensor} = \frac{\text{RGB} - 127.5}{128.0}$ (note: detector uses `/128.0`; embedder uses `/127.5` — different conventions matching each model's original training pipeline).
   * Layout transposed to Channel-First format: `(1, 3, 640, 640)`.
2. **Canonical Alignment**:
   * Uses 5 canonical facial landmarks:
     * Left Eye: `(38.2946, 51.6963)`
     * Right Eye: `(73.5318, 51.5014)`
     * Nose Tip: `(56.0252, 71.7366)`
     * Left Mouth Corner: `(41.5493, 92.3655)`
     * Right Mouth Corner: `(70.7299, 92.2041)`
   * Constructs 10-equation/4-unknown affine system $\mathbf{Ac} = \mathbf{b}$, solved via `np.linalg.lstsq` (least-squares similarity transform).
   * Warps original image to aligned patch of size `112 × 112` via `cv2.warpAffine(..., (112, 112), flags=INTER_LINEAR, borderMode=BORDER_CONSTANT, borderValue=0)`.
3. **Embedder Input Preprocessing**:
   * BGR to RGB color swap.
   * Normalization to range $[-1.0, 1.0]$: $\text{Tensor} = \frac{\text{RGB} - 127.5}{127.5}$.
   * Transpose HWC $\rightarrow$ CHW and add batch dimension → `(1, 3, 112, 112)`.

### 2.4 Post-Processing & Quality Assessment
* **Quality Gating** (optional, `check_quality=False` by default):
  * Evaluated on the aligned `112 × 112` patch using Laplacian variance: $\text{Var}(\nabla^2 \text{Gray}) \ge 60.0$.
  * Raises `ValueError` if blur score falls below threshold.
* **Embedding Post-Processing**:
  * Raw output flattened to 512-dimensional vector.
  * Explicit L2-normalization: $\hat{\mathbf{v}} = \frac{\mathbf{v}}{\max(\|\mathbf{v}\|_2, 10^{-12})}$.
  * Ensures inner products directly compute Cosine Similarity: $\langle \hat{\mathbf{u}}, \hat{\mathbf{v}} \rangle = \cos(\theta)$.

---

## 3. Unimodal Modality 2: Fingerprint Extractor

### 3.1 Overview & Workflow
Fingerprint recognition utilizes a modified DeepPrint Convolutional Neural Network (`FingerprintResNetModel`) designed to extract global ridge structure and local feature representations simultaneously.

```
Raw Scan ──► Aspect-Preserving Padding + Resize (224×224) ──► Grayscale-to-3Ch Replication
                                                                      │
                                                                      ▼
DeepPrintBackbone (ResNet-50, weights=None)
   ├── stem_to_layer2 → shallow features (B, 512, H', W')
   └── deep_layers    → deep features   (B, 2048, H'', W'')
                                                                      │
                                                                      ▼
GAP(shallow)=(B,512) + GAP(deep)=(B,2048) → concat → (B, 2560)
                                                                      │
                                                                      ▼
512-d L2-Norm Embedding ◄── Projector MLP: BN→Identity→Linear(2560→1024)→Identity→BN→Identity→Linear(1024→512)
```

### 3.2 Model Architecture & Weights
* **Primary Extractor Architecture**: `FingerprintResNetModel` in [`models/finger.py`](file:///d:/biometrics_intern/biometrics_fusion/models/finger.py).
  * **Backbone**: `DeepPrintBackbone` based on ResNet-50 without ImageNet weights (`weights=None`).
  * **Multi-Scale Feature Aggregation**:
    * *Shallow Branch*: Output of `layer2` (512 feature channels, fine ridge detail).
    * *Deep Branch*: Output of `layer4` (2048 feature channels, global topological structure).
  * **Pooling & Projection**:
    * Global Average Pooling (`GAP`) applied independently to shallow ($512$) and deep ($2048$) representations.
    * Concatenation into a 2560-dimensional vector: $\mathbf{x}_{\text{concat}} \in \mathbb{R}^{2560}$.
    * Projector MLP: `BN(2560)` → `Identity` → `Linear(2560, 1024)` → `Identity` → `BN(1024)` → `Identity` → `Linear(1024, 512)`.
    * **Note on Identity Placeholders**: `nn.Identity()` layers at indices 1, 3, 5 exist so the `Sequential`'s numbered key names match the original training checkpoint's `state_dict`. Removing them breaks `strict=True` loading.
* **Alternative / Legacy Model Architecture**: `DeepPrintTexMinuModel` (Dual-branch ResNet-18 for Gabor texture + minutiae maps; **currently unused** — dead code).
* **Weights File**: [`weights/finger/finger_extractor_best.pth`](file:///d:/biometrics_intern/biometrics_fusion/weights/finger/finger_extractor_best.pth) — **102.06 MB**. Loaded with PyTorch `strict=True` and `weights_only=True`.

### 3.3 Preprocessing & Normalization
1. **Aspect-Ratio-Preserving Padding** (`pad_and_resize_deepprint()`):
   * Rectangular fingerprint sensor captures (e.g. $300 \times 400$) are centered on a square canvas using symmetric constant border padding (`cv2.copyMakeBorder` with `value=255` white background).
   * Prevents spatial distortion of ridge frequency and ridge orientation that occurs during naive non-uniform resizing.
   * Uniformly resized to target dimensions: `(224, 224)`.
2. **Channel Format & Transformation**:
   * Converted to 8-bit grayscale PIL image (`mode="L"`).
   * Applied `transforms.ToTensor()`, scaling range $[0, 255] \rightarrow [0.0, 1.0]$.
   * Channel Triplication: Grayscale 1-channel tensor replicated 3 times → `(1, 3, 224, 224)` to satisfy ResNet-50's 3-channel stem conv input requirements.
   * *Note*: No ImageNet mean/std subtraction is performed, matching the scratch-trained model's training distribution.

### 3.4 Post-Processing & Quality Assessment
* **Heuristic Quality Scoring**:
  * Weighted score based on two metrics:
    $$\text{Quality} = 0.7 \times \text{Coverage} + 0.3 \times \min\left(1.0, \frac{\text{LaplacianVar}}{1500}\right)$$
  * Foreground pixels defined as intensity $< 210$.
  * Not NFIQ 2.0 compliant — serves as a fast pre-filter only.
* **Embedding Post-Processing**:
  * Internal L2-normalization inside `FingerprintResNetModel.forward()`: $\frac{\mathbf{f}}{\max(\|\mathbf{f}\|_2, 10^{-12})}$.
  * Secondary defensive L2-normalization in [`extractors/finger_extractor.py`](file:///d:/biometrics_intern/biometrics_fusion/extractors/finger_extractor.py). Output is a float32 vector of length 512.

## 4. Unimodal Modality 3: Iris Extractor

### 4.1 Dual-Backend Architecture

The iris extractor provides **two complementary representations** from a single segmentation-and-normalization pass:

| Backend | Algorithm | Output |
|:---|:---|:---|
| `resnet100` | ArcIris IResNet100 deep embedding | 512-d Float32 L2-unit vector |
| `gabor` | Multi-scale Daugman Gabor phase quantization | 512-d bipolar Float32 (subsampled, L2-norm) |

Both backends share the same Hough-circle segmentation and Daugman rubber-sheet normalization via the self-contained `OpenIrisModel` classical pipeline in [`models/iris.py`](file:///d:/biometrics_intern/biometrics_fusion/models/iris.py). `_SelfContainedIrisManager` caches `last_normalized_image` after each segmentation call for reuse by the ResNet100 branch — avoiding a second segmentation run.

### 4.2 Overview & Workflow

```
Raw Eye Image ──► Hough-Circle Segmentation ──► Daugman Rubber-Sheet (64×512 polar)
                                                          │
             ┌────────────────────────────────────────────┴────────────────────────────────────────────────┐
             ▼                                                                                              ▼
CLAHE (clip=2.0, tile=8×8)                                                               estimate_noise_mask()
             │                                                                         (reflections≥230, shadows≤25)
             ▼                                                                                              │
PIL resize (512×64) + Normalize([-1,1])                                                encode_iris() 3 wavelengths {8,16,24}
             │                                                                         2-bit phase quantization
             ▼                                                                                              │
IResNet100 forward() + F.normalize(p=2)                                                subsample→bipolar→L2-norm
             │                                                                                              │
             ▼                                                                                              ▼
      512-d Unit Vector                                                                     512-d Bipolar Vector
```

### 4.3 Self-Contained Classical Iris Pipeline ([`iris_pipeline.py`](file:///d:/biometrics_intern/biometrics_fusion/pipelines/iris_pipeline.py))

Requires no external iris library — pure OpenCV + NumPy:

**Stage 1 — Segmentation (`segment_iris()`)**:
* Median blur `k=5` for noise suppression.
* Adaptive radius ranges from image size: `pupil_r ∈ [max(8, short//16), max(30, short//6)]`.
* Progressive `param2` fallback for pupil `{30, 20, 15}` and iris `{25, 18, 12}`.
* Pupil: lowest mean interior intensity (darkest region = pupil).
* Iris: closest center to detected pupil center (concentricity constraint).

**Stage 2 — Daugman Normalization (`normalize_iris()`)**:
* Fully vectorized: `np.meshgrid` + `cv2.remap` — replaces a 32,768-iteration Python double-loop.
* `r_fracs (64,1)` broadcast against `thetas (1,512)` → `(64,512)` sample coordinate maps.
* `cv2.INTER_LINEAR` bilinear interpolation with `BORDER_CONSTANT=0`.

**Stage 3 — Noise Masking (`estimate_noise_mask()`)**:
* Reflections: `polar_image >= 230`; Shadows/eyelashes: `polar_image <= 25`.
* Output: boolean mask (`True = unreliable`).

**Stage 4 — Gabor Encoding (`encode_iris()`)**:
* Wavelengths $\lambda \in \{8, 16, 24\}$; orientation $\theta = 0.0$; sigma $\sigma = 0.5\lambda$.
* Real kernel `psi=0`; Imaginary kernel `psi=π/2` (quadrature pair).
* 2-bit phase quantization per pixel. Output per scale: `(64, 512, 2)`.

**Stage 5 — Masked Hamming Distance (`masked_hamming_distance()`)**:
* Rotation compensation: `shift ∈ [-8, +8]` (17 total).
* Counts disagreeing bits among `(~mask_a) & (~mask_b)` valid pairs.
* Returns minimum fractional HD across all shifts: $\text{HD} \in [0, 1]$.

### 4.4 ArcIris IResNet100 Architecture

**`IBasicBlock`** (differs from standard ResNet BasicBlock):
* Pre-activation BatchNorm before first conv (`bn1`).
* PReLU (learnable negative slope per channel) instead of ReLU.
* Extra `bn3` after second conv, before residual addition.
* Same block design as ArcFace's official IResNet backbone.

**`IResNet` stem**: Single `3×3 stride-1 conv` — avoids `7×7 stride-2 + maxpool`. Preserves early spatial resolution for fine iris texture.

**Stage depths** (`iresnet100`): `[3, 13, 30, 3]` = 100 layers. Channels: `[64 → 128 → 256 → 512]`.

**FC layer sizing** — mathematically derived:
* Input: `64 (H) × 512 (W)`. After 4× stride-2: `64/16=4`, `512/16=32`.
* Final feature map: `512 ch × 4 × 32 = 65,536`. FC: `Linear(65536, 512)` + `BatchNorm1d(512)`.

### 4.5 Model Weights

| Weight File | Size | Purpose |
|:---|:---|:---|
| [`weights/iris/ResNet100_154000.pt`](file:///d:/biometrics_intern/biometrics_fusion/weights/iris/ResNet100_154000.pt) | **328.15 MB** | ArcIris IResNet100 — production model |
| [`weights/iris/iris_semseg_upp_scse_mobilenetv2.onnx`](file:///d:/biometrics_intern/biometrics_fusion/weights/iris/iris_semseg_upp_scse_mobilenetv2.onnx) | **53.51 MB** | UNet++ SCSE MobileNetV2 deep segmentation — **present but unused** |
| [`weights/iris/iris_extractor_best.pth`](file:///d:/biometrics_intern/biometrics_fusion/weights/iris/iris_extractor_best.pth) | **61.18 MB** | Alternative iris extractor — unused |
| [`weights/iris/resnet18-1543-0.047488-maskIoU-0.934494.pth`](file:///d:/biometrics_intern/biometrics_fusion/weights/iris/resnet18-1543-0.047488-maskIoU-0.934494.pth) | **42.74 MB** | ResNet-18 segmentation (IoU=0.934) — unused |

Weight loading: `torch.load(..., weights_only=True)` + DataParallel `module.` prefix stripping + `load_state_dict(strict=True)`.

### 4.6 Preprocessing (ResNet100 Path)

1. CLAHE: `clipLimit=2.0`, `tileGridSize=(8,8)` on 64×512 polar image.
2. PIL resize: `(512, 64)` (width × height) via `BILINEAR`.
3. `transforms.ToTensor()`: uint8 → float32 `[0,1]`.
4. `Normalize(mean=(0.5,), std=(0.5,))`: maps → `[-1,1]`.
5. `.repeat(1,3,1,1)` → `(1, 3, 64, 512)`.

### 4.7 Post-Processing & Output Formats

* **ResNet100 path**: Internal `F.normalize(p=2)` + external defensive L2-norm. Output: float32 unit vector, dim 512.
* **Gabor path**: 3 scales → concatenate → `flat_code` (196,608 bits) → stride `384` subsample to 512 → bipolar → L2-norm. **Discards 99.74% of available phase information.**
* **Classical Hamming path** (`compute_distance()`): Full 3-scale bit matrix matched via `masked_hamming_distance()` — the correct way to use Gabor codes.
* **Temp-File Bottleneck**: NumPy array inputs trigger unnecessary disk write via `tempfile.NamedTemporaryFile` + `cv2.imwrite()` (flagged as known issue in code).

---

## 5. Summary Matrix of Extractor Specifications

| Feature / Metric | Face Extractor | Fingerprint Extractor | Iris Extractor (`resnet100`) | Iris Extractor (`gabor`) |
| :--- | :--- | :--- | :--- | :--- |
| **Model Type** | ArcFace DCNN (ResNet-50) | DeepPrint Multi-Scale (ResNet-50) | ArcIris DCNN (IResNet-100) | Classical Gabor Wavelet |
| **Primary Weights** | `w600k_r50.onnx` (166.31 MB) | `finger_extractor_best.pth` (102.06 MB) | `ResNet100_154000.pt` (328.15 MB) | Algorithmic (No weights) |
| **Detector Weights** | `det_10g.onnx` (SCRFD, 16.14 MB) | N/A (Rule-based padding) | N/A (Hough Circle Transform) | N/A (Hough Circle Transform) |
| **Input Resolution** | $112 \times 112$ (Aligned) | $224 \times 224$ (Square-padded) | $64 \times 512$ (Polar unrolled) | $64 \times 512$ (Polar unrolled) |
| **Input Channels** | 3 (RGB) | 3 (Gray replicated 3×) | 3 (Polar Gray + CLAHE 3×) | 1 (Polar Gray) |
| **Normalization** | $(x - 127.5)/127.5 \in [-1, 1]$ | $x / 255.0 \in [0, 1]$ | $(x/255.0 - 0.5)/0.5 \in [-1, 1]$ | Bipolar $2x - 1 \in \{-1, 1\}$ |
| **Output Vector** | 512-d Float32 | 512-d Float32 | 512-d Float32 | 512-d Float32 (Subsampled) |
| **Norm Constraint** | Unit L2 Norm | Unit L2 Norm | Unit L2 Norm | Unit L2 Norm |
| **Quality Metric** | Laplacian Var ($\ge 60.0$) | $0.7 \cdot \text{Cov} + 0.3 \cdot \text{Sharp}$ | None explicit | Intensity Noise Mask |
| **GPU Inference** | ONNX CUDAExecProvider | PyTorch `cuda` | PyTorch `cuda` | CPU-only (NumPy) |
| **Backend Framework** | ONNX Runtime | PyTorch | PyTorch | OpenCV + NumPy |

---

## 6. Downstream Modules — Fusion & Template Protection

### 6.1 Concat-L2 Fusion ([`concat_fusion_db.py`](file:///d:/biometrics_intern/biometrics_fusion/concat_fusion_db.py))
Concatenates face (512-d) + finger (512-d) + iris (512-d) → 1536-d vector, then L2-normalizes. Stored as `fusion_type='concat_l2'` in `fused_templates`.

### 6.2 Generalized Compact Bilinear (CBP) Fusion ([`cbp_fusion_db.py`](file:///d:/biometrics_intern/biometrics_fusion/cbp_fusion_db.py))
- Stage 1: Count Sketch(face) ⊗ Count Sketch(iris) via FFT → intermediate fused.
- Stage 2: Count Sketch(intermediate) ⊗ Count Sketch(finger) → final fused.
- Post-processing: Signed Square Root `z' = sign(z)×√|z|` then L2-normalization.
- Learnable `weight1`, `weight2` scalar parameters per fusion pair. Stored as `fusion_type='cbp'`.

### 6.3 BioHashing ([`biohashing.py`](file:///d:/biometrics_intern/biometrics_fusion/biohashing.py)) — Template Protection
Implements Jin et al. (2004) Traditional BioHashing:
- Orthonormal random projection matrix $\mathbf{R} \in \mathbb{R}^{D \times M}$ via QR decomposition of Gaussian random matrix.
- Per-trait deterministic seed: `(master_seed + hash(trait_name) % 100000) & 0x7FFFFFFF`.
- Steps: project → threshold → normalize: $\hat{b} = \text{sign}(\mathbf{x} \cdot \mathbf{R}) / \sqrt{M}$.

### 6.4 Two-Stage Authentication ([`authenticate.py`](file:///d:/biometrics_intern/biometrics_fusion/authenticate.py))
1. **Stage 1 — Trait-Level BioHash Matching**: Extract, BioHash, compare each probe trait against all enrolled `biohash_embeddings` via cosine similarity (1:N or 1:1).
2. **Stage 2 — Fused Embedding Verification**: All three BioHashed traits are CBP-fused and compared against `biohash_embedding_fused`. Default threshold: `fused=0.0337`.

---

## 7. Recommendations for Architecture & Performance Improvement

Based on deep source code inspection, the following targeted recommendations are proposed:

### 1. Upgrade Iris Segmentation to Deep Semantic Segmentation
* **Current Limitation**: Hough circle segmentation assumes strictly circular pupil/iris boundaries; fails on occluded or off-axis eyes.
* **Proposed Enhancement**: Replace `segment_iris()` with the pretrained MobileNetV2 UNet++ SCSE ONNX model at [`weights/iris/iris_semseg_upp_scse_mobilenetv2.onnx`](file:///d:/biometrics_intern/biometrics_fusion/weights/iris/iris_semseg_upp_scse_mobilenetv2.onnx) (53.51 MB). Everything downstream remains unchanged.

### 2. Eliminate Disk I/O Bottlenecks in In-Memory Iris Pipelines
* **Current Limitation**: In [`extractors/iris_extractor.py`](file:///d:/biometrics_intern/biometrics_fusion/extractors/iris_extractor.py), NumPy array inputs trigger write to a temporary PNG file via `tempfile.NamedTemporaryFile` + `cv2.imwrite()`, then read back (flagged as `IMPORTANT ISSUE` in code).
* **Proposed Enhancement**: Refactor `_SelfContainedIrisManager.generate_biometric_template()` to accept NumPy arrays directly — all downstream functions already operate on numpy arrays.

### 3. Replace Strided Subsampling of Classical Iriscodes in Vector Fusion
* **Current Limitation**: The `gabor` backend subsamples 196,608-bit phase code to 512 via `flat_code[::384][:512]`, discarding **99.74%** of information.
* **Proposed Enhancement**: Use `resnet100` exclusively for vector-level fusion. Reserve full `gabor` bit matrix for Masked Fractional Hamming Distance matching. A hybrid score-fusion (ResNet100 cosine + HD) is state-of-the-art for iris.

### 4. Enable Dual-Branch Fingerprint Texture & Minutiae Extraction
* **Current Limitation**: Production code uses single-branch `FingerprintResNetModel` (grayscale ×3), ignoring the defined-but-unused `DeepPrintTexMinuModel` in [`models/finger.py`](file:///d:/biometrics_intern/biometrics_fusion/models/finger.py).
* **Proposed Enhancement**: Pre-compute `[ch0=GaborTexture, ch1=MinutiaeSkeletonMap, ch2=GaussianMinutiaeDensity]` and feed into `DeepPrintTexMinuModel` per the original DeepPrint paper.

### 5. Remove Redundant Double L2-Normalization Passes
* **Current Limitation**: Both `FingerprintExtractor` and `IrisExtractor` apply L2-normalization twice (once inside `forward()` and once in `extract_features()`).
* **Proposed Enhancement**: Remove the defensive external pass once the model's internal normalization is confirmed stable, or document the intent clearly.

### 6. Standardize Exception Handling Across Extractors
* **Current Limitation**: Inconsistent errors (`ValueError` in face, `RuntimeError` in iris, raw torch exceptions in finger) halt batch enrollment loops.
* **Proposed Enhancement**: Implement a unified `ExtractionResult` return pattern: `(success: bool, embedding: np.ndarray | None, quality_score: float, error_message: str | None)`.

### 7. BioHashing Seed Non-Determinism Risk
* **Current Limitation**: `BioHasher.__init__` uses `hash(trait_name)` for seed derivation, which is non-deterministic across Python processes (PYTHONHASHSEED). A seed change makes enrolled and probe BioHashes incompatible.
* **Proposed Enhancement**: Replace with `hashlib.sha256(trait_name.encode()).digest()` for deterministic hashing. Always consult `biohash_keys.json` for existing enrollments.

### 8. SCRFD Landmark Rescaling — Non-Square Input Verification
* **Potential Issue**: `detect_landmarks()` rescaling `best_kps * [w/640.0, h/640.0]` is correct for square inputs but may introduce asymmetric distortion for non-square inputs (e.g., `1280×720`).
* **Proposed Enhancement**: Add aspect-preserving padding before SCRFD inference (analogous to fingerprint padding) and verify on non-square test images.

---
*Report generated as part of a full technical audit of the biometric multimodal fusion framework (`biometrics_fusion`).*
*Analysis based on direct source inspection of: `extractors/`, `models/`, `pipelines/`, `cbp_fusion_db.py`, `concat_fusion_db.py`, `biohashing.py`, `authenticate.py`, `main.py`.*
