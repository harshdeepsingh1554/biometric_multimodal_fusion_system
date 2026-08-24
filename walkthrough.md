# Walkthrough: CircleNet Iris Segmentation, DeepPrint TexMinu & SourceAFIS Integration

We have integrated all three requested biometric components into the project:
1. **CircleNet (`ResNet-18`) Iris Segmentation & Masking**: Direct pupil/iris circle regression and annular masking, active and selectable in the iris pipeline.
2. **DeepPrint (`DeepPrint_TexMinu`) Dual-Branch Model**: 512-D fingerprint representation (256-D Texture + 256-D Minutiae) with checkpoint loading and channel synthesis.
3. **SourceAFIS 3.18.0 Fingerprint Matcher**: Minutiae-based matching engine supporting `java_libs/sourceafis-3.18.0.jar` via JPype with automatic Python Crossing-Number minutiae fallback.

---

## 1. Iris CircleNet (`ResNet-18`) Segmentation

### Implementation Details:
- **File**: [`iris/segmentation/circlenet_segmenter.py`](file:///d:/biometrics_intern/biometrics_fusion/iris/segmentation/circlenet_segmenter.py)
- **Architecture**: ResNet-18 backbone with a 1×1 `ConvHead` (`512 -> 6`) and `FCLayer` (6 linear heads with GELU activation).
- **Weights**: Loaded from [`weights/iris/resnet18-1543-0.047488-maskIoU-0.934494.pth`](file:///d:/biometrics_intern/biometrics_fusion/weights/iris/resnet18-1543-0.047488-maskIoU-0.934494.pth).
- **Prediction**: Regresses 6 normalized parameters scaled to the native image resolution:
  $$\text{pupil} = (p_x, p_y, p_r), \quad \text{iris} = (i_x, i_y, i_r)$$
- **Masks & Quality**: Synthesizes binary pupil mask, annular iris mask, annular visible ratio, and geometric concentricity checks.
- **Pipeline Integration**: [`iris/pipeline/iris_pipeline.py`](file:///d:/biometrics_intern/biometrics_fusion/iris/pipeline/iris_pipeline.py) and [`iris/extractor/iris_extractor.py`](file:///d:/biometrics_intern/biometrics_fusion/iris/extractor/iris_extractor.py) now support `seg_backend="circlenet"` (default), `"unet"`, and `"hough"`.

### Verification Result:
```text
=== 1. Testing CircleNet Iris Segmentation ===
Testing on iris image: data/all_images/setB/Person_001/iris/iris_01.jpg
Success: True, Mode: circlenet, Quality: 0.894
Pupil: (253.6, 240.4), r=29.5 | Iris: (258.3, 237.1), r=92.6
Embedding shape: (512,), L2-norm: 1.0000
```

---

## 2. Fingerprint DeepPrint (`DeepPrint_TexMinu`)

### Implementation Details:
- **Model**: [`models/finger.py`](file:///d:/biometrics_intern/biometrics_fusion/models/finger.py) (`DeepPrintTexMinuModel`)
- **Architecture**:
  - **Texture Branch**: ResNet-18 + AdaptiveAvgPool + Linear(512, 256) + BatchNorm1d -> 256-D vector.
  - **Minutiae Branch**: ResNet-18 + AdaptiveAvgPool + Linear(512, 256) + BatchNorm1d -> 256-D vector.
  - **Fusion & Normalization**: Concatenated to 512-D and L2-normalized: $\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$.
- **Extractor**: [`extractors/finger_extractor.py`](file:///d:/biometrics_intern/biometrics_fusion/extractors/finger_extractor.py) supports `model_type="tex_minu"` and `model_type="resnet50"`, searching for `models/DeepPrint_Tex_512/best_model.pyt` and `weights/finger/finger_extractor_best.pth`.

### Verification Result:
```text
=== 2. Testing DeepPrint TexMinu Fingerprint Extractor ===
DeepPrint TexMinu embedding shape: (512,), L2-norm: 1.0000
```

---

## 3. SourceAFIS 3.18.0 Fingerprint Matcher

### Implementation Details:
- **File**: [`extractors/sourceafis_matcher.py`](file:///d:/biometrics_intern/biometrics_fusion/extractors/sourceafis_matcher.py)
- **Engine**:
  - Attempts JPype initialization to bridge `java_libs/sourceafis-3.18.0.jar` (`FingerprintTemplate`, `FingerprintMatcher`).
  - Implements a pure Python minutiae detection engine using Otsu binarization, morphological thinning, Crossing Number ($CN$) classification for ridge endings/bifurcations, and spatial alignment score computation.

### Verification Result:
```text
=== 3. Testing SourceAFIS Fingerprint Matcher ===
Intra-person match result: {'score': 41.317, 'matched': True, 'engine': 'python_minutiae_cn', 'probe_minutiae_count': 2482, 'gallery_minutiae_count': 2514}
```

---

## 4. CLI Usage

You can enroll datasets and select the new backends via `main.py`:

```bash
# Enroll using CircleNet iris segmentation (default) and standard DeepPrint
python main.py --data-dir data/all_images/setB --db enrolled_templates.json --iris-seg circlenet

# Enroll using DeepPrint TexMinu fingerprint model
python main.py --data-dir data/all_images/setB --db enrolled_templates.json --finger-model-type tex_minu

# Enroll with GPU acceleration and SQLite persistence
python main.py --data-dir data/all_images/setB --db enrolled_templates.json --sqlite database/biometric.db --dataset setB --iris-seg circlenet --gpu
```
