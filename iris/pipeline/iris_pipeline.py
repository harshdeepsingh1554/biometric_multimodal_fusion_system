"""
iris_pipeline.py — Master Orchestrator for the Production Deep Iris Architecture
==================================================================================
Controls the exact sequential execution order of all modular iris components:

  RAW NIR IRIS IMAGE (Path or NumPy array)
          ↓
  1. Preprocessing & Input Validation (IrisPreprocessor)
          ↓
  2. Deep Semantic Segmentation (IrisSegmenter - UNet++ + scSE + MobileNetV2 ONNX)
          ↓
  3. Mask Post-Processing & Boundary Estimation (IrisGeometryEstimator - Direct Ellipse Fitting)
          ↓
  4. Iris Quality Gating (IrisQualityGate - Occlusion, Geometry, Sharpness, Confidence)
          ↓
  5. Daugman Rubber-Sheet Normalization (DaugmanNormalizer - 64 × 512 via cv2.remap)
          ↓
  6. Polar Contrast Enhancement (CLAHE - clipLimit=2.0, tileGridSize=(8, 8))
          ↓
  7. Deep Feature Representation (IrisResNetModel - IResNet-100)
          ↓
  8. L2 Normalization -> 512-D Biometric Template (IrisExtractionResult)
"""

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
import cv2
import torch

from iris.preprocessing import IrisPreprocessor, apply_clahe
from iris.segmentation import (
    IrisSegmenter,
    IrisGeometryEstimator,
    IrisCircleNetSegmenter,
    postprocess_and_estimate_geometry,
)
from iris.quality import IrisQualityFailure, IrisQualityGate
from iris.normalization import DaugmanNormalizer
from iris.models import IrisResNetModel

logger = logging.getLogger(__name__)


@dataclass
class IrisExtractionResult:
    """
    Structured result returned by the Iris Pipeline.
    """
    embedding: Optional[np.ndarray] = None
    success: bool = False
    quality_score: float = 0.0
    failure_reason: IrisQualityFailure = IrisQualityFailure.SUCCESS
    segmentation_confidence: float = 0.0
    geometry_metadata: Dict[str, Any] = field(default_factory=dict)
    normalization_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_mode: str = "circlenet"
    timings_ms: Dict[str, float] = field(default_factory=dict)
    normalized_image: Optional[np.ndarray] = None
    noise_mask: Optional[np.ndarray] = None


class IrisPipeline:
    """
    Master Iris Pipeline Orchestrator.
    Coordinates sequential execution across preprocessing, segmentation (CircleNet / UNet++),
    geometry, quality gating, normalization, and deep feature extraction.
    """

    def __init__(
        self,
        seg_model_path: Optional[str] = None,
        circlenet_model_path: Optional[str] = None,
        embed_model_path: Optional[str] = None,
        use_gpu: bool = False,
        apply_clahe: bool = True,
        radial_res: int = 64,
        angular_res: int = 512,
        quality_gate: Optional[IrisQualityGate] = None,
        seg_backend: str = "circlenet",
    ):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.apply_clahe = apply_clahe
        self.radial_res = radial_res
        self.angular_res = angular_res
        self.seg_backend = seg_backend.lower()

        # 1. Preprocessor
        self.preprocessor = IrisPreprocessor()

        # 2. Segmenters
        self.circlenet_segmenter = None
        self.segmentor = None
        self.geometry_estimator = None

        if self.seg_backend == "circlenet":
            try:
                self.circlenet_segmenter = IrisCircleNetSegmenter(
                    model_path=circlenet_model_path,
                    use_gpu=self.use_gpu,
                )
            except Exception as e:
                logger.warning(f"Failed to load CircleNet segmenter ({e}), falling back to UNet++")
                self.seg_backend = "unet"

        if self.seg_backend == "unet" or self.circlenet_segmenter is None:
            self.segmentor = IrisSegmenter(model_path=seg_model_path, use_gpu=self.use_gpu)
            self.geometry_estimator = IrisGeometryEstimator()

        # 3. Quality Gate
        self.quality_gate = quality_gate if quality_gate is not None else IrisQualityGate()

        # 4. Normalizer
        self.normalizer = DaugmanNormalizer(radial_res=self.radial_res, angular_res=self.angular_res)

        # 5. Deep IResNet-100 Embedding Model
        if embed_model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            embed_model_path = os.path.join(base_dir, "weights", "iris", "ResNet100_154000.pt")

        self.model = IrisResNetModel(embedding_size=512)
        if not self.model.load_weights(embed_model_path, self.device):
            raise RuntimeError(f"Failed to load IResNet-100 weights from: {embed_model_path}")
        self.model.to(self.device)
        self.model.eval()

        # Properties for backward compatibility and test inspection
        self.resnet_model = self.model
        self.transform = self.preprocessor.embed_transform
        self.clahe = self.preprocessor.clahe

        logger.info(
            f"IrisPipeline initialized successfully on {self.device} (backend: {self.seg_backend}, GPU: {self.use_gpu})."
        )

    def extract(
        self,
        image_input: Union[str, np.ndarray],
        eye_side: str = "right",
        mode: str = "deep_segmentation",
        l2_normalize: bool = True,
    ) -> IrisExtractionResult:
        """
        Executes the full pipeline sequentially:
          Raw Image -> Preprocessing -> Segmentation -> Geometry -> Quality Gate -> Normalization -> Polar CLAHE -> IResNet-100 -> 512-D L2 Embedding
        """
        timings = {}
        t_total_start = time.perf_counter()

        # -------------------------------------------------------------
        # Stage 1: Input Validation & Preprocessing
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        image_gray, err_msg = self.preprocessor.validate_and_load(image_input)
        if image_gray is None:
            return IrisExtractionResult(
                embedding=None,
                success=False,
                failure_reason=IrisQualityFailure.INVALID_INPUT,
                processing_mode=mode,
                timings_ms={"total": (time.perf_counter() - t_total_start) * 1000.0}
            )

        if mode == "hough_fallback" or self.seg_backend == "hough":
            return self._extract_hough_fallback(image_gray, l2_normalize=l2_normalize)

        current_mode = "circlenet" if (self.seg_backend == "circlenet" and self.circlenet_segmenter is not None) else "unet"
        if mode in ("circlenet", "unet"):
            current_mode = mode

        # -------------------------------------------------------------
        # Stage 2 & 3: Segmentation & Geometry Estimation
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        if current_mode == "circlenet" and self.circlenet_segmenter is not None:
            try:
                geometry_meta = self.circlenet_segmenter.segment(image_gray)
                timings["segmentation"] = (time.perf_counter() - t0) * 1000.0
                timings["geometry"] = 0.0
                seg_confidence = geometry_meta.get("confidence", 0.90)
            except Exception as e:
                logger.error(f"CircleNet segmentation failed: {e}")
                return IrisExtractionResult(
                    embedding=None,
                    success=False,
                    failure_reason=IrisQualityFailure.SEGMENTATION_FAILED,
                    processing_mode=current_mode,
                    timings_ms={"segmentation": (time.perf_counter() - t0) * 1000.0, "total": (time.perf_counter() - t_total_start) * 1000.0}
                )
        else:
            seg_tensor, orig_hw = self.preprocessor.prepare_segmentation_tensor(image_gray)
            timings["preprocessing"] = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            try:
                seg_dict = self.segmentor.segment(seg_tensor, orig_hw)
                timings["segmentation"] = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                logger.error(f"Segmentation failed: {e}")
                return IrisExtractionResult(
                    embedding=None,
                    success=False,
                    failure_reason=IrisQualityFailure.SEGMENTATION_FAILED,
                    processing_mode=current_mode,
                    timings_ms={"segmentation": (time.perf_counter() - t0) * 1000.0, "total": (time.perf_counter() - t_total_start) * 1000.0}
                )

            seg_confidence = seg_dict["confidence"]

            t0 = time.perf_counter()
            geometry_meta = self.geometry_estimator.estimate(seg_dict)
            timings["geometry"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # Stage 4: Multi-Criteria Iris Quality Gating
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        passed_qg, qg_failure, quality_score, quality_meta = self.quality_gate.evaluate_quality(
            image_gray, seg_confidence, geometry_meta
        )
        timings["quality_gate"] = (time.perf_counter() - t0) * 1000.0

        if not passed_qg:
            return IrisExtractionResult(
                embedding=None,
                success=False,
                quality_score=quality_score,
                failure_reason=qg_failure,
                segmentation_confidence=seg_confidence,
                geometry_metadata=geometry_meta,
                processing_mode=mode,
                timings_ms=timings,
            )

        # -------------------------------------------------------------
        # Stage 5: Daugman Rubber-Sheet Normalization
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            polar_img, polar_noise = self.normalizer.normalize(
                image_gray, geometry_meta, noise_mask=geometry_meta.get("noise_mask")
            )
            timings["normalization"] = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return IrisExtractionResult(
                embedding=None,
                success=False,
                quality_score=quality_score,
                failure_reason=IrisQualityFailure.NORMALIZATION_FAILED,
                segmentation_confidence=seg_confidence,
                geometry_metadata=geometry_meta,
                processing_mode=mode,
                timings_ms=timings,
            )

        # -------------------------------------------------------------
        # Stage 6: CLAHE Contrast Enhancement
        # -------------------------------------------------------------
        norm_for_embed = polar_img
        if self.apply_clahe:
            norm_for_embed = self.preprocessor.apply_clahe(norm_for_embed)

        # -------------------------------------------------------------
        # Stage 7: IResNet-100 Feature Extraction & L2 Normalization
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            embed_tensor = self.preprocessor.prepare_embedding_tensor(norm_for_embed, self.device)
            with torch.no_grad():
                emb_out = self.model(embed_tensor)
                embedding = emb_out.squeeze(0).cpu().numpy().astype(np.float32)

            if l2_normalize:
                norm_val = np.linalg.norm(embedding)
                embedding = (embedding / max(norm_val, 1e-12)).astype(np.float32)

            timings["embedding"] = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            logger.error(f"Embedding forward pass failed: {e}")
            return IrisExtractionResult(
                embedding=None,
                success=False,
                quality_score=quality_score,
                failure_reason=IrisQualityFailure.EMBEDDING_FAILED,
                segmentation_confidence=seg_confidence,
                geometry_metadata=geometry_meta,
                processing_mode=mode,
                timings_ms=timings,
            )

        timings["total"] = (time.perf_counter() - t_total_start) * 1000.0

        return IrisExtractionResult(
            embedding=embedding,
            success=True,
            quality_score=quality_score,
            failure_reason=IrisQualityFailure.SUCCESS,
            segmentation_confidence=seg_confidence,
            geometry_metadata=geometry_meta,
            normalization_metadata={
                "radial_res": self.radial_res,
                "angular_res": self.angular_res,
                "applied_clahe": self.apply_clahe,
            },
            processing_mode=mode,
            timings_ms=timings,
            normalized_image=polar_img,
            noise_mask=polar_noise,
        )

    def extract_aligned(
        self,
        image_input: Union[str, np.ndarray],
        gallery_embedding: Optional[np.ndarray],
        eye_side: str = "right",
        shifts: Tuple[int, ...] = (-24, -16, -8, 0, 8, 16, 24),
        l2_normalize: bool = True,
    ) -> IrisExtractionResult:
        """Extracts embedding with horizontal roll alignment against gallery_embedding."""
        base_res = self.extract(image_input, eye_side=eye_side, mode="deep_segmentation", l2_normalize=l2_normalize)
        if not base_res.success or gallery_embedding is None or base_res.normalized_image is None:
            return base_res

        t0 = time.perf_counter()
        norm_img = base_res.normalized_image
        if self.apply_clahe:
            norm_img = self.preprocessor.apply_clahe(norm_img)

        gal_norm = (gallery_embedding / max(np.linalg.norm(gallery_embedding), 1e-12)).astype(np.float32)

        tensors = []
        for s in shifts:
            rolled = np.roll(norm_img, s, axis=1)
            t = self.preprocessor.prepare_embedding_tensor(rolled, self.device)
            tensors.append(t.squeeze(0))

        batch_tensor = torch.stack(tensors, dim=0).to(self.device)

        with torch.no_grad():
            emb_tensors = self.model(batch_tensor)
            if l2_normalize:
                emb_tensors = torch.nn.functional.normalize(emb_tensors, dim=1)
            embs = emb_tensors.cpu().numpy().astype(np.float32)

        sims = np.dot(embs, gal_norm)
        best_idx = int(np.argmax(sims))

        base_res.embedding = embs[best_idx]
        base_res.timings_ms["alignment"] = (time.perf_counter() - t0) * 1000.0
        return base_res

    def _extract_hough_fallback(self, image_gray: np.ndarray, l2_normalize: bool = True) -> IrisExtractionResult:
        """Legacy Hough transform fallback."""
        t_start = time.perf_counter()
        pupil_circle, iris_circle = segment_iris(image_gray)
        if pupil_circle is None or iris_circle is None:
            return IrisExtractionResult(
                embedding=None,
                success=False,
                failure_reason=IrisQualityFailure.SEGMENTATION_FAILED,
                processing_mode="hough_fallback",
                timings_ms={"total": (time.perf_counter() - t_start) * 1000.0}
            )

        polar_image = normalize_iris(
            image_gray, pupil_circle, iris_circle,
            radial_res=self.radial_res, angular_res=self.angular_res
        )

        norm_for_embed = polar_image
        if self.apply_clahe:
            norm_for_embed = self.preprocessor.apply_clahe(norm_for_embed)

        embed_tensor = self.preprocessor.prepare_embedding_tensor(norm_for_embed, self.device)
        with torch.no_grad():
            emb_out = self.model(embed_tensor)
            embedding = emb_out.squeeze(0).cpu().numpy().astype(np.float32)

        if l2_normalize:
            norm_val = np.linalg.norm(embedding)
            embedding = (embedding / max(norm_val, 1e-12)).astype(np.float32)

        return IrisExtractionResult(
            embedding=embedding,
            success=True,
            quality_score=0.70,
            failure_reason=IrisQualityFailure.SUCCESS,
            segmentation_confidence=0.70,
            geometry_metadata={"pupil_circle": pupil_circle, "iris_circle": iris_circle},
            processing_mode="hough_fallback",
            timings_ms={"total": (time.perf_counter() - t_start) * 1000.0},
            normalized_image=polar_image,
        )


# =========================================================================
# Classical Fallback & Diagnostic Functions
# =========================================================================

def segment_iris(image_gray, pupil_radius_range=None, iris_radius_range=None):
    """Classical Hough Circle segmentation."""
    h, w = image_gray.shape[:2]
    short = min(h, w)
    if pupil_radius_range is None:
        pupil_radius_range = (max(8, short // 16), max(30, short // 6))
    if iris_radius_range is None:
        iris_radius_range = (max(30, short // 8), max(120, short // 2))

    min_dist = max(10, short // 5)
    blurred = cv2.medianBlur(image_gray, 5)

    pupil_circles = None
    for param2 in (30, 20, 15):
        pupil_circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
            param1=50, param2=param2,
            minRadius=pupil_radius_range[0], maxRadius=pupil_radius_range[1],
        )
        if pupil_circles is not None:
            break

    if pupil_circles is None:
        return None, None

    px, py, pr = _pick_darkest_circle(image_gray, pupil_circles[0])
    iris_min_r = max(iris_radius_range[0], int(pr * 1.3))
    iris_max_r = iris_radius_range[1]

    iris_circles = None
    for param2 in (25, 18, 12):
        iris_circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
            param1=50, param2=param2,
            minRadius=iris_min_r, maxRadius=iris_max_r,
        )
        if iris_circles is not None:
            break

    if iris_circles is None:
        return (px, py, pr), None

    ix, iy, ir = _pick_closest_circle(iris_circles[0], center=(px, py))
    center_offset = float(np.hypot(ix - px, iy - py))
    max_allowed_offset = 0.35 * ir
    if center_offset > max_allowed_offset:
        return (px, py, pr), None

    return (px, py, pr), (ix, iy, ir)


def _pick_darkest_circle(image_gray, circles):
    best, best_mean = None, 256.0
    for (x, y, r) in circles:
        mask = np.zeros(image_gray.shape, dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        mean_val = cv2.mean(image_gray, mask=mask)[0]
        if mean_val < best_mean:
            best_mean, best = mean_val, (x, y, r)
    return best


def _pick_closest_circle(circles, center):
    cx, cy = center
    best, best_dist = None, float("inf")
    for (x, y, r) in circles:
        d = (x - cx) ** 2 + (y - cy) ** 2
        if d < best_dist:
            best_dist, best = d, (x, y, r)
    return best


def normalize_iris(image_gray, pupil_circle, iris_circle, radial_res=64, angular_res=512):
    """Classical vectorized Daugman normalization for circular boundaries."""
    px, py, pr = pupil_circle
    ix, iy, ir = iris_circle
    r_fracs = np.linspace(0, 1, radial_res, dtype=np.float32).reshape(-1, 1)
    thetas = np.linspace(0, 2 * np.pi, angular_res, endpoint=False, dtype=np.float32).reshape(1, -1)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    x_p = px + pr * cos_t
    y_p = py + pr * sin_t
    x_i = ix + ir * cos_t
    y_i = iy + ir * sin_t

    map_x = (x_p + r_fracs * (x_i - x_p)).astype(np.float32)
    map_y = (y_p + r_fracs * (y_i - y_p)).astype(np.float32)

    polar_image = cv2.remap(
        image_gray.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(np.uint8)
    return polar_image


def estimate_noise_mask(polar_image, reflection_thresh=230, eyelash_thresh=25):
    """Flags reflection and shadow noise."""
    reflections = polar_image >= reflection_thresh
    shadows = polar_image <= eyelash_thresh
    return reflections | shadows


def encode_iris(polar_image, noise_mask, wavelengths=(8, 16, 24), orientation=0.0):
    """Classical multi-scale Gabor wavelet phase quantization."""
    img_f = polar_image.astype(np.float32)
    codes, masks = [], []
    for wavelength in wavelengths:
        sigma = wavelength * 0.5
        ksize = int(6 * sigma) | 1
        kernel_real = cv2.getGaborKernel((ksize, ksize), sigma, theta=orientation, lambd=wavelength, gamma=1.0, psi=0, ktype=cv2.CV_32F)
        kernel_imag = cv2.getGaborKernel((ksize, ksize), sigma, theta=orientation, lambd=wavelength, gamma=1.0, psi=np.pi / 2, ktype=cv2.CV_32F)
        real = cv2.filter2D(img_f, cv2.CV_32F, kernel_real)
        imag = cv2.filter2D(img_f, cv2.CV_32F, kernel_imag)
        bit0 = (real >= 0).astype(np.uint8)
        bit1 = (imag >= 0).astype(np.uint8)
        codes.append(np.stack([bit0, bit1], axis=-1))
        masks.append(np.stack([noise_mask, noise_mask], axis=-1))
    return codes, masks


def masked_hamming_distance(codes_a, masks_a, codes_b, masks_b, max_shift=8):
    """Masked fractional Hamming distance for Gabor codes."""
    best_dist = 1.0
    for shift in range(-max_shift, max_shift + 1):
        total_bits, disagreeing_bits = 0, 0
        for code_a, mask_a, code_b, mask_b in zip(codes_a, masks_a, codes_b, masks_b):
            shifted_code_b = np.roll(code_b, shift, axis=1)
            shifted_mask_b = np.roll(mask_b, shift, axis=1)
            valid = (~mask_a) & (~shifted_mask_b)
            if not np.any(valid):
                continue
            disagreements = (code_a != shifted_code_b) & valid
            total_bits += int(valid.sum())
            disagreeing_bits += int(disagreements.sum())
        if total_bits > 0:
            dist = disagreeing_bits / total_bits
            best_dist = min(best_dist, dist)
    return best_dist
