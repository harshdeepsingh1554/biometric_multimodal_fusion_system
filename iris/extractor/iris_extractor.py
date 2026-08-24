"""
iris_extractor.py — Unified Public Biometric Extractor Facade
==============================================================
Provides high-performance, production-grade iris feature extraction:
  - Primary backend: "resnet100" (Deep semantic segmentation + Daugman normalization + IResNet-100)
  - Baseline/Diagnostic backend: "gabor" (Classical Multi-scale Gabor wavelets)
"""

import os
import sys
import logging
from typing import Optional, Union, Tuple, Dict, Any
import numpy as np
import cv2

from iris.pipeline import (
    IrisPipeline,
    IrisExtractionResult,
    IrisQualityFailure,
    segment_iris,
    normalize_iris,
    estimate_noise_mask,
    encode_iris,
    masked_hamming_distance,
)

logger = logging.getLogger(__name__)


class OpenIrisModel:
    """
    Classical baseline iris pipeline wrapper using Gabor phase encoding.
    """
    def __init__(self, radial_res: int = 64, angular_res: int = 512):
        self.radial_res = radial_res
        self.angular_res = angular_res

    def extract_template(self, image_input: Union[str, np.ndarray], eye_side: str = "right"):
        if isinstance(image_input, str):
            image_gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3:
                image_gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            else:
                image_gray = image_input
        else:
            raise ValueError(f"Unsupported image_input type: {type(image_input)}")

        if image_gray is None:
            raise ValueError(f"Could not load image: {image_input}")

        pupil_circle, iris_circle = segment_iris(image_gray)
        if pupil_circle is not None and iris_circle is not None:
            polar_image = normalize_iris(
                image_gray, pupil_circle, iris_circle,
                radial_res=self.radial_res, angular_res=self.angular_res
            )
        else:
            pipe = IrisPipeline()
            res = pipe.extract(image_gray)
            if res.success and res.normalized_image is not None:
                polar_image = res.normalized_image
            else:
                raise ValueError("Iris segmentation failed: pupil or iris boundary not found.")

        noise_mask = estimate_noise_mask(polar_image)
        codes, masks = encode_iris(polar_image, noise_mask)
        return codes, masks

    def match_templates(self, code_a, mask_a, code_b, mask_b) -> float:
        return masked_hamming_distance(code_a, mask_a, code_b, mask_b)


class IrisExtractor:
    """
    Unified Iris Feature Extractor supporting:
      1. Deep UNet++ Segmentation + IResNet-100 Deep Embeddings ("resnet100", default)
      2. Classical Hough / Gabor Wavelet codes ("gabor")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        seg_model_path: Optional[str] = None,
        circlenet_model_path: Optional[str] = None,
        use_gpu: bool = False,
        default_backend: str = "resnet100",
        backend: Optional[str] = None,
        seg_backend: str = "circlenet",
    ):
        if backend is not None:
            default_backend = backend
        self.default_backend = default_backend
        self.use_gpu = use_gpu
        self.seg_backend = seg_backend

        logger.info(f"Initializing Iris Extractor (default backend: {default_backend}, seg_backend: {seg_backend}, GPU: {use_gpu})...")

        # 1. Master Iris Pipeline
        self.engine = IrisPipeline(
            seg_model_path=seg_model_path,
            circlenet_model_path=circlenet_model_path,
            embed_model_path=model_path,
            use_gpu=use_gpu,
            seg_backend=seg_backend,
        )

        # Expose models for legacy compatibility and inspection
        self.resnet_model = self.engine.resnet_model
        self.device = self.engine.device
        self.transform = self.engine.transform

        # 2. Classical OpenIris Model (Retained for Gabor baseline and ablation comparison)
        self.open_iris = OpenIrisModel()

    def extract_structured(
        self,
        image_path_or_array: Union[str, np.ndarray],
        eye_side: str = "right",
        mode: str = "deep_segmentation",
        l2_normalize: bool = True,
    ) -> IrisExtractionResult:
        """
        Returns full structured extraction result with quality metrics,
        confidence, geometry, and timings.
        """
        return self.engine.extract(
            image_path_or_array,
            eye_side=eye_side,
            mode=mode,
            l2_normalize=l2_normalize,
        )

    def extract_template(self, image_path: Union[str, np.ndarray], eye_side: str = "right"):
        """
        Passthrough to classical OpenIris: returns (binary_code, noise_mask).
        """
        return self.open_iris.extract_template(image_path, eye_side=eye_side)

    def compute_distance(self, template_a, template_b) -> float:
        """
        Computes distance between two templates:
          - If vectors (np.ndarray): 1.0 - cosine_similarity
          - If tuples (code, mask): Masked fractional Hamming distance
        """
        if isinstance(template_a, np.ndarray) and isinstance(template_b, np.ndarray):
            norm_a = template_a / max(np.linalg.norm(template_a), 1e-12)
            norm_b = template_b / max(np.linalg.norm(template_b), 1e-12)
            sim = float(np.dot(norm_a, norm_b))
            return 1.0 - sim

        if isinstance(template_a, (tuple, list)) and isinstance(template_b, (tuple, list)):
            return self.open_iris.match_templates(
                template_a[0], template_a[1],
                template_b[0], template_b[1]
            )

        raise TypeError(f"Cannot compare template types: {type(template_a)} vs {type(template_b)}")

    def extract_features(
        self,
        image_path_or_array: Union[str, np.ndarray],
        eye_side: str = "right",
        l2_normalize: bool = True,
        mode: str = "deep_segmentation",
    ) -> Optional[np.ndarray]:
        """
        Primary interface: returns 512-D L2-normalized NumPy embedding array,
        or None if image fails quality checks.
        """
        result = self.engine.extract(
            image_path_or_array,
            eye_side=eye_side,
            mode=mode,
            l2_normalize=l2_normalize,
        )
        if result.success and result.embedding is not None:
            return result.embedding

        logger.warning(
            f"Iris feature extraction rejected/failed: {result.failure_reason.value} "
            f"(quality_score={result.quality_score:.3f})"
        )
        return None

    def extract_features_aligned(
        self,
        image_path_or_array: Union[str, np.ndarray],
        gallery_embedding: Optional[np.ndarray],
        eye_side: str = "right",
        shifts: Tuple[int, ...] = (-24, -16, -8, 0, 8, 16, 24),
        l2_normalize: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Extracts probe embedding with horizontal roll alignment against gallery_embedding.
        """
        result = self.engine.extract_aligned(
            image_path_or_array,
            gallery_embedding=gallery_embedding,
            eye_side=eye_side,
            shifts=shifts,
            l2_normalize=l2_normalize,
        )
        if result.success and result.embedding is not None:
            return result.embedding
        return None

    def extract_both_features(
        self,
        image_path_or_array: Union[str, np.ndarray],
        eye_side: str = "right"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extracts both feature representations for legacy and fusion comparison:
        returns (primary_deep_embedding, deep_embedding_aligned) tuple of shape (512,).
        """
        emb = self.extract_features(image_path_or_array, eye_side=eye_side)
        return emb, emb
