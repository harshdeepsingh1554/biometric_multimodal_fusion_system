"""
iris_quality.py — Multi-Criteria Biometric Quality Gate for Iris
================================================================
Evaluates:
  1. Deep segmentation model confidence.
  2. Concentricity / center offset between pupil and limbus boundaries.
  3. Radius ratio (pupil radius / iris radius).
  4. Boundary eccentricities (circularity vs distortion).
  5. Unoccluded visible iris surface area ratio.
  6. Image sharpness (Laplacian variance on unmasked iris texture).
"""

from enum import Enum
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2


class IrisQualityFailure(str, Enum):
    SUCCESS = "SUCCESS"
    INVALID_INPUT = "INVALID_INPUT"
    SEGMENTATION_FAILED = "SEGMENTATION_FAILED"
    LOW_SEGMENTATION_CONFIDENCE = "LOW_SEGMENTATION_CONFIDENCE"
    PUPIL_NOT_FOUND = "PUPIL_NOT_FOUND"
    IRIS_NOT_FOUND = "IRIS_NOT_FOUND"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"
    HIGH_OCCLUSION = "HIGH_OCCLUSION"
    LOW_VISIBLE_IRIS = "LOW_VISIBLE_IRIS"
    LOW_SHARPNESS = "LOW_SHARPNESS"
    NORMALIZATION_FAILED = "NORMALIZATION_FAILED"
    QUALITY_FAILED = "QUALITY_FAILED"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"


class IrisQualityGate:
    """
    Evaluates multi-criteria biometric quality of iris captures.
    """

    def __init__(
        self,
        min_seg_confidence: float = 0.40,
        max_center_offset_ratio: float = 0.35,
        min_radius_ratio: float = 0.15,
        max_radius_ratio: float = 0.75,
        max_eccentricity: float = 0.75,
        min_visible_ratio: float = 0.50,
        min_sharpness: float = 15.0,
    ):
        self.min_seg_confidence = min_seg_confidence
        self.max_center_offset_ratio = max_center_offset_ratio
        self.min_radius_ratio = min_radius_ratio
        self.max_radius_ratio = max_radius_ratio
        self.max_eccentricity = max_eccentricity
        self.min_visible_ratio = min_visible_ratio
        self.min_sharpness = min_sharpness

    def evaluate_quality(
        self,
        image_gray: np.ndarray,
        seg_confidence: float,
        geometry_meta: Dict[str, Any],
    ) -> Tuple[bool, IrisQualityFailure, float, Dict[str, Any]]:
        """
        Returns: (passed_gate: bool, failure_reason: IrisQualityFailure, quality_score: float, quality_meta: dict)
        """
        quality_meta = {
            "seg_confidence": float(seg_confidence),
            "sharpness": 0.0,
            "center_offset": 0.0,
            "radius_ratio": 0.0,
            "visible_ratio": 0.0,
        }

        # 1. Segmentation Confidence
        if seg_confidence < self.min_seg_confidence:
            return False, IrisQualityFailure.LOW_SEGMENTATION_CONFIDENCE, float(seg_confidence), quality_meta

        # 2. Geometry Validity
        if not geometry_meta.get("valid", False):
            return False, IrisQualityFailure.INVALID_GEOMETRY, 0.0, quality_meta

        pr = geometry_meta["pupil_radius"]
        ir = geometry_meta["iris_radius"]
        center_dist = geometry_meta["center_distance"]
        pupil_ecc = geometry_meta.get("pupil_eccentricity", 0.0)
        iris_ecc = geometry_meta.get("iris_eccentricity", 0.0)
        visible_ratio = geometry_meta.get("visible_iris_ratio", 1.0)

        quality_meta["center_offset"] = center_dist
        quality_meta["radius_ratio"] = float(pr / max(ir, 1e-6))
        quality_meta["visible_ratio"] = visible_ratio

        # 3. Concentricity
        if center_dist > self.max_center_offset_ratio * ir:
            return False, IrisQualityFailure.INVALID_GEOMETRY, 0.2, quality_meta

        # 4. Radii Ratios
        radius_ratio = pr / max(ir, 1e-6)
        if radius_ratio < self.min_radius_ratio or radius_ratio > self.max_radius_ratio:
            return False, IrisQualityFailure.INVALID_GEOMETRY, 0.3, quality_meta

        # 5. Eccentricity
        if pupil_ecc > self.max_eccentricity or iris_ecc > self.max_eccentricity:
            return False, IrisQualityFailure.INVALID_GEOMETRY, 0.35, quality_meta

        # 6. Visible Iris Ratio
        if visible_ratio < self.min_visible_ratio:
            return False, IrisQualityFailure.HIGH_OCCLUSION, float(visible_ratio), quality_meta

        # 7. Sharpness Check
        lap_var = float(cv2.Laplacian(image_gray, cv2.CV_64F).var())
        quality_meta["sharpness"] = lap_var
        if lap_var < self.min_sharpness:
            return False, IrisQualityFailure.LOW_SHARPNESS, float(min(1.0, lap_var / 50.0)), quality_meta

        # Aggregate Quality Score [0, 1]
        score_conf = np.clip(seg_confidence, 0.0, 1.0)
        score_geom = np.clip(1.0 - (center_dist / (self.max_center_offset_ratio * ir + 1e-6)), 0.0, 1.0)
        score_vis = np.clip(visible_ratio, 0.0, 1.0)
        score_sharp = np.clip(lap_var / 100.0, 0.0, 1.0)

        quality_score = float(0.35 * score_conf + 0.25 * score_geom + 0.25 * score_vis + 0.15 * score_sharp)
        return True, IrisQualityFailure.SUCCESS, quality_score, quality_meta
