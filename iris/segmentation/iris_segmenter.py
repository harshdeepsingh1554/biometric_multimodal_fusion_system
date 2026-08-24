"""
iris_segmenter.py — Deep Semantic Segmentation & Geometry Estimation
====================================================================
Performs:
  1. 4-class semantic segmentation via UNet++ + scSE + MobileNetV2 ONNX model:
     - 0: eyeball, 1: iris, 2: pupil, 3: eyelashes
  2. Segmentation confidence estimation.
  3. Morphological cleanup and connected component filtering.
  4. Robust algebraic direct ellipse fitting for pupil and iris boundaries.
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
import onnxruntime as ort

logger = logging.getLogger(__name__)


def fit_robust_ellipse(contour: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Fits an ellipse to a contour and extracts center, axes, angle, eccentricity, and radius.
    Falls back to enclosing circle if contour has < 5 points or degenerate geometry.
    """
    if contour is None or len(contour) < 5:
        if contour is not None and len(contour) >= 3:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            return {
                "center": (float(x), float(y)),
                "axes": (float(radius * 2), float(radius * 2)),
                "semi_axes": (float(radius), float(radius)),
                "angle": 0.0,
                "radius": float(radius),
                "eccentricity": 0.0,
                "is_ellipse": False,
            }
        return None

    try:
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (axis_w, axis_h), angle = ellipse

        if axis_w <= 0 or axis_h <= 0 or np.isnan(cx) or np.isnan(cy):
            (x, y), radius = cv2.minEnclosingCircle(contour)
            return {
                "center": (float(x), float(y)),
                "axes": (float(radius * 2), float(radius * 2)),
                "semi_axes": (float(radius), float(radius)),
                "angle": 0.0,
                "radius": float(radius),
                "eccentricity": 0.0,
                "is_ellipse": False,
            }

        semi_a = max(axis_w, axis_h) / 2.0
        semi_b = min(axis_w, axis_h) / 2.0
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (semi_b / max(semi_a, 1e-6)) ** 2)))
        radius = float(np.sqrt(semi_a * semi_b))

        return {
            "center": (float(cx), float(cy)),
            "axes": (float(axis_w), float(axis_h)),
            "semi_axes": (float(semi_a), float(semi_b)),
            "angle": float(angle),
            "radius": float(radius),
            "eccentricity": float(eccentricity),
            "is_ellipse": True,
        }
    except Exception as e:
        logger.debug(f"Ellipse fitting exception: {e}, falling back to enclosing circle")
        (x, y), radius = cv2.minEnclosingCircle(contour)
        return {
            "center": (float(x), float(y)),
            "axes": (float(radius * 2), float(radius * 2)),
            "semi_axes": (float(radius), float(radius)),
            "angle": 0.0,
            "radius": float(radius),
            "eccentricity": 0.0,
            "is_ellipse": False,
        }


def extract_largest_component(binary_mask: np.ndarray, min_area: int = 50) -> Optional[np.ndarray]:
    """Isolates the largest connected component in a binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = int(np.argmax(areas)) + 1
    if stats[max_idx, cv2.CC_STAT_AREA] < min_area:
        return None

    return (labels == max_idx).astype(np.uint8) * 255


def postprocess_and_estimate_geometry(
    seg_dict: Dict[str, Any],
    threshold: float = 0.5,
    min_pupil_area: int = 50,
    min_iris_area: int = 200,
) -> Dict[str, Any]:
    """
    Postprocesses probability maps and extracts pupil and iris boundary parameters.
    """
    pupil_prob = seg_dict.get("pupil_prob")
    iris_prob = seg_dict.get("iris_prob")
    eyelashes_prob = seg_dict.get("eyelashes_prob")

    if pupil_prob is None or iris_prob is None:
        return {"valid": False, "invalid_reason": "Missing segmentation probability maps"}

    pupil_bin = (pupil_prob >= threshold).astype(np.uint8) * 255
    iris_bin = (iris_prob >= threshold).astype(np.uint8) * 255

    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    pupil_clean = cv2.morphologyEx(pupil_bin, cv2.MORPH_OPEN, kernel_3)
    pupil_clean = cv2.morphologyEx(pupil_clean, cv2.MORPH_CLOSE, kernel_5)

    iris_clean = cv2.morphologyEx(iris_bin, cv2.MORPH_OPEN, kernel_5)
    iris_clean = cv2.morphologyEx(iris_clean, cv2.MORPH_CLOSE, kernel_5)

    pupil_comp = extract_largest_component(pupil_clean, min_area=min_pupil_area)
    if pupil_comp is None:
        return {"valid": False, "invalid_reason": "No valid pupil connected component found"}

    iris_comp = extract_largest_component(iris_clean, min_area=min_iris_area)
    if iris_comp is None:
        return {"valid": False, "invalid_reason": "No valid iris connected component found"}

    combined_iris_mask = cv2.bitwise_or(iris_comp, pupil_comp)

    pupil_cnts, _ = cv2.findContours(pupil_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not pupil_cnts:
        return {"valid": False, "invalid_reason": "No pupil contour found"}
    pupil_cnt = max(pupil_cnts, key=cv2.contourArea)

    iris_cnts, _ = cv2.findContours(combined_iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not iris_cnts:
        return {"valid": False, "invalid_reason": "No iris contour found"}
    iris_cnt = max(iris_cnts, key=cv2.contourArea)

    pupil_ellipse = fit_robust_ellipse(pupil_cnt)
    iris_ellipse = fit_robust_ellipse(iris_cnt)

    if pupil_ellipse is None or iris_ellipse is None:
        return {"valid": False, "invalid_reason": "Failed to fit pupil or iris boundary"}

    px, py = pupil_ellipse["center"]
    pr = pupil_ellipse["radius"]
    ix, iy = iris_ellipse["center"]
    ir = iris_ellipse["radius"]

    center_dist = float(np.hypot(px - ix, py - iy))

    if eyelashes_prob is not None:
        eyelash_mask = (eyelashes_prob >= 0.5).astype(np.uint8) * 255
    else:
        eyelash_mask = np.zeros_like(pupil_comp)

    theoretical_annular_area = max(1.0, np.pi * max(1.0, ir**2 - pr**2))
    unoccluded_iris = (combined_iris_mask > 0) & (pupil_comp == 0) & (eyelash_mask == 0)
    visible_pixels = int(np.sum(unoccluded_iris))
    visible_ratio = float(np.clip(visible_pixels / theoretical_annular_area, 0.0, 1.0))

    inside_dist = cv2.pointPolygonTest(iris_cnt, (float(px), float(py)), measureDist=True)
    if inside_dist < -5.0:
        return {
            "valid": False,
            "invalid_reason": "Pupil center lies outside iris contour",
            "pupil_center": (px, py),
            "pupil_radius": pr,
            "iris_center": (ix, iy),
            "iris_radius": ir,
            "center_distance": center_dist,
            "visible_iris_ratio": visible_ratio,
        }

    return {
        "valid": True,
        "invalid_reason": "",
        "pupil_center": (px, py),
        "pupil_radius": pr,
        "pupil_ellipse": pupil_ellipse,
        "iris_center": (ix, iy),
        "iris_radius": ir,
        "iris_ellipse": iris_ellipse,
        "center_distance": center_dist,
        "pupil_eccentricity": pupil_ellipse["eccentricity"],
        "iris_eccentricity": iris_ellipse["eccentricity"],
        "visible_iris_ratio": visible_ratio,
        "pupil_mask": pupil_comp,
        "iris_mask": iris_comp,
        "noise_mask": eyelash_mask,
    }


class IrisGeometryEstimator:
    """Dedicated geometry boundary fitting engine."""
    def __init__(self, threshold: float = 0.5, min_pupil_area: int = 50, min_iris_area: int = 200):
        self.threshold = threshold
        self.min_pupil_area = min_pupil_area
        self.min_iris_area = min_iris_area

    def estimate(self, seg_dict: Dict[str, Any]) -> Dict[str, Any]:
        return postprocess_and_estimate_geometry(
            seg_dict,
            threshold=self.threshold,
            min_pupil_area=self.min_pupil_area,
            min_iris_area=self.min_iris_area,
        )


class IrisSegmenter:
    """
    ONNX UNet++ + scSE + MobileNetV2 Semantic Segmentation Stage.
    """
    INDEX_EYEBALL = 0
    INDEX_IRIS = 1
    INDEX_PUPIL = 2
    INDEX_EYELASHES = 3

    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = False):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "weights", "iris", "iris_semseg_upp_scse_mobilenetv2.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Segmentation ONNX model not found at: {model_path}")

        providers = ["CPUExecutionProvider"]
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        logger.info(f"Loading IrisSegmenter ONNX from {model_path} with providers={providers}")
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.target_h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 480
        self.target_w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 640

    def segment(self, input_tensor: np.ndarray, orig_hw: Tuple[int, int]) -> Dict[str, Any]:
        orig_h, orig_w = orig_hw
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        raw_output = outputs[0]  # (1, 4, 480, 640)

        probs = np.zeros((4, orig_h, orig_w), dtype=np.float32)
        for c in range(4):
            c_map = raw_output[0, c]
            if (orig_h, orig_w) == (self.target_h, self.target_w):
                probs[c] = c_map
            else:
                probs[c] = cv2.resize(c_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        pupil_map = probs[self.INDEX_PUPIL]
        iris_map = probs[self.INDEX_IRIS]

        pupil_top = np.sort(pupil_map.flatten())[-max(1, pupil_map.size // 100):]
        iris_top = np.sort(iris_map.flatten())[-max(1, iris_map.size // 50):]

        pupil_conf = float(np.mean(pupil_top)) if pupil_top.size > 0 else 0.0
        iris_conf = float(np.mean(iris_top)) if iris_top.size > 0 else 0.0
        confidence = float(0.5 * pupil_conf + 0.5 * iris_conf)

        return {
            "probs": probs,
            "eyeball_prob": probs[self.INDEX_EYEBALL],
            "iris_prob": probs[self.INDEX_IRIS],
            "pupil_prob": probs[self.INDEX_PUPIL],
            "eyelashes_prob": probs[self.INDEX_EYELASHES],
            "confidence": confidence,
            "raw_output": raw_output,
        }
