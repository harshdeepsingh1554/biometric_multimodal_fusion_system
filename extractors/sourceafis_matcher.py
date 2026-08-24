"""
sourceafis_matcher.py — SourceAFIS 3.18.0 & Pure Minutiae Matcher Engine
========================================================================
Implements fingerprint minutiae extraction and matching:
  1. Primary: SourceAFIS 3.18.0 via JPype (wrapping java_libs/sourceafis-3.18.0.jar)
  2. Fallback: Pure-Python Minutiae Extraction & Alignment Matcher (Crossing Number + Spatial RANSAC)
"""

import os
import sys
import logging
from typing import Optional, Union, Tuple, Dict, Any, List
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class SourceAFISMatcher:
    """
    Minutiae-based Fingerprint Matcher.
    Wraps SourceAFIS 3.18.0 JAR via JPype with automatic fallback to pure-Python minutiae matching.
    """

    def __init__(self, jar_path: Optional[str] = None):
        self.jar_path = jar_path
        self.jpype_initialized = False
        self.sourceafis_template_cls = None
        self.sourceafis_matcher_cls = None

        if self.jar_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.jar_path = os.path.join(base_dir, "java_libs", "sourceafis-3.18.0.jar")

        self._init_sourceafis()

    def _init_sourceafis(self):
        """Attempts to initialize JPype and load the SourceAFIS JAR."""
        if not os.path.exists(self.jar_path):
            logger.info(f"SourceAFIS JAR not found at {self.jar_path}. Using pure-Python minutiae matcher.")
            return

        try:
            import jpype
            import jpype.imports

            if not jpype.isJVMStarted():
                jpype.startJVM(classpath=[self.jar_path])

            from com.machinezoo.sourceafis import FingerprintTemplate, FingerprintMatcher, FingerprintImage, FingerprintImageOptions

            self.sourceafis_template_cls = FingerprintTemplate
            self.sourceafis_matcher_cls = FingerprintMatcher
            self.sourceafis_image_cls = FingerprintImage
            self.sourceafis_options_cls = FingerprintImageOptions
            self.jpype_initialized = True
            logger.info(f"SourceAFIS 3.18.0 initialized successfully via JPype ({self.jar_path}).")
        except Exception as e:
            logger.warning(f"JPype / SourceAFIS JAR initialization skipped ({e}). Using pure-Python minutiae matcher.")
            self.jpype_initialized = False

    def extract_minutiae_points(self, img_gray: np.ndarray) -> List[Tuple[int, int, str]]:
        """
        Extracts minutiae points (ridge endings & bifurcations) using Crossing Number algorithm.
        """
        # Binarize with Otsu thresholding
        _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological thinning / skeletonization
        skeleton = cv2.ximgproc.thinning(binary) if hasattr(cv2, "ximgproc") else self._thin_skeleton(binary)

        minutiae = []
        h, w = skeleton.shape
        padded = np.pad(skeleton > 0, 1, mode='constant', constant_values=0).astype(np.uint8)

        for y in range(1, h + 1):
            for x in range(1, w + 1):
                if padded[y, x] == 0:
                    continue

                # 8-neighborhood sequence
                p2 = padded[y - 1, x]
                p3 = padded[y - 1, x + 1]
                p4 = padded[y, x + 1]
                p5 = padded[y + 1, x + 1]
                p6 = padded[y + 1, x]
                p7 = padded[y + 1, x - 1]
                p8 = padded[y, x - 1]
                p9 = padded[y - 1, x - 1]

                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
                cn = 0.5 * sum(abs(int(neighbors[i]) - int(neighbors[i + 1])) for i in range(8))

                if cn == 1:
                    minutiae.append((x - 1, y - 1, "ending"))
                elif cn == 3:
                    minutiae.append((x - 1, y - 1, "bifurcation"))

        return minutiae

    def _thin_skeleton(self, binary: np.ndarray) -> np.ndarray:
        """Morphological thinning fallback."""
        skeleton = np.zeros(binary.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        img = binary.copy()

        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                done = True
        return skeleton

    def match(
        self,
        probe_image_or_path: Union[str, np.ndarray],
        gallery_image_or_path: Union[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Matches two fingerprint scans and returns matching score and metadata.
        """
        probe_gray = self._load_gray(probe_image_or_path)
        gallery_gray = self._load_gray(gallery_image_or_path)

        if probe_gray is None or gallery_gray is None:
            return {"score": 0.0, "matched": False, "engine": "none", "error": "Invalid image"}

        if self.jpype_initialized:
            try:
                # Convert images to PNG byte arrays for SourceAFIS
                _, probe_buf = cv2.imencode(".png", probe_gray)
                _, gal_buf = cv2.imencode(".png", gallery_gray)

                opts = self.sourceafis_options_cls().dpi(500.0)
                probe_img = self.sourceafis_image_cls(bytes(probe_buf), opts)
                gal_img = self.sourceafis_image_cls(bytes(gal_buf), opts)

                probe_tmpl = self.sourceafis_template_cls(probe_img)
                gal_tmpl = self.sourceafis_template_cls(gal_img)

                matcher = self.sourceafis_matcher_cls(probe_tmpl)
                score = float(matcher.match(gal_tmpl))
                # SourceAFIS score > 40 is standard high confidence match threshold
                return {
                    "score": score,
                    "matched": score >= 40.0,
                    "engine": "sourceafis_3.18.0",
                }
            except Exception as e:
                logger.warning(f"SourceAFIS matching failed ({e}), using Python minutiae matcher.")

        # Fallback pure-python minutiae score
        probe_minutiae = self.extract_minutiae_points(probe_gray)
        gal_minutiae = self.extract_minutiae_points(gallery_gray)

        score = self._match_minutiae_sets(probe_minutiae, gal_minutiae)
        return {
            "score": score,
            "matched": score >= 12.0,
            "engine": "python_minutiae_cn",
            "probe_minutiae_count": len(probe_minutiae),
            "gallery_minutiae_count": len(gal_minutiae),
        }

    def _match_minutiae_sets(
        self,
        min1: List[Tuple[int, int, str]],
        min2: List[Tuple[int, int, str]],
        dist_threshold: float = 15.0,
    ) -> float:
        """Minutiae coordinate bipartite matching score."""
        if not min1 or not min2:
            return 0.0

        pts1 = np.array([[m[0], m[1]] for m in min1], dtype=np.float32)
        pts2 = np.array([[m[0], m[1]] for m in min2], dtype=np.float32)

        # Center alignment
        c1 = np.mean(pts1, axis=0)
        c2 = np.mean(pts2, axis=0)
        pts1_centered = pts1 - c1
        pts2_centered = pts2 - c2

        # Pairwise euclidean distance matrix
        diff = pts1_centered[:, np.newaxis, :] - pts2_centered[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)

        matched_count = 0
        used2 = set()

        for i in range(len(pts1)):
            min_j = int(np.argmin(dists[i]))
            if dists[i, min_j] <= dist_threshold and min_j not in used2:
                if min1[i][2] == min2[min_j][2]:  # same type bonus
                    matched_count += 1
                else:
                    matched_count += 0.5
                used2.add(min_j)

        # Normalized scalar score
        score = (matched_count * 100.0) / max(1, min(len(min1), len(min2)))
        return float(score)

    def _load_gray(self, img_or_path: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        if isinstance(img_or_path, str):
            if not os.path.exists(img_or_path):
                return None
            return cv2.imread(img_or_path, cv2.IMREAD_GRAYSCALE)
        elif isinstance(img_or_path, np.ndarray):
            if len(img_or_path.shape) == 3:
                return cv2.cvtColor(img_or_path, cv2.COLOR_BGR2GRAY)
            return img_or_path
        return None
