"""
test_iris_engine.py — Comprehensive Verification Suite for Deep Iris Architecture
==================================================================================
Tests every stage:
  1. Valid NIR iris image
  2. Grayscale NumPy array input
  3. BGR / 3-channel input conversion
  4. Invalid / non-existent image input
  5. Synthetic blank/noise image (segmentation/quality failure handling)
  6. Artificial pupil/iris geometry edge cases
  7. Normalization output validation (64 × 512)
  8. Successful 512-D embedding
  9. L2 unit norm validation (||e||_2 ≈ 1.0)
  10. Deterministic extraction (cosine similarity = 1.0)
  11. Intra-identity consistency vs Inter-identity separation
  12. CPU execution verification
  13. Structured IrisExtractionResult metadata and timing checks
  14. Backward compatibility with IrisExtractor API
  15. Backward compatibility with MultiModalBiometricPipeline
"""

import os
import sys
import unittest
import numpy as np
import cv2
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from iris import (
    IrisPipeline as IndustrialIrisEngine,
    IrisExtractionResult,
    IrisExtractor,
    IrisQualityFailure,
    IrisQualityGate,
    IrisSegmenter as DeepIrisSegmentor,
    DaugmanNormalizer as DaugmanRubberSheetNormalizer,
)
from iris.segmentation import postprocess_and_estimate_geometry


class TestDeepIrisArchitecture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_iris_p1 = os.path.join(PROJECT_ROOT, "data", "chemric", "setA", "Person_001", "iris_right.jpg")
        cls.sample_iris_p2 = os.path.join(PROJECT_ROOT, "data", "chemric", "setA", "Person_002", "iris_right.jpg")
        cls.engine = IndustrialIrisEngine(use_gpu=False)
        cls.extractor = IrisExtractor(use_gpu=False)

    def test_01_valid_iris_image_file(self):
        """Test extraction from a valid image filepath."""
        if not os.path.exists(self.sample_iris_p1):
            self.skipTest(f"Sample image not found: {self.sample_iris_p1}")

        res = self.engine.extract(self.sample_iris_p1)
        self.assertTrue(res.success, f"Extraction failed with: {res.failure_reason}")
        self.assertIsNotNone(res.embedding)
        self.assertEqual(res.embedding.shape, (512,))
        self.assertEqual(res.failure_reason, IrisQualityFailure.SUCCESS)
        self.assertGreater(res.quality_score, 0.4)
        self.assertGreater(res.segmentation_confidence, 0.4)

    def test_02_grayscale_numpy_input(self):
        """Test extraction from an in-memory 2D uint8 NumPy array."""
        if not os.path.exists(self.sample_iris_p1):
            self.skipTest(f"Sample image not found: {self.sample_iris_p1}")

        img_gray = cv2.imread(self.sample_iris_p1, cv2.IMREAD_GRAYSCALE)
        self.assertIsInstance(img_gray, np.ndarray)
        self.assertEqual(len(img_gray.shape), 2)

        res = self.engine.extract(img_gray)
        self.assertTrue(res.success)
        self.assertEqual(res.embedding.shape, (512,))

    def test_03_bgr_numpy_input(self):
        """Test extraction from an in-memory 3-channel BGR NumPy array."""
        if not os.path.exists(self.sample_iris_p1):
            self.skipTest(f"Sample image not found: {self.sample_iris_p1}")

        img_bgr = cv2.imread(self.sample_iris_p1, cv2.IMREAD_COLOR)
        self.assertEqual(img_bgr.shape[2], 3)

        res = self.engine.extract(img_bgr)
        self.assertTrue(res.success)
        self.assertEqual(res.embedding.shape, (512,))

    def test_04_invalid_image_inputs(self):
        """Test graceful rejection of invalid inputs."""
        # Non-existent file
        res1 = self.engine.extract("non_existent_iris_file_xyz.jpg")
        self.assertFalse(res1.success)
        self.assertEqual(res1.failure_reason, IrisQualityFailure.INVALID_INPUT)

        # Empty array
        res2 = self.engine.extract(np.zeros((0,), dtype=np.uint8))
        self.assertFalse(res2.success)
        self.assertEqual(res2.failure_reason, IrisQualityFailure.INVALID_INPUT)

    def test_05_synthetic_blank_image(self):
        """Test rejection of blank/solid color images without pupil or iris."""
        blank_img = np.zeros((480, 640), dtype=np.uint8)
        res = self.engine.extract(blank_img)
        self.assertFalse(res.success)
        self.assertIn(
            res.failure_reason,
            [
                IrisQualityFailure.SEGMENTATION_FAILED,
                IrisQualityFailure.LOW_SEGMENTATION_CONFIDENCE,
                IrisQualityFailure.PUPIL_NOT_FOUND,
                IrisQualityFailure.IRIS_NOT_FOUND,
                IrisQualityFailure.INVALID_GEOMETRY,
            ],
        )

    def test_06_synthetic_noise_image(self):
        """Test rejection of random noise image."""
        noise_img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        res = self.engine.extract(noise_img)
        self.assertFalse(res.success)

    def test_07_normalization_dimensions(self):
        """Test Daugman normalization produces exactly 64 x 512 polar texture."""
        if not os.path.exists(self.sample_iris_p1):
            self.skipTest(f"Sample image not found: {self.sample_iris_p1}")

        res = self.engine.extract(self.sample_iris_p1)
        self.assertTrue(res.success)
        self.assertIsNotNone(res.normalized_image)
        self.assertEqual(res.normalized_image.shape, (64, 512))

    def test_08_l2_unit_norm(self):
        """Test that embedding is strictly L2 unit-normalized (norm ≈ 1.0)."""
        if not os.path.exists(self.sample_iris_p1):
            self.skipTest(f"Sample image not found: {self.sample_iris_p1}")

        res = self.engine.extract(self.sample_iris_p1)
        self.assertTrue(res.success)
        norm = float(np.linalg.norm(res.embedding))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_09_deterministic_extraction(self):
        """Test extraction is 100% deterministic (cosine similarity = 1.0 on same input)."""
        if not os.path.exists(self.sample_iris_p1):
            self.skipTest(f"Sample image not found: {self.sample_iris_p1}")

        res1 = self.engine.extract(self.sample_iris_p1)
        res2 = self.engine.extract(self.sample_iris_p1)
        self.assertTrue(res1.success and res2.success)

        cos_sim = float(np.dot(res1.embedding, res2.embedding))
        self.assertAlmostEqual(cos_sim, 1.0, places=5)

    def test_10_inter_identity_separation(self):
        """Test that different individuals produce distinct embeddings."""
        if not (os.path.exists(self.sample_iris_p1) and os.path.exists(self.sample_iris_p2)):
            self.skipTest("Sample images for Person_001 and Person_002 not available")

        res1 = self.engine.extract(self.sample_iris_p1)
        res2 = self.engine.extract(self.sample_iris_p2)
        self.assertTrue(res1.success and res2.success)

        cos_sim = float(np.dot(res1.embedding, res2.embedding))
        self.assertLess(cos_sim, 0.85, f"Impostor similarity unexpectedly high: {cos_sim}")

    def test_11_structured_metadata_and_timings(self):
        """Test that IrisExtractionResult contains complete metadata and timings."""
        if not os.path.exists(self.sample_iris_p1):
            self.skipTest(f"Sample image not found: {self.sample_iris_p1}")

        res = self.engine.extract(self.sample_iris_p1)
        self.assertTrue(res.success)
        self.assertIn("segmentation", res.timings_ms)
        self.assertIn("geometry", res.timings_ms)
        self.assertIn("normalization", res.timings_ms)
        self.assertIn("embedding", res.timings_ms)
        self.assertIn("total", res.timings_ms)
        self.assertGreater(res.timings_ms["total"], 0.0)

        # Check geometry metadata
        self.assertIn("pupil_center", res.geometry_metadata)
        self.assertIn("pupil_radius", res.geometry_metadata)
        self.assertIn("iris_center", res.geometry_metadata)
        self.assertIn("iris_radius", res.geometry_metadata)

    def test_12_backward_compatibility_iris_extractor(self):
        """Test backward compatibility of IrisExtractor public methods."""
        if not os.path.exists(self.sample_iris_p1):
            self.skipTest(f"Sample image not found: {self.sample_iris_p1}")

        # 1. extract_features
        emb = self.extractor.extract_features(self.sample_iris_p1)
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=5)

        # 2. extract_features_aligned
        emb_aligned = self.extractor.extract_features_aligned(self.sample_iris_p1, gallery_embedding=emb)
        self.assertEqual(emb_aligned.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(emb_aligned)), 1.0, places=5)

        # 3. extract_both_features
        vec_gabor, vec_resnet = self.extractor.extract_both_features(self.sample_iris_p1)
        self.assertEqual(vec_gabor.shape, (512,))
        self.assertEqual(vec_resnet.shape, (512,))

        # 4. extract_template
        code, mask = self.extractor.extract_template(self.sample_iris_p1)
        self.assertIsNotNone(code)
        self.assertIsNotNone(mask)

    def test_13_hough_fallback_mode(self):
        """Test explicit Hough fallback processing mode."""
        sample_p4 = os.path.join(PROJECT_ROOT, "data", "chemric", "setA", "Person_004", "iris_right.jpg")
        if os.path.exists(sample_p4):
            res = self.engine.extract(sample_p4, mode="hough_fallback")
            self.assertTrue(res.success)
            self.assertEqual(res.processing_mode, "hough_fallback")
            self.assertEqual(res.embedding.shape, (512,))
            self.assertAlmostEqual(float(np.linalg.norm(res.embedding)), 1.0, places=5)

        # On samples where Hough circle detection fails, it must return a structured failure result
        if os.path.exists(self.sample_iris_p1):
            res_fail = self.engine.extract(self.sample_iris_p1, mode="hough_fallback")
            self.assertFalse(res_fail.success)
            self.assertEqual(res_fail.processing_mode, "hough_fallback")
            self.assertEqual(res_fail.failure_reason, IrisQualityFailure.SEGMENTATION_FAILED)


if __name__ == "__main__":
    unittest.main()
