"""
fingerprint_extractor.py — Unified Public Fingerprint Extractor Facade
======================================================================
Provides a clean, production-grade interface for DeepPrint fixed-length
fingerprint feature extraction:
  - Generates 512-D L2-normalized float32 biometric embeddings.
  - Supports DeepPrint_Tex, DeepPrint_TexMinu, and STN-aligned models.
  - Computes quality scores and cosine / euclidean matching distances.
"""

import os
import logging
from typing import Optional, Union, Tuple, Dict, Any
import numpy as np

from fingerprint.pipeline import FingerprintPipeline, FingerprintExtractionResult

logger = logging.getLogger(__name__)


class DeepPrintFingerprintExtractor:
    """
    Public DeepPrint Fingerprint Extractor Facade.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "deepprint_tex",
        embedding_dims: int = 512,
        use_gpu: bool = False,
        use_stn: bool = False,
        apply_clahe: bool = False,
    ):
        self.pipeline = FingerprintPipeline(
            model_path=model_path,
            model_type=model_type,
            embedding_dims=embedding_dims,
            use_gpu=use_gpu,
            use_stn=use_stn,
            apply_clahe=apply_clahe,
        )
        self.device = self.pipeline.device
        self.model = self.pipeline.model

    def extract_features(
        self,
        image_path_or_array: Union[str, np.ndarray],
        l2_normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Primary interface: Returns 512-D L2-normalized NumPy embedding array or None on failure.
        """
        result = self.pipeline.extract(image_path_or_array, l2_normalize=l2_normalize)
        if result.success and result.embedding is not None:
            return result.embedding
        return None

    def extract_structured(
        self,
        image_path_or_array: Union[str, np.ndarray],
        l2_normalize: bool = True
    ) -> FingerprintExtractionResult:
        """
        Returns full structured extraction result with quality score, model type, and timings.
        """
        return self.pipeline.extract(image_path_or_array, l2_normalize=l2_normalize)

    def compute_quality_score(self, image_path_or_array: Union[str, np.ndarray]) -> float:
        """Computes heuristic fingerprint quality score in [0.0, 1.0]."""
        res = self.pipeline.extract(image_path_or_array)
        return float(res.quality_score)

    def compute_distance(self, template_a: np.ndarray, template_b: np.ndarray) -> float:
        """
        Computes cosine distance (1.0 - cosine_similarity) between two 512-D embeddings.
        """
        if template_a is None or template_b is None:
            return 1.0
        norm_a = template_a / max(np.linalg.norm(template_a), 1e-12)
        norm_b = template_b / max(np.linalg.norm(template_b), 1e-12)
        cosine_sim = float(np.dot(norm_a, norm_b))
        return 1.0 - cosine_sim
