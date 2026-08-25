"""
fingerprint_pipeline.py — Master Pipeline for DeepPrint Fingerprint Extractor
=============================================================================
Orchestrates the complete sequential fingerprint feature extraction workflow:
  Raw Image -> Preprocessing (299x299 padding/resizing) -> STN Rectification -> DeepPrint -> 512-D L2 Embedding
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any
import numpy as np
import cv2
import torch

from fingerprint.preprocessing import FingerprintPreprocessor
from fingerprint.models import (
    DeepPrint_Tex,
    DeepPrint_TexMinu,
    DeepPrint_LocTexMinu,
    DEEPPRINT_INPUT_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass
class FingerprintExtractionResult:
    """Structured extraction output containing embedding, quality, and timings."""
    embedding: Optional[np.ndarray] = None
    success: bool = False
    quality_score: float = 0.0
    model_type: str = "DeepPrint_TexMinu"
    timings_ms: Dict[str, float] = field(default_factory=dict)
    preprocessed_image: Optional[np.ndarray] = None


class FingerprintPipeline:
    """
    Master Fingerprint Pipeline orchestrating preprocessing, model inference, and L2 normalization.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "deepprint_texminu",
        embedding_dims: int = 512,
        use_gpu: bool = False,
        use_stn: bool = False,
        apply_clahe: bool = False,
    ):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model_type = model_type.lower()
        self.embedding_dims = embedding_dims
        self.use_stn = use_stn

        # 1. Preprocessor (299x299 for Inception-v4)
        self.preprocessor = FingerprintPreprocessor(
            target_size=(DEEPPRINT_INPUT_SIZE, DEEPPRINT_INPUT_SIZE),
            apply_clahe=apply_clahe,
        )

        # 2. Select Architecture
        if self.use_stn:
            self.model = DeepPrint_LocTexMinu(embedding_dims=self.embedding_dims)
        else:
            if "tex" in self.model_type and "minu" not in self.model_type:
                self.model = DeepPrint_Tex(embedding_dims=self.embedding_dims)
            else:
                self.model = DeepPrint_TexMinu(embedding_dims=self.embedding_dims)

        # 3. Load Checkpoint Weights
        if model_path is not None and os.path.exists(model_path):
            self.model.load_weights(model_path, self.device)
        else:
            # Check default potential paths
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            default_paths = [
                os.path.join(base_dir, "models", "DeepPrint_Tex_512", "best_model.pyt"),
                os.path.join(base_dir, "weights", "finger", "DeepPrint_Tex_512", "best_model.pyt"),
                os.path.join(base_dir, "weights", "finger", "best_model.pyt"),
                os.path.join(base_dir, "weights", "finger", "finger_extractor_best.pth"),
            ]
            found_path = next((p for p in default_paths if os.path.exists(p)), None)
            if found_path is not None:
                self.model.load_weights(found_path, self.device)
            else:
                logger.info("DeepPrint model initialized (ready to load weights when available).")

        self.model.to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        logger.info(
            f"FingerprintPipeline initialized on {self.device} (model: {self.model.__class__.__name__}, dims: {self.embedding_dims})."
        )

    def extract(
        self,
        image_input: Union[str, np.ndarray],
        l2_normalize: bool = True
    ) -> FingerprintExtractionResult:
        """Runs the complete extraction pipeline."""
        timings = {}
        t_start = time.perf_counter()

        # Step 1: Preprocessing
        t0 = time.perf_counter()
        try:
            tensor, padded_img = self.preprocessor.load_and_preprocess(image_input)
            timings["preprocessing"] = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return FingerprintExtractionResult(
                embedding=None,
                success=False,
                model_type=self.model.__class__.__name__,
                timings_ms={"total": (time.perf_counter() - t_start) * 1000.0}
            )

        # Step 2: Quality estimation
        quality = self.preprocessor.compute_quality(padded_img)

        # Step 3: Forward inference
        t0 = time.perf_counter()
        tensor = tensor.to(self.device)
        try:
            with torch.no_grad():
                emb_out = self.model(tensor)
                emb = emb_out.squeeze(0).cpu().numpy().astype(np.float32)

            if l2_normalize:
                norm = np.linalg.norm(emb)
                if norm > 1e-12:
                    emb = (emb / norm).astype(np.float32)

            timings["forward"] = (time.perf_counter() - t0) * 1000.0
            timings["total"] = (time.perf_counter() - t_start) * 1000.0

            return FingerprintExtractionResult(
                embedding=emb,
                success=True,
                quality_score=quality,
                model_type=self.model.__class__.__name__,
                timings_ms=timings,
                preprocessed_image=padded_img,
            )
        except Exception as e:
            logger.error(f"DeepPrint model inference failed: {e}")
            return FingerprintExtractionResult(
                embedding=None,
                success=False,
                quality_score=quality,
                model_type=self.model.__class__.__name__,
                timings_ms={"total": (time.perf_counter() - t_start) * 1000.0},
                preprocessed_image=padded_img,
            )
