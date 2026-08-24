"""
iris — Modular Deep Iris Biometric System
=========================================
Architecture:
  - iris.preprocessing (IrisPreprocessor, apply_clahe)
  - iris.segmentation (IrisSegmenter, IrisGeometryEstimator, fit_robust_ellipse)
  - iris.normalization (DaugmanNormalizer)
  - iris.models (IrisResNetModel, IResNet, iresnet100, IBasicBlock)
  - iris.quality (IrisQualityGate, IrisQualityFailure)
  - iris.pipeline (IrisPipeline, IrisExtractionResult)
  - iris.extractor (IrisExtractor)
"""

from iris.pipeline import IrisPipeline, IrisExtractionResult
from iris.extractor import IrisExtractor
from iris.quality import IrisQualityGate, IrisQualityFailure
from iris.preprocessing import IrisPreprocessor
from iris.segmentation import IrisSegmenter
from iris.normalization import DaugmanNormalizer
from iris.models import IrisResNetModel

__all__ = [
    "IrisPipeline",
    "IrisExtractor",
    "IrisExtractionResult",
    "IrisQualityGate",
    "IrisQualityFailure",
    "IrisPreprocessor",
    "IrisSegmenter",
    "DaugmanNormalizer",
    "IrisResNetModel",
]
