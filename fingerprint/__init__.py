"""
fingerprint — Modular Fixed-Length Fingerprint Feature Extraction Package
========================================================================
Implements DeepPrint fixed-length representation models from
tim-rohwedder/fixed-length-fingerprint-extractors:
  - DeepPrint_Tex: Inception-v4 Texture model
  - DeepPrint_TexMinu: Dual-branch Texture (256-D) + Minutiae (256-D) -> 512-D
  - DeepPrint_LocTexMinu: Spatial Transformer (STN) + Dual-branch model
  - DeepPrintFingerprintExtractor: Public extraction facade
"""

from fingerprint.extractor import DeepPrintFingerprintExtractor
from fingerprint.pipeline import FingerprintPipeline, FingerprintExtractionResult
from fingerprint.preprocessing import (
    FingerprintPreprocessor,
    pad_and_resize_fingerprint,
    enhance_fingerprint_clahe,
)
from fingerprint.models import (
    DeepPrint_Tex,
    DeepPrint_TexMinu,
    DeepPrint_LocTexMinu,
    TextureBranch,
    MinutiaStem,
    MinutiaEmbedding,
    LocalizationNetwork,
    DEEPPRINT_INPUT_SIZE,
)

__all__ = [
    "DeepPrintFingerprintExtractor",
    "FingerprintPipeline",
    "FingerprintExtractionResult",
    "FingerprintPreprocessor",
    "pad_and_resize_fingerprint",
    "enhance_fingerprint_clahe",
    "DeepPrint_Tex",
    "DeepPrint_TexMinu",
    "DeepPrint_LocTexMinu",
    "TextureBranch",
    "MinutiaStem",
    "MinutiaEmbedding",
    "LocalizationNetwork",
    "DEEPPRINT_INPUT_SIZE",
]
