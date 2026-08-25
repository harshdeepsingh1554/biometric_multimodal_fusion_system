"""
deepprint.py — DeepPrint Fixed-Length Fingerprint Architectures
==============================================================
Implements DeepPrint fixed-length representation models from
tim-rohwedder/fixed-length-fingerprint-extractors:
  - DeepPrint_Tex: Inception-v4 Texture model (256-D / 512-D)
  - DeepPrint_TexMinu: Dual-branch Texture (256-D) + Minutiae (256-D) -> 512-D
  - DeepPrint_LocTexMinu: STN Localization + Dual-branch Texture + Minutiae
"""

import os
import logging
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from fingerprint.models.inception_v4 import (
    BasicConv2d,
    Mixed_3a,
    Mixed_4a,
    Mixed_5a,
    Inception_A,
    Inception_B,
    Inception_C,
    Reduction_A,
    Reduction_B,
)
from fingerprint.models.localization import LocalizationNetwork

logger = logging.getLogger(__name__)

DEEPPRINT_INPUT_SIZE = 299


class InceptionV4_Stem(nn.Module):
    """Initial convolutional stem reducing 299x299 to 35x35 feature maps."""
    def __init__(self, in_channels: int = 1):
        super(InceptionV4_Stem, self).__init__()
        self.features = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class TextureBranch(nn.Module):
    """Texture branch matching official DeepPrint architecture."""
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self._0_block = nn.Sequential(
            Inception_A(), Inception_A(), Inception_A(), Inception_A(), Reduction_A()
        )
        self._1_block = nn.Sequential(
            Inception_B(), Inception_B(), Inception_B(), Inception_B(),
            Inception_B(), Inception_B(), Inception_B(), Reduction_B()
        )
        self._2_block = nn.Sequential(
            Inception_C(), Inception_C(), Inception_C()
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self._6_linear = nn.Linear(1536, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._0_block(x)
        x = self._1_block(x)
        x = self._2_block(x)
        x = self.gap(x).flatten(1)
        return self._6_linear(x)


class MinutiaStem(nn.Module):
    """Minutiae feature stem: 6x Inception_A blocks."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            Inception_A(), Inception_A(), Inception_A(),
            Inception_A(), Inception_A(), Inception_A()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class MinutiaEmbedding(nn.Module):
    """Minutiae embedding projection head."""
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self._0_block = nn.Sequential(
            nn.Conv2d(384, 768, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(768, 896, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(896, 1024, kernel_size=3, stride=1, padding=1),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self._4_linear = nn.Linear(1024, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._0_block(x)
        x = self.gap(x).flatten(1)
        return self._4_linear(x)


class DeepPrint_TexMinu(nn.Module):
    """
    Official DeepPrint Dual-Branch Model (best_model.pyt):
    Stem -> Texture Branch (256-D) + Minutiae Branch (256-D) -> 512-D Embedding.
    """
    def __init__(self, embedding_dims: int = 512):
        super(DeepPrint_TexMinu, self).__init__()
        self.embedding_dims = embedding_dims
        tex_dim = embedding_dims // 2
        minu_dim = embedding_dims - tex_dim

        self.stem = InceptionV4_Stem(in_channels=1)
        self.texture_branch = TextureBranch(emb_dim=tex_dim)
        self.minutia_stem = MinutiaStem()
        self.minutia_embedding = MinutiaEmbedding(emb_dim=minu_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > 1:
            x = x[:, 0:1, :, :]

        feat = self.stem(x)
        tex = self.texture_branch(feat)
        m_stem = self.minutia_stem(feat)
        minu = self.minutia_embedding(m_stem)

        emb = torch.cat([tex, minu], dim=1)
        norm = torch.norm(emb, p=2, dim=1, keepdim=True)
        return emb / torch.clamp(norm, min=1e-12)

    def load_weights(self, model_path: str, device: torch.device) -> bool:
        return _load_checkpoint_helper(self, model_path, device)


class DeepPrint_Tex(nn.Module):
    """DeepPrint Texture-Only Model."""
    def __init__(self, embedding_dims: int = 512):
        super(DeepPrint_Tex, self).__init__()
        self.embedding_dims = embedding_dims
        self.stem = InceptionV4_Stem(in_channels=1)
        self.texture_branch = TextureBranch(emb_dim=embedding_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > 1:
            x = x[:, 0:1, :, :]
        feat = self.stem(x)
        emb = self.texture_branch(feat)
        norm = torch.norm(emb, p=2, dim=1, keepdim=True)
        return emb / torch.clamp(norm, min=1e-12)

    def load_weights(self, model_path: str, device: torch.device) -> bool:
        return _load_checkpoint_helper(self, model_path, device)


class DeepPrint_LocTexMinu(nn.Module):
    """STN Localization + Dual-Branch DeepPrint Model."""
    def __init__(self, embedding_dims: int = 512):
        super(DeepPrint_LocTexMinu, self).__init__()
        self.embedding_dims = embedding_dims
        tex_dim = embedding_dims // 2
        minu_dim = embedding_dims - tex_dim

        self.localization = LocalizationNetwork()
        self.stem = InceptionV4_Stem(in_channels=1)
        self.texture_branch = TextureBranch(emb_dim=tex_dim)
        self.minutia_stem = MinutiaStem()
        self.minutia_embedding = MinutiaEmbedding(emb_dim=minu_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > 1:
            x = x[:, 0:1, :, :]
        x_aligned = self.localization(x)
        feat = self.stem(x_aligned)
        tex = self.texture_branch(feat)
        m_stem = self.minutia_stem(feat)
        minu = self.minutia_embedding(m_stem)

        emb = torch.cat([tex, minu], dim=1)
        norm = torch.norm(emb, p=2, dim=1, keepdim=True)
        return emb / torch.clamp(norm, min=1e-12)

    def load_weights(self, model_path: str, device: torch.device) -> bool:
        return _load_checkpoint_helper(self, model_path, device)


def _load_checkpoint_helper(model: nn.Module, model_path: str, device: torch.device) -> bool:
    """Helper to flexibly load PyTorch state dicts from .pyt, .pth, or .pt files."""
    if not os.path.exists(model_path):
        logger.warning(f"DeepPrint weights file not found at: {model_path}")
        return False

    logger.info(f"Loading DeepPrint weights from: {model_path} on {device}")
    try:
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]

        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        res = model.load_state_dict(cleaned, strict=False)
        logger.info(f"Successfully loaded DeepPrint weights from: {model_path} (missing: {len(res.missing_keys)}, unexpected: {len(res.unexpected_keys)})")
        return True
    except Exception as e:
        logger.error(f"Failed to load DeepPrint weights from {model_path}: {e}")
        return False
