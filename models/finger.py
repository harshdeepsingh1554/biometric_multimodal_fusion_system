"""
finger_model.py — ANNOTATED
=============================
Defines the fingerprint embedding architecture(s). Only FingerprintResNetModel
is actually used by finger_extractor.py — see note at the bottom about
DeepPrintTexMinuModel.
"""

import os
import logging
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class DeepPrintTexMinuModel(nn.Module):
    """
    Dual-branch fingerprint model: one ResNet18 branch sees Gabor-enhanced
    ridge texture, the other sees minutiae structure. This mirrors the
    real DeepPrint paper's design philosophy (texture + minutiae are
    complementary fingerprint cues) but is currently UNUSED —
    finger_extractor.py only imports FingerprintResNetModel below.
    Keep this if you plan to train/use this dual-branch variant later,
    otherwise it's dead code.
    """
    def __init__(self, embedding_size=512):
        super().__init__()
        # Two independent ResNet18 backbones, no pretrained weights
        # (weights=None) since fingerprints look nothing like ImageNet
        # photos — pretrained ImageNet features wouldn't transfer well
        # here anyway, so training from scratch (or from a fingerprint-
        # specific checkpoint) is the right call.
        tex_resnet = models.resnet18(weights=None)
        self.tex_branch = nn.Sequential(
            tex_resnet.conv1, tex_resnet.bn1, tex_resnet.relu, tex_resnet.maxpool,
            tex_resnet.layer1, tex_resnet.layer2, tex_resnet.layer3, tex_resnet.layer4
        )

        minu_resnet = models.resnet18(weights=None)
        self.minu_branch = nn.Sequential(
            minu_resnet.conv1, minu_resnet.bn1, minu_resnet.relu, minu_resnet.maxpool,
            minu_resnet.layer1, minu_resnet.layer2, minu_resnet.layer3, minu_resnet.layer4
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 + 512, embedding_size)  # concat both branches

    def forward(self, x):
        """
        Input x: [B, 3, H, W] where the 3 "channels" aren't RGB at all —
        they're 3 different fingerprint representation maps stacked
        together:
          x[:, 0] = Gabor-enhanced ridge texture
          x[:, 1] = Skeleton / minutiae structure map
          x[:, 2] = Gaussian-blurred minutiae density map
        """
        # Texture branch expects 3 identical channels (since ResNet's
        # first conv layer is built for 3-channel input) -- repeat the
        # single texture channel 3x rather than wasting 2 channels on
        # nothing.
        x_tex = x[:, 0:1, :, :].repeat(1, 3, 1, 1)

        # Minutiae branch instead gets the two real minutiae-related
        # channels plus a zero-filled third channel, again just to
        # satisfy the 3-channel input requirement.
        x_min = torch.stack([x[:, 1, :, :], x[:, 2, :, :], torch.zeros_like(x[:, 1, :, :])], dim=1)

        f_tex = self.gap(self.tex_branch(x_tex)).flatten(1)   # (B, 512)
        f_min = self.gap(self.minu_branch(x_min)).flatten(1)  # (B, 512)

        concat = torch.cat([f_tex, f_min], dim=1)  # (B, 1024)
        features = self.fc(concat)                 # (B, embedding_size)

        # Standard L2 normalization so cosine similarity == dot product later.
        norm = torch.norm(features, p=2, dim=1, keepdim=True)
        return features / torch.clamp(norm, min=1e-12)


class DeepPrintBackbone(nn.Module):
    """
    ResNet50 split into two stages so the model can use BOTH mid-level
    and high-level features:
      - shallow (after layer2): 512 channels, captures finer ridge detail
      - deep (after layer4): 2048 channels, captures higher-level structure
    This is a common "multi-scale feature" trick — combining both gives
    richer representations than using only the final layer's output.
    """
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)

        self.stem_to_layer2 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2,  # -> 512 channels
        )
        self.deep_layers = nn.Sequential(
            backbone.layer3, backbone.layer4,  # -> 2048 channels
        )

    def forward(self, x):
        shallow = self.stem_to_layer2(x)
        deep = self.deep_layers(shallow)
        return shallow, deep


class FingerprintResNetModel(nn.Module):
    """
    The model actually used in production (finger_extractor.py imports
    this one). Combines DeepPrintBackbone's shallow+deep features into a
    single 512-d embedding via a small projector head.

    Architecture (must match finger_extractor_best.pth exactly for
    load_state_dict(strict=True) to succeed):
      backbone -> GAP(shallow) + GAP(deep) -> concat(512+2048=2560)
               -> BN -> Linear(2560->1024) -> BN -> Linear(1024->512)
    """
    def __init__(self, embedding_size=512):
        super(FingerprintResNetModel, self).__init__()
        self.backbone = DeepPrintBackbone()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        in_features = 512 + 2048  # shallow_pool + deep_pool concatenated

        # The nn.Identity() entries at indices 1, 3, 5 are placeholders,
        # not functional layers -- they exist purely so this Sequential's
        # layer indices (projector.0, projector.1, ... projector.6) line
        # up with the key names saved in the checkpoint file. This is a
        # common pattern when the original training script had extra
        # layers (e.g. Dropout, activation) at those positions that were
        # later removed for inference, but the checkpoint's state_dict
        # keys still expect something at those indices for strict=True
        # loading to succeed without renaming.
        self.projector = nn.Sequential(
            nn.BatchNorm1d(in_features),      # projector.0 -- real
            nn.Identity(),                    # projector.1 -- placeholder
            nn.Linear(in_features, 1024),      # projector.2 -- real
            nn.Identity(),                    # projector.3 -- placeholder
            nn.BatchNorm1d(1024),              # projector.4 -- real
            nn.Identity(),                    # projector.5 -- placeholder
            nn.Linear(1024, embedding_size),   # projector.6 -- real
        )

    def forward(self, x):
        shallow, deep = self.backbone(x)
        shallow_pool = self.gap(shallow).flatten(1)  # (B, 512)
        deep_pool = self.gap(deep).flatten(1)         # (B, 2048)
        concat = torch.cat([shallow_pool, deep_pool], dim=1)  # (B, 2560)
        features = self.projector(concat)  # (B, 512)

        norm = torch.norm(features, p=2, dim=1, keepdim=True)
        return features / torch.clamp(norm, min=1e-12)

    def load_weights(self, model_path, device):
        """
        Loads a trained checkpoint into this architecture. strict=True
        means every single layer name in the checkpoint must match a
        layer name in this model exactly (and vice versa) -- this is
        intentionally strict so a silent architecture mismatch (e.g.
        wrong projector shape) raises an error instead of loading
        garbage weights into some layers.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Fingerprint model weights not found at: {model_path}. "
                "Ensure the .pth file exists at the configured path."
            )

        logger.info(f"Loading DeepPrint Fingerprint PyTorch weights from: {model_path}")
        try:
            # weights_only=True is a security best-practice for torch.load:
            # restricts unpickling to tensors/basic types only, protecting
            # against arbitrary code execution from a malicious .pth file.
            state_dict = torch.load(model_path, map_location=device, weights_only=True)

            # Some training scripts save extra metadata (epoch, optimizer
            # state, etc.) alongside the weights under a
            # "model_state_dict" key -- unwrap it if present.
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            result = self.load_state_dict(state_dict, strict=True)
            logger.info("Successfully loaded DeepPrint fingerprint weights with strict=True (0 missing, 0 unexpected keys)!")
            return True
        except Exception as e:
            logger.error(f"Error loading DeepPrint fingerprint weights: {e}")
            return False
