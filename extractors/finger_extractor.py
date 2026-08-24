"""
finger_extractor.py — ANNOTATED
==================================
Full fingerprint pipeline: load grayscale image -> aspect-ratio-preserving
pad + resize -> tensor -> DeepPrint CNN -> L2-normalized 512-d embedding.
Also provides a standalone quality score (coverage + sharpness heuristic).
"""

import os
import sys
import cv2
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from models.finger import FingerprintResNetModel

logger = logging.getLogger(__name__)


def pad_and_resize_deepprint(img_gray, target_size=(224, 224), fill=255):
    """
    Pads a rectangular fingerprint scan to a square canvas (white
    background, since fingerprint ridges are typically dark-on-white)
    BEFORE resizing to the model's 224x224 input.

    Why this matters: fingerprint sensors often produce non-square
    captures (e.g. 300x400). A naive resize straight to 224x224 would
    stretch ridge spacing/orientation differently in each axis, which
    directly corrupts the ridge frequency/orientation cues the model
    relies on. Padding to square FIRST preserves the true aspect ratio
    of the ridge pattern; only then is it safe to resize uniformly.
    """
    h, w = img_gray.shape[:2]
    if h == w:
        return cv2.resize(img_gray, target_size)

    max_dim = max(h, w)
    # Center the original image within a max_dim x max_dim square,
    # padding the shorter dimension symmetrically.
    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left
    padded = cv2.copyMakeBorder(img_gray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill)
    return cv2.resize(padded, target_size)


class FingerprintExtractor:
    """
    DeepPrint-based fingerprint feature extractor.
    """
    def __init__(self, model_path=None, use_gpu=False):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        if model_path is None:
            model_path = os.path.join(PARENT_DIR, "weights", "finger", "finger_extractor_best.pth")

        logger.info(f"Initializing DeepPrint Extractor model on device: {self.device}")

        # Build the architecture, then load trained weights into it --
        # two separate steps, unlike ONNX where they're bundled together.
        self.model = FingerprintResNetModel()
        self.model.load_weights(model_path, self.device)
        self.model.to(self.device)
        self.model.eval()  # disables dropout/batchnorm-update behavior for inference

        # Freeze all parameters -- this is an inference-only extractor,
        # never trained further here, so no gradients are ever needed.
        # Saves memory and prevents accidental fine-tuning.
        for param in self.model.parameters():
            param.requires_grad = False

        # ToTensor(): converts a PIL image (H,W) uint8 [0,255] into a
        # torch tensor (1,H,W) float32 [0,1] -- this is the ONLY
        # normalization applied; there's no further mean/std subtraction
        # like the typical ImageNet (0.485, 0.456, 0.406) normalization
        # you'd see on a pretrained-ImageNet ResNet. That's expected here
        # since this ResNet50 backbone was trained from scratch on
        # fingerprints (weights=None in finger_model.py), not fine-tuned
        # from ImageNet weights -- so it doesn't need ImageNet's specific
        # input statistics, just whatever normalization convention this
        # checkpoint was actually trained with. If you didn't train this
        # checkpoint yourself, confirm with whoever did that plain [0,1]
        # scaling (no further mean/std normalization) matches their
        # training preprocessing -- a mismatch here would silently hurt
        # accuracy without throwing any errors.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # The model's first conv layer expects 3 input channels
            # (inherited from ResNet50's standard stem), but fingerprint
            # scans are grayscale (1 channel) -- so triplicate the single
            # channel into 3 identical channels rather than treating it
            # as RGB.
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
        ])

    def preprocess_image(self, image_path_or_array):
        """
        Loads (if needed) and preprocesses a fingerprint image into the
        exact tensor shape/format the model expects: (1, 3, 224, 224).
        """
        if isinstance(image_path_or_array, str):
            if not os.path.exists(image_path_or_array):
                raise FileNotFoundError(f"Fingerprint image file not found: {image_path_or_array}")
            # IMREAD_GRAYSCALE loads directly as single-channel --
            # correct, since fingerprint sensors are inherently grayscale.
            img_gray = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                raise ValueError(f"Failed to read fingerprint image at: {image_path_or_array}")
        elif isinstance(image_path_or_array, np.ndarray):
            # Handle the case where an already-loaded array might still
            # be 3-channel (e.g. accidentally read as color) -- convert
            # down to grayscale rather than assuming the caller got it right.
            if len(image_path_or_array.shape) == 3:
                img_gray = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = image_path_or_array
        else:
            raise TypeError("Expected image path string or numpy array.")

        padded_224 = pad_and_resize_deepprint(img_gray, target_size=(224, 224), fill=255)
        img_pil = Image.fromarray(padded_224).convert("L")  # "L" = 8-bit grayscale mode

        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)  # add batch dim
        return tensor

    def extract_features(self, image_path_or_array, l2_normalize=True):
        """
        Runs the full pipeline and returns a 512-d embedding.
        """
        tensor = self.preprocess_image(image_path_or_array)

        # torch.no_grad(): disables gradient tracking during the forward
        # pass -- not strictly required since params already have
        # requires_grad=False, but it also reduces memory overhead for
        # intermediate activations, which matters more for the larger
        # ResNet50 backbone here than it did for ArcFace's smaller net.
        with torch.no_grad():
            features = self.model(tensor).squeeze(0).cpu().numpy()

        # Note: FingerprintResNetModel.forward() ALREADY L2-normalizes
        # internally (see finger_model.py). Normalizing again here is
        # mathematically a no-op on an already-unit-length vector (aside
        # from floating point rounding) -- redundant but harmless, kept
        # here as a defensive safety net in case the model's internal
        # normalization is ever changed or this extractor is pointed at
        # a different model class that doesn't normalize internally.
        if l2_normalize:
            norm = np.linalg.norm(features)
            if norm > 1e-12:
                features = features / norm

        return features.astype(np.float32)

    def compute_quality_score(self, image_path_or_array):
        """
        A lightweight, heuristic fingerprint quality score in [0,1] --
        NOT a replacement for a real standard like NFIQ 2.0, but useful
        as a fast pre-filter to reject obviously bad captures (e.g. a
        finger barely touching the sensor, or heavy motion blur) before
        spending a CNN forward pass on them.

        Two components, weighted:
          - coverage (70%): what fraction of the image is actual ridge
            content (dark pixels) vs. blank background. A wide/mostly
            white image usually means poor sensor contact.
          - sharpness (30%): Laplacian variance, same blur metric used
            for face quality gating -- low variance means the ridge
            edges are smeared/out of focus.
        """
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
        else:
            img = image_path_or_array

        if img is None:
            return 0.0

        # Pixels below 210 are treated as "ridge" (dark) rather than
        # background (near-white) -- threshold tuned for typical
        # white-background optical/capacitive fingerprint scans.
        foreground = img < 210
        coverage = float(np.sum(foreground) / img.size)

        sharpness = float(cv2.Laplacian(img, cv2.CV_64F).var())
        # Cap at 1500 (empirically chosen ceiling) so a very sharp image
        # doesn't blow the normalized score past 1.0.
        sharpness_norm = min(1.0, sharpness / 1500.0)

        quality = 0.7 * coverage + 0.3 * sharpness_norm
        return float(np.clip(quality, 0.0, 1.0))