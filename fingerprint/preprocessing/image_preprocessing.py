"""
image_preprocessing.py — Fingerprint Preprocessing & Enhancement
================================================================
Implements image loading, aspect-preserving padding to 299x299 (Inception-v4),
contrast enhancement (CLAHE), and quality scoring.
"""

import os
from typing import Union, Tuple, Optional
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image


def pad_and_resize_fingerprint(
    img_gray: np.ndarray,
    target_size: Tuple[int, int] = (299, 299),
    fill: int = 255
) -> np.ndarray:
    """
    Pads a fingerprint image to a square canvas preserving ridge aspect ratio,
    then resizes to target dimensions (e.g. 299x299 for Inception-v4).
    """
    h, w = img_gray.shape[:2]
    if h == w:
        return cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA if (h > target_size[0]) else cv2.INTER_LINEAR)

    max_dim = max(h, w)
    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left

    padded = cv2.copyMakeBorder(img_gray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill)
    return cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA if (max_dim > target_size[0]) else cv2.INTER_LINEAR)


def enhance_fingerprint_clahe(
    img_gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Applies CLAHE local contrast enhancement to clarify ridge valley structures."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_gray)


class FingerprintPreprocessor:
    """
    Production Preprocessor for DeepPrint.
    Converts raw image path/array to (1, 1, 299, 299) PyTorch normalized tensor.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (299, 299),
        apply_clahe: bool = False,
        normalize_mean: float = 0.5,
        normalize_std: float = 0.5,
    ):
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # [0, 255] -> [0.0, 1.0]
            transforms.Normalize(mean=(normalize_mean,), std=(normalize_std,)),  # -> [-1.0, 1.0]
        ])

    def load_and_preprocess(self, image_input: Union[str, np.ndarray, Image.Image]) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Loads and prepares image. Returns (tensor_1x1xHxW, processed_gray_uint8).
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Fingerprint file not found: {image_input}")
            img_gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                raise ValueError(f"Failed to read image at: {image_input}")
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3:
                img_gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = image_input.copy()
        elif isinstance(image_input, Image.Image):
            img_gray = np.array(image_input.convert("L"))
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        padded = pad_and_resize_fingerprint(img_gray, target_size=self.target_size, fill=255)
        if self.apply_clahe:
            padded = enhance_fingerprint_clahe(padded)

        pil_img = Image.fromarray(padded, mode="L")
        tensor = self.transform(pil_img).unsqueeze(0)  # (1, 1, H, W)
        return tensor, padded

    def compute_quality(self, img_gray: np.ndarray) -> float:
        """
        Computes composite fingerprint quality score in [0.0, 1.0]:
        70% ridge coverage + 30% Laplacian sharpness variance.
        """
        if img_gray is None or img_gray.size == 0:
            return 0.0

        foreground = img_gray < 210
        coverage = float(np.sum(foreground) / img_gray.size)

        sharpness = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
        sharpness_norm = min(1.0, sharpness / 1500.0)

        quality = 0.7 * coverage + 0.3 * sharpness_norm
        return float(np.clip(quality, 0.0, 1.0))
