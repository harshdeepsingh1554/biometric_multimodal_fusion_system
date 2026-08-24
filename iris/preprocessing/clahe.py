"""
clahe.py — Preprocessing & Contrast Enhancement for Iris Pipeline
==================================================================
Handles:
  1. Input image validation and loading (filepaths, 2D/3D uint8 arrays).
  2. Segmentation tensor formatting (640x480, ImageNet z-score normalization).
  3. CLAHE polar enhancement (clipLimit=2.0, tileGridSize=(8, 8)).
  4. Deep embedding tensor preparation ([-1, 1] normalization, 3-channel tiling).
"""

import os
import logging
from typing import Tuple, Optional, Union
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)


def apply_clahe(polar_image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Applies CLAHE contrast enhancement matching training configuration."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(polar_image)


class IrisPreprocessor:
    """
    Dedicated preprocessor for iris image validation, normalization, and tensor formatting.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        # ImageNet normalization parameters for MobileNetV2 segmentation backbone
        self.seg_means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.seg_stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # PyTorch transform for IResNet-100 polar input (maps [0, 1] to [-1, 1])
        self.embed_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

    def validate_and_load(self, image_input: Union[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Converts any valid input (filepath, BGR array, Grayscale array) to a 2D uint8 grayscale array.
        Returns: (image_gray, error_message)
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return None, f"Image file not found: {image_input}"
            img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None, f"Failed to decode image from path: {image_input}"
            return img, None

        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                return None, "Empty numpy array provided"

            if len(image_input.shape) == 3:
                if image_input.shape[2] == 3:
                    img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
                elif image_input.shape[2] == 4:
                    img = cv2.cvtColor(image_input, cv2.COLOR_BGRA2GRAY)
                elif image_input.shape[2] == 1:
                    img = image_input.squeeze(axis=-1)
                else:
                    return None, f"Unsupported channel dimension: {image_input.shape}"
            elif len(image_input.shape) == 2:
                img = image_input
            else:
                return None, f"Unsupported array shape: {image_input.shape}"

            if img.dtype != np.uint8:
                if img.max() <= 1.0 and img.min() >= 0.0:
                    img = (img * 255.0).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            return img, None

        else:
            return None, f"Expected filepath or numpy array, got {type(image_input).__name__}"

    def prepare_segmentation_tensor(
        self,
        image_gray: np.ndarray,
        target_w: int = 640,
        target_h: int = 480,
        denoise: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Formats raw grayscale image into an NCHW float32 tensor for UNet++ segmentation.
        Returns: (tensor of shape (1, 3, 480, 640), (orig_h, orig_w))
        """
        orig_h, orig_w = image_gray.shape[:2]

        proc_img = image_gray
        if denoise:
            proc_img = cv2.bilateralFilter(image_gray, d=5, sigmaColor=75, sigmaSpace=10)

        # Resize to network input dimensions (W, H) = (640, 480)
        resized = cv2.resize(proc_img.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Scale [0, 255] -> [0.0, 1.0] and replicate to 3 channels
        norm_img = resized / 255.0
        norm_img = np.expand_dims(norm_img, axis=-1)
        norm_img = np.tile(norm_img, (1, 1, 3))

        # ImageNet z-score normalization
        norm_img = (norm_img - self.seg_means) / self.seg_stds

        # HWC -> CHW -> NCHW
        tensor = norm_img.transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

        return tensor, (orig_h, orig_w)

    def apply_clahe(self, polar_image: np.ndarray) -> np.ndarray:
        """Applies CLAHE contrast enhancement to the 64x512 polar unwrapped image."""
        return self.clahe.apply(polar_image)

    def prepare_embedding_tensor(self, polar_image: np.ndarray, device: torch.device) -> torch.Tensor:
        """Converts 64x512 polar image to a (1, 3, 64, 512) PyTorch tensor on the target device."""
        im_pil = Image.fromarray(polar_image, "L").resize((512, 64), Image.Resampling.BILINEAR)
        tensor = self.embed_transform(im_pil).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
        return tensor
