"""
circlenet_segmenter.py — CircleNet (ResNet-18) Deep Iris Segmentation & Geometry
================================================================================
Implements CircleNet based on ResNet-18 with conv + fclayer heads for direct,
robust estimation of pupil and iris circular boundaries:
  - Pupil center & radius (px, py, pr)
  - Iris center & radius (ix, iy, ir)
  - Synthetic / annotated segmentation masks and confidence estimation.
"""

import os
import math
import logging
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image

logger = logging.getLogger(__name__)


class ConvHead(nn.Module):
    """1x1 conv reduction layer for CircleNet."""
    def __init__(self, in_channels: int = 512, out_n: int = 6):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_n, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FCLayer(nn.Module):
    """Multi-channel FC projection head with GELU activation."""
    def __init__(self, in_h: int = 8, in_w: int = 10, out_n: int = 6):
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.out_n = out_n
        self.fc_list = nn.ModuleList([
            nn.Linear(in_h * in_w, 6) for _ in range(out_n)
        ])
        self.act = nn.GELU()
        self.fc2 = nn.Linear(36, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.out_n, self.in_h, self.in_w)
        outs = []
        for i in range(self.out_n):
            outs.append(self.fc_list[i](x[:, i, :, :].reshape(-1, self.in_h * self.in_w)))
        out = torch.cat(outs, dim=1)
        out = self.act(out)
        out = self.fc2(out)
        return out


def build_circlenet_resnet18() -> nn.Module:
    """Builds CircleNet modified ResNet-18 architecture."""
    model = models.resnet18(weights=None)
    model.avgpool = ConvHead(in_channels=512, out_n=6)
    model.fc = FCLayer(in_h=8, in_w=10, out_n=6)
    return model


class IrisCircleNetSegmenter:
    """
    Production CircleNet Iris Segmenter using ResNet-18 checkpoint.
    """
    NET_INPUT_SIZE = (320, 240)  # (width, height)

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = False,
        min_pupil_radius: float = 8.0,
        min_iris_radius: float = 25.0,
    ):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.min_pupil_radius = min_pupil_radius
        self.min_iris_radius = min_iris_radius

        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            primary_path = os.path.join(base_dir, "weights", "iris", "resnet18-1543-0.047488-maskIoU-0.934494.pth")
            alt_path = os.path.join(
                base_dir, "OpenSourceIrisRecognition", "methods", "ArcIris", "Python", "models", "resnet18-1543-0.047488-maskIoU-0.934494.pth"
            )
            model_path = primary_path if os.path.exists(primary_path) else alt_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CircleNet weights file not found at: {model_path}")

        logger.info(f"Loading CircleNet (ResNet-18) segmentation model from: {model_path} (device: {self.device})")
        self.model = build_circlenet_resnet18()
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.transform = Compose([
            ToTensor(),
            Normalize(mean=(0.5,), std=(0.5,))
        ])

    def predict_circles(self, image_input: Union[str, np.ndarray, Image.Image]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Runs inference and returns ((px, py, pr), (ix, iy, ir)) in original pixel coordinates.
        """
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("L")
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3:
                gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_input
            pil_img = Image.fromarray(gray, mode="L")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("L")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        orig_w, orig_h = pil_img.size

        # Resize to NET_INPUT_SIZE (320x240) using cv2.INTER_LINEAR_EXACT
        im_np = np.array(pil_img)
        resized_np = cv2.resize(im_np, self.NET_INPUT_SIZE, interpolation=cv2.INTER_LINEAR_EXACT)
        
        # Tensor shape (1, 3, 240, 320)
        tensor = self.transform(resized_np).unsqueeze(0).repeat(1, 3, 1, 1).to(self.device)

        with torch.no_grad():
            out = self.model(tensor).cpu().numpy()[0]

        diag = math.sqrt(orig_w**2 + orig_h**2)
        pupil_x = float(out[0] * orig_w)
        pupil_y = float(out[1] * orig_h)
        pupil_r = float(out[2] * 0.5 * 0.8 * diag)
        iris_x = float(out[3] * orig_w)
        iris_y = float(out[4] * orig_h)
        iris_r = float(out[5] * 0.5 * diag)

        return (pupil_x, pupil_y, pupil_r), (iris_x, iris_y, iris_r)

    def segment(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Executes CircleNet segmentation and generates structured geometry and mask dictionary.
        """
        if isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                orig_h, orig_w = image_input.shape
            else:
                orig_h, orig_w = image_input.shape[:2]
        elif isinstance(image_input, str):
            img_tmp = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            orig_h, orig_w = img_tmp.shape
        elif isinstance(image_input, Image.Image):
            orig_w, orig_h = image_input.size
        else:
            orig_h, orig_w = 480, 640

        (px, py, pr), (ix, iy, ir) = self.predict_circles(image_input)

        valid = True
        invalid_reason = ""

        if pr < self.min_pupil_radius:
            valid = False
            invalid_reason = f"Pupil radius too small ({pr:.1f} < {self.min_pupil_radius})"
        elif ir < self.min_iris_radius:
            valid = False
            invalid_reason = f"Iris radius too small ({ir:.1f} < {self.min_iris_radius})"
        elif ir <= pr:
            valid = False
            invalid_reason = "Pupil radius exceeds or equals iris radius"

        center_dist = float(math.hypot(px - ix, py - iy))
        if center_dist / max(ir, 1e-6) > 0.5:
            valid = False
            invalid_reason = f"Pupil and iris centers are too far apart (offset={center_dist:.1f}, iris_r={ir:.1f})"

        # Generate binary masks safely
        pupil_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        iris_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        safe_pr = max(1, int(round(max(0.0, pr))))
        safe_ir = max(1, int(round(max(0.0, ir))))
        cv2.circle(pupil_mask, (int(round(px)), int(round(py))), safe_pr, 255, -1)
        cv2.circle(iris_mask, (int(round(ix)), int(round(iy))), safe_ir, 255, -1)

        annular_mask = cv2.subtract(iris_mask, pupil_mask)
        annular_area = max(1.0, np.pi * max(1.0, ir**2 - pr**2))
        visible_ratio = float(np.clip(np.sum(annular_mask > 0) / annular_area, 0.0, 1.0))

        confidence = 0.95 if valid else 0.40

        pupil_ellipse = {
            "center": (px, py),
            "axes": (pr * 2, pr * 2),
            "semi_axes": (pr, pr),
            "angle": 0.0,
            "radius": pr,
            "eccentricity": 0.0,
            "is_ellipse": False,
        }
        iris_ellipse = {
            "center": (ix, iy),
            "axes": (ir * 2, ir * 2),
            "semi_axes": (ir, ir),
            "angle": 0.0,
            "radius": ir,
            "eccentricity": 0.0,
            "is_ellipse": False,
        }

        geometry_dict = {
            "valid": valid,
            "invalid_reason": invalid_reason,
            "pupil_center": (px, py),
            "pupil_radius": pr,
            "pupil_ellipse": pupil_ellipse,
            "iris_center": (ix, iy),
            "iris_radius": ir,
            "iris_ellipse": iris_ellipse,
            "center_distance": center_dist,
            "pupil_eccentricity": 0.0,
            "iris_eccentricity": 0.0,
            "visible_iris_ratio": visible_ratio,
            "pupil_mask": pupil_mask,
            "iris_mask": iris_mask,
            "noise_mask": np.zeros((orig_h, orig_w), dtype=np.uint8),
            "confidence": confidence,
            "segmenter_type": "circlenet",
        }

        return geometry_dict
