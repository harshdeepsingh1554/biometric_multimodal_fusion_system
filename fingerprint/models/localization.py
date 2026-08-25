"""
localization.py — Spatial Transformer Network (STN) for DeepPrint
==================================================================
Implements the Localization Network that estimates 2x3 affine transformation
parameters to rectify fingerprint rotation and translation.
Matches the official fixed-length-fingerprint-extractors STN design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class LocalizationNetwork(nn.Module):
    """
    Spatial Transformer Network (STN) Localization Module.
    Predicts affine transformation matrix theta and applies grid sampling.
    """
    def __init__(self, input_size: tuple = (128, 128)):
        super().__init__()
        self.input_size = input_size
        self.resize = transforms.Resize(size=self.input_size, antialias=True)

        # Convolutional localization feature extractor
        self.localization = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3x2 (or 2x3) affine matrix
        # Output shape after 4 maxpools on 128x128: 128 -> 64 -> 32 -> 16 -> 8 -> (64, 8, 8) = 4096
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(True),
            nn.Linear(128, 6),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, C, H, W) where C=1
        Returns: (B, C, H, W) geometrically aligned / rectified fingerprint tensor
        """
        # Downsample for fast localization regression
        x_down = self.resize(x)
        xs = self.localization(x_down)
        xs = xs.flatten(1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # Generate sampling grid and apply bilinear transformation
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        rectified = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
        return rectified
