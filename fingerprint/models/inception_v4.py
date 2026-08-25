"""
inception_v4.py — Inception-v4 Backbone for DeepPrint
=====================================================
PyTorch implementation of Inception-v4 matching the exact architecture
used in the fixed-length-fingerprint-extractors (DeepPrint) framework.
Based on Szegedy et al., "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size, stride: int = 1, padding: int = 0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):
    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch0 = self.maxpool(x)
        branch1 = self.conv(x)
        out = torch.cat([branch0, branch1], dim=1)
        return out


class Mixed_4a(nn.Module):
    def __init__(self):
        super(Mixed_4a, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(64, 96, kernel_size=3, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        out = torch.cat([branch0, branch1], dim=1)
        return out


class Mixed_5a(nn.Module):
    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch0 = self.conv(x)
        branch1 = self.maxpool(x)
        out = torch.cat([branch0, branch1], dim=1)
        return out


class Inception_A(nn.Module):
    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        out = torch.cat([branch0, branch1, branch2, branch3], dim=1)
        return out


class Reduction_A(nn.Module):
    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2),
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        out = torch.cat([branch0, branch1, branch2], dim=1)
        return out


class Inception_B(nn.Module):
    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, kernel_size=1, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        out = torch.cat([branch0, branch1, branch2, branch3], dim=1)
        return out


class Reduction_B(nn.Module):
    def __init__(self):
        super(Reduction_B, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2),
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        out = torch.cat([branch0, branch1, branch2], dim=1)
        return out


class Inception_C(nn.Module):
    def __init__(self):
        super(Inception_C, self).__init__()
        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch0 = self.branch0(x)

        branch1_0 = self.branch1_0(x)
        branch1_1a = self.branch1_1a(branch1_0)
        branch1_1b = self.branch1_1b(branch1_0)
        branch1 = torch.cat([branch1_1a, branch1_1b], dim=1)

        branch2_0 = self.branch2_0(x)
        branch2_1 = self.branch2_1(branch2_0)
        branch2_2 = self.branch2_2(branch2_1)
        branch2_3a = self.branch2_3a(branch2_2)
        branch2_3b = self.branch2_3b(branch2_2)
        branch2 = torch.cat([branch2_3a, branch2_3b], dim=1)

        branch3 = self.branch3(x)
        out = torch.cat([branch0, branch1, branch2, branch3], dim=1)
        return out
