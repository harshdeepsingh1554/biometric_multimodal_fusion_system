"""
iresnet.py — IResNet-100 Deep Feature Representation Architecture
=================================================================
Implements the 4-stage IResNet-100 backbone with PReLU activations,
pre-activation batch normalization, and 512-D L2-normalized embedding projection.
"""

import os
import logging
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('IBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in IBasicBlock")

        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class IResNet(nn.Module):
    def __init__(self, block, layers, dropout=0, num_features=512, groups=1, width_per_group=64):
        super(IResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.fc = nn.Linear(512 * 4 * 32, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        return x


def iresnet100(pretrained=False, progress=True, **kwargs):
    return IResNet(IBasicBlock, [3, 13, 30, 3], **kwargs)


class IrisResNetModel(nn.Module):
    """
    Wraps iresnet100 as the deep iris embedding model.
    """
    def __init__(self, embedding_size: int = 512):
        super().__init__()
        self.model = iresnet100(num_features=embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        return nn.functional.normalize(features, p=2, dim=1)

    def load_weights(self, model_path: str, device: torch.device) -> bool:
        if not os.path.exists(model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            alt_path = os.path.join(base_dir, "weights", "iris", "ResNet100_154000.pt")
            if os.path.exists(alt_path):
                model_path = alt_path

        logger.info(f"Loading Iris ResNet100 weights from: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            self.model.load_state_dict(clean_state_dict, strict=True)
            logger.info("Successfully loaded Iris ResNet100 weights (strict=True)!")
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            return True
        except Exception as e:
            logger.error(f"Error loading Iris ResNet100 weights: {e}")
            return False
