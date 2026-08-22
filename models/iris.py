import os
import sys
import logging
import torch    
import torch.nn as nn
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(PARENT_DIR)

from pipelines.iris_pipeline import (
    segment_iris,
    normalize_iris,
    estimate_noise_mask,
    encode_iris,
    masked_hamming_distance,
)

logger = logging.getLogger(__name__)


class _SelfContainedIrisManager:
    def __init__(self, radial_res=64, angular_res=512):
        self.radial_res = radial_res
        self.angular_res = angular_res
        self.last_normalized_image = None


    def generate_biometric_template(self,image_path, eye_side="right"):
        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            raise ValueError(f"Failed to read image from path: {image_path}")

        pupil_circle, iris_circle = segment_iris(image_gray)
        if pupil_circle is None or iris_circle is None:
            which = "pupil" if pupil_circle is None else "iris outer boundary"
            raise RuntimeError(
                f"Iris segmentation failed ({which} not detected) for: {image_path}. "
                f"Check image quality or adjust radius ranges in segment_iris()."
            )

        polar_image = normalize_iris(
            image_gray, pupil_circle, iris_circle,
            radial_res=self.radial_res, angular_res=self.angular_res
        )
        self.last_normalized_image = polar_image  # cached for the resnet100 embedding path

        noise_mask = estimate_noise_mask(polar_image)
        code, mask = encode_iris(polar_image, noise_mask)
        return code, mask      
    def compute_masked_distance(self, code_a, mask_a, code_b, mask_b):
        return masked_hamming_distance(code_a, mask_a, code_b, mask_b)


class OpenIrisModel:
    """
    Self-contained classical iris pipeline: Hough-circle segmentation,
    Daugman rubber-sheet normalization, multi-scale Gabor wavelet
    encoding, and masked Hamming distance matching -- all implemented
    directly in this project (see iris_pipeline.py), no external
    package required. The only files this needs on disk are your own
    trained model weights (used elsewhere, e.g. ArcIris ResNet100) --
    segmentation and encoding here need no weights at all, since they're
    classical algorithms, not learned models.
    """
    def __init__(self):
        self.manager = _SelfContainedIrisManager()
        logger.info("Self-contained classical iris pipeline initialized (no external dependencies).")

    def extract_template(self, image_path, eye_side="right"):
        """
        Runs segmentation -> Daugman polar normalization -> Gabor
        wavelet encoding. Returns a (binary_code, noise_mask) pair. The
        noise mask marks pixels that are unreliable (specular reflection,
        likely eyelash/shadow occlusion) so they're excluded from matching.
        """
        return self.manager.generate_biometric_template(image_path, eye_side=eye_side)

    def match_templates(self, code_a, mask_a, code_b, mask_b):
        """
        Computes Masked Fractional Hamming Distance between two iris
        codes -- the classical gold-standard iris matching metric.
        "Masked" means bits marked unreliable in EITHER mask are
        excluded from the distance calculation. Also performs axial
        bit-rolling: tries several rotational shifts to compensate for
        eye rotation between two captures, keeping the best (lowest)
        distance found.
        """
        return self.manager.compute_masked_distance(code_a, mask_a, code_b, mask_b)


class IBasicBlock(nn.Module):
    """
    A single residual block for IResNet -- structurally different from
    a standard torchvision ResNet BasicBlock in a few deliberate ways
    that matter for face/iris recognition specifically:
      - BatchNorm BEFORE the first conv (pre-activation style), not after
      - PReLU instead of ReLU (learns a per-channel negative slope
        rather than hard-clipping negative values to zero -- generally
        gives finer-grained gradients for embedding-style training)
      - An extra bn3 after the second conv, before the residual add
    This exact block design is the same one ArcFace's official
    "IResNet" backbone uses -- reusing it here for iris means the same
    proven architecture family is applied to a different biometric trait.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
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
        self.downsample = downsample  # projects the identity path when shape changes
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
            # Needed whenever stride > 1 or channel count changes,
            # so the identity shortcut's shape matches `out`'s shape
            # before they're added together.
            identity = self.downsample(x)

        out += identity  # the residual connection
        return out

class IResNet(nn.Module):
    """
    Full IResNet backbone -- 4 stages of IBasicBlocks, each downsampling
    spatial resolution by 2x (stride=2), ending in a linear projection
    to a fixed-size embedding.
    """
    def __init__(self, block, layers, dropout=0, num_features=512, groups=1, width_per_group=64):
        super(IResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        # Stem: single 3x3 conv (stride 1) -- notably NOT the aggressive
        # 7x7 stride-2 + maxpool stem that standard ImageNet ResNets use.
        # Recognition backbones like this keep more spatial resolution
        # early on since fine detail (ridges, iris texture) matters more
        # than it does for generic object classification.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)

        # 4 stages, each halving spatial resolution (stride=2) --
        # combined with the stem's stride=1, total downsampling factor
        # across the whole network is 2^4 = 16x.
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)

        # This flattened size (512 * 4 * 32 = 65536) is NOT arbitrary --
        # it's derived from the expected input resolution. The iris
        # pipeline resizes normalized polar images to 64 (height) x 512
        # (width). After 16x downsampling: height 64/16=4, width
        # 512/16=32, giving a final feature map of 512 channels x 4 x 32
        # spatial = 512*4*32 flattened. If you ever change the polar
        # image resize dimensions in iris_extractor.py, this fc layer's
        # input size must be recalculated to match, or loading/forward
        # will fail with a shape mismatch.
        self.fc = nn.Linear(512 * 4 * 32, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 1x1 conv + BN to reshape the identity path whenever this
            # stage changes channel count or spatial resolution.
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
        x = self.features(x)  # final BatchNorm1d, no activation -- raw embedding output
        return x


def iresnet100(pretrained=False, progress=True, **kwargs):
    """
    layers=[3,13,30,3] is the exact stage configuration for the "100"
    variant (100 refers to depth) -- this matches the official ArcFace
    IResNet100 backbone definition, so any IResNet100 checkpoint trained
    elsewhere (e.g. on faces) is architecturally compatible with this
    class, which is presumably why it was reused here as ArcIris's
    backbone for a different biometric trait.
    """
    return IResNet(IBasicBlock, [3, 13, 30, 3], **kwargs)


class IrisResNetModel(nn.Module):
    """
    Wraps iresnet100 as the deep iris embedding model ("ArcIris").
    """
    def __init__(self, embedding_size=512):
        super().__init__()
        self.model = iresnet100(num_features=embedding_size)

    def forward(self, x):
        features = self.model(x)
        # F.normalize with p=2 is functionally identical to the manual
        # norm/clamp pattern used elsewhere in this codebase -- just a
        # more concise way to write the same L2 normalization.
        return nn.functional.normalize(features, p=2, dim=1)

    def load_weights(self, model_path, device):
        if not os.path.exists(model_path):
            _model_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(_model_dir, "..", "weights", "iris", "ResNet100_154000.pt")
            alt_path = os.path.normpath(alt_path)
            if os.path.exists(alt_path):
                model_path = alt_path

        logger.info(f"Loading ArcIris ResNet100 weights from: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)

            # Checkpoints saved from a model wrapped in
            # torch.nn.DataParallel have all keys prefixed with
            # "module." (e.g. "module.layer1.0.conv1.weight") -- strip
            # it so the keys match this un-wrapped model's own naming.
            clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

            result = self.model.load_state_dict(clean_state_dict, strict=True)
            logger.info("Successfully loaded ArcIris ResNet100 weights with strict=True (0 missing, 0 unexpected keys)!")
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            return True
        except Exception as e:
            logger.error(f"Error loading ArcIris ResNet100 weights: {e}")
            return False
