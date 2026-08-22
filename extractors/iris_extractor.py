"""
iris_extractor.py — ANNOTATED
================================
Supports TWO parallel iris representations from the same segmented,
normalized iris image:
  - "gabor"     : classical binary iriscode from OpenIris, converted to
                  a bipolar vector so it's cosine-comparable alongside
                  other traits in a fused biometric pipeline.
  - "resnet100" : deep 512-d embedding from ArcIris (IResNet100).

!!! IMPORTANT ISSUE FLAGGED BELOW in extract_features/extract_both_features:
the exception fallback returns a FIXED, deterministic "dummy" embedding
on any failure. See the detailed warning at that code.
"""

import os
import sys
import logging
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(PARENT_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.iris import OpenIrisModel, IrisResNetModel

logger = logging.getLogger(__name__)


class IrisExtractor:
    """
    Combines OpenIris (segmentation + classical Gabor codes) with
    ArcIris ResNet100 (deep embedding) into one extractor.
    """
    def __init__(self, model_path=None, use_gpu=False, default_backend="resnet100", backend=None):
        if backend is not None:
            default_backend = backend  # `backend` kwarg overrides default_backend if both passed
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.default_backend = default_backend
        logger.info(f"Initializing Iris Extractor (default backend: {default_backend}) on device: {self.device}...")

        # OpenIris handles segmentation + polar normalization + Gabor
        # coding -- needed regardless of which backend you ultimately
        # extract embeddings with, since ResNet100 also runs on the
        # OpenIris-normalized polar image, not the raw eye photo.
        self.open_iris = OpenIrisModel()

        if model_path is None:
            model_path = os.path.join(PARENT_DIR, "weights", "iris", "ResNet100_154000.pt")

        self.resnet_model = IrisResNetModel(embedding_size=512)
        self.resnet_model.load_weights(model_path, self.device)
        self.resnet_model.to(self.device)
        self.resnet_model.eval()

        # Normalize(mean=0.5, std=0.5) maps [0,1] -> [-1,1] for a
        # single-channel grayscale polar image -- standard simple
        # normalization when there's no specific pretrained-model
        # convention to match (unlike ArcFace's face preprocessing,
        # which had to match InsightFace's exact training convention).
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

    def extract_template(self, image_path, eye_side="right"):
        """
        Passthrough to OpenIris: returns (binary_code, noise_mask) for
        classical Hamming-distance-based matching (see compute_distance).
        """
        return self.open_iris.extract_template(image_path, eye_side=eye_side)

    def extract_features(self, image_path_or_array, eye_side="right", backend=None, l2_normalize=True):
        """
        Runs segmentation once, then produces ONE embedding using
        whichever backend is selected ("resnet100" or "gabor").

        Accepts either a file-path string or a pre-loaded numpy array.
        For numpy array inputs, the array must be a grayscale uint8 image
        (H, W) — matching what cv2.imread returns with IMREAD_GRAYSCALE.
        """
        if backend is None:
            backend = self.default_backend

        # Validate input type early so errors are obvious.
        if isinstance(image_path_or_array, str):
            if not os.path.exists(image_path_or_array):
                raise FileNotFoundError(
                    f"Iris image not found: {image_path_or_array}"
                )
            source_is_path = True
        elif isinstance(image_path_or_array, np.ndarray):
            source_is_path = False
        else:
            raise TypeError(
                f"Expected a file-path string or numpy array, got {type(image_path_or_array).__name__}"
            )

        try:
            if source_is_path:
                # Running extract_template here has a side effect: it
                # populates self.open_iris.manager.last_normalized_image,
                # which is read below for the resnet100 path. That's an
                # implicit dependency worth knowing about if you ever
                # refactor this -- the resnet100 branch silently depends
                # on this call having just happened.
                _ = self.open_iris.extract_template(image_path_or_array, eye_side=eye_side)
            else:
                # For array inputs, write to a temp file so OpenIris (which
                # expects a path) can read it. This keeps the downstream
                # classical pipeline unchanged.
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                import cv2 as _cv2
                _cv2.imwrite(tmp_path, image_path_or_array)
                try:
                    _ = self.open_iris.extract_template(tmp_path, eye_side=eye_side)
                finally:
                    os.unlink(tmp_path)
                image_path_or_array = tmp_path  # reuse path string for logging only

            if backend == "resnet100":
                norm_img = self.open_iris.manager.last_normalized_image
                if norm_img is not None:
                    # CLAHE: Contrast Limited Adaptive Histogram
                    # Equalization -- boosts local contrast in the
                    # iris texture (which is often low-contrast,
                    # especially in darker irises) without
                    # over-amplifying noise the way global histogram
                    # equalization would.
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    norm_img = clahe.apply(norm_img)

                    # Resize to (512, 64) -- PIL's resize() takes
                    # (width, height), so this produces a 64-tall,
                    # 512-wide image. This exact size is why
                    # IResNet's fc layer in iris_model.py is sized
                    # for 512*4*32 -- these two numbers must stay in
                    # sync (see the note in iris_model.py).
                    im_polar = Image.fromarray(norm_img, "L").resize((512, 64), Image.Resampling.BILINEAR)
                    im_tensor = self.transform(im_polar).unsqueeze(0)

                    # .repeat(1, 3, 1, 1): duplicates the single
                    # normalized channel into 3 channels, since
                    # IResNet's stem conv expects 3-channel input
                    # (same pattern used for fingerprint's grayscale
                    # -> 3-channel handling).
                    im_tensor = im_tensor.repeat(1, 3, 1, 1).to(self.device)

                    with torch.no_grad():
                        emb_tensor = self.resnet_model(im_tensor)
                        features = emb_tensor.squeeze(0).cpu().numpy().astype(np.float32)

                    if l2_normalize:
                        norm = np.linalg.norm(features)
                        features = (features / max(norm, 1e-12)).astype(np.float32)
                    return features

            elif backend == "gabor":
                code, mask = self.open_iris.extract_template(
                    image_path_or_array if source_is_path else tmp_path,
                    eye_side=eye_side
                )

                # `code` is typically a LIST of binary arrays (one
                # per Gabor wavelet scale) -- flatten and concatenate
                # them all into one long binary vector.
                flat_code = np.concatenate([c.flatten() for c in code]).astype(np.float32)

                # Subsample down to exactly 512 dimensions via
                # uniform strided sampling, so this vector has the
                # same length as the resnet100 embedding and can be
                # fused/compared the same way downstream.
                if len(flat_code) > 512:
                    step = len(flat_code) // 512
                    flat_code = flat_code[::step][:512]

                # Binary bits (0/1) converted to bipolar (-1/+1) so
                # that cosine similarity behaves sensibly.
                bipolar_code = 2.0 * flat_code - 1.0
                norm = np.linalg.norm(bipolar_code)
                return (bipolar_code / max(norm, 1e-12)).astype(np.float32)

        except Exception as e:
            logger.warning(f"Iris extraction warning ({backend}) for {image_path_or_array}: {e}")

        # Raise rather than silently returning a fixed dummy embedding
        # that would falsely match every other failed extraction.
        raise RuntimeError(
            f"Iris feature extraction failed for {image_path_or_array} (backend={backend}). "
            f"See preceding warning log for the underlying cause."
        )

    def extract_both_features(self, image_path_or_array, eye_side="right", l2_normalize=True):
        """
        Same idea as extract_features, but computes BOTH gabor and
        resnet100 embeddings from a single segmentation pass (avoids
        running the expensive OpenIris segmentation twice if you want
        both representations).

        Accepts either a file-path string or a pre-loaded numpy array.
        """
        # Validate and resolve input to a usable file path.
        tmp_path = None
        if isinstance(image_path_or_array, str):
            if not os.path.exists(image_path_or_array):
                raise FileNotFoundError(
                    f"Iris image not found: {image_path_or_array}"
                )
            path_for_template = image_path_or_array
        elif isinstance(image_path_or_array, np.ndarray):
            import tempfile
            import cv2 as _cv2
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            _cv2.imwrite(tmp_path, image_path_or_array)
            path_for_template = tmp_path
        else:
            raise TypeError(
                f"Expected a file-path string or numpy array, got {type(image_path_or_array).__name__}"
            )

        try:
            code, mask = self.open_iris.extract_template(path_for_template, eye_side=eye_side)

            # --- Gabor code (same logic as extract_features above) ---
            flat_code = np.concatenate([c.flatten() for c in code]).astype(np.float32)
            if len(flat_code) > 512:
                step = len(flat_code) // 512
                flat_code = flat_code[::step][:512]
            bipolar_code = 2.0 * flat_code - 1.0
            norm_gab = np.linalg.norm(bipolar_code)
            vec_gabor = (bipolar_code / max(norm_gab, 1e-12)).astype(np.float32)

            # --- ResNet100 deep embedding (reuses the normalized
            # image already produced by the extract_template call
            # above -- no second segmentation run needed) ---
            norm_img = self.open_iris.manager.last_normalized_image
            if norm_img is not None:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                norm_img = clahe.apply(norm_img)
                im_polar = Image.fromarray(norm_img, "L").resize((512, 64), Image.Resampling.BILINEAR)
                im_tensor = self.transform(im_polar).unsqueeze(0).repeat(1, 3, 1, 1).to(self.device)

                with torch.no_grad():
                    emb_tensor = self.resnet_model(im_tensor)
                    features = emb_tensor.squeeze(0).cpu().numpy().astype(np.float32)

                if l2_normalize:
                    norm_res = np.linalg.norm(features)
                    features = (features / max(norm_res, 1e-12)).astype(np.float32)
                vec_resnet = features
            else:
                # If normalization failed to produce an image but
                # the gabor code somehow still succeeded, fall back
                # to duplicating the gabor vector rather than
                # crashing -- still worth logging/monitoring how
                # often this branch triggers, since it means the two
                # returned "different" representations are actually
                # identical, which could mislead a fusion layer
                # expecting them to be independent signals.
                logger.warning(
                    "ResNet100 path: last_normalized_image is None after extract_template. "
                    "Returning gabor vector for both outputs."
                )
                vec_resnet = vec_gabor

            return vec_gabor, vec_resnet

        except Exception as e:
            logger.warning(f"Failed extraction for {image_path_or_array}: {e}")
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # Same reasoning as extract_features() above -- raise rather than
        # silently return a fixed dummy pair that would falsely "match"
        # every other failed extraction.
        raise RuntimeError(
            f"Iris feature extraction (both backends) failed for {image_path_or_array}. "
            f"See preceding warning log for the underlying cause."
        )

    def compute_distance(self, code_a, mask_a, code_b, mask_b):
        """
        The CORRECT way to compare two classical iris codes -- masked
        fractional Hamming distance with rotation compensation, via
        OpenIris. This is what should be used for real gabor-code
        matching, rather than cosine similarity on the subsampled
        bipolar vector from extract_features().
        """
        return self.open_iris.match_templates(code_a, mask_a, code_b, mask_b)