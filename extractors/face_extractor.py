import os
import sys
import cv2
import logging
import numpy as np
import onnxruntime as ort
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0,PARENT_DIR)

from models.face import ArcFaceONNXModel

logger = logging.getLogger(__name__)


CANONICAL_LANDMARKS = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.2041],  # Right mouth corner
], dtype=np.float32)


class FaceExtractor:

    def __init__(self, model_path=None, detector_path=None, use_gpu_detector=False,det_score_threshold=0.3):
        if model_path is None:
            model_path = os.path.join(PARENT_DIR, "weights", "face", "w600k_r50.onnx")
        if detector_path is None:
            detector_path = os.path.join(PARENT_DIR, "weights", "face", "det_10g.onnx")

        self.det_score_threshold = det_score_threshold

        self.model = ArcFaceONNXModel(model_path=model_path)

        if os.path.exists(detector_path):
            providers = ["CPUExecutionProvider"]
            if use_gpu_detector and "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.detector_session = ort.InferenceSession(detector_path, providers=providers)
            logger.info(f"FaceExtractor initialized with SCRFD det_10g. Providers: {providers}")
        else:
            self.detector_session = None
            logger.warning(f"Face detector missing at: {detector_path}. Falling back to center crop.")
                   

    def detect_landmarks(self, image):
        
        if self.detector_session is None:
            return None

        h, w = image.shape[:2]

       
        img_resized = cv2.resize(image, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = (img_rgb.astype(np.float32) - 127.5) / 128.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        try:
            outputs = self.detector_session.run(
                None, {self.detector_session.get_inputs()[0].name: tensor}
            )

            
            strides = [8, 16, 32]
            scores_list, bboxes_list, kps_list = [], [], []

            for idx, stride in enumerate(strides):
                score = outputs[idx]
                bbox = outputs[idx + 3] * stride   # scale distances back to pixels
                kps = outputs[idx + 6] * stride    # scale landmark offsets back to pixels

               
                grid_h, grid_w = 640 // stride, 640 // stride
                anchor_grid = np.stack(
                    np.meshgrid(np.arange(grid_w), np.arange(grid_h)), axis=-1
                ).reshape(-1, 2) * stride
                anchor_grid = np.repeat(anchor_grid, 2, axis=0)

               
                x1 = anchor_grid[:, 0] - bbox[:, 0]
                y1 = anchor_grid[:, 1] - bbox[:, 1]
                x2 = anchor_grid[:, 0] + bbox[:, 2]
                y2 = anchor_grid[:, 1] + bbox[:, 3]

               
                landmarks = np.zeros((len(kps), 5, 2), dtype=np.float32)
                for p in range(5):
                    landmarks[:, p, 0] = anchor_grid[:, 0] + kps[:, p * 2]
                    landmarks[:, p, 1] = anchor_grid[:, 1] + kps[:, p * 2 + 1]

                scores_list.append(score.flatten())
                bboxes_list.append(np.stack([x1, y1, x2, y2], axis=-1))
                kps_list.append(landmarks)

            scores = np.concatenate(scores_list)
            kps = np.concatenate(kps_list, axis=0)

           
            valid_idx = np.where(scores > self.det_score_threshold)[0]
            if len(valid_idx) == 0:
                return None

            best_i = valid_idx[np.argmax(scores[valid_idx])]

            best_kps = kps[best_i] * np.array([w / 640.0, h / 640.0])
            return best_kps

        except Exception as e:
            logger.warning(f"Error in face detection: {e}")
            return None

    # ------------------------------------------------------------------        
    # Alignment
    # ------------------------------------------------------------------
    def align_face(self, image, landmarks=None):
        if landmarks is None:
            landmarks = self.detect_landmarks(image)
        if landmarks is None:
            raise ValueError("No landmarks detected for alignment. detect_landmark not working")
       
        A, B = [], []
        for i in range(5):
            x, y = landmarks[i]
            tx_ref, ty_ref = CANONICAL_LANDMARKS[i]
            A.append([x, -y, 1, 0])
            A.append([y, x, 0, 1])
            B.append(tx_ref)
            B.append(ty_ref)

        c, _, _, _ = np.linalg.lstsq(np.array(A, dtype=np.float32), np.array(B, dtype=np.float32), rcond=None)

        # Reassemble the solved parameters into a standard 2x3 affine
        # matrix for cv2.warpAffine.
        M = np.array([[c[0], -c[1], c[2]], [c[1], c[0], c[3]]], dtype=np.float32)

        aligned_patch = cv2.warpAffine(
            image, M, (112, 112),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        return aligned_patch


    def preprocess(self, aligned_patch):
        
        img_rgb = cv2.cvtColor(aligned_patch, cv2.COLOR_BGR2RGB)   # swapRB
        img_float = img_rgb.astype(np.float32)
        img_normalized = (img_float - 127.5) / 127.5               # -> [-1, 1]
        img_chw = np.transpose(img_normalized, (2, 0, 1))          # HWC -> CHW
        tensor = np.expand_dims(img_chw, axis=0)                   # add batch dim
        return tensor

    # ------------------------------------------------------------------
    # Optional: capture quality gating
    # ------------------------------------------------------------------
    @staticmethod
    def assess_quality(aligned_patch, blur_threshold=60.0):
        gray = cv2.cvtColor(aligned_patch, cv2.COLOR_BGR2GRAY)
       
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return blur_score >= blur_threshold, blur_score

    # ------------------------------------------------------------------
    # Full pipeline entry point
    # ------------------------------------------------------------------
    def extract_features(self, image_path_or_array, landmarks=None, check_quality=False):
        """
        Run the full pipeline: load -> detect -> align -> preprocess ->embed -> L2-normalize.
        Returns np.ndarray
            512-dimensional, L2-normalized face embedding, dtype float32.
        """
        if isinstance(image_path_or_array, str):
            if not os.path.exists(image_path_or_array):
                raise FileNotFoundError(f"Face image file not found: {image_path_or_array}")
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise ValueError(f"Failed to read image at: {image_path_or_array}")
        elif isinstance(image_path_or_array, np.ndarray):
            image = image_path_or_array
        else:
            raise TypeError("Expected image file path string or numpy array.")

        # Step 1 + 2: detect + align
        aligned_patch = self.align_face(image, landmarks=landmarks)

        # Optional quality gate, run on the aligned crop before it's
        # normalized and sent to the CNN
        if check_quality:
            ok, score = self.assess_quality(aligned_patch)
            if not ok:
                raise ValueError(f"Rejected capture: blur score {score:.1f} below threshold.")

        # Step 3: recognizer-specific preprocessing
        tensor = self.preprocess(aligned_patch)

        # Step 4: CNN forward pass
        raw_embedding = self.model.forward(tensor).flatten()

        # Step 5: L2 normalize so cosine similarity == dot product later
        norm = np.linalg.norm(raw_embedding, ord=2)
        normalized_embedding = raw_embedding / max(norm, 1e-12)

        return normalized_embedding.astype(np.float32)


