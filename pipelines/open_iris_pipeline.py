import os
import sys
import cv2
import numpy as np


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
open_iris_path = os.path.join(project_root, "open-iris", "src")
if open_iris_path not in sys.path:
    sys.path.insert(0, open_iris_path)

try:
    import pydantic.v1 as _pydantic_v1
    sys.modules['pydantic'] = _pydantic_v1
except ImportError:
    pass

import iris
from iris.io.dataclasses import IRImage, IrisTemplate


class BiometricQualityFailure(Exception):
    """Exception raised when the biometric sample quality or overlap area is insufficient."""
    pass

class OpenIrisPipelineManager:
    def __init__(self):
        # Initializes the native UNet++ segmentation network and Gabor kernel space
        self.engine = iris.IRISPipeline()
        self.last_metadata = None

    def generate_biometric_template(self, image_path_or_array, eye_side: str = "right"):
        """
        Reads a raw grayscale image (file path string or numpy array), prepares an IRImage,
        runs the Open-IRIS pipeline, and returns the multi-wavelet binary iris code and noise mask.
        """
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
            img_id = os.path.basename(image_path_or_array)
            if img is None:
                raise ValueError(f"Target frame loading failed at: {image_path_or_array}")
        elif isinstance(image_path_or_array, np.ndarray):
            if len(image_path_or_array.shape) == 3:
                img = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2GRAY)
            else:
                img = image_path_or_array
            img_id = "in_memory_image.png"
        else:
            raise TypeError(f"Expected image path string or numpy array, got {type(image_path_or_array).__name__}")

        # Map eye_side string to "left" or "right" as required by Open-IRIS
        side = "right" if "right" in str(eye_side).lower() or "r" in str(eye_side).lower() else "left"

        # Create IRImage dataclass
        ir_image = IRImage(img_data=img, image_id=img_id, eye_side=side)

        # Run end-to-end segmentation and extraction
        pipeline_output = self.engine(ir_image)
        self.last_metadata = pipeline_output.get("metadata")
        
        # Capture normalized image if present in call trace
        norm_obj = self.engine.call_trace.get("normalization")
        self.last_normalized_image = norm_obj.normalized_image if norm_obj is not None else None

        # Check if pipeline output reported an extraction error
        if pipeline_output.get("error") is not None:
            err_msg = pipeline_output["error"].get("message", "Unknown Open-IRIS error")
            raise ValueError(f"Open-IRIS pipeline execution failed: {err_msg}")

        template = pipeline_output.get("iris_template")
        if template is None:
            raise ValueError("Open-IRIS failed to generate template for sample.")

        # Extract binary codes and valid/invalid masks list
        iris_code = template.iris_codes  # List of np.ndarray, shape (16, 256, 2), dtype bool
        noise_mask = template.mask_codes  # List of np.ndarray, shape (16, 256, 2), dtype bool

        return iris_code, noise_mask

    @staticmethod
    def compute_masked_distance(code_a, mask_a, code_b, mask_b, rotation_shifts: int = 8) -> float:
        """
        Computes the Masked Fractional Hamming Distance between two templates.
        Uses horizontal bit-rolling to handle axial rotation (head tilts).
        Implements a Quality Gate requiring at least 15% overlap area.
        """
        best_hd = 1.0
        best_valid_bits = 0
        
        # Calculate total possible bits in the mask across all wavelets
        total_bits = sum(m.size for m in mask_b)
        quality_gate_threshold = 0.15 * total_bits

        valid_shift_found = False

        # Shift bits horizontally to handle axial head variations
        for shift in range(-rotation_shifts, rotation_shifts + 1):
            total_xor = 0
            total_valid = 0
            
            for c_a, m_a, c_b, m_b in zip(code_a, mask_a, code_b, mask_b):
                # Roll across the horizontal (columns) axis (axis 1)
                shifted_c_a = np.roll(c_a, shift, axis=1)
                shifted_m_a = np.roll(m_a, shift, axis=1)
                
                # In Open-IRIS, mask is True for valid bits, False for noise/eyelashes
                combined_mask = shifted_m_a & m_b
                
                # XOR comparison strictly on overlapping valid pixels
                xor_result = np.bitwise_xor(shifted_c_a, c_b) & combined_mask
                
                total_xor += np.sum(xor_result)
                total_valid += np.sum(combined_mask)

            # Apply quality check: if overlapping area is too small, skip or throw
            if total_valid < 100:
                continue

            valid_shift_found = True
            hd = total_xor / total_valid
            if hd < best_hd:
                best_hd = hd
                best_valid_bits = total_valid

        # If we couldn't find a valid matching window or the best overlap area is under the 15% threshold,
        # throw a Biometric Quality Failure exception.
        if not valid_shift_found or (best_valid_bits < quality_gate_threshold):
            raise BiometricQualityFailure(
                f"Biometric Quality Failure (Uncooperative Probe): Overlapping visible area "
                f"({best_valid_bits} bits) is below the 15% minimum threshold ({int(quality_gate_threshold)} bits)."
            )

        return best_hd
