"""
daugman_normalizer.py — Daugman Rubber-Sheet Polar Normalization
================================================================
Unwraps the annular iris region between the pupil and limbic boundaries
into a normalized rectangular polar representation of fixed dimension (64 × 512).
Supports both elliptical and circular boundary parameters via vectorized cv2.remap.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2


class DaugmanNormalizer:
    """
    Implements deterministic Daugman rubber-sheet normalization.
    Target resolution: 64 (radial) × 512 (angular).
    """

    def __init__(self, radial_res: int = 64, angular_res: int = 512):
        self.radial_res = radial_res
        self.angular_res = angular_res

    def _get_boundary_coords(self, geom: Dict[str, Any], thetas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates boundary (x, y) coordinates for an array of angles thetas."""
        if geom.get("is_ellipse", False):
            cx, cy = geom["center"]
            semi_a, semi_b = geom["semi_axes"]
            phi = np.deg2rad(geom["angle"])

            cos_t = np.cos(thetas)
            sin_t = np.sin(thetas)

            x_rot = semi_a * cos_t * np.cos(phi) - semi_b * sin_t * np.sin(phi)
            y_rot = semi_a * cos_t * np.sin(phi) + semi_b * sin_t * np.cos(phi)

            x_bound = cx + x_rot
            y_bound = cy + y_rot
        else:
            cx, cy = geom["center"]
            r = geom["radius"]
            x_bound = cx + r * np.cos(thetas)
            y_bound = cy + r * np.sin(thetas)

        return x_bound.astype(np.float32), y_bound.astype(np.float32)

    def normalize(
        self,
        image_gray: np.ndarray,
        geometry_meta: Dict[str, Any],
        noise_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Unwraps the annular iris into a 64 × 512 polar texture image.

        Parameters
        ----------
        image_gray : np.ndarray
            Original grayscale NIR image (H, W) uint8.
        geometry_meta : Dict[str, Any]
            Pupil and iris boundary metadata.
        noise_mask : np.ndarray, optional
            Binary noise / eyelash mask.

        Returns
        -------
        polar_image : np.ndarray
            (64, 512) uint8 normalized iris image.
        polar_noise_mask : np.ndarray or None
            (64, 512) bool normalized noise mask.
        """
        pupil_geom = geometry_meta.get("pupil_ellipse") or {
            "center": geometry_meta["pupil_center"],
            "radius": geometry_meta["pupil_radius"],
            "is_ellipse": False,
        }
        iris_geom = geometry_meta.get("iris_ellipse") or {
            "center": geometry_meta["iris_center"],
            "radius": geometry_meta["iris_radius"],
            "is_ellipse": False,
        }

        r_fracs = np.linspace(0.0, 1.0, self.radial_res, dtype=np.float32).reshape(-1, 1)
        thetas = np.linspace(0.0, 2.0 * np.pi, self.angular_res, endpoint=False, dtype=np.float32).reshape(1, -1)

        xp, yp = self._get_boundary_coords(pupil_geom, thetas)
        xi, yi = self._get_boundary_coords(iris_geom, thetas)

        map_x = ((1.0 - r_fracs) * xp + r_fracs * xi).astype(np.float32)
        map_y = ((1.0 - r_fracs) * yp + r_fracs * yi).astype(np.float32)

        polar_image = cv2.remap(
            image_gray.astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.uint8)

        polar_noise = None
        if noise_mask is not None:
            remap_mask = cv2.remap(
                noise_mask.astype(np.float32),
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            polar_noise = (remap_mask > 127)

        return polar_image, polar_noise
