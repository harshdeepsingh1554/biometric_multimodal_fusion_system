"""
iris_pipeline.py — Self-contained classical Daugman iris pipeline
=====================================================================
Replaces the external `src.pipelines.open_iris_pipeline.OpenIrisPipelineManager`
dependency with a fully self-contained implementation. No external iris
library required, no additional trained segmentation weights required --
everything here is classical computer vision (OpenCV + NumPy), which is
exactly how iris recognition worked for ~20 years before deep-learning
segmentation networks like UNet++ became common.

Pipeline stages:
  1. segment_iris()        -> find pupil + iris circle boundaries (Hough transform)
  2. normalize_iris()      -> Daugman rubber-sheet unwrap to a fixed polar image
  3. estimate_noise_mask() -> flag unreliable pixels (reflections, likely eyelash)
  4. encode_iris()         -> multi-scale Gabor phase-quantization -> binary iriscode
  5. masked_hamming_distance() -> compare two iriscodes with rotation compensation

Known limitation vs. a trained UNet++ segmentation model: Hough-circle
segmentation assumes the pupil/iris boundaries are roughly circular and
can struggle on heavily occluded or off-axis captures. If you have (or
later train) a segmentation checkpoint, swap segment_iris()'s internals
for a model forward pass -- everything downstream (normalize/encode/match)
stays the same since it only needs the two boundary circles as input.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 1. Segmentation
# ----------------------------------------------------------------------
def segment_iris(image_gray, pupil_radius_range=None, iris_radius_range=None):
    """
    Locate the pupil and iris boundary circles using the Hough Circle
    Transform -- the same classical technique Daugman's original system
    and most pre-deep-learning iris systems used.

    Radius ranges default to image-size-relative values so the detector
    works across different sensor resolutions without manual tuning.

    Parameters
    ----------
    image_gray : np.ndarray
        Grayscale NIR eye image.
    pupil_radius_range, iris_radius_range : tuple(int, int) or None
        Expected pixel radius ranges for each boundary. When None,
        sensible defaults are derived from the image dimensions.

    Returns
    -------
    (pupil_circle, iris_circle) : each a tuple (x, y, r), or (None, None)
        if detection failed.
    """
    h, w = image_gray.shape[:2]
    short = min(h, w)

    # Derive adaptive defaults from image size so the detector works
    # across different sensor resolutions (e.g. 640px vs 1024px images)
    # without requiring manual radius tuning per dataset.
    if pupil_radius_range is None:
        pupil_radius_range = (max(8, short // 16), max(30, short // 6))
    if iris_radius_range is None:
        iris_radius_range = (max(30, short // 8), max(120, short // 2))

    # minDist: minimum distance between detected circle centers.
    # Using the full image height (old value) was far too strict --
    # it often prevented finding the iris circle at all when the pupil
    # center is close to the iris center. A fraction of the short side
    # is large enough to avoid duplicate detections while still finding
    # both boundaries.
    min_dist = max(10, short // 5)

    # Median blur suppresses eyelash/sensor noise while preserving the
    # sharp pupil/iris edges Hough transform relies on.
    blurred = cv2.medianBlur(image_gray, 5)

    # --- Pupil: darkest, most well-defined circular region ---
    # We try progressively lower param2 (accumulator threshold) values
    # if no circles are found at the stricter level, so images with
    # lower contrast pupil boundaries still produce a result.
    pupil_circles = None
    for param2 in (30, 20, 15):
        pupil_circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
            param1=50, param2=param2,
            minRadius=pupil_radius_range[0], maxRadius=pupil_radius_range[1],
        )
        if pupil_circles is not None:
            break

    if pupil_circles is None:
        logger.warning(
            f"Pupil boundary not detected (image size {w}x{h}, "
            f"radius range {pupil_radius_range})."
        )
        return None, None

    # Among candidates, pick the one whose interior is darkest on
    # average -- the pupil is the darkest region of the eye by a wide
    # margin, which is a more reliable disambiguator than accumulator
    # score alone.
    px, py, pr = _pick_darkest_circle(image_gray, pupil_circles[0])

    # --- Iris outer boundary: concentric-ish, larger radius ---
    iris_min_r = max(iris_radius_range[0], int(pr * 1.3))
    iris_max_r = iris_radius_range[1]

    iris_circles = None
    for param2 in (25, 18, 12):
        iris_circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
            param1=50, param2=param2,
            minRadius=iris_min_r, maxRadius=iris_max_r,
        )
        if iris_circles is not None:
            break

    if iris_circles is None:
        logger.warning(
            f"Iris outer boundary not detected (image size {w}x{h}, "
            f"radius range ({iris_min_r}, {iris_max_r}))."
        )
        return (px, py, pr), None

    # Prefer the iris candidate whose center is closest to the pupil's
    # center -- the two boundaries are approximately concentric for a
    # forward-facing eye capture.
    ix, iy, ir = _pick_closest_circle(iris_circles[0], center=(px, py))

    # --- Concentricity sanity check ---
    # "Closest of the candidates Hough found" is not the same as "close
    # enough to be the real iris boundary". A genuine iris/pupil pair is
    # near-concentric; a center offset that's a large fraction of the
    # iris radius almost always means Hough locked onto an eyelid crease,
    # eyebrow shadow, or reflection edge instead of the true limbus.
    # Rather than silently returning a bad circle (which then corrupts
    # the polar unwrap and every downstream embedding), reject it and
    # report failure explicitly.
    center_offset = float(np.hypot(ix - px, iy - py))
    max_allowed_offset = 0.35 * ir  # tune per dataset if needed
    if center_offset > max_allowed_offset:
        logger.warning(
            f"Iris boundary rejected: center offset {center_offset:.1f}px exceeds "
            f"{max_allowed_offset:.1f}px allowed for radius {ir}px (image size {w}x{h}). "
            f"Likely a false detection (eyelid/eyebrow/reflection) rather than the true iris boundary."
        )
        return (px, py, pr), None

    return (px, py, pr), (ix, iy, ir)


def _pick_darkest_circle(image_gray, circles):
    best, best_mean = None, 256.0
    for (x, y, r) in circles:
        mask = np.zeros(image_gray.shape, dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        mean_val = cv2.mean(image_gray, mask=mask)[0]
        if mean_val < best_mean:
            best_mean, best = mean_val, (x, y, r)
    return best


def _pick_closest_circle(circles, center):
    cx, cy = center
    best, best_dist = None, float("inf")
    for (x, y, r) in circles:
        d = (x - cx) ** 2 + (y - cy) ** 2
        if d < best_dist:
            best_dist, best = d, (x, y, r)
    return best


# ----------------------------------------------------------------------
# 2. Daugman rubber-sheet normalization
# ----------------------------------------------------------------------
def normalize_iris(image_gray, pupil_circle, iris_circle, radial_res=64, angular_res=512):
    """
    Unwraps the annular iris region into a fixed-size rectangular polar
    image -- Daugman's "rubber sheet model". Every angle theta gets a
    radial scan line from the pupil boundary out to the iris boundary,
    so pupil dilation/constriction (which changes the iris ring's raw
    width) doesn't distort the resulting fixed-size representation --
    it's normalized away by construction.

    Implementation: fully vectorized using np.meshgrid + cv2.remap
    (replaces the previous 32,768-iteration Python double-loop).
    cv2.remap applies bilinear interpolation in a single C-level call,
    making this significantly faster while producing identical output.

    Returns
    -------
    polar_image : np.ndarray, shape (radial_res, angular_res), uint8
    """
    px, py, pr = pupil_circle
    ix, iy, ir = iris_circle

    # Build 2-D grids of (radial_fraction, angle) coordinates in one shot.
    # r_fracs: shape (radial_res, 1)   -- fraction along the pupil->iris radius
    # thetas:  shape (1, angular_res)  -- angle in [0, 2π)
    r_fracs = np.linspace(0, 1, radial_res, dtype=np.float32).reshape(-1, 1)
    thetas = np.linspace(0, 2 * np.pi, angular_res, endpoint=False, dtype=np.float32).reshape(1, -1)

    cos_t = np.cos(thetas)  # (1, angular_res)
    sin_t = np.sin(thetas)  # (1, angular_res)

    # Pupil-boundary and iris-boundary points for every angle, broadcast
    # so shapes become (1, angular_res) for the boundary coords.
    x_p = px + pr * cos_t
    y_p = py + pr * sin_t
    x_i = ix + ir * cos_t
    y_i = iy + ir * sin_t

    # Linear interpolation: (radial_res, angular_res) sample coordinates.
    # Broadcasting: r_fracs (radial_res, 1) × (1, angular_res) → (radial_res, angular_res)
    map_x = (x_p + r_fracs * (x_i - x_p)).astype(np.float32)
    map_y = (y_p + r_fracs * (y_i - y_p)).astype(np.float32)

    # cv2.remap applies the bilinear sampling at every (map_x, map_y) coordinate
    # in a single vectorized C call -- same result as the _bilinear_sample loop.
    polar_image = cv2.remap(
        image_gray.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(np.uint8)

    return polar_image


def _bilinear_sample(image, x, y):
    """Bilinear-interpolated pixel value at fractional coordinate (x, y)."""
    h, w = image.shape[:2]
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return 0
    dx, dy = x - x0, y - y0
    top = image[y0, x0] * (1 - dx) + image[y0, x1] * dx
    bottom = image[y1, x0] * (1 - dx) + image[y1, x1] * dx
    return int(top * (1 - dy) + bottom * dy)


# ----------------------------------------------------------------------
# 3. Noise mask estimation (reflections / likely eyelash occlusion)
# ----------------------------------------------------------------------
def estimate_noise_mask(polar_image, reflection_thresh=230, eyelash_thresh=25):
    """
    Flags unreliable pixels in the normalized polar image so they can be
    excluded from matching. This is a coarse intensity-based heuristic --
    NOT as accurate as a trained segmentation network that explicitly
    learns eyelash/eyelid shapes, but catches the two most common and
    highest-impact sources of noise cheaply:
      - specular reflections from the NIR illuminator (very bright pixels)
      - deep shadow regions typical of eyelash occlusion (very dark pixels,
        excluding the pupil boundary itself which normalize_iris already
        excludes by construction)

    Returns
    -------
    noise_mask : np.ndarray, same shape as polar_image, dtype=bool
        True = unreliable / excluded from matching.
    """
    reflections = polar_image >= reflection_thresh
    shadows = polar_image <= eyelash_thresh
    return reflections | shadows


# ----------------------------------------------------------------------
# 4. Multi-scale Gabor wavelet encoding (Daugman phase quantization)
# ----------------------------------------------------------------------
def encode_iris(polar_image, noise_mask, wavelengths=(8, 16, 24), orientation=0.0):
    """
    Generates a binary iris code using 2D Gabor wavelet phase
    quantization -- the same core idea as Daugman's original encoding:
    at each pixel, the complex response of a Gabor filter is computed,
    and its phase is quantized into one of 4 quadrants using 2 bits
    (sign of the real part, sign of the imaginary part). Doing this at
    multiple wavelengths (scales) captures both fine and coarse texture
    patterns in the iris.

    Returns
    -------
    (codes, masks) : each a list of np.ndarray, one per wavelength scale.
        codes[i] has shape (radial_res, angular_res, 2) -- 2 bits/pixel.
        masks[i] has shape (radial_res, angular_res, 2) -- noise_mask
        broadcast to match, both bits at a noisy pixel marked unreliable.
    """
    img_f = polar_image.astype(np.float32)
    codes, masks = [], []

    for wavelength in wavelengths:
        # sigma (Gaussian envelope width) scaled relative to wavelength
        # is a standard Gabor-filter convention -- keeps the number of
        # oscillations under the envelope roughly constant across scales.
        sigma = wavelength * 0.5
        ksize = int(6 * sigma) | 1  # ensure odd kernel size

        kernel_real = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta=orientation, lambd=wavelength,
            gamma=1.0, psi=0, ktype=cv2.CV_32F
        )
        # psi=pi/2 gives the quadrature (90 deg phase-shifted) component,
        # forming a complex Gabor pair -- real + imaginary parts, needed
        # to determine which phase quadrant each pixel falls into.
        kernel_imag = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta=orientation, lambd=wavelength,
            gamma=1.0, psi=np.pi / 2, ktype=cv2.CV_32F
        )

        real = cv2.filter2D(img_f, cv2.CV_32F, kernel_real)
        imag = cv2.filter2D(img_f, cv2.CV_32F, kernel_imag)

        # 2 bits per pixel: sign of real part, sign of imaginary part.
        # This directly encodes which of the 4 phase quadrants the
        # complex Gabor response falls in -- Daugman's original scheme.
        bit0 = (real >= 0).astype(np.uint8)
        bit1 = (imag >= 0).astype(np.uint8)
        code = np.stack([bit0, bit1], axis=-1)

        mask = np.stack([noise_mask, noise_mask], axis=-1)

        codes.append(code)
        masks.append(mask)

    return codes, masks


# ----------------------------------------------------------------------
# 5. Masked fractional Hamming distance with rotation compensation
# ----------------------------------------------------------------------
def masked_hamming_distance(codes_a, masks_a, codes_b, masks_b, max_shift=8):
    """
    Standard iris comparison metric: fraction of bits that disagree,
    counting only bits both codes consider reliable (unmasked). Tries
    several angular "bit-roll" shifts to compensate for eye rotation
    between the two captures (e.g. head tilt), keeping the best (lowest)
    distance found across all shifts -- exactly what real iris matchers
    do, since two genuine captures of the same iris are rarely at the
    exact same rotational alignment.

    Returns
    -------
    float in [0, 1] -- lower means more similar. 0 = identical, ~0.5 =
    what two DIFFERENT irises typically score (since random bits agree
    about half the time by chance).
    """
    best_dist = 1.0

    for shift in range(-max_shift, max_shift + 1):
        total_bits, disagreeing_bits = 0, 0

        for code_a, mask_a, code_b, mask_b in zip(codes_a, masks_a, codes_b, masks_b):
            # Roll along the angular axis (axis=1) to simulate rotating
            # one iris code relative to the other.
            shifted_code_b = np.roll(code_b, shift, axis=1)
            shifted_mask_b = np.roll(mask_b, shift, axis=1)

            valid = (~mask_a) & (~shifted_mask_b)
            if not np.any(valid):
                continue

            disagreements = (code_a != shifted_code_b) & valid
            total_bits += int(valid.sum())
            disagreeing_bits += int(disagreements.sum())

        if total_bits > 0:
            dist = disagreeing_bits / total_bits
            best_dist = min(best_dist, dist)

    return best_dist