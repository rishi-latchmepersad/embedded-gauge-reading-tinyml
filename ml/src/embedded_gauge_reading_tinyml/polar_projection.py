"""Polar projection utilities for gauge reading.

Converting a circular gauge face into polar coordinates (angle vs radius)
turns the needle into a vertical line feature. This makes angle detection
much easier for a CNN than working in raw Cartesian space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import cv2
import numpy as np
from PIL import Image

# The gauge sweep in polar space: needle angle maps to horizontal position.
# For our gauge: -30C = 225deg, +50C = 315deg (with 270deg sweep).
# In polar projection, angle varies along the horizontal axis.
DEFAULT_POLAR_SIZE: Final[int] = 224


def polar_project_image(
    image: np.ndarray,
    center_xy: tuple[float, float] | None = None,
    max_radius: float | None = None,
    polar_size: int = DEFAULT_POLAR_SIZE,
) -> np.ndarray:
    """Project a gauge image into polar coordinates.

    Args:
        image: RGB image as uint8 array (H, W, 3).
        center_xy: Dial center in pixels. If None, uses image center.
        max_radius: Maximum radius to sample. If None, uses half the min dimension.
        polar_size: Output square size (produces polar_size x polar_size).

    Returns:
        Polar-projected image as float32 array [0, 1] of shape
        (polar_size, polar_size, 3). The horizontal axis is angle (0 to 360deg)
        and the vertical axis is radius (0 to max_radius).
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    height, width = image.shape[:2]

    if center_xy is None:
        center_xy = (float(width) * 0.5, float(height) * 0.5)
    if max_radius is None:
        max_radius = float(min(height, width)) * 0.5

    # OpenCV warpPolar expects center as a tuple of floats.
    center = (float(center_xy[0]), float(center_xy[1]))
    max_r = float(max_radius)

    # OpenCV warpPolar: angle across horizontal, radius down vertical.
    polar = cv2.warpPolar(
        image,
        (polar_size, polar_size),
        center,
        max_r,
        cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS,
    )

    # Ensure correct shape and normalize.
    if polar.shape[0] != polar_size or polar.shape[1] != polar_size:
        polar = cv2.resize(
            polar, (polar_size, polar_size), interpolation=cv2.INTER_LINEAR
        )

    return polar.astype(np.float32) / 255.0


def polar_project_image_path(
    image_path: str | Path,
    center_xy: tuple[float, float] | None = None,
    max_radius: float | None = None,
    polar_size: int = DEFAULT_POLAR_SIZE,
) -> np.ndarray:
    """Load an image from disk and polar-project it."""
    path = Path(image_path)
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    else:
        # Fallback: try OpenCV.
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return polar_project_image(img, center_xy, max_radius, polar_size)


def needle_mask_from_polar(
    polar_image: np.ndarray,
    needle_angle_deg: float,
    mask_sigma: float = 2.0,
) -> np.ndarray:
    """Create a soft needle mask in polar space for supervision.

    In polar space, the needle is a vertical line at a specific horizontal
    position (corresponding to its angle). This generates a Gaussian mask
    centered on that horizontal column.

    Args:
        polar_image: Polar image of shape (H, W, 3).
        needle_angle_deg: Needle angle in degrees (0-360).
        mask_sigma: Width of the Gaussian mask in pixels.

    Returns:
        Binary mask of shape (H, W, 1) with values in [0, 1].
    """
    height, width = polar_image.shape[:2]
    # Map angle to horizontal position in polar image.
    # warpPolar maps 0-360deg across the full width.
    center_x = (needle_angle_deg / 360.0) * float(width)

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dist_sq = (xx - center_x) ** 2
    mask = np.exp(-dist_sq / (2.0 * mask_sigma**2))
    return mask[..., np.newaxis].astype(np.float32)


def angle_from_polar_prediction(
    predicted_mask: np.ndarray,
) -> float:
    """Extract the needle angle from a predicted polar needle mask.

    The decode is intentionally more robust than a plain argmax:
    1. Collapse the mask vertically into a 1D angular profile.
    2. Remove the background floor with a median subtraction.
    3. Smooth the profile slightly so quantization noise matters less.
    4. Estimate the peak angle with a local quadratic fit and a weighted fallback.

    Args:
        predicted_mask: Predicted mask of shape (H, W, 1) or (H, W).

    Returns:
        Estimated needle angle in degrees [0, 360).
    """
    mask = np.squeeze(predicted_mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape {mask.shape}")

    height, width = mask.shape
    if height <= 0 or width <= 0:
        return 0.0

    # Sum vertically to get a 1D profile across angles.
    profile_raw = np.sum(mask.astype(np.float32), axis=0)
    if float(np.sum(profile_raw)) <= 1e-6:
        return 0.0  # Fallback for empty mask.

    # Remove the background floor and smooth the profile a little.
    profile = np.clip(profile_raw - float(np.median(profile_raw)), 0.0, None)
    padded = np.pad(profile, (1, 1), mode="edge")
    profile = 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]

    peak_index = int(np.argmax(profile))
    window_radius = max(3, min(8, width // 32))
    window_start = max(0, peak_index - window_radius)
    window_end = min(width, peak_index + window_radius + 1)
    window_indices = np.arange(window_start, window_end, dtype=np.float32)
    window_weights = profile[window_start:window_end].astype(np.float32)
    window_weights = np.clip(window_weights, 0.0, None)
    window_weights = np.power(window_weights, 1.5)

    center_col: float | None = None
    if width >= 3:
        if peak_index <= 0:
            x_coords = np.array([0.0, 1.0, 2.0], dtype=np.float32)
            y_coords = profile_raw[:3]
            x_origin = 0.0
        elif peak_index >= width - 1:
            x_coords = np.array([-2.0, -1.0, 0.0], dtype=np.float32)
            y_coords = profile_raw[-3:]
            x_origin = float(width - 1)
        else:
            x_coords = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
            y_coords = profile_raw[peak_index - 1 : peak_index + 2]
            x_origin = float(peak_index)

        coeffs = np.polyfit(x_coords, y_coords, deg=2)
        if abs(float(coeffs[0])) > 1e-8:
            vertex = x_origin - (float(coeffs[1]) / (2.0 * float(coeffs[0])))
            if np.isfinite(vertex):
                center_col = float(vertex)

    if center_col is None:
        if float(np.sum(window_weights)) <= 1e-6:
            center_col = float(peak_index)
        else:
            center_col = float(np.sum(window_indices * window_weights) / np.sum(window_weights))

    # The angular axis is linear in the image, so keep the estimate inside the
    # finite raster instead of letting an extrapolated peak drift outside it.
    center_col = min(max(center_col, 0.0), float(width - 1))

    # Convert column back to angle.
    angle_deg = (center_col / float(width)) * 360.0
    return angle_deg % 360.0


def augment_polar_image(
    polar_image: np.ndarray,
    angle_shift_deg: float = 0.0,
    brightness_delta: float = 0.0,
    contrast_factor: float = 1.0,
    blur_sigma: float = 0.0,
    noise_std: float = 0.0,
) -> np.ndarray:
    """Apply augmentation in polar space.

    A horizontal shift in polar space corresponds to a rotation of the gauge.
    This is the correct way to augment gauge images without corrupting labels.
    """
    img = polar_image.copy()
    height, width = img.shape[:2]

    # Horizontal shift = gauge rotation (label-preserving in polar space).
    if abs(angle_shift_deg) > 1e-6:
        shift_px = int((angle_shift_deg / 360.0) * float(width))
        img = np.roll(img, shift_px, axis=1)

    # Photometric augmentations.
    img = np.clip(img + brightness_delta, 0.0, 1.0)
    img = np.clip((img - 0.5) * contrast_factor + 0.5, 0.0, 1.0)

    if blur_sigma > 0.0:
        ksize = max(3, int(blur_sigma * 2) * 2 + 1)
        img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)

    if noise_std > 0.0:
        img = np.clip(img + np.random.normal(0.0, noise_std, img.shape), 0.0, 1.0)

    return img.astype(np.float32)
