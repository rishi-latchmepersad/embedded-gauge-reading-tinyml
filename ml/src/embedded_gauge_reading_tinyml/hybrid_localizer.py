"""
Hybrid localizer: classical center hypotheses + simplified polar spoke vote.

Provides the Python-side implementations of the firmware's classical gauge
reading pipeline. Used for training the CNN center selector and evaluating
the end-to-end hybrid pipeline on board captures.

Center hypotheses (matching app_baseline_runtime.c):
  1. bright_centroid  — luma bright-centroid detection
  2. crop_center      — fixed training crop center (inner dial pivot)
  3. rim_center       — coarse-to-fine Hough rim search
  4. image_center     — geometric center of the crop

Polar spoke vote (simplified from the firmware):
  - Sobel gradient → tangential alignment → angular bin accumulation
  - Smooth → peak finding → sub-bin refinement
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Firmware-matching constants
# ---------------------------------------------------------------------------

INNER_DIAL_CENTER_X_RATIO: float = 0.5000
INNER_DIAL_CENTER_Y_RATIO: float = 0.4460
BOARD_PRIOR_CENTER_X_RATIO: float = 0.4900
BOARD_PRIOR_CENTER_Y_RATIO: float = 0.4460
BRIGHT_THRESHOLD: float = 150.0
SATURATION_THRESHOLD: float = 235.0
MIN_BRIGHT_PIXELS: int = 64  # relaxed for 224x224 crops vs full frame
DIAL_RADIUS_FROM_CROP_HEIGHT_RATIO: float = 0.56
CENTER_SEARCH_RIM_MIN_FRACTION: float = 0.84
CENTER_SEARCH_RIM_MAX_FRACTION: float = 1.04
CENTER_SEARCH_COARSE_STEP: int = 8
CENTER_SEARCH_FINE_STEP: int = 4
CENTER_SEARCH_SAMPLE_STEP: int = 4
SCAN_BORDER_PIXELS: int = 8
CENTER_PRIOR_PENALTY_RATE: float = 0.25
CENTER_PRIOR_MIN: float = 0.20
MIN_RADIUS_PIXELS: float = 16.0
SUBDIAL_ANGLE_MIN: float = 55.0
SUBDIAL_ANGLE_MAX: float = 130.0
SWEEP_START_DEG: float = 135.0
SWEEP_END_DEG: float = 45.0  # wraps through 360 → actually 135+270=405 → mod 360 = 45

# Polar vote constants
POLAR_ANNULUS_INNER: float = 0.30
POLAR_ANNULUS_OUTER: float = 0.70
NUM_ANGLE_BINS: int = 360
SOBEL_THRESHOLD: float = 8.0


# ---------------------------------------------------------------------------
# Luma and edge helpers
# ---------------------------------------------------------------------------

def rgb_to_luma(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """BT.601 luma from RGB."""
    return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.float32)


def _sobel_edges(luma: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Compute Sobel gradient magnitude and direction."""
    # Simple Sobel without OpenCV
    # Simpler: just use central differences
    grad_x = np.zeros_like(luma)
    grad_y = np.zeros_like(luma)
    grad_x[:, 1:-1] = luma[:, 2:] - luma[:, :-2]
    grad_y[1:-1, :] = luma[2:, :] - luma[:-2, :]
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return mag, grad_x, grad_y


def _sobel_magnitude(luma: NDArray[np.float32]) -> NDArray[np.float32]:
    """Sobel edge magnitude."""
    mag, _, _ = _sobel_edges(luma)
    return mag


# ---------------------------------------------------------------------------
# Center hypotheses
# ---------------------------------------------------------------------------

def estimate_dial_radius(height: int) -> float:
    """Estimate dial radius from crop height (matching firmware)."""
    training_crop_y_span = 0.8071 - 0.2573  # APP_GAUGE_TRAINING_CROP_Y_MAX_RATIO - Y_MIN_RATIO
    crop_height = height * training_crop_y_span
    radius = crop_height * DIAL_RADIUS_FROM_CROP_HEIGHT_RATIO
    frame_limit = 0.49 * min(height, height)
    return max(MIN_RADIUS_PIXELS, min(radius, frame_limit))


def estimate_bright_centroid_on_crop(
    image: NDArray[np.uint8],
) -> tuple[float, float, bool]:
    """Bright-centroid detection on a 224×224 crop (matching firmware logic).

    Returns:
        (cx, cy, detected) in pixel coordinates.
    """
    height, width = image.shape[:2]
    inner_cx = width * INNER_DIAL_CENTER_X_RATIO
    inner_cy = height * INNER_DIAL_CENTER_Y_RATIO
    dial_radius = estimate_dial_radius(height)
    scan_radius = int(dial_radius * 1.5)

    scan_x_min = max(0, int(inner_cx) - scan_radius)
    scan_x_max = min(width, int(inner_cx) + scan_radius)
    scan_y_min = max(0, int(inner_cy) - scan_radius)
    scan_y_max = min(height, int(inner_cy) + scan_radius)

    luma = rgb_to_luma(image)
    bright_y, bright_x = np.where(
        (luma >= BRIGHT_THRESHOLD) & (luma <= SATURATION_THRESHOLD)
    )
    # Filter to scan region
    in_region = (
        (bright_x >= scan_x_min) & (bright_x < scan_x_max) &
        (bright_y >= scan_y_min) & (bright_y < scan_y_max)
    )
    bright_x = bright_x[in_region]
    bright_y = bright_y[in_region]

    if len(bright_x) < MIN_BRIGHT_PIXELS:
        return inner_cx, inner_cy, False

    raw_cx = float(np.mean(bright_x))
    raw_cy = float(np.mean(bright_y))

    # Y bias (upward shift toward inner dial)
    y_min = int(np.min(bright_y))
    y_max = int(np.max(bright_y))
    crop_h = y_max - y_min
    bias = int(0.11 * crop_h + 0.5)
    bias = max(8, min(bias, 18))
    biased_cy = max(0.0, raw_cy - bias)

    return raw_cx, biased_cy, True


def compute_crop_center(width: int, height: int) -> tuple[float, float]:
    """Fixed training-crop center (inner dial pivot)."""
    return width * INNER_DIAL_CENTER_X_RATIO, height * INNER_DIAL_CENTER_Y_RATIO


def compute_image_center(width: int, height: int) -> tuple[float, float]:
    """Geometric image center."""
    return width / 2.0, height / 2.0


def compute_board_prior_center(width: int, height: int) -> tuple[float, float]:
    """Board-prior center from observed firmware prior (matching app_baseline_runtime.c).

    Uses the same ratios as APP_BASELINE_BOARD_PRIOR_CENTER_X/Y_RATIO.
    """
    return width * BOARD_PRIOR_CENTER_X_RATIO, height * BOARD_PRIOR_CENTER_Y_RATIO


def score_rim_candidate(
    luma: NDArray[np.float32],
    width: int,
    height: int,
    candidate_x: int,
    candidate_y: int,
    dial_radius_px: float,
    scan_x_min: int,
    scan_y_min: int,
    scan_x_max: int,
    scan_y_max: int,
    crop_center_x: float,
    crop_center_y: float,
    crop_half_diag: float,
) -> float:
    """Score one candidate center for rim-edge alignment (matching firmware)."""
    rim_min = dial_radius_px * CENTER_SEARCH_RIM_MIN_FRACTION
    rim_max = dial_radius_px * CENTER_SEARCH_RIM_MAX_FRACTION
    mag, grad_x, grad_y = _sobel_edges(luma)
    score = 0.0
    count = 0

    for y in range(scan_y_min + 1, scan_y_max - 1, CENTER_SEARCH_SAMPLE_STEP):
        for x in range(scan_x_min + 1, scan_x_max - 1, CENTER_SEARCH_SAMPLE_STEP):
            if luma[y, x] > SATURATION_THRESHOLD:
                continue
            dx = x - candidate_x
            dy = y - candidate_y
            r = math.sqrt(dx * dx + dy * dy)
            if r < rim_min or r > rim_max:
                continue
            if mag[y, x] <= SOBEL_THRESHOLD:
                continue
            gx = grad_x[y, x] / max(mag[y, x], 1.0)
            gy = grad_y[y, x] / max(mag[y, x], 1.0)
            rx = dx / max(r, 1.0)
            ry = dy / max(r, 1.0)
            alignment = abs(gx * rx + gy * ry)
            rim_bias = 1.0 - min(abs(r - dial_radius_px) / max(dial_radius_px, 1.0), 1.0)
            rim_weight = rim_bias ** 2
            alignment_weight = alignment ** 2
            vote = mag[y, x] * alignment_weight * rim_weight
            if vote > 0.0:
                score += vote
                count += 1

    if count == 0:
        return 0.0

    center_dist = math.sqrt(
        (candidate_x - crop_center_x) ** 2 + (candidate_y - crop_center_y) ** 2
    )
    center_prior = max(CENTER_PRIOR_MIN,
                       1.0 - CENTER_PRIOR_PENALTY_RATE * center_dist / max(crop_half_diag, 1.0))
    return (score / count) * center_prior


def estimate_rim_center(
    luma: NDArray[np.float32],
    width: int,
    height: int,
    dial_radius_px: float,
) -> tuple[float, float, bool]:
    """Coarse-to-fine rim center search (matching firmware logic)."""
    crop_center_x, crop_center_y = compute_crop_center(width, height)
    crop_half_diag = math.sqrt((0.5 * width) ** 2 + (0.5 * height) ** 2)

    scan_x_min = SCAN_BORDER_PIXELS
    scan_y_min = SCAN_BORDER_PIXELS
    scan_x_max = width - SCAN_BORDER_PIXELS
    scan_y_max = height - SCAN_BORDER_PIXELS

    best_cx, best_cy = int(crop_center_x), int(crop_center_y)
    best_q = -1.0

    # Coarse pass
    for cy in range(scan_y_min, scan_y_max, CENTER_SEARCH_COARSE_STEP):
        for cx in range(scan_x_min, scan_x_max, CENTER_SEARCH_COARSE_STEP):
            q = score_rim_candidate(
                luma, width, height, cx, cy, dial_radius_px,
                scan_x_min, scan_y_min, scan_x_max, scan_y_max,
                crop_center_x, crop_center_y, crop_half_diag,
            )
            if q > best_q:
                best_q, best_cx, best_cy = q, cx, cy

    if best_q < 0:
        return crop_center_x, crop_center_y, False

    # Fine pass around coarse winner
    fine_min_x = max(scan_x_min, best_cx - CENTER_SEARCH_COARSE_STEP)
    fine_max_x = min(scan_x_max, best_cx + CENTER_SEARCH_COARSE_STEP)
    fine_min_y = max(scan_y_min, best_cy - CENTER_SEARCH_COARSE_STEP)
    fine_max_y = min(scan_y_max, best_cy + CENTER_SEARCH_COARSE_STEP)

    for cy in range(fine_min_y, fine_max_y + 1, CENTER_SEARCH_FINE_STEP):
        for cx in range(fine_min_x, fine_max_x + 1, CENTER_SEARCH_FINE_STEP):
            q = score_rim_candidate(
                luma, width, height, cx, cy, dial_radius_px,
                scan_x_min, scan_y_min, scan_x_max, scan_y_max,
                crop_center_x, crop_center_y, crop_half_diag,
            )
            if q > best_q:
                best_q, best_cx, best_cy = q, cx, cy

    return float(best_cx), float(best_cy), True


def compute_fast_hypotheses(
    image: NDArray[np.uint8],
) -> np.ndarray:
    """Compute 4 fast center hypotheses (skip slow rim search).

    Hypothesis order:
      1. bright_centroid
      2. crop_center
      3. board_prior
      4. image_center

    Returns:
        (4, 2) array.
    """
    height, width = image.shape[:2]
    h1 = estimate_bright_centroid_on_crop(image)
    h2 = compute_crop_center(width, height)
    h3 = compute_board_prior_center(width, height)
    h4 = compute_image_center(width, height)
    return np.array([
        [h1[0], h1[1]],
        [h2[0], h2[1]],
        [h3[0], h3[1]],
        [h4[0], h4[1]],
    ], dtype=np.float32)


def compute_all_hypotheses(
    image: NDArray[np.uint8],
) -> np.ndarray:
    """Compute all 5 classical center hypotheses in pixel coordinates.

    Hypothesis order (matching firmware baseline):
      1. bright_centroid  — luma bright-centroid detection
      2. crop_center      — fixed training crop center (inner dial pivot)
      3. board_prior      — observed board-prior center from firmware
      4. rim_center       — coarse-to-fine Hough rim search
      5. image_center     — geometric center of the crop

    Returns:
        (5, 2) array: [[bright_cx, bright_cy],
                       [crop_cx, crop_cy],
                       [board_prior_cx, board_prior_cy],
                       [rim_cx, rim_cy],
                       [image_cx, image_cy]]
    """
    height, width = image.shape[:2]
    luma = rgb_to_luma(image)
    dial_radius = estimate_dial_radius(height)

    h1 = estimate_bright_centroid_on_crop(image)
    h2 = compute_crop_center(width, height)
    h3 = compute_board_prior_center(width, height)
    h4 = estimate_rim_center(luma, width, height, dial_radius)
    h5 = compute_image_center(width, height)

    return np.array([
        [h1[0], h1[1]],
        [h2[0], h2[1]],
        [h3[0], h3[1]],
        [h4[0], h4[1]],
        [h5[0], h5[1]],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Simplified polar spoke vote
# ---------------------------------------------------------------------------

def polar_spoke_vote(
    luma: NDArray[np.float32],
    center_x: float,
    center_y: float,
    dial_radius: float,
    *,
    num_bins: int = NUM_ANGLE_BINS,
) -> np.ndarray:
    """Simplified polar spoke voting (gradient-based, no continuity scoring).

    For each pixel in the annular region around the center:
      1. Compute Sobel edge magnitude and gradient direction
      2. Compute tangential alignment (cross product of gradient × radial)
      3. Weight by edge magnitude and darkness (255 - luma)
      4. Accumulate into angular bins

    Returns:
        (num_bins,) vote array.
    """
    height, width = luma.shape
    mag, grad_x, grad_y = _sobel_edges(luma)
    votes = np.zeros(num_bins, dtype=np.float64)
    r_min = dial_radius * POLAR_ANNULUS_INNER
    r_max = dial_radius * POLAR_ANNULUS_OUTER

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            dx = x - center_x
            dy = y - center_y
            r = math.sqrt(dx * dx + dy * dy)
            if r < r_min or r > r_max:
                continue
            if mag[y, x] <= SOBEL_THRESHOLD:
                continue
            if luma[y, x] > SATURATION_THRESHOLD:
                continue

            darkness = (255.0 - luma[y, x]) / 255.0
            # Tangential alignment: cross product of unit gradient × unit radial
            gx_norm = grad_x[y, x] / max(mag[y, x], 1.0)
            gy_norm = grad_y[y, x] / max(mag[y, x], 1.0)
            rx = dx / max(r, 1.0)
            ry = dy / max(r, 1.0)
            tangential = abs(gx_norm * ry - gy_norm * rx)

            # Radial angle of this pixel
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad) % 360.0
            bin_idx = int(angle_deg * num_bins / 360.0) % num_bins

            vote = mag[y, x] * tangential * darkness
            votes[bin_idx] += vote

    return votes


def smooth_and_find_peak(
    votes: np.ndarray,
    *,
    num_bins: int = NUM_ANGLE_BINS,
) -> tuple[float, float, float]:
    """3-bin boxcar smooth, find peak, sub-bin refine.

    Returns:
        (best_angle_deg, best_vote, mean_vote)
    """
    smoothed = np.zeros_like(votes)
    for i in range(num_bins):
        smoothed[i] = (votes[(i - 1) % num_bins] + votes[i] + votes[(i + 1) % num_bins]) / 3.0

    best_idx = int(np.argmax(smoothed))

    # Sub-bin refinement via weighted average of 3 neighbors
    prev_idx = (best_idx - 1) % num_bins
    next_idx = (best_idx + 1) % num_bins
    total = smoothed[prev_idx] + smoothed[best_idx] + smoothed[next_idx]
    if total > 0:
        refined = (
            prev_idx * smoothed[prev_idx] +
            best_idx * smoothed[best_idx] +
            next_idx * smoothed[next_idx]
        ) / total
    else:
        refined = float(best_idx)

    # Convert to degrees in [0, 360)
    best_angle = ((refined / num_bins) * 360.0) % 360.0
    best_vote = float(smoothed[best_idx])
    mean_vote = float(np.mean(smoothed))

    return best_angle, best_vote, mean_vote


def is_angle_in_sweep(angle_deg: float, *, margin: float = 5.0) -> bool:
    """Check if angle is in the Celsius sweep [135, 405)° with margin."""
    angle_norm = angle_deg % 360.0
    sweep_start = SWEEP_START_DEG - margin
    sweep_end = (SWEEP_START_DEG + 270.0 + margin) % 360.0

    if sweep_start < sweep_end:
        return sweep_start <= angle_norm <= sweep_end
    else:
        return angle_norm >= sweep_start or angle_norm <= sweep_end


def needle_angle_from_polar_vote(
    image: NDArray[np.uint8],
    center_x: float,
    center_y: float,
    dial_radius: float | None = None,
) -> float:
    """Full pipeline: polar vote → smooth → peak → return angle in degrees.

    Args:
        image: 224×224 uint8 RGB crop.
        center_x, center_y: Predicted center in pixel coords.
        dial_radius: Optional override; auto-estimated if None.

    Returns:
        Best needle angle in degrees [0, 360).
    """
    height, width = image.shape[:2]
    if dial_radius is None:
        dial_radius = estimate_dial_radius(height)
    luma = rgb_to_luma(image)
    votes = polar_spoke_vote(luma, center_x, center_y, dial_radius)
    angle, _, _ = smooth_and_find_peak(votes)
    return angle
