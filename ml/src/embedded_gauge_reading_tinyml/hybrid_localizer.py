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

# Subdial clutter mask (matching APP_BASELINE_SUBDIAL_*)
SUBDIAL_X_FRACTION: float = 0.35
SUBDIAL_Y_MIN_FRACTION: float = 0.10
SUBDIAL_Y_MAX_FRACTION: float = 0.58

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
    """3×3 Sobel gradient magnitude and direction (matching firmware)."""
    grad_x = np.zeros_like(luma)
    grad_y = np.zeros_like(luma)
    # Sobel X: [-1, 0, +1] * 3 with weights
    grad_x[1:-1, 1:-1] = (
        -1.0 * luma[:-2, :-2] + 1.0 * luma[:-2, 2:] +
        -2.0 * luma[1:-1, :-2] + 2.0 * luma[1:-1, 2:] +
        -1.0 * luma[2:, :-2] + 1.0 * luma[2:, 2:]
    )
    # Sobel Y: [-1, -2, -1] / [+1, +2, +1]
    grad_y[1:-1, 1:-1] = (
        -1.0 * luma[:-2, :-2] + -2.0 * luma[:-2, 1:-1] + -1.0 * luma[:-2, 2:] +
         1.0 * luma[2:, :-2] + 2.0 * luma[2:, 1:-1] + 1.0 * luma[2:, 2:]
    )
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return mag, grad_x, grad_y


def _sobel_magnitude(luma: NDArray[np.float32]) -> NDArray[np.float32]:
    """Sobel edge magnitude."""
    mag, _, _ = _sobel_edges(luma)
    return mag


def _middle_shaft_weight(sample_progress: float) -> float:
    """Bell-shaped weight peaking at 50% annulus position (matching firmware)."""
    return 1.0 - 4.0 * (sample_progress - 0.5) ** 2


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
    """Geometric centre of the training-crop rectangle, matching firmware
    AppGaugeGeometry_TrainingCrop.  The firmware uses the training-crop
    geometric centre for the centre-prior anchor in the rim search,
    NOT the inner-dial pivot."""
    x_min = width * 0.1027   # APP_GAUGE_TRAINING_CROP_X_MIN_RATIO
    y_min = height * 0.2573  # APP_GAUGE_TRAINING_CROP_Y_MIN_RATIO
    w = width * (0.7987 - 0.1027)
    h = height * (0.8071 - 0.2573)
    return (x_min + w / 2.0, y_min + h / 2.0)


def compute_image_center(width: int, height: int) -> tuple[float, float]:
    """Geometric image center."""
    return width / 2.0, height / 2.0


def compute_board_prior_center(width: int, height: int) -> tuple[float, float]:
    """Board-prior center from observed firmware prior (matching app_baseline_runtime.c).

    Uses the same ratios as APP_BASELINE_BOARD_PRIOR_CENTER_X/Y_RATIO.
    """
    return width * BOARD_PRIOR_CENTER_X_RATIO, height * BOARD_PRIOR_CENTER_Y_RATIO


def _is_in_subdial_mask(
    center_x: float, center_y: float,
    x: int, y: int, radius_px: float,
) -> bool:
    """Check whether (x, y) falls inside the subdial clutter band,
    matching firmware AppBaselineRuntime_IsInSubdialMask."""
    dx = abs(x - center_x)
    dy = abs(y - center_y)
    return (
        dx < (SUBDIAL_X_FRACTION * radius_px)
        and y > (center_y + SUBDIAL_Y_MIN_FRACTION * radius_px)
        and y < (center_y + SUBDIAL_Y_MAX_FRACTION * radius_px)
        and dy > (SUBDIAL_Y_MIN_FRACTION * radius_px)
    )


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
            if _is_in_subdial_mask(candidate_x, candidate_y, x, y, dial_radius_px):
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
# Firmware-matching polar spoke vote with structural needle scoring
# ---------------------------------------------------------------------------

def polar_spoke_vote(
    luma: NDArray[np.float32],
    center_x: float,
    center_y: float,
    dial_radius: float,
    *,
    num_bins: int = NUM_ANGLE_BINS,
    edge_threshold: float = 8.0,
    use_structural_boost: bool = True,
) -> np.ndarray:
    """Polar spoke vote with Sobel edges, shaft weight, optional structural boost.

    When use_structural_boost=True, applies hub-connection + tip-extension checks
    to boost needle-like radial features over isolated dial markings.

    Returns:
        (num_bins,) vote array.
    """
    height, width = luma.shape
    mag, grad_x, grad_y = _sobel_edges(luma)
    votes = np.zeros(num_bins, dtype=np.float64)
    r_min = dial_radius * POLAR_ANNULUS_INNER
    r_max = dial_radius * POLAR_ANNULUS_OUTER
    r_range = r_max - r_min + 1e-6
    saturation = SATURATION_THRESHOLD

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            dx = x - center_x
            dy = y - center_y
            r = math.sqrt(dx * dx + dy * dy)
            if r < r_min or r > r_max:
                continue
            if mag[y, x] <= edge_threshold:
                continue
            if luma[y, x] > saturation:
                continue

            # Radial angle of this pixel → bin
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad) % 360.0

            darkness = (255.0 - luma[y, x]) / 255.0

            # Tangential alignment: cross product of unit gradient × unit radial
            gx_norm = grad_x[y, x] / max(mag[y, x], 1.0)
            gy_norm = grad_y[y, x] / max(mag[y, x], 1.0)
            rx = dx / max(r, 1.0)
            ry = dy / max(r, 1.0)
            tangential = abs(gx_norm * ry - gy_norm * rx)

            # Shaft weight: focus on middle of annulus
            sample_progress = (r - r_min) / r_range
            shaft_weight = 0.35 + 0.65 * _middle_shaft_weight(sample_progress)

            vote = mag[y, x] * tangential * darkness * shaft_weight
            if vote <= 0.0:
                continue

            if use_structural_boost:
                # Hub connection: sample points toward center
                dx_norm = dx / r
                dy_norm = dy / r
                hub_count = 0.0
                for step in range(1, 8):
                    t = step / 8.0 * 0.6
                    hx = int(round(center_x + dx * t))
                    hy = int(round(center_y + dy * t))
                    if 0 <= hx < width and 0 <= hy < height:
                        hub_count += (255.0 - luma[hy, hx]) / 255.0
                hub_connection = hub_count / 7.0

                # Tip extension: sample outward
                tip_count = 0.0
                for step in range(5):
                    r_frac = 0.70 + 0.25 * step / 4.0
                    tx = int(round(center_x + dx_norm * r_frac * dial_radius))
                    ty = int(round(center_y + dy_norm * r_frac * dial_radius))
                    if 0 <= tx < width and 0 <= ty < height:
                        tip_count += (255.0 - luma[ty, tx]) / 255.0
                tip_extension = tip_count / 5.0

                spoke_score = (hub_connection + tip_extension) * 0.5
                vote *= (0.3 + 3.7 * spoke_score * spoke_score)

            bin_idx = int(angle_deg * num_bins / 360.0) % num_bins
            votes[bin_idx] += vote

    return votes


def smooth_and_find_peak(
    votes: np.ndarray,
    *,
    num_bins: int = NUM_ANGLE_BINS,
    luma: NDArray[np.float32] | None = None,
    center_x: float | None = None,
    center_y: float | None = None,
    dial_radius: float | None = None,
    continuity_threshold: float = 0.35,
    hub_threshold: float = 0.25,
) -> tuple[float, float, float]:
    """3-bin boxcar smooth, find peak with spoke-continuity voting, sub-bin refine.

    When luma/center/radius are provided, selects the peak with best
    spoke-continuity * hub-darkness weighting (matching firmware's continuity
    scoring in AppBaselineRuntime_EstimatePolarNeedle).

    Returns:
        (best_angle_deg, best_vote, mean_vote)
    """
    smoothed = np.zeros_like(votes)
    for i in range(num_bins):
        smoothed[i] = (votes[(i - 1) % num_bins] + votes[i] + votes[(i + 1) % num_bins]) / 3.0

    best_idx = int(np.argmax(smoothed))

    # Spoke-continuity weighted peak selection
    if luma is not None and center_x is not None and center_y is not None and dial_radius is not None:
        height, width = luma.shape
        # Get top 16 peaks
        top_count = min(16, num_bins)
        top_indices = np.argpartition(smoothed, -top_count)[-top_count:]
        top_order = np.argsort(-smoothed[top_indices])
        top_indices = top_indices[top_order]
        top_scores = smoothed[top_indices]

        best_weighted_score = 0.0
        best_weighted_idx = best_idx

        for idx, score_val in zip(top_indices, top_scores):
            if score_val <= 0.0:
                continue

            angle_deg_atan2 = (idx / num_bins) * 360.0

            # Only consider peaks within the valid sweep: [135°, 405°) in atan2 convention
            # This maps to [135°, 360°) ∪ [0°, 45°)
            sweep_ok = (angle_deg_atan2 >= 135.0) or (angle_deg_atan2 <= 45.0)
            if not sweep_ok:
                continue

            # Convert bin to atan2 physical angle [0, 2π)
            # bin 0 = atan2 = 0 = +x (right); bin 180 = atan2 = π = -x (left)
            angle_rad = (idx / num_bins) * 2.0 * math.pi
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            # Spoke continuity: sample 12 points from 20%-80% of radius
            continuity = 0.0
            valid = 0
            for i in range(12):
                r_frac = 0.20 + 0.60 * i / 11.0
                sx = int(round(center_x + cos_a * r_frac * dial_radius))
                sy = int(round(center_y + sin_a * r_frac * dial_radius))
                if 0 <= sx < width and 0 <= sy < height:
                    continuity += (255.0 - luma[sy, sx]) / 255.0
                    valid += 1
            if valid == 0:
                continue
            continuity /= valid

            # Hub darkness: sample 3 points near center (10%-25% of radius)
            hub_darkness = 0.0
            hub_valid = 0
            for h in range(3):
                r_frac = 0.10 + 0.15 * h / 2.0
                hx = int(round(center_x + cos_a * r_frac * dial_radius))
                hy = int(round(center_y + sin_a * r_frac * dial_radius))
                if 0 <= hx < width and 0 <= hy < height:
                    hub_darkness += (255.0 - luma[hy, hx]) / 255.0
                    hub_valid += 1
            if hub_valid > 0:
                hub_darkness /= hub_valid

            # Relax thresholds for smaller crop images (224x224) vs full-frame 640x480
            cont_ok = continuity >= (continuity_threshold * 0.75)
            hub_ok = hub_darkness >= (hub_threshold * 0.75)
            if cont_ok and hub_ok:
                weighted_score = continuity * continuity * hub_darkness * score_val
                if weighted_score > best_weighted_score:
                    best_weighted_score = weighted_score
                    best_weighted_idx = idx

        best_idx = best_weighted_idx

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


# ---------------------------------------------------------------------------
# Center quality scoring and refinement (matching firmware)
# ---------------------------------------------------------------------------

def score_polar_quality(
    votes: np.ndarray,
    luma: NDArray[np.float32],
    center_x: float,
    center_y: float,
    dial_radius: float,
    *,
    num_bins: int = NUM_ANGLE_BINS,
    suppression_bins: int = 15,
    continuity_threshold: float = 0.27,
    hub_threshold: float = 0.19,
) -> float:
    """Score a center candidate by polar vote peak quality (matching firmware).

    Higher score = better center. Factors:
      - Sweep-valid peak (peak must be in [135°, 45°] atan2 range)
      - Peak ratio (best_score / runner_up after suppression)
      - Confidence (best_score / mean_vote)
      - Spoke continuity (darkness along the spoke direction)
      - Hub darkness (darkness near center along the spoke)

    Returns:
        Combined quality score (0 = reject).
    """
    # Smooth and find the best peak within the sweep range [135°, 45°]
    smoothed = np.zeros(num_bins, dtype=np.float64)
    for i in range(num_bins):
        smoothed[i] = (votes[(i - 1) % num_bins] + votes[i] + votes[(i + 1) % num_bins]) / 3.0

    best_score = 0.0
    best_idx = 0
    for i in range(num_bins):
        ang = (i / num_bins) * 360.0
        in_sweep = (ang >= 135.0) or (ang <= 45.0)
        if in_sweep and smoothed[i] > best_score:
            best_score = float(smoothed[i])
            best_idx = i

    if best_score <= 0.0:
        return 0.0

    # Suppress neighbors to find runner-up
    suppressed = smoothed.copy()
    start = max(0, best_idx - suppression_bins)
    end = min(num_bins, best_idx + suppression_bins + 1)
    suppressed[start:end] = 0.0
    # Also suppress the wrap-around
    if start == 0:
        suppressed[num_bins - suppression_bins:] = 0.0
    if end == num_bins:
        suppressed[:best_idx + suppression_bins + 1 - num_bins] = 0.0

    runner_up = float(np.max(suppressed))
    mean_vote = float(np.mean(smoothed))

    # Compute peak ratio and confidence
    peak_ratio = best_score / max(runner_up, 1.0)
    confidence = best_score / max(mean_vote, 1.0)

    # Spoke continuity at the peak angle
    height, width = luma.shape
    angle_rad = (best_idx / num_bins) * 2.0 * math.pi
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    cont = 0.0
    cont_valid = 0
    for i in range(12):
        r_frac = 0.20 + 0.60 * i / 11.0
        sx = int(round(center_x + cos_a * r_frac * dial_radius))
        sy = int(round(center_y + sin_a * r_frac * dial_radius))
        if 0 <= sx < width and 0 <= sy < height:
            cont += (255.0 - luma[sy, sx]) / 255.0
            cont_valid += 1
    cont /= max(cont_valid, 1)

    # Hub darkness
    hub = 0.0
    hub_valid = 0
    for h in range(3):
        r_frac = 0.10 + 0.15 * h / 2.0
        hx = int(round(center_x + cos_a * r_frac * dial_radius))
        hy = int(round(center_y + sin_a * r_frac * dial_radius))
        if 0 <= hx < width and 0 <= hy < height:
            hub += (255.0 - luma[hy, hx]) / 255.0
            hub_valid += 1
    hub /= max(hub_valid, 1)

    return peak_ratio * confidence * max(cont, 0.1) * max(hub, 0.1)


def refine_center(
    image: NDArray[np.uint8],
    seed_cx: float,
    seed_cy: float,
    dial_radius: float,
    *,
    offsets_px: tuple[float, ...] = (-8, -4, 0, 4, 8),
    edge_threshold: float = 8.0,
    use_structural_boost: bool = True,
) -> tuple[float, float, float, float]:
    """Refine a center hypothesis by probing a local neighbourhood (matching firmware).

    Tries every (dx, dy) combination from offsets_px around the seed center,
    runs the polar vote, scores each, and returns the best (cx, cy, angle, quality).

    Args:
        image: 224×224 uint8 RGB crop.
        seed_cx, seed_cy: Starting centre in pixel coords.
        dial_radius: Dial radius in pixels.

    Returns:
        (best_cx, best_cy, best_angle_deg, best_quality).
    """
    height, width = image.shape[:2]
    luma = rgb_to_luma(image)
    best_cx, best_cy = seed_cx, seed_cy
    best_angle = 0.0
    best_quality = -1.0

    for dy in offsets_px:
        for dx in offsets_px:
            cx = seed_cx + dx
            cy = seed_cy + dy
            cx = max(1.0, min(float(width - 2), cx))
            cy = max(1.0, min(float(height - 2), cy))

            votes = polar_spoke_vote(luma, cx, cy, dial_radius,
                                     edge_threshold=edge_threshold,
                                     use_structural_boost=use_structural_boost)
            quality = score_polar_quality(votes, luma, cx, cy, dial_radius)

            if quality > best_quality:
                best_quality = quality
                best_cx, best_cy = cx, cy
                angle, _, _ = smooth_and_find_peak(votes)
                best_angle = angle

    return best_cx, best_cy, best_angle, best_quality


def needle_angle_from_polar_vote(
    image: NDArray[np.uint8],
    center_x: float,
    center_y: float,
    dial_radius: float | None = None,
    *,
    edge_threshold: float = 8.0,
    continuity_threshold: float = 0.27,
    hub_threshold: float = 0.19,
    refine: bool = True,
    use_structural_boost: bool = True,
) -> float:
    """Full pipeline: polar vote → continuity-scored peak → angle in degrees.

    When refine=True, runs a local centre refinement sweep around the seed
    (matching firmware's AppBaselineRuntime_RefineEstimateAroundSeed).

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

    if refine:
        _, _, angle, quality = refine_center(
            image, center_x, center_y, dial_radius,
            edge_threshold=edge_threshold,
            use_structural_boost=use_structural_boost,
        )
        if quality > 0.0:
            return angle

    # Fallthrough: use simple peak (no sweep filter, no continuity)
    luma = rgb_to_luma(image)
    votes = polar_spoke_vote(luma, center_x, center_y, dial_radius,
                             edge_threshold=edge_threshold,
                             use_structural_boost=use_structural_boost)
    angle, _, _ = smooth_and_find_peak(votes)
    return angle
