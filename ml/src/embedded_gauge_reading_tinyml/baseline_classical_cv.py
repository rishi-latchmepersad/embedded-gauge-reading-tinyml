"""Classical computer-vision baseline for analog gauge reading.

This module provides a non-neural baseline that can be used to benchmark
whether a learned model is improving over simple image processing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Final, Iterable, Sequence

import cv2
import numpy as np

from embedded_gauge_reading_tinyml.dataset import Sample
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, needle_value


@dataclass(frozen=True)
class NeedleDetection:
    """Needle detection output plus the angular peak-shape metadata."""

    unit_dx: float
    unit_dy: float
    confidence: float
    peak_value: float
    runner_up_value: float
    peak_ratio: float
    peak_margin: float




def needle_detection_quality(detection: NeedleDetection) -> float:
    """Return a scalar quality score for a needle detection."""
    if detection is None:
        return -1.0
    return float(detection.confidence * detection.peak_ratio)

@dataclass(frozen=True)
class ClassicalPrediction:
    """Predicted gauge value and metadata for one sample."""

    image_path: str
    true_value: float
    predicted_value: float
    abs_error: float
    confidence: float


@dataclass(frozen=True)
class ClassicalBaselineResult:
    """Aggregate metrics for a full classical-CV baseline run."""

    attempted_samples: int
    successful_samples: int
    failed_samples: int
    mae: float
    rmse: float
    predictions: list[ClassicalPrediction]


@dataclass(frozen=True)
class GeometryCandidate:
    """One candidate dial geometry hypothesis for image-based inference."""

    label: str
    center_xy: tuple[float, float]
    dial_radius_px: float


@dataclass(frozen=True)
class GeometrySelection:
    """Best needle detection found for one geometry candidate set."""

    candidate: GeometryCandidate
    detection: NeedleDetection
    quality: float
    shaft_support: float


# Fixed geometry constants (matches embedded C baseline)
BRIGHT_CENTROID_MIN_PIXELS: Final[int] = 1024
BRIGHT_CENTROID_MIN_LUMA: Final[int] = 150
BRIGHT_CENTROID_MAX_LUMA: Final[int] = 220
FIXED_CROP_CENTER_X: Final[float] = 100.0
FIXED_CROP_CENTER_Y: Final[float] = 118.0
IMAGE_CENTER_X: Final[float] = 112.0
IMAGE_CENTER_Y: Final[float] = 112.0
DEFAULT_DIAL_RADIUS_PX: Final[float] = 80.0


@dataclass(frozen=True)
class CenterHypothesis:
    """One needle-center hypothesis for the classical baseline."""

    label: str
    center_xy: tuple[float, float]
    dial_radius_px: float = DEFAULT_DIAL_RADIUS_PX


def _bright_centroid_hypothesis(image_bgr: np.ndarray) -> CenterHypothesis | None:
    """Find the bright centroid of the dial face (matches C baseline).

    Scans the fixed gauge crop for pixels with 150 â‰¤ luma â‰¤ 220,
    computes the centroid, and requires â‰¥ 1024 qualifying pixels.
    """
    gray: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape[:2]

    # Fixed gauge crop region (matches C baseline)
    x_start, x_end = 23, 178
    y_start, y_end = 57, 180

    # Clamp to image bounds
    x_start = max(0, min(x_start, w_img - 1))
    x_end = max(x_start + 1, min(x_end, w_img))
    y_start = max(0, min(y_start, h_img - 1))
    y_end = max(y_start + 1, min(y_end, h_img))

    crop: np.ndarray = gray[y_start:y_end, x_start:x_end]
    mask: np.ndarray = (
        (crop >= BRIGHT_CENTROID_MIN_LUMA) & (crop <= BRIGHT_CENTROID_MAX_LUMA)
    )

    num_pixels: int = int(np.sum(mask))
    if num_pixels < BRIGHT_CENTROID_MIN_PIXELS:
        return None

    # Compute centroid
    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        return None

    centroid_x_crop: float = float(np.mean(xs))
    centroid_y_crop: float = float(np.mean(ys))

    # Convert to full image coordinates
    centroid_x: float = float(x_start) + centroid_x_crop
    centroid_y: float = float(y_start) + centroid_y_crop

    return CenterHypothesis(
        label="bright_centroid",
        center_xy=(centroid_x, centroid_y),
    )


def _fixed_crop_hypothesis(image_bgr: np.ndarray) -> CenterHypothesis:
    """Return the fixed-crop center hypothesis (matches C baseline)."""
    return CenterHypothesis(
        label="fixed_crop",
        center_xy=(FIXED_CROP_CENTER_X, FIXED_CROP_CENTER_Y),
    )


def _image_center_hypothesis(image_bgr: np.ndarray) -> CenterHypothesis:
    """Return the image center hypothesis (matches C baseline)."""
    h_img, w_img = image_bgr.shape[:2]
    return CenterHypothesis(
        label="image_center",
        center_xy=(float(w_img) / 2.0, float(h_img) / 2.0),
    )


def _center_hypotheses(
    image_bgr: np.ndarray,
) -> list[CenterHypothesis]:
    """Generate all center hypotheses (matches C baseline)."""
    hypotheses: list[CenterHypothesis] = []

    # Try bright centroid first
    bright: CenterHypothesis | None = _bright_centroid_hypothesis(image_bgr)
    if bright is not None:
        hypotheses.append(bright)

    # Always include fixed crop and image center
    hypotheses.append(_fixed_crop_hypothesis(image_bgr))
    hypotheses.append(_image_center_hypothesis(image_bgr))

    return hypotheses


def board_prior_geometry_candidate(image_bgr: np.ndarray) -> GeometryCandidate:
    """Build a small classical neighborhood around the observed board prior."""
    height, width = image_bgr.shape[:2]
    min_dim: float = float(min(height, width))
    return GeometryCandidate(
        label="board_prior",
        center_xy=(
            BOARD_PRIOR_CENTER_X_RATIO * float(width),
            BOARD_PRIOR_CENTER_Y_RATIO * float(height),
        ),
        dial_radius_px=BOARD_PRIOR_RADIUS_RATIO * min_dim,
    )


def board_prior_geometry_candidates(image_bgr: np.ndarray) -> list[GeometryCandidate]:
    """Build a small classical neighborhood around the observed board prior."""
    base_candidate: GeometryCandidate = board_prior_geometry_candidate(image_bgr)
    candidates: list[GeometryCandidate] = []
    for dx in BOARD_PRIOR_CENTER_OFFSETS_PX:
        for dy in BOARD_PRIOR_CENTER_OFFSETS_PX:
            for radius_scale in BOARD_PRIOR_RADIUS_SCALES:
                candidates.append(
                    GeometryCandidate(
                        label=(
                            f"board_prior_{int(dx):+d}_{int(dy):+d}_"
                            f"{radius_scale:.2f}"
                        ),
                        center_xy=(
                            base_candidate.center_xy[0] + dx,
                            base_candidate.center_xy[1] + dy,
                        ),
                        dial_radius_px=base_candidate.dial_radius_px * radius_scale,
                    )
                )
    return candidates


def needle_vector_to_value(unit_dx: float, unit_dy: float, spec: GaugeSpec) -> float:
    """Convert a unit direction vector to calibrated gauge value.

    The conversion mirrors the sweep calibration math used by training/evaluation.
    """
    raw_angle: float = math.atan2(unit_dy, unit_dx)

    # Shift angle into the gauge's calibrated frame and wrap into [0, 2*pi).
    shifted: float = (raw_angle - spec.min_angle_rad) % (2.0 * math.pi)

    # Clamp non-strictly so out-of-sweep detections saturate at max value.
    fraction: float = min(max(shifted / spec.sweep_rad, 0.0), 1.0)
    return spec.min_value + fraction * (spec.max_value - spec.min_value)


def _runner_up_peak_after_suppression(
    peak_values: np.ndarray,
    *,
    best_index: int,
    suppression_bins: int,
) -> float:
    """Find the strongest non-neighbor peak after suppressing the main peak."""
    if peak_values.size == 0:
        return 0.0

    window: int = max(1, int(suppression_bins))
    if peak_values.size <= (2 * window + 1):
        return 0.0

    rolled: np.ndarray = np.roll(peak_values, -int(best_index))
    competitor_values: np.ndarray = rolled[window + 1 : peak_values.size - window]
    if competitor_values.size == 0:
        return 0.0
    return float(np.max(competitor_values))


def _angle_in_sweep(
    angle_rad: float, spec: GaugeSpec, *, margin_rad: float = 0.0
) -> bool:
    """Return True if angle_rad falls within the gauge's calibrated sweep arc."""
    shifted: float = (angle_rad - spec.min_angle_rad) % (2.0 * math.pi)
    return shifted <= (spec.sweep_rad + margin_rad)


def _middle_shaft_weight(rr: np.ndarray, dial_radius_px: float) -> np.ndarray:
    """Return a radial weight that peaks in the middle-shaft band (~0.45 × radius).

    Falls off toward the hub and rim so that the middle of the needle shaft
    contributes the most to a spoke vote.
    """
    peak_r: float = 0.45 * dial_radius_px
    sigma: float = max(0.15 * dial_radius_px, 1.0)
    return np.exp(-0.5 * ((rr - peak_r) / sigma) ** 2)


def _find_top_local_peaks(
    values: np.ndarray,
    num_peaks: int = 5,
    min_distance: int = 30,
) -> list[tuple[int, float]]:
    """Return the (index, value) of the top N local maxima.

    Each local maximum must be at least `min_distance` bins away from any
    higher-ranked peak so we avoid returning multiple bins of the same bump.
    """
    if values.size == 0:
        return []

    # Local maxima are points higher than both neighbours (wrap-around).
    left: np.ndarray = np.roll(values, 1)
    right: np.ndarray = np.roll(values, -1)
    is_peak: np.ndarray = (values >= left) & (values >= right)

    candidates: list[tuple[int, float]] = [
        (int(i), float(values[i])) for i in range(values.size) if is_peak[i]
    ]
    candidates.sort(key=lambda pair: pair[1], reverse=True)

    selected: list[tuple[int, float]] = []
    for idx, val in candidates:
        # Check if this peak is far enough from already-selected peaks.
        if all(abs(idx - s_idx) >= min_distance for s_idx, _ in selected):
            selected.append((idx, val))
            if len(selected) >= num_peaks:
                break

    return selected


def _score_radial_spoke(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    angle_rad: float,
    dial_radius_px: float,
    gray: np.ndarray | None = None,
) -> float:
    """Score how needle-like a radial spoke is at the given angle.

    Uses the existing _sample_line_darkness function and returns a
    composite score. This is used as a tiebreaker when the polar histogram
    is ambiguous.
    """
    if gray is None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    cx, cy = center_xy
    tip_x: float = cx + math.cos(angle_rad) * 0.80 * dial_radius_px
    tip_y: float = cy + math.sin(angle_rad) * 0.80 * dial_radius_px
    tail_x: float = cx + math.cos(angle_rad) * 0.25 * dial_radius_px
    tail_y: float = cy + math.sin(angle_rad) * 0.25 * dial_radius_px

    contrast, dark_fraction = _sample_line_darkness(
        gray,
        x1=tail_x,
        y1=tail_y,
        x2=tip_x,
        y2=tip_y,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
    )
    return max(contrast, 0.0) + 0.80 * dark_fraction


def _needle_detection_shaft_score(
    image_bgr: np.ndarray,
    detection: NeedleDetection,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
) -> float:
    """Score how needle-like a detection is by sampling its shaft darkness.

    Returns a score in [0, ~2] where higher means the detection looks more
    like a dark needle shaft.
    """
    angle_rad: float = math.atan2(detection.unit_dy, detection.unit_dx)
    contrast, dark_fraction = _sample_line_darkness(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY),
        x1=center_xy[0] + math.cos(angle_rad) * 0.25 * dial_radius_px,
        y1=center_xy[1] + math.sin(angle_rad) * 0.25 * dial_radius_px,
        x2=center_xy[0] + math.cos(angle_rad) * 0.80 * dial_radius_px,
        y2=center_xy[1] + math.sin(angle_rad) * 0.80 * dial_radius_px,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
    )
    return max(contrast, 0.0) + 0.80 * dark_fraction


def _detect_needle_unit_vector_spoked_arc(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec | None = None,
) -> NeedleDetection | None:
    """Detect needle direction with polar + Hough-line fusion.

    Runs the baseline polar voter AND the Hough line-segment detector
    independently, scores each by shaft darkness, and returns the one
    that looks more like a dark needle.

    The polar histogram is excellent when the needle is the dominant dark
    radial feature.  It occasionally locks onto a tick mark or shadow
    (wrong-spoke failure).  The Hough line-segment detector explicitly
    looks for a line that passes from near the hub toward the rim — a
    structural constraint the polar vote lacks.

    By picking whichever method produces a higher shaft-darkness score
    we fuse the strengths of both approaches.
    """
    polar_result: NeedleDetection | None = _detect_needle_unit_vector_polar(
        image_bgr,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
        gauge_spec=gauge_spec,
    )
    hough_result: NeedleDetection | None = _detect_needle_unit_vector_line_segment(
        image_bgr,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
        gauge_spec=gauge_spec,
    )

    candidates: list[tuple[NeedleDetection, str]] = []
    if polar_result is not None:
        candidates.append((polar_result, "polar"))
    if hough_result is not None:
        candidates.append((hough_result, "hough"))

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0][0]

    # Score each candidate by shaft darkness and pick the best.
    best_score: float = float("-inf")
    best_detection: NeedleDetection | None = None
    best_name: str = ""

    for detection, name in candidates:
        shaft_score: float = _needle_detection_shaft_score(
            image_bgr, detection,
            center_xy=center_xy,
            dial_radius_px=dial_radius_px,
        )
        # Boost the polar score slightly to prefer it when scores are close.
        adjusted_score: float = shaft_score * (1.15 if name == "polar" else 1.0)
        if adjusted_score > best_score:
            best_score = adjusted_score
            best_detection = detection
            best_name = name

    # Confidence floor: if the winner has a very low shaft score, reject it.
    if best_score < 0.08:
        return None

    return best_detection


def _detect_needle_unit_vector_polar(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec | None = None,
) -> NeedleDetection | None:
    """Detect the needle via polar edge voting (matches embedded C baseline).

    This is a simplified implementation that mirrors the embedded C code:
    - 360 angle bins (1Â° resolution)
    - Grayscale only (no HSV saturation weighting)
    - Simple annulus mask (no middle-shaft Gaussian)
    - Basic contrast + edge strength voting
    - Confidence = peak / mean
    """
    if dial_radius_px <= 1.0:
        return None

    center_x, center_y = center_xy

    # Simple grayscale conversion (no CLAHE, no HSV)
    gray: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred: np.ndarray = cv2.GaussianBlur(gray, (5, 5), 0)

    h_img, w_img = gray.shape[:2]
    yy, xx = np.meshgrid(
        np.arange(h_img, dtype=np.float32),
        np.arange(w_img, dtype=np.float32),
        indexing="ij",
    )
    dx: np.ndarray = xx - center_x
    dy: np.ndarray = yy - center_y
    rr: np.ndarray = np.sqrt(dx**2 + dy**2)

    # Simple annulus mask (skip subdial and outer ring)
    inner_mask: np.ndarray = (rr > 0.30 * dial_radius_px) & (rr < 0.70 * dial_radius_px)

    # Skip saturated samples (luma > 220)
    saturation_mask: np.ndarray = gray < 220.0

    # Compute simple edge strength (Sobel magnitude)
    gx: np.ndarray = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy: np.ndarray = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag: np.ndarray = np.sqrt(gx * gx + gy * gy)

    # Compute radial/tangential alignment
    rr_safe: np.ndarray = np.where(rr > 0.5, rr, 1.0)
    radial_x: np.ndarray = -dx / rr_safe
    radial_y: np.ndarray = -dy / rr_safe

    grad_mag_safe: np.ndarray = np.where(grad_mag > 1.0, grad_mag, 1.0)
    gx_n: np.ndarray = gx / grad_mag_safe
    gy_n: np.ndarray = gy / grad_mag_safe
    tangential_weight: np.ndarray = np.abs(gx_n * radial_y - gy_n * radial_x)

    # Dark contrast: local mean minus center pixel
    dark_contrast: np.ndarray = np.zeros_like(gray)
    for i in range(1, h_img - 1):
        for j in range(1, w_img - 1):
            if inner_mask[i, j] and saturation_mask[i, j]:
                neighborhood = gray[i-1:i+2, j-1:j+2].ravel()
                dark_contrast[i, j] = np.mean(neighborhood) - gray[i, j]

    # Combined vote weight
    vote_mask: np.ndarray = inner_mask & saturation_mask & (grad_mag > 8.0)
    vote_weight: np.ndarray = np.where(
        vote_mask,
        grad_mag * tangential_weight * np.clip(dark_contrast / 50.0, 0.0, 1.0),
        0.0,
    )

    # Accumulate into 360 angle bins (1Â° resolution)
    spoke_angle: np.ndarray = np.arctan2(dy, dx)
    num_bins: int = 360
    angle_bins: np.ndarray = (
        (spoke_angle + math.pi) / (2.0 * math.pi) * num_bins
    ).astype(np.int32)
    angle_bins = np.clip(angle_bins, 0, num_bins - 1)

    histogram: np.ndarray = np.zeros(num_bins, dtype=np.float32)
    np.add.at(histogram, angle_bins.ravel(), vote_weight.ravel())

    # Apply gauge sweep mask if spec provided
    if gauge_spec is not None:
        for bin_index in range(num_bins):
            angle_rad: float = (bin_index / num_bins) * 2.0 * math.pi - math.pi
            if not _angle_in_sweep(angle_rad, gauge_spec, margin_rad=math.radians(6.0)):
                histogram[bin_index] = 0.0

    # Simple smoothing
    histogram_smooth: np.ndarray = cv2.GaussianBlur(
        histogram[np.newaxis, :],
        (1, 15),
        0,
    ).ravel()

    best_bin: int = int(np.argmax(histogram_smooth))
    peak_val: float = float(histogram_smooth[best_bin])
    mean_val: float = float(np.mean(histogram)) + 1e-6

    # Confidence = peak / mean (matches C code)
    confidence: float = peak_val / mean_val

    # Find runner-up (simple: max of rest)
    rolled: np.ndarray = np.roll(histogram_smooth, -best_bin)
    runner_up: float = float(np.max(rolled[1:20]))

    peak_ratio: float = peak_val / max(runner_up, 1e-6)
    best_angle: float = (best_bin / num_bins) * 2.0 * math.pi - math.pi

    # Apply confidence gates (matches C thresholds)
    if confidence < 1.25 or peak_val < 75.0 or peak_ratio < 1.05:
        return None

    if gauge_spec is not None and not _angle_in_sweep(
        best_angle, gauge_spec, margin_rad=math.radians(6.0)
    ):
        return None

    return NeedleDetection(
        unit_dx=float(math.cos(best_angle)),
        unit_dy=float(math.sin(best_angle)),
        confidence=float(confidence),
        peak_value=float(peak_val),
        runner_up_value=float(runner_up),
        peak_ratio=float(peak_ratio),
        peak_margin=float(peak_val - runner_up),
    )


def _point_to_segment_distance(
    point_x: float,
    point_y: float,
    seg_x1: float,
    seg_y1: float,
    seg_x2: float,
    seg_y2: float,
) -> float:
    """Return the shortest distance from a point to a finite line segment."""
    seg_dx: float = seg_x2 - seg_x1
    seg_dy: float = seg_y2 - seg_y1
    seg_len_sq: float = seg_dx * seg_dx + seg_dy * seg_dy
    if seg_len_sq <= 1e-12:
        return math.hypot(point_x - seg_x1, point_y - seg_y1)

    # Project the point onto the segment and clamp the result to the segment.
    t: float = ((point_x - seg_x1) * seg_dx + (point_y - seg_y1) * seg_dy) / seg_len_sq
    t = min(max(t, 0.0), 1.0)
    proj_x: float = seg_x1 + t * seg_dx
    proj_y: float = seg_y1 + t * seg_dy
    return math.hypot(point_x - proj_x, point_y - proj_y)


def _is_in_lower_subdial_region(
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    dial_radius_px: float,
) -> bool:
    """Return True when a point falls into the lower subdial clutter region."""
    dx: float = abs(x - center_x)
    dy: float = y - center_y
    return (
        dx < 0.36 * dial_radius_px
        and dy > 0.10 * dial_radius_px
        and dy < 0.60 * dial_radius_px
    )


def _sample_line_darkness(
    image: np.ndarray,
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    center_xy: tuple[float, float] | None = None,
    dial_radius_px: float | None = None,
) -> tuple[float, float]:
    """Estimate how needle-like a line segment is from local darkness.

    A true needle usually reads as a thin dark stroke with darker pixels along
    the segment than in the immediate neighborhood on either side.

    When a color image is available we also reward low saturation so the dark
    neutral needle beats colorful dial markings.
    """
    if image.ndim == 2:
        gray_image = image
        sat_image: np.ndarray | None = None
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat_image = hsv_image[:, :, 1].astype(np.float32)

    seg_dx: float = x2 - x1
    seg_dy: float = y2 - y1
    seg_len: float = math.hypot(seg_dx, seg_dy)
    if seg_len <= 1e-6:
        return 0.0, 0.0

    unit_x: float = seg_dx / seg_len
    unit_y: float = seg_dy / seg_len
    perp_x: float = -unit_y
    perp_y: float = unit_x

    sample_contrasts: list[float] = []
    dark_hits: int = 0
    sample_count: int = 0
    weighted_contrast_sum: float = 0.0
    weighted_dark_hits: float = 0.0
    weight_sum: float = 0.0

    # Avoid the exact endpoints so the hub and rim do not dominate the score.
    # The middle shaft is the least ambiguous part of the needle, so we bias
    # the score toward the central band instead of the outer dial ring.
    sample_fractions: np.ndarray = np.linspace(0.44, 0.78, 7, dtype=np.float32)
    for fraction in sample_fractions:
        sample_count += 1
        sample_weight: float = float(
            math.exp(-0.5 * ((float(fraction) - 0.60) / 0.07) ** 2)
        )
        sample_x: float = x1 + float(fraction) * seg_dx
        sample_y: float = y1 + float(fraction) * seg_dy
        ix: int = int(round(min(max(sample_x, 0.0), gray_image.shape[1] - 1.0)))
        iy: int = int(round(min(max(sample_y, 0.0), gray_image.shape[0] - 1.0)))
        line_px: float = float(gray_image[iy, ix])
        line_sat: float = float(sat_image[iy, ix]) if sat_image is not None else line_px

        neighbor_values: list[float] = []
        neighbor_sats: list[float] = []
        for offset_px in (2.0, 4.0):
            for direction in (-1.0, 1.0):
                nx: float = sample_x + direction * offset_px * perp_x
                ny: float = sample_y + direction * offset_px * perp_y
                ix: int = int(round(min(max(nx, 0.0), gray_image.shape[1] - 1.0)))
                iy: int = int(round(min(max(ny, 0.0), gray_image.shape[0] - 1.0)))
                neighbor_values.append(float(gray_image[iy, ix]))
                if sat_image is not None:
                    neighbor_sats.append(float(sat_image[iy, ix]))

        if not neighbor_values:
            continue

        local_mean: float = float(np.mean(neighbor_values))
        if sat_image is not None and neighbor_sats:
            local_sat_mean: float = float(np.mean(neighbor_sats))
            saturation_bonus: float = max(local_sat_mean - line_sat, 0.0) / 255.0
        else:
            saturation_bonus = 0.0

        contrast: float = (local_mean - line_px) / 255.0 + 0.60 * saturation_bonus
        sample_contrasts.append(contrast)
        weighted_contrast_sum += sample_weight * contrast
        weight_sum += sample_weight
        if (line_px + 2.0 < local_mean) and (saturation_bonus >= 0.0):
            dark_hits += 1
            weighted_dark_hits += sample_weight

    if not sample_contrasts or sample_count == 0:
        return 0.0, 0.0

    if weight_sum <= 1e-6:
        return 0.0, 0.0

    contrast_mean: float = weighted_contrast_sum / weight_sum
    dark_fraction: float = weighted_dark_hits / weight_sum
    return contrast_mean, dark_fraction


def _angle_in_sweep(
    angle_rad: float, spec: GaugeSpec, *, margin_rad: float = 0.0
) -> bool:
    """Return True if angle_rad falls within the gauge's calibrated sweep arc.

    An optional margin widens the arc on each side to tolerate small detection
    offsets near the min/max ticks.
    """
    shifted: float = (angle_rad - spec.min_angle_rad) % (2.0 * math.pi)
    return shifted <= (spec.sweep_rad + margin_rad)


def _detect_needle_unit_vector_spoke_improved(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec | None = None,
    snr_threshold: float = 2.0,
) -> NeedleDetection | None:
    """Detect the needle with a tightened spoke vote and subdial suppression."""
    if dial_radius_px <= 1.0:
        return None

    center_x, center_y = center_xy

    gray: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced: np.ndarray = clahe.apply(gray)
    blurred: np.ndarray = cv2.GaussianBlur(enhanced, (5, 5), 0)
    hsv: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation: np.ndarray = hsv[:, :, 1].astype(np.float32) / 255.0

    h_img, w_img = gray.shape[:2]
    yy, xx = np.meshgrid(
        np.arange(h_img, dtype=np.float32),
        np.arange(w_img, dtype=np.float32),
        indexing="ij",
    )
    dx: np.ndarray = xx - center_x
    dy: np.ndarray = yy - center_y
    rr: np.ndarray = np.sqrt(dx**2 + dy**2)

    # Keep the inner annulus and suppress the lower-center humidity subdial.
    # The middle shaft is usually the least ambiguous part of the needle, so we
    # bias the vote toward that band instead of letting the outer dial edge win.
    inner_mask: np.ndarray = (rr > 0.30 * dial_radius_px) & (rr < 0.70 * dial_radius_px)
    gx: np.ndarray = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy: np.ndarray = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag: np.ndarray = np.sqrt(gx * gx + gy * gy)

    rr_safe: np.ndarray = np.where(rr > 0.5, rr, 1.0)
    radial_x: np.ndarray = -dx / rr_safe
    radial_y: np.ndarray = -dy / rr_safe

    grad_mag_safe: np.ndarray = np.where(grad_mag > 1.0, grad_mag, 1.0)
    gx_n: np.ndarray = gx / grad_mag_safe
    gy_n: np.ndarray = gy / grad_mag_safe
    tangential_weight: np.ndarray = np.abs(gx_n * radial_y - gy_n * radial_x)
    shaft_weight: np.ndarray = _middle_shaft_weight(rr, dial_radius_px)
    neutral_weight: np.ndarray = np.clip(1.0 - saturation, 0.0, 1.0)

    vote_weight: np.ndarray = np.where(
        inner_mask & (grad_mag > 8.0),
        grad_mag * tangential_weight * shaft_weight * neutral_weight,
        0.0,
    )

    spoke_angle: np.ndarray = np.arctan2(dy, dx)
    num_bins: int = 720
    angle_bins: np.ndarray = (
        (spoke_angle + math.pi) / (2.0 * math.pi) * num_bins
    ).astype(np.int32)
    angle_bins = np.clip(angle_bins, 0, num_bins - 1)

    histogram: np.ndarray = np.zeros(num_bins, dtype=np.float32)
    np.add.at(histogram, angle_bins.ravel(), vote_weight.ravel())

    if gauge_spec is not None:
        for bin_index in range(num_bins):
            angle_rad: float = (bin_index / num_bins) * 2.0 * math.pi - math.pi
            if not _angle_in_sweep(
                angle_rad,
                gauge_spec,
                margin_rad=math.radians(6.0),
            ):
                histogram[bin_index] = 0.0

    kernel_width: int = 15
    histogram_smooth: np.ndarray = cv2.GaussianBlur(
        histogram[np.newaxis, :],
        (1, kernel_width * 2 + 1),
        0,
    ).ravel()

    best_bin: int = int(np.argmax(histogram_smooth))
    peak_val: float = float(histogram_smooth[best_bin])
    noise: float = float(np.mean(histogram)) + 1e-6
    snr: float = peak_val / noise
    if snr < snr_threshold:
        return None

    runner_up: float = _runner_up_peak_after_suppression(
        histogram_smooth,
        best_index=best_bin,
        suppression_bins=kernel_width,
    )
    peak_ratio: float = peak_val / max(runner_up, 1e-6)
    best_angle: float = (best_bin / num_bins) * 2.0 * math.pi - math.pi

    if gauge_spec is not None and not _angle_in_sweep(
        best_angle,
        gauge_spec,
        margin_rad=math.radians(6.0),
    ):
        return None

    return NeedleDetection(
        unit_dx=float(math.cos(best_angle)),
        unit_dy=float(math.sin(best_angle)),
        confidence=float(snr),
        peak_value=float(peak_val),
        runner_up_value=float(runner_up),
        peak_ratio=float(peak_ratio),
        peak_margin=float(peak_val - runner_up),
    )


def _detect_needle_unit_vector_center_weighted(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec | None = None,
) -> NeedleDetection | None:
    """Detect the needle with a center-weighted dark-arc accumulator."""
    if dial_radius_px <= 1.0:
        return None

    center_x, center_y = center_xy

    gray: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced: np.ndarray = clahe.apply(gray)
    blurred: np.ndarray = cv2.GaussianBlur(enhanced, (5, 5), 0)
    hsv: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation: np.ndarray = hsv[:, :, 1].astype(np.float32) / 255.0

    h_img, w_img = gray.shape[:2]
    yy, xx = np.meshgrid(
        np.arange(h_img, dtype=np.float32),
        np.arange(w_img, dtype=np.float32),
        indexing="ij",
    )
    dx: np.ndarray = xx - center_x
    dy: np.ndarray = yy - center_y
    rr: np.ndarray = np.sqrt(dx**2 + dy**2)

    inner_mask: np.ndarray = (rr > 0.22 * dial_radius_px) & (rr < 0.62 * dial_radius_px)
    rr_safe: np.ndarray = np.where(rr > 0.5, rr, 1.0)
    radial_x: np.ndarray = -dx / rr_safe
    radial_y: np.ndarray = -dy / rr_safe

    gx: np.ndarray = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy: np.ndarray = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag: np.ndarray = np.sqrt(gx * gx + gy * gy)
    grad_mag_safe: np.ndarray = np.where(grad_mag > 1.0, grad_mag, 1.0)
    gx_n: np.ndarray = gx / grad_mag_safe
    gy_n: np.ndarray = gy / grad_mag_safe
    tangential_weight: np.ndarray = np.abs(gx_n * radial_y - gy_n * radial_x)
    shaft_weight: np.ndarray = _middle_shaft_weight(rr, dial_radius_px)
    neutral_weight: np.ndarray = np.clip(1.0 - saturation, 0.0, 1.0)

    # Keep a mild preference for the shaft moving away from the hub, but focus
    # the vote on the middle band instead of the outer dial ring.
    center_weight: np.ndarray = np.where(
        inner_mask,
        np.clip(1.0 - rr / (0.72 * dial_radius_px), 0.0, 1.0),
        0.0,
    )
    center_weight = center_weight * shaft_weight

    dark_mask: np.ndarray = (blurred < 130) & inner_mask
    vote_weight: np.ndarray = np.where(
        dark_mask,
        center_weight * tangential_weight * shaft_weight * neutral_weight * 50.0,
        0.0,
    )

    spoke_angle: np.ndarray = np.arctan2(dy, dx)
    num_bins: int = 720
    angle_bins: np.ndarray = (
        (spoke_angle + math.pi) / (2.0 * math.pi) * num_bins
    ).astype(np.int32)
    angle_bins = np.clip(angle_bins, 0, num_bins - 1)

    histogram: np.ndarray = np.zeros(num_bins, dtype=np.float32)
    np.add.at(histogram, angle_bins.ravel(), vote_weight.ravel())

    if gauge_spec is not None:
        for bin_index in range(num_bins):
            angle_rad: float = (bin_index / num_bins) * 2.0 * math.pi - math.pi
            if not _angle_in_sweep(
                angle_rad,
                gauge_spec,
                margin_rad=math.radians(6.0),
            ):
                histogram[bin_index] = 0.0

    histogram_smooth: np.ndarray = cv2.GaussianBlur(
        histogram[np.newaxis, :],
        (1, 21),
        0,
    ).ravel()

    best_bin: int = int(np.argmax(histogram_smooth))
    peak_val: float = float(histogram_smooth[best_bin])
    runner_up: float = float(np.max(np.roll(histogram_smooth, -best_bin)[1:20]))
    peak_ratio: float = peak_val / max(runner_up, 1e-6)
    noise: float = float(np.mean(histogram_smooth)) + 1e-6
    snr: float = peak_val / noise
    if snr < 1.15 or peak_ratio < 1.03:
        return None

    best_angle: float = (best_bin / num_bins) * 2.0 * math.pi - math.pi
    if gauge_spec is not None and not _angle_in_sweep(
        best_angle,
        gauge_spec,
        margin_rad=math.radians(6.0),
    ):
        return None

    return NeedleDetection(
        unit_dx=float(math.cos(best_angle)),
        unit_dy=float(math.sin(best_angle)),
        confidence=float(snr),
        peak_value=float(peak_val),
        runner_up_value=float(runner_up),
        peak_ratio=float(peak_ratio),
        peak_margin=float(peak_val - runner_up),
    )


def _detect_needle_unit_vector_hough_lines(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec | None = None,
) -> NeedleDetection | None:
    """Detect the needle with a probabilistic Hough line search.

    The spoke vote is good at finding radial dark structure, but it can still
    lock onto the dial markings. This detector looks for a long dark line
    segment that passes near the gauge center and reaches far enough toward the
    rim to look like a needle rather than a tick mark.
    """
    if dial_radius_px <= 1.0:
        return None

    gray: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced: np.ndarray = clahe.apply(gray)
    blurred: np.ndarray = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges: np.ndarray = cv2.Canny(blurred, 40, 120, apertureSize=3, L2gradient=True)

    min_line_length: int = max(18, int(round(0.24 * dial_radius_px)))
    max_line_gap: int = max(4, int(round(0.08 * dial_radius_px)))
    threshold: int = max(14, int(round(0.10 * dial_radius_px)))
    lines: np.ndarray | None = cv2.HoughLinesP(
        edges,
        1,
        math.pi / 180.0,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return None

    center_x, center_y = center_xy
    best_score: float = float("-inf")
    second_score: float = float("-inf")
    best_vec: tuple[float, float] | None = None

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, line)
        seg_dx: float = x2 - x1
        seg_dy: float = y2 - y1
        seg_len: float = math.hypot(seg_dx, seg_dy)
        if seg_len < min_line_length:
            continue

        center_dist: float = _point_to_segment_distance(
            center_x, center_y, x1, y1, x2, y2
        )
        if center_dist > 0.24 * dial_radius_px:
            continue

        d1: float = math.hypot(x1 - center_x, y1 - center_y)
        d2: float = math.hypot(x2 - center_x, y2 - center_y)
        near: float = min(d1, d2)
        far: float = max(d1, d2)
        if near > 0.36 * dial_radius_px or far < 0.50 * dial_radius_px:
            continue

        far_x, far_y = (x1, y1) if d1 > d2 else (x2, y2)
        angle_rad: float = math.atan2(far_y - center_y, far_x - center_x)
        if gauge_spec is not None and not _angle_in_sweep(
            angle_rad,
            gauge_spec,
            margin_rad=math.radians(16.0),
        ):
            continue

        line_contrast, dark_fraction = _sample_line_darkness(
            image_bgr,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            center_xy=center_xy,
            dial_radius_px=dial_radius_px,
        )
        darkness: float = max(0.0, line_contrast) + 0.60 * max(0.0, dark_fraction)
        if darkness <= 0.0:
            continue

        radiality: float = 1.0 - min(center_dist / max(0.24 * dial_radius_px, 1.0), 1.0)
        reach: float = min(far / max(0.95 * dial_radius_px, 1.0), 1.0)
        shaft_balance: float = 1.0 - min(
            abs(near - 0.15 * dial_radius_px) / max(0.25 * dial_radius_px, 1.0),
            1.0,
        )
        score: float = (
            seg_len
            * darkness
            * (0.35 + 0.65 * radiality)
            * (0.35 + 0.65 * reach)
            * (0.35 + 0.65 * shaft_balance)
        )

        if score > best_score:
            second_score = best_score
            best_score = score
            unit_len = max(far, 1e-6)
            best_vec = ((far_x - center_x) / unit_len, (far_y - center_y) / unit_len)
        elif score > second_score:
            second_score = score

    if best_vec is None:
        return None

    confidence: float = (
        best_score / max(second_score, 1e-6) if second_score > 0.0 else best_score
    )
    if confidence < 1.08 or best_score < 2.0:
        return None

    return NeedleDetection(
        unit_dx=best_vec[0],
        unit_dy=best_vec[1],
        confidence=float(confidence),
        peak_value=float(best_score),
        runner_up_value=float(second_score if second_score > 0.0 else 0.0),
        peak_ratio=float(
            best_score / max(second_score, 1e-6) if second_score > 0.0 else 1.0
        ),
        peak_margin=float(
            best_score - second_score if second_score > 0.0 else best_score
        ),
    )


def _detect_needle_unit_vector_line_segment(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec | None = None,
) -> NeedleDetection | None:
    """Detect the needle as a dark line segment from hub to tip.

    This is the primary detector. It finds line segments with HoughLinesP,
    scores them by length + darkness + center proximity, and determines
    tip direction unambiguously: the endpoint farther from the center is
    the tip; the endpoint near the bright hub is the tail.

    This eliminates needle inversion because a line segment has two distinct
    endpoints, unlike histogram voting which only knows an angle.
    """
    if dial_radius_px <= 1.0:
        return None

    center_x, center_y = center_xy

    # Preprocess: grayscale, CLAHE, blur, Canny edges.
    gray: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced: np.ndarray = clahe.apply(gray)
    blurred: np.ndarray = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges: np.ndarray = cv2.Canny(blurred, 30, 100, apertureSize=3, L2gradient=True)

    # Find line segments. Use relatively permissive parameters so we catch
    # the needle even when it is faint.
    min_line_length: int = max(15, int(round(0.20 * dial_radius_px)))
    max_line_gap: int = max(6, int(round(0.12 * dial_radius_px)))
    threshold: int = max(10, int(round(0.08 * dial_radius_px)))
    lines: np.ndarray | None = cv2.HoughLinesP(
        edges,
        1,
        math.pi / 180.0,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return None

    best_score: float = float("-inf")
    second_score: float = float("-inf")
    best_tip_vec: tuple[float, float] | None = None

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, line)
        seg_len: float = math.hypot(x2 - x1, y2 - y1)
        if seg_len < min_line_length:
            continue

        # Distance from center to the line segment.
        center_dist: float = _point_to_segment_distance(
            center_x, center_y, x1, y1, x2, y2
        )
        # Needle must pass near the hub.
        if center_dist > 0.30 * dial_radius_px:
            continue

        # Distances from center to each endpoint.
        d1: float = math.hypot(x1 - center_x, y1 - center_y)
        d2: float = math.hypot(x2 - center_x, y2 - center_y)
        near: float = min(d1, d2)
        far: float = max(d1, d2)

        # One end must be near the hub (but not exactly at center), the other
        # must reach toward the rim.
        if near > 0.30 * dial_radius_px or far < 0.45 * dial_radius_px:
            continue

        # Determine tip vs tail: the farther endpoint is the tip.
        tip_x, tip_y = (x1, y1) if d1 > d2 else (x2, y2)
        tail_x, tail_y = (x2, y2) if d1 > d2 else (x1, y1)

        # Compute tip angle from center.
        tip_dx: float = tip_x - center_x
        tip_dy: float = tip_y - center_y
        tip_angle: float = math.atan2(tip_dy, tip_dx)

        # Reject if outside the gauge sweep.
        if gauge_spec is not None and not _angle_in_sweep(
            tip_angle,
            gauge_spec,
            margin_rad=math.radians(8.0),
        ):
            continue

        # Score darkness along the segment.
        line_contrast, dark_fraction = _sample_line_darkness(
            image_bgr,
            x1=tail_x,
            y1=tail_y,
            x2=tip_x,
            y2=tip_y,
            center_xy=center_xy,
            dial_radius_px=dial_radius_px,
        )
        darkness: float = max(0.0, line_contrast) + 0.60 * max(0.0, dark_fraction)
        if darkness <= 0.0:
            continue

        # Score how radial the segment is (needle should be roughly radial).
        # Project segment direction onto radial direction at midpoint.
        mid_x: float = (x1 + x2) / 2.0
        mid_y: float = (y1 + y2) / 2.0
        mid_dx: float = mid_x - center_x
        mid_dy: float = mid_y - center_y
        mid_dist: float = math.hypot(mid_dx, mid_dy)
        if mid_dist < 1.0:
            continue
        radial_unit_x: float = mid_dx / mid_dist
        radial_unit_y: float = mid_dy / mid_dist
        seg_unit_x: float = (x2 - x1) / seg_len
        seg_unit_y: float = (y2 - y1) / seg_len
        # Dot product of segment direction with radial direction.
        radiality: float = abs(radial_unit_x * seg_unit_x + radial_unit_y * seg_unit_y)
        if radiality < 0.60:
            continue

        # Composite score: length * darkness * radiality.
        # Length matters most â€” the needle is the longest dark feature.
        score: float = seg_len * darkness * (0.3 + 0.7 * radiality)

        if score > best_score:
            second_score = best_score
            best_score = score
            tip_dist = max(far, 1e-6)
            best_tip_vec = (tip_dx / tip_dist, tip_dy / tip_dist)
        elif score > second_score:
            second_score = score

    if best_tip_vec is None:
        return None

    confidence: float = (
        best_score / max(second_score, 1e-6) if second_score > 0.0 else best_score
    )
    if confidence < 1.05 or best_score < 1.5:
        return None

    return NeedleDetection(
        unit_dx=best_tip_vec[0],
        unit_dy=best_tip_vec[1],
        confidence=float(confidence),
        peak_value=float(best_score),
        runner_up_value=float(second_score if second_score > 0.0 else 0.0),
        peak_ratio=float(
            best_score / max(second_score, 1e-6) if second_score > 0.0 else 1.0
        ),
        peak_margin=float(
            best_score - second_score if second_score > 0.0 else best_score
        ),
    )


def _detect_needle_unit_vector_combined(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec | None = None,
) -> NeedleDetection | None:
    """Run the stable radial detectors and keep the best result.

    The line-segment and Hough-line paths remain in the module as experiments,
    but they were too eager to win on clean captures and masked the better
    spoke/center-weighted votes.
    """
    candidates: list[NeedleDetection] = []

    # Primary: line segment detector â€” no inversion ambiguity.
    # Favor the middle-shaft detectors first because they are the most stable
    # on the bright, clean gauge photos we use as the thesis baseline.
    spoke_detection: NeedleDetection | None = _detect_needle_unit_vector_spoke_improved(
        image_bgr,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
        gauge_spec=gauge_spec,
    )
    if spoke_detection is not None:
        candidates.append(spoke_detection)

    center_detection: NeedleDetection | None = (
        _detect_needle_unit_vector_center_weighted(
            image_bgr,
            center_xy=center_xy,
            dial_radius_px=dial_radius_px,
            gauge_spec=gauge_spec,
        )
    )
    if center_detection is not None:
        candidates.append(center_detection)

    if not candidates:
        return None

    return max(candidates, key=needle_detection_quality)



def _detect_needle_unit_vector_shaft_scan(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec | None = None,
) -> NeedleDetection | None:
    """Alias for combined radial detection (backward compatibility)."""
    return _detect_needle_unit_vector_combined(
        image_bgr,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
        gauge_spec=gauge_spec,
    )

def _select_best_detection(
    image_bgr: np.ndarray,
    hypotheses: list[CenterHypothesis],
    gauge_spec: GaugeSpec,
) -> tuple[CenterHypothesis, NeedleDetection] | None:
    """Select the best needle detection across all hypotheses (matches C baseline).

    For each hypothesis, run the polar vote detector and track the best peak.
    The hypothesis with the sharpest peak (highest confidence) wins.
    """
    best_hypothesis: CenterHypothesis | None = None
    best_detection: NeedleDetection | None = None
    best_confidence: float = 0.0

    for hypothesis in hypotheses:
        detection: NeedleDetection | None = _detect_needle_unit_vector_polar(
            image_bgr,
            center_xy=hypothesis.center_xy,
            dial_radius_px=hypothesis.dial_radius_px,
            gauge_spec=gauge_spec,
        )

        if detection is None:
            continue

        # Select based on confidence (peak / mean)
        if detection.confidence > best_confidence:
            best_hypothesis = hypothesis
            best_detection = detection
            best_confidence = detection.confidence

    if best_hypothesis is None or best_detection is None:
        return None

    return best_hypothesis, best_detection


def detect_needle_unit_vector(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec | None = None,
) -> NeedleDetection | None:
    """Detect needle direction using the polar voting baseline (matches C code).

    This is the main entry point for the classical baseline. It uses the
    simplified polar voting detector that mirrors the embedded C implementation.
    """
    return _detect_needle_unit_vector_polar(
        image_bgr,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
        gauge_spec=gauge_spec,
    )


def run_classical_baseline(
    image_bgr: np.ndarray,
    gauge_spec: GaugeSpec,
) -> tuple[float, float, float, NeedleDetection | None]:
    """Run the classical baseline and return (center_x, center_y, radius, detection).

    This matches the C baseline flow:
    1. Generate center hypotheses (bright centroid, fixed crop, image center)
    2. Run polar voting for each hypothesis
    3. Select the hypothesis with the highest confidence peak
    4. Return the winning geometry and detection
    """
    hypotheses: list[CenterHypothesis] = _center_hypotheses(image_bgr)
    result: tuple[CenterHypothesis, NeedleDetection] | None = _select_best_detection(
        image_bgr, hypotheses, gauge_spec
    )

    if result is None:
        # Fallback to image center with no detection
        fallback: CenterHypothesis = _image_center_hypothesis(image_bgr)
        return fallback.center_xy[0], fallback.center_xy[1], fallback.dial_radius_px, None

    hypothesis, detection = result
    return hypothesis.center_xy[0], hypothesis.center_xy[1], hypothesis.dial_radius_px, detection


def detect_needle_unit_vector_with_geometry_fallback(
    image_bgr: np.ndarray,
    *,
    primary: GeometryCandidate,
    secondary: GeometryCandidate | None = None,
    gauge_spec: GaugeSpec | None = None,
    confidence_threshold: float = 4.0,
) -> NeedleDetection | None:
    """Run the needle detector with a conservative geometry fallback.

    The Hough-circle estimate is still the preferred geometry. When it produces
    a weak spoke vote, we try a simpler center-based fallback before giving up.
    This keeps the hard-case script resilient without changing the core detector
    logic for callers that already provide trustworthy geometry.
    """
    primary_detection: NeedleDetection | None = detect_needle_unit_vector(
        image_bgr,
        center_xy=primary.center_xy,
        dial_radius_px=primary.dial_radius_px,
        gauge_spec=gauge_spec,
    )
    if primary_detection is None:
        if secondary is None:
            return None
        return detect_needle_unit_vector(
            image_bgr,
            center_xy=secondary.center_xy,
            dial_radius_px=secondary.dial_radius_px,
            gauge_spec=gauge_spec,
        )

    if secondary is None:
        return primary_detection

    secondary_detection: NeedleDetection | None = detect_needle_unit_vector(
        image_bgr,
        center_xy=secondary.center_xy,
        dial_radius_px=secondary.dial_radius_px,
        gauge_spec=gauge_spec,
    )
    if secondary_detection is None:
        return primary_detection

    # If the primary geometry is only marginally separated from its runner-up,
    # allow a sharper secondary hypothesis to take over even when raw quality is
    # close. This keeps weak Hough fits from pinning the result to a bad crop.
    if (
        primary_detection.peak_ratio < 1.10
        and secondary_detection.peak_ratio > primary_detection.peak_ratio
    ):
        return secondary_detection

    # The old gate preferred the higher-confidence geometry. The new rule keeps
    # that preference only when it also has the stronger peak separation.
    primary_quality: float = needle_detection_quality(primary_detection)
    secondary_quality: float = needle_detection_quality(secondary_detection)
    if secondary_quality > primary_quality:
        return secondary_detection
    return primary_detection


def select_best_geometry_detection(
    image_bgr: np.ndarray,
    *,
    candidates: Sequence[GeometryCandidate],
    gauge_spec: GaugeSpec | None = None,
    detectors: (
        Sequence[
            Callable[
                [np.ndarray, tuple[float, float], float, GaugeSpec | None],
                NeedleDetection | None,
            ]
        ]
        | None
    ) = None,
) -> GeometrySelection | None:
    """Run the detector over several geometry hypotheses and keep the best one.

    The default is still the highest-quality detector, but if several geometry
    hypotheses land within a few degrees of each other, we prefer that small
    agreement cluster over a lone high-score outlier.
    """
    detector_fns: Sequence[
        Callable[
            [np.ndarray, tuple[float, float], float, GaugeSpec | None],
            NeedleDetection | None,
        ]
    ] = (
        detectors if detectors is not None else (detect_needle_unit_vector,)
    )
    best_selection: GeometrySelection | None = None
    best_index: int = -1
    selections: list[GeometrySelection] = []
    for candidate in candidates:
        for detector_fn in detector_fns:
            detection: NeedleDetection | None = detector_fn(
                image_bgr,
                center_xy=candidate.center_xy,
                dial_radius_px=candidate.dial_radius_px,
                gauge_spec=gauge_spec,
            )
            if detection is None:
                continue

            raw_quality: float = needle_detection_quality(detection)
            line_contrast, dark_fraction = _sample_line_darkness(
                image_bgr,
                x1=candidate.center_xy[0]
                + detection.unit_dx * (0.30 * candidate.dial_radius_px),
                y1=candidate.center_xy[1]
                + detection.unit_dy * (0.30 * candidate.dial_radius_px),
                x2=candidate.center_xy[0]
                # Look a little farther toward the tip so the middle shaft,
                # not the dial rim or nearby markings, drives the score.
                + detection.unit_dx * (0.78 * candidate.dial_radius_px),
                y2=candidate.center_xy[1]
                + detection.unit_dy * (0.78 * candidate.dial_radius_px),
                center_xy=candidate.center_xy,
                dial_radius_px=candidate.dial_radius_px,
            )
            line_contrast = float(line_contrast)
            dark_fraction = float(dark_fraction)
            shaft_support: float = max(line_contrast, 0.0) + 0.90 * dark_fraction
            quality: float = raw_quality * (0.15 + shaft_support)
            selection = GeometrySelection(
                candidate=candidate,
                detection=detection,
                quality=quality,
                shaft_support=shaft_support,
            )
            selections.append(selection)
            if best_selection is None:
                best_selection = selection
                best_index = len(selections) - 1
                continue

            if _geometry_selection_key(selection) > _geometry_selection_key(
                best_selection
            ):
                best_selection = selection
                best_index = len(selections) - 1

    if (best_selection is None) or (gauge_spec is None) or (len(selections) < 2):
        return best_selection

    best_selection_quality: float = best_selection.quality

    # Count how many other geometry hypotheses land near each candidate.
    predicted_values: list[float] = [
        needle_vector_to_value(
            selection.detection.unit_dx,
            selection.detection.unit_dy,
            gauge_spec,
        )
        for selection in selections
    ]
    support_counts: list[int] = []
    best_support: int = 0
    for i, predicted_value in enumerate(predicted_values):
        support: int = 1
        for j, other_value in enumerate(predicted_values):
            if i == j:
                continue
            if abs(predicted_value - other_value) <= CONSENSUS_TEMP_DELTA_C:
                support += 1
        support_counts.append(support)
        if support > best_support:
            best_support = support

    if best_support < 2:
        return best_selection

    # Keep the strongest candidate inside the best-agreement cluster.
    consensus_indexes = [
        index for index, support in enumerate(support_counts) if support == best_support
    ]
    consensus_index = max(
        consensus_indexes,
        key=lambda index: _geometry_selection_key(selections[index]),
    )
    if consensus_index == best_index:
        return best_selection

    consensus_selection: GeometrySelection = selections[consensus_index]
    if consensus_selection.quality < (
        best_selection_quality * CONSENSUS_MIN_QUALITY_RATIO
    ):
        return best_selection

    return consensus_selection


def evaluate_classical_baseline(
    samples: Iterable[Sample],
    spec: GaugeSpec,
    *,
    max_samples: int | None = None,
) -> ClassicalBaselineResult:
    """Evaluate the classical-CV baseline against labelled samples.

    Args:
        samples: Iterable of labelled samples.
        spec: Gauge calibration for angle-to-value conversion.
        max_samples: Optional cap for faster benchmark loops.

    Returns:
        Aggregate result with MAE/RMSE computed over successful detections.
    """
    predictions: list[ClassicalPrediction] = []
    attempted: int = 0

    for sample in samples:
        if max_samples is not None and attempted >= max_samples:
            break
        attempted += 1

        image_bgr: np.ndarray | None = cv2.imread(
            str(sample.image_path), cv2.IMREAD_COLOR
        )
        if image_bgr is None:
            continue

        # Use a slightly shrunken version of the smaller ellipse axis as the
        # effective dial radius. This keeps the annulus away from border clutter
        # when the label ellipse is stretched by perspective.
        dial_radius_px: float = 0.95 * min(sample.dial.rx, sample.dial.ry)
        detection: NeedleDetection | None = detect_needle_unit_vector(
            image_bgr,
            center_xy=(sample.center.x, sample.center.y),
            dial_radius_px=dial_radius_px,
            gauge_spec=spec,
        )
        if detection is None:
            continue

        true_value: float = needle_value(sample, spec, strict=False)
        predicted_value: float = needle_vector_to_value(
            detection.unit_dx,
            detection.unit_dy,
            spec,
        )
        abs_error: float = abs(predicted_value - true_value)

        predictions.append(
            ClassicalPrediction(
                image_path=str(sample.image_path),
                true_value=true_value,
                predicted_value=predicted_value,
                abs_error=abs_error,
                confidence=detection.confidence,
            )
        )

    successful: int = len(predictions)
    failed: int = attempted - successful

    if successful == 0:
        return ClassicalBaselineResult(
            attempted_samples=attempted,
            successful_samples=0,
            failed_samples=failed,
            mae=float("nan"),
            rmse=float("nan"),
            predictions=[],
        )

    errors: np.ndarray = np.array([p.abs_error for p in predictions], dtype=np.float32)
    mae: float = float(np.mean(errors))
    rmse: float = float(np.sqrt(np.mean(np.square(errors))))

    return ClassicalBaselineResult(
        attempted_samples=attempted,
        successful_samples=successful,
        failed_samples=failed,
        mae=mae,
        rmse=rmse,
        predictions=predictions,
    )


# ── Firmware-matched classical baseline ──────────────────────────────────
# These constants match the STM32N6 firmware's app_baseline_runtime.c,
# mapped into 224x224 crop-space coordinates.

_FW_DIAL_RADIUS_PX: float = 0.56 * 224.0  # ~125 px
_FW_BRIGHT_THRESHOLD: int = 150
_FW_SATURATION_THRESHOLD: int = 235
_FW_MIN_BRIGHT_PIXELS: int = 1024
_FW_BOARD_PRIOR_CENTER_X: float = 0.49 * 224.0
_FW_BOARD_PRIOR_CENTER_Y: float = 0.446 * 224.0
_FW_BOARD_PRIOR_RADIUS_PX: float = 0.35 * 224.0
_FW_INNER_DIAL_CENTER_X: float = 0.5 * 224.0
_FW_INNER_DIAL_CENTER_Y: float = 0.446 * 224.0
_FW_FIXED_CROP_BIAS_PX: int = 18
_FW_REFINEMENT_OFFSETS: tuple[int, ...] = (-4, 0, 4)
_FW_CONFIDENCE_THRESHOLD: float = 1.25
_FW_MIN_ACCEPT_SCORE: float = 2.0
_FW_MIN_PEAK_RATIO: float = 0.35
_FW_CONSENSUS_TEMP_DELTA_C: float = 4.0
_FW_CONSENSUS_MIN_QUALITY_RATIO: float = 0.85
_FW_CENTER_MAX_DIST: float = 40.0
_FW_RIM_SEARCH_COARSE_STEP: int = 8
_FW_RIM_SEARCH_FINE_STEP: int = 4
_FW_RIM_SEARCH_RADIUS: float = 20.0


@dataclass
class _FwEstimate:
    valid: bool = False
    center_x: float = 0.0
    center_y: float = 0.0
    angle_deg: float = 0.0
    temperature_c: float = 0.0
    confidence: float = 0.0
    best_score: float = 0.0
    runner_up_score: float = 0.0
    source_label: str = ""
    detection: NeedleDetection | None = None


def _fw_bright_centroid_hypothesis(
    image_bgr: np.ndarray,
) -> tuple[float, float] | None:
    """Luma-weighted bright centroid (matches firmware)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = image_bgr.shape[:2]
    scan_x_min = max(0, round(_FW_INNER_DIAL_CENTER_X - _FW_DIAL_RADIUS_PX * 1.5))
    scan_y_min = max(0, round(_FW_INNER_DIAL_CENTER_Y - _FW_DIAL_RADIUS_PX * 1.5))
    scan_x_max = min(w, round(_FW_INNER_DIAL_CENTER_X + _FW_DIAL_RADIUS_PX * 1.5))
    scan_y_max = min(h, round(_FW_INNER_DIAL_CENTER_Y + _FW_DIAL_RADIUS_PX * 1.5))
    bright_sum_x: int = 0
    bright_sum_y: int = 0
    bright_count: int = 0
    for y in range(scan_y_min, scan_y_max):
        for x in range(scan_x_min, scan_x_max):
            luma = float(gray[y, x])
            if luma < _FW_BRIGHT_THRESHOLD:
                continue
            if luma > _FW_SATURATION_THRESHOLD:
                continue
            bright_count += 1
            bright_sum_x += x
            bright_sum_y += y
    if bright_count < _FW_MIN_BRIGHT_PIXELS:
        return None
    cx = bright_sum_x / bright_count
    cy = bright_sum_y / bright_count
    biased_cy = max(cy - _FW_FIXED_CROP_BIAS_PX, 0.0)
    return (cx, biased_cy)


def _fw_fixed_crop_center() -> tuple[float, float]:
    """Training-crop inner dial center with upward bias (matches firmware)."""
    cx = _FW_INNER_DIAL_CENTER_X
    cy = max(_FW_INNER_DIAL_CENTER_Y - _FW_FIXED_CROP_BIAS_PX, 0.0)
    return (cx, cy)


def _fw_board_prior_center() -> tuple[float, float]:
    """Board prior center (matches firmware)."""
    return (_FW_BOARD_PRIOR_CENTER_X, _FW_BOARD_PRIOR_CENTER_Y)


def _fw_rim_search_center(
    image_bgr: np.ndarray,
) -> tuple[float, float] | None:
    """Score centers near inner dial by rim edge radial alignment (matches firmware).
    
    Uses numpy for vectorized computation instead of Python loops.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = image_bgr.shape[:2]
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    
    sat_mask = gray > _FW_SATURATION_THRESHOLD
    weak_mask = mag < 2.0
    rim_mask = np.zeros((h, w), dtype=bool)
    
    cx0, cy0 = _FW_INNER_DIAL_CENTER_X, _FW_INNER_DIAL_CENTER_Y
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((xx - cx0)**2 + (yy - cy0)**2)
    rim_min = _FW_DIAL_RADIUS_PX * 0.84
    rim_max = _FW_DIAL_RADIUS_PX * 1.04
    rim_mask = (rr >= rim_min) & (rr <= rim_max)
    
    valid = rim_mask & ~sat_mask & ~weak_mask
    if valid.sum() < 10:
        return None
    
    rim_y, rim_x = np.where(valid)
    rim_mag = mag[rim_y, rim_x]
    
    sx = max(1, round(cx0 - _FW_RIM_SEARCH_RADIUS))
    sy = max(1, round(cy0 - _FW_RIM_SEARCH_RADIUS))
    ex = min(w - 1, round(cx0 + _FW_RIM_SEARCH_RADIUS))
    ey = min(h - 1, round(cy0 + _FW_RIM_SEARCH_RADIUS))
    
    best_score_val = -1.0
    best_cx = cx0
    best_cy = cy0
    
    for step in (_FW_RIM_SEARCH_COARSE_STEP, _FW_RIM_SEARCH_FINE_STEP):
        for cy in range(sy, ey + 1, step):
            for cx in range(sx, ex + 1, step):
                drx = rim_x.astype(np.float32) - cx
                dry = rim_y.astype(np.float32) - cy
                dr = np.sqrt(drx * drx + dry * dry)
                dr_safe = np.maximum(dr, 1.0)
                rx = drx / dr_safe
                ry = dry / dr_safe
                gx_vals = gx[rim_y, rim_x] / np.maximum(rim_mag, 1.0)
                gy_vals = gy[rim_y, rim_x] / np.maximum(rim_mag, 1.0)
                align = np.abs(gx_vals * rx + gy_vals * ry)
                score = float(np.sum(align * rim_mag))
                if score > best_score_val:
                    best_score_val = score
                    best_cx = float(cx)
                    best_cy = float(cy)
    
    return (best_cx, best_cy)


def _fw_center_hypotheses(
    image_bgr: np.ndarray,
) -> list[tuple[str, tuple[float, float], float]]:
    """Generate all 5 center hypotheses (matches firmware)."""
    hypotheses: list[tuple[str, tuple[float, float], float]] = []
    bright = _fw_bright_centroid_hypothesis(image_bgr)
    if bright is not None:
        hypotheses.append(("bright-center-polar", bright, _FW_DIAL_RADIUS_PX))
    hypotheses.append(("fixed-crop-polar", _fw_fixed_crop_center(), _FW_DIAL_RADIUS_PX))
    hypotheses.append(("board-prior-polar", _fw_board_prior_center(), _FW_BOARD_PRIOR_RADIUS_PX))
    rim = _fw_rim_search_center(image_bgr)
    if rim is not None:
        hypotheses.append(("rim-center-polar", rim, _FW_DIAL_RADIUS_PX))
    hypotheses.append(("image-center-polar", (_FW_INNER_DIAL_CENTER_X, _FW_INNER_DIAL_CENTER_Y), _FW_DIAL_RADIUS_PX))
    return hypotheses


def _fw_run_polar_vote(
    image_bgr: np.ndarray,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec | None,
) -> _FwEstimate | None:
    """Run the polar vote at a given center (matches firmware)."""
    det = _detect_needle_unit_vector_polar(
        image_bgr,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
        gauge_spec=gauge_spec,
    )
    if det is None:
        return None
    angle_deg = math.degrees(math.atan2(det.unit_dy, det.unit_dx)) % 360
    temp_c = float("nan")
    if gauge_spec is not None:
        temp_c = needle_vector_to_value(det.unit_dx, det.unit_dy, gauge_spec)
    est = _FwEstimate(
        valid=True,
        center_x=center_xy[0],
        center_y=center_xy[1],
        angle_deg=angle_deg,
        temperature_c=temp_c,
        confidence=det.confidence,
        best_score=det.peak_value,
        runner_up_score=det.runner_up_value,
        detection=det,
    )
    return est


def _fw_refine_estimate(
    image_bgr: np.ndarray,
    seed: _FwEstimate,
    dial_radius_px: float,
    gauge_spec: GaugeSpec | None,
) -> _FwEstimate | None:
    """5x5 offset sweep around seed center (matches firmware)."""
    h, w = image_bgr.shape[:2]
    min_cx = 1
    min_cy = 1
    max_cx = w - 2
    max_cy = h - 2
    best = seed
    found = True
    for dy in _FW_REFINEMENT_OFFSETS:
        for dx in _FW_REFINEMENT_OFFSETS:
            cx = int(round(seed.center_x + dx))
            cy = int(round(seed.center_y + dy))
            cx = max(min_cx, min(cx, max_cx))
            cy = max(min_cy, min(cy, max_cy))
            est = _fw_run_polar_vote(image_bgr, (float(cx), float(cy)), dial_radius_px, gauge_spec)
            if est is None:
                continue
            if _fw_is_better_estimate(est, best):
                best = est
    return best if found else None


def _fw_compute_quality(est: _FwEstimate) -> float:
    """Quality = confidence / peak_ratio (matches firmware)."""
    if not est.valid or est.best_score <= 0.0:
        return 0.0
    if est.runner_up_score > 0.0:
        peak_ratio = est.best_score / est.runner_up_score
    else:
        peak_ratio = est.best_score
    peak_ratio = max(peak_ratio, 1.0)
    return est.confidence / peak_ratio


def _fw_source_priority(source_label: str) -> int:
    """Source priority (matches firmware)."""
    if source_label in ("fixed-crop-polar", "image-center-polar", "board-prior-polar", "rim-center-polar"):
        return 4
    if source_label == "bright-center-polar":
        return 2
    return 0


def _fw_is_better_estimate(
    candidate: _FwEstimate,
    incumbent: _FwEstimate,
) -> bool:
    """Compare two estimates using priority + quality (matches firmware)."""
    if not candidate.valid:
        return False
    if not incumbent.valid:
        return True
    cq = _fw_compute_quality(candidate)
    iq = _fw_compute_quality(incumbent)
    if candidate.source_label == "bright-center-polar":
        dx = candidate.center_x - _FW_INNER_DIAL_CENTER_X
        dy = candidate.center_y - _FW_INNER_DIAL_CENTER_Y
        if dx * dx + dy * dy > 150.0 * 150.0:
            return False
    if cq > iq:
        cp = _fw_source_priority(candidate.source_label)
        ip_prior = _fw_source_priority(incumbent.source_label)
        ratio = cq / iq if iq > 0 else float("inf")
        if cp > ip_prior:
            return True
        if cp < ip_prior:
            return ratio >= 2.0
        return True
    if cq < iq:
        cp = _fw_source_priority(candidate.source_label)
        ip_prior = _fw_source_priority(incumbent.source_label)
        return cp > ip_prior
    return False


def _fw_pass_acceptance_gate(
    est: _FwEstimate,
    image_bgr: np.ndarray,
) -> bool:
    """Acceptance gate (matches firmware)."""
    if not est.valid:
        return False
    if est.confidence < _FW_CONFIDENCE_THRESHOLD:
        return False
    if est.best_score < _FW_MIN_ACCEPT_SCORE:
        return False
    if est.runner_up_score > 0.0:
        pk_ratio = est.best_score / est.runner_up_score
        if pk_ratio < _FW_MIN_PEAK_RATIO:
            return False
    dx = est.center_x - _FW_INNER_DIAL_CENTER_X
    dy = est.center_y - _FW_INNER_DIAL_CENTER_Y
    if dx * dx + dy * dy > _FW_CENTER_MAX_DIST * _FW_CENTER_MAX_DIST:
        return False
    return True


def _fw_select_consensus(
    estimates: list[_FwEstimate | None],
    fallback: _FwEstimate | None,
) -> _FwEstimate | None:
    """Consensus clustering (matches firmware)."""
    valid: list[tuple[int, _FwEstimate]] = [
        (i, e) for i, e in enumerate(estimates) if e is not None and e.valid
    ]
    if len(valid) < 2:
        return fallback
    support: list[int] = [0] * len(estimates)
    best_support: int = 0
    for i, e_i in valid:
        count: int = 1
        for j, e_j in valid:
            if i == j:
                continue
            delta = abs(e_i.temperature_c - e_j.temperature_c)
            if delta <= _FW_CONSENSUS_TEMP_DELTA_C:
                count += 1
        support[i] = count
        if count > best_support:
            best_support = count
    if best_support < 2:
        return fallback
    fallback_quality = _fw_compute_quality(fallback) if fallback is not None else 0.0
    fallback_priority = _fw_source_priority(fallback.source_label) if fallback is not None else 0
    best: _FwEstimate | None = None
    best_priority: int = -1
    best_quality: float = -1.0
    best_peak_ratio: float = -1.0
    best_confidence: float = -1.0
    best_score: float = -1.0
    for i, e_i in valid:
        if support[i] != best_support:
            continue
        q = _fw_compute_quality(e_i)
        pk_ratio = (e_i.best_score / e_i.runner_up_score) if e_i.runner_up_score > 0 else e_i.best_score
        p = _fw_source_priority(e_i.source_label)
        better = (
            best is None
            or p > best_priority
            or (p == best_priority and q > best_quality)
            or (p == best_priority and q == best_quality and pk_ratio > best_peak_ratio)
            or (p == best_priority and q == best_quality and pk_ratio == best_peak_ratio and e_i.confidence > best_confidence)
            or (p == best_priority and q == best_quality and pk_ratio == best_peak_ratio and e_i.confidence == best_confidence and e_i.best_score > best_score)
        )
        if better:
            best = e_i
            best_priority = p
            best_quality = q
            best_peak_ratio = pk_ratio
            best_confidence = e_i.confidence
            best_score = e_i.best_score
    if best is None or fallback_quality <= 0.0:
        return fallback
    if best_priority < fallback_priority:
        return fallback
    if best_priority > fallback_priority:
        return best
    if best_quality >= fallback_quality * _FW_CONSENSUS_MIN_QUALITY_RATIO:
        return best
    return fallback


def run_firmware_baseline(
    image_bgr: np.ndarray,
    gauge_spec: GaugeSpec | None = None,
) -> tuple[float, float, float, NeedleDetection | None]:
    """Run the firmware-matched classical baseline.

    Matches the STM32N6 firmware's 5-hypothesis + refinement + consensus
    algorithm in app_baseline_runtime.c.

    Returns (center_x, center_y, dial_radius_px, detection).
    """
    hypotheses = _fw_center_hypotheses(image_bgr)
    raw_estimates: list[_FwEstimate] = []
    for label, (cx, cy), radius in hypotheses:
        est = _fw_run_polar_vote(image_bgr, (cx, cy), radius, gauge_spec)
        if est is not None:
            est.source_label = label
            raw_estimates.append(est)
    if not raw_estimates:
        return (
            _FW_INNER_DIAL_CENTER_X,
            _FW_INNER_DIAL_CENTER_Y,
            _FW_DIAL_RADIUS_PX,
            None,
        )
    refined: list[_FwEstimate | None] = []
    for est in raw_estimates:
        r = _fw_refine_estimate(image_bgr, est, _FW_DIAL_RADIUS_PX, gauge_spec)
        # Locate matching hypothesis radius for source label
        hyp_radius = _FW_DIAL_RADIUS_PX
        for label2, _, rad in hypotheses:
            if label2 == est.source_label:
                hyp_radius = rad
                break
        if r is not None:
            r.source_label = est.source_label
            refined.append(r)
        else:
            refined.append(None)
    best: _FwEstimate | None = None
    for r in refined:
        if r is None:
            continue
        if best is None or _fw_is_better_estimate(r, best):
            best = r
    consensus = _fw_select_consensus(refined, best)
    if consensus is not None and consensus.valid:
        return (consensus.center_x, consensus.center_y, _FW_DIAL_RADIUS_PX, consensus.detection)
    return (
        _FW_INNER_DIAL_CENTER_X,
        _FW_INNER_DIAL_CENTER_Y,
        _FW_DIAL_RADIUS_PX,
        None,
    )