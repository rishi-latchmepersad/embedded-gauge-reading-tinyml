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


# When multiple candidates agree within a few degrees, keep that consensus
# cluster instead of letting a lone outlier win on score alone.
CONSENSUS_TEMP_DELTA_C: Final[float] = 4.0
"""Temperature tolerance for candidate-agreement consensus."""

# The board capture is centered slightly above and left of the dial midpoint,
# and the useful inner gauge radius is much smaller than the full crop.
# These ratios give us a lightweight classical prior without introducing ML.
BOARD_PRIOR_CENTER_X_RATIO: Final[float] = 0.490
BOARD_PRIOR_CENTER_Y_RATIO: Final[float] = 0.446
BOARD_PRIOR_RADIUS_RATIO: Final[float] = 0.290
BOARD_PRIOR_CENTER_OFFSETS_PX: Final[tuple[float, ...]] = (-4.0, 0.0, 4.0)
BOARD_PRIOR_RADIUS_SCALES: Final[tuple[float, ...]] = (0.94, 1.0, 1.06)


def board_prior_geometry_candidate(image_bgr: np.ndarray) -> GeometryCandidate:
    """Build the fixed board prior used by the classical sweep.

    The prior is intentionally simple: it biases the detector toward the inner
    Celsius dial seen in the board capture, which has a slightly off-center
    pivot and a smaller effective radius than the full crop.
    """
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


def needle_detection_quality(detection: NeedleDetection) -> float:
    """Return a bounded scalar score that favors strong, supported peaks.

    The earlier confidence * peak-ratio score could explode when the runner-up
    bin collapsed to zero on a spurious geometry. That made the selector chase
    isolated spikes instead of stable needle-like responses. We now reward the
    main peak directly, but only give a candidate full credit when it still has
    some supporting runner-up energy nearby.
    """
    peak_value: float = max(detection.peak_value, 0.0)
    runner_value: float = max(detection.runner_up_value, 0.0)
    support_term: float = 0.1 + math.log1p(runner_value)
    return peak_value * support_term


def _geometry_selection_key(selection: GeometrySelection) -> tuple[float, float, float]:
    """Return the tie-break tuple used when two geometry candidates agree."""
    return (
        selection.quality,
        selection.shaft_support,
        selection.detection.peak_value,
    )


def _runner_up_peak_after_suppression(
    peak_values: np.ndarray,
    *,
    best_index: int,
    suppression_bins: int,
) -> float:
    """Find the strongest non-neighbor peak after suppressing the main peak.

    Using the global second-highest bin can overstate how clean a candidate is
    when the histogram has a broad plateau. We instead suppress a small window
    around the best bin and measure the strongest remaining competitor.
    """
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


def _middle_shaft_weight(rr: np.ndarray, dial_radius_px: float) -> np.ndarray:
    """Return a smooth radial weight that favors the middle of the needle shaft.

    The gauge face is visually busy near the outer tick ring and around the hub,
    so we emphasize the middle band of the shaft where the needle is usually
    darkest and least confused with dial graphics.
    """
    if dial_radius_px <= 1.0:
        return np.zeros_like(rr, dtype=np.float32)

    shaft_center: float = 0.56 * dial_radius_px
    shaft_sigma: float = max(0.09 * dial_radius_px, 1.0)
    normalized: np.ndarray = (rr - shaft_center) / shaft_sigma
    return np.exp(-0.5 * np.square(normalized)).astype(np.float32)


def _detect_needle_unit_vector_shaft_scan(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec | None = None,
) -> NeedleDetection | None:
    """Scan the sweep for a colored shaft on a bright dial face.

    The live board captures show a saturated needle-like shaft that is easy to
    miss when the detector only rewards grayscale darkness. This pass samples
    the middle part of the candidate shaft directly and looks for saturation
    and color-spread contrast against the immediate neighborhood on either side
    of the line.
    """
    if dial_radius_px <= 1.0:
        return None

    center_x, center_y = center_xy
    hsv: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation: np.ndarray = hsv[:, :, 1].astype(np.float32) / 255.0
    color_spread: np.ndarray = (
        np.max(image_bgr.astype(np.float32), axis=2)
        - np.min(image_bgr.astype(np.float32), axis=2)
    ) / 255.0

    angle_bins: int = 720
    fractions: np.ndarray = np.linspace(0.34, 0.78, 7, dtype=np.float32)
    shaft_center_fraction: float = 0.54
    shaft_sigma_fraction: float = 0.12

    h_img, w_img = image_bgr.shape[:2]
    angle_scores: np.ndarray = np.zeros(angle_bins, dtype=np.float32)

    for bin_index in range(angle_bins):
        angle_rad: float = (2.0 * math.pi * float(bin_index)) / float(angle_bins) - math.pi
        if gauge_spec is not None and not _angle_in_sweep(
            angle_rad,
            gauge_spec,
            margin_rad=math.radians(8.0),
        ):
            continue

        unit_dx: float = math.cos(angle_rad)
        unit_dy: float = math.sin(angle_rad)
        perp_dx: float = -unit_dy
        perp_dy: float = unit_dx

        sample_contrasts: list[float] = []
        for fraction in fractions:
            sample_x: float = center_x + float(fraction) * dial_radius_px * unit_dx
            sample_y: float = center_y + float(fraction) * dial_radius_px * unit_dy
            sample_ix: int = int(round(min(max(sample_x, 0.0), w_img - 1.0)))
            sample_iy: int = int(round(min(max(sample_y, 0.0), h_img - 1.0)))

            line_sat: float = float(saturation[sample_iy, sample_ix])
            line_spread: float = float(color_spread[sample_iy, sample_ix])
            neighbor_sats: list[float] = []
            neighbor_spreads: list[float] = []
            for offset_px in (2.0, 4.0, 6.0):
                for direction in (-1.0, 1.0):
                    nx: float = sample_x + direction * offset_px * perp_dx
                    ny: float = sample_y + direction * offset_px * perp_dy
                    neighbor_ix: int = int(round(min(max(nx, 0.0), w_img - 1.0)))
                    neighbor_iy: int = int(round(min(max(ny, 0.0), h_img - 1.0)))
                    neighbor_sats.append(float(saturation[neighbor_iy, neighbor_ix]))
                    neighbor_spreads.append(float(color_spread[neighbor_iy, neighbor_ix]))

            if not neighbor_sats or not neighbor_spreads:
                continue
            local_sat: float = float(np.mean(neighbor_sats)) if neighbor_sats else 0.0
            local_spread: float = (
                float(np.mean(neighbor_spreads)) if neighbor_spreads else 0.0
            )
            shaft_weight: float = math.exp(
                -0.5
                * ((float(fraction) - shaft_center_fraction) / shaft_sigma_fraction) ** 2
            )
            contrast: float = shaft_weight * (
                0.50 * max(line_sat - local_sat, 0.0)
                + 0.50 * max(line_spread - local_spread, 0.0)
            )
            sample_contrasts.append(contrast)

        if sample_contrasts:
            angle_scores[bin_index] = float(np.quantile(np.asarray(sample_contrasts), 0.25))

    if not np.any(angle_scores > 0.0):
        return None

    # Smooth the angular profile a little so a single noisy bin does not win.
    angle_scores_smooth: np.ndarray = cv2.GaussianBlur(
        angle_scores[np.newaxis, :],
        (1, 9),
        0,
    ).ravel()
    best_bin: int = int(np.argmax(angle_scores_smooth))
    best_score: float = float(angle_scores_smooth[best_bin])
    if best_score <= 0.0:
        return None

    runner_up: float = _runner_up_peak_after_suppression(
        angle_scores_smooth,
        best_index=best_bin,
        suppression_bins=6,
    )
    peak_ratio: float = best_score / max(runner_up, 1e-6)

    best_angle: float = (2.0 * math.pi * float(best_bin)) / float(angle_bins) - math.pi
    if gauge_spec is not None and not _angle_in_sweep(
        best_angle,
        gauge_spec,
        margin_rad=math.radians(8.0),
    ):
        return None

    # Scale the score back into the same rough range as the other detectors so
    # the shared ranking code can compare them without special casing.
    scaled_peak: float = best_score * 255.0
    scaled_runner_up: float = runner_up * 255.0
    confidence: float = scaled_peak / max(scaled_runner_up, 1e-6) if scaled_runner_up > 0.0 else scaled_peak

    return NeedleDetection(
        unit_dx=float(math.cos(best_angle)),
        unit_dy=float(math.sin(best_angle)),
        confidence=float(confidence),
        peak_value=float(scaled_peak),
        runner_up_value=float(scaled_runner_up),
        peak_ratio=float(peak_ratio),
        peak_margin=float((best_score - runner_up) * 255.0),
    )


def _detect_needle_unit_vector_polar(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    angle_bounds_rad: tuple[float, float] | None = None,
) -> NeedleDetection | None:
    """Detect the needle by looking for a dark radial stripe in polar space.

    When a gauge sweep is known, the search is restricted to that angular span
    so the fallback does not get distracted by the surrounding dial clutter.
    """
    if dial_radius_px <= 1.0:
        return None

    center_x, center_y = center_xy

    # Normalize contrast first so the radial stripe stands out more cleanly.
    gray: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced: np.ndarray = clahe.apply(gray)
    blurred: np.ndarray = cv2.GaussianBlur(enhanced, (5, 5), 0)

    angle_bins: int = 720
    radius_bins: int = max(64, int(round(dial_radius_px)))
    max_radius: float = min(float(min(blurred.shape[:2])) / 2.0, dial_radius_px * 0.98)
    if max_radius <= 1.0:
        return None

    # In polar view, the needle should appear as a dark vertical band.
    polar: np.ndarray = cv2.warpPolar(
        blurred,
        (angle_bins, radius_bins),
        (center_x, center_y),
        max_radius,
        cv2.WARP_POLAR_LINEAR,
    )
    if polar.size == 0:
        return None

    # Ignore the center hub and the outer dial edge to reduce false matches.
    # We bias a little farther out so the lower subdial and hub clutter do not
    # dominate the angle score on frames where the main needle is hard to see.
    start_row: int = max(1, int(0.22 * radius_bins))
    end_row: int = max(start_row + 1, int(0.95 * radius_bins))
    radial_slice: np.ndarray = polar[start_row:end_row, :].astype(np.float32)
    if radial_slice.size == 0:
        return None

    # Blend the mean and a low percentile so thin dark needles still stand out
    # even when the dial face has text or tick marks nearby.
    column_means: np.ndarray = np.mean(radial_slice, axis=0)
    column_q20: np.ndarray = np.percentile(radial_slice, 20.0, axis=0)
    angular_profile: np.ndarray = 0.55 * column_means + 0.45 * column_q20
    smoothed: np.ndarray = cv2.GaussianBlur(
        angular_profile[np.newaxis, :],
        (1, 31),
        0,
    ).ravel()

    # Use a local baseline rather than a global threshold so a valid needle can
    # still win on frames where the whole dial is relatively busy.
    local_baseline: np.ndarray = cv2.blur(
        smoothed[np.newaxis, :],
        (1, 61),
        borderType=cv2.BORDER_REFLECT,
    ).ravel()
    contrast_profile: np.ndarray = local_baseline - smoothed

    if angle_bounds_rad is not None:
        start_angle_rad, sweep_rad = angle_bounds_rad
        angle_span_rad = max(1e-6, sweep_rad)
        start_index: int = int(
            round((start_angle_rad % (2.0 * math.pi)) / (2.0 * math.pi) * angle_bins)
        )
        sweep_bins: int = max(
            1,
            int(round((angle_span_rad / (2.0 * math.pi)) * angle_bins)),
        )
        candidate_indices = (start_index + np.arange(sweep_bins + 1)) % angle_bins
    else:
        candidate_indices = np.arange(angle_bins)

    candidate_contrasts: np.ndarray = contrast_profile[candidate_indices]
    best_offset: int = int(np.argmax(candidate_contrasts))
    best_index: int = int(candidate_indices[best_offset])
    best_contrast: float = float(candidate_contrasts[best_offset])
    noise: float = float(np.std(candidate_contrasts)) + 1e-6
    contrast_score: float = best_contrast / noise

    # Keep only peaks that clearly beat the local background.
    if best_contrast <= 0.0 or contrast_score < 0.45:
        return None

    angle_rad: float = (2.0 * math.pi * best_index) / float(angle_bins)
    unit_dx: float = math.cos(angle_rad)
    unit_dy: float = math.sin(angle_rad)
    runner_up_contrast: float = _runner_up_peak_after_suppression(
        candidate_contrasts,
        best_index=best_offset,
        suppression_bins=max(1, kernel_width // 2),
    )
    return NeedleDetection(
        unit_dx=unit_dx,
        unit_dy=unit_dy,
        confidence=contrast_score,
        peak_value=best_contrast,
        runner_up_value=runner_up_contrast,
        peak_ratio=best_contrast / max(runner_up_contrast, 1e-6),
        peak_margin=best_contrast - runner_up_contrast,
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


def _angle_in_sweep(angle_rad: float, spec: GaugeSpec, *, margin_rad: float = 0.0) -> bool:
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
    angle_bins: np.ndarray = ((spoke_angle + math.pi) / (2.0 * math.pi) * num_bins).astype(
        np.int32
    )
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
    angle_bins: np.ndarray = ((spoke_angle + math.pi) / (2.0 * math.pi) * num_bins).astype(
        np.int32
    )
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

        center_dist: float = _point_to_segment_distance(center_x, center_y, x1, y1, x2, y2)
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
        peak_margin=float(best_score - second_score if second_score > 0.0 else best_score),
    )


def _detect_needle_unit_vector_combined(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec | None = None,
) -> NeedleDetection | None:
    """Run the improved classical detectors and keep the best result."""
    candidates: list[NeedleDetection] = []

    spoke_detection: NeedleDetection | None = _detect_needle_unit_vector_spoke_improved(
        image_bgr,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
        gauge_spec=gauge_spec,
    )
    if spoke_detection is not None:
        candidates.append(spoke_detection)

    center_detection: NeedleDetection | None = _detect_needle_unit_vector_center_weighted(
        image_bgr,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
        gauge_spec=gauge_spec,
    )
    if center_detection is not None:
        candidates.append(center_detection)

    line_detection: NeedleDetection | None = _detect_needle_unit_vector_hough_lines(
        image_bgr,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
        gauge_spec=gauge_spec,
    )
    if line_detection is not None:
        candidates.append(line_detection)

    if not candidates:
        return None

    return max(candidates, key=needle_detection_quality)


def detect_needle_unit_vector(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec | None = None,
) -> NeedleDetection | None:
    """Detect needle direction using the strongest improved classical vote."""
    return _detect_needle_unit_vector_combined(
        image_bgr,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
        gauge_spec=gauge_spec,
    )


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
    detectors: Sequence[
        Callable[[np.ndarray, tuple[float, float], float, GaugeSpec | None], NeedleDetection | None]
    ]
    | None = None,
) -> GeometrySelection | None:
    """Run the detector over several geometry hypotheses and keep the best one.

    The default is still the highest-quality detector, but if several geometry
    hypotheses land within a few degrees of each other, we prefer that small
    agreement cluster over a lone high-score outlier.
    """
    detector_fns: Sequence[
        Callable[[np.ndarray, tuple[float, float], float, GaugeSpec | None], NeedleDetection | None]
    ] = detectors if detectors is not None else (detect_needle_unit_vector,)
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
                + detection.unit_dx * (0.68 * candidate.dial_radius_px),
                y2=candidate.center_xy[1]
                + detection.unit_dy * (0.68 * candidate.dial_radius_px),
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

            if _geometry_selection_key(selection) > _geometry_selection_key(best_selection):
                best_selection = selection
                best_index = len(selections) - 1

    if (best_selection is None) or (gauge_spec is None) or (len(selections) < 2):
        return best_selection

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
    return selections[consensus_index]


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

        image_bgr: np.ndarray | None = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
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
