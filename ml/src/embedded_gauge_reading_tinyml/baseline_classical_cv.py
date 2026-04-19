"""Classical computer-vision baseline for analog gauge reading.

This module provides a non-neural baseline that can be used to benchmark
whether a learned model is improving over simple image processing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2
import numpy as np

from embedded_gauge_reading_tinyml.dataset import Sample
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, needle_value


@dataclass(frozen=True)
class NeedleDetection:
    """Needle detection output represented as a center-origin unit vector."""

    unit_dx: float
    unit_dy: float
    confidence: float


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
    return NeedleDetection(unit_dx=unit_dx, unit_dy=unit_dy, confidence=contrast_score)


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


def _sample_line_darkness(
    gray_image: np.ndarray,
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[float, float]:
    """Estimate how needle-like a line segment is from local darkness.

    A true needle usually reads as a thin dark stroke with darker pixels along
    the segment than in the immediate neighborhood on either side.
    """
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

    # Avoid the exact endpoints so the hub and rim do not dominate the score.
    for fraction in np.linspace(0.15, 0.85, 9, dtype=np.float32):
        sample_count += 1
        sample_x: float = x1 + float(fraction) * seg_dx
        sample_y: float = y1 + float(fraction) * seg_dy
        ix: int = int(round(min(max(sample_x, 0.0), gray_image.shape[1] - 1.0)))
        iy: int = int(round(min(max(sample_y, 0.0), gray_image.shape[0] - 1.0)))
        line_px: float = float(gray_image[iy, ix])

        neighbor_values: list[float] = []
        for offset_px in (2.0, 4.0):
            for direction in (-1.0, 1.0):
                nx: float = sample_x + direction * offset_px * perp_x
                ny: float = sample_y + direction * offset_px * perp_y
                ix: int = int(round(min(max(nx, 0.0), gray_image.shape[1] - 1.0)))
                iy: int = int(round(min(max(ny, 0.0), gray_image.shape[0] - 1.0)))
                neighbor_values.append(float(gray_image[iy, ix]))

        if not neighbor_values:
            continue

        local_mean: float = float(np.mean(neighbor_values))
        sample_contrasts.append(local_mean - line_px)
        if line_px + 2.0 < local_mean:
            dark_hits += 1

    if not sample_contrasts or sample_count == 0:
        return 0.0, 0.0

    contrast_mean: float = float(np.mean(sample_contrasts)) / 255.0
    dark_fraction: float = float(dark_hits) / float(sample_count)
    return contrast_mean, dark_fraction


def _angle_in_sweep(angle_rad: float, spec: GaugeSpec, *, margin_rad: float = 0.0) -> bool:
    """Return True if angle_rad falls within the gauge's calibrated sweep arc.

    An optional margin widens the arc on each side to tolerate small detection
    offsets near the min/max ticks.
    """
    shifted: float = (angle_rad - spec.min_angle_rad) % (2.0 * math.pi)
    return shifted <= (spec.sweep_rad + margin_rad)


def detect_needle_unit_vector(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec | None = None,
) -> NeedleDetection | None:
    """Detect needle direction using radial-spoke voting on edge gradients.

    For every dark edge pixel in the inner dial annulus we ask: how well does
    this pixel's image gradient align with the radial direction toward the dial
    center?  A needle is a radial stroke, so its edge pixels have gradients that
    are nearly perpendicular to the radial direction, meaning the pixel itself
    points radially.  We accumulate a smoothed angular histogram of
    gradient-weighted radial votes and pick the dominant angle.

    This is robust to the heavily-printed tick-mark background because tick marks
    are short tangential arcs: their gradient is radial (perpendicular to the
    arc), so they vote for *many* angles equally and don't produce a sharp peak.
    The needle's gradient is tangential to the needle direction and radial to the
    dial center — it votes strongly for one specific spoke angle.
    """
    if dial_radius_px <= 1.0:
        return None

    center_x, center_y = center_xy

    # Enhance local contrast and extract Sobel gradients.
    gray: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced: np.ndarray = clahe.apply(gray)
    blurred: np.ndarray = cv2.GaussianBlur(enhanced, (5, 5), 0)

    gx: np.ndarray = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy: np.ndarray = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag: np.ndarray = np.sqrt(gx * gx + gy * gy)

    # Work only in the inner annulus: exclude the hub and the outer tick band.
    # Tick marks live at 75–95% radius; the needle crosses all radii from the
    # hub outward, so restricting to 15–75% eliminates most tick-mark clutter
    # while keeping the needle's strongest radial portion.
    h_img, w_img = gray.shape[:2]
    yy, xx = np.meshgrid(
        np.arange(h_img, dtype=np.float32),
        np.arange(w_img, dtype=np.float32),
        indexing="ij",
    )
    dx_from_center: np.ndarray = xx - center_x
    dy_from_center: np.ndarray = yy - center_y
    rr: np.ndarray = np.sqrt(dx_from_center ** 2 + dy_from_center ** 2)

    inner_mask: np.ndarray = (rr > 0.15 * dial_radius_px) & (rr < 0.75 * dial_radius_px)

    # Suppress the humidity subdial region.
    # The subdial sits at (cx, cy+0.25r) and its needle sweeps through the
    # lower-center of the main dial, injecting spurious votes.
    # We mask pixels that are:
    #   - far enough from the main hub that they can't be the main needle root
    #   - within the horizontal and vertical extents of the subdial
    # The r > 0.20r guard preserves the main needle near the hub regardless of
    # which direction it points.
    dx_sub: np.ndarray = np.abs(xx - center_x)
    dy_sub: np.ndarray = yy - center_y
    in_subdial_zone: np.ndarray = (
        (rr > 0.20 * dial_radius_px)
        & (dx_sub < 0.35 * dial_radius_px)
        & (dy_sub > 0.10 * dial_radius_px)
        & (dy_sub < 0.58 * dial_radius_px)
    )
    inner_mask = inner_mask & ~in_subdial_zone

    # For each pixel, compute the unit radial direction (toward center).
    rr_safe: np.ndarray = np.where(rr > 0.5, rr, 1.0)
    radial_x: np.ndarray = -dx_from_center / rr_safe   # points toward center
    radial_y: np.ndarray = -dy_from_center / rr_safe

    # The gradient of a dark needle on a light background is perpendicular to
    # the needle.  The needle points radially, so its gradient is tangential.
    # A pixel votes for spoke angle θ = atan2(dy_from_center, dx_from_center)
    # weighted by how much its gradient is *tangential* to the radial direction,
    # i.e. how perpendicular the gradient is to the radial direction.
    # perpendicularity = |grad × radial| / (|grad| * 1) = |cross product|
    grad_mag_safe: np.ndarray = np.where(grad_mag > 1.0, grad_mag, 1.0)
    gx_n: np.ndarray = gx / grad_mag_safe
    gy_n: np.ndarray = gy / grad_mag_safe
    # 2-D cross product magnitude = |gx*radial_y - gy*radial_x|
    tangential_weight: np.ndarray = np.abs(gx_n * radial_y - gy_n * radial_x)

    # Vote weight = gradient magnitude * tangential alignment, inside annulus.
    vote_weight: np.ndarray = np.where(
        inner_mask & (grad_mag > 8.0),
        grad_mag * tangential_weight,
        0.0,
    )

    # Each pixel's spoke angle: direction *from* center *to* pixel.
    spoke_angle: np.ndarray = np.arctan2(dy_from_center, dx_from_center)  # (H, W)

    # Accumulate into a 1-D histogram with 720 bins (0.5° resolution).
    num_bins: int = 720
    angle_bins: np.ndarray = ((spoke_angle + math.pi) / (2.0 * math.pi) * num_bins).astype(
        np.int32
    )
    angle_bins = np.clip(angle_bins, 0, num_bins - 1)

    vote_flat: np.ndarray = vote_weight.ravel()
    bin_flat: np.ndarray = angle_bins.ravel()
    histogram: np.ndarray = np.zeros(num_bins, dtype=np.float32)
    np.add.at(histogram, bin_flat, vote_flat)

    # If gauge_spec is provided, zero out bins outside the valid sweep arc so
    # the subdial region and dead-zone don't compete.
    if gauge_spec is not None:
        for b in range(num_bins):
            angle_rad: float = (b / num_bins) * 2.0 * math.pi - math.pi
            if not _angle_in_sweep(angle_rad, gauge_spec, margin_rad=math.radians(12.0)):
                histogram[b] = 0.0

    # Smooth the histogram to merge nearby votes from a slightly curved needle.
    kernel_width: int = 15  # ~7.5° half-width
    histogram_smooth: np.ndarray = cv2.GaussianBlur(
        histogram[np.newaxis, :], (1, kernel_width * 2 + 1), 0
    ).ravel()

    best_bin: int = int(np.argmax(histogram_smooth))
    peak_val: float = float(histogram_smooth[best_bin])
    noise: float = float(np.mean(histogram_smooth)) + 1e-6
    snr: float = peak_val / noise

    # Require the peak to stand clearly above the background noise level.
    if snr < 2.0:
        return None

    best_angle: float = (best_bin / num_bins) * 2.0 * math.pi - math.pi
    unit_dx: float = math.cos(best_angle)
    unit_dy: float = math.sin(best_angle)

    return NeedleDetection(
        unit_dx=float(unit_dx),
        unit_dy=float(unit_dy),
        confidence=float(snr),
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

    if primary_detection.confidence >= confidence_threshold:
        return primary_detection

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
    return secondary_detection


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
