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
        line_px: float = float(
            gray_image[int(round(sample_y)), int(round(sample_x))]
        )

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


def detect_needle_unit_vector(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec | None = None,
) -> NeedleDetection | None:
    """Detect needle direction using a line-first classical CV strategy.

    Args:
        image_bgr: Input color image loaded by OpenCV (BGR).
        center_xy: Dial center in image pixels.
        dial_radius_px: Approximate dial radius in pixels.

    Returns:
        A center-origin unit vector (dx, dy) and score, or ``None`` if no robust
        line candidate is found.
    """
    if dial_radius_px <= 1.0:
        return None

    center_x, center_y = center_xy

    # Normalize local contrast and smooth noise to stabilize edges.
    gray: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced: np.ndarray = clahe.apply(gray)
    blurred: np.ndarray = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Extract edge structure for line detection.
    edges: np.ndarray = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Keep an annulus: reject center hub and outer frame clutter.
    yy, xx = np.indices(edges.shape)
    rr: np.ndarray = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    annulus_mask: np.ndarray = (rr > 0.15 * dial_radius_px) & (rr < 0.95 * dial_radius_px)
    masked_edges: np.ndarray = np.where(annulus_mask, edges, 0).astype(np.uint8)

    # Probabilistic Hough gives concrete segment endpoints we can score.
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=40,
        minLineLength=int(0.25 * dial_radius_px),
        maxLineGap=8,
    )
    if lines is None:
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=28,
            minLineLength=int(0.18 * dial_radius_px),
            maxLineGap=12,
        )

    best_score: float = float("-inf")
    best_unit_vec: tuple[float, float] | None = None

    if lines is not None:
        for line in lines[:, 0, :]:
            x1_i, y1_i, x2_i, y2_i = line
            x1: float = float(x1_i)
            y1: float = float(y1_i)
            x2: float = float(x2_i)
            y2: float = float(y2_i)

            seg_dx: float = x2 - x1
            seg_dy: float = y2 - y1
            seg_len: float = math.hypot(seg_dx, seg_dy)
            if seg_len <= 1e-6:
                continue

            endpoint1_r: float = math.hypot(x1 - center_x, y1 - center_y)
            endpoint2_r: float = math.hypot(x2 - center_x, y2 - center_y)
            tail_r: float = min(endpoint1_r, endpoint2_r)
            tip_r: float = max(endpoint1_r, endpoint2_r)
            tail_norm: float = tail_r / dial_radius_px
            tip_norm: float = tip_r / dial_radius_px
            tip_x: float = x1 if endpoint1_r > endpoint2_r else x2
            tip_y: float = y1 if endpoint1_r > endpoint2_r else y2

            # Needle candidates should originate near the hub and extend outward.
            if tail_norm > 0.40 or tip_norm < 0.55:
                continue

            # Favor lines that pass through the dial center and point outward.
            mid_x: float = 0.5 * (x1 + x2)
            mid_y: float = 0.5 * (y1 + y2)
            midpoint_dist_to_center: float = math.hypot(
                mid_x - center_x,
                mid_y - center_y,
            )
            center_dist: float = _point_to_segment_distance(
                center_x,
                center_y,
                x1,
                y1,
                x2,
                y2,
            )
            line_contrast: float = 0.0
            dark_fraction: float = 0.0
            line_contrast, dark_fraction = _sample_line_darkness(
                enhanced,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
            score: float = (
                4.0 * (tip_norm - tail_norm)
                - 3.0 * (center_dist / dial_radius_px)
                - 0.75 * (midpoint_dist_to_center / dial_radius_px)
                + 1.6 * max(line_contrast, 0.0)
                + 0.8 * dark_fraction
            )

            if score > best_score:
                best_score = score

                tip_dx: float = tip_x - center_x
                tip_dy: float = tip_y - center_y
                tip_norm_final: float = math.hypot(tip_dx, tip_dy)
                if tip_norm_final <= 1e-6:
                    continue
                best_unit_vec = (tip_dx / tip_norm_final, tip_dy / tip_norm_final)

    if best_unit_vec is None:
        # Keep the polar detector only as a conservative fallback when the
        # line-based detector cannot find a plausible segment.
        angle_bounds_rad: tuple[float, float] | None = None
        if gauge_spec is not None:
            angle_bounds_rad = (gauge_spec.min_angle_rad, gauge_spec.sweep_rad)
        return _detect_needle_unit_vector_polar(
            image_bgr,
            center_xy=center_xy,
            dial_radius_px=dial_radius_px,
            angle_bounds_rad=angle_bounds_rad,
        )

    return NeedleDetection(
        unit_dx=float(best_unit_vec[0]),
        unit_dy=float(best_unit_vec[1]),
        confidence=float(max(best_score, 0.0)),
    )


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
