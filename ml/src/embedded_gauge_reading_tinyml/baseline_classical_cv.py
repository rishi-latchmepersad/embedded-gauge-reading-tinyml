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


def _needle_vector_to_value(unit_dx: float, unit_dy: float, spec: GaugeSpec) -> float:
    """Convert a unit direction vector to calibrated gauge value.

    The conversion mirrors the sweep calibration math used by training/evaluation.
    """
    raw_angle: float = math.atan2(unit_dy, unit_dx)

    # Shift angle into the gauge's calibrated frame and wrap into [0, 2*pi).
    shifted: float = (raw_angle - spec.min_angle_rad) % (2.0 * math.pi)

    # Clamp non-strictly so out-of-sweep detections saturate at max value.
    fraction: float = min(max(shifted / spec.sweep_rad, 0.0), 1.0)
    return spec.min_value + fraction * (spec.max_value - spec.min_value)


def detect_needle_unit_vector(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
) -> NeedleDetection | None:
    """Detect needle direction using Canny + Hough line transform.

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
        return None

    best_score: float = float("-inf")
    best_unit_vec: tuple[float, float] | None = None

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

        # Favor lines that are long and pass near the dial center.
        mid_x: float = 0.5 * (x1 + x2)
        mid_y: float = 0.5 * (y1 + y2)
        midpoint_dist_to_center: float = math.hypot(mid_x - center_x, mid_y - center_y)
        score: float = seg_len - 1.5 * midpoint_dist_to_center

        if score > best_score:
            best_score = score

            # Needle tip should be farther from center than the tail endpoint.
            endpoint1_r: float = math.hypot(x1 - center_x, y1 - center_y)
            endpoint2_r: float = math.hypot(x2 - center_x, y2 - center_y)
            tip_x: float = x1 if endpoint1_r > endpoint2_r else x2
            tip_y: float = y1 if endpoint1_r > endpoint2_r else y2

            tip_dx: float = tip_x - center_x
            tip_dy: float = tip_y - center_y
            norm: float = math.hypot(tip_dx, tip_dy)
            if norm <= 1e-6:
                continue
            best_unit_vec = (tip_dx / norm, tip_dy / norm)

    if best_unit_vec is None:
        return None

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

        dial_radius_px: float = max(sample.dial.rx, sample.dial.ry)
        detection: NeedleDetection | None = detect_needle_unit_vector(
            image_bgr,
            center_xy=(sample.center.x, sample.center.y),
            dial_radius_px=dial_radius_px,
        )
        if detection is None:
            continue

        true_value: float = needle_value(sample, spec, strict=False)
        predicted_value: float = _needle_vector_to_value(
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
