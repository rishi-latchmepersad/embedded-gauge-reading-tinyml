"""Single-image classical baseline runner.

This module runs the existing Canny + Hough baseline on one camera frame and
optionally writes a small annotated preview plus JSON summary for inspection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    NeedleDetection,
    detect_needle_unit_vector,
    needle_vector_to_value,
)
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, load_gauge_specs

ML_ROOT: Path = Path(__file__).resolve().parents[2]
"""Project root resolved from the package location."""

DEFAULT_ARTIFACTS_DIR: Path = ML_ROOT / "artifacts" / "single_image_baseline"
"""Default folder for single-image baseline artifacts."""


@dataclass(frozen=True)
class SingleImageBaselineConfig:
    """Configuration for one single-image classical baseline run."""

    image_path: Path
    gauge_id: str = "littlegood_home_temp_gauge_c"
    center_x: float | None = None
    center_y: float | None = None
    dial_radius_px: float | None = None
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    run_name: str = ""


@dataclass(frozen=True)
class SingleImageBaselineResult:
    """Structured output for a single-image classical baseline run."""

    image_path: Path
    gauge_spec: GaugeSpec
    center_xy: tuple[float, float]
    dial_radius_px: float
    detection: NeedleDetection | None
    predicted_value: float | None
    annotated_image_path: Path | None


def _timestamp_run_name() -> str:
    """Build a stable timestamp-based directory name for a fresh run."""
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _estimate_dial_geometry(image_bgr: np.ndarray) -> tuple[tuple[float, float], float] | None:
    """Estimate the dial center and radius with a coarse Hough circle search."""
    gray: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred: np.ndarray = cv2.GaussianBlur(gray, (9, 9), 2.0)

    height, width = gray.shape[:2]
    min_radius: int = max(8, int(min(height, width) * 0.18))
    max_radius: int = max(min_radius + 1, int(min(height, width) * 0.48))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, min(height, width) // 4),
        param1=120,
        param2=28,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None

    candidates: np.ndarray = np.squeeze(circles, axis=0)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        return None

    image_cx: float = 0.5 * width
    image_cy: float = 0.5 * height

    # Prefer circles near the image center and with a large radius.
    best_score: float = float("-inf")
    best_circle: tuple[float, float, float] | None = None
    for cx_f, cy_f, radius_f in candidates:
        cx: float = float(cx_f)
        cy: float = float(cy_f)
        radius: float = float(radius_f)
        center_dist: float = math.hypot(cx - image_cx, cy - image_cy)
        score: float = radius - 0.25 * center_dist
        if score > best_score:
            best_score = score
            best_circle = (cx, cy, radius)

    if best_circle is None:
        return None

    return (best_circle[0], best_circle[1]), best_circle[2]


def estimate_dial_geometry(image_bgr: np.ndarray) -> tuple[tuple[float, float], float] | None:
    """Public wrapper for the dial geometry estimator used by the baseline.

    The manifest evaluator reuses this helper so the same geometry logic drives
    both the one-off preview mode and the batch benchmark path.
    """
    return _estimate_dial_geometry(image_bgr)


def _draw_annotation(
    image_bgr: np.ndarray,
    *,
    gauge_spec: GaugeSpec,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    detection: NeedleDetection | None,
    output_path: Path,
) -> None:
    """Save a simple annotated preview for human inspection.

    The preview now includes the calibrated sweep start/end rays so we can
    visually confirm the angle-to-value mapping used by the classical baseline.
    """
    annotated: np.ndarray = image_bgr.copy()
    center_x, center_y = center_xy
    center_i: tuple[int, int] = (int(round(center_x)), int(round(center_y)))
    sweep_len: float = dial_radius_px * 0.95

    def point_on_circle(angle_rad: float) -> tuple[int, int]:
        """Project an angle onto the dial circle for annotation."""
        x: int = int(round(center_x + math.cos(angle_rad) * sweep_len))
        y: int = int(round(center_y + math.sin(angle_rad) * sweep_len))
        return (x, y)

    cv2.circle(
        annotated,
        center_i,
        int(round(dial_radius_px)),
        (0, 255, 0),
        2,
    )
    cv2.circle(
        annotated,
        center_i,
        3,
        (0, 0, 255),
        -1,
    )

    # Draw the calibrated sweep span so the preview shows the value geometry.
    min_angle_deg: float = math.degrees(gauge_spec.min_angle_rad) % 360.0
    max_angle_deg: float = (min_angle_deg + math.degrees(gauge_spec.sweep_rad)) % 360.0
    min_angle_rad: float = gauge_spec.min_angle_rad
    max_angle_rad: float = gauge_spec.min_angle_rad + gauge_spec.sweep_rad
    min_tip: tuple[int, int] = point_on_circle(min_angle_rad)
    max_tip: tuple[int, int] = point_on_circle(max_angle_rad)
    cv2.line(annotated, center_i, min_tip, (0, 165, 255), 2)
    cv2.line(annotated, center_i, max_tip, (255, 0, 255), 2)
    cv2.putText(
        annotated,
        f"min={min_angle_deg:.0f} deg",
        (min_tip[0] + 6, min_tip[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 165, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"max={max_angle_deg:.0f} deg",
        (max_tip[0] + 6, max_tip[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 0, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"sweep={math.degrees(gauge_spec.sweep_rad):.0f} deg",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if detection is not None:
        needle_len: float = dial_radius_px * 0.9
        tip_x: int = int(round(center_x + detection.unit_dx * needle_len))
        tip_y: int = int(round(center_y + detection.unit_dy * needle_len))
        cv2.line(
            annotated,
            (int(round(center_x)), int(round(center_y))),
            (tip_x, tip_y),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"conf={detection.confidence:.1f}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), annotated)


def run_single_image_baseline(
    config: SingleImageBaselineConfig,
) -> SingleImageBaselineResult:
    """Run the classical baseline against one image and save artifacts."""
    run_name: str = config.run_name or _timestamp_run_name()
    run_dir: Path = config.artifacts_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    specs: dict[str, GaugeSpec] = load_gauge_specs()
    if config.gauge_id not in specs:
        raise ValueError(
            f"Unknown gauge_id '{config.gauge_id}'. Available: {list(specs)}"
        )
    spec: GaugeSpec = specs[config.gauge_id]

    image_bgr: np.ndarray | None = cv2.imread(str(config.image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image '{config.image_path}'.")

    if config.center_x is not None and config.center_y is not None:
        center_xy: tuple[float, float] = (config.center_x, config.center_y)
    else:
        estimated = _estimate_dial_geometry(image_bgr)
        if estimated is None:
            height, width = image_bgr.shape[:2]
            center_xy = (0.5 * width, 0.5 * height)
        else:
            center_xy = estimated[0]

    if config.dial_radius_px is not None:
        dial_radius_px: float = config.dial_radius_px
    else:
        estimated = _estimate_dial_geometry(image_bgr)
        if estimated is None:
            height, width = image_bgr.shape[:2]
            dial_radius_px = 0.45 * float(min(height, width))
        else:
            dial_radius_px = estimated[1]

    detection: NeedleDetection | None = detect_needle_unit_vector(
        image_bgr,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
        gauge_spec=spec,
    )
    predicted_value: float | None = None
    if detection is not None:
        predicted_value = needle_vector_to_value(
            detection.unit_dx,
            detection.unit_dy,
            spec,
        )

    annotated_path: Path = run_dir / f"{config.image_path.stem}_annotated.png"
    _draw_annotation(
        image_bgr,
        gauge_spec=spec,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
        detection=detection,
        output_path=annotated_path,
    )

    result = SingleImageBaselineResult(
        image_path=config.image_path,
        gauge_spec=spec,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
        detection=detection,
        predicted_value=predicted_value,
        annotated_image_path=annotated_path,
    )

    summary_path: Path = run_dir / "summary.json"
    summary_payload: dict[str, Any] = {
        "config": {
            "image_path": str(config.image_path),
            "gauge_id": config.gauge_id,
            "center_x": config.center_x,
            "center_y": config.center_y,
            "dial_radius_px": config.dial_radius_px,
            "artifacts_dir": str(config.artifacts_dir),
            "run_name": config.run_name,
        },
        "gauge_spec": asdict(spec),
        "center_xy": [center_xy[0], center_xy[1]],
        "dial_radius_px": dial_radius_px,
        "detection": None
        if detection is None
        else {
            "unit_dx": detection.unit_dx,
            "unit_dy": detection.unit_dy,
            "confidence": detection.confidence,
        },
        "predicted_value": predicted_value,
        "annotated_image_path": str(annotated_path),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return result
