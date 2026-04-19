"""Unit and integration tests for the classical CV baseline module."""

from __future__ import annotations

from pathlib import Path
import math

import cv2
import numpy as np
import pytest

import embedded_gauge_reading_tinyml.baseline_classical_cv as classical_cv
from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    GeometryCandidate,
    NeedleDetection,
    _sample_line_darkness,
    detect_needle_unit_vector,
    detect_needle_unit_vector_with_geometry_fallback,
    evaluate_classical_baseline,
)
from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs


def test_detect_needle_unit_vector_on_synthetic_image() -> None:
    """Detector should recover a clear synthetic needle direction."""
    # Build a blank dial-like image with a bright line as the needle.
    image: np.ndarray = np.zeros((240, 240, 3), dtype=np.uint8)
    center_xy: tuple[int, int] = (120, 120)
    cv2.circle(image, center_xy, 90, (40, 40, 40), 2)

    # Draw a synthetic needle at +45 degrees in image coordinates.
    angle_rad: float = math.pi / 4.0
    tip_x: int = int(center_xy[0] + 70 * math.cos(angle_rad))
    tip_y: int = int(center_xy[1] + 70 * math.sin(angle_rad))
    cv2.line(image, center_xy, (tip_x, tip_y), (255, 255, 255), 3)

    detection = detect_needle_unit_vector(
        image,
        center_xy=(float(center_xy[0]), float(center_xy[1])),
        dial_radius_px=90.0,
    )

    assert detection is not None
    assert detection.confidence >= 0.0

    detected_angle: float = math.atan2(detection.unit_dy, detection.unit_dx)
    assert detected_angle == pytest.approx(angle_rad, abs=0.2)


def test_sample_line_darkness_clamps_out_of_bounds_samples() -> None:
    """The line sampler should stay safe when a ray extends past the image."""
    # Build a tiny grayscale frame so the ray needs to sample outside the image.
    gray_image: np.ndarray = np.full((10, 10), 128, dtype=np.uint8)

    contrast_mean, dark_fraction = _sample_line_darkness(
        gray_image,
        x1=1.0,
        y1=1.0,
        x2=100.0,
        y2=100.0,
    )

    assert math.isfinite(contrast_mean)
    assert math.isfinite(dark_fraction)
    assert dark_fraction >= 0.0


def test_detect_needle_unit_vector_with_geometry_fallback_uses_secondary_when_primary_is_weak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The geometry fallback should prefer the secondary geometry on weak primaries."""
    image: np.ndarray = np.zeros((16, 16, 3), dtype=np.uint8)
    primary = GeometryCandidate(
        label="hough",
        center_xy=(1.0, 1.0),
        dial_radius_px=8.0,
    )
    secondary = GeometryCandidate(
        label="image_center",
        center_xy=(8.0, 8.0),
        dial_radius_px=6.0,
    )

    def fake_detect(
        image_bgr: np.ndarray,
        *,
        center_xy: tuple[float, float],
        dial_radius_px: float,
        gauge_spec=None,
    ) -> NeedleDetection | None:
        if center_xy == primary.center_xy and dial_radius_px == primary.dial_radius_px:
            return NeedleDetection(unit_dx=1.0, unit_dy=0.0, confidence=2.0)
        if center_xy == secondary.center_xy and dial_radius_px == secondary.dial_radius_px:
            return NeedleDetection(unit_dx=0.0, unit_dy=1.0, confidence=8.0)
        return None

    monkeypatch.setattr(classical_cv, "detect_needle_unit_vector", fake_detect)

    detection = detect_needle_unit_vector_with_geometry_fallback(
        image,
        primary=primary,
        secondary=secondary,
        confidence_threshold=4.0,
    )

    assert detection is not None
    assert detection.unit_dx == pytest.approx(0.0)
    assert detection.unit_dy == pytest.approx(1.0)
    assert detection.confidence == pytest.approx(8.0)


def test_evaluate_classical_baseline_with_project_data_smoke() -> None:
    """Run a small smoke evaluation on real project data when available."""
    # Resolve dataset/specs from the package defaults.
    samples = load_dataset()
    if not samples:
        pytest.skip("No labelled samples found in ml/data/labelled.")

    specs = load_gauge_specs()
    gauge_id: str = "littlegood_home_temp_gauge_c"
    if gauge_id not in specs:
        pytest.skip(f"Gauge spec '{gauge_id}' not found.")

    # Limit sample count so the test stays fast and deterministic enough for CI.
    result = evaluate_classical_baseline(samples, specs[gauge_id], max_samples=24)

    assert result.attempted_samples > 0
    assert result.successful_samples > 0
    assert result.failed_samples >= 0
    assert result.mae >= 0.0
    assert result.rmse >= 0.0

    # Ensure each prediction records the path and finite values for downstream reporting.
    first = result.predictions[0]
    assert Path(first.image_path).suffix.lower() in {".jpg", ".jpeg", ".png"}
    assert math.isfinite(first.true_value)
    assert math.isfinite(first.predicted_value)
    assert first.abs_error >= 0.0
