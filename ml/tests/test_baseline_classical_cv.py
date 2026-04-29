"""Unit and integration tests for the classical CV baseline module."""

from __future__ import annotations

from pathlib import Path
import math

import cv2
import numpy as np
import pytest

import embedded_gauge_reading_tinyml.baseline_classical_cv as classical_cv
import embedded_gauge_reading_tinyml.single_image_baseline as single_image_baseline
from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    GeometryCandidate,
    NeedleDetection,
    _detect_needle_unit_vector_shaft_scan,
    _sample_line_darkness,
    detect_needle_unit_vector,
    detect_needle_unit_vector_with_geometry_fallback,
    evaluate_classical_baseline,
    select_best_geometry_detection,
)
from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, load_gauge_specs
from embedded_gauge_reading_tinyml.single_image_baseline import (
    HOUGH_GEOMETRY_RADIUS_SCALE,
    estimate_dial_geometry,
)


def _make_detection(
    *,
    unit_dx: float,
    unit_dy: float,
    confidence: float,
    peak_ratio: float,
) -> NeedleDetection:
    """Build a compact fake detection for ranking tests."""
    peak_value: float = confidence * peak_ratio
    runner_up_value: float = peak_value / peak_ratio if peak_ratio > 0.0 else 0.0
    return NeedleDetection(
        unit_dx=unit_dx,
        unit_dy=unit_dy,
        confidence=confidence,
        peak_value=peak_value,
        runner_up_value=runner_up_value,
        peak_ratio=peak_ratio,
        peak_margin=peak_value - runner_up_value,
    )


def _unit_vector_for_value(value: float, spec: GaugeSpec) -> tuple[float, float]:
    """Convert one gauge value back into the unit vector on the calibrated sweep."""
    fraction: float = (value - spec.min_value) / (spec.max_value - spec.min_value)
    angle_rad: float = spec.min_angle_rad + (fraction * spec.sweep_rad)
    return math.cos(angle_rad), math.sin(angle_rad)


def test_detect_needle_unit_vector_on_synthetic_image() -> None:
    """Detector should recover a clear synthetic needle direction."""
    # Build a simple dial-like image with a dark needle on a light face.
    #
    # The detector favors dark radial spokes, so the synthetic fixture should
    # match that assumption instead of asking it to invert a bright line.
    image: np.ndarray = np.full((240, 240, 3), 220, dtype=np.uint8)
    center_xy: tuple[int, int] = (120, 120)
    cv2.circle(image, center_xy, 90, (40, 40, 40), 2)

    # Draw a synthetic needle at +45 degrees in image coordinates.
    angle_rad: float = math.pi / 4.0
    tip_x: int = int(center_xy[0] + 70 * math.cos(angle_rad))
    tip_y: int = int(center_xy[1] + 70 * math.sin(angle_rad))
    cv2.line(image, center_xy, (tip_x, tip_y), (10, 10, 10), 3)

    detection = detect_needle_unit_vector(
        image,
        center_xy=(float(center_xy[0]), float(center_xy[1])),
        dial_radius_px=90.0,
    )

    assert detection is not None
    assert detection.confidence >= 0.0

    detected_angle: float = math.atan2(detection.unit_dy, detection.unit_dx)
    assert detected_angle == pytest.approx(angle_rad, abs=0.2)


def test_shaft_scan_detector_prefers_long_colored_needle_over_short_distractor() -> None:
    """The shaft scan should favor a colored needle shaft over a short tick mark."""
    image: np.ndarray = np.full((240, 240, 3), 225, dtype=np.uint8)
    center_xy: tuple[int, int] = (120, 120)

    # Draw the target needle as a long, saturated radial shaft in the valid sweep.
    target_angle_rad: float = math.radians(300.0)
    target_tip_x: int = int(center_xy[0] + 88 * math.cos(target_angle_rad))
    target_tip_y: int = int(center_xy[1] + 88 * math.sin(target_angle_rad))
    cv2.line(image, center_xy, (target_tip_x, target_tip_y), (40, 120, 220), 3)

    # Add a short distractor tick in a different direction so the scan has to
    # prefer the longer, more saturated shaft.
    distractor_angle_rad: float = math.radians(210.0)
    distractor_tip_x: int = int(center_xy[0] + 28 * math.cos(distractor_angle_rad))
    distractor_tip_y: int = int(center_xy[1] + 28 * math.sin(distractor_angle_rad))
    cv2.line(image, center_xy, (distractor_tip_x, distractor_tip_y), (64, 64, 64), 2)

    detection = _detect_needle_unit_vector_shaft_scan(
        image,
        center_xy=(float(center_xy[0]), float(center_xy[1])),
        dial_radius_px=90.0,
    )

    assert detection is not None
    detected_angle: float = math.atan2(detection.unit_dy, detection.unit_dx)
    detected_angle = detected_angle % (2.0 * math.pi)
    assert detected_angle == pytest.approx(target_angle_rad, abs=0.2)


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
            return _make_detection(
                unit_dx=1.0,
                unit_dy=0.0,
                confidence=8.0,
                peak_ratio=1.02,
            )
        if center_xy == secondary.center_xy and dial_radius_px == secondary.dial_radius_px:
            return _make_detection(
                unit_dx=0.0,
                unit_dy=1.0,
                confidence=5.0,
                peak_ratio=1.30,
            )
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
    assert detection.confidence == pytest.approx(5.0)


def test_select_best_geometry_detection_prefers_consensus_cluster_over_outlier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lone high-quality outlier should not beat a small agreeing cluster."""
    image: np.ndarray = np.zeros((16, 16, 3), dtype=np.uint8)
    spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
    candidates = [
        GeometryCandidate(label="cluster_a", center_xy=(4.0, 4.0), dial_radius_px=6.0),
        GeometryCandidate(label="cluster_b", center_xy=(8.0, 4.0), dial_radius_px=6.0),
        GeometryCandidate(label="outlier", center_xy=(12.0, 4.0), dial_radius_px=6.0),
    ]

    def fake_detect(
        image_bgr: np.ndarray,
        *,
        center_xy: tuple[float, float],
        dial_radius_px: float,
        gauge_spec=None,
    ) -> NeedleDetection | None:
        if center_xy == candidates[0].center_xy:
            dx, dy = _unit_vector_for_value(18.0, spec)
            return _make_detection(unit_dx=dx, unit_dy=dy, confidence=5.0, peak_ratio=1.20)
        if center_xy == candidates[1].center_xy:
            dx, dy = _unit_vector_for_value(20.0, spec)
            return _make_detection(unit_dx=dx, unit_dy=dy, confidence=5.0, peak_ratio=1.30)
        if center_xy == candidates[2].center_xy:
            dx, dy = _unit_vector_for_value(45.0, spec)
            return _make_detection(unit_dx=dx, unit_dy=dy, confidence=20.0, peak_ratio=1.40)
        return None

    monkeypatch.setattr(classical_cv, "detect_needle_unit_vector", fake_detect)

    selection = select_best_geometry_detection(
        image,
        candidates=candidates,
        gauge_spec=spec,
    )

    assert selection is not None
    assert selection.candidate.label == "cluster_b"
    assert selection.detection.confidence == pytest.approx(5.0)


def test_run_single_image_baseline_uses_hough_geometry_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default single-image path should stay with the Hough seed."""
    image_path: Path = tmp_path / "frame.png"
    image_bgr: np.ndarray = np.full((32, 32, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image_bgr)

    hough_center_xy: tuple[float, float] = (11.0, 12.0)
    hough_radius_px: float = 9.5

    def fake_estimate_dial_geometry(image_bgr: np.ndarray) -> tuple[tuple[float, float], float] | None:
        return (hough_center_xy, hough_radius_px)

    def fake_detect(
        image_bgr: np.ndarray,
        *,
        center_xy: tuple[float, float],
        dial_radius_px: float,
        gauge_spec=None,
    ) -> NeedleDetection | None:
        if center_xy == hough_center_xy and dial_radius_px == hough_radius_px:
            return _make_detection(
                unit_dx=1.0,
                unit_dy=0.0,
                confidence=6.0,
                peak_ratio=1.40,
            )
        if center_xy == (16.0, 16.0):
            pytest.fail("The default path should not sweep to image-center offsets.")
        return None

    monkeypatch.setattr(single_image_baseline, "_estimate_dial_geometry", fake_estimate_dial_geometry)
    monkeypatch.setattr(single_image_baseline, "detect_needle_unit_vector", fake_detect)

    result = single_image_baseline.run_single_image_baseline(
        single_image_baseline.SingleImageBaselineConfig(
            image_path=image_path,
            artifacts_dir=tmp_path,
            run_name="probe",
        )
    )

    assert result.center_xy[0] == pytest.approx(hough_center_xy[0])
    assert result.center_xy[1] == pytest.approx(hough_center_xy[1])
    assert result.dial_radius_px == pytest.approx(hough_radius_px)
    assert result.detection is not None
    assert result.detection.confidence == pytest.approx(6.0)


def test_run_single_image_baseline_falls_back_to_board_prior_when_hough_is_weak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A weak Hough seed should allow the fixed board prior to win."""
    image_path: Path = tmp_path / "frame.png"
    image_bgr: np.ndarray = np.full((32, 32, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image_bgr)

    hough_center_xy: tuple[float, float] = (11.0, 12.0)
    hough_radius_px: float = 9.5
    board_prior_candidate = classical_cv.board_prior_geometry_candidate(image_bgr)

    def fake_estimate_dial_geometry(
        image_bgr: np.ndarray,
    ) -> tuple[tuple[float, float], float] | None:
        return (hough_center_xy, hough_radius_px)

    def fake_detect(
        image_bgr: np.ndarray,
        *,
        center_xy: tuple[float, float],
        dial_radius_px: float,
        gauge_spec=None,
    ) -> NeedleDetection | None:
        if center_xy == hough_center_xy and dial_radius_px == hough_radius_px:
            return _make_detection(
                unit_dx=1.0,
                unit_dy=0.0,
                confidence=2.0,
                peak_ratio=1.05,
            )
        if (
            center_xy == board_prior_candidate.center_xy
            and dial_radius_px == board_prior_candidate.dial_radius_px
        ):
            return _make_detection(
                unit_dx=0.0,
                unit_dy=1.0,
                confidence=9.0,
                peak_ratio=1.30,
            )
        if center_xy == (16.0, 16.0):
            return _make_detection(
                unit_dx=-1.0,
                unit_dy=0.0,
                confidence=4.0,
                peak_ratio=1.10,
            )
        return None

    monkeypatch.setattr(single_image_baseline, "_estimate_dial_geometry", fake_estimate_dial_geometry)
    monkeypatch.setattr(single_image_baseline, "detect_needle_unit_vector", fake_detect)

    result = single_image_baseline.run_single_image_baseline(
        single_image_baseline.SingleImageBaselineConfig(
            image_path=image_path,
            artifacts_dir=tmp_path,
            run_name="probe_board_prior",
        )
    )

    assert result.center_xy[0] == pytest.approx(board_prior_candidate.center_xy[0])
    assert result.center_xy[1] == pytest.approx(board_prior_candidate.center_xy[1])
    assert result.dial_radius_px == pytest.approx(board_prior_candidate.dial_radius_px)
    assert result.detection is not None
    assert result.detection.confidence == pytest.approx(9.0)


def test_run_single_image_baseline_prefers_board_prior_scan_over_confident_hough(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strong board-prior shaft scan should beat a confident but wrong Hough seed."""
    image_path: Path = tmp_path / "frame.png"
    image_bgr: np.ndarray = np.full((32, 32, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image_bgr)

    hough_center_xy: tuple[float, float] = (11.0, 12.0)
    hough_radius_px: float = 9.5
    board_prior_candidate = classical_cv.board_prior_geometry_candidate(image_bgr)

    def fake_estimate_dial_geometry(
        image_bgr: np.ndarray,
    ) -> tuple[tuple[float, float], float] | None:
        return (hough_center_xy, hough_radius_px)

    def fake_detect(
        image_bgr: np.ndarray,
        *,
        center_xy: tuple[float, float],
        dial_radius_px: float,
        gauge_spec=None,
    ) -> NeedleDetection | None:
        if center_xy == hough_center_xy and dial_radius_px == hough_radius_px:
            return _make_detection(
                unit_dx=1.0,
                unit_dy=0.0,
                confidence=12.0,
                peak_ratio=1.40,
            )
        if center_xy == (16.0, 16.0):
            return _make_detection(
                unit_dx=-1.0,
                unit_dy=0.0,
                confidence=4.0,
                peak_ratio=1.10,
            )
        return None

    def fake_shaft_scan(
        image_bgr: np.ndarray,
        *,
        center_xy: tuple[float, float],
        dial_radius_px: float,
        gauge_spec=None,
    ) -> NeedleDetection | None:
        if (
            center_xy == board_prior_candidate.center_xy
            and dial_radius_px == board_prior_candidate.dial_radius_px
        ):
            return _make_detection(
                unit_dx=0.0,
                unit_dy=1.0,
                confidence=1.5,
                peak_ratio=1.30,
            )
        return None

    monkeypatch.setattr(single_image_baseline, "_estimate_dial_geometry", fake_estimate_dial_geometry)
    monkeypatch.setattr(single_image_baseline, "detect_needle_unit_vector", fake_detect)
    monkeypatch.setattr(
        single_image_baseline,
        "_detect_needle_unit_vector_shaft_scan",
        fake_shaft_scan,
    )

    result = single_image_baseline.run_single_image_baseline(
        single_image_baseline.SingleImageBaselineConfig(
            image_path=image_path,
            artifacts_dir=tmp_path,
            run_name="probe_board_prior_scan",
        )
    )

    assert result.center_xy[0] == pytest.approx(board_prior_candidate.center_xy[0])
    assert result.center_xy[1] == pytest.approx(board_prior_candidate.center_xy[1])
    assert result.dial_radius_px == pytest.approx(board_prior_candidate.dial_radius_px)
    assert result.detection is not None
    assert result.detection.confidence == pytest.approx(1.5)


def test_estimate_dial_geometry_rejects_implausible_off_center_circle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Hough geometry helper should ignore circles that are clearly bogus."""
    image: np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)
    circles: np.ndarray = np.array(
        [
            [
                [90.0, 10.0, 60.0],
                [50.0, 50.0, 30.0],
            ]
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr(
        cv2,
        "HoughCircles",
        lambda *args, **kwargs: circles,
    )

    geometry = estimate_dial_geometry(image)

    assert geometry is not None
    assert geometry[0][0] == pytest.approx(50.0)
    assert geometry[0][1] == pytest.approx(50.0)
    assert geometry[1] == pytest.approx(30.0 * HOUGH_GEOMETRY_RADIUS_SCALE)


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
