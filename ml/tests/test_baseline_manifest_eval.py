"""Tests for shared classical-baseline manifest evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import embedded_gauge_reading_tinyml.baseline_manifest_eval as manifest_eval
from embedded_gauge_reading_tinyml.baseline_classical_cv import NeedleDetection
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs


def test_detect_with_geometry_mode_hough_only_returns_none_without_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pure Hough strategy should fail cleanly when no circle is detected."""
    image_bgr: np.ndarray = np.zeros((16, 16, 3), dtype=np.uint8)

    monkeypatch.setattr(manifest_eval, "estimate_dial_geometry", lambda _: None)

    called: list[bool] = []

    def fake_detect(
        image_bgr: np.ndarray,
        *,
        center_xy: tuple[float, float],
        dial_radius_px: float,
        gauge_spec=None,
    ) -> NeedleDetection | None:
        called.append(True)
        return NeedleDetection(unit_dx=1.0, unit_dy=0.0, confidence=1.0)

    monkeypatch.setattr(manifest_eval, "detect_needle_unit_vector", fake_detect)

    spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
    detection = manifest_eval._detect_with_geometry_mode(
        image_bgr,
        spec=spec,
        config=manifest_eval.GeometryEvaluationConfig(mode="hough_only"),
    )

    assert detection is None
    assert not called


def test_evaluate_manifest_counts_attempts_and_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest evaluator should parse rows and produce predictions."""
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("image_path,value\ncaptured_images/example.png,12.5\n", encoding="utf-8")

    monkeypatch.setattr(
        manifest_eval,
        "_load_image",
        lambda path: np.zeros((12, 12, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        manifest_eval,
        "_detect_with_geometry_mode",
        lambda image_bgr, *, spec, config: NeedleDetection(
            unit_dx=1.0,
            unit_dy=0.0,
            confidence=7.0,
        ),
    )

    spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
    result = manifest_eval.evaluate_manifest(
        manifest_path,
        spec,
        config=manifest_eval.GeometryEvaluationConfig(mode="center_only"),
        repo_root=tmp_path,
    )

    assert result.attempted_samples == 1
    assert result.result.successful_samples == 1
    assert result.result.failed_samples == 0
    assert len(result.result.predictions) == 1
    assert result.result.predictions[0].image_path.endswith("captured_images/example.png")
