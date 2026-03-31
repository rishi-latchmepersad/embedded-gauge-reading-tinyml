"""Tests for the classical baseline experiment runner."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

import embedded_gauge_reading_tinyml.baseline_runner as baseline_runner
from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    ClassicalBaselineResult,
    ClassicalPrediction,
)
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec
from embedded_gauge_reading_tinyml.labels import LabelSummary


def test_run_classical_baseline_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """run_classical_baseline should save a metrics JSON and predictions CSV."""
    spec = GaugeSpec(
        gauge_id="test_gauge",
        min_angle_rad=0.0,
        sweep_rad=3.14159,
        min_value=0.0,
        max_value=100.0,
    )
    label_summary = LabelSummary(
        total_samples=2,
        in_sweep=2,
        out_of_sweep=0,
        min_fraction=0.0,
        max_fraction=1.0,
    )
    result = ClassicalBaselineResult(
        attempted_samples=2,
        successful_samples=1,
        failed_samples=1,
        mae=12.5,
        rmse=13.0,
        predictions=[
            ClassicalPrediction(
                image_path="sample.jpg",
                true_value=42.0,
                predicted_value=50.0,
                abs_error=8.0,
                confidence=99.0,
            )
        ],
    )

    monkeypatch.setattr(
        baseline_runner,
        "load_gauge_specs",
        lambda: {spec.gauge_id: spec},
    )
    monkeypatch.setattr(
        baseline_runner,
        "load_dataset",
        lambda labelled_dir, raw_dir: [object(), object()],
    )
    monkeypatch.setattr(
        baseline_runner,
        "summarize_label_sweep",
        lambda samples, gauge_spec: label_summary,
    )
    monkeypatch.setattr(
        baseline_runner,
        "evaluate_classical_baseline",
        lambda samples, gauge_spec, max_samples=None: result,
    )

    run_result = baseline_runner.run_classical_baseline(
        baseline_runner.ClassicalBaselineRunConfig(
            gauge_id=spec.gauge_id,
            artifacts_dir=tmp_path,
            labelled_dir=tmp_path / "labelled",
            raw_dir=tmp_path / "raw",
            run_name="demo",
            max_samples=1,
        )
    )

    assert run_result.run_dir == tmp_path / "demo"
    assert run_result.spec == spec
    assert run_result.label_summary == label_summary
    assert run_result.result == result
    assert run_result.sample_count == 2

    metrics_path = run_result.run_dir / "metrics.json"
    predictions_path = run_result.run_dir / "predictions.csv"
    assert metrics_path.exists()
    assert predictions_path.exists()

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["config"]["gauge_id"] == "test_gauge"
    assert metrics_payload["result"]["successful_samples"] == 1
    assert metrics_payload["predictions_path"] == str(predictions_path)

    csv_text = predictions_path.read_text(encoding="utf-8")
    assert "image_path,true_value,predicted_value,abs_error,confidence" in csv_text
    assert "sample.jpg,42.0,50.0,8.0,99.0" in csv_text
