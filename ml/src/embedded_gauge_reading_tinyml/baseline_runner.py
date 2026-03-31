"""Experiment runner for the classical CV gauge baseline.

This module turns the existing Canny + Hough baseline into a repeatable
benchmark that loads the labelled dataset, evaluates a chosen gauge, and writes
simple artifact files for later comparison against CNN results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import csv
import json
import math
from pathlib import Path
from typing import Any

from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    ClassicalBaselineResult,
    evaluate_classical_baseline,
)
from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, load_gauge_specs
from embedded_gauge_reading_tinyml.labels import LabelSummary, summarize_label_sweep


ML_ROOT: Path = Path(__file__).resolve().parents[2]
"""Project root resolved from the package location."""

DEFAULT_ARTIFACTS_DIR: Path = ML_ROOT / "artifacts" / "baseline"
"""Default folder where baseline run artifacts are stored."""


@dataclass(frozen=True)
class ClassicalBaselineRunConfig:
    """Configuration for one classical baseline benchmark run."""

    gauge_id: str = "littlegood_home_temp_gauge_c"
    max_samples: int | None = None
    run_name: str = ""
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    labelled_dir: Path = ML_ROOT / "data" / "labelled"
    raw_dir: Path = ML_ROOT / "data" / "raw"


@dataclass(frozen=True)
class ClassicalBaselineRunResult:
    """Returned metadata for a completed baseline benchmark run."""

    run_dir: Path
    spec: GaugeSpec
    label_summary: LabelSummary
    result: ClassicalBaselineResult
    sample_count: int


def _timestamp_run_name() -> str:
    """Build a stable timestamp-based directory name for a fresh run."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_safe_float(value: float) -> float | None:
    """Convert NaN/inf values to JSON-friendly nulls."""
    if math.isfinite(value):
        return float(value)
    return None


def _build_metrics_payload(
    *,
    config: ClassicalBaselineRunConfig,
    spec: GaugeSpec,
    label_summary: LabelSummary,
    result: ClassicalBaselineResult,
    sample_count: int,
    predictions_path: Path,
) -> dict[str, Any]:
    """Assemble the structured JSON payload for a baseline run."""
    return {
        "config": {
            "gauge_id": config.gauge_id,
            "max_samples": config.max_samples,
            "run_name": config.run_name,
            "artifacts_dir": str(config.artifacts_dir),
            "labelled_dir": str(config.labelled_dir),
            "raw_dir": str(config.raw_dir),
        },
        "gauge_spec": asdict(spec),
        "label_summary": asdict(label_summary),
        "sample_count": sample_count,
        "result": {
            "attempted_samples": result.attempted_samples,
            "successful_samples": result.successful_samples,
            "failed_samples": result.failed_samples,
            "mae": _json_safe_float(result.mae),
            "rmse": _json_safe_float(result.rmse),
        },
        "predictions_path": str(predictions_path),
    }


def _write_predictions_csv(
    predictions_path: Path,
    result: ClassicalBaselineResult,
) -> None:
    """Write the per-sample classical baseline predictions to CSV."""
    fieldnames: list[str] = [
        "image_path",
        "true_value",
        "predicted_value",
        "abs_error",
        "confidence",
    ]

    with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in result.predictions:
            writer.writerow(
                {
                    "image_path": prediction.image_path,
                    "true_value": prediction.true_value,
                    "predicted_value": prediction.predicted_value,
                    "abs_error": prediction.abs_error,
                    "confidence": prediction.confidence,
                }
            )


def run_classical_baseline(
    config: ClassicalBaselineRunConfig,
) -> ClassicalBaselineRunResult:
    """Load labelled samples, evaluate the baseline, and save run artifacts."""
    run_name: str = config.run_name or _timestamp_run_name()
    run_dir: Path = config.artifacts_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    specs: dict[str, GaugeSpec] = load_gauge_specs()
    if config.gauge_id not in specs:
        raise ValueError(
            f"Unknown gauge_id '{config.gauge_id}'. Available: {list(specs)}"
        )
    spec: GaugeSpec = specs[config.gauge_id]

    samples = load_dataset(labelled_dir=config.labelled_dir, raw_dir=config.raw_dir)
    if not samples:
        raise ValueError("No samples found. Check labelled/raw paths and annotations.")

    label_summary: LabelSummary = summarize_label_sweep(samples, spec)
    result: ClassicalBaselineResult = evaluate_classical_baseline(
        samples,
        spec,
        max_samples=config.max_samples,
    )

    metrics_path: Path = run_dir / "metrics.json"
    predictions_path: Path = run_dir / "predictions.csv"

    metrics_payload: dict[str, Any] = _build_metrics_payload(
        config=config,
        spec=spec,
        label_summary=label_summary,
        result=result,
        sample_count=len(samples),
        predictions_path=predictions_path,
    )
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_predictions_csv(predictions_path, result)

    return ClassicalBaselineRunResult(
        run_dir=run_dir,
        spec=spec,
        label_summary=label_summary,
        result=result,
        sample_count=len(samples),
    )
