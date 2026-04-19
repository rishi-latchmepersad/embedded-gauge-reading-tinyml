"""Helpers for ranking and grouping classical-baseline failures.

The hard-case benchmark is most useful when it surfaces the worst examples and
groups them in ways that make debugging easier, so this module turns a list of
per-sample predictions into a small set of report artifacts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from embedded_gauge_reading_tinyml.baseline_classical_cv import ClassicalPrediction


@dataclass(frozen=True)
class GroupFailureSummary:
    """Aggregate failure metrics for one bucket or family."""

    group_key: str
    sample_count: int
    mae: float
    rmse: float
    max_abs_error: float


@dataclass(frozen=True)
class ClassicalFailureReport:
    """Structured summary for a classical-baseline failure report."""

    attempted_samples: int
    successful_samples: int
    failed_samples: int
    mae: float
    rmse: float
    value_bucket_size: float
    worst_cases: list[ClassicalPrediction]
    by_image_family: list[GroupFailureSummary]
    by_value_bucket: list[GroupFailureSummary]


def _json_safe_float(value: float) -> float | None:
    """Convert infinities and NaNs into JSON-friendly nulls."""
    if math.isfinite(value):
        return float(value)
    return None


def _image_family_key(image_path: str) -> str:
    """Derive a short family label from a manifest image path.

    The hard-case captures are named with a shared `capture_` prefix, so we use
    the first token after that prefix to group related shots like `p30c` or
    `2026-04-09`.
    """
    path = Path(image_path)
    stem = path.stem
    if stem.startswith("capture_"):
        stem = stem.removeprefix("capture_")
    if "_" in stem:
        stem = stem.split("_", 1)[0]
    if stem:
        return stem
    if path.parent.name:
        return path.parent.name
    return "(root)"


def _value_bucket_label(value: float, bucket_size: float) -> str:
    """Bucket a value into a human-friendly inclusive range label."""
    if bucket_size <= 0.0:
        raise ValueError("bucket_size must be positive.")
    lower: float = math.floor(value / bucket_size) * bucket_size
    upper: float = lower + bucket_size
    return f"{lower:.0f}..{upper:.0f}"


def _group_predictions(
    predictions: Sequence[ClassicalPrediction],
    *,
    key_fn: Callable[[ClassicalPrediction], str],
) -> list[GroupFailureSummary]:
    """Aggregate error stats for a sequence of predictions."""
    grouped_errors: dict[str, list[float]] = defaultdict(list)
    for prediction in predictions:
        grouped_errors[key_fn(prediction)].append(prediction.abs_error)

    summaries: list[GroupFailureSummary] = []
    for group_key, errors in grouped_errors.items():
        error_array: np.ndarray = np.asarray(errors, dtype=np.float32)
        summaries.append(
            GroupFailureSummary(
                group_key=group_key,
                sample_count=len(errors),
                mae=float(np.mean(error_array)),
                rmse=float(np.sqrt(np.mean(np.square(error_array)))),
                max_abs_error=float(np.max(error_array)),
            )
        )

    # Sort by the worst average error first so the report highlights the most
    # painful groups at the top of the CSV.
    summaries.sort(key=lambda item: (item.mae, item.max_abs_error), reverse=True)
    return summaries


def _write_predictions_csv(
    predictions_path: Path,
    predictions: Sequence[ClassicalPrediction],
) -> None:
    """Write the raw per-sample predictions to CSV."""
    fieldnames: list[str] = [
        "image_path",
        "true_value",
        "predicted_value",
        "abs_error",
        "confidence",
    ]
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(
                {
                    "image_path": prediction.image_path,
                    "true_value": prediction.true_value,
                    "predicted_value": prediction.predicted_value,
                    "abs_error": prediction.abs_error,
                    "confidence": prediction.confidence,
                }
            )


def _write_group_csv(
    output_path: Path,
    group_summaries: Sequence[GroupFailureSummary],
) -> None:
    """Write one grouped summary table to CSV."""
    fieldnames: list[str] = [
        "group_key",
        "sample_count",
        "mae",
        "rmse",
        "max_abs_error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in group_summaries:
            writer.writerow(asdict(summary))


def _write_worst_cases_csv(
    output_path: Path,
    worst_cases: Sequence[ClassicalPrediction],
) -> None:
    """Write the top ranked failures to CSV with an explicit rank column."""
    fieldnames: list[str] = [
        "rank",
        "image_path",
        "true_value",
        "predicted_value",
        "abs_error",
        "confidence",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, prediction in enumerate(worst_cases, start=1):
            writer.writerow(
                {
                    "rank": index,
                    "image_path": prediction.image_path,
                    "true_value": prediction.true_value,
                    "predicted_value": prediction.predicted_value,
                    "abs_error": prediction.abs_error,
                    "confidence": prediction.confidence,
                }
            )


def build_failure_report(
    predictions: Sequence[ClassicalPrediction],
    *,
    attempted_samples: int | None = None,
    top_n: int = 10,
    value_bucket_size: float = 10.0,
) -> ClassicalFailureReport:
    """Build a structured report from the raw predictions."""
    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    successful_samples: int = len(predictions)
    attempted = attempted_samples if attempted_samples is not None else successful_samples
    failed_samples: int = max(attempted - successful_samples, 0)

    if successful_samples == 0:
        return ClassicalFailureReport(
            attempted_samples=attempted,
            successful_samples=0,
            failed_samples=failed_samples,
            mae=float("nan"),
            rmse=float("nan"),
            value_bucket_size=value_bucket_size,
            worst_cases=[],
            by_image_family=[],
            by_value_bucket=[],
        )

    error_array: np.ndarray = np.asarray([p.abs_error for p in predictions], dtype=np.float32)
    ranked_predictions: list[ClassicalPrediction] = sorted(
        predictions,
        key=lambda item: item.abs_error,
        reverse=True,
    )
    worst_cases: list[ClassicalPrediction] = ranked_predictions[:top_n]
    family_summaries: list[GroupFailureSummary] = _group_predictions(
        predictions,
        key_fn=lambda prediction: _image_family_key(prediction.image_path),
    )
    bucket_summaries: list[GroupFailureSummary] = _group_predictions(
        predictions,
        key_fn=lambda prediction: _value_bucket_label(prediction.true_value, value_bucket_size),
    )

    return ClassicalFailureReport(
        attempted_samples=attempted,
        successful_samples=successful_samples,
        failed_samples=failed_samples,
        mae=float(np.mean(error_array)),
        rmse=float(np.sqrt(np.mean(np.square(error_array)))),
        value_bucket_size=value_bucket_size,
        worst_cases=worst_cases,
        by_image_family=family_summaries,
        by_value_bucket=bucket_summaries,
    )


def write_failure_report(
    report_dir: Path,
    predictions: Sequence[ClassicalPrediction],
    *,
    attempted_samples: int | None = None,
    top_n: int = 10,
    value_bucket_size: float = 10.0,
) -> ClassicalFailureReport:
    """Write the structured failure report artifacts and return the summary."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report: ClassicalFailureReport = build_failure_report(
        predictions,
        attempted_samples=attempted_samples,
        top_n=top_n,
        value_bucket_size=value_bucket_size,
    )

    summary_path: Path = report_dir / "summary.json"
    summary_payload = {
        "attempted_samples": report.attempted_samples,
        "successful_samples": report.successful_samples,
        "failed_samples": report.failed_samples,
        "mae": _json_safe_float(report.mae),
        "rmse": _json_safe_float(report.rmse),
        "value_bucket_size": report.value_bucket_size,
        "worst_cases": [asdict(prediction) for prediction in report.worst_cases],
        "by_image_family": [asdict(summary) for summary in report.by_image_family],
        "by_value_bucket": [asdict(summary) for summary in report.by_value_bucket],
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    _write_predictions_csv(report_dir / "predictions.csv", predictions)
    _write_group_csv(report_dir / "by_image_family.csv", report.by_image_family)
    _write_group_csv(report_dir / "by_value_bucket.csv", report.by_value_bucket)
    _write_worst_cases_csv(report_dir / "worst_cases.csv", report.worst_cases)
    return report
