"""Tests for the classical baseline failure-report helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from embedded_gauge_reading_tinyml.baseline_classical_cv import ClassicalPrediction
from embedded_gauge_reading_tinyml.baseline_report import write_failure_report


def test_write_failure_report_groups_and_ranks_predictions(tmp_path: Path) -> None:
    """The failure report should rank worst cases and group them by family/value."""
    predictions = [
        ClassicalPrediction(
            image_path="captured_images/capture_p30c.jpg",
            true_value=30.0,
            predicted_value=45.0,
            abs_error=15.0,
            confidence=7.0,
        ),
        ClassicalPrediction(
            image_path="captured_images/today_converted/capture_2026-04-09_06-41-57.png",
            true_value=30.0,
            predicted_value=34.0,
            abs_error=4.0,
            confidence=6.0,
        ),
        ClassicalPrediction(
            image_path="captured_images/capture_m25c.jpg",
            true_value=-25.0,
            predicted_value=10.0,
            abs_error=35.0,
            confidence=3.0,
        ),
    ]

    report = write_failure_report(
        tmp_path,
        predictions,
        attempted_samples=4,
        top_n=2,
        value_bucket_size=10.0,
    )

    assert report.attempted_samples == 4
    assert report.successful_samples == 3
    assert report.failed_samples == 1
    assert report.worst_cases[0].image_path == "captured_images/capture_m25c.jpg"
    assert report.by_image_family[0].group_key == "m25c"
    assert report.by_value_bucket[0].group_key == "-30..-20"

    summary_path = tmp_path / "summary.json"
    worst_cases_path = tmp_path / "worst_cases.csv"
    family_path = tmp_path / "by_image_family.csv"
    bucket_path = tmp_path / "by_value_bucket.csv"

    assert summary_path.exists()
    assert worst_cases_path.exists()
    assert family_path.exists()
    assert bucket_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["attempted_samples"] == 4
    assert summary["worst_cases"][0]["image_path"] == "captured_images/capture_m25c.jpg"

    with worst_cases_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["rank"] == "1"
    assert rows[0]["image_path"] == "captured_images/capture_m25c.jpg"
