#!/usr/bin/env python3
"""Evaluate geometry heatmap v2 under deterministic crop jitter levels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import (
    decode_prediction_row,
    load_clean_geometry_examples,
    load_heatmap_sample,
    load_selected_calibration_candidate,
    sample_jitter_params,
    select_examples_from_split,
)


JITTER_LEVELS = {
    "identity": {
        "shift_min_px": 0,
        "shift_max_px": 0,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "aspect_min": 1.0,
        "aspect_max": 1.0,
    },
    "mild": {
        "shift_min_px": 4,
        "shift_max_px": 4,
        "scale_min": 0.97,
        "scale_max": 1.03,
        "aspect_min": 0.98,
        "aspect_max": 1.02,
    },
    "medium": {
        "shift_min_px": 8,
        "shift_max_px": 8,
        "scale_min": 0.93,
        "scale_max": 1.08,
        "aspect_min": 0.95,
        "aspect_max": 1.05,
    },
    "strong": {
        "shift_min_px": 12,
        "shift_max_px": 12,
        "scale_min": 0.90,
        "scale_max": 1.15,
        "aspect_min": 0.95,
        "aspect_max": 1.08,
    },
}


def _write_json(data: dict[str, Any], output_path: Path) -> None:
    """Write a JSON file with stable formatting."""

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write prediction rows to CSV, dropping large array fields."""

    if not rows:
        raise ValueError("Cannot write an empty prediction table.")
    skip_keys = {"pred_center_heatmap_array", "pred_tip_heatmap_array"}
    fieldnames = [key for key in rows[0].keys() if key not in skip_keys]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize one jitter level."""

    calibrated_errors = np.asarray([float(row["absolute_error_c_calibrated"]) for row in rows], dtype=np.float64)
    center_errors = np.asarray([float(row["center_px_mae_224"]) for row in rows], dtype=np.float64)
    tip_errors = np.asarray([float(row["tip_px_mae_224"]) for row in rows], dtype=np.float64)
    center_peaks = np.asarray([float(row["center_heatmap_peak_value"]) for row in rows], dtype=np.float64)
    tip_peaks = np.asarray([float(row["tip_heatmap_peak_value"]) for row in rows], dtype=np.float64)
    confidences = np.asarray([float(row["confidence"]) for row in rows], dtype=np.float64)

    return {
        "count": float(len(rows)),
        "temperature_mae_c_calibrated": float(np.mean(calibrated_errors)),
        "worst_error_c": float(np.max(calibrated_errors)),
        "percentage_under_5c": float(np.mean(calibrated_errors < 5.0) * 100.0),
        "percentage_under_10c": float(np.mean(calibrated_errors < 10.0) * 100.0),
        "center_px_mae_224": float(np.mean(center_errors)),
        "tip_px_mae_224": float(np.mean(tip_errors)),
        "center_heatmap_peak_mean": float(np.mean(center_peaks)),
        "center_heatmap_peak_median": float(np.median(center_peaks)),
        "tip_heatmap_peak_mean": float(np.mean(tip_peaks)),
        "tip_heatmap_peak_median": float(np.median(tip_peaks)),
        "confidence_mean": float(np.mean(confidences)),
        "confidence_median": float(np.median(confidences)),
    }


def _write_report(
    *,
    report_path: Path,
    level_metrics: dict[str, dict[str, float]],
    model_path: Path,
    selected_stage: str,
) -> None:
    """Write the markdown robustness summary."""

    lines = [
        "# Geometry Heatmap v2 Jitter Robustness",
        "",
        "## Run Summary",
        "",
        f"- Model: `{model_path}`",
        f"- Selected stage: {selected_stage}",
        "",
        "| level | count | calibrated_mae | worst_error | under_5c_% | under_10c_% | center_mae | tip_mae | center_peak_mean | tip_peak_mean | confidence_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for level in ["identity", "mild", "medium", "strong"]:
        metrics = level_metrics[level]
        lines.append(
            "| {level} | {count:.0f} | {temperature_mae_c_calibrated:.3f} | {worst_error_c:.3f} | {percentage_under_5c:.1f} | {percentage_under_10c:.1f} | {center_px_mae_224:.3f} | {tip_px_mae_224:.3f} | {center_heatmap_peak_mean:.4f} | {tip_heatmap_peak_mean:.4f} | {confidence_mean:.4f} |".format(
                level=level, **metrics
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Identity crops establish the clean baseline for board-style replay.",
            "- Mild, medium, and strong jitter show whether the crop pipeline is robust or brittle.",
            "- If the strong-jitter MAE remains close to the identity crop MAE, the heatmap model is tolerant to practical localizer noise.",
            "",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """Run the calibrated jitter robustness probe."""

    parser = argparse.ArgumentParser(description="Evaluate jitter robustness for geometry heatmap v2")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2/model.keras"),
        help="Trained heatmap model artifact.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"),
        help="Clean geometry manifest.",
    )
    parser.add_argument(
        "--calibration-json-path",
        type=Path,
        default=Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json"),
        help="Calibration artifact from Phase 4.7.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2/jitter_robustness_predictions.csv"),
        help="CSV path for jitter robustness predictions.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_v2_jitter_robustness.md"),
        help="Markdown report path.",
    )
    parser.add_argument("--heatmap-size", type=int, default=56, help="Heatmap size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    model_path = base_path / args.model_path if not args.model_path.is_absolute() else args.model_path
    manifest_path = base_path / args.manifest_path if not args.manifest_path.is_absolute() else args.manifest_path
    calibration_json_path = (
        base_path / args.calibration_json_path if not args.calibration_json_path.is_absolute() else args.calibration_json_path
    )
    output_csv = base_path / args.output_csv if not args.output_csv.is_absolute() else args.output_csv
    report_path = base_path / args.report_path if not args.report_path.is_absolute() else args.report_path
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(model_path, compile=False)
    calibration_candidate, _ = load_selected_calibration_candidate(calibration_json_path)

    examples = load_clean_geometry_examples(manifest_path)
    test_examples = select_examples_from_split(examples, split="test")

    rows: list[dict[str, Any]] = []
    level_metrics: dict[str, dict[str, float]] = {}

    for level_index, (level_name, jitter_spec) in enumerate(JITTER_LEVELS.items()):
        level_rows: list[dict[str, Any]] = []
        for example_index, example in enumerate(test_examples):
            if level_name == "identity":
                sample = load_heatmap_sample(example, base_path, heatmap_size=args.heatmap_size, sigma_pixels=5.0, jitter=None)
            else:
                jitter_rng = np.random.default_rng(args.seed + level_index * 1000 + example_index)
                jitter = sample_jitter_params(jitter_rng, **jitter_spec)
                sample = load_heatmap_sample(example, base_path, heatmap_size=args.heatmap_size, sigma_pixels=5.0, jitter=jitter)

            prediction = model.predict(sample.crop_image[np.newaxis, ...], verbose=0)
            row = decode_prediction_row(
                sample,
                prediction[0][0],
                prediction[1][0],
                float(np.ravel(prediction[2][0])[0]),
                calibration_candidate=calibration_candidate,
            )
            row["jitter_level"] = level_name
            level_rows.append(row)
            rows.append(row)

        level_metrics[level_name] = _summarize_rows(level_rows)

    _write_csv(rows, output_csv)

    selected_stage = "unknown"
    config_path = base_path / "ml/artifacts/training/geometry_heatmap_v2/config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        selected_stage = str(config.get("selected_stage", config.get("stage", "unknown")))

    report_payload = {
        "model_path": str(model_path),
        "selected_stage": selected_stage,
        "level_metrics": level_metrics,
    }
    _write_json(report_payload, output_csv.with_name("jitter_robustness_summary.json"))
    _write_report(report_path=report_path, level_metrics=level_metrics, model_path=model_path, selected_stage=selected_stage)

    print(f"Jitter robustness CSV: {output_csv}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
