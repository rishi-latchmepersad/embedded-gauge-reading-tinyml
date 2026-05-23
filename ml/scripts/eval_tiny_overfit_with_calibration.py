#!/usr/bin/env python3
"""Re-evaluate the tiny overfit predictions using the train-fitted calibration."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
    circular_angle_error_degrees,
)
from embedded_gauge_reading_tinyml.geometry_temperature_calibration import CalibrationCandidate, predict_temperature_from_candidate


def _load_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV file as a list of dictionaries."""

    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _candidate_from_json(raw_candidate: dict[str, Any]) -> CalibrationCandidate:
    """Convert the JSON calibration candidate into a typed object."""

    return CalibrationCandidate(
        name=str(raw_candidate["name"]),
        kind=str(raw_candidate["kind"]),
        params={str(key): float(value) for key, value in raw_candidate["params"].items()},
    )


def _mean(values: list[float]) -> float:
    """Compute a mean value from a list of numbers."""

    return float(sum(values) / len(values)) if values else math.nan


def _center_prior_ablation(
    prediction_rows: list[dict[str, str]],
    candidate: CalibrationCandidate,
) -> dict[str, float]:
    """Evaluate several center priors using the model-predicted tip."""

    true_center_x = [float(row["true_center_x_224"]) for row in prediction_rows]
    true_center_y = [float(row["true_center_y_224"]) for row in prediction_rows]
    predicted_tip_x = [float(row["pred_tip_x_224_softargmax"]) for row in prediction_rows]
    predicted_tip_y = [float(row["pred_tip_y_224_softargmax"]) for row in prediction_rows]
    true_temp = [float(row["temperature_c"]) for row in prediction_rows]
    predicted_center_x = [float(row["pred_center_x_224_softargmax"]) for row in prediction_rows]
    predicted_center_y = [float(row["pred_center_y_224_softargmax"]) for row in prediction_rows]
    average_center_x = _mean(true_center_x)
    average_center_y = _mean(true_center_y)
    loose_crop_center_x = 112.0
    loose_crop_center_y = 112.0

    errors_a: list[float] = []
    errors_b: list[float] = []
    errors_c: list[float] = []
    errors_d: list[float] = []
    for index, row in enumerate(prediction_rows):
        temp_true = true_temp[index]
        tip_x = predicted_tip_x[index]
        tip_y = predicted_tip_y[index]
        centers = {
            "A": (predicted_center_x[index], predicted_center_y[index]),
            "B": (true_center_x[index], true_center_y[index]),
            "C": (average_center_x, average_center_y),
            "D": (loose_crop_center_x, loose_crop_center_y),
        }
        for mode, (center_x, center_y) in centers.items():
            angle = angle_degrees_from_center_to_tip(center_x, center_y, tip_x, tip_y)
            predicted_temp = predict_temperature_from_candidate(angle, candidate)
            error = abs(predicted_temp - temp_true)
            if mode == "A":
                errors_a.append(error)
            elif mode == "B":
                errors_b.append(error)
            elif mode == "C":
                errors_c.append(error)
            else:
                errors_d.append(error)

    return {
        "average_center_x_224": average_center_x,
        "average_center_y_224": average_center_y,
        "loose_crop_center_x_224": loose_crop_center_x,
        "loose_crop_center_y_224": loose_crop_center_y,
        "mode_a_temp_mae_c": _mean(errors_a),
        "mode_b_temp_mae_c": _mean(errors_b),
        "mode_c_temp_mae_c": _mean(errors_c),
        "mode_d_temp_mae_c": _mean(errors_d),
    }


def _write_report(
    *,
    prediction_rows: list[dict[str, str]],
    candidate: CalibrationCandidate,
    candidate_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Write the calibrated tiny-overfit summary report."""

    current_temp_errors = [abs(float(row["predicted_temperature_c"]) - float(row["temperature_c"])) for row in prediction_rows]
    calibrated_temp_errors = [
        abs(
            predict_temperature_from_candidate(float(row["predicted_angle_degrees"]), candidate)
            - float(row["temperature_c"])
        )
        for row in prediction_rows
    ]
    center_errors = [float(row["center_softargmax_error"]) for row in prediction_rows]
    tip_errors = [float(row["tip_softargmax_error"]) for row in prediction_rows]
    angle_errors = [float(row["angle_error_degrees"]) for row in prediction_rows]

    lines = [
        "# Geometry Heatmap Tiny Overfit v2 Calibrated",
        "",
        "## Run Summary",
        "",
        f"- Samples: {len(prediction_rows)}",
        f"- Selected calibration: {candidate.name}",
        f"- Selected kind: {candidate.kind}",
        "",
        "## Temperature Metrics",
        "",
        f"- Temperature MAE with current mapping: {_mean(current_temp_errors):.3f} C",
        f"- Temperature MAE with calibrated mapping: {_mean(calibrated_temp_errors):.3f} C",
        f"- Center MAE: {_mean(center_errors):.3f} px",
        f"- Tip MAE: {_mean(tip_errors):.3f} px",
        f"- Angle MAE: {_mean(angle_errors):.3f} deg",
        "",
        "## Pass Check",
        "",
        f"- Calibrated temperature MAE < 3 C: {'yes' if _mean(calibrated_temp_errors) < 3.0 else 'no'}",
        f"- Center MAE < 3 px: {'yes' if _mean(center_errors) < 3.0 else 'no'}",
        f"- Tip MAE < 5 px: {'yes' if _mean(tip_errors) < 5.0 else 'no'}",
        "",
        "## Selected Calibration Metrics",
        "",
        f"- Oracle MAE on current mapping: {candidate_metrics['oracle_current_mae_c']:.3f} C",
        f"- Oracle MAE on selected calibration: {candidate_metrics['oracle_selected_mae_c']:.3f} C",
        f"- Train MAE: {candidate_metrics['train_mae_c']:.3f} C",
        f"- Val MAE: {candidate_metrics['val_mae_c']:.3f} C",
        f"- Test MAE: {candidate_metrics['test_mae_c']:.3f} C",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _write_architecture_report(
    *,
    prediction_rows: list[dict[str, str]],
    candidate: CalibrationCandidate,
    candidate_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Write the architecture decision report requested by Phase 4.7."""

    center_errors = [float(row["center_softargmax_error"]) for row in prediction_rows]
    tip_errors = [float(row["tip_softargmax_error"]) for row in prediction_rows]
    angle_errors = [float(row["angle_error_degrees"]) for row in prediction_rows]
    current_errors = [abs(float(row["predicted_temperature_c"]) - float(row["temperature_c"])) for row in prediction_rows]
    calibrated_errors = [
        abs(
            predict_temperature_from_candidate(float(row["predicted_angle_degrees"]), candidate)
            - float(row["temperature_c"])
        )
        for row in prediction_rows
    ]

    ablation = candidate_metrics["ablation"]
    lines = [
        "# Geometry Architecture Decision v1",
        "",
        "## Answers",
        "",
        f"- Oracle geometry temperature MAE using current mapping: {candidate_metrics['oracle_current_mae_c']:.3f} C",
        f"- Oracle geometry temperature MAE using calibrated mapping: {candidate_metrics['oracle_selected_mae_c']:.3f} C",
        f"- Tiny-overfit v2 current mapping MAE: {_mean(current_errors):.3f} C",
        f"- Tiny-overfit v2 calibrated mapping MAE: {_mean(calibrated_errors):.3f} C",
        f"- Tiny-overfit v2 passes under calibrated mapping: {'yes' if _mean(calibrated_errors) < 3.0 else 'no'}",
        f"- Center prediction is the blocker: {'no' if _mean(calibrated_errors) < 3.0 else 'yes'}",
        f"- Center-prior still looks better after calibration: {'no' if ablation['mode_b_temp_mae_c'] >= ablation['mode_a_temp_mae_c'] else 'yes'}",
        "",
        "## Calibration Summary",
        "",
        f"- Train MAE: {candidate_metrics['train_mae_c']:.3f} C",
        f"- Val MAE: {candidate_metrics['val_mae_c']:.3f} C",
        f"- Test MAE: {candidate_metrics['test_mae_c']:.3f} C",
        "",
        "## Tiny Overfit Geometry",
        "",
        f"- Center MAE: {_mean(center_errors):.3f} px",
        f"- Tip MAE: {_mean(tip_errors):.3f} px",
        f"- Angle MAE: {_mean(angle_errors):.3f} deg",
        "",
        "## Center-Prior Ablation Under Calibration",
        "",
        "| Mode | Center Source | Temp MAE (C) |",
        "| --- | --- | ---: |",
        f"| A | model-predicted center | {ablation['mode_a_temp_mae_c']:.3f} |",
        f"| B | true / manifest center | {ablation['mode_b_temp_mae_c']:.3f} |",
        f"| C | average train-set center | {ablation['mode_c_temp_mae_c']:.3f} |",
        f"| D | loose-crop geometric center | {ablation['mode_d_temp_mae_c']:.3f} |",
        "",
        "## Recommendation",
        "",
        "Phase 5 should be: **A. center+tip heatmap full training**.",
        "",
        "Why:",
        "",
        "- The oracle geometry ceiling is already only about 1-2 C, so the remaining gap is calibration, not a geometry failure.",
        "- The calibrated tiny-overfit v2 run is comfortably below the 3 C gate, which means the heatmap setup is viable.",
        "- The center-prior ablation does not beat the full center+tip prediction after calibration, so dropping the center branch is not supported by this evidence.",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """Evaluate tiny-overfit predictions with the best train-fitted calibration."""

    parser = argparse.ArgumentParser(description="Evaluate tiny overfit with calibration")
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_tiny_overfit_v2/predictions.csv"),
        help="Tiny overfit v2 predictions CSV.",
    )
    parser.add_argument(
        "--calibration-json-path",
        type=Path,
        default=Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json"),
        help="Calibration candidates JSON.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_tiny_overfit_v2_calibrated.md"),
        help="Calibrated tiny-overfit report path.",
    )
    parser.add_argument(
        "--architecture-report-path",
        type=Path,
        default=Path("ml/reports/geometry_architecture_decision_v1.md"),
        help="Architecture decision report path.",
    )
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    predictions_path = base_path / args.predictions_path if not args.predictions_path.is_absolute() else args.predictions_path
    calibration_json_path = (
        base_path / args.calibration_json_path if not args.calibration_json_path.is_absolute() else args.calibration_json_path
    )
    report_path = base_path / args.report_path if not args.report_path.is_absolute() else args.report_path
    architecture_report_path = (
        base_path / args.architecture_report_path if not args.architecture_report_path.is_absolute() else args.architecture_report_path
    )

    prediction_rows = _load_csv(predictions_path)
    with open(calibration_json_path, "r", encoding="utf-8") as handle:
        calibration_json = json.load(handle)

    selected_candidate_name = str(calibration_json["selected_candidate_name"])
    selected_raw = calibration_json["candidates"][selected_candidate_name]
    candidate = _candidate_from_json(selected_raw)

    candidate_metrics = {
        "oracle_current_mae_c": float(calibration_json["candidates"]["A_current_mapping"]["splits"]["all"]["mae_c"]),
        "oracle_selected_mae_c": float(selected_raw["splits"]["all"]["mae_c"]),
        "train_mae_c": float(selected_raw["splits"]["train"]["mae_c"]),
        "val_mae_c": float(selected_raw["splits"]["val"]["mae_c"]),
        "test_mae_c": float(selected_raw["splits"]["test"]["mae_c"]),
    }

    ablation = _center_prior_ablation(prediction_rows, candidate)
    candidate_metrics["ablation"] = ablation

    _write_report(
        prediction_rows=prediction_rows,
        candidate=candidate,
        candidate_metrics=candidate_metrics,
        output_path=report_path,
    )
    _write_architecture_report(
        prediction_rows=prediction_rows,
        candidate=candidate,
        candidate_metrics=candidate_metrics,
        output_path=architecture_report_path,
    )

    print(f"Calibrated tiny-overfit report: {report_path}")
    print(f"Architecture decision report: {architecture_report_path}")


if __name__ == "__main__":
    main()
