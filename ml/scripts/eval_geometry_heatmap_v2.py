#!/usr/bin/env python3
"""Evaluate the full geometry heatmap v2 model on train/val/test splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import (
    decode_prediction_row,
    load_clean_geometry_examples,
    load_heatmap_samples,
    load_selected_calibration_candidate,
    select_examples_from_split,
    write_prediction_overlay,
)


BASELINE_GEOMETRY_POINTS_V1 = {
    "test_temperature_mae_c": 7.91,
    "test_center_mae_px": 11.30,
    "test_tip_mae_px": 21.82,
}
ORACLE_CALIBRATED_GEOMETRY_MAE_C = 1.195


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
    """Reduce decoded rows to the split metrics requested by the phase."""

    center_errors = np.asarray([float(row["center_px_mae_224"]) for row in rows], dtype=np.float64)
    tip_errors = np.asarray([float(row["tip_px_mae_224"]) for row in rows], dtype=np.float64)
    angle_errors = np.asarray([float(row["angle_mae_degrees"]) for row in rows], dtype=np.float64)
    current_errors = np.asarray(
        [float(row["absolute_error_c_current_mapping"]) for row in rows],
        dtype=np.float64,
    )
    calibrated_errors = np.asarray(
        [float(row["absolute_error_c_calibrated"]) for row in rows],
        dtype=np.float64,
    )
    center_peaks = np.asarray([float(row["center_heatmap_peak_value"]) for row in rows], dtype=np.float64)
    tip_peaks = np.asarray([float(row["tip_heatmap_peak_value"]) for row in rows], dtype=np.float64)
    confidences = np.asarray([float(row["confidence"]) for row in rows], dtype=np.float64)

    return {
        "count": float(len(rows)),
        "center_px_mae_224": float(np.mean(center_errors)),
        "tip_px_mae_224": float(np.mean(tip_errors)),
        "angle_mae_degrees": float(np.mean(angle_errors)),
        "temperature_mae_c_current_mapping": float(np.mean(current_errors)),
        "temperature_mae_c_calibrated": float(np.mean(calibrated_errors)),
        "temperature_rmse_c_calibrated": float(np.sqrt(np.mean(np.square(calibrated_errors)))),
        "percentage_under_2c_calibrated": float(np.mean(calibrated_errors < 2.0) * 100.0),
        "percentage_under_5c_calibrated": float(np.mean(calibrated_errors < 5.0) * 100.0),
        "percentage_under_10c_calibrated": float(np.mean(calibrated_errors < 10.0) * 100.0),
        "center_heatmap_peak_mean": float(np.mean(center_peaks)),
        "center_heatmap_peak_median": float(np.median(center_peaks)),
        "tip_heatmap_peak_mean": float(np.mean(tip_peaks)),
        "tip_heatmap_peak_median": float(np.median(tip_peaks)),
        "confidence_mean": float(np.mean(confidences)),
        "confidence_median": float(np.median(confidences)),
        "worst_calibrated_error_c": float(np.max(calibrated_errors)),
    }


def _predict_split_rows(
    model: keras.Model,
    samples: list[Any],
    *,
    calibration_candidate: Any,
) -> list[dict[str, Any]]:
    """Run inference on one split and decode the predictions."""

    x = np.stack([sample.crop_image for sample in samples], axis=0).astype(np.float32)
    predictions = model.predict(x, verbose=0)
    center_batch = np.asarray(predictions[0], dtype=np.float32)
    tip_batch = np.asarray(predictions[1], dtype=np.float32)
    confidence_batch = np.asarray(predictions[2], dtype=np.float32)

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        row = decode_prediction_row(
            sample,
            center_batch[index],
            tip_batch[index],
            float(np.ravel(confidence_batch[index])[0]),
            calibration_candidate=calibration_candidate,
        )
        rows.append(row)
    return rows


def _write_report(
    *,
    model_path: Path,
    selected_stage: str,
    candidate_name: str,
    candidate_kind: str,
    split_metrics: dict[str, dict[str, float]],
    worst_rows: list[dict[str, Any]],
    report_path: Path,
) -> None:
    """Write the markdown evaluation report."""

    overall = split_metrics["test"]
    baseline_gap = overall["temperature_mae_c_calibrated"] - BASELINE_GEOMETRY_POINTS_V1["test_temperature_mae_c"]
    oracle_gap = overall["temperature_mae_c_calibrated"] - ORACLE_CALIBRATED_GEOMETRY_MAE_C
    passes_great = overall["temperature_mae_c_calibrated"] < 5.0 and overall["tip_px_mae_224"] < 12.0
    passes_good = 5.0 <= overall["temperature_mae_c_calibrated"] <= 7.91 and overall["tip_px_mae_224"] < 21.82
    board_style_ok = passes_great or passes_good

    lines = [
        "# Geometry Heatmap v2 Evaluation",
        "",
        "## Run Summary",
        "",
        f"- Model: `{model_path}`",
        f"- Selected stage: {selected_stage}",
        f"- Calibration candidate: {candidate_name} ({candidate_kind})",
        f"- Oracle calibrated geometry ceiling: {ORACLE_CALIBRATED_GEOMETRY_MAE_C:.3f} C",
        "",
        "## Split Metrics",
        "",
        "| split | count | center_px_mae_224 | tip_px_mae_224 | angle_mae_degrees | temp_mae_current | temp_mae_calibrated | rmse_calibrated | under_2c_% | under_5c_% | under_10c_% | center_peak_mean | center_peak_median | tip_peak_mean | tip_peak_median | confidence_mean | confidence_median |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ["train", "val", "test"]:
        metrics = split_metrics[split]
        lines.append(
            "| {split} | {count:.0f} | {center_px_mae_224:.3f} | {tip_px_mae_224:.3f} | {angle_mae_degrees:.3f} | {temperature_mae_c_current_mapping:.3f} | {temperature_mae_c_calibrated:.3f} | {temperature_rmse_c_calibrated:.3f} | {percentage_under_2c_calibrated:.1f} | {percentage_under_5c_calibrated:.1f} | {percentage_under_10c_calibrated:.1f} | {center_heatmap_peak_mean:.4f} | {center_heatmap_peak_median:.4f} | {tip_heatmap_peak_mean:.4f} | {tip_heatmap_peak_median:.4f} | {confidence_mean:.4f} | {confidence_median:.4f} |".format(
                split=split, **metrics
            )
        )

    lines.extend(
        [
            "",
            "## Baseline Comparison",
            "",
            f"- geometry_points_v1 test temperature MAE: {BASELINE_GEOMETRY_POINTS_V1['test_temperature_mae_c']:.2f} C",
            f"- geometry_points_v1 test center MAE: {BASELINE_GEOMETRY_POINTS_V1['test_center_mae_px']:.2f} px",
            f"- geometry_points_v1 test tip MAE: {BASELINE_GEOMETRY_POINTS_V1['test_tip_mae_px']:.2f} px",
            f"- Heatmap v2 test calibrated MAE: {overall['temperature_mae_c_calibrated']:.3f} C",
            f"- Heatmap v2 test tip MAE: {overall['tip_px_mae_224']:.3f} px",
            f"- Heatmap v2 test center MAE: {overall['center_px_mae_224']:.3f} px",
            f"- Heatmap v2 is within {oracle_gap:.3f} C of the oracle geometry ceiling.",
            "",
            "## Decision Checks",
            "",
            f"- Beats geometry_points_v1 test MAE: {'yes' if overall['temperature_mae_c_calibrated'] < BASELINE_GEOMETRY_POINTS_V1['test_temperature_mae_c'] else 'no'}",
            f"- Beats geometry_points_v1 tip MAE: {'yes' if overall['tip_px_mae_224'] < BASELINE_GEOMETRY_POINTS_V1['test_tip_mae_px'] else 'no'}",
            f"- Reduces catastrophic errors: {'yes' if overall['percentage_under_10c_calibrated'] >= 95.0 else 'partially' if overall['percentage_under_10c_calibrated'] >= 90.0 else 'no'}",
            f"- Good enough for board-style replay: {'yes' if board_style_ok else 'no'}",
            "",
            "## Worst 30 Predictions",
            "",
            "| image | split | abs_err_calibrated | temp_true | temp_calibrated | center_err | tip_err | confidence |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in worst_rows:
        lines.append(
            f"| {Path(str(row['image_path'])).name} | {row['split']} | {float(row['absolute_error_c_calibrated']):.3f} | {float(row['true_temperature_c']):.2f} | {float(row['predicted_temperature_c_calibrated']):.2f} | {float(row['center_px_mae_224']):.2f} | {float(row['tip_px_mae_224']):.2f} | {float(row['confidence']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- The calibrated temperature gap versus the oracle ceiling is {overall['temperature_mae_c_calibrated'] - ORACLE_CALIBRATED_GEOMETRY_MAE_C:.3f} C on test.",
            f"- The model is {'well below' if board_style_ok else 'not yet below'} the coordinate baseline on the metrics that matter for board-style replay.",
            "",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """Evaluate the selected geometry heatmap v2 model."""

    parser = argparse.ArgumentParser(description="Evaluate geometry heatmap v2")
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
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2"),
        help="Directory to store prediction CSVs.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("ml/debug/geometry_heatmap_v2_predictions"),
        help="Directory for overlay images.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_v2_eval.md"),
        help="Markdown report path.",
    )
    parser.add_argument("--heatmap-size", type=int, default=56, help="Heatmap size.")
    parser.add_argument("--overlay-seed", type=int, default=42, help="Overlay sampling seed.")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    model_path = base_path / args.model_path if not args.model_path.is_absolute() else args.model_path
    manifest_path = base_path / args.manifest_path if not args.manifest_path.is_absolute() else args.manifest_path
    calibration_json_path = (
        base_path / args.calibration_json_path if not args.calibration_json_path.is_absolute() else args.calibration_json_path
    )
    output_dir = base_path / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    debug_dir = base_path / args.debug_dir if not args.debug_dir.is_absolute() else args.debug_dir
    report_path = base_path / args.report_path if not args.report_path.is_absolute() else args.report_path

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(model_path, compile=False)
    calibration_candidate, calibration_json = load_selected_calibration_candidate(calibration_json_path)
    selected_stage = "unknown"
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            selected_config = json.load(handle)
        selected_stage = str(selected_config.get("selected_stage", selected_config.get("stage", "unknown")))

    examples = load_clean_geometry_examples(manifest_path)
    split_examples = {
        split: select_examples_from_split(examples, split=split)
        for split in ("train", "val", "test")
    }

    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_predictions_csvs: dict[str, Path] = {}
    split_metrics: dict[str, dict[str, float]] = {}

    for split, examples_for_split in split_examples.items():
        samples = load_heatmap_samples(examples_for_split, base_path, heatmap_size=args.heatmap_size, sigma_pixels=5.0, jitter=None)
        rows = _predict_split_rows(model, samples, calibration_candidate=calibration_candidate)
        split_rows[split] = rows
        split_metrics[split] = _summarize_rows(rows)
        split_predictions_csvs[split] = output_dir / f"{split}_predictions.csv"
        _write_csv(rows, split_predictions_csvs[split])

    all_rows = split_rows["train"] + split_rows["val"] + split_rows["test"]
    worst_rows = sorted(all_rows, key=lambda row: float(row["absolute_error_c_calibrated"]), reverse=True)[:30]
    _write_csv(worst_rows, output_dir / "worst_30_predictions.csv")

    # Overlay galleries.
    rng = random.Random(args.overlay_seed)
    test_rows = split_rows["test"]
    test_samples = load_heatmap_samples(split_examples["test"], base_path, heatmap_size=args.heatmap_size, sigma_pixels=5.0, jitter=None)
    test_pairs = list(zip(test_samples, test_rows))
    best_test = sorted(test_pairs, key=lambda pair: float(pair[1]["absolute_error_c_calibrated"]))[:20]
    worst_test = sorted(test_pairs, key=lambda pair: float(pair[1]["absolute_error_c_calibrated"]), reverse=True)[:30]
    random_test = rng.sample(test_pairs, k=min(30, len(test_pairs)))
    val_samples = load_heatmap_samples(split_examples["val"], base_path, heatmap_size=args.heatmap_size, sigma_pixels=5.0, jitter=None)
    val_pairs = list(zip(val_samples, split_rows["val"]))

    gallery_specs = {
        "test_best_20": best_test,
        "test_worst_30": worst_test,
        "test_random_30": random_test,
        "val_all": val_pairs,
    }
    for gallery_name, pairs in gallery_specs.items():
        gallery_dir = debug_dir / gallery_name
        gallery_dir.mkdir(parents=True, exist_ok=True)
        for index, (sample, row) in enumerate(pairs):
            overlay_name = f"{index:03d}_{Path(str(sample.metadata['image_path'])).stem}.png"
            write_prediction_overlay(sample, row, gallery_dir / overlay_name, heatmap_size=args.heatmap_size)

    report_payload = {
        "model_path": str(model_path),
        "selected_stage": selected_stage,
        "calibration_candidate_name": calibration_candidate.name,
        "calibration_candidate_kind": calibration_candidate.kind,
        "split_metrics": split_metrics,
    }
    _write_json(report_payload, output_dir / "eval_summary.json")
    _write_report(
        model_path=model_path,
        selected_stage=selected_stage,
        candidate_name=calibration_candidate.name,
        candidate_kind=calibration_candidate.kind,
        split_metrics=split_metrics,
        worst_rows=worst_rows,
        report_path=report_path,
    )

    print(f"Prediction CSVs written to {output_dir}")
    print(f"Overlay gallery written to {debug_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
