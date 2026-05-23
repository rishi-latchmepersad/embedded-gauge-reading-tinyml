#!/usr/bin/env python3
"""Evaluate the board replay on test split with the selected guardrails."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_board_replay import (
    BoardReplayMode,
    build_board_replay_sample,
    write_board_replay_overlay,
)
from embedded_gauge_reading_tinyml.geometry_board_replay_guardrails import (
    BoardReplayGuardrailDecision,
    evaluate_board_replay_row,
    summarize_board_replay_rows,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import (
    load_clean_geometry_examples,
    load_selected_calibration_candidate,
    select_examples_from_split,
)
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import (
    GeometryGuardrailThresholds,
    guarded_temperature_from_prediction,
)
from embedded_gauge_reading_tinyml.gauge_geometry import circular_angle_error_degrees
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import decode_heatmap_geometry_prediction


SELECTED_PREPROCESSING_MODE: BoardReplayMode = "python_training_rgb_bilinear"
PREVIOUS_TEST_ACCEPTED_MAE_C = 2.517
PREVIOUS_TEST_ACCEPTANCE_RATE = 0.644
PREVIOUS_TEST_WORST_ACCEPTED_ERROR_C = 9.060


def _resolve_path(base_path: Path, maybe_relative: Path) -> Path:
    """Resolve a path against the repo root when needed."""

    return maybe_relative if maybe_relative.is_absolute() else base_path / maybe_relative


def _load_rows(csv_path: Path, *, preprocessing_mode: str) -> list[dict[str, Any]]:
    """Load the previously replayed rows for one preprocessing mode."""

    rows: list[dict[str, Any]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row["preprocessing_mode"]) != preprocessing_mode:
                continue
            rows.append(row)
    return rows


def _load_selected_thresholds(json_path: Path) -> GeometryGuardrailThresholds:
    """Load the selected threshold artifact produced by the micro sweep."""

    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    thresholds = payload["selected_thresholds"]
    return GeometryGuardrailThresholds(
        center_peak_min=float(thresholds["center_peak_min"]),
        tip_peak_min=float(thresholds["tip_peak_min"]),
        confidence_min=float(thresholds["confidence_min"]),
        max_heatmap_entropy=float(thresholds["max_heatmap_entropy"]),
        max_heatmap_spread_px=float(thresholds["max_heatmap_spread_px"]),
        center_tip_distance_ratio_min=float(thresholds["center_tip_distance_ratio_min"]),
        center_tip_distance_ratio_max=float(thresholds["center_tip_distance_ratio_max"]),
        edge_margin_px=float(thresholds["edge_margin_px"]),
        temperature_physical_margin_c=float(thresholds["temperature_physical_range_margin_c"]),
        minimum_celsius=float(thresholds["minimum_celsius"]),
        maximum_celsius=float(thresholds["maximum_celsius"]),
        clamp_temperature_to_physical_range=bool(thresholds["clamp_temperature_to_physical_range"]),
    )


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a stable CSV file for selected-threshold predictions."""

    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    skip_keys = {"pred_center_heatmap_array", "pred_tip_heatmap_array"}
    fieldnames = [key for key in rows[0].keys() if key not in skip_keys]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def _predict_in_batches(model: keras.Model, inputs: list[np.ndarray], *, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference on a list of preprocessed inputs."""

    centers: list[np.ndarray] = []
    tips: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    for start in range(0, len(inputs), batch_size):
        batch = np.stack(inputs[start : start + batch_size], axis=0).astype(np.float32)
        outputs = model.predict(batch, verbose=0)
        if isinstance(outputs, dict):
            ordered = list(outputs.values())
        else:
            ordered = list(outputs)
        if len(ordered) != 3:
            raise ValueError(f"Expected 3 outputs, got {len(ordered)}")
        centers.append(np.asarray(ordered[0], dtype=np.float32))
        tips.append(np.asarray(ordered[1], dtype=np.float32))
        confidences.append(np.asarray(ordered[2], dtype=np.float32))
    return np.concatenate(centers, axis=0), np.concatenate(tips, axis=0), np.concatenate(confidences, axis=0)


def _prediction_row(
    sample: Any,
    old_row: Mapping[str, Any],
    guarded: BoardReplayGuardrailDecision,
    center_heatmap: np.ndarray,
    tip_heatmap: np.ndarray,
) -> dict[str, Any]:
    """Flatten one selected-threshold replay decision to a CSV row."""

    true_temperature_c = float(old_row["true_temperature_c"])
    old_status = str(old_row["guardrail_status"])
    old_reasons = str(old_row["rejection_reasons"])
    if not old_reasons:
        old_reasons = "none"
    new_error = float(abs(guarded.temperature_c - true_temperature_c)) if guarded.status != "rejected" else math.nan
    old_guarded_temperature = float(old_row["guarded_temperature_c"]) if str(old_row["guarded_temperature_c"]) != "nan" else math.nan
    old_error = float(abs(old_guarded_temperature - true_temperature_c)) if math.isfinite(old_guarded_temperature) else math.nan

    row = {
        "image_path": old_row["image_path"],
        "split": old_row["split"],
        "preprocessing_mode": old_row["preprocessing_mode"],
        "source_kind": old_row["source_kind"],
        "true_temperature_c": true_temperature_c,
        "predicted_temperature_c_current_mapping": float(old_row["predicted_temperature_c_current_mapping"]),
        "predicted_temperature_c_current_mapping_argmax": float(old_row["predicted_temperature_c_current_mapping_argmax"]),
        "predicted_temperature_c_calibrated": float(old_row["predicted_temperature_c_calibrated"]),
        "predicted_temperature_c_calibrated_argmax": float(old_row["predicted_temperature_c_calibrated_argmax"]),
        "old_guarded_temperature_c": old_guarded_temperature,
        "guarded_temperature_c": float(guarded.temperature_c),
        "absolute_error_c_current_mapping": float(old_row["absolute_error_c_current_mapping"]),
        "absolute_error_c_current_mapping_argmax": float(old_row["absolute_error_c_current_mapping_argmax"]),
        "absolute_error_c_calibrated_raw": float(old_row["absolute_error_c_calibrated_raw"]),
        "absolute_error_c_calibrated_raw_argmax": float(old_row["absolute_error_c_calibrated_raw_argmax"]),
        "old_absolute_error_c_guarded": old_error,
        "absolute_error_c_guarded": new_error,
        "old_guardrail_status": old_status,
        "old_rejection_reasons": old_reasons,
        "guardrail_status": guarded.status,
        "rejection_reasons": ";".join(guarded.rejection_reasons),
        "true_angle_degrees": float(old_row["true_angle_degrees"]),
        "predicted_angle_degrees": float(old_row["predicted_angle_degrees"]),
        "predicted_angle_degrees_argmax": float(old_row["predicted_angle_degrees_argmax"]),
        "angle_mae_degrees": float(old_row["angle_mae_degrees"]),
        "angle_mae_degrees_argmax": float(old_row["angle_mae_degrees_argmax"]),
        "true_center_x_224": float(old_row["true_center_x_224"]),
        "true_center_y_224": float(old_row["true_center_y_224"]),
        "true_tip_x_224": float(old_row["true_tip_x_224"]),
        "true_tip_y_224": float(old_row["true_tip_y_224"]),
        "predicted_center_x_224": float(old_row["predicted_center_x_224"]),
        "predicted_center_y_224": float(old_row["predicted_center_y_224"]),
        "predicted_tip_x_224": float(old_row["predicted_tip_x_224"]),
        "predicted_tip_y_224": float(old_row["predicted_tip_y_224"]),
        "predicted_center_x_224_argmax": float(old_row["predicted_center_x_224_argmax"]),
        "predicted_center_y_224_argmax": float(old_row["predicted_center_y_224_argmax"]),
        "predicted_tip_x_224_argmax": float(old_row["predicted_tip_x_224_argmax"]),
        "predicted_tip_y_224_argmax": float(old_row["predicted_tip_y_224_argmax"]),
        "center_px_mae_224": float(old_row["center_px_mae_224"]),
        "tip_px_mae_224": float(old_row["tip_px_mae_224"]),
        "center_px_mae_224_argmax": float(old_row["center_px_mae_224_argmax"]),
        "tip_px_mae_224_argmax": float(old_row["tip_px_mae_224_argmax"]),
        "center_heatmap_peak_value": float(old_row["center_heatmap_peak_value"]),
        "tip_heatmap_peak_value": float(old_row["tip_heatmap_peak_value"]),
        "center_heatmap_mean_value": float(old_row["center_heatmap_mean_value"]),
        "tip_heatmap_mean_value": float(old_row["tip_heatmap_mean_value"]),
        "center_heatmap_entropy": float(old_row["center_heatmap_entropy"]),
        "tip_heatmap_entropy": float(old_row["tip_heatmap_entropy"]),
        "center_heatmap_spread_px": float(old_row["center_heatmap_spread_px"]),
        "tip_heatmap_spread_px": float(old_row["tip_heatmap_spread_px"]),
        "confidence": float(old_row["confidence"]),
        "center_normalized_in_bounds": bool(str(old_row["center_normalized_in_bounds"]) == "True"),
        "tip_normalized_in_bounds": bool(str(old_row["tip_normalized_in_bounds"]) == "True"),
        "center_edge_margin_px": float(old_row["center_edge_margin_px"]),
        "tip_edge_margin_px": float(old_row["tip_edge_margin_px"]),
        "min_edge_margin_px": float(old_row["min_edge_margin_px"]),
        "predicted_center_tip_distance_px": float(old_row["predicted_center_tip_distance_px"]),
        "true_center_tip_distance_px": float(old_row["true_center_tip_distance_px"]),
        "expected_center_tip_distance_px": float(old_row["expected_center_tip_distance_px"]),
        "center_tip_distance_ratio": float(old_row["center_tip_distance_ratio"]),
        "angle_unwrapped_from_cold_degrees": float(old_row["angle_unwrapped_from_cold_degrees"]),
        "angle_within_valid_sweep": bool(str(old_row["angle_within_valid_sweep"]) == "True"),
        "current_temperature_within_physical_range": bool(str(old_row["current_temperature_within_physical_range"]) == "True"),
        "calibrated_temperature_within_physical_range": bool(str(old_row["calibrated_temperature_within_physical_range"]) == "True"),
        "calibrated_temperature_outside_physical_range": bool(str(old_row["calibrated_temperature_outside_physical_range"]) == "True"),
        "jitter_shift_x": int(old_row["jitter_shift_x"]),
        "jitter_shift_y": int(old_row["jitter_shift_y"]),
        "jitter_scale": float(old_row["jitter_scale"]),
        "jitter_aspect": float(old_row["jitter_aspect"]),
        "crop_x1": int(old_row["crop_x1"]),
        "crop_y1": int(old_row["crop_y1"]),
        "crop_x2": int(old_row["crop_x2"]),
        "crop_y2": int(old_row["crop_y2"]),
        "crop_width": int(old_row["crop_width"]),
        "crop_height": int(old_row["crop_height"]),
        "source_width": int(old_row["source_width"]),
        "source_height": int(old_row["source_height"]),
        "source_manifest": str(old_row["source_manifest"]),
        "quality_flag": str(old_row["quality_flag"]),
        "dial_radius_source": float(old_row["dial_radius_source"]),
        "resize_method": str(old_row["resize_method"]),
        "channel_strategy": str(old_row["channel_strategy"]),
        "normalization": str(old_row["normalization"]),
        "scale": float(old_row["scale"]),
        "resized_width": int(old_row["resized_width"]),
        "resized_height": int(old_row["resized_height"]),
        "pad_x": int(old_row["pad_x"]),
        "pad_y": int(old_row["pad_y"]),
        "pad_bottom": int(old_row["pad_bottom"]),
        "pad_right": int(old_row["pad_right"]),
        "input_size": int(old_row["input_size"]),
        "heatmap_size": int(old_row["heatmap_size"]),
        "sigma_pixels": float(old_row["sigma_pixels"]),
        "selected_guardrail_status": guarded.status,
        "selected_guardrail_rejection_reasons": ";".join(guarded.rejection_reasons),
        "selected_guarded_temperature_c": float(guarded.temperature_c),
    }
    return row


def _write_report(
    *,
    report_path: Path,
    selected_thresholds: GeometryGuardrailThresholds,
    selected_summary: dict[str, Any],
) -> None:
    """Write the final selected-threshold test report."""

    lines = [
        "# Geometry Heatmap v2 Board Replay Selected Guardrails",
        "",
        "## Selected Thresholds",
        "",
        "| threshold | value |",
        "| --- | ---: |",
        f"| center_peak_min | {selected_thresholds.center_peak_min} |",
        f"| tip_peak_min | {selected_thresholds.tip_peak_min} |",
        f"| confidence_min | {selected_thresholds.confidence_min} |",
        f"| max_heatmap_entropy | {selected_thresholds.max_heatmap_entropy} |",
        f"| max_heatmap_spread_px | {selected_thresholds.max_heatmap_spread_px} |",
        f"| center_tip_distance_ratio_min | {selected_thresholds.center_tip_distance_ratio_min} |",
        f"| center_tip_distance_ratio_max | {selected_thresholds.center_tip_distance_ratio_max} |",
        f"| edge_margin_px | {selected_thresholds.edge_margin_px} |",
        f"| temperature_physical_range_margin_c | {selected_thresholds.temperature_physical_margin_c} |",
        "",
        "## Test Metrics",
        "",
        f"- accepted MAE: {selected_summary['accepted_mae_c']:.3f} C",
        f"- acceptance rate: {selected_summary['acceptance_rate']:.3f}",
        f"- worst accepted error: {selected_summary['accepted_worst_error_c']:.3f} C",
        f"- accepted >20 C failures: {int(selected_summary['accepted_gt20c_failures'])}",
        f"- under 2 C: {selected_summary['percentage_under_2c']:.1f}%",
        f"- under 5 C: {selected_summary['percentage_under_5c']:.1f}%",
        f"- under 10 C: {selected_summary['percentage_under_10c']:.1f}%",
        "",
        "## Comparison To Previous Test Result",
        "",
        f"- Previous accepted MAE: {PREVIOUS_TEST_ACCEPTED_MAE_C:.3f} C",
        f"- Previous acceptance: {PREVIOUS_TEST_ACCEPTANCE_RATE:.3f}",
        f"- Previous worst accepted error: {PREVIOUS_TEST_WORST_ACCEPTED_ERROR_C:.3f} C",
        f"- Delta accepted MAE: {selected_summary['accepted_mae_c'] - PREVIOUS_TEST_ACCEPTED_MAE_C:.3f} C",
        f"- Delta acceptance: {selected_summary['acceptance_rate'] - PREVIOUS_TEST_ACCEPTANCE_RATE:.3f}",
        f"- Delta worst accepted error: {selected_summary['accepted_worst_error_c'] - PREVIOUS_TEST_WORST_ACCEPTED_ERROR_C:.3f} C",
        "",
        "## Top Rejection Reasons",
        "",
        "| reason | count |",
        "| --- | ---: |",
    ]
    for reason, count in selected_summary["top_rejection_reasons"]:
        lines.append(f"| {reason} | {count} |")

    lines.extend(
        [
            "",
            f"- Export gate status: {'pass' if selected_summary['accepted_mae_c'] <= 4.5 and selected_summary['acceptance_rate'] >= 0.65 and selected_summary['accepted_worst_error_c'] < 20.0 and selected_summary['accepted_gt20c_failures'] == 0 else 'fail'}",
            "",
        ]
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """Evaluate the selected guardrails on the test split once."""

    parser = argparse.ArgumentParser(description="Evaluate board replay with selected guardrails")
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
        help="Calibration artifact.",
    )
    parser.add_argument(
        "--selected-thresholds-path",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2_board_replay/selected_board_guardrail_thresholds.json"),
        help="Selected guardrail thresholds from the validation sweep.",
    )
    parser.add_argument(
        "--old-predictions-csv",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2_board_replay/board_replay_predictions.csv"),
        help="First-pass board replay predictions with the old thresholds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2_board_replay"),
        help="Directory for selected-threshold artifacts.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("ml/debug/geometry_heatmap_v2_board_replay_selected_guardrails"),
        help="Directory for selected-threshold overlays.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_v2_board_replay_selected_guardrails_eval.md"),
        help="Markdown report path.",
    )
    parser.add_argument("--input-size", type=int, default=224, help="Model input size.")
    parser.add_argument("--heatmap-size", type=int, default=56, help="Heatmap size.")
    parser.add_argument("--sigma-pixels", type=float, default=5.0, help="Heatmap sigma.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    model_path = _resolve_path(base_path, args.model_path)
    manifest_path = _resolve_path(base_path, args.manifest_path)
    calibration_json_path = _resolve_path(base_path, args.calibration_json_path)
    selected_thresholds_path = _resolve_path(base_path, args.selected_thresholds_path)
    old_predictions_csv = _resolve_path(base_path, args.old_predictions_csv)
    output_dir = _resolve_path(base_path, args.output_dir)
    debug_dir = _resolve_path(base_path, args.debug_dir)
    report_path = _resolve_path(base_path, args.report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(model_path, compile=False)
    calibration_candidate, _ = load_selected_calibration_candidate(calibration_json_path)
    selected_thresholds = _load_selected_thresholds(selected_thresholds_path)
    examples = load_clean_geometry_examples(manifest_path)
    test_examples = select_examples_from_split(examples, split="test")
    old_rows = {
        str(row["image_path"]): row
        for row in _load_rows(old_predictions_csv, preprocessing_mode=SELECTED_PREPROCESSING_MODE)
        if str(row["split"]) == "test"
    }

    samples = [
        build_board_replay_sample(
            example,
            base_path,
            mode=SELECTED_PREPROCESSING_MODE,
            input_size=args.input_size,
            heatmap_size=args.heatmap_size,
            sigma_pixels=args.sigma_pixels,
        )
        for example in test_examples
    ]
    inputs = [np.asarray(sample.crop_image, dtype=np.float32) for sample in samples]
    center_batch, tip_batch, confidence_batch = _predict_in_batches(model, inputs, batch_size=args.batch_size)

    rows: list[dict[str, Any]] = []
    heatmap_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, sample in enumerate(samples):
        center_heatmap = np.asarray(center_batch[index], dtype=np.float32)
        tip_heatmap = np.asarray(tip_batch[index], dtype=np.float32)
        confidence = float(np.ravel(np.asarray(confidence_batch[index], dtype=np.float32))[0])
        guarded = guarded_temperature_from_prediction(
            sample,
            center_heatmap,
            tip_heatmap,
            confidence,
            calibration_candidate,
            thresholds=selected_thresholds,
        )
        old_row = old_rows[str(sample.metadata["image_path"])]
        row = _prediction_row(sample, old_row, guarded, center_heatmap, tip_heatmap)
        rows.append(row)
        heatmap_cache[str(sample.metadata["image_path"])] = (center_heatmap, tip_heatmap)

    summary = summarize_board_replay_rows(rows, selected_thresholds)
    _write_csv(rows, output_dir / "selected_guardrail_test_predictions.csv")
    remaining_worst = [
        row
        for row in sorted(
            [row for row in rows if str(row["guardrail_status"]) != "rejected"],
            key=lambda row: float(row["absolute_error_c_guarded"]),
            reverse=True,
        )[:30]
    ]
    _write_csv(remaining_worst, output_dir / "selected_guardrail_remaining_worst_accepted.csv")
    with open(output_dir / "selected_board_guardrail_thresholds.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_thresholds": {
                    "center_peak_min": float(selected_thresholds.center_peak_min),
                    "tip_peak_min": float(selected_thresholds.tip_peak_min),
                    "confidence_min": float(selected_thresholds.confidence_min),
                    "max_heatmap_entropy": float(selected_thresholds.max_heatmap_entropy),
                    "max_heatmap_spread_px": float(selected_thresholds.max_heatmap_spread_px),
                    "center_tip_distance_ratio_min": float(selected_thresholds.center_tip_distance_ratio_min),
                    "center_tip_distance_ratio_max": float(selected_thresholds.center_tip_distance_ratio_max),
                    "edge_margin_px": float(selected_thresholds.edge_margin_px),
                    "temperature_physical_range_margin_c": float(selected_thresholds.temperature_physical_margin_c),
                    "minimum_celsius": float(selected_thresholds.minimum_celsius),
                    "maximum_celsius": float(selected_thresholds.maximum_celsius),
                    "clamp_temperature_to_physical_range": bool(selected_thresholds.clamp_temperature_to_physical_range),
                },
                "selected_mode": SELECTED_PREPROCESSING_MODE,
                "selection_source": str(selected_thresholds_path),
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    _write_report(
        report_path=report_path,
        selected_thresholds=selected_thresholds,
        selected_summary=summary,
    )

    # Overlay groups requested by the phase.
    newly_accepted = [
        (sample, row)
        for sample, row in zip(samples, rows, strict=True)
        if str(old_rows[str(sample.metadata["image_path"])]["guardrail_status"]) == "rejected"
        and str(row["guardrail_status"]) != "rejected"
    ]
    worst_accepted = sorted(
        [(sample, row) for sample, row in zip(samples, rows, strict=True) if str(row["guardrail_status"]) != "rejected"],
        key=lambda pair: float(pair[1]["absolute_error_c_guarded"]),
        reverse=True,
    )[:30]
    over_8c = [
        (sample, row)
        for sample, row in zip(samples, rows, strict=True)
        if str(row["guardrail_status"]) != "rejected" and float(row["absolute_error_c_guarded"]) > 8.0
    ]
    rejected = [
        (sample, row)
        for sample, row in zip(samples, rows, strict=True)
        if str(row["guardrail_status"]) == "rejected"
    ][:50]

    galleries = {
        "newly_accepted": newly_accepted,
        "worst_30_accepted": worst_accepted,
        "accepted_error_gt_8c": over_8c,
        "rejected_test": rejected,
    }
    for gallery_name, pairs in galleries.items():
        gallery_dir = debug_dir / gallery_name
        gallery_dir.mkdir(parents=True, exist_ok=True)
        for index, (sample, row) in enumerate(pairs):
            overlay_name = f"{index:03d}_{Path(str(sample.metadata['image_path'])).stem}.png"
            center_heatmap, tip_heatmap = heatmap_cache[str(sample.metadata["image_path"])]
            write_board_replay_overlay(
                sample,
                row,
                center_heatmap,
                tip_heatmap,
                gallery_dir / overlay_name,
            )

    print(f"Selected guardrail test predictions written to {output_dir}")
    print(f"Selected guardrail overlays written to {debug_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
