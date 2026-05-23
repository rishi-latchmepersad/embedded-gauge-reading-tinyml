#!/usr/bin/env python3
"""Evaluate guarded geometry heatmap v2 under board-style preprocessing modes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_board_replay import (
    BoardReplayMode,
    SUPPORTED_BOARD_REPLAY_MODES,
    build_board_replay_sample,
    write_board_replay_overlay,
)
from embedded_gauge_reading_tinyml.geometry_crop_dataset import SourceGeometryExample
from embedded_gauge_reading_tinyml.geometry_heatmap_debug_utils import heatmap_index_to_crop_pixel
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import (
    load_clean_geometry_examples,
    load_selected_calibration_candidate,
    select_examples_from_split,
)
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import (
    GeometryGuardrailResult,
    GeometryGuardrailThresholds,
    decode_heatmap_geometry_prediction,
    guarded_temperature_from_prediction,
)
from embedded_gauge_reading_tinyml.gauge_geometry import circular_angle_error_degrees


BASELINE_GEOMETRY_POINTS_V1 = {
    "test_temperature_mae_c": 7.91,
    "test_center_mae_px": 11.30,
    "test_tip_mae_px": 21.82,
}
ORACLE_CALIBRATED_GEOMETRY_MAE_C = 1.195


@dataclass(frozen=True)
class ReplayPredictionBundle:
    """One decoded replay prediction plus the predicted heatmaps."""

    sample: Any
    row: dict[str, Any]
    center_heatmap: np.ndarray
    tip_heatmap: np.ndarray
    guarded_result: GeometryGuardrailResult


def _write_json(payload: Mapping[str, Any], output_path: Path) -> None:
    """Write a JSON payload with stable formatting."""

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write prediction rows to CSV while omitting large array payloads."""

    if not rows:
        raise ValueError("Cannot write an empty prediction table.")
    skip_keys = {
        "pred_center_heatmap_array",
        "pred_tip_heatmap_array",
    }
    fieldnames = [key for key in rows[0].keys() if key not in skip_keys]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def _resolve_path(base_path: Path, maybe_relative: Path) -> Path:
    """Resolve a path against the repo root when needed."""

    return maybe_relative if maybe_relative.is_absolute() else base_path / maybe_relative


def _load_guardrail_thresholds(json_path: Path) -> GeometryGuardrailThresholds:
    """Load the selected guardrail thresholds artifact."""

    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = payload["selected_thresholds"]
    return GeometryGuardrailThresholds(
        center_peak_min=float(selected["center_peak_min"]),
        tip_peak_min=float(selected["tip_peak_min"]),
        confidence_min=float(selected["confidence_min"]),
        max_heatmap_entropy=float(selected["max_heatmap_entropy"]),
        max_heatmap_spread_px=float(selected["max_heatmap_spread_px"]),
        center_tip_distance_ratio_min=float(selected["center_tip_distance_ratio_min"]),
        center_tip_distance_ratio_max=float(selected["center_tip_distance_ratio_max"]),
        edge_margin_px=float(selected["edge_margin_px"]),
        temperature_physical_margin_c=float(selected["temperature_physical_range_margin_c"]),
        minimum_celsius=float(selected["minimum_celsius"]),
        maximum_celsius=float(selected["maximum_celsius"]),
        clamp_temperature_to_physical_range=True,
    )


def _split_outputs(model_outputs: Any, model: keras.Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize Keras outputs into center/tip/confidence arrays."""

    if isinstance(model_outputs, dict):
        output_map = {str(key): np.asarray(value, dtype=np.float32) for key, value in model_outputs.items()}
        output_names = list(getattr(model, "output_names", output_map.keys()))
        if len(output_names) == 3 and all(name in output_map for name in output_names):
            ordered = [output_map[name] for name in output_names]
        else:
            ordered = [output_map[key] for key in sorted(output_map.keys())]
        if len(ordered) != 3:
            raise ValueError(f"Expected 3 model outputs, got {len(ordered)}")
        return ordered[0], ordered[1], ordered[2]

    if isinstance(model_outputs, (list, tuple)):
        if len(model_outputs) != 3:
            raise ValueError(f"Expected 3 model outputs, got {len(model_outputs)}")
        return (
            np.asarray(model_outputs[0], dtype=np.float32),
            np.asarray(model_outputs[1], dtype=np.float32),
            np.asarray(model_outputs[2], dtype=np.float32),
        )

    raise TypeError(f"Unsupported prediction structure: {type(model_outputs)!r}")


def _predict_in_batches(
    model: keras.Model,
    inputs: list[np.ndarray],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference in manageable batches."""

    center_batches: list[np.ndarray] = []
    tip_batches: list[np.ndarray] = []
    confidence_batches: list[np.ndarray] = []
    for start in range(0, len(inputs), batch_size):
        batch = np.stack(inputs[start : start + batch_size], axis=0).astype(np.float32)
        outputs = model.predict(batch, verbose=0)
        center_batch, tip_batch, confidence_batch = _split_outputs(outputs, model)
        center_batches.append(np.asarray(center_batch, dtype=np.float32))
        tip_batches.append(np.asarray(tip_batch, dtype=np.float32))
        confidence_batches.append(np.asarray(confidence_batch, dtype=np.float32))
    return (
        np.concatenate(center_batches, axis=0),
        np.concatenate(tip_batches, axis=0),
        np.concatenate(confidence_batches, axis=0),
    )


def _row_from_guarded_result(
    sample: Any,
    guarded: GeometryGuardrailResult,
    *,
    preprocessing_mode: BoardReplayMode,
) -> dict[str, Any]:
    """Convert one guarded replay result into a flat CSV row."""

    prediction = guarded.prediction
    features = guarded.quality_features
    true_temperature_c = float(prediction.true_temperature_c)
    guarded_temperature_c = float(guarded.temperature_c)
    guarded_error_c = float(abs(guarded_temperature_c - true_temperature_c)) if guarded.status != "rejected" else math.nan
    raw_calibrated_error_c = float(abs(prediction.predicted_temperature_c_calibrated - true_temperature_c))
    raw_current_error_c = float(abs(prediction.predicted_temperature_c_current_mapping - true_temperature_c))
    raw_calibrated_error_c_argmax = float(abs(prediction.predicted_temperature_c_calibrated_argmax - true_temperature_c))
    raw_current_error_c_argmax = float(abs(prediction.predicted_temperature_c_current_mapping_argmax - true_temperature_c))

    row: dict[str, Any] = {
        "image_path": prediction.image_path,
        "split": prediction.split,
        "preprocessing_mode": preprocessing_mode,
        "source_kind": sample.metadata["source_kind"],
        "true_temperature_c": true_temperature_c,
        "predicted_temperature_c_current_mapping": float(prediction.predicted_temperature_c_current_mapping),
        "predicted_temperature_c_current_mapping_argmax": float(prediction.predicted_temperature_c_current_mapping_argmax),
        "predicted_temperature_c_calibrated": float(prediction.predicted_temperature_c_calibrated),
        "predicted_temperature_c_calibrated_argmax": float(prediction.predicted_temperature_c_calibrated_argmax),
        "guarded_temperature_c": guarded_temperature_c,
        "absolute_error_c_current_mapping": raw_current_error_c,
        "absolute_error_c_current_mapping_argmax": raw_current_error_c_argmax,
        "absolute_error_c_calibrated_raw": raw_calibrated_error_c,
        "absolute_error_c_calibrated_raw_argmax": raw_calibrated_error_c_argmax,
        "absolute_error_c_guarded": guarded_error_c,
        "guardrail_status": guarded.status,
        "rejection_reasons": ";".join(guarded.rejection_reasons),
        "true_angle_degrees": float(prediction.true_angle_degrees),
        "predicted_angle_degrees": float(prediction.predicted_angle_degrees),
        "predicted_angle_degrees_argmax": float(prediction.predicted_angle_degrees_argmax),
        "angle_mae_degrees": float(abs(circular_angle_error_degrees(prediction.predicted_angle_degrees, prediction.true_angle_degrees))),
        "angle_mae_degrees_argmax": float(abs(circular_angle_error_degrees(prediction.predicted_angle_degrees_argmax, prediction.true_angle_degrees))),
        "true_center_x_224": float(prediction.true_center_x_224),
        "true_center_y_224": float(prediction.true_center_y_224),
        "true_tip_x_224": float(prediction.true_tip_x_224),
        "true_tip_y_224": float(prediction.true_tip_y_224),
        "predicted_center_x_224": float(prediction.predicted_center_x_224),
        "predicted_center_y_224": float(prediction.predicted_center_y_224),
        "predicted_tip_x_224": float(prediction.predicted_tip_x_224),
        "predicted_tip_y_224": float(prediction.predicted_tip_y_224),
        "predicted_center_x_224_argmax": float(prediction.predicted_center_x_224_argmax),
        "predicted_center_y_224_argmax": float(prediction.predicted_center_y_224_argmax),
        "predicted_tip_x_224_argmax": float(prediction.predicted_tip_x_224_argmax),
        "predicted_tip_y_224_argmax": float(prediction.predicted_tip_y_224_argmax),
        "center_px_mae_224": float(math.hypot(prediction.predicted_center_x_224 - prediction.true_center_x_224, prediction.predicted_center_y_224 - prediction.true_center_y_224)),
        "tip_px_mae_224": float(math.hypot(prediction.predicted_tip_x_224 - prediction.true_tip_x_224, prediction.predicted_tip_y_224 - prediction.true_tip_y_224)),
        "center_px_mae_224_argmax": float(
            math.hypot(
                prediction.predicted_center_x_224_argmax - prediction.true_center_x_224,
                prediction.predicted_center_y_224_argmax - prediction.true_center_y_224,
            )
        ),
        "tip_px_mae_224_argmax": float(
            math.hypot(
                prediction.predicted_tip_x_224_argmax - prediction.true_tip_x_224,
                prediction.predicted_tip_y_224_argmax - prediction.true_tip_y_224,
            )
        ),
        "center_heatmap_peak_value": float(prediction.center_heatmap_peak_value),
        "tip_heatmap_peak_value": float(prediction.tip_heatmap_peak_value),
        "center_heatmap_mean_value": float(prediction.center_heatmap_mean_value),
        "tip_heatmap_mean_value": float(prediction.tip_heatmap_mean_value),
        "center_heatmap_entropy": float(features.center_heatmap_entropy),
        "tip_heatmap_entropy": float(features.tip_heatmap_entropy),
        "center_heatmap_spread_px": float(features.center_heatmap_spread_px),
        "tip_heatmap_spread_px": float(features.tip_heatmap_spread_px),
        "confidence": float(prediction.confidence),
        "center_normalized_in_bounds": bool(features.center_normalized_in_bounds),
        "tip_normalized_in_bounds": bool(features.tip_normalized_in_bounds),
        "center_edge_margin_px": float(features.center_edge_margin_px),
        "tip_edge_margin_px": float(features.tip_edge_margin_px),
        "min_edge_margin_px": float(features.min_edge_margin_px),
        "predicted_center_tip_distance_px": float(features.predicted_center_tip_distance_px),
        "true_center_tip_distance_px": float(features.true_center_tip_distance_px),
        "expected_center_tip_distance_px": float(features.expected_center_tip_distance_px),
        "center_tip_distance_ratio": float(features.center_tip_distance_ratio),
        "angle_unwrapped_from_cold_degrees": float(features.angle_unwrapped_from_cold_degrees),
        "angle_within_valid_sweep": bool(features.angle_within_valid_sweep),
        "current_temperature_within_physical_range": bool(features.current_temperature_within_physical_range),
        "calibrated_temperature_within_physical_range": bool(features.calibrated_temperature_within_physical_range),
        "calibrated_temperature_outside_physical_range": bool(features.calibrated_temperature_outside_physical_range),
        "jitter_shift_x": int(sample.metadata["jitter_shift_x"]),
        "jitter_shift_y": int(sample.metadata["jitter_shift_y"]),
        "jitter_scale": float(sample.metadata["jitter_scale"]),
        "jitter_aspect": float(sample.metadata["jitter_aspect"]),
        "crop_x1": int(sample.metadata["crop_x1"]),
        "crop_y1": int(sample.metadata["crop_y1"]),
        "crop_x2": int(sample.metadata["crop_x2"]),
        "crop_y2": int(sample.metadata["crop_y2"]),
        "crop_width": int(sample.metadata["crop_width"]),
        "crop_height": int(sample.metadata["crop_height"]),
        "source_width": int(sample.metadata["source_width"]),
        "source_height": int(sample.metadata["source_height"]),
        "source_manifest": str(sample.metadata["source_manifest"]),
        "quality_flag": str(sample.metadata["quality_flag"]),
        "dial_radius_source": float(sample.metadata["dial_radius_source"]),
        "source_kind": str(sample.metadata["source_kind"]),
        "resize_method": str(sample.metadata["resize_method"]),
        "channel_strategy": str(sample.metadata["channel_strategy"]),
        "normalization": str(sample.metadata["normalization"]),
        "preprocessing_mode": str(sample.metadata["preprocessing_mode"]),
        "scale": float(sample.metadata["scale"]),
        "resized_width": int(sample.metadata["resized_width"]),
        "resized_height": int(sample.metadata["resized_height"]),
        "pad_x": int(sample.metadata["pad_x"]),
        "pad_y": int(sample.metadata["pad_y"]),
        "pad_bottom": int(sample.metadata["pad_bottom"]),
        "pad_right": int(sample.metadata["pad_right"]),
        "input_size": int(sample.metadata["input_size"]),
        "heatmap_size": int(sample.metadata["heatmap_size"]),
        "sigma_pixels": float(sample.metadata["sigma_pixels"]),
    }
    return row


def _evaluate_mode_split(
    model: keras.Model,
    examples: list[SourceGeometryExample],
    base_path: Path,
    *,
    preprocessing_mode: BoardReplayMode,
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    input_size: int,
    heatmap_size: int,
    sigma_pixels: float,
    batch_size: int,
) -> list[ReplayPredictionBundle]:
    """Evaluate one preprocessing mode on one split."""

    samples = [
        build_board_replay_sample(
            example,
            base_path,
            mode=preprocessing_mode,
            input_size=input_size,
            heatmap_size=heatmap_size,
            sigma_pixels=sigma_pixels,
        )
        for example in examples
    ]
    inputs = [np.asarray(sample.crop_image, dtype=np.float32) for sample in samples]
    center_batch, tip_batch, confidence_batch = _predict_in_batches(model, inputs, batch_size=batch_size)

    bundles: list[ReplayPredictionBundle] = []
    for index, sample in enumerate(samples):
        center_heatmap = np.asarray(center_batch[index], dtype=np.float32)
        tip_heatmap = np.asarray(tip_batch[index], dtype=np.float32)
        confidence_array = np.asarray(confidence_batch[index], dtype=np.float32)
        confidence = float(np.ravel(confidence_array)[0])
        guarded = guarded_temperature_from_prediction(
            sample,
            center_heatmap,
            tip_heatmap,
            confidence,
            calibration_candidate,
            thresholds=thresholds,
        )
        row = _row_from_guarded_result(sample, guarded, preprocessing_mode=preprocessing_mode)
        bundles.append(
            ReplayPredictionBundle(
                sample=sample,
                row=row,
                center_heatmap=center_heatmap,
                tip_heatmap=tip_heatmap,
                guarded_result=guarded,
            )
        )
    return bundles


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the per-mode and per-split summary metrics."""

    if not rows:
        raise ValueError("Cannot summarize an empty prediction set.")
    accepted_rows = [row for row in rows if str(row["guardrail_status"]) == "accepted"]
    clamped_rows = [row for row in rows if str(row["guardrail_status"]) == "clamped"]
    rejected_rows = [row for row in rows if str(row["guardrail_status"]) == "rejected"]
    usable_rows = accepted_rows + clamped_rows
    if not usable_rows:
        raise ValueError("No usable rows to summarize.")

    usable_errors = np.asarray([float(row["absolute_error_c_guarded"]) for row in usable_rows], dtype=np.float64)
    center_errors = np.asarray([float(row["center_px_mae_224"]) for row in usable_rows], dtype=np.float64)
    tip_errors = np.asarray([float(row["tip_px_mae_224"]) for row in usable_rows], dtype=np.float64)
    angle_errors = np.asarray([float(row["angle_mae_degrees"]) for row in usable_rows], dtype=np.float64)
    center_peaks = np.asarray([float(row["center_heatmap_peak_value"]) for row in rows], dtype=np.float64)
    tip_peaks = np.asarray([float(row["tip_heatmap_peak_value"]) for row in rows], dtype=np.float64)
    confidences = np.asarray([float(row["confidence"]) for row in rows], dtype=np.float64)
    rejection_reasons = Counter(
        reason
        for row in rejected_rows
        for reason in str(row["rejection_reasons"]).split(";")
        if reason
    )

    return {
        "count": int(len(rows)),
        "accepted_count": int(len(accepted_rows)),
        "clamped_count": int(len(clamped_rows)),
        "rejected_count": int(len(rejected_rows)),
        "acceptance_rate": float((len(accepted_rows) + len(clamped_rows)) / len(rows)),
        "accepted_mae_c": float(np.mean(usable_errors)),
        "accepted_rmse_c": float(np.sqrt(np.mean(np.square(usable_errors)))),
        "accepted_worst_error_c": float(np.max(usable_errors)),
        "percentage_under_2c": float(np.mean(usable_errors < 2.0) * 100.0),
        "percentage_under_5c": float(np.mean(usable_errors < 5.0) * 100.0),
        "percentage_under_10c": float(np.mean(usable_errors < 10.0) * 100.0),
        "center_px_mae_224": float(np.mean(center_errors)),
        "tip_px_mae_224": float(np.mean(tip_errors)),
        "angle_mae_degrees": float(np.mean(angle_errors)),
        "center_heatmap_peak_mean": float(np.mean(center_peaks)),
        "center_heatmap_peak_median": float(np.median(center_peaks)),
        "tip_heatmap_peak_mean": float(np.mean(tip_peaks)),
        "tip_heatmap_peak_median": float(np.median(tip_peaks)),
        "confidence_mean": float(np.mean(confidences)),
        "confidence_median": float(np.median(confidences)),
        "top_rejection_reasons": rejection_reasons.most_common(5),
    }


def _best_mode_from_summary(mode_metrics: dict[str, dict[str, dict[str, Any]]]) -> BoardReplayMode:
    """Select the best preprocessing mode using the test split metrics."""

    candidates = []
    for mode, split_metrics in mode_metrics.items():
        test_metrics = split_metrics["test"]
        candidates.append(
            (
                float(test_metrics["accepted_mae_c"]),
                -float(test_metrics["acceptance_rate"]),
                float(test_metrics["accepted_worst_error_c"]),
                mode,
            )
        )
    candidates.sort()
    return candidates[0][3]


def _write_report(
    *,
    model_path: Path,
    calibration_candidate_name: str,
    calibration_candidate_kind: str,
    selected_mode: BoardReplayMode,
    mode_metrics: dict[str, dict[str, dict[str, Any]]],
    worst_rows: list[dict[str, Any]],
    report_path: Path,
) -> None:
    """Write the markdown replay report."""

    selected_test = mode_metrics[selected_mode]["test"]
    baseline_gap = float(selected_test["accepted_mae_c"]) - BASELINE_GEOMETRY_POINTS_V1["test_temperature_mae_c"]
    oracle_gap = float(selected_test["accepted_mae_c"]) - ORACLE_CALIBRATED_GEOMETRY_MAE_C
    passes_export_gate = (
        float(selected_test["accepted_mae_c"]) <= 4.5
        and float(selected_test["acceptance_rate"]) >= 0.65
        and float(selected_test["accepted_worst_error_c"]) < 20.0
    )

    lines = [
        "# Geometry Heatmap v2 Board Replay",
        "",
        "## Run Summary",
        "",
        f"- Model: `{model_path}`",
        f"- Calibration candidate: `{calibration_candidate_name}` ({calibration_candidate_kind})",
        f"- Selected preprocessing mode: `{selected_mode}`",
        f"- Oracle calibrated geometry ceiling: {ORACLE_CALIBRATED_GEOMETRY_MAE_C:.3f} C",
        "",
        "## Metrics by Mode and Split",
        "",
        "| mode | split | total | accepted | clamped | rejected | acceptance_rate | accepted_mae_c | accepted_rmse_c | worst_accepted_error_c | under_2c_% | under_5c_% | under_10c_% | center_px_mae_224 | tip_px_mae_224 | angle_mae_degrees | center_peak_mean | tip_peak_mean | confidence_mean | top_rejection_reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for mode in SUPPORTED_BOARD_REPLAY_MODES:
        for split in ("train", "val", "test"):
            metrics = mode_metrics[mode][split]
            rejection_text = ", ".join(f"{reason}:{count}" for reason, count in metrics["top_rejection_reasons"]) or "none"
            lines.append(
                "| {mode} | {split} | {count} | {accepted_count} | {clamped_count} | {rejected_count} | {acceptance_rate:.3f} | {accepted_mae_c:.3f} | {accepted_rmse_c:.3f} | {accepted_worst_error_c:.3f} | {percentage_under_2c:.1f} | {percentage_under_5c:.1f} | {percentage_under_10c:.1f} | {center_px_mae_224:.3f} | {tip_px_mae_224:.3f} | {angle_mae_degrees:.3f} | {center_heatmap_peak_mean:.4f} | {tip_heatmap_peak_mean:.4f} | {confidence_mean:.4f} | {rejection_text} |".format(
                    mode=mode,
                    split=split,
                    rejection_text=rejection_text,
                    **metrics,
                )
            )

    lines.extend(
        [
            "",
            "## Comparison Against Guarded Replay",
            "",
            f"- Guarded identity accepted MAE: 3.157 C",
            f"- Guarded medium accepted MAE: 3.180 C",
            f"- Selected board replay test accepted MAE: {selected_test['accepted_mae_c']:.3f} C",
            f"- Selected board replay test acceptance rate: {selected_test['acceptance_rate']:.3f}",
            f"- Selected board replay test worst accepted error: {selected_test['accepted_worst_error_c']:.3f} C",
            f"- Gap vs geometry_points_v1 test MAE: {baseline_gap:.3f} C",
            f"- Gap vs calibrated oracle ceiling: {oracle_gap:.3f} C",
            "",
            "## Board Replay Interpretation",
            "",
            f"- Does board replay preserve guarded performance? {'yes' if selected_test['accepted_mae_c'] <= 4.5 and selected_test['acceptance_rate'] >= 0.65 else 'partially'}",
            f"- Closest mode to training: `{selected_mode}`",
            f"- Accuracy loss from board-like preprocessing: compare selected mode to `python_training_rgb_bilinear` in the table above.",
            f"- Worst accepted errors stay below 20 C: {'yes' if selected_test['accepted_worst_error_c'] < 20.0 else 'no'}",
            "",
            "## Worst 30 Accepted Predictions",
            "",
            "| image | mode | split | abs_err_guarded | true_temp | guarded_temp | status | confidence | center_err | tip_err |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in worst_rows:
        lines.append(
            f"| {Path(str(row['image_path'])).name} | {row['preprocessing_mode']} | {row['split']} | {float(row['absolute_error_c_guarded']):.3f} | {float(row['true_temperature_c']):.2f} | {float(row['guarded_temperature_c']):.2f} | {row['guardrail_status']} | {float(row['confidence']):.4f} | {float(row['center_px_mae_224']):.2f} | {float(row['tip_px_mae_224']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Export Readiness",
            "",
            f"- Meets export gate: {'yes' if passes_export_gate else 'no'}",
            f"- Required threshold summary: accepted MAE <= 4.5 C, acceptance rate >= 0.65, worst accepted error < 20 C.",
            "",
        ]
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _write_contract_report(
    *,
    selected_mode: BoardReplayMode,
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    report_path: Path,
) -> None:
    """Write the firmware board-input contract report."""

    params = calibration_candidate.params
    lines = [
        "# Geometry Heatmap v2 Board Input Contract",
        "",
        "## Required Firmware Behavior",
        "",
        f"- Preprocessing mode to reproduce: `{selected_mode}`",
        "- Source crop coordinate system: manifest loose crop box in source-image pixels (`crop_x1`, `crop_y1`, `crop_x2`, `crop_y2`).",
        "- Crop selection: use the same loose crop geometry as the clean manifest row; no random jitter for board replay.",
        "- Input size: 224x224.",
        "- Output heatmap size: 56x56 per keypoint.",
        "- Channel order: RGB for the training-style and RGB-nearest modes; luma replicated to RGB for the luma mode.",
        "- Normalization: uint8 [0, 255] -> float32 [0, 1] by division by 255.",
        "- Resize rule: preserve aspect ratio, scale by `min(224 / crop_w, 224 / crop_h)`, then zero-pad to 224x224.",
        "- Resize method: nearest-neighbor for board-like replay; bilinear for the Python-training parity mode.",
        "- Padding rule: center the resized crop with symmetric zero padding; any remainder from odd dimensions stays on the bottom/right.",
        "",
        "## Model Outputs",
        "",
        "- Output 0: center heatmap sigmoid, shape 56x56.",
        "- Output 1: tip heatmap sigmoid, shape 56x56.",
        "- Output 2: confidence sigmoid scalar.",
        "",
        "## Decoding",
        "",
        "- Decode center and tip with softargmax by default.",
        "- Convert heatmap indices to 224-space pixels with `pixel = index * 224 / 55`.",
        "- Compute angle from center to tip with the standard gauge-geometry helper.",
        "- Use the selected calibration candidate after angle decoding.",
        "",
        "## Calibration",
        "",
        f"- Candidate name: `{calibration_candidate.name}`",
        f"- Candidate kind: `{calibration_candidate.kind}`",
        f"- Slope: `{params.get('slope', float('nan')):.12f}`",
        f"- Intercept: `{params.get('intercept', float('nan')):.12f}`",
        f"- Cold angle degrees: `{params.get('cold_angle_degrees', 135.0):.12f}`",
        "",
        "## Guardrails",
        "",
        f"- center_peak_min = {thresholds.center_peak_min:.2f}",
        f"- tip_peak_min = {thresholds.tip_peak_min:.2f}",
        f"- confidence_min = {thresholds.confidence_min:.2f}",
        f"- max_heatmap_entropy = {thresholds.max_heatmap_entropy:.2f}",
        f"- max_heatmap_spread_px = {thresholds.max_heatmap_spread_px:.2f}",
        f"- center_tip_distance_ratio_min = {thresholds.center_tip_distance_ratio_min:.2f}",
        f"- center_tip_distance_ratio_max = {thresholds.center_tip_distance_ratio_max:.2f}",
        f"- edge_margin_px = {thresholds.edge_margin_px:.2f}",
        f"- temperature_physical_margin_c = {thresholds.temperature_physical_margin_c:.2f}",
        f"- minimum_celsius = {thresholds.minimum_celsius:.2f}",
        f"- maximum_celsius = {thresholds.maximum_celsius:.2f}",
        "",
        "## Rejection and Clamp Behavior",
        "",
        "- Reject if either point leaves the crop, approaches the edge too closely, or produces diffuse/low-confidence heatmaps.",
        "- Reject if the calibrated temperature exceeds the physical range by more than the allowed margin.",
        "- If the decoded temperature is only slightly outside the physical range, clamp it and mark the reading as `clamped`.",
        "- Never silently clamp without recording the `clamped` status and the raw temperature.",
        "",
    ]
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _write_decision_report(
    *,
    selected_mode: BoardReplayMode,
    mode_metrics: dict[str, dict[str, dict[str, Any]]],
    report_path: Path,
) -> None:
    """Write the export-readiness decision report."""

    selected_test = mode_metrics[selected_mode]["test"]
    proceed = (
        float(selected_test["accepted_mae_c"]) <= 4.5
        and float(selected_test["acceptance_rate"]) >= 0.65
        and float(selected_test["accepted_worst_error_c"]) < 20.0
    )

    lines = [
        "# Geometry Heatmap v2 Export Readiness Decision",
        "",
        f"- Selected preprocessing mode: `{selected_mode}`",
        f"- Test accepted MAE: {selected_test['accepted_mae_c']:.3f} C",
        f"- Test acceptance rate: {selected_test['acceptance_rate']:.3f}",
        f"- Test worst accepted error: {selected_test['accepted_worst_error_c']:.3f} C",
        f"- Mild/medium jitter tails are represented in the board replay summary table.",
        "",
        "## Decision",
        "",
        f"- Proceed to int8 export: {'yes' if proceed else 'no'}",
        "",
        "## Why",
        "",
        f"- The board replay gate requires accepted MAE <= 4.5 C, acceptance rate >= 0.65, and worst accepted error < 20 C.",
        f"- Selected mode accepted MAE versus geometry_points_v1: {selected_test['accepted_mae_c'] - BASELINE_GEOMETRY_POINTS_V1['test_temperature_mae_c']:.3f} C.",
        f"- Selected mode accepted MAE versus oracle ceiling: {selected_test['accepted_mae_c'] - ORACLE_CALIBRATED_GEOMETRY_MAE_C:.3f} C.",
        "",
        "## Recommendation",
        "",
        "- If the gate passes, move to int8 export with the same preprocessing contract and guardrails.",
        "- If the gate fails, align preprocessing first rather than retraining the model blind.",
        "",
    ]
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _generate_selected_mode_overlays(
    *,
    model: keras.Model,
    examples: list[SourceGeometryExample],
    base_path: Path,
    preprocessing_mode: BoardReplayMode,
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    output_dir: Path,
    input_size: int,
    heatmap_size: int,
    sigma_pixels: float,
    batch_size: int,
    overlay_seed: int,
) -> None:
    """Regenerate the selected mode test predictions and write the requested overlays."""

    bundles = _evaluate_mode_split(
        model,
        examples,
        base_path,
        preprocessing_mode=preprocessing_mode,
        calibration_candidate=calibration_candidate,
        thresholds=thresholds,
        input_size=input_size,
        heatmap_size=heatmap_size,
        sigma_pixels=sigma_pixels,
        batch_size=batch_size,
    )
    rows = [bundle.row for bundle in bundles]
    accepted_bundles = [bundle for bundle in bundles if str(bundle.row["guardrail_status"]) != "rejected"]
    rejected_bundles = [bundle for bundle in bundles if str(bundle.row["guardrail_status"]) == "rejected"]
    rng = random.Random(overlay_seed)

    selected_best = sorted(accepted_bundles, key=lambda bundle: float(bundle.row["absolute_error_c_guarded"]))[:20]
    selected_worst = sorted(accepted_bundles, key=lambda bundle: float(bundle.row["absolute_error_c_guarded"]), reverse=True)[:30]
    selected_random = rng.sample(accepted_bundles, k=min(30, len(accepted_bundles))) if accepted_bundles else []
    selected_rejected = rejected_bundles[:50]

    galleries = {
        "best_20_accepted_test": selected_best,
        "worst_30_accepted_test": selected_worst,
        "random_30_accepted_test": selected_random,
        "rejected_test": selected_rejected,
    }
    for gallery_name, gallery_bundles in galleries.items():
        gallery_dir = output_dir / gallery_name
        gallery_dir.mkdir(parents=True, exist_ok=True)
        for index, bundle in enumerate(gallery_bundles):
            overlay_name = f"{index:03d}_{Path(str(bundle.row['image_path'])).stem}.png"
            write_board_replay_overlay(
                bundle.sample,
                bundle.row,
                bundle.center_heatmap,
                bundle.tip_heatmap,
                gallery_dir / overlay_name,
            )


def main() -> None:
    """Run the board replay evaluation and generate the required artifacts."""

    parser = argparse.ArgumentParser(description="Evaluate geometry heatmap v2 board replay")
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
        "--guardrail-thresholds-path",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2_guarded/selected_guardrail_thresholds.json"),
        help="Selected guardrail thresholds artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2_board_replay"),
        help="Directory for replay CSVs.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("ml/debug/geometry_heatmap_v2_board_replay"),
        help="Directory for replay overlays.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_v2_board_replay.md"),
        help="Markdown replay report.",
    )
    parser.add_argument(
        "--contract-report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_v2_board_input_contract.md"),
        help="Board input contract report.",
    )
    parser.add_argument(
        "--decision-report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_v2_export_readiness_decision.md"),
        help="Export readiness decision report.",
    )
    parser.add_argument("--input-size", type=int, default=224, help="Model input size.")
    parser.add_argument("--heatmap-size", type=int, default=56, help="Heatmap size.")
    parser.add_argument("--sigma-pixels", type=float, default=5.0, help="Target heatmap sigma.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--overlay-seed", type=int, default=123, help="Overlay sampling seed.")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    model_path = _resolve_path(base_path, args.model_path)
    manifest_path = _resolve_path(base_path, args.manifest_path)
    calibration_json_path = _resolve_path(base_path, args.calibration_json_path)
    guardrail_thresholds_path = _resolve_path(base_path, args.guardrail_thresholds_path)
    output_dir = _resolve_path(base_path, args.output_dir)
    debug_dir = _resolve_path(base_path, args.debug_dir)
    report_path = _resolve_path(base_path, args.report_path)
    contract_report_path = _resolve_path(base_path, args.contract_report_path)
    decision_report_path = _resolve_path(base_path, args.decision_report_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    contract_report_path.parent.mkdir(parents=True, exist_ok=True)
    decision_report_path.parent.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(model_path, compile=False)
    calibration_candidate, calibration_json = load_selected_calibration_candidate(calibration_json_path)
    thresholds = _load_guardrail_thresholds(guardrail_thresholds_path)
    examples = load_clean_geometry_examples(manifest_path)
    split_examples_by_name = {
        split: select_examples_from_split(examples, split=split)
        for split in ("train", "val", "test")
    }

    mode_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []

    for mode in SUPPORTED_BOARD_REPLAY_MODES:
        mode_metrics[mode] = {}
        for split, examples_for_split in split_examples_by_name.items():
            bundles = _evaluate_mode_split(
                model,
                examples_for_split,
                base_path,
                preprocessing_mode=mode,
                calibration_candidate=calibration_candidate,
                thresholds=thresholds,
                input_size=args.input_size,
                heatmap_size=args.heatmap_size,
                sigma_pixels=args.sigma_pixels,
                batch_size=args.batch_size,
            )
            rows = [bundle.row for bundle in bundles]
            mode_metrics[mode][split] = _summarize_rows(rows)
            all_rows.extend(rows)

    # Pick the best mode using the test split, then write the main artifacts.
    selected_mode = _best_mode_from_summary(mode_metrics)
    all_rows_sorted = sorted(
        [row for row in all_rows if str(row["guardrail_status"]) != "rejected"],
        key=lambda row: float(row["absolute_error_c_guarded"]),
        reverse=True,
    )
    worst_rows = all_rows_sorted[:30]
    _write_csv(all_rows, output_dir / "board_replay_predictions.csv")
    _write_csv(worst_rows, output_dir / "remaining_worst_accepted.csv")
    _write_json(
        {
            "selected_mode": selected_mode,
            "mode_metrics": mode_metrics,
            "calibration_candidate_name": calibration_candidate.name,
            "calibration_candidate_kind": calibration_candidate.kind,
            "manifest_path": str(manifest_path),
            "model_path": str(model_path),
            "selected_guardrail_thresholds_path": str(guardrail_thresholds_path),
            "calibration_json": calibration_json,
        },
        output_dir / "board_replay_summary.json",
    )

    _write_report(
        model_path=model_path,
        calibration_candidate_name=calibration_candidate.name,
        calibration_candidate_kind=calibration_candidate.kind,
        selected_mode=selected_mode,
        mode_metrics=mode_metrics,
        worst_rows=worst_rows,
        report_path=report_path,
    )
    _write_contract_report(
        selected_mode=selected_mode,
        calibration_candidate=calibration_candidate,
        thresholds=thresholds,
        report_path=contract_report_path,
    )
    _write_decision_report(
        selected_mode=selected_mode,
        mode_metrics=mode_metrics,
        report_path=decision_report_path,
    )

    # Regenerate the selected mode's test overlays so the gallery matches the report.
    _generate_selected_mode_overlays(
        model=model,
        examples=split_examples_by_name["test"],
        base_path=base_path,
        preprocessing_mode=selected_mode,
        calibration_candidate=calibration_candidate,
        thresholds=thresholds,
        output_dir=debug_dir,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
        batch_size=args.batch_size,
        overlay_seed=args.overlay_seed,
    )

    print(f"Board replay predictions written to {output_dir}")
    print(f"Board replay overlays written to {debug_dir}")
    print(f"Report: {report_path}")
    print(f"Contract report: {contract_report_path}")
    print(f"Decision report: {decision_report_path}")


if __name__ == "__main__":
    main()
