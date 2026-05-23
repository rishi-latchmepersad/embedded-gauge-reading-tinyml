#!/usr/bin/env python3
"""Replay geometry_heatmap_v2 through Keras and TFLite and compare drift."""

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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import circular_angle_error_degrees
from embedded_gauge_reading_tinyml.geometry_board_replay import build_board_replay_sample
from embedded_gauge_reading_tinyml.geometry_board_replay_guardrails import (
    BoardReplayGuardrailDecision,
    evaluate_board_replay_row,
    summarize_board_replay_rows,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_debug_utils import load_clean_geometry_examples
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import (
    LoadedTFLiteModel,
    iterate_batched,
    load_geometry_heatmap_keras_model,
    load_tflite_model,
    run_tflite_model,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import (
    HeatmapSample,
    load_selected_calibration_candidate,
    select_examples_from_split,
)
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import (
    GeometryDecodedPrediction,
    GeometryGuardrailResult,
    GeometryGuardrailThresholds,
    apply_geometry_guardrails,
    decode_heatmap_geometry_prediction,
)


SELECTED_PREPROCESSING_MODE = "python_training_rgb_bilinear"
MODEL_TYPES = ("keras_fp32", "tflite_fp32", "tflite_int8")
DEFAULT_SELECTED_DECODE_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json")
DEFAULT_OUTPUT_DIR = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite")
DEFAULT_REPORT_PATH = Path("ml/reports/geometry_heatmap_v2_tflite_replay.md")
DEFAULT_PREDICTIONS_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/tflite_replay_predictions.csv")
DEFAULT_SUMMARY_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/tflite_replay_summary.csv")
DEFAULT_REMAINING_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/tflite_remaining_worst_accepted.csv")


@dataclass(frozen=True)
class ModelPredictionArtifacts:
    """Per-sample predictions for one model type."""

    sample: HeatmapSample
    model_type: str
    row: dict[str, Any]
    center_heatmap: np.ndarray
    tip_heatmap: np.ndarray
    decoded: GeometryDecodedPrediction
    guarded: GeometryGuardrailResult


def _resolve_path(base_path: Path, path: Path) -> Path:
    """Resolve relative paths against the repository root."""

    return path if path.is_absolute() else base_path / path


def _with_output_suffix(path: Path, output_suffix: str) -> Path:
    """Append an output suffix to a file stem when requested."""

    if not output_suffix:
        return path
    return path.with_name(f"{path.stem}_{output_suffix}{path.suffix}")


def _load_selected_decode_spec(selection_path: Path) -> tuple[str, str, int]:
    """Load the locked decode selection artifact."""

    with selection_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "decode_method" in payload and "window_size" in payload:
        selected_method = str(payload["decode_method"])
        selected_window_size = int(payload["window_size"])
        selected_name = f"{selected_method}_w{selected_window_size}"
    else:
        selected_name = str(payload["selected_decode_method"])
        selected_window_size = int(payload["selected_window_size"])
        if "_w" in selected_name:
            selected_method, selected_window_suffix = selected_name.rsplit("_w", 1)
            try:
                selected_window_size = int(selected_window_suffix)
            except ValueError:
                pass
        else:
            selected_method = selected_name
    if selected_method != "softargmax" or int(selected_window_size) != 3:
        raise RuntimeError(
            f"Expected corrected decode softargmax w3, found {selected_name}/{selected_method} w{selected_window_size} in {selection_path}."
        )
    return selected_name, selected_method, selected_window_size


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a deterministic CSV file."""

    if not rows:
        raise ValueError("Cannot write an empty CSV file.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_selected_thresholds(thresholds_path: Path) -> GeometryGuardrailThresholds:
    """Load the selected board guardrails from JSON."""

    with thresholds_path.open("r", encoding="utf-8") as handle:
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
        clamp_temperature_to_physical_range=bool(selected["clamp_temperature_to_physical_range"]),
    )


def _ordered_model_outputs(outputs: Any, output_names: list[str]) -> list[np.ndarray]:
    """Normalize model outputs to a list ordered like the saved model signature."""

    if isinstance(outputs, Mapping):
        return [np.asarray(outputs[name], dtype=np.float32) for name in output_names]
    return [np.asarray(output, dtype=np.float32) for output in list(outputs)]


def _predict_keras_batches(
    model: keras.Model,
    inputs: list[np.ndarray],
    *,
    batch_size: int,
) -> list[list[np.ndarray]]:
    """Predict all Keras outputs for one list of preprocessed inputs."""

    outputs_per_batch: list[list[np.ndarray]] = []
    for batch in iterate_batched(inputs, batch_size=batch_size):
        outputs = model(batch, training=False)
        ordered = _ordered_model_outputs(outputs, list(model.output_names))
        outputs_per_batch.append(ordered)

    if not outputs_per_batch:
        raise ValueError("No Keras batches were predicted.")

    merged: list[np.ndarray] = []
    for output_index in range(len(outputs_per_batch[0])):
        merged.append(np.concatenate([batch_outputs[output_index] for batch_outputs in outputs_per_batch], axis=0))
    return [merged]


def _predict_keras_outputs(
    model: keras.Model,
    inputs: list[np.ndarray],
    *,
    batch_size: int,
) -> list[np.ndarray]:
    """Predict Keras outputs and return the per-output tensors."""

    batches: list[list[np.ndarray]] = []
    for batch in iterate_batched(inputs, batch_size=batch_size):
        outputs = model(batch, training=False)
        ordered = _ordered_model_outputs(outputs, list(model.output_names))
        batches.append(ordered)

    if not batches:
        raise ValueError("No Keras batches were predicted.")

    merged: list[np.ndarray] = []
    for output_index in range(len(batches[0])):
        merged.append(np.concatenate([batch_outputs[output_index] for batch_outputs in batches], axis=0))
    return merged


def _predict_tflite_outputs(
    bundle: LoadedTFLiteModel,
    inputs: list[np.ndarray],
) -> list[np.ndarray]:
    """Predict TFLite outputs sample by sample and return per-output tensors."""

    outputs_accumulator: list[list[np.ndarray]] | None = None
    for input_array in inputs:
        model_outputs = run_tflite_model(bundle, np.expand_dims(np.asarray(input_array, dtype=np.float32), axis=0))
        if outputs_accumulator is None:
            outputs_accumulator = [[] for _ in model_outputs]
        for index, tensor in enumerate(model_outputs):
            outputs_accumulator[index].append(np.asarray(tensor, dtype=np.float32))

    if outputs_accumulator is None:
        raise ValueError("No TFLite outputs were predicted.")

    return [np.concatenate(output_tensors, axis=0) for output_tensors in outputs_accumulator]


def _prediction_row(
    sample: HeatmapSample,
    *,
    model_type: str,
    decoded: GeometryDecodedPrediction,
    guarded: GeometryGuardrailResult,
) -> dict[str, Any]:
    """Flatten one model prediction into a board-replay-style row."""

    features = guarded.quality_features
    rejection_reasons = ";".join(guarded.rejection_reasons)
    if not rejection_reasons:
        rejection_reasons = "none"
    row: dict[str, Any] = {
        "model_type": model_type,
        "image_path": str(sample.metadata["image_path"]),
        "split": str(sample.metadata["split"]),
        "source_kind": str(sample.metadata["source_kind"]),
        "preprocessing_mode": str(sample.metadata["preprocessing_mode"]),
        "true_temperature_c": float(decoded.true_temperature_c),
        "true_angle_degrees": float(decoded.true_angle_degrees),
        "true_center_x_224": float(decoded.true_center_x_224),
        "true_center_y_224": float(decoded.true_center_y_224),
        "true_tip_x_224": float(decoded.true_tip_x_224),
        "true_tip_y_224": float(decoded.true_tip_y_224),
        "predicted_center_x_224": float(decoded.predicted_center_x_224),
        "predicted_center_y_224": float(decoded.predicted_center_y_224),
        "predicted_tip_x_224": float(decoded.predicted_tip_x_224),
        "predicted_tip_y_224": float(decoded.predicted_tip_y_224),
        "predicted_center_x_224_argmax": float(decoded.predicted_center_x_224_argmax),
        "predicted_center_y_224_argmax": float(decoded.predicted_center_y_224_argmax),
        "predicted_tip_x_224_argmax": float(decoded.predicted_tip_x_224_argmax),
        "predicted_tip_y_224_argmax": float(decoded.predicted_tip_y_224_argmax),
        "predicted_angle_degrees": float(decoded.predicted_angle_degrees),
        "predicted_angle_degrees_argmax": float(decoded.predicted_angle_degrees_argmax),
        "angle_mae_degrees": float(circular_angle_error_degrees(decoded.predicted_angle_degrees, decoded.true_angle_degrees)),
        "angle_mae_degrees_argmax": float(
            circular_angle_error_degrees(decoded.predicted_angle_degrees_argmax, decoded.true_angle_degrees)
        ),
        "predicted_temperature_c_current_mapping": float(decoded.predicted_temperature_c_current_mapping),
        "predicted_temperature_c_current_mapping_argmax": float(decoded.predicted_temperature_c_current_mapping_argmax),
        "predicted_temperature_c_calibrated": float(decoded.predicted_temperature_c_calibrated),
        "predicted_temperature_c_calibrated_argmax": float(decoded.predicted_temperature_c_calibrated_argmax),
        "absolute_error_c_current_mapping": float(decoded.absolute_error_c_current_mapping),
        "absolute_error_c_current_mapping_argmax": float(decoded.absolute_error_c_current_mapping_argmax),
        "absolute_error_c_calibrated": float(decoded.absolute_error_c_calibrated),
        "absolute_error_c_calibrated_argmax": float(decoded.absolute_error_c_calibrated_argmax),
        "center_px_mae_224": float(
            math.hypot(
                decoded.predicted_center_x_224 - decoded.true_center_x_224,
                decoded.predicted_center_y_224 - decoded.true_center_y_224,
            )
        ),
        "tip_px_mae_224": float(
            math.hypot(
                decoded.predicted_tip_x_224 - decoded.true_tip_x_224,
                decoded.predicted_tip_y_224 - decoded.true_tip_y_224,
            )
        ),
        "center_px_mae_224_argmax": float(
            math.hypot(
                decoded.predicted_center_x_224_argmax - decoded.true_center_x_224,
                decoded.predicted_center_y_224_argmax - decoded.true_center_y_224,
            )
        ),
        "tip_px_mae_224_argmax": float(
            math.hypot(
                decoded.predicted_tip_x_224_argmax - decoded.true_tip_x_224,
                decoded.predicted_tip_y_224_argmax - decoded.true_tip_y_224,
            )
        ),
        "center_heatmap_peak_value": float(decoded.center_heatmap_peak_value),
        "tip_heatmap_peak_value": float(decoded.tip_heatmap_peak_value),
        "center_heatmap_mean_value": float(decoded.center_heatmap_mean_value),
        "tip_heatmap_mean_value": float(decoded.tip_heatmap_mean_value),
        "center_heatmap_entropy": float(features.center_heatmap_entropy),
        "tip_heatmap_entropy": float(features.tip_heatmap_entropy),
        "center_heatmap_spread_px": float(features.center_heatmap_spread_px),
        "tip_heatmap_spread_px": float(features.tip_heatmap_spread_px),
        "confidence": float(decoded.confidence),
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
        "guardrail_status": str(guarded.status),
        "rejection_reasons": rejection_reasons,
        "guarded_temperature_c": float(guarded.temperature_c),
        "raw_temperature_c": float(guarded.raw_temperature_c),
        "absolute_error_c_guarded": float(abs(guarded.temperature_c - decoded.true_temperature_c))
        if guarded.status != "rejected"
        else math.nan,
        "crop_x1": int(decoded.crop_x1),
        "crop_y1": int(decoded.crop_y1),
        "crop_x2": int(decoded.crop_x2),
        "crop_y2": int(decoded.crop_y2),
        "crop_width": int(decoded.crop_width),
        "crop_height": int(decoded.crop_height),
        "source_width": int(sample.metadata["source_width"]),
        "source_height": int(sample.metadata["source_height"]),
        "source_manifest": str(sample.metadata["source_manifest"]),
        "quality_flag": str(sample.metadata["quality_flag"]),
        "dial_radius_source": float(decoded.dial_radius_source),
        "resize_method": str(sample.metadata["resize_method"]),
        "channel_strategy": str(sample.metadata["channel_strategy"]),
        "normalization": str(sample.metadata["normalization"]),
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
        "jitter_shift_x": int(decoded.jitter_shift_x),
        "jitter_shift_y": int(decoded.jitter_shift_y),
        "jitter_scale": float(decoded.jitter_scale),
        "jitter_aspect": float(decoded.jitter_aspect),
    }
    return row


def _compute_drift_rows(
    keras_row: Mapping[str, Any],
    fp32_row: Mapping[str, Any],
    int8_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute cross-model drift statistics for one sample."""

    def _finite_temperature_delta(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> float:
        if str(row_a["guardrail_status"]) == "rejected" or str(row_b["guardrail_status"]) == "rejected":
            return math.nan
        return float(abs(float(row_a["guarded_temperature_c"]) - float(row_b["guarded_temperature_c"])))

    return {
        "keras_vs_tflite_fp32_temperature_delta_c": _finite_temperature_delta(keras_row, fp32_row),
        "keras_vs_tflite_int8_temperature_delta_c": _finite_temperature_delta(keras_row, int8_row),
        "keras_vs_tflite_int8_center_point_delta_px": float(
            math.hypot(
                float(keras_row["predicted_center_x_224"]) - float(int8_row["predicted_center_x_224"]),
                float(keras_row["predicted_center_y_224"]) - float(int8_row["predicted_center_y_224"]),
            )
        ),
        "keras_vs_tflite_int8_tip_point_delta_px": float(
            math.hypot(
                float(keras_row["predicted_tip_x_224"]) - float(int8_row["predicted_tip_x_224"]),
                float(keras_row["predicted_tip_y_224"]) - float(int8_row["predicted_tip_y_224"]),
            )
        ),
        "keras_vs_tflite_int8_center_heatmap_peak_delta": float(
            abs(float(keras_row["center_heatmap_peak_value"]) - float(int8_row["center_heatmap_peak_value"]))
        ),
        "keras_vs_tflite_int8_tip_heatmap_peak_delta": float(
            abs(float(keras_row["tip_heatmap_peak_value"]) - float(int8_row["tip_heatmap_peak_value"]))
        ),
        "keras_vs_tflite_int8_rejection_status_disagreement": bool(
            str(keras_row["guardrail_status"]) != str(int8_row["guardrail_status"])
        ),
    }


def _summarize_rows(rows: list[dict[str, Any]], thresholds: GeometryGuardrailThresholds) -> dict[str, Any]:
    """Summarize one model's replay rows for a split."""

    accepted_summary = summarize_board_replay_rows(rows, thresholds)
    center_errors = np.asarray([float(row["center_px_mae_224"]) for row in rows], dtype=np.float64)
    tip_errors = np.asarray([float(row["tip_px_mae_224"]) for row in rows], dtype=np.float64)
    angle_errors = np.asarray([float(row["angle_mae_degrees"]) for row in rows], dtype=np.float64)
    center_peaks = np.asarray([float(row["center_heatmap_peak_value"]) for row in rows], dtype=np.float64)
    tip_peaks = np.asarray([float(row["tip_heatmap_peak_value"]) for row in rows], dtype=np.float64)
    confidences = np.asarray([float(row["confidence"]) for row in rows], dtype=np.float64)

    return {
        **accepted_summary,
        "center_mae_px_224": float(np.mean(center_errors)),
        "tip_mae_px_224": float(np.mean(tip_errors)),
        "angle_mae_degrees": float(np.mean(angle_errors)),
        "center_heatmap_peak_mean": float(np.mean(center_peaks)),
        "center_heatmap_peak_median": float(np.median(center_peaks)),
        "tip_heatmap_peak_mean": float(np.mean(tip_peaks)),
        "tip_heatmap_peak_median": float(np.median(tip_peaks)),
        "confidence_mean": float(np.mean(confidences)),
        "confidence_median": float(np.median(confidences)),
    }


def _score_drift(
    keras_rows: list[dict[str, Any]],
    fp32_rows: list[dict[str, Any]],
    int8_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate cross-model drift statistics for one split."""

    paired_keras_fp32 = [
        abs(float(k["guarded_temperature_c"]) - float(f["guarded_temperature_c"]))
        for k, f in zip(keras_rows, fp32_rows, strict=True)
        if str(k["guardrail_status"]) != "rejected" and str(f["guardrail_status"]) != "rejected"
    ]
    paired_keras_int8 = [
        abs(float(k["guarded_temperature_c"]) - float(i["guarded_temperature_c"]))
        for k, i in zip(keras_rows, int8_rows, strict=True)
        if str(k["guardrail_status"]) != "rejected" and str(i["guardrail_status"]) != "rejected"
    ]
    center_point_delta = [
        math.hypot(
            float(k["predicted_center_x_224"]) - float(i["predicted_center_x_224"]),
            float(k["predicted_center_y_224"]) - float(i["predicted_center_y_224"]),
        )
        for k, i in zip(keras_rows, int8_rows, strict=True)
    ]
    tip_point_delta = [
        math.hypot(
            float(k["predicted_tip_x_224"]) - float(i["predicted_tip_x_224"]),
            float(k["predicted_tip_y_224"]) - float(i["predicted_tip_y_224"]),
        )
        for k, i in zip(keras_rows, int8_rows, strict=True)
    ]
    center_peak_delta = [
        abs(float(k["center_heatmap_peak_value"]) - float(i["center_heatmap_peak_value"]))
        for k, i in zip(keras_rows, int8_rows, strict=True)
    ]
    tip_peak_delta = [
        abs(float(k["tip_heatmap_peak_value"]) - float(i["tip_heatmap_peak_value"]))
        for k, i in zip(keras_rows, int8_rows, strict=True)
    ]
    rejection_disagreements = sum(
        1
        for k, i in zip(keras_rows, int8_rows, strict=True)
        if str(k["guardrail_status"]) != str(i["guardrail_status"])
    )

    return {
        "keras_vs_tflite_fp32_temperature_delta_mean_c": float(np.mean(paired_keras_fp32)) if paired_keras_fp32 else math.nan,
        "keras_vs_tflite_fp32_temperature_delta_median_c": float(np.median(paired_keras_fp32)) if paired_keras_fp32 else math.nan,
        "keras_vs_tflite_int8_temperature_delta_mean_c": float(np.mean(paired_keras_int8)) if paired_keras_int8 else math.nan,
        "keras_vs_tflite_int8_temperature_delta_median_c": float(np.median(paired_keras_int8)) if paired_keras_int8 else math.nan,
        "keras_vs_tflite_int8_center_point_delta_mean_px": float(np.mean(center_point_delta)) if center_point_delta else math.nan,
        "keras_vs_tflite_int8_tip_point_delta_mean_px": float(np.mean(tip_point_delta)) if tip_point_delta else math.nan,
        "keras_vs_tflite_int8_center_heatmap_peak_delta_mean": float(np.mean(center_peak_delta)) if center_peak_delta else math.nan,
        "keras_vs_tflite_int8_tip_heatmap_peak_delta_mean": float(np.mean(tip_peak_delta)) if tip_peak_delta else math.nan,
        "keras_vs_tflite_int8_rejection_status_disagreement_count": int(rejection_disagreements),
    }


def _summarize_split_drift(split: str, drift_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one split's drift rows for the markdown report."""

    split_rows = [row for row in drift_rows if str(row["split"]) == split]
    fp32_temp = np.asarray(
        [
            float(row["keras_vs_tflite_fp32_temperature_delta_c"])
            for row in split_rows
            if math.isfinite(float(row["keras_vs_tflite_fp32_temperature_delta_c"]))
        ],
        dtype=np.float64,
    )
    int8_temp = np.asarray(
        [
            float(row["keras_vs_tflite_int8_temperature_delta_c"])
            for row in split_rows
            if math.isfinite(float(row["keras_vs_tflite_int8_temperature_delta_c"]))
        ],
        dtype=np.float64,
    )
    center_delta = np.asarray([float(row["keras_vs_tflite_int8_center_point_delta_px"]) for row in split_rows], dtype=np.float64)
    tip_delta = np.asarray([float(row["keras_vs_tflite_int8_tip_point_delta_px"]) for row in split_rows], dtype=np.float64)
    center_peak_delta = np.asarray(
        [float(row["keras_vs_tflite_int8_center_heatmap_peak_delta"]) for row in split_rows],
        dtype=np.float64,
    )
    tip_peak_delta = np.asarray([float(row["keras_vs_tflite_int8_tip_heatmap_peak_delta"]) for row in split_rows], dtype=np.float64)
    disagreements = int(sum(1 for row in split_rows if bool(row["keras_vs_tflite_int8_rejection_status_disagreement"])))

    return {
        "split": split,
        "keras_vs_tflite_fp32_temperature_delta_mean_c": float(np.mean(fp32_temp)) if fp32_temp.size else math.nan,
        "keras_vs_tflite_fp32_temperature_delta_median_c": float(np.median(fp32_temp)) if fp32_temp.size else math.nan,
        "keras_vs_tflite_int8_temperature_delta_mean_c": float(np.mean(int8_temp)) if int8_temp.size else math.nan,
        "keras_vs_tflite_int8_temperature_delta_median_c": float(np.median(int8_temp)) if int8_temp.size else math.nan,
        "keras_vs_tflite_int8_center_point_delta_mean_px": float(np.mean(center_delta)) if center_delta.size else math.nan,
        "keras_vs_tflite_int8_tip_point_delta_mean_px": float(np.mean(tip_delta)) if tip_delta.size else math.nan,
        "keras_vs_tflite_int8_center_heatmap_peak_delta_mean": float(np.mean(center_peak_delta)) if center_peak_delta.size else math.nan,
        "keras_vs_tflite_int8_tip_heatmap_peak_delta_mean": float(np.mean(tip_peak_delta)) if tip_peak_delta.size else math.nan,
        "keras_vs_tflite_int8_rejection_status_disagreement_count": disagreements,
    }


def _make_overlay(
    *,
    sample: HeatmapSample,
    keras_row: Mapping[str, Any],
    keras_center_heatmap: np.ndarray,
    keras_tip_heatmap: np.ndarray,
    int8_row: Mapping[str, Any],
    int8_center_heatmap: np.ndarray,
    int8_tip_heatmap: np.ndarray,
    output_path: Path,
) -> None:
    """Render one comparison overlay with Keras and INT8 side by side."""

    fig = plt.figure(figsize=(19, 11), dpi=150)
    grid = fig.add_gridspec(2, 4, width_ratios=(1.45, 0.95, 0.95, 1.15), height_ratios=(1.0, 1.0))

    ax_crop = fig.add_subplot(grid[:, 0])
    ax_keras_center = fig.add_subplot(grid[0, 1])
    ax_keras_tip = fig.add_subplot(grid[1, 1])
    ax_int8_center = fig.add_subplot(grid[0, 2])
    ax_int8_tip = fig.add_subplot(grid[1, 2])
    ax_text = fig.add_subplot(grid[:, 3])

    crop = np.asarray(sample.crop_image, dtype=np.float32)
    ax_crop.imshow(np.clip(crop, 0.0, 1.0))
    ax_crop.scatter(
        [float(sample.metadata["center_x_224"]), float(keras_row["predicted_center_x_224"]), float(int8_row["predicted_center_x_224"])],
        [float(sample.metadata["center_y_224"]), float(keras_row["predicted_center_y_224"]), float(int8_row["predicted_center_y_224"])],
        c=["lime", "deepskyblue", "orange"],
        s=[70, 50, 50],
        marker="o",
        edgecolors="white",
        linewidths=1.0,
    )
    ax_crop.scatter(
        [float(sample.metadata["tip_x_224"]), float(keras_row["predicted_tip_x_224"]), float(int8_row["predicted_tip_x_224"])],
        [float(sample.metadata["tip_y_224"]), float(keras_row["predicted_tip_y_224"]), float(int8_row["predicted_tip_y_224"])],
        c=["red", "deepskyblue", "orange"],
        s=[75, 55, 55],
        marker="x",
        linewidths=2.0,
    )
    ax_crop.plot(
        [float(sample.metadata["center_x_224"]), float(sample.metadata["tip_x_224"])],
        [float(sample.metadata["center_y_224"]), float(sample.metadata["tip_y_224"])],
        color="white",
        linewidth=2.0,
        alpha=0.9,
    )
    ax_crop.plot(
        [float(keras_row["predicted_center_x_224"]), float(keras_row["predicted_tip_x_224"])],
        [float(keras_row["predicted_center_y_224"]), float(keras_row["predicted_tip_y_224"])],
        color="deepskyblue",
        linewidth=2.0,
        alpha=0.9,
    )
    ax_crop.plot(
        [float(int8_row["predicted_center_x_224"]), float(int8_row["predicted_tip_x_224"])],
        [float(int8_row["predicted_center_y_224"]), float(int8_row["predicted_tip_y_224"])],
        color="orange",
        linewidth=2.0,
        alpha=0.9,
    )
    ax_crop.set_title("Crop fed to the model")
    ax_crop.set_axis_off()

    def _plot_heatmap(
        ax: plt.Axes,
        heatmap: np.ndarray,
        *,
        title: str,
        true_x: float,
        true_y: float,
        pred_x: float,
        pred_y: float,
    ) -> None:
        """Show one predicted heatmap and annotate true vs predicted points."""

        squeezed = np.squeeze(np.asarray(heatmap, dtype=np.float32))
        ax.imshow(squeezed, cmap="magma", origin="upper")
        ax.scatter(
            [true_x * (squeezed.shape[1] - 1) / 223.0],
            [true_y * (squeezed.shape[0] - 1) / 223.0],
            c="white",
            s=40,
            marker="o",
            edgecolors="black",
            linewidths=0.8,
        )
        ax.scatter(
            [pred_x * (squeezed.shape[1] - 1) / 223.0],
            [pred_y * (squeezed.shape[0] - 1) / 223.0],
            c="cyan",
            s=45,
            marker="x",
            linewidths=2.0,
        )
        ax.set_xlim(-0.5, squeezed.shape[1] - 0.5)
        ax.set_ylim(squeezed.shape[0] - 0.5, -0.5)
        ax.set_title(f"{title}\npeak={float(np.max(squeezed)):.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    _plot_heatmap(
        ax_keras_center,
        keras_center_heatmap,
        title="Keras center heatmap",
        true_x=float(sample.metadata["center_x_224"]),
        true_y=float(sample.metadata["center_y_224"]),
        pred_x=float(keras_row["predicted_center_x_224"]),
        pred_y=float(keras_row["predicted_center_y_224"]),
    )
    _plot_heatmap(
        ax_keras_tip,
        keras_tip_heatmap,
        title="Keras tip heatmap",
        true_x=float(sample.metadata["tip_x_224"]),
        true_y=float(sample.metadata["tip_y_224"]),
        pred_x=float(keras_row["predicted_tip_x_224"]),
        pred_y=float(keras_row["predicted_tip_y_224"]),
    )
    _plot_heatmap(
        ax_int8_center,
        int8_center_heatmap,
        title="INT8 center heatmap",
        true_x=float(sample.metadata["center_x_224"]),
        true_y=float(sample.metadata["center_y_224"]),
        pred_x=float(int8_row["predicted_center_x_224"]),
        pred_y=float(int8_row["predicted_center_y_224"]),
    )
    _plot_heatmap(
        ax_int8_tip,
        int8_tip_heatmap,
        title="INT8 tip heatmap",
        true_x=float(sample.metadata["tip_x_224"]),
        true_y=float(sample.metadata["tip_y_224"]),
        pred_x=float(int8_row["predicted_tip_x_224"]),
        pred_y=float(int8_row["predicted_tip_y_224"]),
    )

    keras_reasons = str(keras_row["rejection_reasons"]) or "none"
    int8_reasons = str(int8_row["rejection_reasons"]) or "none"
    text_lines = [
        f"file: {Path(str(sample.metadata['image_path'])).name}",
        f"split: {sample.metadata['split']}",
        f"mode: {sample.metadata['preprocessing_mode']}",
        "",
        f"true temp: {float(sample.metadata['temperature_c']):.2f} C",
        f"Keras temp: {float(keras_row['guarded_temperature_c']):.2f} C",
        f"INT8 temp: {float(int8_row['guarded_temperature_c']):.2f} C",
        f"Keras abs err: {float(keras_row['absolute_error_c_guarded']) if math.isfinite(float(keras_row['absolute_error_c_guarded'])) else float('nan'):.2f} C",
        f"INT8 abs err: {float(int8_row['absolute_error_c_guarded']) if math.isfinite(float(int8_row['absolute_error_c_guarded'])) else float('nan'):.2f} C",
        "",
        f"Keras status: {keras_row['guardrail_status']}",
        f"Keras reasons: {keras_reasons}",
        f"INT8 status: {int8_row['guardrail_status']}",
        f"INT8 reasons: {int8_reasons}",
        "",
        f"Keras confidence: {float(keras_row['confidence']):.4f}",
        f"INT8 confidence: {float(int8_row['confidence']):.4f}",
        f"Keras center peak: {float(keras_row['center_heatmap_peak_value']):.4f}",
        f"INT8 center peak: {float(int8_row['center_heatmap_peak_value']):.4f}",
        f"Keras tip peak: {float(keras_row['tip_heatmap_peak_value']):.4f}",
        f"INT8 tip peak: {float(int8_row['tip_heatmap_peak_value']):.4f}",
        "",
        f"center delta: {math.hypot(float(keras_row['predicted_center_x_224']) - float(int8_row['predicted_center_x_224']), float(keras_row['predicted_center_y_224']) - float(int8_row['predicted_center_y_224'])):.2f} px",
        f"tip delta: {math.hypot(float(keras_row['predicted_tip_x_224']) - float(int8_row['predicted_tip_x_224']), float(keras_row['predicted_tip_y_224']) - float(int8_row['predicted_tip_y_224'])):.2f} px",
        f"temp delta: {abs(float(keras_row['guarded_temperature_c']) - float(int8_row['guarded_temperature_c'])) if str(keras_row['guardrail_status']) != 'rejected' and str(int8_row['guardrail_status']) != 'rejected' else float('nan'):.2f} C",
    ]
    ax_text.set_axis_off()
    ax_text.text(0.0, 1.0, "\n".join(text_lines), family="monospace", fontsize=9.0, va="top")

    fig.suptitle(
        f"{Path(str(sample.metadata['image_path'])).name} | Keras {keras_row['guardrail_status']} / INT8 {int8_row['guardrail_status']}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _select_overlay_rows(
    rows_by_model: dict[str, list[dict[str, Any]]],
    *,
    split: str,
) -> dict[str, list[dict[str, Any]]]:
    """Select the drift-focused overlay rows for the test split."""

    keras_rows = {str(row["image_path"]): row for row in rows_by_model["keras_fp32"] if str(row["split"]) == split}
    int8_rows = {str(row["image_path"]): row for row in rows_by_model["tflite_int8"] if str(row["split"]) == split}

    accepted_int8 = [
        row for row in int8_rows.values() if str(row["guardrail_status"]) != "rejected"
    ]
    worst_accepted = sorted(accepted_int8, key=lambda row: float(row["absolute_error_c_guarded"]), reverse=True)[:30]
    accepted_gt10 = [
        row for row in accepted_int8 if float(row["absolute_error_c_guarded"]) > 10.0
    ]
    keras_accept_int8_reject = [
        int8_rows[path]
        for path, keras_row in keras_rows.items()
        if str(keras_row["guardrail_status"]) != "rejected" and str(int8_rows[path]["guardrail_status"]) == "rejected"
    ]
    keras_reject_int8_accept = [
        int8_rows[path]
        for path, keras_row in keras_rows.items()
        if str(keras_row["guardrail_status"]) == "rejected" and str(int8_rows[path]["guardrail_status"]) != "rejected"
    ]

    common_temp_delta = [
        (
            path,
            abs(float(keras_rows[path]["guarded_temperature_c"]) - float(int8_rows[path]["guarded_temperature_c"])),
        )
        for path in keras_rows
        if str(keras_rows[path]["guardrail_status"]) != "rejected" and str(int8_rows[path]["guardrail_status"]) != "rejected"
    ]
    largest_delta = [
        int8_rows[path]
        for path, _ in sorted(common_temp_delta, key=lambda item: item[1], reverse=True)[:30]
    ]

    return {
        "test_worst_30_accepted": worst_accepted,
        "test_accepted_gt10c": accepted_gt10,
        "keras_accept_int8_reject": keras_accept_int8_reject,
        "keras_reject_int8_accept": keras_reject_int8_accept,
        "largest_keras_int8_temp_delta": largest_delta,
    }


def _write_report(
    *,
    report_path: Path,
    summary_rows: list[dict[str, Any]],
    drift_rows: list[dict[str, Any]],
    selected_decode_name: str,
    selected_decode_method: str,
    selected_decode_window_size: int,
) -> None:
    """Write the markdown replay report."""

    lines = [
        "# Geometry Heatmap v2 TFLite Replay",
        "",
        "## Decode Selection",
        "",
        f"- Selected decode name: `{selected_decode_name}`",
        f"- Selected decode method: `{selected_decode_method}`",
        f"- Selected decode window size: `{selected_decode_window_size}`",
        "",
        "## Replay Summary",
        "",
        "| model | split | accepted MAE | acceptance rate | worst accepted error | rejected | clamped | center MAE | tip MAE | angle MAE | center peak mean | tip peak mean | confidence mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model_type']} | {row['split']} | {row['accepted_mae_c']:.3f} | {row['acceptance_rate']:.3f} | {row['accepted_worst_error_c']:.3f} | {int(row['rejected_count'])} | {int(row['clamped_count'])} | {row['center_mae_px_224']:.3f} | {row['tip_mae_px_224']:.3f} | {row['angle_mae_degrees']:.3f} | {row['center_heatmap_peak_mean']:.4f} | {row['tip_heatmap_peak_mean']:.4f} | {row['confidence_mean']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Drift Against Keras",
            "",
            "| split | Keras vs FP32 temp delta mean | Keras vs FP32 temp delta median | Keras vs INT8 temp delta mean | Keras vs INT8 temp delta median | INT8 center delta mean | INT8 tip delta mean | INT8 center peak delta mean | INT8 tip peak delta mean | rejection disagreements |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in drift_rows:
        lines.append(
            f"| {row['split']} | {row['keras_vs_tflite_fp32_temperature_delta_mean_c']:.3f} | {row['keras_vs_tflite_fp32_temperature_delta_median_c']:.3f} | {row['keras_vs_tflite_int8_temperature_delta_mean_c']:.3f} | {row['keras_vs_tflite_int8_temperature_delta_median_c']:.3f} | {row['keras_vs_tflite_int8_center_point_delta_mean_px']:.3f} | {row['keras_vs_tflite_int8_tip_point_delta_mean_px']:.3f} | {row['keras_vs_tflite_int8_center_heatmap_peak_delta_mean']:.4f} | {row['keras_vs_tflite_int8_tip_heatmap_peak_delta_mean']:.4f} | {int(row['keras_vs_tflite_int8_rejection_status_disagreement_count'])} |"
        )

    lines.extend(["", "## Notes", "", "- Accepted metrics are computed on accepted and clamped predictions.", "- Center/tip/angle/peak/confidence metrics are reported over all rows.", ""])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _prefix_row(row: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Prefix a row's keys for wide CSV storage."""

    return {f"{prefix}{key}": value for key, value in row.items()}


def main() -> None:
    """Run Keras and TFLite replay on the clean manifest and summarize drift."""

    parser = argparse.ArgumentParser(description="Replay geometry_heatmap_v2 through TFLite")
    parser.add_argument(
        "--keras-model-path",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2/model.keras"),
        help="Source Keras model.",
    )
    parser.add_argument(
        "--float32-model-path",
        type=Path,
        default=Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/model_float32.tflite"),
        help="Float32 TFLite model.",
    )
    parser.add_argument(
        "--int8-model-path",
        type=Path,
        default=Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/model_int8.tflite"),
        help="Int8 TFLite model.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"),
        help="Clean manifest path.",
    )
    parser.add_argument(
        "--calibration-json-path",
        type=Path,
        default=Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json"),
        help="Calibration artifact path.",
    )
    parser.add_argument(
        "--selected-thresholds-path",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2_board_replay/selected_board_guardrail_thresholds.json"),
        help="Selected board thresholds.",
    )
    parser.add_argument(
        "--selected-decode-path",
        type=Path,
        default=DEFAULT_SELECTED_DECODE_PATH,
        help="Locked decode selection artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for replay CSV artifacts.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown report path.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Wide replay CSV path.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Summary CSV path.",
    )
    parser.add_argument(
        "--remaining-path",
        type=Path,
        default=DEFAULT_REMAINING_PATH,
        help="Worst accepted rows CSV path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "val", "test"),
        default="test",
        help="Which split to replay.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional smoke-test cap on the number of split samples.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix appended to report and CSV filenames.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("ml/debug/geometry_heatmap_v2_tflite_replay"),
        help="Directory for overlay images.",
    )
    parser.add_argument("--input-size", type=int, default=224, help="Model input size.")
    parser.add_argument("--heatmap-size", type=int, default=56, help="Heatmap size.")
    parser.add_argument("--sigma-pixels", type=float, default=5.0, help="Heatmap sigma.")
    parser.add_argument("--batch-size", type=int, default=32, help="Keras inference batch size.")
    parser.add_argument("--overlay-seed", type=int, default=42, help="Overlay sampling seed.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    keras_model_path = _resolve_path(repo_root, args.keras_model_path)
    float32_model_path = _resolve_path(repo_root, args.float32_model_path)
    int8_model_path = _resolve_path(repo_root, args.int8_model_path)
    manifest_path = _resolve_path(repo_root, args.manifest_path)
    calibration_json_path = _resolve_path(repo_root, args.calibration_json_path)
    selected_thresholds_path = _resolve_path(repo_root, args.selected_thresholds_path)
    selected_decode_path = _resolve_path(repo_root, args.selected_decode_path)
    output_dir = _resolve_path(repo_root, args.output_dir)
    report_path = _with_output_suffix(_resolve_path(repo_root, args.report_path), args.output_suffix)
    predictions_path = _with_output_suffix(_resolve_path(repo_root, args.predictions_path), args.output_suffix)
    summary_path = _with_output_suffix(_resolve_path(repo_root, args.summary_path), args.output_suffix)
    remaining_path = _with_output_suffix(_resolve_path(repo_root, args.remaining_path), args.output_suffix)
    debug_dir = _resolve_path(repo_root, args.debug_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    calibration_candidate, _ = load_selected_calibration_candidate(calibration_json_path)
    thresholds = _load_selected_thresholds(selected_thresholds_path)
    selected_decode_name, selected_decode_method, selected_decode_window_size = _load_selected_decode_spec(selected_decode_path)
    examples = load_clean_geometry_examples(manifest_path)
    split_examples = {
        split: select_examples_from_split(examples, split=split)
        for split in (args.split,)
    }

    keras_model = load_geometry_heatmap_keras_model(keras_model_path)
    float32_bundle = load_tflite_model(float32_model_path)
    int8_bundle = load_tflite_model(int8_model_path)

    all_wide_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    all_drift_rows: list[dict[str, Any]] = []
    split_rows_by_model: dict[str, dict[str, list[dict[str, Any]]]] = {}
    split_artifacts_by_model: dict[str, dict[str, list[ModelPredictionArtifacts]]] = {}

    for split, examples_for_split in split_examples.items():
        samples = [
            build_board_replay_sample(
                example,
                repo_root,
                mode=SELECTED_PREPROCESSING_MODE,
                input_size=args.input_size,
                heatmap_size=args.heatmap_size,
                sigma_pixels=args.sigma_pixels,
            )
            for example in examples_for_split
        ]
        if args.max_samples is not None:
            samples = samples[: int(args.max_samples)]
        print(
            f"[REPLAY] Starting split {split} with {len(samples)} samples "
            f"(output suffix: {args.output_suffix or 'none'})",
            flush=True,
        )
        inputs = [np.asarray(sample.crop_image, dtype=np.float32) for sample in samples]

        keras_outputs = _predict_keras_outputs(keras_model, inputs, batch_size=args.batch_size)
        float32_outputs = _predict_tflite_outputs(float32_bundle, inputs)
        int8_outputs = _predict_tflite_outputs(int8_bundle, inputs)

        # The exported TFLite graph emits the two heatmap tensors in reverse
        # semantic order relative to the original Keras model signature.
        float32_outputs = [float32_outputs[1], float32_outputs[0], float32_outputs[2]]
        int8_outputs = [int8_outputs[1], int8_outputs[0], int8_outputs[2]]

        split_rows_by_model[split] = {model_type: [] for model_type in MODEL_TYPES}
        split_artifacts_by_model[split] = {model_type: [] for model_type in MODEL_TYPES}

        for index, sample in enumerate(samples):
            model_output_map = {
                "keras_fp32": keras_outputs,
                "tflite_fp32": float32_outputs,
                "tflite_int8": int8_outputs,
            }
            if index % 5 == 0 or index + 1 == len(samples):
                print(f"[REPLAY] split={split} sample {index + 1}/{len(samples)}", flush=True)
            for model_type, outputs in model_output_map.items():
                center_heatmap = np.asarray(outputs[0][index], dtype=np.float32)
                tip_heatmap = np.asarray(outputs[1][index], dtype=np.float32)
                confidence_tensor = np.asarray(outputs[2][index], dtype=np.float32)
                confidence = float(np.ravel(confidence_tensor)[0])
                decoded = decode_heatmap_geometry_prediction(
                    sample,
                    center_heatmap,
                    tip_heatmap,
                    confidence,
                    calibration_candidate,
                    decode_method=selected_decode_method,
                    window_size=selected_decode_window_size,
                )
                guarded = apply_geometry_guardrails(decoded, thresholds)
                row = _prediction_row(sample, model_type=model_type, decoded=decoded, guarded=guarded)
                split_rows_by_model[split][model_type].append(row)
                split_artifacts_by_model[split][model_type].append(
                    ModelPredictionArtifacts(
                        sample=sample,
                        model_type=model_type,
                        row=row,
                        center_heatmap=center_heatmap,
                        tip_heatmap=tip_heatmap,
                        decoded=decoded,
                        guarded=guarded,
                    )
                )

            keras_row = split_rows_by_model[split]["keras_fp32"][-1]
            float32_row = split_rows_by_model[split]["tflite_fp32"][-1]
            int8_row = split_rows_by_model[split]["tflite_int8"][-1]
            drift_row = {
                "image_path": str(sample.metadata["image_path"]),
                "split": split,
                **_compute_drift_rows(keras_row, float32_row, int8_row),
            }
            all_drift_rows.append(drift_row)

            wide_row = {
                "image_path": str(sample.metadata["image_path"]),
                "split": split,
                "source_kind": str(sample.metadata["source_kind"]),
                "preprocessing_mode": str(sample.metadata["preprocessing_mode"]),
                "decode_name": selected_decode_name,
                "decode_method": selected_decode_method,
                "decode_window_size": int(selected_decode_window_size),
                "selected_decode_path": str(selected_decode_path),
                "calibration_path": str(calibration_json_path),
                "guardrail_thresholds_path": str(selected_thresholds_path),
                "true_temperature_c": float(sample.metadata["temperature_c"]),
                "true_angle_degrees": float(keras_row["true_angle_degrees"]),
                "true_center_x_224": float(keras_row["true_center_x_224"]),
                "true_center_y_224": float(keras_row["true_center_y_224"]),
                "true_tip_x_224": float(keras_row["true_tip_x_224"]),
                "true_tip_y_224": float(keras_row["true_tip_y_224"]),
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
                "resize_method": str(sample.metadata["resize_method"]),
                "channel_strategy": str(sample.metadata["channel_strategy"]),
                "normalization": str(sample.metadata["normalization"]),
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
                "model_variant_name": str(model_type),
            }
            wide_row.update(_prefix_row(keras_row, "keras_"))
            wide_row.update(_prefix_row(float32_row, "tflite_fp32_"))
            wide_row.update(_prefix_row(int8_row, "tflite_int8_"))
            wide_row.update(
                {
                    "keras_vs_tflite_fp32_temperature_delta_c": drift_row["keras_vs_tflite_fp32_temperature_delta_c"],
                    "keras_vs_tflite_int8_temperature_delta_c": drift_row["keras_vs_tflite_int8_temperature_delta_c"],
                    "keras_vs_tflite_int8_center_point_delta_px": drift_row["keras_vs_tflite_int8_center_point_delta_px"],
                    "keras_vs_tflite_int8_tip_point_delta_px": drift_row["keras_vs_tflite_int8_tip_point_delta_px"],
                    "keras_vs_tflite_int8_center_heatmap_peak_delta": drift_row["keras_vs_tflite_int8_center_heatmap_peak_delta"],
                    "keras_vs_tflite_int8_tip_heatmap_peak_delta": drift_row["keras_vs_tflite_int8_tip_heatmap_peak_delta"],
                    "keras_vs_tflite_int8_rejection_status_disagreement": drift_row["keras_vs_tflite_int8_rejection_status_disagreement"],
                }
            )
            all_wide_rows.append(wide_row)

        for model_type in MODEL_TYPES:
            model_rows = split_rows_by_model[split][model_type]
            summary = _summarize_rows(model_rows, thresholds)
            summary_row = {
                "model_type": model_type,
                "split": split,
                **summary,
            }
            all_summary_rows.append(summary_row)

    selected_int8_rows = [row for row in all_wide_rows if str(row["split"]) == "test" and str(row["tflite_int8_guardrail_status"]) != "rejected"]
    remaining_worst = sorted(selected_int8_rows, key=lambda row: float(row["tflite_int8_absolute_error_c_guarded"]), reverse=True)[:30]

    _write_csv(all_wide_rows, predictions_path)
    _write_csv(all_summary_rows, summary_path)
    if remaining_worst:
        _write_csv(remaining_worst, remaining_path)
    elif remaining_path.exists():
        remaining_path.unlink()

    if "test" in split_rows_by_model:
        # Overlay selection focuses on the test split because that is the export gate.
        test_selected = _select_overlay_rows(split_rows_by_model["test"], split="test")
        rng = random.Random(args.overlay_seed)
        overlay_specs = {
            "test_worst_30_int8_accepted": test_selected["test_worst_30_accepted"],
            "test_int8_gt10c": test_selected["test_accepted_gt10c"],
            "keras_accept_int8_reject": test_selected["keras_accept_int8_reject"],
            "keras_reject_int8_accept": test_selected["keras_reject_int8_accept"],
            "largest_keras_int8_temp_delta": test_selected["largest_keras_int8_temp_delta"],
        }

        # Make the overlay filenames deterministic by basing them on the underlying image path.
        for category, rows in overlay_specs.items():
            category_dir = debug_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            for row in rows:
                image_path = str(row["image_path"])
                sample_index = next(
                    index
                    for index, artifact in enumerate(split_artifacts_by_model["test"]["keras_fp32"])
                    if str(artifact.sample.metadata["image_path"]) == image_path
                )
                sample = split_artifacts_by_model["test"]["keras_fp32"][sample_index].sample
                keras_artifact = split_artifacts_by_model["test"]["keras_fp32"][sample_index]
                int8_artifact = split_artifacts_by_model["test"]["tflite_int8"][sample_index]
                output_path = category_dir / f"{Path(image_path).stem}.png"
                _make_overlay(
                    sample=sample,
                    keras_row=keras_artifact.row,
                    keras_center_heatmap=keras_artifact.center_heatmap,
                    keras_tip_heatmap=keras_artifact.tip_heatmap,
                    int8_row=int8_artifact.row,
                    int8_center_heatmap=int8_artifact.center_heatmap,
                    int8_tip_heatmap=int8_artifact.tip_heatmap,
                    output_path=output_path,
                )

    drift_summary_rows = [_summarize_split_drift(split, all_drift_rows) for split in ("train", "val", "test")]
    _write_report(
        report_path=report_path,
        summary_rows=all_summary_rows,
        drift_rows=drift_summary_rows,
        selected_decode_name=selected_decode_name,
        selected_decode_method=selected_decode_method,
        selected_decode_window_size=selected_decode_window_size,
    )

    print(f"[REPLAY] Wrote predictions to {predictions_path}", flush=True)
    print(f"[REPLAY] Wrote summary to {summary_path}", flush=True)
    print(f"[REPLAY] Wrote worst accepted rows to {remaining_path}", flush=True)


if __name__ == "__main__":
    main()
