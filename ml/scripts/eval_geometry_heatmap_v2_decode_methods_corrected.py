#!/usr/bin/env python3
"""Evaluate corrected geometry_heatmap_v2 decode methods and select one on validation.

This script uses the fixed 224-space projection path and keeps selection strictly
validation-driven.  Test is only used after a decoder has been selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import numpy as np
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_heatmap_quantization_common import (
    DEFAULT_HEATMAP_SIZE,
    DEFAULT_INPUT_SIZE,
    DEFAULT_PREPROCESSING_MODE,
    load_split_samples,
    load_semantic_output_order_indices,
    predict_tflite_outputs,
    resolve_repo_path,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import load_geometry_heatmap_keras_model, load_tflite_model
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import load_selected_calibration_candidate
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import (
    GeometryDecodedPrediction,
    GeometryGuardrailResult,
    GeometryGuardrailThresholds,
    apply_geometry_guardrails,
    decode_heatmap_geometry_prediction,
)
from embedded_gauge_reading_tinyml.gauge_geometry import circular_angle_error_degrees


SPLITS = ("train", "val", "test")
DEFAULT_MODEL_PATH = Path("ml/artifacts/training/geometry_heatmap_v2/model.keras")
DEFAULT_CALIBRATION_PATH = Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json")
DEFAULT_THRESHOLDS_PATH = Path("ml/artifacts/training/geometry_heatmap_v2_board_replay/selected_board_guardrail_thresholds.json")
DEFAULT_OUTPUT_DIR = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2")
DEFAULT_REPORT_PATH = Path("ml/reports/geometry_heatmap_v2_decode_method_comparison_corrected.md")
DEFAULT_SUMMARY_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/corrected_decode_method_summary.csv")
DEFAULT_SELECTED_DECODE_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json")
DEFAULT_SELECTED_GUARDRAILS_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_corrected_decoder_guardrails.json")
DEFAULT_GUARDRAIL_SWEEP_REPORT_PATH = Path("ml/reports/geometry_heatmap_v2_corrected_decoder_guardrail_sweep.md")
DEFAULT_TEST_REPORT_PATH = Path("ml/reports/geometry_heatmap_v2_corrected_decoder_test_eval.md")
DEFAULT_TEST_PREDICTIONS_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/corrected_decoder_test_predictions.csv")
DEFAULT_CURRENT_INT8_MODEL_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/model_int8.tflite")
DEFAULT_CURRENT_INT8_CONTRACT_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/tflite_tensor_contract.json")
DEFAULT_CURRENT_INT8_REPORT_PATH = Path("ml/reports/geometry_heatmap_v2_corrected_decoder_current_int8_eval.md")


@dataclass(frozen=True)
class DecodeSpec:
    """One decoder and its local window size."""

    name: str
    method: str
    window_size: int


@dataclass(frozen=True)
class DecodedSample:
    """A cached decoded prediction for one decoder and one sample."""

    sample: Any
    spec: DecodeSpec
    decoded: GeometryDecodedPrediction
    guarded: GeometryGuardrailResult


DECODE_SPECS: tuple[DecodeSpec, ...] = (
    DecodeSpec(name="softargmax", method="softargmax", window_size=3),
    DecodeSpec(name="argmax", method="argmax", window_size=3),
    DecodeSpec(name="local_window_softargmax_w3", method="local_window_softargmax", window_size=3),
    DecodeSpec(name="local_window_softargmax_w5", method="local_window_softargmax", window_size=5),
    DecodeSpec(name="peak_weighted_centroid_w3", method="peak_weighted_centroid", window_size=3),
    DecodeSpec(name="peak_weighted_centroid_w5", method="peak_weighted_centroid", window_size=5),
)


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write rows to CSV with deterministic field ordering."""

    if not rows:
        raise ValueError("Cannot write an empty CSV file.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write a JSON artifact with stable formatting."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _ordered_model_outputs(outputs: Any, output_names: list[str]) -> list[np.ndarray]:
    """Normalize Keras model outputs into signature order."""

    if isinstance(outputs, Mapping):
        return [np.asarray(outputs[name], dtype=np.float32) for name in output_names]
    return [np.asarray(output, dtype=np.float32) for output in list(outputs)]


def _predict_keras_outputs(
    model: keras.Model,
    inputs: list[np.ndarray],
    *,
    batch_size: int,
) -> list[np.ndarray]:
    """Predict all Keras outputs for one list of inputs."""

    batches: list[list[np.ndarray]] = []
    for start in range(0, len(inputs), batch_size):
        batch = np.stack(inputs[start : start + batch_size], axis=0).astype(np.float32)
        outputs = model(batch, training=False)
        ordered = _ordered_model_outputs(outputs, list(model.output_names))
        batches.append(ordered)
    if not batches:
        raise ValueError("No Keras batches were predicted.")
    merged: list[np.ndarray] = []
    for output_index in range(len(batches[0])):
        merged.append(np.concatenate([batch_outputs[output_index] for batch_outputs in batches], axis=0))
    return merged


def _load_thresholds(thresholds_path: Path) -> GeometryGuardrailThresholds:
    """Load the selected board guardrail thresholds."""

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


def _decode_records_for_split(
    samples: list[Any],
    keras_center: np.ndarray,
    keras_tip: np.ndarray,
    keras_confidence: np.ndarray,
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    spec: DecodeSpec,
) -> list[DecodedSample]:
    """Decode one split for a single decoder."""

    records: list[DecodedSample] = []
    for index, sample in enumerate(samples):
        decoded = decode_heatmap_geometry_prediction(
            sample,
            keras_center[index],
            keras_tip[index],
            float(np.ravel(keras_confidence[index])[0]),
            calibration_candidate,
            decode_method=spec.method,
            window_size=spec.window_size,
        )
        guarded = apply_geometry_guardrails(decoded, thresholds)
        records.append(DecodedSample(sample=sample, spec=spec, decoded=decoded, guarded=guarded))
    return records


def _status_is_accepted(status: str) -> bool:
    """Treat accepted and clamped predictions as usable."""

    return status in {"accepted", "clamped"}


def _summary_from_records(records: list[DecodedSample], thresholds: GeometryGuardrailThresholds) -> dict[str, Any]:
    """Summarize one decoder under one threshold set."""

    decisions = [apply_geometry_guardrails(record.decoded, thresholds) for record in records]
    accepted_rows = [
        (record, decision)
        for record, decision in zip(records, decisions, strict=True)
        if decision.status in {"accepted", "clamped"}
    ]
    rejected_rows = [
        (record, decision)
        for record, decision in zip(records, decisions, strict=True)
        if decision.status == "rejected"
    ]
    accepted_errors = np.asarray(
        [abs(float(decision.temperature_c) - float(record.decoded.true_temperature_c)) for record, decision in accepted_rows],
        dtype=np.float64,
    )
    if accepted_errors.size:
        accepted_mae = float(np.mean(accepted_errors))
        accepted_worst = float(np.max(accepted_errors))
        under_2 = float(np.mean(accepted_errors < 2.0) * 100.0)
        under_5 = float(np.mean(accepted_errors < 5.0) * 100.0)
        under_10 = float(np.mean(accepted_errors < 10.0) * 100.0)
        accepted_gt20 = int(np.sum(accepted_errors > 20.0))
    else:
        accepted_mae = math.nan
        accepted_worst = math.nan
        under_2 = 0.0
        under_5 = 0.0
        under_10 = 0.0
        accepted_gt20 = 0

    center_errors = np.asarray(
        [
            math.hypot(
                float(record.decoded.predicted_center_x_224) - float(record.decoded.true_center_x_224),
                float(record.decoded.predicted_center_y_224) - float(record.decoded.true_center_y_224),
            )
            for record in records
        ],
        dtype=np.float64,
    )
    tip_errors = np.asarray(
        [
            math.hypot(
                float(record.decoded.predicted_tip_x_224) - float(record.decoded.true_tip_x_224),
                float(record.decoded.predicted_tip_y_224) - float(record.decoded.true_tip_y_224),
            )
            for record in records
        ],
        dtype=np.float64,
    )
    angle_errors = np.asarray(
        [
            abs(circular_angle_error_degrees(float(record.decoded.predicted_angle_degrees), float(record.decoded.true_angle_degrees)))
            for record in records
        ],
        dtype=np.float64,
    )
    rejection_reasons = Counter(
        reason
        for _, decision in rejected_rows
        for reason in decision.rejection_reasons
    )
    return {
        "count": int(len(records)),
        "accepted_count": int(sum(1 for _, decision in accepted_rows if decision.status == "accepted")),
        "clamped_count": int(sum(1 for _, decision in accepted_rows if decision.status == "clamped")),
        "rejected_count": int(len(rejected_rows)),
        "acceptance_rate": float(len(accepted_rows) / len(records)) if records else math.nan,
        "accepted_mae_c": accepted_mae,
        "accepted_worst_error_c": accepted_worst,
        "accepted_gt20c_failures": accepted_gt20,
        "percentage_under_2c": under_2,
        "percentage_under_5c": under_5,
        "percentage_under_10c": under_10,
        "center_mae_px_224": float(np.mean(center_errors)) if center_errors.size else math.nan,
        "tip_mae_px_224": float(np.mean(tip_errors)) if tip_errors.size else math.nan,
        "angle_mae_degrees": float(np.mean(angle_errors)) if angle_errors.size else math.nan,
        "center_heatmap_peak_mean": float(np.mean([record.decoded.center_heatmap_peak_value for record in records])) if records else math.nan,
        "tip_heatmap_peak_mean": float(np.mean([record.decoded.tip_heatmap_peak_value for record in records])) if records else math.nan,
        "confidence_mean": float(np.mean([record.decoded.confidence for record in records])) if records else math.nan,
        "top_rejection_reasons": ";".join(f"{reason}:{count}" for reason, count in rejection_reasons.most_common(5)) or "none",
    }


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
    """Rank decoders by the validation gate and then by robustness."""

    return (
        float(row["accepted_worst_error_c"]),
        float(row["accepted_mae_c"]),
        -float(row["acceptance_rate"]),
    )


def _validation_passes(row: Mapping[str, Any]) -> bool:
    """Check the validation gate specified by the user."""

    return (
        float(row["accepted_mae_c"]) <= 4.5
        and float(row["acceptance_rate"]) >= 0.65
        and float(row["accepted_worst_error_c"]) < 20.0
        and int(row["accepted_gt20c_failures"]) == 0
    )


def _rows_for_decoder(split_records: dict[str, dict[str, list[DecodedSample]]], spec_name: str, split: str) -> list[DecodedSample]:
    """Return the cached decoded records for one decoder/split pair."""

    return split_records[split][spec_name]


def _build_prediction_rows(
    records: list[DecodedSample],
    thresholds: GeometryGuardrailThresholds,
    *,
    guardrail_thresholds_path: Path,
) -> list[dict[str, Any]]:
    """Flatten one set of decoded records into CSV rows."""

    rows: list[dict[str, Any]] = []
    for record in records:
        guarded = apply_geometry_guardrails(record.decoded, thresholds)
        features = guarded.quality_features
        rejection_reasons = ";".join(guarded.rejection_reasons) if guarded.rejection_reasons else "none"
        row = {
            "decode_method": record.spec.name,
            "window_size": int(record.spec.window_size),
            "image_path": str(record.decoded.image_path),
            "split": str(record.decoded.split),
            "true_temperature_c": float(record.decoded.true_temperature_c),
            "true_angle_degrees": float(record.decoded.true_angle_degrees),
            "true_center_x_224": float(record.decoded.true_center_x_224),
            "true_center_y_224": float(record.decoded.true_center_y_224),
            "true_tip_x_224": float(record.decoded.true_tip_x_224),
            "true_tip_y_224": float(record.decoded.true_tip_y_224),
            "predicted_center_x_224": float(record.decoded.predicted_center_x_224),
            "predicted_center_y_224": float(record.decoded.predicted_center_y_224),
            "predicted_tip_x_224": float(record.decoded.predicted_tip_x_224),
            "predicted_tip_y_224": float(record.decoded.predicted_tip_y_224),
            "predicted_center_x_224_argmax": float(record.decoded.predicted_center_x_224_argmax),
            "predicted_center_y_224_argmax": float(record.decoded.predicted_center_y_224_argmax),
            "predicted_tip_x_224_argmax": float(record.decoded.predicted_tip_x_224_argmax),
            "predicted_tip_y_224_argmax": float(record.decoded.predicted_tip_y_224_argmax),
            "predicted_angle_degrees": float(record.decoded.predicted_angle_degrees),
            "predicted_angle_degrees_argmax": float(record.decoded.predicted_angle_degrees_argmax),
            "predicted_temperature_c_calibrated": float(record.decoded.predicted_temperature_c_calibrated),
            "predicted_temperature_c_calibrated_argmax": float(record.decoded.predicted_temperature_c_calibrated_argmax),
            "absolute_error_c_calibrated": float(record.decoded.absolute_error_c_calibrated),
            "absolute_error_c_calibrated_argmax": float(record.decoded.absolute_error_c_calibrated_argmax),
            "center_heatmap_peak_value": float(record.decoded.center_heatmap_peak_value),
            "tip_heatmap_peak_value": float(record.decoded.tip_heatmap_peak_value),
            "center_heatmap_mean_value": float(record.decoded.center_heatmap_mean_value),
            "tip_heatmap_mean_value": float(record.decoded.tip_heatmap_mean_value),
            "center_heatmap_entropy": float(features.center_heatmap_entropy),
            "tip_heatmap_entropy": float(features.tip_heatmap_entropy),
            "center_heatmap_spread_px": float(features.center_heatmap_spread_px),
            "tip_heatmap_spread_px": float(features.tip_heatmap_spread_px),
            "confidence": float(record.decoded.confidence),
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
            "absolute_error_c_guarded": float(abs(guarded.temperature_c - record.decoded.true_temperature_c))
            if guarded.status != "rejected"
            else math.nan,
            "crop_x1": int(record.decoded.crop_x1),
            "crop_y1": int(record.decoded.crop_y1),
            "crop_x2": int(record.decoded.crop_x2),
            "crop_y2": int(record.decoded.crop_y2),
            "crop_width": int(record.decoded.crop_width),
            "crop_height": int(record.decoded.crop_height),
            "source_width": int(record.sample.metadata["source_width"]),
            "source_height": int(record.sample.metadata["source_height"]),
            "source_manifest": str(record.sample.metadata["source_manifest"]),
            "quality_flag": str(record.sample.metadata["quality_flag"]),
            "dial_radius_source": float(record.decoded.dial_radius_source),
            "resize_method": str(record.sample.metadata["resize_method"]),
            "channel_strategy": str(record.sample.metadata["channel_strategy"]),
            "normalization": str(record.sample.metadata["normalization"]),
            "scale": float(record.sample.metadata["scale"]),
            "resized_width": int(record.sample.metadata["resized_width"]),
            "resized_height": int(record.sample.metadata["resized_height"]),
            "pad_x": int(record.sample.metadata["pad_x"]),
            "pad_y": int(record.sample.metadata["pad_y"]),
            "pad_bottom": int(record.sample.metadata["pad_bottom"]),
            "pad_right": int(record.sample.metadata["pad_right"]),
            "input_size": int(record.sample.metadata["input_size"]),
            "heatmap_size": int(record.sample.metadata["heatmap_size"]),
            "sigma_pixels": float(record.sample.metadata["sigma_pixels"]),
            "jitter_shift_x": int(record.decoded.jitter_shift_x),
            "jitter_shift_y": int(record.decoded.jitter_shift_y),
            "jitter_scale": float(record.decoded.jitter_scale),
            "jitter_aspect": float(record.decoded.jitter_aspect),
            "selected_guardrail_thresholds_path": str(guardrail_thresholds_path),
        }
        rows.append(row)
    return rows


def _format_metric_table_row(summary: Mapping[str, Any]) -> str:
    """Format one summary row for markdown tables."""

    return (
        f"| {summary['decode_method']} | {int(summary['window_size'])} | {summary['split']} | "
        f"{float(summary['accepted_mae_c']):.4f} | {float(summary['acceptance_rate']):.4f} | {float(summary['accepted_worst_error_c']):.4f} | "
        f"{int(summary['accepted_gt20c_failures'])} | {float(summary['percentage_under_2c']):.1f} | {float(summary['percentage_under_5c']):.1f} | "
        f"{float(summary['percentage_under_10c']):.1f} | {float(summary['center_mae_px_224']):.2f} | {float(summary['tip_mae_px_224']):.2f} | "
        f"{float(summary['angle_mae_degrees']):.2f} | {summary['top_rejection_reasons']} |"
    )


def _build_markdown_report(
    *,
    summary_rows: list[dict[str, Any]],
    selected_row: Mapping[str, Any],
    selected_thresholds: GeometryGuardrailThresholds,
    selected_thresholds_path: Path,
    guardrails_were_reswept: bool,
    test_summary: Mapping[str, Any] | None,
    int8_summary: Mapping[str, Any] | None,
) -> list[str]:
    """Build the corrected decoder comparison report."""

    lines = [
        "# Geometry Heatmap v2 Decode Method Comparison, Corrected",
        "",
        "## Selection",
        "",
        f"- Selected decoder: `{selected_row['decode_method']}`",
        f"- Selected window size: `{int(selected_row['window_size'])}`",
        f"- Selected on split: `val`",
        f"- Guardrails re-swept? `{guardrails_were_reswept}`",
        f"- Selected guardrail thresholds path: `{selected_thresholds_path}`",
        "",
        "## Selected Thresholds",
        "",
        f"- center_peak_min: `{selected_thresholds.center_peak_min}`",
        f"- tip_peak_min: `{selected_thresholds.tip_peak_min}`",
        f"- confidence_min: `{selected_thresholds.confidence_min}`",
        f"- max_heatmap_entropy: `{selected_thresholds.max_heatmap_entropy}`",
        f"- max_heatmap_spread_px: `{selected_thresholds.max_heatmap_spread_px}`",
        f"- center_tip_distance_ratio_min: `{selected_thresholds.center_tip_distance_ratio_min}`",
        f"- center_tip_distance_ratio_max: `{selected_thresholds.center_tip_distance_ratio_max}`",
        f"- edge_margin_px: `{selected_thresholds.edge_margin_px}`",
        f"- temperature_physical_margin_c: `{selected_thresholds.temperature_physical_margin_c}`",
        "",
        "## Summary",
        "",
        "| decode method | window | split | accepted MAE | acceptance rate | worst accepted | accepted >20 C | under 2 C | under 5 C | under 10 C | center MAE | tip MAE | angle MAE | top rejection reasons |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(summary_rows, key=lambda item: (item["split"], item["decode_method"], int(item["window_size"]))):
        lines.append(_format_metric_table_row(row))

    if test_summary is not None:
        lines.extend(
            [
                "",
                "## Test Check",
                "",
                f"- Test accepted MAE: `{float(test_summary['accepted_mae_c']):.4f} C`",
                f"- Test acceptance rate: `{float(test_summary['acceptance_rate']):.4f}`",
                f"- Test worst accepted error: `{float(test_summary['accepted_worst_error_c']):.4f} C`",
                f"- Test accepted >20 C failures: `{int(test_summary['accepted_gt20c_failures'])}`",
                f"- Test under 2 C / 5 C / 10 C: `{float(test_summary['percentage_under_2c']):.1f}` / `{float(test_summary['percentage_under_5c']):.1f}` / `{float(test_summary['percentage_under_10c']):.1f}`",
                f"- Test center MAE: `{float(test_summary['center_mae_px_224']):.2f} px`",
                f"- Test tip MAE: `{float(test_summary['tip_mae_px_224']):.2f} px`",
                f"- Test angle MAE: `{float(test_summary['angle_mae_degrees']):.2f} deg`",
            ]
        )

    if int8_summary is not None:
        lines.extend(
            [
                "",
                "## Current INT8 Check",
                "",
                f"- INT8 accepted MAE: `{float(int8_summary['accepted_mae_c']):.4f} C`",
                f"- INT8 acceptance rate: `{float(int8_summary['acceptance_rate']):.4f}`",
                f"- INT8 worst accepted error: `{float(int8_summary['accepted_worst_error_c']):.4f} C`",
                f"- INT8 accepted >20 C failures: `{int(int8_summary['accepted_gt20c_failures'])}`",
                f"- Keras-vs-INT8 temperature delta mean: `{float(int8_summary['temperature_delta_mean']):.4f} C`",
                f"- Keras-vs-INT8 temperature delta median: `{float(int8_summary['temperature_delta_median']):.4f} C`",
                f"- Keras-vs-INT8 center delta mean: `{float(int8_summary['center_delta_mean']):.4f} px`",
                f"- Keras-vs-INT8 center delta median: `{float(int8_summary['center_delta_median']):.4f} px`",
                f"- Keras-vs-INT8 tip delta mean: `{float(int8_summary['tip_delta_mean']):.4f} px`",
                f"- Keras-vs-INT8 tip delta median: `{float(int8_summary['tip_delta_median']):.4f} px`",
                f"- Guardrail disagreement count: `{int(int8_summary['guardrail_disagreement_count'])}`",
            ]
        )

    return lines


def _best_decoder_row(summary_rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    """Pick the best decoder row for one split."""

    split_rows = [row for row in summary_rows if row["split"] == split]
    passing_rows = [row for row in split_rows if _validation_passes(row)]
    if passing_rows:
        return sorted(passing_rows, key=_selection_key)[0]
    return sorted(
        split_rows,
        key=lambda row: (
            float("inf") if math.isnan(float(row["acceptance_rate"])) else -float(row["acceptance_rate"]),
            float("inf") if math.isnan(float(row["accepted_mae_c"])) else float(row["accepted_mae_c"]),
            float("inf") if math.isnan(float(row["accepted_worst_error_c"])) else float(row["accepted_worst_error_c"]),
        ),
    )[0]


def _build_guardrail_candidate_grid(base_thresholds: GeometryGuardrailThresholds) -> list[GeometryGuardrailThresholds]:
    """Build the small guardrail sweep grid requested by the user."""

    candidates: list[GeometryGuardrailThresholds] = []
    for tip_peak_min in (0.30, 0.35, 0.40):
        for max_heatmap_spread_px in (25.0, 30.0, 35.0):
            for center_tip_distance_ratio_min in (0.35, 0.40):
                for center_tip_distance_ratio_max in (1.40, 1.50):
                    for confidence_min in (0.35, 0.40):
                        candidates.append(
                            GeometryGuardrailThresholds(
                                center_peak_min=base_thresholds.center_peak_min,
                                tip_peak_min=tip_peak_min,
                                confidence_min=confidence_min,
                                max_heatmap_entropy=base_thresholds.max_heatmap_entropy,
                                max_heatmap_spread_px=max_heatmap_spread_px,
                                center_tip_distance_ratio_min=center_tip_distance_ratio_min,
                                center_tip_distance_ratio_max=center_tip_distance_ratio_max,
                                edge_margin_px=base_thresholds.edge_margin_px,
                                temperature_physical_margin_c=2.0,
                                clamp_temperature_to_physical_range=True,
                                minimum_celsius=base_thresholds.minimum_celsius,
                                maximum_celsius=base_thresholds.maximum_celsius,
                                cold_angle_degrees=base_thresholds.cold_angle_degrees,
                                sweep_degrees=base_thresholds.sweep_degrees,
                            )
                        )
    return candidates


def _sweep_key(summary: Mapping[str, Any], thresholds: GeometryGuardrailThresholds) -> tuple[float, float, float, float]:
    """Rank sweep candidates using the user-provided selection criteria."""

    return (
        0.0 if _validation_passes(summary) else 1.0,
        float(summary["accepted_worst_error_c"]),
        float(summary["accepted_mae_c"]),
        -float(summary["acceptance_rate"]),
    )


def main() -> None:
    """Evaluate corrected decoders and select one using validation only."""

    parser = argparse.ArgumentParser(description="Corrected geometry_heatmap_v2 decode method comparison")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest-path", type=Path, default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"))
    parser.add_argument("--calibration-json-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--selected-guardrails-path", type=Path, default=DEFAULT_SELECTED_GUARDRAILS_PATH)
    parser.add_argument("--guardrail-sweep-report-path", type=Path, default=DEFAULT_GUARDRAIL_SWEEP_REPORT_PATH)
    parser.add_argument("--test-report-path", type=Path, default=DEFAULT_TEST_REPORT_PATH)
    parser.add_argument("--test-predictions-path", type=Path, default=DEFAULT_TEST_PREDICTIONS_PATH)
    parser.add_argument("--current-int8-model-path", type=Path, default=DEFAULT_CURRENT_INT8_MODEL_PATH)
    parser.add_argument("--current-int8-contract-path", type=Path, default=DEFAULT_CURRENT_INT8_CONTRACT_PATH)
    parser.add_argument("--current-int8-report-path", type=Path, default=DEFAULT_CURRENT_INT8_REPORT_PATH)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--heatmap-size", type=int, default=DEFAULT_HEATMAP_SIZE)
    parser.add_argument("--sigma-pixels", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    model_path = resolve_repo_path(repo_root, args.model_path)
    manifest_path = resolve_repo_path(repo_root, args.manifest_path)
    calibration_json_path = resolve_repo_path(repo_root, args.calibration_json_path)
    thresholds_path = resolve_repo_path(repo_root, args.thresholds_path)
    output_dir = resolve_repo_path(repo_root, args.output_dir)
    report_path = resolve_repo_path(repo_root, args.report_path)
    summary_path = resolve_repo_path(repo_root, args.summary_path)
    selected_decode_path = resolve_repo_path(repo_root, DEFAULT_SELECTED_DECODE_PATH)
    selected_guardrails_path = resolve_repo_path(repo_root, args.selected_guardrails_path)
    guardrail_sweep_report_path = resolve_repo_path(repo_root, args.guardrail_sweep_report_path)
    test_report_path = resolve_repo_path(repo_root, args.test_report_path)
    test_predictions_path = resolve_repo_path(repo_root, args.test_predictions_path)
    current_int8_model_path = resolve_repo_path(repo_root, args.current_int8_model_path)
    current_int8_contract_path = resolve_repo_path(repo_root, args.current_int8_contract_path)
    current_int8_report_path = resolve_repo_path(repo_root, args.current_int8_report_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_candidate, _ = load_selected_calibration_candidate(calibration_json_path)
    selected_thresholds = _load_thresholds(thresholds_path)

    keras_model = load_geometry_heatmap_keras_model(model_path)

    split_records: dict[str, dict[str, list[DecodedSample]]] = {split: {} for split in SPLITS}
    summary_rows: list[dict[str, Any]] = []

    for split in SPLITS:
        samples = load_split_samples(
            manifest_path,
            repo_root,
            split=split,
            mode=DEFAULT_PREPROCESSING_MODE,
            input_size=args.input_size,
            heatmap_size=args.heatmap_size,
            sigma_pixels=args.sigma_pixels,
        ).samples
        inputs = [np.asarray(sample.crop_image, dtype=np.float32) for sample in samples]
        keras_center, keras_tip, keras_confidence = _predict_keras_outputs(keras_model, inputs, batch_size=args.batch_size)
        split_records[split] = {}
        for spec in DECODE_SPECS:
            records = _decode_records_for_split(
                samples,
                keras_center,
                keras_tip,
                keras_confidence,
                calibration_candidate,
                selected_thresholds,
                spec,
            )
            split_records[split][spec.name] = records
            summary = _summary_from_records(records, selected_thresholds)
            summary.update(
                {
                    "decode_method": spec.name,
                    "window_size": int(spec.window_size),
                    "split": split,
                }
            )
            summary_rows.append(summary)

    _write_csv(summary_rows, summary_path)

    val_rows = [row for row in summary_rows if row["split"] == "val"]
    passing_val_rows = [row for row in val_rows if _validation_passes(row)]
    guardrails_were_reswept = False
    final_thresholds = selected_thresholds
    final_thresholds_path = thresholds_path
    selection_source = "selected_board_guardrail_thresholds.json"

    if passing_val_rows:
        selected_row = sorted(passing_val_rows, key=_selection_key)[0]
    else:
        best_row = _best_decoder_row(summary_rows, "val")
        best_spec = next(spec for spec in DECODE_SPECS if spec.name == str(best_row["decode_method"]))
        guardrails_were_reswept = True
        selection_source = "re-swept"
        sweep_candidates = _build_guardrail_candidate_grid(selected_thresholds)
        sweep_rows: list[dict[str, Any]] = []
        selected_decoder_records = split_records["val"][best_spec.name]
        for candidate in sweep_candidates:
            candidate_summary = _summary_from_records(selected_decoder_records, candidate)
            candidate_summary.update(
                {
                    "decode_method": best_spec.name,
                    "window_size": int(best_spec.window_size),
                    "split": "val",
                    "guardrail_thresholds": {
                        "center_peak_min": candidate.center_peak_min,
                        "tip_peak_min": candidate.tip_peak_min,
                        "confidence_min": candidate.confidence_min,
                        "max_heatmap_entropy": candidate.max_heatmap_entropy,
                        "max_heatmap_spread_px": candidate.max_heatmap_spread_px,
                        "center_tip_distance_ratio_min": candidate.center_tip_distance_ratio_min,
                        "center_tip_distance_ratio_max": candidate.center_tip_distance_ratio_max,
                        "edge_margin_px": candidate.edge_margin_px,
                        "temperature_physical_margin_c": candidate.temperature_physical_margin_c,
                    },
                }
            )
            sweep_rows.append(candidate_summary)
        sweep_rows_sorted = sorted(sweep_rows, key=lambda row: _sweep_key(row, final_thresholds))
        selected_candidate = sweep_rows_sorted[0]
        final_thresholds = GeometryGuardrailThresholds(
            center_peak_min=float(selected_candidate["guardrail_thresholds"]["center_peak_min"]),
            tip_peak_min=float(selected_candidate["guardrail_thresholds"]["tip_peak_min"]),
            confidence_min=float(selected_candidate["guardrail_thresholds"]["confidence_min"]),
            max_heatmap_entropy=float(selected_candidate["guardrail_thresholds"]["max_heatmap_entropy"]),
            max_heatmap_spread_px=float(selected_candidate["guardrail_thresholds"]["max_heatmap_spread_px"]),
            center_tip_distance_ratio_min=float(selected_candidate["guardrail_thresholds"]["center_tip_distance_ratio_min"]),
            center_tip_distance_ratio_max=float(selected_candidate["guardrail_thresholds"]["center_tip_distance_ratio_max"]),
            edge_margin_px=float(selected_candidate["guardrail_thresholds"]["edge_margin_px"]),
            temperature_physical_margin_c=float(selected_candidate["guardrail_thresholds"]["temperature_physical_margin_c"]),
            minimum_celsius=selected_thresholds.minimum_celsius,
            maximum_celsius=selected_thresholds.maximum_celsius,
            cold_angle_degrees=selected_thresholds.cold_angle_degrees,
            sweep_degrees=selected_thresholds.sweep_degrees,
            clamp_temperature_to_physical_range=selected_thresholds.clamp_temperature_to_physical_range,
        )
        final_thresholds_path = selected_guardrails_path
        selected_row = {
            "decode_method": best_spec.name,
            "window_size": int(best_spec.window_size),
            "split": "val",
            **{
                key: selected_candidate[key]
                for key in (
                    "count",
                    "accepted_count",
                    "clamped_count",
                    "rejected_count",
                    "acceptance_rate",
                    "accepted_mae_c",
                    "accepted_worst_error_c",
                    "accepted_gt20c_failures",
                    "percentage_under_2c",
                    "percentage_under_5c",
                    "percentage_under_10c",
                    "center_mae_px_224",
                    "tip_mae_px_224",
                    "angle_mae_degrees",
                    "top_rejection_reasons",
                    "center_heatmap_peak_mean",
                    "tip_heatmap_peak_mean",
                    "confidence_mean",
                )
            },
            "guardrail_thresholds_path": str(selected_guardrails_path),
        }
        _write_csv(sweep_rows_sorted, guardrail_sweep_report_path.with_suffix(".csv"))
        guardrail_sweep_lines = [
            "# Geometry Heatmap v2 Corrected Decoder Guardrail Sweep",
            "",
            f"- Best decoder under sweep: `{best_spec.name}`",
            f"- Best window size under sweep: `{best_spec.window_size}`",
            f"- Selected on split: `val`",
            f"- Guardrails were re-swept: `True`",
            "",
            "## Top Sweep Results",
            "",
            "| decode method | window | tip peak min | spread max | ratio min | ratio max | confidence min | accepted MAE | acceptance rate | worst accepted | accepted >20 C |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for candidate in sweep_rows_sorted[:10]:
            thresholds_dict = candidate["guardrail_thresholds"]
            guardrail_sweep_lines.append(
                f"| {candidate['decode_method']} | {int(candidate['window_size'])} | {thresholds_dict['tip_peak_min']:.2f} | {thresholds_dict['max_heatmap_spread_px']:.1f} | "
                f"{thresholds_dict['center_tip_distance_ratio_min']:.2f} | {thresholds_dict['center_tip_distance_ratio_max']:.2f} | {thresholds_dict['confidence_min']:.2f} | "
                f"{float(candidate['accepted_mae_c']):.4f} | {float(candidate['acceptance_rate']):.4f} | {float(candidate['accepted_worst_error_c']):.4f} | {int(candidate['accepted_gt20c_failures'])} |"
            )
        guardrail_sweep_lines.extend(
            [
                "",
                f"- Selected thresholds path: `{selected_guardrails_path}`",
            ]
        )
        guardrail_sweep_report_path.parent.mkdir(parents=True, exist_ok=True)
        guardrail_sweep_report_path.write_text("\n".join(guardrail_sweep_lines), encoding="utf-8")
        _write_json(
            {
                "selected_decode_method": best_spec.name,
                "selected_window_size": int(best_spec.window_size),
                "selected_on_split": "val",
                "guardrails_were_reswept": True,
                "guardrail_thresholds_path": str(selected_guardrails_path),
                "guardrail_thresholds": {
                    "center_peak_min": final_thresholds.center_peak_min,
                    "tip_peak_min": final_thresholds.tip_peak_min,
                    "confidence_min": final_thresholds.confidence_min,
                    "max_heatmap_entropy": final_thresholds.max_heatmap_entropy,
                    "max_heatmap_spread_px": final_thresholds.max_heatmap_spread_px,
                    "center_tip_distance_ratio_min": final_thresholds.center_tip_distance_ratio_min,
                    "center_tip_distance_ratio_max": final_thresholds.center_tip_distance_ratio_max,
                    "edge_margin_px": final_thresholds.edge_margin_px,
                    "temperature_physical_margin_c": final_thresholds.temperature_physical_margin_c,
                },
                "selection_metrics": selected_candidate,
                "selection_source": selection_source,
            },
            selected_guardrails_path,
        )

    _write_json(
        {
            "decode_method": str(selected_row["decode_method"]),
            "window_size": int(selected_row["window_size"]),
            "selected_on_split": "val",
            "guardrails_were_reswept": guardrails_were_reswept,
            "guardrail_thresholds_path": str(final_thresholds_path),
            "val_metrics": {
                key: selected_row[key]
                for key in (
                    "count",
                    "accepted_count",
                    "clamped_count",
                    "rejected_count",
                    "acceptance_rate",
                    "accepted_mae_c",
                    "accepted_worst_error_c",
                    "accepted_gt20c_failures",
                    "percentage_under_2c",
                    "percentage_under_5c",
                    "percentage_under_10c",
                    "center_mae_px_224",
                    "tip_mae_px_224",
                    "angle_mae_degrees",
                    "top_rejection_reasons",
                    "center_heatmap_peak_mean",
                    "tip_heatmap_peak_mean",
                    "confidence_mean",
                )
            },
        },
        selected_decode_path,
    )

    selected_spec = next(spec for spec in DECODE_SPECS if spec.name == str(selected_row["decode_method"]))
    test_records = _rows_for_decoder(split_records, selected_spec.name, "test")
    test_summary = _summary_from_records(test_records, final_thresholds)
    test_predictions = _build_prediction_rows(
        test_records,
        final_thresholds,
        guardrail_thresholds_path=final_thresholds_path,
    )
    _write_csv(test_predictions, test_predictions_path)

    comparison_report_lines = _build_markdown_report(
        summary_rows=summary_rows,
        selected_row=selected_row,
        selected_thresholds=final_thresholds,
        selected_thresholds_path=final_thresholds_path,
        guardrails_were_reswept=guardrails_were_reswept,
        test_summary=test_summary,
        int8_summary=None,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(comparison_report_lines), encoding="utf-8")
    test_report_path.parent.mkdir(parents=True, exist_ok=True)
    test_report_path.write_text("\n".join(
        _build_markdown_report(
            summary_rows=summary_rows,
            selected_row=selected_row,
            selected_thresholds=final_thresholds,
            selected_thresholds_path=final_thresholds_path,
            guardrails_were_reswept=guardrails_were_reswept,
            test_summary=test_summary,
            int8_summary=None,
        )
    ), encoding="utf-8")

    keras_test_pass = _validation_passes(test_summary)
    print(
        f"[CORRECTED] Selected decoder {selected_spec.name} w{selected_spec.window_size} | "
        f"val acceptance {float(selected_row['acceptance_rate']):.4f} | test acceptance {float(test_summary['acceptance_rate']):.4f}",
        flush=True,
    )

    int8_summary: dict[str, Any] | None = None
    if keras_test_pass:
        current_int8_bundle = load_tflite_model(current_int8_model_path)
        semantic_order = load_semantic_output_order_indices(current_int8_contract_path)
        test_samples = [record.sample for record in test_records]
        test_inputs = [np.asarray(sample.crop_image, dtype=np.float32) for sample in test_samples]
        int8_outputs = predict_tflite_outputs(
            current_int8_bundle,
            test_inputs,
            semantic_output_order_indices=semantic_order,
        )
        int8_records: list[DecodedSample] = []
        for index, sample in enumerate(test_samples):
            decoded = decode_heatmap_geometry_prediction(
                sample,
                int8_outputs[0][index],
                int8_outputs[1][index],
                float(np.ravel(int8_outputs[2][index])[0]),
                calibration_candidate,
                decode_method=selected_spec.method,
                window_size=selected_spec.window_size,
            )
            guarded = apply_geometry_guardrails(decoded, final_thresholds)
            int8_records.append(DecodedSample(sample=sample, spec=selected_spec, decoded=decoded, guarded=guarded))
        int8_test_summary = _summary_from_records(int8_records, final_thresholds)
        temp_deltas = [
            abs(
                float(test_records[index].guarded.temperature_c) - float(int8_records[index].guarded.temperature_c)
            )
            for index in range(len(test_records))
            if test_records[index].guarded.status != "rejected" and int8_records[index].guarded.status != "rejected"
        ]
        center_deltas = [
            math.hypot(
                float(test_records[index].decoded.predicted_center_x_224) - float(int8_records[index].decoded.predicted_center_x_224),
                float(test_records[index].decoded.predicted_center_y_224) - float(int8_records[index].decoded.predicted_center_y_224),
            )
            for index in range(len(test_records))
        ]
        tip_deltas = [
            math.hypot(
                float(test_records[index].decoded.predicted_tip_x_224) - float(int8_records[index].decoded.predicted_tip_x_224),
                float(test_records[index].decoded.predicted_tip_y_224) - float(int8_records[index].decoded.predicted_tip_y_224),
            )
            for index in range(len(test_records))
        ]
        int8_summary = {
            **int8_test_summary,
            "temperature_delta_mean": float(np.mean(temp_deltas)) if temp_deltas else math.nan,
            "temperature_delta_median": float(np.median(temp_deltas)) if temp_deltas else math.nan,
            "center_delta_mean": float(np.mean(center_deltas)) if center_deltas else math.nan,
            "center_delta_median": float(np.median(center_deltas)) if center_deltas else math.nan,
            "tip_delta_mean": float(np.mean(tip_deltas)) if tip_deltas else math.nan,
            "tip_delta_median": float(np.median(tip_deltas)) if tip_deltas else math.nan,
            "guardrail_disagreement_count": int(
                sum(
                    1
                    for index in range(len(test_records))
                    if test_records[index].guarded.status != int8_records[index].guarded.status
                )
            ),
        }
        int8_report_lines = _build_markdown_report(
            summary_rows=summary_rows,
            selected_row=selected_row,
            selected_thresholds=final_thresholds,
            selected_thresholds_path=final_thresholds_path,
            guardrails_were_reswept=guardrails_were_reswept,
            test_summary=test_summary,
            int8_summary=int8_summary,
        )
        current_int8_report_path.parent.mkdir(parents=True, exist_ok=True)
        current_int8_report_path.write_text("\n".join(int8_report_lines), encoding="utf-8")

    print(f"[CORRECTED] Wrote summary to {summary_path}", flush=True)
    print(f"[CORRECTED] Wrote selection JSON to {selected_decode_path}", flush=True)
    print(f"[CORRECTED] Wrote test predictions to {test_predictions_path}", flush=True)
    print(f"[CORRECTED] Wrote test report to {test_report_path}", flush=True)
    if guardrails_were_reswept:
        print(f"[CORRECTED] Wrote guardrail sweep report to {guardrail_sweep_report_path}", flush=True)
        print(f"[CORRECTED] Wrote guardrail selection JSON to {selected_guardrails_path}", flush=True)
    if int8_summary is not None:
        print(f"[CORRECTED] Wrote current INT8 report to {current_int8_report_path}", flush=True)


if __name__ == "__main__":
    main()
