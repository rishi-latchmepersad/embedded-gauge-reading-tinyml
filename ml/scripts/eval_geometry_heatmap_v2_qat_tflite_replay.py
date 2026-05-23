#!/usr/bin/env python3
"""Evaluate the QAT-fine-tuned geometry_heatmap_v2 models on board replay data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_heatmap_quantization_common import (
    DEFAULT_HEATMAP_SIZE,
    DEFAULT_INPUT_SIZE,
    DEFAULT_PREPROCESSING_MODE,
    load_split_samples,
    predict_tflite_outputs,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import load_geometry_heatmap_keras_model, load_tflite_model
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import load_selected_calibration_candidate, write_prediction_overlay
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import GeometryGuardrailThresholds, apply_geometry_guardrails, decode_heatmap_geometry_prediction
from embedded_gauge_reading_tinyml.gauge_geometry import circular_angle_error_degrees


SPLITS = ("train", "val", "test")
DEFAULT_BASE_MODEL_PATH = Path("ml/artifacts/training/geometry_heatmap_v2/model.keras")
DEFAULT_QAT_MODEL_PATH = Path("ml/artifacts/training/geometry_heatmap_v2_qat/model_qat.keras")
DEFAULT_QAT_FLOAT32_TFLITE_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_qat_tflite/model_qat_float32.tflite")
DEFAULT_QAT_INT8_TFLITE_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_qat_tflite/model_qat_int8.tflite")
DEFAULT_MANIFEST_PATH = Path("ml/data/geometry_reader_manifest_v2_clean.csv")
DEFAULT_CALIBRATION_PATH = Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json")
DEFAULT_THRESHOLDS_PATH = Path("ml/artifacts/training/geometry_heatmap_v2_board_replay/selected_board_guardrail_thresholds.json")
DEFAULT_DECODER_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json")
DEFAULT_CURRENT_INT8_PREDICTIONS_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/tflite_replay_predictions_corrected_decoder_test.csv")
DEFAULT_DYNAMIC_RANGE_PREDICTIONS_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/corrected_decoder_selected_variant_test_predictions.csv")
DEFAULT_SUMMARY_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_qat_tflite/qat_tflite_replay_summary.csv")
DEFAULT_PREDICTIONS_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_qat_tflite/qat_tflite_replay_predictions.csv")
DEFAULT_REMAINING_WORST_ACCEPTED_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_qat_tflite/qat_remaining_worst_accepted.csv")
DEFAULT_REPORT_PATH = Path("ml/reports/geometry_heatmap_v2_qat_tflite_replay.md")
DEFAULT_DEBUG_DIR = Path("ml/debug/geometry_heatmap_v2_qat_tflite_replay")


@dataclass(frozen=True)
class ReplayRecord:
    """One decoded model prediction."""

    model_name: str
    split: str
    image_path: str
    true_temperature_c: float
    true_angle_degrees: float
    predicted_center_x_224: float
    predicted_center_y_224: float
    predicted_tip_x_224: float
    predicted_tip_y_224: float
    predicted_angle_degrees: float
    predicted_temperature_c_calibrated: float
    absolute_error_c_calibrated: float
    guardrail_status: str
    guarded_temperature_c: float
    rejection_reasons: str
    center_px_mae_224: float
    tip_px_mae_224: float
    angle_mae_degrees: float
    center_heatmap_peak_value: float
    tip_heatmap_peak_value: float
    center_heatmap_entropy: float
    tip_heatmap_entropy: float
    center_heatmap_spread_px: float
    tip_heatmap_spread_px: float
    confidence: float
    pred_center_heatmap_array: np.ndarray | None = None
    pred_tip_heatmap_array: np.ndarray | None = None


def _resolve_path(repo_root: Path, path: Path) -> Path:
    """Resolve a repository-relative path."""

    return path if path.is_absolute() else repo_root / path


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a deterministic CSV file."""

    if not rows:
        raise ValueError("Cannot write an empty CSV file.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skip_keys = {"pred_center_heatmap_array", "pred_tip_heatmap_array"}
    fieldnames = [key for key in rows[0].keys() if key not in skip_keys]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def _write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write a JSON file with stable formatting."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
        clamp_temperature_to_physical_range=bool(selected["clamp_temperature_to_physical_range"]),
        minimum_celsius=float(selected["minimum_celsius"]),
        maximum_celsius=float(selected["maximum_celsius"]),
    )


def _status_is_accepted(status: str) -> bool:
    """Treat accepted and clamped predictions as usable."""

    return status in {"accepted", "clamped"}


def _load_prediction_rows(
    predictions_path: Path,
    *,
    split: str,
    decode_method: str,
    window_size: int,
) -> dict[str, dict[str, Any]]:
    """Load a cached prediction table keyed by image path."""

    if not predictions_path.exists():
        return {}
    rows_by_image: dict[str, dict[str, Any]] = {}
    with predictions_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("split", "")) != split:
                continue
            row_decode_method = str(row.get("decode_method", row.get("selected_decode_method", "")))
            row_window_size = row.get("decode_window_size", row.get("selected_decode_window_size", ""))
            if row_decode_method and row_decode_method != decode_method:
                continue
            if row_window_size and int(float(row_window_size)) != int(window_size):
                continue
            rows_by_image[str(row["image_path"])] = row
    return rows_by_image


def _parse_metric_from_report(report_path: Path, label: str) -> float | None:
    """Extract one metric value from a Markdown report if possible."""

    if not report_path.exists():
        return None
    pattern = re.compile(rf"{re.escape(label)}: `([0-9.]+)")
    text = report_path.read_text(encoding="utf-8")
    match = pattern.search(text)
    if match is None:
        return None
    return float(match.group(1))


def _normalize_outputs(outputs: Any) -> dict[str, np.ndarray]:
    """Normalize Keras or TFLite outputs into the semantic model order."""

    if isinstance(outputs, dict):
        return {
            "center_heatmap": np.asarray(outputs["center_heatmap"], dtype=np.float32),
            "tip_heatmap": np.asarray(outputs["tip_heatmap"], dtype=np.float32),
            "confidence": np.asarray(outputs["confidence"], dtype=np.float32),
        }
    center_heatmap, tip_heatmap, confidence = outputs
    return {
        "center_heatmap": np.asarray(center_heatmap, dtype=np.float32),
        "tip_heatmap": np.asarray(tip_heatmap, dtype=np.float32),
        "confidence": np.asarray(confidence, dtype=np.float32),
    }


def _geometry_error_metrics(decoded: Any) -> tuple[float, float, float]:
    """Compute center, tip, and angle errors for one decoded prediction."""

    center_px_mae_224 = float(
        math.hypot(
            float(decoded.predicted_center_x_224) - float(decoded.true_center_x_224),
            float(decoded.predicted_center_y_224) - float(decoded.true_center_y_224),
        )
    )
    tip_px_mae_224 = float(
        math.hypot(
            float(decoded.predicted_tip_x_224) - float(decoded.true_tip_x_224),
            float(decoded.predicted_tip_y_224) - float(decoded.true_tip_y_224),
        )
    )
    angle_mae_degrees = float(abs(circular_angle_error_degrees(decoded.predicted_angle_degrees, decoded.true_angle_degrees)))
    return center_px_mae_224, tip_px_mae_224, angle_mae_degrees


def _evaluate_keras_model(
    model: Any,
    samples: list[Any],
    *,
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    decode_method: str,
    window_size: int,
    model_name: str,
    max_samples: int | None = None,
) -> list[ReplayRecord]:
    """Evaluate one Keras model on a split."""

    chosen_samples = samples if max_samples is None else samples[:max_samples]
    x = np.stack([sample.crop_image for sample in chosen_samples], axis=0).astype(np.float32)
    outputs = _normalize_outputs(model.predict(x, verbose=0))
    records: list[ReplayRecord] = []
    for index, sample in enumerate(chosen_samples):
        confidence = float(np.ravel(outputs["confidence"][index])[0])
        decoded = decode_heatmap_geometry_prediction(
            sample,
            outputs["center_heatmap"][index],
            outputs["tip_heatmap"][index],
            confidence,
            calibration_candidate,
            decode_method=decode_method,
            window_size=window_size,
        )
        guarded = apply_geometry_guardrails(decoded, thresholds)
        center_px_mae_224, tip_px_mae_224, angle_mae_degrees = _geometry_error_metrics(decoded)
        records.append(
            ReplayRecord(
                model_name=model_name,
                split=str(sample.metadata["split"]),
                image_path=str(sample.metadata["image_path"]),
                true_temperature_c=float(sample.metadata["temperature_c"]),
                true_angle_degrees=float(sample.metadata["angle_degrees"]),
                predicted_center_x_224=float(decoded.predicted_center_x_224),
                predicted_center_y_224=float(decoded.predicted_center_y_224),
                predicted_tip_x_224=float(decoded.predicted_tip_x_224),
                predicted_tip_y_224=float(decoded.predicted_tip_y_224),
                predicted_angle_degrees=float(decoded.predicted_angle_degrees),
                predicted_temperature_c_calibrated=float(decoded.predicted_temperature_c_calibrated),
                absolute_error_c_calibrated=float(abs(decoded.predicted_temperature_c_calibrated - float(sample.metadata["temperature_c"]))),
                guardrail_status=str(guarded.status),
                guarded_temperature_c=float(guarded.temperature_c),
                rejection_reasons=";".join(guarded.rejection_reasons) if guarded.rejection_reasons else "none",
                center_px_mae_224=center_px_mae_224,
                tip_px_mae_224=tip_px_mae_224,
                angle_mae_degrees=angle_mae_degrees,
                center_heatmap_peak_value=float(decoded.center_heatmap_peak_value),
                tip_heatmap_peak_value=float(decoded.tip_heatmap_peak_value),
                center_heatmap_entropy=float(guarded.quality_features.center_heatmap_entropy),
                tip_heatmap_entropy=float(guarded.quality_features.tip_heatmap_entropy),
                center_heatmap_spread_px=float(guarded.quality_features.center_heatmap_spread_px),
                tip_heatmap_spread_px=float(guarded.quality_features.tip_heatmap_spread_px),
                confidence=float(confidence),
                pred_center_heatmap_array=np.asarray(outputs["center_heatmap"][index], dtype=np.float32),
                pred_tip_heatmap_array=np.asarray(outputs["tip_heatmap"][index], dtype=np.float32),
            )
        )
    return records


def _evaluate_tflite_model(
    bundle: Any,
    samples: list[Any],
    *,
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    decode_method: str,
    window_size: int,
    model_name: str,
    semantic_output_order_indices: list[int],
    max_samples: int | None = None,
) -> list[ReplayRecord]:
    """Evaluate one TFLite model on a split."""

    chosen_samples = samples if max_samples is None else samples[:max_samples]
    inputs = [np.asarray(sample.crop_image, dtype=np.float32) for sample in chosen_samples]
    outputs = predict_tflite_outputs(bundle, inputs, semantic_output_order_indices=semantic_output_order_indices)
    records: list[ReplayRecord] = []
    for index, sample in enumerate(chosen_samples):
        confidence = float(np.ravel(outputs[2][index])[0])
        decoded = decode_heatmap_geometry_prediction(
            sample,
            outputs[0][index],
            outputs[1][index],
            confidence,
            calibration_candidate,
            decode_method=decode_method,
            window_size=window_size,
        )
        guarded = apply_geometry_guardrails(decoded, thresholds)
        center_px_mae_224, tip_px_mae_224, angle_mae_degrees = _geometry_error_metrics(decoded)
        records.append(
            ReplayRecord(
                model_name=model_name,
                split=str(sample.metadata["split"]),
                image_path=str(sample.metadata["image_path"]),
                true_temperature_c=float(sample.metadata["temperature_c"]),
                true_angle_degrees=float(sample.metadata["angle_degrees"]),
                predicted_center_x_224=float(decoded.predicted_center_x_224),
                predicted_center_y_224=float(decoded.predicted_center_y_224),
                predicted_tip_x_224=float(decoded.predicted_tip_x_224),
                predicted_tip_y_224=float(decoded.predicted_tip_y_224),
                predicted_angle_degrees=float(decoded.predicted_angle_degrees),
                predicted_temperature_c_calibrated=float(decoded.predicted_temperature_c_calibrated),
                absolute_error_c_calibrated=float(abs(decoded.predicted_temperature_c_calibrated - float(sample.metadata["temperature_c"]))),
                guardrail_status=str(guarded.status),
                guarded_temperature_c=float(guarded.temperature_c),
                rejection_reasons=";".join(guarded.rejection_reasons) if guarded.rejection_reasons else "none",
                center_px_mae_224=center_px_mae_224,
                tip_px_mae_224=tip_px_mae_224,
                angle_mae_degrees=angle_mae_degrees,
                center_heatmap_peak_value=float(decoded.center_heatmap_peak_value),
                tip_heatmap_peak_value=float(decoded.tip_heatmap_peak_value),
                center_heatmap_entropy=float(guarded.quality_features.center_heatmap_entropy),
                tip_heatmap_entropy=float(guarded.quality_features.tip_heatmap_entropy),
                center_heatmap_spread_px=float(guarded.quality_features.center_heatmap_spread_px),
                tip_heatmap_spread_px=float(guarded.quality_features.tip_heatmap_spread_px),
                confidence=float(confidence),
                pred_center_heatmap_array=np.asarray(outputs[0][index], dtype=np.float32),
                pred_tip_heatmap_array=np.asarray(outputs[1][index], dtype=np.float32),
            )
        )
    return records


def _records_to_rows(records: list[ReplayRecord]) -> list[dict[str, Any]]:
    """Convert replay records into CSV rows."""

    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "model_name": record.model_name,
                "split": record.split,
                "image_path": record.image_path,
                "true_temperature_c": record.true_temperature_c,
                "true_angle_degrees": record.true_angle_degrees,
                "predicted_center_x_224": record.predicted_center_x_224,
                "predicted_center_y_224": record.predicted_center_y_224,
                "predicted_tip_x_224": record.predicted_tip_x_224,
                "predicted_tip_y_224": record.predicted_tip_y_224,
                "predicted_angle_degrees": record.predicted_angle_degrees,
                "predicted_temperature_c_calibrated": record.predicted_temperature_c_calibrated,
                "absolute_error_c_calibrated": record.absolute_error_c_calibrated,
                "guardrail_status": record.guardrail_status,
                "guarded_temperature_c": record.guarded_temperature_c,
                "rejection_reasons": record.rejection_reasons,
                "center_px_mae_224": record.center_px_mae_224,
                "tip_px_mae_224": record.tip_px_mae_224,
                "angle_mae_degrees": record.angle_mae_degrees,
                "center_heatmap_peak_value": record.center_heatmap_peak_value,
                "tip_heatmap_peak_value": record.tip_heatmap_peak_value,
                "center_heatmap_entropy": record.center_heatmap_entropy,
                "tip_heatmap_entropy": record.tip_heatmap_entropy,
                "center_heatmap_spread_px": record.center_heatmap_spread_px,
                "tip_heatmap_spread_px": record.tip_heatmap_spread_px,
                "confidence": record.confidence,
                "pred_center_heatmap_array": record.pred_center_heatmap_array,
                "pred_tip_heatmap_array": record.pred_tip_heatmap_array,
            }
        )
    return rows


def _summary_from_records(records: list[ReplayRecord]) -> dict[str, float]:
    """Summarize one model's replay records."""

    accepted = [record for record in records if _status_is_accepted(record.guardrail_status)]
    accepted_errors = np.asarray([abs(record.guarded_temperature_c - record.true_temperature_c) for record in accepted], dtype=np.float64)
    all_errors = np.asarray([abs(record.guarded_temperature_c - record.true_temperature_c) for record in accepted], dtype=np.float64)
    return {
        "count": float(len(records)),
        "accepted_count": float(len(accepted)),
        "accepted_mae_c": float(np.mean(accepted_errors)) if accepted_errors.size else math.nan,
        "acceptance_rate": float(len(accepted) / len(records)) if records else math.nan,
        "worst_accepted_error_c": float(np.max(accepted_errors)) if accepted_errors.size else math.nan,
        "accepted_gt20_failures": float(sum(1 for record in accepted if abs(record.guarded_temperature_c - record.true_temperature_c) > 20.0)),
        "percentage_under_2c": float(np.mean(all_errors < 2.0) * 100.0) if all_errors.size else math.nan,
        "percentage_under_5c": float(np.mean(all_errors < 5.0) * 100.0) if all_errors.size else math.nan,
        "percentage_under_10c": float(np.mean(all_errors < 10.0) * 100.0) if all_errors.size else math.nan,
        "center_mae_px_224": float(np.mean([record.center_px_mae_224 for record in records])) if records else math.nan,
        "tip_mae_px_224": float(np.mean([record.tip_px_mae_224 for record in records])) if records else math.nan,
        "angle_mae_degrees": float(np.mean([record.angle_mae_degrees for record in records])) if records else math.nan,
        "center_heatmap_peak_mean": float(np.mean([record.center_heatmap_peak_value for record in records])) if records else math.nan,
        "tip_heatmap_peak_mean": float(np.mean([record.tip_heatmap_peak_value for record in records])) if records else math.nan,
        "center_heatmap_spread_mean": float(np.mean([record.center_heatmap_spread_px for record in records])) if records else math.nan,
        "tip_heatmap_spread_mean": float(np.mean([record.tip_heatmap_spread_px for record in records])) if records else math.nan,
        "confidence_mean": float(np.mean([record.confidence for record in records])) if records else math.nan,
        "guardrail_disagreement_count": float(sum(1 for record in records if record.guardrail_status != "accepted")),
        "top_rejection_reasons": _top_rejection_reason_string(records),
    }


def _top_rejection_reason_string(records: list[ReplayRecord]) -> str:
    """Format the most common rejection reasons for reporting."""

    counts: Counter[str] = Counter()
    for record in records:
        if record.guardrail_status == "accepted":
            continue
        for reason in record.rejection_reasons.split(";"):
            if reason and reason != "none":
                counts[reason] += 1
    if not counts:
        return "none"
    return ";".join(f"{reason}:{count}" for reason, count in counts.most_common(5))


def _paired_deltas(
    base_records: list[ReplayRecord],
    other_records: list[ReplayRecord],
) -> dict[str, float]:
    """Compute paired deltas between two models on the same samples."""

    base_by_image = {record.image_path: record for record in base_records}
    other_by_image = {record.image_path: record for record in other_records}
    common_images = sorted(base_by_image.keys() & other_by_image.keys())
    temp_deltas: list[float] = []
    center_deltas: list[float] = []
    tip_deltas: list[float] = []
    peak_deltas: list[float] = []
    spread_deltas: list[float] = []
    guardrail_disagreements = 0
    for image_path in common_images:
        base_record = base_by_image[image_path]
        other_record = other_by_image[image_path]
        if _status_is_accepted(base_record.guardrail_status) and _status_is_accepted(other_record.guardrail_status):
            temp_deltas.append(abs(base_record.guarded_temperature_c - other_record.guarded_temperature_c))
        center_deltas.append(
            math.hypot(
                base_record.predicted_center_x_224 - other_record.predicted_center_x_224,
                base_record.predicted_center_y_224 - other_record.predicted_center_y_224,
            )
        )
        tip_deltas.append(
            math.hypot(
                base_record.predicted_tip_x_224 - other_record.predicted_tip_x_224,
                base_record.predicted_tip_y_224 - other_record.predicted_tip_y_224,
            )
        )
        peak_deltas.append(other_record.tip_heatmap_peak_value - base_record.tip_heatmap_peak_value)
        spread_deltas.append(other_record.tip_heatmap_spread_px - base_record.tip_heatmap_spread_px)
        if base_record.guardrail_status != other_record.guardrail_status:
            guardrail_disagreements += 1
    return {
        "temperature_delta_mean": float(np.mean(temp_deltas)) if temp_deltas else math.nan,
        "temperature_delta_median": float(np.median(temp_deltas)) if temp_deltas else math.nan,
        "temperature_delta_p90": float(np.percentile(temp_deltas, 90)) if temp_deltas else math.nan,
        "center_delta_mean": float(np.mean(center_deltas)) if center_deltas else math.nan,
        "center_delta_median": float(np.median(center_deltas)) if center_deltas else math.nan,
        "center_delta_p90": float(np.percentile(center_deltas, 90)) if center_deltas else math.nan,
        "tip_delta_mean": float(np.mean(tip_deltas)) if tip_deltas else math.nan,
        "tip_delta_median": float(np.median(tip_deltas)) if tip_deltas else math.nan,
        "tip_delta_p90": float(np.percentile(tip_deltas, 90)) if tip_deltas else math.nan,
        "tip_peak_delta_mean": float(np.mean(peak_deltas)) if peak_deltas else math.nan,
        "tip_spread_delta_mean": float(np.mean(spread_deltas)) if spread_deltas else math.nan,
        "guardrail_disagreements": float(guardrail_disagreements),
    }


def _reference_baseline_summary(
    current_int8_path: Path,
    dynamic_range_path: Path,
    split: str,
    decode_method: str,
    window_size: int,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """Load prior baseline prediction tables if they are available for the split."""

    current_int8_summary: dict[str, float] | None = None
    dynamic_range_summary: dict[str, float] | None = None

    if current_int8_path.exists():
        current_rows = _load_prediction_rows(current_int8_path, split=split, decode_method=decode_method, window_size=window_size)
        if current_rows:
            accepted = [
                row
                for row in current_rows.values()
                if _status_is_accepted(str(row.get("int8_guardrail_status", row.get("guardrail_status", ""))))
            ]
            accepted_errors = np.asarray(
                [
                    abs(float(row.get("int8_guarded_temperature_c", row.get("guarded_temperature_c", math.nan))) - float(row["true_temperature_c"]))
                    for row in accepted
                ],
                dtype=np.float64,
            )
            current_int8_summary = {
                "accepted_mae_c": float(np.mean(accepted_errors)) if accepted_errors.size else math.nan,
                "acceptance_rate": float(len(accepted) / len(current_rows)) if current_rows else math.nan,
                "worst_accepted_error_c": float(np.max(accepted_errors)) if accepted_errors.size else math.nan,
                "accepted_gt20_failures": float(
                    sum(
                        1
                        for row in accepted
                        if abs(float(row.get("int8_guarded_temperature_c", row.get("guarded_temperature_c", math.nan))) - float(row["true_temperature_c"])) > 20.0
                    )
                ),
            }

    if dynamic_range_path.exists():
        dynamic_rows = _load_prediction_rows(dynamic_range_path, split=split, decode_method=decode_method, window_size=window_size)
        if dynamic_rows:
            accepted = [
                row
                for row in dynamic_rows.values()
                if _status_is_accepted(str(row.get("candidate_guardrail_status", row.get("guardrail_status", ""))))
            ]
            accepted_errors = np.asarray(
                [
                    abs(float(row.get("candidate_guarded_temperature_c", row.get("guarded_temperature_c", math.nan))) - float(row["true_temperature_c"]))
                    for row in accepted
                ],
                dtype=np.float64,
            )
            dynamic_range_summary = {
                "accepted_mae_c": float(np.mean(accepted_errors)) if accepted_errors.size else math.nan,
                "acceptance_rate": float(len(accepted) / len(dynamic_rows)) if dynamic_rows else math.nan,
                "worst_accepted_error_c": float(np.max(accepted_errors)) if accepted_errors.size else math.nan,
                "accepted_gt20_failures": float(
                    sum(
                        1
                        for row in accepted
                        if abs(float(row.get("candidate_guarded_temperature_c", row.get("guarded_temperature_c", math.nan))) - float(row["true_temperature_c"])) > 20.0
                    )
                ),
            }

    return current_int8_summary, dynamic_range_summary


def _write_report(
    *,
    output_path: Path,
    split: str,
    decode_method: str,
    window_size: int,
    selected_decode_path: Path,
    thresholds_path: Path,
    calibration_candidate_name: str,
    summaries: dict[str, dict[str, float]],
    drift_by_model: dict[str, dict[str, float]],
    current_int8_reference: dict[str, float] | None,
    dynamic_range_reference: dict[str, float] | None,
) -> None:
    """Render the replay findings as a Markdown report."""

    lines = [
        "# Geometry Heatmap v2 QAT TFLite Replay",
        "",
        f"- Split: {split}",
        f"- Decoder: {decode_method} w{window_size}",
        f"- Selected decode artifact: {selected_decode_path}",
        f"- Guardrail thresholds: {thresholds_path}",
        f"- Calibration candidate: {calibration_candidate_name}",
        "",
        "## Model Summary",
        "| model | accepted MAE | acceptance rate | worst accepted | >20 C fails | temp delta mean | temp delta median | temp delta p90 | center delta mean | tip delta mean | guardrail disagreements |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name in sorted(summaries.keys()):
        summary = summaries[model_name]
        drift = drift_by_model.get(model_name, {})
        lines.append(
            "| {model} | {mae:.4f} | {acc:.4f} | {worst:.4f} | {fails:.0f} | {temp_mean:.4f} | {temp_med:.4f} | {temp_p90:.4f} | {center_mean:.4f} | {tip_mean:.4f} | {disagree:.0f} |".format(
                model=model_name,
                mae=summary["accepted_mae_c"],
                acc=summary["acceptance_rate"],
                worst=summary["worst_accepted_error_c"],
                fails=summary["accepted_gt20_failures"],
                temp_mean=drift.get("temperature_delta_mean", math.nan),
                temp_med=drift.get("temperature_delta_median", math.nan),
                temp_p90=drift.get("temperature_delta_p90", math.nan),
                center_mean=drift.get("center_delta_mean", math.nan),
                tip_mean=drift.get("tip_delta_mean", math.nan),
                disagree=drift.get("guardrail_disagreements", summary["guardrail_disagreement_count"]),
            )
        )

    if current_int8_reference is not None:
        lines.extend(
            [
                "",
                "## Current INT8 Reference",
                f"- Accepted MAE: {current_int8_reference['accepted_mae_c']:.4f} C",
                f"- Acceptance rate: {current_int8_reference['acceptance_rate']:.4f}",
                f"- Worst accepted error: {current_int8_reference['worst_accepted_error_c']:.4f} C",
                f"- Accepted >20 C failures: {int(current_int8_reference['accepted_gt20_failures'])}",
            ]
        )
    if dynamic_range_reference is not None:
        lines.extend(
            [
                "",
                "## Dynamic Range Reference",
                f"- Accepted MAE: {dynamic_range_reference['accepted_mae_c']:.4f} C",
                f"- Acceptance rate: {dynamic_range_reference['acceptance_rate']:.4f}",
                f"- Worst accepted error: {dynamic_range_reference['worst_accepted_error_c']:.4f} C",
                f"- Accepted >20 C failures: {int(dynamic_range_reference['accepted_gt20_failures'])}",
            ]
        )

    lines.extend(
        [
            "",
            "## Decision",
            "- Cube.AI packaging is allowed only if the QAT INT8 model satisfies the accuracy and drift gates.",
            "- If the QAT INT8 model still misses the temperature-drift target, the next fix is QAT follow-up training or a higher-capacity heatmap v3.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Evaluate the QAT fine-tuned models and write replay artifacts."""

    parser = argparse.ArgumentParser(description="Evaluate geometry_heatmap_v2 QAT TFLite replay")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--qat-model-path", type=Path, default=DEFAULT_QAT_MODEL_PATH)
    parser.add_argument("--qat-float32-tflite-path", type=Path, default=DEFAULT_QAT_FLOAT32_TFLITE_PATH)
    parser.add_argument("--qat-int8-tflite-path", type=Path, default=DEFAULT_QAT_INT8_TFLITE_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--calibration-json-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--decoder-path", type=Path, default=DEFAULT_DECODER_PATH)
    parser.add_argument("--current-int8-predictions-path", type=Path, default=DEFAULT_CURRENT_INT8_PREDICTIONS_PATH)
    parser.add_argument("--dynamic-range-predictions-path", type=Path, default=DEFAULT_DYNAMIC_RANGE_PREDICTIONS_PATH)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--remaining-worst-accepted-path", type=Path, default=DEFAULT_REMAINING_WORST_ACCEPTED_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    model_path = _resolve_path(repo_root, args.model_path)
    qat_model_path = _resolve_path(repo_root, args.qat_model_path)
    qat_float32_tflite_path = _resolve_path(repo_root, args.qat_float32_tflite_path)
    qat_int8_tflite_path = _resolve_path(repo_root, args.qat_int8_tflite_path)
    manifest_path = _resolve_path(repo_root, args.manifest_path)
    calibration_json_path = _resolve_path(repo_root, args.calibration_json_path)
    thresholds_path = _resolve_path(repo_root, args.thresholds_path)
    decoder_path = _resolve_path(repo_root, args.decoder_path)
    current_int8_predictions_path = _resolve_path(repo_root, args.current_int8_predictions_path)
    dynamic_range_predictions_path = _resolve_path(repo_root, args.dynamic_range_predictions_path)
    predictions_path = _resolve_path(repo_root, args.predictions_path)
    summary_path = _resolve_path(repo_root, args.summary_path)
    remaining_worst_accepted_path = _resolve_path(repo_root, args.remaining_worst_accepted_path)
    report_path = _resolve_path(repo_root, args.report_path)
    debug_dir = _resolve_path(repo_root, args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    split_specific_current = current_int8_predictions_path.with_name(
        f"tflite_replay_predictions_corrected_decoder_{args.split}.csv"
    )
    if split_specific_current.exists():
        current_int8_predictions_path = split_specific_current
    split_specific_dynamic = dynamic_range_predictions_path.with_name(
        f"corrected_decoder_selected_variant_{args.split}_predictions.csv"
    )
    if split_specific_dynamic.exists():
        dynamic_range_predictions_path = split_specific_dynamic

    with decoder_path.open("r", encoding="utf-8") as handle:
        decoder_payload = json.load(handle)
    decode_method = str(decoder_payload["decode_method"])
    window_size = int(decoder_payload["window_size"])
    if decode_method != "softargmax" or window_size != 3:
        raise RuntimeError(f"Expected corrected decoder softargmax w3, found {decode_method} w{window_size}.")

    calibration_candidate, _ = load_selected_calibration_candidate(calibration_json_path)
    thresholds = _load_thresholds(thresholds_path)
    semantic_output_order_indices = [1, 0, 2]

    samples_by_split = {
        split: load_split_samples(
            manifest_path,
            repo_root,
            split=split,
            mode=DEFAULT_PREPROCESSING_MODE,
            input_size=DEFAULT_INPUT_SIZE,
            heatmap_size=DEFAULT_HEATMAP_SIZE,
        ).samples
        for split in (args.split,)
    }
    if args.max_samples is not None:
        for split in list(samples_by_split.keys()):
            samples_by_split[split] = samples_by_split[split][: args.max_samples]

    base_model = load_geometry_heatmap_keras_model(model_path)
    qat_model = load_geometry_heatmap_keras_model(qat_model_path) if qat_model_path.exists() else None
    qat_float32_bundle = load_tflite_model(qat_float32_tflite_path) if qat_float32_tflite_path.exists() else None
    qat_int8_bundle = load_tflite_model(qat_int8_tflite_path) if qat_int8_tflite_path.exists() else None

    current_int8_reference, dynamic_range_reference = _reference_baseline_summary(
        current_int8_predictions_path,
        dynamic_range_predictions_path,
        args.split,
        decode_method,
        window_size,
    )

    all_records: list[ReplayRecord] = []
    model_summaries: dict[str, dict[str, float]] = {}
    drift_by_model: dict[str, dict[str, float]] = {}

    for split, samples in samples_by_split.items():
        print(f"[QAT REPLAY] Evaluating split={split} with {len(samples)} samples.", flush=True)
        base_records = _evaluate_keras_model(
            base_model,
            samples,
            calibration_candidate=calibration_candidate,
            thresholds=thresholds,
            decode_method=decode_method,
            window_size=window_size,
            model_name="base_keras",
            max_samples=args.max_samples,
        )
        base_summary = _summary_from_records(base_records)
        model_summaries["base_keras"] = base_summary
        all_records.extend(base_records)

        if qat_model is not None:
            qat_records = _evaluate_keras_model(
                qat_model,
                samples,
                calibration_candidate=calibration_candidate,
                thresholds=thresholds,
                decode_method=decode_method,
                window_size=window_size,
                model_name="qat_keras",
                max_samples=args.max_samples,
            )
            model_summaries["qat_keras"] = _summary_from_records(qat_records)
            drift_by_model["qat_keras"] = _paired_deltas(base_records, qat_records)
            all_records.extend(qat_records)

        if qat_float32_bundle is not None:
            qat_float32_records = _evaluate_tflite_model(
                qat_float32_bundle,
                samples,
                calibration_candidate=calibration_candidate,
                thresholds=thresholds,
                decode_method=decode_method,
                window_size=window_size,
                model_name="qat_tflite_float32",
                semantic_output_order_indices=semantic_output_order_indices,
                max_samples=args.max_samples,
            )
            model_summaries["qat_tflite_float32"] = _summary_from_records(qat_float32_records)
            drift_by_model["qat_tflite_float32"] = _paired_deltas(base_records, qat_float32_records)
            all_records.extend(qat_float32_records)

        if qat_int8_bundle is not None:
            qat_int8_records = _evaluate_tflite_model(
                qat_int8_bundle,
                samples,
                calibration_candidate=calibration_candidate,
                thresholds=thresholds,
                decode_method=decode_method,
                window_size=window_size,
                model_name="qat_tflite_int8",
                semantic_output_order_indices=semantic_output_order_indices,
                max_samples=args.max_samples,
            )
            model_summaries["qat_tflite_int8"] = _summary_from_records(qat_int8_records)
            drift_by_model["qat_tflite_int8"] = _paired_deltas(base_records, qat_int8_records)
            all_records.extend(qat_int8_records)

            # Overlay generation for the selected test split.
            if args.split == "test":
                # Worst accepted accepted predictions.
                accepted = [
                    record
                    for record in qat_int8_records
                    if _status_is_accepted(record.guardrail_status)
                ]
                accepted_sorted = sorted(
                    accepted,
                    key=lambda record: abs(record.guarded_temperature_c - record.true_temperature_c),
                    reverse=True,
                )
                worst_rows = _records_to_rows(accepted_sorted[:30])
                _write_csv(worst_rows, remaining_worst_accepted_path)

                base_by_image = {record.image_path: record for record in base_records}
                q8_by_image = {record.image_path: record for record in qat_int8_records}
                improved = []
                worse = []
                for image_path in base_by_image.keys() & q8_by_image.keys():
                    base_err = abs(base_by_image[image_path].guarded_temperature_c - base_by_image[image_path].true_temperature_c)
                    q8_err = abs(q8_by_image[image_path].guarded_temperature_c - q8_by_image[image_path].true_temperature_c)
                    if q8_err < base_err:
                        improved.append(q8_by_image[image_path])
                    elif q8_err > base_err:
                        worse.append(q8_by_image[image_path])

                categories = {
                    "worst_accepted": accepted_sorted[:30],
                    "largest_temp_delta": sorted(
                        qat_int8_records,
                        key=lambda record: abs(
                            base_by_image[record.image_path].guarded_temperature_c - record.guarded_temperature_c
                        )
                        if record.image_path in base_by_image
                        else -1.0,
                        reverse=True,
                    )[:30],
                    "largest_tip_delta": sorted(
                        qat_int8_records,
                        key=lambda record: math.hypot(
                            base_by_image[record.image_path].predicted_tip_x_224 - record.predicted_tip_x_224,
                            base_by_image[record.image_path].predicted_tip_y_224 - record.predicted_tip_y_224,
                        )
                        if record.image_path in base_by_image
                        else -1.0,
                        reverse=True,
                    )[:30],
                    "guardrail_disagreements": [
                        record
                        for record in qat_int8_records
                        if record.image_path in base_by_image and base_by_image[record.image_path].guardrail_status != record.guardrail_status
                    ],
                    "improved_vs_current_int8": improved[:30],
                    "worse_than_current_int8": worse[:30],
                }
                for category_name, records in categories.items():
                    for index, record in enumerate(records, start=1):
                        sample_lookup = next(sample for sample in samples if str(sample.metadata["image_path"]) == record.image_path)
                        output_path = debug_dir / category_name / f"{index:03d}_{Path(record.image_path).stem}.png"
                        write_prediction_overlay(
                            sample_lookup,
                            {
                                "predicted_center_x_224": record.predicted_center_x_224,
                                "predicted_center_y_224": record.predicted_center_y_224,
                                "predicted_tip_x_224": record.predicted_tip_x_224,
                                "predicted_tip_y_224": record.predicted_tip_y_224,
                                "predicted_temperature_c_current_mapping": record.guarded_temperature_c,
                                "predicted_temperature_c_calibrated": record.guarded_temperature_c,
                                "absolute_error_c_calibrated": abs(record.guarded_temperature_c - record.true_temperature_c),
                                "confidence": record.confidence,
                                "center_px_mae_224": record.center_px_mae_224,
                                "tip_px_mae_224": record.tip_px_mae_224,
                                "pred_center_heatmap_array": np.squeeze(record.pred_center_heatmap_array).astype(np.float32)
                                if record.pred_center_heatmap_array is not None
                                else np.zeros((56, 56), dtype=np.float32),
                                "pred_tip_heatmap_array": np.squeeze(record.pred_tip_heatmap_array).astype(np.float32)
                                if record.pred_tip_heatmap_array is not None
                                else np.zeros((56, 56), dtype=np.float32),
                            },
                            output_path,
                        )

        # For the report, record the split-specific current INT8 / dynamic range reference if available.
    rows = _records_to_rows(all_records)
    _write_csv(rows, predictions_path)
    _write_csv(
        [
            {"model_name": name, **summary}
            for name, summary in model_summaries.items()
        ],
        summary_path,
    )
    _write_report(
        output_path=report_path,
        split=args.split,
        decode_method=decode_method,
        window_size=window_size,
        selected_decode_path=decoder_path,
        thresholds_path=thresholds_path,
        calibration_candidate_name=calibration_candidate.name,
        summaries=model_summaries,
        drift_by_model=drift_by_model,
        current_int8_reference=current_int8_reference,
        dynamic_range_reference=dynamic_range_reference,
    )

    print(f"[QAT REPLAY] Wrote predictions to {predictions_path}", flush=True)
    print(f"[QAT REPLAY] Wrote summary to {summary_path}", flush=True)
    print(f"[QAT REPLAY] Wrote report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
