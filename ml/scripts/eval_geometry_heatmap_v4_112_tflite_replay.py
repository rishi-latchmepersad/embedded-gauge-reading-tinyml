#!/usr/bin/env python3
"""Replay geometry_heatmap_v4_112 through Keras and TFLite on one split.

The evaluator is intentionally narrow: it evaluates one split at a time,
reuses the corrected decoder lock, and compares the exported float32 and int8
TFLite artifacts against the v3 Keras checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_heatmap_quantization_common import (
    DEFAULT_HEATMAP_SIZE,
    DEFAULT_INPUT_SIZE,
    DEFAULT_PREPROCESSING_MODE,
    decode_and_guard,
    load_semantic_output_order_indices,
    load_split_samples,
    predict_tflite_outputs,
    resolve_repo_path,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import load_geometry_heatmap_keras_model, load_tflite_model, summarize_tflite_contract
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import load_selected_calibration_candidate
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import GeometryGuardrailThresholds


DEFAULT_MODEL_PATH = Path("ml/artifacts/training/geometry_heatmap_v4_112_quant_native_30epoch_smoke/model_v4_112.keras")
DEFAULT_FLOAT_TFLITE_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v4_112_tflite/model_v4_112_float32.tflite")
DEFAULT_INT8_TFLITE_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v4_112_tflite/model_v4_112_int8.tflite")
DEFAULT_CONTRACT_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v4_112_tflite/tflite_tensor_contract.json")
DEFAULT_MANIFEST_PATH = Path("ml/data/geometry_reader_manifest_v2_clean.csv")
DEFAULT_CALIBRATION_PATH = Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json")
DEFAULT_THRESHOLDS_PATH = Path("ml/artifacts/training/geometry_heatmap_v4_112_quant_native/v4_112_guardrail_thresholds.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/deployment/geometry_heatmap_v4_112_tflite")
DEFAULT_PREDICTIONS_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v4_112_tflite/v4_112_tflite_replay_predictions.csv")
DEFAULT_SUMMARY_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v4_112_tflite/v4_112_tflite_replay_summary.csv")
DEFAULT_REMAINING_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v4_112_tflite/v4_112_remaining_worst_accepted.csv")
DEFAULT_REPORT_PATH = Path("ml/reports/geometry_heatmap_v4_112_tflite_replay.md")
DEFAULT_KERAS_VALIDATION_REPORT_PATH = Path("ml/reports/geometry_heatmap_v4_112_keras_validation.md")
DEFAULT_SELECTED_DECODE_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json")

MODEL_TYPES = ("keras_v3", "tflite_float32", "tflite_int8")
V4_HEATMAP_SIZE = 112
V4_SIGMA_PIXELS = 2.5


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a deterministic CSV table."""

    if not rows:
        raise ValueError("Cannot write an empty CSV file.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write a deterministic JSON artifact."""

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
        minimum_celsius=float(selected["minimum_celsius"]),
        maximum_celsius=float(selected["maximum_celsius"]),
        clamp_temperature_to_physical_range=bool(selected["clamp_temperature_to_physical_range"]),
    )


def _load_selected_decode_spec(selection_path: Path) -> tuple[str, int]:
    """Load the locked decoder selection and verify it remains softargmax w3."""

    with selection_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    decode_method = str(payload.get("decode_method", payload.get("selected_decode_method", "")))
    window_size = int(payload.get("window_size", payload.get("selected_window_size", 0)))
    if "_w" in decode_method:
        decode_method, suffix = decode_method.rsplit("_w", 1)
        if suffix.isdigit():
            window_size = int(suffix)
    if decode_method != "softargmax" or window_size != 3:
        raise RuntimeError(
            f"Expected corrected decode softargmax w3, found {decode_method} w{window_size} in {selection_path}."
        )
    return decode_method, window_size


def _load_semantic_names_from_contract(contract_path: Path) -> list[str]:
    """Load the semantic output names from the TFLite tensor contract JSON."""
    with contract_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "semantic_output_names" in payload:
        return [str(name) for name in payload["semantic_output_names"]]
    return []


def _as_output_dict(outputs: Any) -> dict[str, tf.Tensor]:
    """Normalize Keras outputs into the semantic heatmap dictionary.

    Supports 3-output (center, tip, confidence),
    4-output with aux_coords, and 4-output with aux_offset_map.
    """

    if isinstance(outputs, dict):
        result = {
            "center_heatmap": tf.cast(outputs["center_heatmap"], tf.float32),
            "tip_heatmap": tf.cast(outputs["tip_heatmap"], tf.float32),
            "confidence": tf.cast(outputs["confidence"], tf.float32),
        }
        if "aux_coords" in outputs:
            result["aux_coords"] = tf.cast(outputs["aux_coords"], tf.float32)
        if "aux_offset_map" in outputs:
            result["aux_offset_map"] = tf.cast(outputs["aux_offset_map"], tf.float32)
        return result
    center_heatmap, tip_heatmap, confidence, *extra = outputs
    result = {
        "center_heatmap": tf.cast(center_heatmap, tf.float32),
        "tip_heatmap": tf.cast(tip_heatmap, tf.float32),
        "confidence": tf.cast(confidence, tf.float32),
    }
    if extra:
        extra_tensor = extra[0]
        extra_ndim = getattr(extra_tensor, 'ndim', len(extra_tensor.shape) if hasattr(extra_tensor, 'shape') else 0)
        if extra_ndim >= 3:
            result["aux_offset_map"] = tf.cast(extra_tensor, tf.float32)
        else:
            result["aux_coords"] = tf.cast(extra_tensor, tf.float32)
    return result


def _predict_keras_outputs(model: keras.Model, inputs: np.ndarray, *, batch_size: int) -> dict[str, np.ndarray]:
    """Run the Keras model on a batch of inputs and return semantic outputs."""

    outputs = model.predict(inputs, batch_size=batch_size, verbose=0)
    ordered = _as_output_dict(outputs)
    return {name: np.asarray(tensor, dtype=np.float32) for name, tensor in ordered.items()}


def _predict_tflite_outputs(
    model_path: Path,
    inputs: list[np.ndarray],
    *,
    semantic_output_order_indices: list[int],
    semantic_output_names: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run one exported TFLite model and return semantic outputs plus contract metadata."""

    bundle = load_tflite_model(model_path)
    outputs = predict_tflite_outputs(bundle, inputs, semantic_output_order_indices=semantic_output_order_indices)
    contract = summarize_tflite_contract(model_path)
    semantic_outputs: dict[str, np.ndarray] = {
        "center_heatmap": np.asarray(outputs[0], dtype=np.float32),
        "tip_heatmap": np.asarray(outputs[1], dtype=np.float32),
        "confidence": np.asarray(outputs[2], dtype=np.float32),
    }
    if len(outputs) >= 4:
        # Detect aux output type from semantic output names.
        aux_name = semantic_output_names[3] if semantic_output_names and len(semantic_output_names) > 3 else ""
        if "offset_map" in aux_name:
            semantic_outputs["aux_offset_map"] = np.asarray(outputs[3], dtype=np.float32)
        else:
            semantic_outputs["aux_coords"] = np.asarray(outputs[3], dtype=np.float32)
    return semantic_outputs, contract


def _tensor_stats(array: np.ndarray) -> dict[str, float]:
    """Return compact statistics for a tensor-like array."""

    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    return {
        "min": float(np.min(safe)),
        "max": float(np.max(safe)),
        "mean": float(np.mean(safe)),
        "finite_fraction": float(np.mean(finite.astype(np.float32))),
    }


def _status_is_accepted(status: str) -> bool:
    """Treat accepted and clamped rows as usable."""

    return status in {"accepted", "clamped"}


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize replay-style metrics for one model."""

    accepted = [row for row in rows if _status_is_accepted(str(row["guardrail_status"]))]
    accepted_errors = np.asarray(
        [abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"])) for row in accepted],
        dtype=np.float64,
    )
    all_errors = np.asarray(
        [abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"])) for row in rows],
        dtype=np.float64,
    )
    top_rejection_reasons = Counter()
    for row in rows:
        if _status_is_accepted(str(row["guardrail_status"])):
            continue
        for reason in str(row["rejection_reasons"]).split(";"):
            if reason and reason != "none":
                top_rejection_reasons[reason] += 1
    return {
        "count": float(len(rows)),
        "accepted_count": float(len(accepted)),
        "accepted_mae_c": float(np.mean(accepted_errors)) if accepted_errors.size else math.nan,
        "acceptance_rate": float(len(accepted) / len(rows)) if rows else math.nan,
        "worst_accepted_error_c": float(np.max(accepted_errors)) if accepted_errors.size else math.nan,
        "accepted_gt20_failures": float(
            sum(
                1
                for row in accepted
                if abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"])) > 20.0
            )
        ),
        "percentage_under_2c": float(np.mean(all_errors < 2.0) * 100.0) if all_errors.size else math.nan,
        "percentage_under_5c": float(np.mean(all_errors < 5.0) * 100.0) if all_errors.size else math.nan,
        "percentage_under_10c": float(np.mean(all_errors < 10.0) * 100.0) if all_errors.size else math.nan,
        "center_mae_px_224": float(
            np.mean(
                [
                    math.hypot(
                        float(row["predicted_center_x_224"]) - float(row["true_center_x_224"]),
                        float(row["predicted_center_y_224"]) - float(row["true_center_y_224"]),
                    )
                    for row in rows
                ]
            )
        ),
        "tip_mae_px_224": float(
            np.mean(
                [
                    math.hypot(
                        float(row["predicted_tip_x_224"]) - float(row["true_tip_x_224"]),
                        float(row["predicted_tip_y_224"]) - float(row["true_tip_y_224"]),
                    )
                    for row in rows
                ]
            )
        ),
        "angle_mae_degrees": float(
            np.mean([abs(float(row["predicted_angle_degrees"]) - float(row["true_angle_degrees"])) for row in rows])
        ),
        "center_heatmap_peak_mean": float(np.mean([float(row["center_heatmap_peak_value"]) for row in rows])),
        "tip_heatmap_peak_mean": float(np.mean([float(row["tip_heatmap_peak_value"]) for row in rows])),
        "center_heatmap_spread_mean": float(np.mean([float(row["center_heatmap_spread_px"]) for row in rows])),
        "tip_heatmap_spread_mean": float(np.mean([float(row["tip_heatmap_spread_px"]) for row in rows])),
        "confidence_mean": float(np.mean([float(row["confidence"]) for row in rows])),
        "guardrail_disagreement_count": float(sum(1 for row in rows if not _status_is_accepted(str(row["guardrail_status"])))),
        "top_rejection_reasons": ";".join(
            f"{reason}:{count}" for reason, count in top_rejection_reasons.most_common(5)
        )
        if top_rejection_reasons
        else "none",
    }


def _compare_against_reference(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute Keras-vs-candidate drift metrics."""

    reference_by_image = {row["image_path"]: row for row in reference_rows}
    candidate_by_image = {row["image_path"]: row for row in candidate_rows}
    common_images = sorted(reference_by_image.keys() & candidate_by_image.keys())
    temp_deltas: list[float] = []
    center_deltas: list[float] = []
    tip_deltas: list[float] = []
    guardrail_disagreements = 0
    for image_path in common_images:
        reference = reference_by_image[image_path]
        candidate = candidate_by_image[image_path]
        if _status_is_accepted(str(reference["guardrail_status"])) and _status_is_accepted(str(candidate["guardrail_status"])):
            temp_deltas.append(abs(float(reference["guarded_temperature_c"]) - float(candidate["guarded_temperature_c"])))
        center_deltas.append(
            math.hypot(
                float(reference["predicted_center_x_224"]) - float(candidate["predicted_center_x_224"]),
                float(reference["predicted_center_y_224"]) - float(candidate["predicted_center_y_224"]),
            )
        )
        tip_deltas.append(
            math.hypot(
                float(reference["predicted_tip_x_224"]) - float(candidate["predicted_tip_x_224"]),
                float(reference["predicted_tip_y_224"]) - float(candidate["predicted_tip_y_224"]),
            )
        )
        if str(reference["guardrail_status"]) != str(candidate["guardrail_status"]):
            guardrail_disagreements += 1
    return {
        "temperature_delta_mean": float(np.mean(temp_deltas)) if temp_deltas else math.nan,
        "temperature_delta_median": float(np.median(temp_deltas)) if temp_deltas else math.nan,
        "temperature_delta_p90": float(np.percentile(temp_deltas, 90)) if temp_deltas else math.nan,
        "center_delta_mean": float(np.mean(center_deltas)) if center_deltas else math.nan,
        "center_delta_median": float(np.median(center_deltas)) if center_deltas else math.nan,
        "tip_delta_mean": float(np.mean(tip_deltas)) if tip_deltas else math.nan,
        "tip_delta_median": float(np.median(tip_deltas)) if tip_deltas else math.nan,
        "guardrail_disagreements": float(guardrail_disagreements),
    }


def _evaluate_model(
    *,
    model_type: str,
    samples: list[Any],
    outputs: dict[str, np.ndarray],
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    decode_method: str,
    window_size: int,
    offset_scale_px: float = 8.0,
) -> list[dict[str, Any]]:
    """Decode one model on the provided split."""

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        confidence = float(np.ravel(outputs["confidence"][index])[0])
        decoded, guarded = decode_and_guard(
            sample,
            outputs["center_heatmap"][index],
            outputs["tip_heatmap"][index],
            confidence,
            calibration_candidate,
            thresholds,
            decode_method=decode_method,
            window_size=window_size,
            aux_coords=outputs["aux_coords"][index] if "aux_coords" in outputs else None,
            aux_offset_map=outputs["aux_offset_map"][index] if "aux_offset_map" in outputs else None,
            offset_scale_px=offset_scale_px,
        )
        reasons = ";".join(guarded.rejection_reasons) if guarded.rejection_reasons else "none"
        rows.append(
            {
                "model_type": model_type,
                "image_path": str(sample.metadata["image_path"]),
                "split": str(sample.metadata["split"]),
                "source_kind": str(sample.metadata["source_kind"]),
                "preprocessing_mode": str(sample.metadata["preprocessing_mode"]),
                "true_temperature_c": float(sample.metadata["temperature_c"]),
                "true_angle_degrees": float(sample.metadata["angle_degrees"]),
                "true_center_x_224": float(sample.metadata["center_x_224"]),
                "true_center_y_224": float(sample.metadata["center_y_224"]),
                "true_tip_x_224": float(sample.metadata["tip_x_224"]),
                "true_tip_y_224": float(sample.metadata["tip_y_224"]),
                "predicted_center_x_224": float(decoded.predicted_center_x_224),
                "predicted_center_y_224": float(decoded.predicted_center_y_224),
                "predicted_tip_x_224": float(decoded.predicted_tip_x_224),
                "predicted_tip_y_224": float(decoded.predicted_tip_y_224),
                "predicted_angle_degrees": float(decoded.predicted_angle_degrees),
                "predicted_temperature_c": float(decoded.predicted_temperature_c_calibrated),
                "guarded_temperature_c": float(guarded.temperature_c),
                "guardrail_status": guarded.status,
                "center_heatmap_peak_value": float(decoded.center_heatmap_peak_value),
                "tip_heatmap_peak_value": float(decoded.tip_heatmap_peak_value),
                "center_heatmap_spread_px": float(guarded.quality_features.center_heatmap_spread_px),
                "tip_heatmap_spread_px": float(guarded.quality_features.tip_heatmap_spread_px),
                "confidence": float(confidence),
                "rejection_reasons": reasons,
            }
        )
    return rows


def _write_report(
    *,
    report_path: Path,
    split: str,
    decode_method: str,
    window_size: int,
    calibration_name: str,
    keras_summary: dict[str, Any],
    float_summary: dict[str, Any],
    int8_summary: dict[str, Any],
    float_drift: dict[str, float],
    int8_drift: dict[str, float],
    float_contract: dict[str, Any],
    int8_contract: dict[str, Any],
    semantic_output_order_indices: list[int],
    split_allowed: bool,
) -> None:
    """Write a markdown replay report."""

    lines = [
        "# Geometry Heatmap v4 112 TFLite Replay",
        "",
        f"- Split: {split}",
        f"- Decoder: {decode_method} w{window_size}",
        f"- Calibration candidate: {calibration_name}",
        f"- Validation gate passed: {'yes' if split_allowed else 'no'}",
        "",
        "## Keras (val)",
        f"- Accepted MAE: {keras_summary['accepted_mae_c']:.4f} C",
        f"- Acceptance rate: {keras_summary['acceptance_rate']:.4f}",
        f"- Worst accepted error: {keras_summary['worst_accepted_error_c']:.4f} C",
        f"- Accepted >20 C failures: {int(keras_summary['accepted_gt20_failures'])}",
        f"- Under 2/5/10 C: {keras_summary['percentage_under_2c']:.2f}% / {keras_summary['percentage_under_5c']:.2f}% / {keras_summary['percentage_under_10c']:.2f}%",
        f"- Center MAE px: {keras_summary['center_mae_px_224']:.4f}",
        f"- Tip MAE px: {keras_summary['tip_mae_px_224']:.4f}",
        f"- Angle MAE deg: {keras_summary['angle_mae_degrees']:.4f}",
        f"- Temp drift mean/median/p90: {float('nan'):.4f} / {float('nan'):.4f} / {float('nan'):.4f}",
        "",
        "## TFLite FP32",
        f"- Accepted MAE: {float_summary['accepted_mae_c']:.4f} C",
        f"- Acceptance rate: {float_summary['acceptance_rate']:.4f}",
        f"- Worst accepted error: {float_summary['worst_accepted_error_c']:.4f} C",
        f"- Accepted >20 C failures: {int(float_summary['accepted_gt20_failures'])}",
        f"- Temp drift mean/median/p90: {float_drift['temperature_delta_mean']:.4f} / {float_drift['temperature_delta_median']:.4f} / {float_drift['temperature_delta_p90']:.4f}",
        f"- Center drift mean/median: {float_drift['center_delta_mean']:.4f} / {float_drift['center_delta_median']:.4f}",
        f"- Tip drift mean/median: {float_drift['tip_delta_mean']:.4f} / {float_drift['tip_delta_median']:.4f}",
        f"- Guardrail disagreements: {int(float_drift['guardrail_disagreements'])}",
        "",
        "## TFLite INT8",
        f"- Accepted MAE: {int8_summary['accepted_mae_c']:.4f} C",
        f"- Acceptance rate: {int8_summary['acceptance_rate']:.4f}",
        f"- Worst accepted error: {int8_summary['worst_accepted_error_c']:.4f} C",
        f"- Accepted >20 C failures: {int(int8_summary['accepted_gt20_failures'])}",
        f"- Temp drift mean/median/p90: {int8_drift['temperature_delta_mean']:.4f} / {int8_drift['temperature_delta_median']:.4f} / {int8_drift['temperature_delta_p90']:.4f}",
        f"- Center drift mean/median: {int8_drift['center_delta_mean']:.4f} / {int8_drift['center_delta_median']:.4f}",
        f"- Tip drift mean/median: {int8_drift['tip_delta_mean']:.4f} / {int8_drift['tip_delta_median']:.4f}",
        f"- Guardrail disagreements: {int(int8_drift['guardrail_disagreements'])}",
        "",
        "## Tensor Contract",
        f"- FP32 input dtype: {float_contract['input']['dtype']}",
        f"- INT8 input dtype: {int8_contract['input']['dtype']}",
        f"- FP32 output dtypes: {', '.join(output['dtype'] for output in float_contract['outputs'])}",
        f"- INT8 output dtypes: {', '.join(output['dtype'] for output in int8_contract['outputs'])}",
        f"- Semantic output reorder: {semantic_output_order_indices}",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Replay geometry_heatmap_v4_112 on one split and save evaluation artifacts."""

    parser = argparse.ArgumentParser(description="Replay geometry_heatmap_v4_112 TFLite exports")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--float-tflite-path", type=Path, default=DEFAULT_FLOAT_TFLITE_PATH)
    parser.add_argument("--int8-tflite-path", type=Path, default=DEFAULT_INT8_TFLITE_PATH)
    parser.add_argument("--contract-path", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--selected-decode-path", type=Path, default=DEFAULT_SELECTED_DECODE_PATH)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--remaining-path", type=Path, default=DEFAULT_REMAINING_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--keras-validation-report-path", type=Path, default=DEFAULT_KERAS_VALIDATION_REPORT_PATH)
    parser.add_argument("--offset-scale-px", type=float, default=8.0,
                        help="Heatmap pixels per unit tanh range for aux_offset_map decode")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    manifest_path = resolve_repo_path(repo_root, args.manifest_path)
    model_path = resolve_repo_path(repo_root, args.model_path)
    float_tflite_path = resolve_repo_path(repo_root, args.float_tflite_path)
    int8_tflite_path = resolve_repo_path(repo_root, args.int8_tflite_path)
    contract_path = resolve_repo_path(repo_root, args.contract_path)
    calibration_path = resolve_repo_path(repo_root, args.calibration_path)
    thresholds_path = resolve_repo_path(repo_root, args.thresholds_path)
    selected_decode_path = resolve_repo_path(repo_root, args.selected_decode_path)
    predictions_path = resolve_repo_path(repo_root, args.predictions_path)
    summary_path = resolve_repo_path(repo_root, args.summary_path)
    remaining_path = resolve_repo_path(repo_root, args.remaining_path)
    report_path = resolve_repo_path(repo_root, args.report_path)
    keras_validation_report_path = resolve_repo_path(repo_root, args.keras_validation_report_path)

    if args.output_suffix:
        predictions_path = predictions_path.with_name(f"{predictions_path.stem}_{args.output_suffix}{predictions_path.suffix}")
        summary_path = summary_path.with_name(f"{summary_path.stem}_{args.output_suffix}{summary_path.suffix}")
        remaining_path = remaining_path.with_name(f"{remaining_path.stem}_{args.output_suffix}{remaining_path.suffix}")
        report_path = report_path.with_name(f"{report_path.stem}_{args.output_suffix}{report_path.suffix}")
        keras_validation_report_path = keras_validation_report_path.with_name(
            f"{keras_validation_report_path.stem}_{args.output_suffix}{keras_validation_report_path.suffix}"
        )

    decode_method, window_size = _load_selected_decode_spec(selected_decode_path)
    calibration_candidate, calibration_json = load_selected_calibration_candidate(calibration_path)
    thresholds = _load_thresholds(thresholds_path)
    semantic_output_order_indices = load_semantic_output_order_indices(contract_path)
    semantic_output_names = _load_semantic_names_from_contract(contract_path)

    examples = load_split_samples(
        manifest_path,
        repo_root,
        split=args.split,
        mode=DEFAULT_PREPROCESSING_MODE,
        input_size=DEFAULT_INPUT_SIZE,
        heatmap_size=V4_HEATMAP_SIZE,
        sigma_pixels=V4_SIGMA_PIXELS,
    ).samples
    inputs = np.stack([sample.crop_image for sample in examples], axis=0).astype(np.float32)

    keras_model = load_geometry_heatmap_keras_model(model_path)
    keras_outputs = _predict_keras_outputs(keras_model, inputs, batch_size=args.batch_size)
    keras_rows = _evaluate_model(
        model_type="keras_v3",
        samples=examples,
        outputs=keras_outputs,
        calibration_candidate=calibration_candidate,
        thresholds=thresholds,
        decode_method=decode_method,
        window_size=window_size,
        offset_scale_px=args.offset_scale_px,
    )
    keras_summary = _summarize_rows(keras_rows)

    float_outputs, float_contract = _predict_tflite_outputs(
        float_tflite_path,
        list(inputs),
        semantic_output_order_indices=semantic_output_order_indices,
        semantic_output_names=semantic_output_names,
    )
    float_rows = _evaluate_model(
        model_type="tflite_float32",
        samples=examples,
        outputs=float_outputs,
        calibration_candidate=calibration_candidate,
        thresholds=thresholds,
        decode_method=decode_method,
        window_size=window_size,
        offset_scale_px=args.offset_scale_px,
    )
    float_summary = _summarize_rows(float_rows)
    float_drift = _compare_against_reference(keras_rows, float_rows)

    int8_outputs, int8_contract = _predict_tflite_outputs(
        int8_tflite_path,
        list(inputs),
        semantic_output_order_indices=semantic_output_order_indices,
        semantic_output_names=semantic_output_names,
    )
    int8_rows = _evaluate_model(
        model_type="tflite_int8",
        samples=examples,
        outputs=int8_outputs,
        calibration_candidate=calibration_candidate,
        thresholds=thresholds,
        decode_method=decode_method,
        window_size=window_size,
        offset_scale_px=args.offset_scale_px,
    )
    int8_summary = _summarize_rows(int8_rows)
    int8_drift = _compare_against_reference(keras_rows, int8_rows)

    all_rows = keras_rows + float_rows + int8_rows
    _write_csv(all_rows, predictions_path)
    _write_csv(
        [
            {
                "model_type": "keras_v3",
                **keras_summary,
                "temperature_delta_mean": math.nan,
                "temperature_delta_median": math.nan,
                "temperature_delta_p90": math.nan,
                "center_delta_mean": math.nan,
                "center_delta_median": math.nan,
                "tip_delta_mean": math.nan,
                "tip_delta_median": math.nan,
                "guardrail_disagreements_vs_keras": 0.0,
            },
            {
                "model_type": "tflite_float32",
                **float_summary,
                **float_drift,
            },
            {
                "model_type": "tflite_int8",
                **int8_summary,
                **int8_drift,
            },
        ],
        summary_path,
    )

    remaining_rows = [
        row
        for row in int8_rows
        if _status_is_accepted(str(row["guardrail_status"])) and abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"])) > 10.0
    ]
    _write_csv(remaining_rows if remaining_rows else int8_rows[:1], remaining_path)
    contract_payload = {
        "selected_decode_path": str(selected_decode_path),
        "selected_decode_method": decode_method,
        "selected_window_size": window_size,
        "semantic_output_order_indices": semantic_output_order_indices,
        "calibration_path": str(calibration_path),
        "calibration_candidate": calibration_candidate.name,
        "calibration_json": calibration_json,
        "float32": float_contract,
        "int8": int8_contract,
    }
    _write_json(contract_payload, contract_path)

    split_allowed = (
        int8_summary["accepted_mae_c"] <= 4.5
        and int8_summary["acceptance_rate"] >= 0.65
        and int8_summary["worst_accepted_error_c"] < 20.0
        and int8_summary["accepted_gt20_failures"] <= 0
        and int8_drift["temperature_delta_mean"] <= 1.0
        and int8_drift["tip_delta_mean"] < 14.82
    )
    _write_report(
        report_path=report_path,
        split=args.split,
        decode_method=decode_method,
        window_size=window_size,
        calibration_name=calibration_candidate.name,
        keras_summary=keras_summary,
        float_summary=float_summary,
        int8_summary=int8_summary,
        float_drift=float_drift,
        int8_drift=int8_drift,
        float_contract=float_contract,
        int8_contract=int8_contract,
        semantic_output_order_indices=semantic_output_order_indices,
        split_allowed=split_allowed,
    )
    if args.split == "val":
        _write_report(
            report_path=keras_validation_report_path,
            split=args.split,
            decode_method=decode_method,
            window_size=window_size,
            calibration_name=calibration_candidate.name,
            keras_summary=keras_summary,
            float_summary=float_summary,
            int8_summary=int8_summary,
            float_drift=float_drift,
            int8_drift=int8_drift,
            float_contract=float_contract,
            int8_contract=int8_contract,
            semantic_output_order_indices=semantic_output_order_indices,
            split_allowed=split_allowed,
        )

    print(f"[V4 REPLAY] Split: {args.split}", flush=True)
    print(f"[V4 REPLAY] Keras accepted MAE: {keras_summary['accepted_mae_c']:.4f} C", flush=True)
    print(f"[V4 REPLAY] Keras acceptance rate: {keras_summary['acceptance_rate']:.4f}", flush=True)
    print(f"[V4 REPLAY] Keras worst accepted error: {keras_summary['worst_accepted_error_c']:.4f} C", flush=True)
    print(f"[V4 REPLAY] INT8 accepted MAE: {int8_summary['accepted_mae_c']:.4f} C", flush=True)
    print(f"[V4 REPLAY] INT8 acceptance rate: {int8_summary['acceptance_rate']:.4f}", flush=True)
    print(f"[V4 REPLAY] INT8 worst accepted error: {int8_summary['worst_accepted_error_c']:.4f} C", flush=True)
    print(
        f"[V4 REPLAY] INT8 temp drift mean/median/p90: "
        f"{int8_drift['temperature_delta_mean']:.4f} / {int8_drift['temperature_delta_median']:.4f} / {int8_drift['temperature_delta_p90']:.4f}",
        flush=True,
    )
    print(
        f"[V4 REPLAY] INT8 tip drift mean/median: {int8_drift['tip_delta_mean']:.4f} / {int8_drift['tip_delta_median']:.4f}",
        flush=True,
    )
    print(f"[V4 REPLAY] Guardrail disagreements: {int(int8_drift['guardrail_disagreements'])}", flush=True)
    print(f"[V4 REPLAY] Tensor contract: {contract_path}", flush=True)


if __name__ == "__main__":
    main()
