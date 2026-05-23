#!/usr/bin/env python3
"""Evaluate the exported geometry heatmap v2 TFLite variants against Keras."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import circular_angle_error_degrees
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
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import load_tflite_model
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import load_selected_calibration_candidate
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import GeometryGuardrailThresholds


SPLITS = ("train", "val", "test")
DEFAULT_VARIANT_ROOT = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2")
DEFAULT_CURRENT_REFERENCE_MODEL = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/model_int8.tflite")
DEFAULT_CURRENT_REFERENCE_CONTRACT = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/tflite_tensor_contract.json")
DEFAULT_TRAINED_MODEL = Path("ml/artifacts/training/geometry_heatmap_v2/model.keras")
DEFAULT_CALIBRATION_PATH = Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json")
DEFAULT_THRESHOLDS_PATH = Path("ml/artifacts/training/geometry_heatmap_v2_board_replay/selected_board_guardrail_thresholds.json")
DEFAULT_SELECTED_DECODE_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json")
DEFAULT_OUTPUT_DIR = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2")
DEFAULT_REPORT_PATH = Path("ml/reports/geometry_heatmap_v2_tflite_variant_comparison.md")
DEFAULT_DECISION_REPORT_PATH = Path("ml/reports/geometry_heatmap_v2_quantization_readiness_decision.md")
DEFAULT_DEBUG_DIR = Path("ml/debug/geometry_heatmap_v2_quantization_drift_v2")

VARIANT_PRIORITY = {
    "variant_a_full_int8_identity": 0,
    "variant_a_full_int8_identity_mild": 1,
    "variant_a_full_int8_identity_mild_medium": 2,
    "variant_a_full_int8_stratified": 3,
    "variant_c_int8_input_float_output": 4,
    "variant_b_float_io_internal_int8": 5,
    "variant_d_dynamic_range": 6,
}


@dataclass(frozen=True)
class SplitCache:
    """Cached samples and baseline predictions for one split."""

    split: str
    samples: list[Any]


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a stable CSV file."""

    if not rows:
        raise ValueError("Cannot write an empty CSV file.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write a JSON payload with stable formatting."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_thresholds(thresholds_path: Path) -> GeometryGuardrailThresholds:
    """Load the selected board-replay guardrail thresholds."""

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


def _load_selected_decode_method(output_dir: Path, fallback_method: str, fallback_window_size: int) -> tuple[str, int]:
    """Load the preferred decode method from the saved selection artifact if available."""

    selection_path = output_dir / "selected_decode_method_corrected.json"
    if not selection_path.exists():
        selection_path = output_dir / "selected_decode_method.json"
    if not selection_path.exists():
        selection_path = DEFAULT_SELECTED_DECODE_PATH
    if not selection_path.exists():
        raise RuntimeError(
            f"Missing selected decode artifact at {selection_path}. "
            "Phase 7.5 replay scripts require the saved decode selection and must not fall back to softargmax."
        )
    with selection_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "decode_method" in payload and "window_size" in payload:
        selected_method = str(payload["decode_method"])
        selected_window_size = int(payload["window_size"])
    elif "selected_decode_method" in payload and "selected_window_size" in payload:
        selected_method = str(payload["selected_decode_method"])
        selected_window_size = int(payload["selected_window_size"])
    else:
        raise RuntimeError(f"Decode selection artifact at {selection_path} is missing required fields.")
    if selected_method != "softargmax" or int(selected_window_size) != 3:
        raise RuntimeError(
            f"Expected corrected decode softargmax w3, found {selected_method} w{selected_window_size} in {selection_path}."
        )
    return selected_method, selected_window_size


def _load_baseline_decode_rows(predictions_path: Path, decode_method: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the saved Keras and current-int8 decode rows for one decode method."""

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    with predictions_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["decode_method"] != decode_method:
                continue
            rows_by_key[(row["split"], row["image_path"])] = row
    if not rows_by_key:
        raise RuntimeError(f"No baseline decode rows found for decode method {decode_method!r} in {predictions_path}.")
    return rows_by_key


def _load_baseline_summary_rows(summary_path: Path, decode_method: str) -> dict[str, dict[str, Any]]:
    """Load the saved decode summary rows for one decode method."""

    rows_by_split: dict[str, dict[str, Any]] = {}
    with summary_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["decode_method"] != decode_method:
                continue
            rows_by_split[str(row["split"])] = row
    if not rows_by_split:
        raise RuntimeError(f"No decode summary rows found for decode method {decode_method!r} in {summary_path}.")
    return rows_by_split


def _baseline_decoded_and_guarded(
    sample: Any,
    baseline_row: dict[str, Any],
    *,
    prefix: str,
) -> tuple[Any, Any]:
    """Rebuild a lightweight decoded/guarded bundle from cached CSV rows."""

    decoded = SimpleNamespace(
        true_temperature_c=float(baseline_row["true_temperature_c"]),
        true_angle_degrees=float(baseline_row["true_angle_degrees"]),
        true_center_x_224=float(sample.metadata["center_x_224"]),
        true_center_y_224=float(sample.metadata["center_y_224"]),
        true_tip_x_224=float(sample.metadata["tip_x_224"]),
        true_tip_y_224=float(sample.metadata["tip_y_224"]),
        predicted_center_x_224=float(baseline_row[f"{prefix}_predicted_center_x_224"]),
        predicted_center_y_224=float(baseline_row[f"{prefix}_predicted_center_y_224"]),
        predicted_tip_x_224=float(baseline_row[f"{prefix}_predicted_tip_x_224"]),
        predicted_tip_y_224=float(baseline_row[f"{prefix}_predicted_tip_y_224"]),
        predicted_angle_degrees=float(baseline_row[f"{prefix}_predicted_angle_degrees"]),
        predicted_temperature_c_current_mapping=float(baseline_row[f"{prefix}_predicted_temperature_c_current_mapping"]),
        predicted_temperature_c_calibrated=float(baseline_row[f"{prefix}_predicted_temperature_c_calibrated"]),
        absolute_error_c_current_mapping=float(baseline_row[f"{prefix}_absolute_error_c_current_mapping"]),
        absolute_error_c_calibrated=float(baseline_row[f"{prefix}_absolute_error_c_calibrated"]),
        center_heatmap_peak_value=float(baseline_row[f"{prefix}_center_heatmap_peak_value"]),
        tip_heatmap_peak_value=float(baseline_row[f"{prefix}_tip_heatmap_peak_value"]),
        center_heatmap_entropy=float(baseline_row[f"{prefix}_center_heatmap_entropy"]),
        tip_heatmap_entropy=float(baseline_row[f"{prefix}_tip_heatmap_entropy"]),
        center_heatmap_spread_px=float(baseline_row[f"{prefix}_center_heatmap_spread_px"]),
        tip_heatmap_spread_px=float(baseline_row[f"{prefix}_tip_heatmap_spread_px"]),
        confidence=float(baseline_row[f"{prefix}_confidence"]),
    )
    guarded = SimpleNamespace(
        status=str(baseline_row[f"{prefix}_guardrail_status"]),
        rejection_reasons=[]
        if str(baseline_row[f"{prefix}_rejection_reasons"]) in {"", "none"}
        else str(baseline_row[f"{prefix}_rejection_reasons"]).split(";"),
        temperature_c=float(baseline_row[f"{prefix}_guarded_temperature_c"]),
        quality_features=SimpleNamespace(
            center_heatmap_entropy=float(baseline_row[f"{prefix}_center_heatmap_entropy"]),
            tip_heatmap_entropy=float(baseline_row[f"{prefix}_tip_heatmap_entropy"]),
            center_heatmap_spread_px=float(baseline_row[f"{prefix}_center_heatmap_spread_px"]),
            tip_heatmap_spread_px=float(baseline_row[f"{prefix}_tip_heatmap_spread_px"]),
        ),
    )
    return decoded, guarded


def _baseline_reference_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the cached current-reference INT8 predictions for one split."""

    accepted_errors = np.asarray(
        [
            abs(float(row["int8_guarded_temperature_c"]) - float(row["true_temperature_c"]))
            for row in rows
            if row["int8_guardrail_status"] != "rejected"
        ],
        dtype=np.float64,
    )
    return {
        "accepted_mae_c": float(np.mean(accepted_errors)) if accepted_errors.size else math.nan,
        "acceptance_rate": float(np.mean([_status_is_accepted(str(row["int8_guardrail_status"])) for row in rows])),
        "worst_accepted_error_c": float(np.max(accepted_errors)) if accepted_errors.size else math.nan,
        "accepted_gt20_failures": int(
            sum(
                1
                for row in rows
                if row["int8_guardrail_status"] != "rejected"
                and abs(float(row["int8_guarded_temperature_c"]) - float(row["true_temperature_c"])) > 20.0
            )
        ),
        "guardrail_disagreement_count": int(sum(1 for row in rows if row["keras_guardrail_status"] != row["int8_guardrail_status"])),
        "keras_vs_reference_temperature_delta_mean": float(
            np.mean([float(row["keras_vs_int8_temperature_delta_c"]) for row in rows])
        ),
        "keras_vs_reference_tip_delta_mean": float(np.mean([float(row["keras_vs_int8_tip_delta_px"]) for row in rows])),
        "keras_vs_reference_center_delta_mean": float(
            np.mean([float(row["keras_vs_int8_center_delta_px"]) for row in rows])
        ),
    }


def _status_is_accepted(status: str) -> bool:
    """Treat accepted and clamped results as usable predictions."""

    return status in {"accepted", "clamped"}


def _variant_priority(variant_name: str) -> int:
    """Rank Cube.AI-facing export families from simplest to least preferred."""

    return int(VARIANT_PRIORITY.get(variant_name, len(VARIANT_PRIORITY)))


def _decode_prediction(
    sample: Any,
    center_heatmap: np.ndarray,
    tip_heatmap: np.ndarray,
    confidence: float,
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    *,
    decode_method: str,
    window_size: int,
) -> tuple[Any, Any]:
    """Decode heatmaps and apply guardrails."""

    return decode_and_guard(
        sample,
        center_heatmap,
        tip_heatmap,
        confidence,
        calibration_candidate,
        thresholds,
        decode_method=decode_method,  # type: ignore[arg-type]
        window_size=window_size,
    )


def _summary(rows: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    """Summarize one set of per-sample predictions."""

    accepted_errors = np.asarray(
        [
            abs(float(row["variant_guarded_temperature_c"]) - float(row["true_temperature_c"]))
            for row in rows
            if row["variant_guardrail_status"] != "rejected"
        ],
        dtype=np.float64,
    )
    temp_deltas = np.asarray([float(row["keras_vs_variant_temperature_delta_c"]) for row in rows], dtype=np.float64)
    tip_deltas = np.asarray([float(row["keras_vs_variant_tip_delta_px"]) for row in rows], dtype=np.float64)
    center_deltas = np.asarray([float(row["keras_vs_variant_center_delta_px"]) for row in rows], dtype=np.float64)
    center_peak_deltas = np.asarray([float(row["keras_vs_variant_center_peak_delta"]) for row in rows], dtype=np.float64)
    tip_peak_deltas = np.asarray([float(row["keras_vs_variant_tip_peak_delta"]) for row in rows], dtype=np.float64)
    center_spread_deltas = np.asarray([float(row["keras_vs_variant_center_spread_delta_px"]) for row in rows], dtype=np.float64)
    tip_spread_deltas = np.asarray([float(row["keras_vs_variant_tip_spread_delta_px"]) for row in rows], dtype=np.float64)
    disagreement_count = sum(1 for row in rows if row["keras_guardrail_status"] != row["variant_guardrail_status"])
    gt20_failures = sum(
        1
        for row in rows
        if row["variant_guardrail_status"] != "rejected"
        and abs(float(row["variant_guarded_temperature_c"]) - float(row["true_temperature_c"])) > 20.0
    )
    return {
        "label": label,
        "count": int(len(rows)),
        "accepted_count": int(sum(_status_is_accepted(str(row["variant_guardrail_status"])) for row in rows)),
        "acceptance_rate": float(np.mean([_status_is_accepted(str(row["variant_guardrail_status"])) for row in rows])),
        "accepted_mae_c": float(np.mean(accepted_errors)) if accepted_errors.size else math.nan,
        "accepted_rmse_c": float(np.sqrt(np.mean(np.square(accepted_errors)))) if accepted_errors.size else math.nan,
        "worst_accepted_error_c": float(np.max(accepted_errors)) if accepted_errors.size else math.nan,
        "percentage_under_2c": float(np.mean(accepted_errors < 2.0) * 100.0) if accepted_errors.size else 0.0,
        "percentage_under_5c": float(np.mean(accepted_errors < 5.0) * 100.0) if accepted_errors.size else 0.0,
        "percentage_under_10c": float(np.mean(accepted_errors < 10.0) * 100.0) if accepted_errors.size else 0.0,
        "accepted_gt20_failures": int(gt20_failures),
        "keras_vs_variant_temperature_delta_mean": float(np.mean(temp_deltas)),
        "keras_vs_variant_temperature_delta_median": float(np.median(temp_deltas)),
        "keras_vs_variant_temperature_delta_p90": float(np.percentile(temp_deltas, 90)),
        "keras_vs_variant_center_delta_mean": float(np.mean(center_deltas)),
        "keras_vs_variant_center_delta_median": float(np.median(center_deltas)),
        "keras_vs_variant_center_delta_p90": float(np.percentile(center_deltas, 90)),
        "keras_vs_variant_tip_delta_mean": float(np.mean(tip_deltas)),
        "keras_vs_variant_tip_delta_median": float(np.median(tip_deltas)),
        "keras_vs_variant_tip_delta_p90": float(np.percentile(tip_deltas, 90)),
        "keras_vs_variant_center_peak_delta_mean": float(np.mean(center_peak_deltas)),
        "keras_vs_variant_center_peak_delta_median": float(np.median(center_peak_deltas)),
        "keras_vs_variant_center_peak_delta_p90": float(np.percentile(center_peak_deltas, 90)),
        "keras_vs_variant_tip_peak_delta_mean": float(np.mean(tip_peak_deltas)),
        "keras_vs_variant_tip_peak_delta_median": float(np.median(tip_peak_deltas)),
        "keras_vs_variant_tip_peak_delta_p90": float(np.percentile(tip_peak_deltas, 90)),
        "keras_vs_variant_center_spread_delta_px_mean": float(np.mean(center_spread_deltas)),
        "keras_vs_variant_center_spread_delta_px_median": float(np.median(center_spread_deltas)),
        "keras_vs_variant_center_spread_delta_px_p90": float(np.percentile(center_spread_deltas, 90)),
        "keras_vs_variant_tip_spread_delta_px_mean": float(np.mean(tip_spread_deltas)),
        "keras_vs_variant_tip_spread_delta_px_median": float(np.median(tip_spread_deltas)),
        "keras_vs_variant_tip_spread_delta_px_p90": float(np.percentile(tip_spread_deltas, 90)),
        "guardrail_disagreement_count": int(disagreement_count),
        "top_rejection_reasons": Counter(
            reason for row in rows if row["variant_guardrail_status"] == "rejected" for reason in str(row["variant_rejection_reasons"]).split(";")
        ),
    }


def _json_safe_summary(summary_row: dict[str, Any]) -> dict[str, Any]:
    """Convert non-JSON-native fields into serializable values."""

    safe_row: dict[str, Any] = {}
    for key, value in summary_row.items():
        if isinstance(value, Counter):
            safe_row[key] = dict(value)
        elif isinstance(value, np.generic):
            safe_row[key] = value.item()
        else:
            safe_row[key] = value
    return safe_row


def _reference_summary_for_split(
    *,
    cache: dict[str, Any],
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    decode_method: str,
    window_size: int,
) -> dict[str, Any]:
    """Summarize the current reference int8 model against Keras for one split."""

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(cache["samples"]):
        keras_decoded, keras_guarded = _decode_prediction(
            sample,
            cache["keras_center"][index],
            cache["keras_tip"][index],
            float(np.ravel(cache["keras_confidence"][index])[0]),
            calibration_candidate,
            thresholds,
            decode_method=decode_method,
            window_size=window_size,
        )
        reference_decoded, reference_guarded = _decode_prediction(
            sample,
            cache["reference_center"][index],
            cache["reference_tip"][index],
            float(np.ravel(cache["reference_confidence"][index])[0]),
            calibration_candidate,
            thresholds,
            decode_method=decode_method,
            window_size=window_size,
        )
        rows.append(
            {
                "true_temperature_c": float(sample.metadata["temperature_c"]),
                "keras_guardrail_status": str(keras_guarded.status),
                "reference_guardrail_status": str(reference_guarded.status),
                "reference_guarded_temperature_c": float(reference_guarded.temperature_c),
                "keras_vs_reference_temperature_delta_c": float(
                    abs(float(keras_decoded.predicted_temperature_c_calibrated) - float(reference_decoded.predicted_temperature_c_calibrated))
                ),
                "keras_vs_reference_tip_delta_px": float(
                    math.hypot(
                        float(keras_decoded.predicted_tip_x_224) - float(reference_decoded.predicted_tip_x_224),
                        float(keras_decoded.predicted_tip_y_224) - float(reference_decoded.predicted_tip_y_224),
                    )
                ),
                "keras_vs_reference_center_delta_px": float(
                    math.hypot(
                        float(keras_decoded.predicted_center_x_224) - float(reference_decoded.predicted_center_x_224),
                        float(keras_decoded.predicted_center_y_224) - float(reference_decoded.predicted_center_y_224),
                    )
                ),
            }
        )

    accepted_errors = np.asarray(
        [
            abs(float(row["reference_guarded_temperature_c"]) - float(row["true_temperature_c"]))
            for row in rows
            if row["reference_guardrail_status"] != "rejected"
        ],
        dtype=np.float64,
    )
    return {
        "accepted_mae_c": float(np.mean(accepted_errors)) if accepted_errors.size else math.nan,
        "acceptance_rate": float(np.mean([_status_is_accepted(str(row["reference_guardrail_status"])) for row in rows])),
        "worst_accepted_error_c": float(np.max(accepted_errors)) if accepted_errors.size else math.nan,
        "accepted_gt20_failures": int(
            sum(
                1
                for row in rows
                if row["reference_guardrail_status"] != "rejected"
                and abs(float(row["reference_guarded_temperature_c"]) - float(row["true_temperature_c"])) > 20.0
            )
        ),
        "guardrail_disagreement_count": int(sum(1 for row in rows if row["keras_guardrail_status"] != row["reference_guardrail_status"])),
        "keras_vs_reference_temperature_delta_mean": float(np.mean([row["keras_vs_reference_temperature_delta_c"] for row in rows])),
        "keras_vs_reference_tip_delta_mean": float(np.mean([row["keras_vs_reference_tip_delta_px"] for row in rows])),
        "keras_vs_reference_center_delta_mean": float(np.mean([row["keras_vs_reference_center_delta_px"] for row in rows])),
    }


def _prediction_row(
    *,
    sample: Any,
    keras_decoded: Any,
    keras_guarded: Any,
    reference_decoded: Any,
    reference_guarded: Any,
    variant_name: str,
    variant_decoded: Any,
    variant_guarded: Any,
    decode_method: str,
) -> dict[str, Any]:
    """Flatten one sample prediction bundle into a CSV row."""

    row: dict[str, Any] = {
        "variant_name": variant_name,
        "decode_method": decode_method,
        "split": str(sample.metadata["split"]),
        "image_path": str(sample.metadata["image_path"]),
        "source_manifest": str(sample.metadata.get("source_manifest", "")),
        "source_folder": Path(str(sample.metadata.get("image_path", ""))).parent.name,
        "quality_flag": str(sample.metadata.get("quality_flag", "")),
        "preprocessing_mode": str(sample.metadata.get("preprocessing_mode", "")),
        "true_temperature_c": float(sample.metadata["temperature_c"]),
        "true_angle_degrees": float(keras_decoded.true_angle_degrees),
        "true_center_x_224": float(keras_decoded.true_center_x_224),
        "true_center_y_224": float(keras_decoded.true_center_y_224),
        "true_tip_x_224": float(keras_decoded.true_tip_x_224),
        "true_tip_y_224": float(keras_decoded.true_tip_y_224),
        "keras_predicted_temperature_c_calibrated": float(keras_decoded.predicted_temperature_c_calibrated),
        "reference_predicted_temperature_c_calibrated": float(reference_decoded.predicted_temperature_c_calibrated),
        "variant_predicted_temperature_c_calibrated": float(variant_decoded.predicted_temperature_c_calibrated),
        "keras_absolute_error_c_calibrated": float(keras_decoded.absolute_error_c_calibrated),
        "reference_absolute_error_c_calibrated": float(reference_decoded.absolute_error_c_calibrated),
        "variant_absolute_error_c_calibrated": float(variant_decoded.absolute_error_c_calibrated),
        "keras_guardrail_status": str(keras_guarded.status),
        "reference_guardrail_status": str(reference_guarded.status),
        "variant_guardrail_status": str(variant_guarded.status),
        "keras_rejection_reasons": ";".join(keras_guarded.rejection_reasons) if keras_guarded.rejection_reasons else "none",
        "reference_rejection_reasons": ";".join(reference_guarded.rejection_reasons) if reference_guarded.rejection_reasons else "none",
        "variant_rejection_reasons": ";".join(variant_guarded.rejection_reasons) if variant_guarded.rejection_reasons else "none",
        "keras_center_heatmap_peak_value": float(keras_decoded.center_heatmap_peak_value),
        "keras_tip_heatmap_peak_value": float(keras_decoded.tip_heatmap_peak_value),
        "variant_center_heatmap_peak_value": float(variant_decoded.center_heatmap_peak_value),
        "variant_tip_heatmap_peak_value": float(variant_decoded.tip_heatmap_peak_value),
        "keras_center_heatmap_entropy": float(keras_guarded.quality_features.center_heatmap_entropy),
        "keras_tip_heatmap_entropy": float(keras_guarded.quality_features.tip_heatmap_entropy),
        "variant_center_heatmap_entropy": float(variant_guarded.quality_features.center_heatmap_entropy),
        "variant_tip_heatmap_entropy": float(variant_guarded.quality_features.tip_heatmap_entropy),
        "keras_center_heatmap_spread_px": float(keras_guarded.quality_features.center_heatmap_spread_px),
        "keras_tip_heatmap_spread_px": float(keras_guarded.quality_features.tip_heatmap_spread_px),
        "variant_center_heatmap_spread_px": float(variant_guarded.quality_features.center_heatmap_spread_px),
        "variant_tip_heatmap_spread_px": float(variant_guarded.quality_features.tip_heatmap_spread_px),
        "keras_confidence": float(keras_decoded.confidence),
        "variant_confidence": float(variant_decoded.confidence),
        "keras_vs_variant_temperature_delta_c": float(
            abs(float(keras_decoded.predicted_temperature_c_calibrated) - float(variant_decoded.predicted_temperature_c_calibrated))
        ),
        "keras_vs_variant_center_delta_px": float(
            math.hypot(
                float(keras_decoded.predicted_center_x_224) - float(variant_decoded.predicted_center_x_224),
                float(keras_decoded.predicted_center_y_224) - float(variant_decoded.predicted_center_y_224),
            )
        ),
        "keras_vs_variant_tip_delta_px": float(
            math.hypot(
                float(keras_decoded.predicted_tip_x_224) - float(variant_decoded.predicted_tip_x_224),
                float(keras_decoded.predicted_tip_y_224) - float(variant_decoded.predicted_tip_y_224),
            )
        ),
        "keras_vs_variant_angle_delta_degrees": float(
            circular_angle_error_degrees(
                float(keras_decoded.predicted_angle_degrees),
                float(variant_decoded.predicted_angle_degrees),
            )
        ),
        "keras_vs_variant_center_peak_delta": float(
            float(keras_decoded.center_heatmap_peak_value) - float(variant_decoded.center_heatmap_peak_value)
        ),
        "keras_vs_variant_tip_peak_delta": float(
            float(keras_decoded.tip_heatmap_peak_value) - float(variant_decoded.tip_heatmap_peak_value)
        ),
        "keras_vs_variant_center_spread_delta_px": float(
            float(variant_guarded.quality_features.center_heatmap_spread_px)
            - float(keras_guarded.quality_features.center_heatmap_spread_px)
        ),
        "keras_vs_variant_tip_spread_delta_px": float(
            float(variant_guarded.quality_features.tip_heatmap_spread_px)
            - float(keras_guarded.quality_features.tip_heatmap_spread_px)
        ),
        "keras_vs_variant_confidence_delta": float(float(keras_decoded.confidence) - float(variant_decoded.confidence)),
        "guardrail_disagreement": bool(keras_guarded.status != variant_guarded.status),
        "rescued_from_reference": bool(
            reference_guarded.status == "rejected" and variant_guarded.status != "rejected"
        ),
    }
    return row


def _plot_variant_overlay(
    *,
    sample: Any,
    keras_decoded: Any,
    keras_guarded: Any,
    reference_decoded: Any,
    reference_guarded: Any,
    variant_decoded: Any,
    variant_guarded: Any,
    variant_name: str,
    output_path: Path,
) -> None:
    """Render a selected-variant overlay against Keras and the current reference."""

    fig = plt.figure(figsize=(18, 11), dpi=150)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.35, 1.0, 1.0), height_ratios=(1.0, 1.0))
    ax_crop = fig.add_subplot(grid[:, 0])
    ax_keras = fig.add_subplot(grid[0, 1])
    ax_reference = fig.add_subplot(grid[1, 1])
    ax_variant = fig.add_subplot(grid[:, 2])

    crop = np.asarray(sample.crop_image, dtype=np.float32)
    ax_crop.imshow(crop)
    ax_crop.scatter(
        [float(sample.metadata["center_x_224"]), float(keras_decoded.predicted_center_x_224), float(reference_decoded.predicted_center_x_224), float(variant_decoded.predicted_center_x_224)],
        [float(sample.metadata["center_y_224"]), float(keras_decoded.predicted_center_y_224), float(reference_decoded.predicted_center_y_224), float(variant_decoded.predicted_center_y_224)],
        c=["lime", "cyan", "orange", "yellow"],
        s=[70, 60, 60, 60],
        marker="o",
        edgecolors="white",
        linewidths=1.0,
    )
    ax_crop.scatter(
        [float(sample.metadata["tip_x_224"]), float(keras_decoded.predicted_tip_x_224), float(reference_decoded.predicted_tip_x_224), float(variant_decoded.predicted_tip_x_224)],
        [float(sample.metadata["tip_y_224"]), float(keras_decoded.predicted_tip_y_224), float(reference_decoded.predicted_tip_y_224), float(variant_decoded.predicted_tip_y_224)],
        c=["red", "deepskyblue", "orange", "yellow"],
        s=[70, 60, 60, 60],
        marker="x",
        linewidths=2.0,
    )
    ax_crop.plot(
        [float(sample.metadata["center_x_224"]), float(sample.metadata["tip_x_224"])],
        [float(sample.metadata["center_y_224"]), float(sample.metadata["tip_y_224"])],
        color="white",
        linewidth=2.0,
        alpha=0.9,
        label="true line",
    )
    ax_crop.plot(
        [float(keras_decoded.predicted_center_x_224), float(keras_decoded.predicted_tip_x_224)],
        [float(keras_decoded.predicted_center_y_224), float(keras_decoded.predicted_tip_y_224)],
        color="deepskyblue",
        linewidth=2.0,
        alpha=0.9,
        label="keras",
    )
    ax_crop.plot(
        [float(reference_decoded.predicted_center_x_224), float(reference_decoded.predicted_tip_x_224)],
        [float(reference_decoded.predicted_center_y_224), float(reference_decoded.predicted_tip_y_224)],
        color="orange",
        linewidth=2.0,
        alpha=0.9,
        label="current int8",
    )
    ax_crop.plot(
        [float(variant_decoded.predicted_center_x_224), float(variant_decoded.predicted_tip_x_224)],
        [float(variant_decoded.predicted_center_y_224), float(variant_decoded.predicted_tip_y_224)],
        color="yellow",
        linewidth=2.0,
        alpha=0.9,
        label=variant_name,
    )
    ax_crop.set_axis_off()
    ax_crop.set_title("Crop fed to the model")
    ax_crop.legend(loc="lower right", fontsize=8, framealpha=0.9)

    def _plot_heatmap(ax: plt.Axes, heatmap: np.ndarray, *, title: str, true_x: float, true_y: float, pred_x: float, pred_y: float) -> None:
        """Plot one heatmap with label markers."""

        ax.imshow(np.asarray(heatmap, dtype=np.float32), cmap="magma", origin="upper")
        ax.scatter([true_x * 55.0 / 223.0], [true_y * 55.0 / 223.0], c="white", s=40, marker="o", edgecolors="black", linewidths=0.8)
        ax.scatter([pred_x * 55.0 / 223.0], [pred_y * 55.0 / 223.0], c="cyan", s=50, marker="x", linewidths=2.0)
        ax.set_title(title)
        ax.set_xlim(-0.5, heatmap.shape[1] - 0.5)
        ax.set_ylim(heatmap.shape[0] - 0.5, -0.5)
        ax.set_xlabel("x / col")
        ax.set_ylabel("y / row")

    _plot_heatmap(
        ax_keras,
        keras_decoded.center_heatmap,
        title=f"Keras center\nstatus={keras_guarded.status}, peak={keras_decoded.center_heatmap_peak_value:.4f}",
        true_x=float(sample.metadata["center_x_224"]),
        true_y=float(sample.metadata["center_y_224"]),
        pred_x=float(keras_decoded.predicted_center_x_224),
        pred_y=float(keras_decoded.predicted_center_y_224),
    )
    _plot_heatmap(
        ax_reference,
        reference_decoded.tip_heatmap,
        title=f"Current int8 tip\nstatus={reference_guarded.status}, peak={reference_decoded.tip_heatmap_peak_value:.4f}",
        true_x=float(sample.metadata["tip_x_224"]),
        true_y=float(sample.metadata["tip_y_224"]),
        pred_x=float(reference_decoded.predicted_tip_x_224),
        pred_y=float(reference_decoded.predicted_tip_y_224),
    )
    _plot_heatmap(
        ax_variant,
        variant_decoded.tip_heatmap,
        title=f"{variant_name} tip\nstatus={variant_guarded.status}, peak={variant_decoded.tip_heatmap_peak_value:.4f}",
        true_x=float(sample.metadata["tip_x_224"]),
        true_y=float(sample.metadata["tip_y_224"]),
        pred_x=float(variant_decoded.predicted_tip_x_224),
        pred_y=float(variant_decoded.predicted_tip_y_224),
    )
    ax_variant.set_title(
        f"{variant_name}\n"
        f"Keras temp={float(keras_decoded.predicted_temperature_c_calibrated):.2f} C, "
        f"ref int8 temp={float(reference_decoded.predicted_temperature_c_calibrated):.2f} C, "
        f"variant temp={float(variant_decoded.predicted_temperature_c_calibrated):.2f} C",
        fontsize=10,
    )

    text_lines = [
        f"file: {Path(str(sample.metadata['image_path'])).name}",
        f"split: {sample.metadata['split']}",
        f"true temp: {float(sample.metadata['temperature_c']):.2f} C",
        f"keras status: {keras_guarded.status} ({';'.join(keras_guarded.rejection_reasons) or 'none'})",
        f"reference status: {reference_guarded.status} ({';'.join(reference_guarded.rejection_reasons) or 'none'})",
        f"variant status: {variant_guarded.status} ({';'.join(variant_guarded.rejection_reasons) or 'none'})",
        f"keras-vs-variant temp delta: {abs(float(keras_decoded.predicted_temperature_c_calibrated) - float(variant_decoded.predicted_temperature_c_calibrated)):.2f} C",
        f"keras-vs-variant tip delta: {math.hypot(float(keras_decoded.predicted_tip_x_224) - float(variant_decoded.predicted_tip_x_224), float(keras_decoded.predicted_tip_y_224) - float(variant_decoded.predicted_tip_y_224)):.2f} px",
        f"reference-vs-variant status: {reference_guarded.status} -> {variant_guarded.status}",
        f"reference-vs-variant rescue: {reference_guarded.status == 'rejected' and variant_guarded.status != 'rejected'}",
    ]
    fig.text(0.01, 0.01, "\n".join(text_lines), family="monospace", fontsize=9, va="bottom")
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Evaluate the exported TFLite variants and choose a selected candidate."""

    parser = argparse.ArgumentParser(description="Evaluate geometry heatmap v2 TFLite variants")
    parser.add_argument("--variant-root-dir", type=Path, default=DEFAULT_VARIANT_ROOT)
    parser.add_argument("--current-reference-model-path", type=Path, default=DEFAULT_CURRENT_REFERENCE_MODEL)
    parser.add_argument("--current-reference-contract-path", type=Path, default=DEFAULT_CURRENT_REFERENCE_CONTRACT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_TRAINED_MODEL)
    parser.add_argument("--manifest-path", type=Path, default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"))
    parser.add_argument("--calibration-json-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--decision-report-path", type=Path, default=DEFAULT_DECISION_REPORT_PATH)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--decode-method", type=str, default="softargmax")
    parser.add_argument("--decode-window-size", type=int, default=3)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--heatmap-size", type=int, default=DEFAULT_HEATMAP_SIZE)
    parser.add_argument("--sigma-pixels", type=float, default=5.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    variant_root = resolve_repo_path(repo_root, args.variant_root_dir)
    model_path = resolve_repo_path(repo_root, args.model_path)
    manifest_path = resolve_repo_path(repo_root, args.manifest_path)
    calibration_json_path = resolve_repo_path(repo_root, args.calibration_json_path)
    thresholds_path = resolve_repo_path(repo_root, args.thresholds_path)
    output_dir = resolve_repo_path(repo_root, args.output_dir)
    report_path = resolve_repo_path(repo_root, args.report_path)
    decision_report_path = resolve_repo_path(repo_root, args.decision_report_path)
    debug_dir = resolve_repo_path(repo_root, args.debug_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _load_thresholds(thresholds_path)
    calibration_candidate, calibration_json = load_selected_calibration_candidate(calibration_json_path)
    selected_decode_method, selected_decode_window_size = _load_selected_decode_method(
        output_dir,
        args.decode_method,
        args.decode_window_size,
    )
    baseline_prediction_path = output_dir / "decode_method_predictions.csv"
    baseline_summary_path = output_dir / "decode_method_summary.csv"
    baseline_rows = _load_baseline_decode_rows(baseline_prediction_path, selected_decode_method)

    variant_index_path = variant_root / "variant_index.csv"
    variant_index_rows: list[dict[str, Any]] = []
    with variant_index_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            variant_index_rows.append(row)

    successful_variants = [row for row in variant_index_rows if row["status"] == "success"]
    if not successful_variants:
        raise RuntimeError("No successful TFLite variants were exported.")

    split_cache: dict[str, SplitCache] = {}
    for split in SPLITS:
        print(f"[VARIANTS] Building split samples for {split}", flush=True)
        split_samples = load_split_samples(
            manifest_path,
            repo_root,
            split=split,
            mode=DEFAULT_PREPROCESSING_MODE,
            input_size=args.input_size,
            heatmap_size=args.heatmap_size,
            sigma_pixels=args.sigma_pixels,
        ).samples
        split_cache[split] = SplitCache(split=split, samples=split_samples)

    reference_split_summary: dict[str, dict[str, Any]] = {}
    baseline_rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (split, _image_path), row in baseline_rows.items():
        baseline_rows_by_split[split].append(row)
    for split in SPLITS:
        print(f"[VARIANTS] Loading baseline summary for {split}", flush=True)
        split_rows = baseline_rows_by_split.get(split, [])
        if not split_rows:
            raise RuntimeError(f"No baseline rows available for split {split!r} and decode method {selected_decode_method!r}.")
        reference_split_summary[split] = _baseline_reference_summary(split_rows)

    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    variant_rows_by_split: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for variant_row in successful_variants:
        variant_name = str(variant_row["variant_name"])
        variant_dir = variant_root / variant_name
        variant_model_path = variant_dir / "model.tflite"
        variant_contract_path = variant_dir / "tflite_tensor_contract.json"
        print(f"[VARIANTS] Loading {variant_name}", flush=True)
        variant_bundle = load_tflite_model(variant_model_path)
        variant_order = load_semantic_output_order_indices(variant_contract_path)

        for split, cache in split_cache.items():
            print(f"[VARIANTS] Evaluating {variant_name} on {split}", flush=True)
            variant_center, variant_tip, variant_confidence = predict_tflite_outputs(
                variant_bundle,
                [sample.crop_image for sample in cache.samples],
                semantic_output_order_indices=variant_order,
            )
            split_rows: list[dict[str, Any]] = []
            for index, sample in enumerate(cache.samples):
                baseline_row = baseline_rows[(split, str(sample.metadata["image_path"]))]
                keras_decoded, keras_guarded = _baseline_decoded_and_guarded(sample, baseline_row, prefix="keras")
                reference_decoded, reference_guarded = _baseline_decoded_and_guarded(sample, baseline_row, prefix="int8")
                variant_decoded, variant_guarded = _decode_prediction(
                    sample,
                    variant_center[index],
                    variant_tip[index],
                    float(np.ravel(variant_confidence[index])[0]),
                    calibration_candidate,
                    thresholds,
                    decode_method=selected_decode_method,
                    window_size=selected_decode_window_size,
                )
                row = _prediction_row(
                    sample=sample,
                    keras_decoded=keras_decoded,
                    keras_guarded=keras_guarded,
                    reference_decoded=reference_decoded,
                    reference_guarded=reference_guarded,
                    variant_name=variant_name,
                    variant_decoded=variant_decoded,
                    variant_guarded=variant_guarded,
                    decode_method=selected_decode_method,
                )
                split_rows.append(row)
                prediction_rows.append(row)
            variant_rows_by_split[(variant_name, split)] = split_rows
            summary = _summary(split_rows, label=f"{variant_name}:{split}")
            summary.update(
                {
                    "variant_name": variant_name,
                    "split": split,
                    "reference_current_int8_accepted_mae_c": float(reference_split_summary[split]["accepted_mae_c"]),
                    "reference_current_int8_acceptance_rate": float(reference_split_summary[split]["acceptance_rate"]),
                    "reference_current_int8_worst_accepted_error_c": float(reference_split_summary[split]["worst_accepted_error_c"]),
                    "reference_current_int8_guardrail_disagreement_count": int(reference_split_summary[split]["guardrail_disagreement_count"]),
                }
            )
            summary_rows.append(summary)

    _write_csv(prediction_rows, output_dir / "tflite_variant_predictions.csv")
    _write_csv(summary_rows, output_dir / "tflite_variant_replay_summary.csv")

    # Pick the best validation-safe variant.
    candidate_rows = [row for row in summary_rows if row["split"] == "val"]
    reference_val_disagreements = int(reference_split_summary["val"]["guardrail_disagreement_count"])
    passing_candidates = [
        row
        for row in candidate_rows
        if row["accepted_mae_c"] <= 4.5
        and row["acceptance_rate"] >= 0.65
        and row["worst_accepted_error_c"] < 20.0
        and row["accepted_gt20_failures"] == 0
        and row["keras_vs_variant_temperature_delta_mean"] <= 1.0
        and row["guardrail_disagreement_count"] <= max(0, reference_val_disagreements - 2)
    ]
    if passing_candidates:
        selected_variant_row = sorted(
            passing_candidates,
            key=lambda row: (
                _variant_priority(str(row["variant_name"])),
                float(row["keras_vs_variant_temperature_delta_mean"]),
                float(row["keras_vs_variant_tip_delta_mean"]),
                float(row["accepted_mae_c"]),
                float(row["worst_accepted_error_c"]),
                int(row["guardrail_disagreement_count"]),
            ),
        )[0]
    else:
        selected_variant_row = sorted(
            candidate_rows,
            key=lambda row: (
                _variant_priority(str(row["variant_name"])),
                float(row["keras_vs_variant_temperature_delta_mean"]),
                float(row["keras_vs_variant_tip_delta_mean"]),
                float(row["accepted_mae_c"]) if math.isfinite(float(row["accepted_mae_c"])) else math.inf,
                float(row["worst_accepted_error_c"]) if math.isfinite(float(row["worst_accepted_error_c"])) else math.inf,
                int(row["guardrail_disagreement_count"]),
            ),
        )[0]

    selected_variant_name = str(selected_variant_row["variant_name"])
    selected_variant_json = _json_safe_summary(selected_variant_row)
    selected_threshold_payload = {
        "selected_variant_name": selected_variant_name,
        "decode_method": selected_decode_method,
        "selection_split": "val",
        "selection_metrics": selected_variant_json,
    }
    _write_json(selected_threshold_payload, output_dir / "selected_variant.json")

    # Generate overlays for the selected variant on the test split.
    selected_rows = variant_rows_by_split[(selected_variant_name, "test")]
    selected_rows_sorted = sorted(selected_rows, key=lambda row: float(row["keras_vs_variant_tip_delta_px"]), reverse=True)
    selected_cache = split_cache["test"]
    selected_variant_dir = variant_root / selected_variant_name
    selected_variant_bundle = load_tflite_model(selected_variant_dir / "model.tflite")
    selected_variant_order = load_semantic_output_order_indices(selected_variant_dir / "tflite_tensor_contract.json")
    selected_variant_center, selected_variant_tip, selected_variant_conf = predict_tflite_outputs(
        selected_variant_bundle,
        [sample.crop_image for sample in selected_cache.samples],
        semantic_output_order_indices=selected_variant_order,
    )
    # Build per-sample decoded bundles for overlay generation.
    selected_overlay_results: dict[str, tuple[Any, Any, Any, Any, Any, Any]] = {}
    for index, sample in enumerate(selected_cache.samples):
        baseline_row = baseline_rows[("test", str(sample.metadata["image_path"]))]
        keras_decoded, keras_guarded = _baseline_decoded_and_guarded(sample, baseline_row, prefix="keras")
        reference_decoded, reference_guarded = _baseline_decoded_and_guarded(sample, baseline_row, prefix="int8")
        variant_decoded, variant_guarded = _decode_prediction(
            sample,
            selected_variant_center[index],
            selected_variant_tip[index],
            float(np.ravel(selected_variant_conf[index])[0]),
            calibration_candidate,
            thresholds,
            decode_method=selected_decode_method,
            window_size=selected_decode_window_size,
        )
        selected_overlay_results[str(sample.metadata["image_path"])] = (
            sample,
            keras_decoded,
            keras_guarded,
            reference_decoded,
            reference_guarded,
            variant_decoded,
            variant_guarded,
        )

    top_tip_rows = selected_rows_sorted[:30]
    top_temp_rows = sorted(selected_rows, key=lambda row: float(row["keras_vs_variant_temperature_delta_c"]), reverse=True)[:30]
    disagreement_rows = [row for row in selected_rows if row["guardrail_disagreement"]]
    rescue_rows = [row for row in selected_rows if row["rescued_from_reference"]]

    def _emit_overlay_rows(rows: list[dict[str, Any]], subdir: str) -> None:
        """Render overlays for one selected row subset."""

        for row in rows:
            sample, keras_decoded, keras_guarded, reference_decoded, reference_guarded, variant_decoded, variant_guarded = selected_overlay_results[str(row["image_path"])]
            output_path = debug_dir / selected_variant_name / subdir / f"{Path(str(row['image_path'])).stem}.png"
            _plot_variant_overlay(
                sample=sample,
                keras_decoded=keras_decoded,
                keras_guarded=keras_guarded,
                reference_decoded=reference_decoded,
                reference_guarded=reference_guarded,
                variant_decoded=variant_decoded,
                variant_guarded=variant_guarded,
                variant_name=selected_variant_name,
                output_path=output_path,
            )

    _emit_overlay_rows(top_tip_rows, "top_tip")
    _emit_overlay_rows(top_temp_rows, "top_temp")
    _emit_overlay_rows(disagreement_rows, "disagreements")
    _emit_overlay_rows(rescue_rows, "rescues")

    # Produce the markdown report.
    lines: list[str] = [
        "# Geometry Heatmap v2 TFLite Variant Comparison",
        "",
        "## Decode Method",
        "",
        f"- Decode method used for this comparison: `{selected_decode_method}` with window size `{selected_decode_window_size}`",
        "",
        "## Selected Variant",
        "",
        f"- Selected variant: `{selected_variant_name}`",
        f"- Selected on validation with accepted MAE `{float(selected_variant_row['accepted_mae_c']):.4f} C`",
        f"- Validation acceptance rate `{float(selected_variant_row['acceptance_rate']):.4f}`",
        f"- Validation worst accepted error `{float(selected_variant_row['worst_accepted_error_c']):.4f} C`",
        f"- Validation Keras-vs-variant temp delta mean `{float(selected_variant_row['keras_vs_variant_temperature_delta_mean']):.4f} C`",
        f"- Validation Keras-vs-variant tip delta mean `{float(selected_variant_row['keras_vs_variant_tip_delta_mean']):.4f} px`",
        "",
        "## Current Reference Current INT8",
        "",
    ]
    # Compare against the current champion export.
    # The reference summary is recorded in each candidate row, so report it directly from the selected variant's validation row.
    lines.extend(
        [
            f"- Current reference accepted MAE on validation (from the selected variant table): `{float(selected_variant_row['reference_current_int8_accepted_mae_c']):.4f} C`",
            f"- Current reference acceptance rate on validation: `{float(selected_variant_row['reference_current_int8_acceptance_rate']):.4f}`",
            f"- Current reference worst accepted error on validation: `{float(selected_variant_row['reference_current_int8_worst_accepted_error_c']):.4f} C`",
            f"- Current reference guardrail disagreements on validation: `{int(selected_variant_row['reference_current_int8_guardrail_disagreement_count'])}`",
            "",
            "## Variant Summary",
            "",
        ]
    )

    for split in SPLITS:
        split_rows = [row for row in summary_rows if row["split"] == split]
        lines.extend(
            [
                f"### {split}",
                "",
                "| variant | accepted MAE | acceptance | worst accepted | Keras vs variant temp delta mean | Keras vs variant tip delta mean | disagreement count |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(split_rows, key=lambda item: (float(item["accepted_mae_c"]), -float(item["acceptance_rate"]))):
            lines.append(
                f"| {row['variant_name']} | {float(row['accepted_mae_c']):.4f} | {float(row['acceptance_rate']):.4f} | {float(row['worst_accepted_error_c']):.4f} | {float(row['keras_vs_variant_temperature_delta_mean']):.4f} | {float(row['keras_vs_variant_tip_delta_mean']):.4f} | {int(row['guardrail_disagreement_count'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Test Split Selected Variant",
            "",
            "| metric | value |",
            "| --- | ---: |",
        ]
    )
    test_selected_rows = [row for row in summary_rows if row["variant_name"] == selected_variant_name and row["split"] == "test"]
    if not test_selected_rows:
        raise RuntimeError("Could not locate selected variant test summary.")
    test_selected = test_selected_rows[0]
    for key, label in [
        ("accepted_mae_c", "accepted MAE"),
        ("acceptance_rate", "acceptance rate"),
        ("worst_accepted_error_c", "worst accepted error"),
        ("accepted_gt20_failures", "accepted >20 C failures"),
        ("percentage_under_2c", "% under 2 C"),
        ("percentage_under_5c", "% under 5 C"),
        ("percentage_under_10c", "% under 10 C"),
        ("keras_vs_variant_temperature_delta_mean", "Keras vs variant temp delta mean"),
        ("keras_vs_variant_temperature_delta_median", "Keras vs variant temp delta median"),
        ("keras_vs_variant_temperature_delta_p90", "Keras vs variant temp delta p90"),
        ("keras_vs_variant_tip_delta_mean", "Keras vs variant tip delta mean"),
        ("keras_vs_variant_tip_delta_median", "Keras vs variant tip delta median"),
        ("keras_vs_variant_tip_delta_p90", "Keras vs variant tip delta p90"),
        ("keras_vs_variant_center_peak_delta_mean", "Keras vs variant center peak delta mean"),
        ("keras_vs_variant_center_peak_delta_median", "Keras vs variant center peak delta median"),
        ("keras_vs_variant_center_peak_delta_p90", "Keras vs variant center peak delta p90"),
        ("keras_vs_variant_tip_peak_delta_mean", "Keras vs variant tip peak delta mean"),
        ("keras_vs_variant_tip_peak_delta_median", "Keras vs variant tip peak delta median"),
        ("keras_vs_variant_tip_peak_delta_p90", "Keras vs variant tip peak delta p90"),
        ("keras_vs_variant_center_spread_delta_px_mean", "Keras vs variant center spread delta mean"),
        ("keras_vs_variant_center_spread_delta_px_median", "Keras vs variant center spread delta median"),
        ("keras_vs_variant_center_spread_delta_px_p90", "Keras vs variant center spread delta p90"),
        ("keras_vs_variant_tip_spread_delta_px_mean", "Keras vs variant tip spread delta mean"),
        ("keras_vs_variant_tip_spread_delta_px_median", "Keras vs variant tip spread delta median"),
        ("keras_vs_variant_tip_spread_delta_px_p90", "Keras vs variant tip spread delta p90"),
        ("guardrail_disagreement_count", "guardrail disagreements"),
        ("reference_current_int8_guardrail_disagreement_count", "reference guardrail disagreements"),
    ]:
        lines.append(f"| {label} | {float(test_selected[key]):.4f} |" if isinstance(test_selected[key], float) else f"| {label} | {test_selected[key]} |")

    lines.extend(
        [
            "",
            "## Decision Inputs",
            "",
            f"- The selected variant passes the board gate on test? {float(test_selected['accepted_mae_c']) <= 4.5 and float(test_selected['acceptance_rate']) >= 0.65 and float(test_selected['worst_accepted_error_c']) < 20.0 and int(test_selected['accepted_gt20_failures']) == 0}",
            f"- Keras-vs-selected temperature delta mean <= 1.0 C? {float(test_selected['keras_vs_variant_temperature_delta_mean']) <= 1.0}",
            f"- Keras-vs-selected tip delta mean <= 8 px? {float(test_selected['keras_vs_variant_tip_delta_mean']) <= 8.0}",
            f"- Guardrail disagreement count on test: {int(test_selected['guardrail_disagreement_count'])}",
            f"- Reference current INT8 disagreement count on test: {int(test_selected['reference_current_int8_guardrail_disagreement_count'])}",
            "",
            "## Overlay Sets",
            "",
            f"- Top 30 tip deltas: `{debug_dir / selected_variant_name / 'top_tip'}`",
            f"- Top 30 temperature deltas: `{debug_dir / selected_variant_name / 'top_temp'}`",
            f"- Guardrail disagreements: `{debug_dir / selected_variant_name / 'disagreements'}`",
            f"- Cases rescued from the reference int8 path: `{debug_dir / selected_variant_name / 'rescues'}`",
            "",
        ]
    )

    _write_json(selected_variant_json, output_dir / "selected_variant_summary.json")
    _write_csv(summary_rows, output_dir / "tflite_variant_replay_summary.csv")
    _write_csv(prediction_rows, output_dir / "tflite_variant_predictions.csv")
    _write_json(
        {
            "selected_variant_name": selected_variant_name,
            "selected_variant_summary": selected_variant_json,
            "decode_method": selected_decode_method,
            "decode_window_size": selected_decode_window_size,
            "board_gate_pass": bool(
                float(test_selected["accepted_mae_c"]) <= 4.5
                and float(test_selected["acceptance_rate"]) >= 0.65
                and float(test_selected["worst_accepted_error_c"]) < 20.0
                and int(test_selected["accepted_gt20_failures"]) == 0
                and float(test_selected["keras_vs_variant_temperature_delta_mean"]) <= 1.0
            ),
        },
        output_dir / "selected_variant.json",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    decode_rows: list[dict[str, Any]] = []
    if baseline_summary_path.exists():
        with baseline_summary_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            decode_rows = list(reader)

    def _decode_summary_row(method_name: str, split_name: str) -> dict[str, Any] | None:
        """Find one decode-method summary row by method and split."""

        for row in decode_rows:
            if row["decode_method"] == method_name and row["split"] == split_name:
                return row
        return None

    softargmax_val_row = _decode_summary_row("softargmax_w3", "val")
    selected_decode_val_row = _decode_summary_row(selected_decode_method, "val")
    decode_method_reduced_drift = False
    decode_method_notes = "Decode comparison summary was unavailable."
    if softargmax_val_row is not None and selected_decode_val_row is not None:
        decode_method_reduced_drift = (
            float(selected_decode_val_row["keras_vs_int8_temperature_delta_mean"]) <= float(softargmax_val_row["keras_vs_int8_temperature_delta_mean"])
            and float(selected_decode_val_row["keras_vs_int8_tip_delta_mean"]) <= float(softargmax_val_row["keras_vs_int8_tip_delta_mean"])
            and float(selected_decode_val_row["keras_acceptance_rate"]) >= float(softargmax_val_row["keras_acceptance_rate"])
        )
        decode_method_notes = (
            f"Selected decode method `{selected_decode_method}` on validation has "
            f"temp delta mean {float(selected_decode_val_row['keras_vs_int8_temperature_delta_mean']):.4f} C "
            f"vs softargmax_w3 {float(softargmax_val_row['keras_vs_int8_temperature_delta_mean']):.4f} C."
        )

    validation_reference = reference_split_summary["val"]
    test_reference = reference_split_summary["test"]
    selected_variant_contract_path = variant_root / selected_variant_name / "tflite_tensor_contract.json"
    selected_variant_cubeai_friendly = not selected_variant_name.startswith("variant_d_")
    selected_variant_gate_pass = bool(
        float(test_selected["accepted_mae_c"]) <= 4.5
        and float(test_selected["acceptance_rate"]) >= 0.65
        and float(test_selected["worst_accepted_error_c"]) < 20.0
        and int(test_selected["accepted_gt20_failures"]) == 0
        and float(test_selected["keras_vs_variant_temperature_delta_mean"]) <= 1.0
    )
    selected_variant_guardrail_acceptable = bool(
        int(test_selected["guardrail_disagreement_count"]) <= int(test_reference["guardrail_disagreement_count"])
    )

    float_io_rows = [
        row
        for row in candidate_rows
        if row["variant_name"].startswith(("variant_b_", "variant_c_"))
    ]
    best_float_io_row = (
        sorted(
            float_io_rows,
            key=lambda row: (
                float(row["keras_vs_variant_temperature_delta_mean"]),
                float(row["keras_vs_variant_tip_delta_mean"]),
                float(row["accepted_mae_c"]) if math.isfinite(float(row["accepted_mae_c"])) else math.inf,
                int(row["guardrail_disagreement_count"]),
            ),
        )[0]
        if float_io_rows
        else None
    )
    best_float_io_reduced_drift = bool(
        best_float_io_row is not None
        and float(best_float_io_row["keras_vs_variant_temperature_delta_mean"]) <= float(validation_reference["keras_vs_reference_temperature_delta_mean"])
        and float(best_float_io_row["keras_vs_variant_tip_delta_mean"]) <= float(validation_reference["keras_vs_reference_tip_delta_mean"])
    )
    representative_data_reduced_drift = bool(
        selected_variant_name.startswith("variant_a_")
        and float(selected_variant_row["keras_vs_variant_temperature_delta_mean"]) <= float(validation_reference["keras_vs_reference_temperature_delta_mean"])
        and float(selected_variant_row["keras_vs_variant_tip_delta_mean"]) <= float(validation_reference["keras_vs_reference_tip_delta_mean"])
    )
    selected_variant_selected_by_decode = decode_method_reduced_drift and selected_decode_method != "softargmax_w3"
    proceed_to_cubeai = bool(
        selected_variant_gate_pass
        and selected_variant_cubeai_friendly
        and selected_variant_guardrail_acceptable
        and selected_variant_contract_path.exists()
    )

    decision_lines: list[str] = [
        "# Geometry Heatmap v2 Quantization Readiness Decision",
        "",
        "## Decision",
        "",
        f"- Selected variant: `{selected_variant_name}`",
        f"- Selected decode method: `{selected_decode_method}` with window size `{selected_decode_window_size}`",
        f"- Cube.AI-friendly export family? `{selected_variant_cubeai_friendly}`",
        f"- Selected variant test gate passed? `{selected_variant_gate_pass}`",
        f"- Guardrail disagreements acceptable on test? `{selected_variant_guardrail_acceptable}`",
        f"- Tensor contract documented? `{selected_variant_contract_path.exists()}`",
        f"- Allowed to proceed to Cube.AI packaging? `{proceed_to_cubeai}`",
        "",
        "## Root Cause",
        "",
        "- The drift is dominated by tip heatmap flattening and a wider tip heatmap spread after INT8 quantization.",
        "- Softargmax is sensitive to that flattened tip mass, so small quantization changes move the decoded tip more than the center.",
        "- The raw TFLite FP32 path already matched Keras in Phase 7, so output order and dequantization are not the primary cause.",
        "",
        "## What Helped",
        "",
        f"- Improved representative data reduced drift? `{representative_data_reduced_drift}`",
        f"- Float input/output reduced drift? `{best_float_io_reduced_drift}`",
        f"- Different decode method reduced drift? `{decode_method_reduced_drift}`",
        f"- Selected decode method was helpful enough to recommend? `{selected_variant_selected_by_decode}`",
        "",
        "## Selected Variant Metrics",
        "",
        f"- Validation accepted MAE: `{float(selected_variant_row['accepted_mae_c']):.4f} C`",
        f"- Validation acceptance rate: `{float(selected_variant_row['acceptance_rate']):.4f}`",
        f"- Validation worst accepted error: `{float(selected_variant_row['worst_accepted_error_c']):.4f} C`",
        f"- Validation Keras-vs-variant temp delta mean: `{float(selected_variant_row['keras_vs_variant_temperature_delta_mean']):.4f} C`",
        f"- Validation Keras-vs-variant tip delta mean: `{float(selected_variant_row['keras_vs_variant_tip_delta_mean']):.4f} px`",
        f"- Test accepted MAE: `{float(test_selected['accepted_mae_c']):.4f} C`",
        f"- Test acceptance rate: `{float(test_selected['acceptance_rate']):.4f}`",
        f"- Test worst accepted error: `{float(test_selected['worst_accepted_error_c']):.4f} C`",
        f"- Test Keras-vs-variant temp delta mean: `{float(test_selected['keras_vs_variant_temperature_delta_mean']):.4f} C`",
        f"- Test Keras-vs-variant tip delta mean: `{float(test_selected['keras_vs_variant_tip_delta_mean']):.4f} px`",
        f"- Test guardrail disagreements: `{int(test_selected['guardrail_disagreement_count'])}`",
        f"- Selected variant contract path: `{selected_variant_contract_path}`",
        "",
        "## Notes",
        "",
        decode_method_notes,
    ]

    if best_float_io_row is not None:
        decision_lines.extend(
            [
                "",
                "## Best Float-IO Candidate",
                "",
                f"- Best float-IO candidate on validation: `{best_float_io_row['variant_name']}`",
                f"- Validation Keras-vs-variant temp delta mean: `{float(best_float_io_row['keras_vs_variant_temperature_delta_mean']):.4f} C`",
                f"- Validation Keras-vs-variant tip delta mean: `{float(best_float_io_row['keras_vs_variant_tip_delta_mean']):.4f} px`",
                f"- Validation accepted MAE: `{float(best_float_io_row['accepted_mae_c']):.4f} C`",
            ]
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    decision_report_path.parent.mkdir(parents=True, exist_ok=True)
    decision_report_path.write_text("\n".join(decision_lines), encoding="utf-8")
    print(f"[VARIANTS] Wrote prediction CSV to {output_dir / 'tflite_variant_predictions.csv'}", flush=True)
    print(f"[VARIANTS] Wrote summary CSV to {output_dir / 'tflite_variant_replay_summary.csv'}", flush=True)
    print(f"[VARIANTS] Wrote report to {report_path}", flush=True)
    print(f"[VARIANTS] Wrote decision report to {decision_report_path}", flush=True)


if __name__ == "__main__":
    main()
