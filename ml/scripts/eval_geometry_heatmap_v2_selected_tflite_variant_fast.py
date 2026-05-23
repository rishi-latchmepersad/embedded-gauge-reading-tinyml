#!/usr/bin/env python3
"""Fast replay for the selected geometry_heatmap_v2 TFLite candidate variant.

This script intentionally avoids the broad all-variant replay path. It reuses
cached baseline decode rows for Keras and the current INT8 model when available
and only runs fresh inference for one selected candidate variant.
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
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np

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
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import load_tflite_model, summarize_tflite_contract
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import load_selected_calibration_candidate
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import GeometryGuardrailThresholds


SPLITS = ("train", "val", "test")
DEFAULT_VARIANT_ROOT = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2")
DEFAULT_BASELINE_REPLAY_PREDICTIONS_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/tflite_replay_predictions.csv")
DEFAULT_KERAS_REPLAY_PREDICTIONS_PATH = DEFAULT_BASELINE_REPLAY_PREDICTIONS_PATH
DEFAULT_CURRENT_INT8_REPLAY_PREDICTIONS_PATH = DEFAULT_BASELINE_REPLAY_PREDICTIONS_PATH
DEFAULT_SELECTED_DECODE_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json")
DEFAULT_VARIANT_INDEX_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/variant_index.csv")
DEFAULT_CALIBRATION_PATH = Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json")
DEFAULT_THRESHOLDS_PATH = Path("ml/artifacts/training/geometry_heatmap_v2_board_replay/selected_board_guardrail_thresholds.json")
DEFAULT_OUTPUT_DIR = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2")
DEFAULT_REPORT_PATH = Path("ml/reports/geometry_heatmap_v2_selected_variant_fast_replay.md")
DEFAULT_PREDICTIONS_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_variant_fast_predictions.csv")

VARIANT_PRIORITY = {
    "variant_b_float_io_internal_int8": 0,
    "variant_c_int8_input_float_output": 1,
    "variant_a_full_int8": 2,
    "variant_d_dynamic_range": 3,
}

FULL_INT8_STRATEGY_PRIORITY = {
    "identity_mild_medium": 0,
    "identity_mild": 1,
    "identity": 2,
    "stratified": 3,
}


@dataclass(frozen=True)
class SelectedVariantSpec:
    """Metadata describing the single candidate variant to evaluate."""

    variant_name: str
    variant_dir: Path
    model_path: Path
    contract_path: Path
    representative_strategy: str
    representative_dataset_count: int


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a deterministic CSV file."""

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


def _status_is_accepted(status: str) -> bool:
    """Treat accepted and clamped predictions as usable."""

    return status in {"accepted", "clamped"}


def _load_selected_decode_spec(selection_path: Path) -> tuple[str, str, int]:
    """Load the selected decode spec and split it into method plus window."""

    if not selection_path.exists():
        raise RuntimeError(f"Missing selected decode selection artifact: {selection_path}")
    with selection_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "decode_method" in payload and "window_size" in payload:
        decode_method = str(payload["decode_method"])
        selected_window_size = int(payload["window_size"])
        selected_name = f"{decode_method}_w{selected_window_size}"
    else:
        selected_name = str(payload["selected_decode_method"])
        selected_window_size = int(payload["selected_window_size"])
        if "_w" in selected_name:
            decode_method, window_suffix = selected_name.rsplit("_w", 1)
            try:
                selected_window_size = int(window_suffix)
            except ValueError:
                pass
        else:
            decode_method = selected_name
    if decode_method != "softargmax" or int(selected_window_size) != 3:
        raise RuntimeError(
            f"Expected corrected decode softargmax w3, found {selected_name}/{decode_method} w{selected_window_size} in {selection_path}."
        )
    return selected_name, decode_method, selected_window_size


def _load_cached_rows(
    predictions_path: Path,
    *,
    selected_decode_path: Path,
    selected_decode_name: str,
    selected_decode_method: str,
    selected_decode_window_size: int,
    calibration_path: Path,
    thresholds_path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load cached replay rows keyed by split and image path."""

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    required_columns = {
        "decode_name",
        "decode_method",
        "decode_window_size",
        "selected_decode_path",
        "preprocessing_mode",
        "resize_method",
        "channel_strategy",
        "normalization",
        "calibration_path",
        "guardrail_thresholds_path",
        "model_variant_name",
        "keras_guardrail_status",
        "tflite_int8_guardrail_status",
        "keras_predicted_center_x_224",
        "keras_predicted_center_y_224",
        "keras_predicted_tip_x_224",
        "keras_predicted_tip_y_224",
        "tflite_int8_predicted_center_x_224",
        "tflite_int8_predicted_center_y_224",
        "tflite_int8_predicted_tip_x_224",
        "tflite_int8_predicted_tip_y_224",
    }
    with predictions_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            missing = sorted(required_columns.difference(row))
            if missing:
                raise RuntimeError(
                    f"Cached replay rows in {predictions_path} are missing provenance columns: {', '.join(missing)}"
                )
            if str(row["decode_name"]) != selected_decode_name or str(row["decode_method"]) != selected_decode_method:
                raise RuntimeError(
                    f"Cached replay rows in {predictions_path} were generated with decode {row['decode_name']}/{row['decode_method']}, "
                    f"not the selected decode {selected_decode_name}/{selected_decode_method}."
                )
            if int(float(row["decode_window_size"])) != int(selected_decode_window_size):
                raise RuntimeError(
                    f"Cached replay rows in {predictions_path} were generated with window size {row['decode_window_size']}, "
                    f"not the selected decode window {selected_decode_window_size}."
                )
            if str(row["selected_decode_path"]) != str(selected_decode_path):
                raise RuntimeError(
                    f"Cached replay rows in {predictions_path} were generated with selected decode path {row['selected_decode_path']}, "
                    f"not the corrected decode artifact {selected_decode_path}."
                )
            if str(row["preprocessing_mode"]) != "python_training_rgb_bilinear":
                raise RuntimeError(
                    f"Cached replay rows in {predictions_path} were generated with preprocessing {row['preprocessing_mode']}, "
                    "not the selected RGB bilinear board contract."
                )
            if str(row["resize_method"]) != "rgb_bilinear" or str(row["channel_strategy"]) != "rgb":
                raise RuntimeError(
                    f"Cached replay rows in {predictions_path} were generated with crop/resize provenance "
                    f"{row['resize_method']}/{row['channel_strategy']}, which does not match the selected board contract."
                )
            if str(row["normalization"]) != "uint8_to_float32_0_1":
                raise RuntimeError(
                    f"Cached replay rows in {predictions_path} were generated with normalization {row['normalization']}, "
                    "which does not match the selected board contract."
                )
            if str(row["calibration_path"]) != str(calibration_path):
                raise RuntimeError(
                    f"Cached replay rows in {predictions_path} were generated with calibration {row['calibration_path']}, "
                    f"not the selected calibration artifact {calibration_path}."
                )
            if str(row["guardrail_thresholds_path"]) != str(thresholds_path):
                raise RuntimeError(
                    f"Cached replay rows in {predictions_path} were generated with guardrails {row['guardrail_thresholds_path']}, "
                    f"not the selected thresholds artifact {thresholds_path}."
                )
            rows_by_key[(row["split"], row["image_path"])] = row
    if not rows_by_key:
        raise RuntimeError(f"No cached replay rows found in {predictions_path}.")
    return rows_by_key


def _load_variant_index(variant_index_path: Path) -> list[dict[str, Any]]:
    """Load the exported-variant index table."""

    rows: list[dict[str, Any]] = []
    with variant_index_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    if not rows:
        raise RuntimeError(f"No variant rows found in {variant_index_path}.")
    return rows


def _select_candidate_variant(variant_rows: list[dict[str, Any]], variant_name: str | None = None) -> SelectedVariantSpec:
    """Pick one candidate variant using the documented priority order."""

    successful_rows = [row for row in variant_rows if str(row["status"]) == "success"]
    if variant_name is not None:
        match = next((row for row in successful_rows if row["variant_name"] == variant_name), None)
        if match is None:
            raise RuntimeError(f"Requested variant {variant_name!r} was not exported successfully.")
        return SelectedVariantSpec(
            variant_name=str(match["variant_name"]),
            variant_dir=Path(str(match["variant_dir"])),
            model_path=Path(str(match["model_path"])),
            contract_path=Path(str(match["tensor_contract_path"])),
            representative_strategy=str(match["representative_strategy"]),
            representative_dataset_count=int(float(match["representative_dataset_count"])),
        )

    def _build(row: dict[str, Any]) -> SelectedVariantSpec:
        return SelectedVariantSpec(
            variant_name=str(row["variant_name"]),
            variant_dir=Path(str(row["variant_dir"])),
            model_path=Path(str(row["model_path"])),
            contract_path=Path(str(row["tensor_contract_path"])),
            representative_strategy=str(row["representative_strategy"]),
            representative_dataset_count=int(float(row["representative_dataset_count"])),
        )

    for preferred_name in ("variant_b_float_io_internal_int8", "variant_c_int8_input_float_output"):
        match = next((row for row in successful_rows if row["variant_name"] == preferred_name), None)
        if match is not None:
            return _build(match)

    full_int8_rows = [row for row in successful_rows if str(row["variant_name"]).startswith("variant_a_full_int8")]
    if full_int8_rows:
        best_full_int8 = sorted(
            full_int8_rows,
            key=lambda row: (
                -int(float(row["representative_dataset_count"])),
                FULL_INT8_STRATEGY_PRIORITY.get(str(row["representative_strategy"]), len(FULL_INT8_STRATEGY_PRIORITY)),
                str(row["variant_name"]),
            ),
        )[0]
        return _build(best_full_int8)

    dynamic_range_row = next((row for row in successful_rows if row["variant_name"] == "variant_d_dynamic_range"), None)
    if dynamic_range_row is not None:
        return _build(dynamic_range_row)

    raise RuntimeError("No candidate variants were available.")


def _decode_keras_cached_row(sample: Any, cached_row: dict[str, Any]) -> tuple[Any, Any]:
    """Rebuild a lightweight Keras decoded/guarded bundle from the replay cache."""

    decoded = SimpleNamespace(
        true_temperature_c=float(cached_row["keras_true_temperature_c"]),
        true_angle_degrees=float(cached_row["keras_true_angle_degrees"]),
        true_center_x_224=float(sample.metadata["center_x_224"]),
        true_center_y_224=float(sample.metadata["center_y_224"]),
        true_tip_x_224=float(sample.metadata["tip_x_224"]),
        true_tip_y_224=float(sample.metadata["tip_y_224"]),
        predicted_center_x_224=float(cached_row["keras_predicted_center_x_224"]),
        predicted_center_y_224=float(cached_row["keras_predicted_center_y_224"]),
        predicted_tip_x_224=float(cached_row["keras_predicted_tip_x_224"]),
        predicted_tip_y_224=float(cached_row["keras_predicted_tip_y_224"]),
        predicted_angle_degrees=float(cached_row["keras_predicted_angle_degrees"]),
        predicted_temperature_c_current_mapping=float(cached_row["keras_predicted_temperature_c_current_mapping"]),
        predicted_temperature_c_calibrated=float(cached_row["keras_predicted_temperature_c_calibrated"]),
        absolute_error_c_current_mapping=float(cached_row["keras_absolute_error_c_current_mapping"]),
        absolute_error_c_calibrated=float(cached_row["keras_absolute_error_c_calibrated"]),
        center_heatmap_peak_value=float(cached_row["keras_center_heatmap_peak_value"]),
        tip_heatmap_peak_value=float(cached_row["keras_tip_heatmap_peak_value"]),
        center_heatmap_entropy=float(cached_row["keras_center_heatmap_entropy"]),
        tip_heatmap_entropy=float(cached_row["keras_tip_heatmap_entropy"]),
        center_heatmap_spread_px=float(cached_row["keras_center_heatmap_spread_px"]),
        tip_heatmap_spread_px=float(cached_row["keras_tip_heatmap_spread_px"]),
        confidence=float(cached_row["keras_confidence"]),
    )
    guarded = SimpleNamespace(
        status=str(cached_row["keras_guardrail_status"]),
        rejection_reasons=[]
        if str(cached_row["keras_rejection_reasons"]) in {"", "none"}
        else str(cached_row["keras_rejection_reasons"]).split(";"),
        temperature_c=float(cached_row["keras_guarded_temperature_c"]),
        quality_features=SimpleNamespace(
            center_heatmap_entropy=float(cached_row["keras_center_heatmap_entropy"]),
            tip_heatmap_entropy=float(cached_row["keras_tip_heatmap_entropy"]),
            center_heatmap_spread_px=float(cached_row["keras_center_heatmap_spread_px"]),
            tip_heatmap_spread_px=float(cached_row["keras_tip_heatmap_spread_px"]),
        ),
    )
    return decoded, guarded


def _decode_int8_cached_row(sample: Any, cached_row: dict[str, Any]) -> tuple[Any, Any]:
    """Rebuild a lightweight current-INT8 decoded/guarded bundle from the replay cache."""

    decoded = SimpleNamespace(
        true_temperature_c=float(cached_row["tflite_int8_true_temperature_c"]),
        true_angle_degrees=float(cached_row["tflite_int8_true_angle_degrees"]),
        true_center_x_224=float(sample.metadata["center_x_224"]),
        true_center_y_224=float(sample.metadata["center_y_224"]),
        true_tip_x_224=float(sample.metadata["tip_x_224"]),
        true_tip_y_224=float(sample.metadata["tip_y_224"]),
        predicted_center_x_224=float(cached_row["tflite_int8_predicted_center_x_224"]),
        predicted_center_y_224=float(cached_row["tflite_int8_predicted_center_y_224"]),
        predicted_tip_x_224=float(cached_row["tflite_int8_predicted_tip_x_224"]),
        predicted_tip_y_224=float(cached_row["tflite_int8_predicted_tip_y_224"]),
        predicted_angle_degrees=float(cached_row["tflite_int8_predicted_angle_degrees"]),
        predicted_temperature_c_current_mapping=float(cached_row["tflite_int8_predicted_temperature_c_current_mapping"]),
        predicted_temperature_c_calibrated=float(cached_row["tflite_int8_predicted_temperature_c_calibrated"]),
        absolute_error_c_current_mapping=float(cached_row["tflite_int8_absolute_error_c_current_mapping"]),
        absolute_error_c_calibrated=float(cached_row["tflite_int8_absolute_error_c_calibrated"]),
        center_heatmap_peak_value=float(cached_row["tflite_int8_center_heatmap_peak_value"]),
        tip_heatmap_peak_value=float(cached_row["tflite_int8_tip_heatmap_peak_value"]),
        center_heatmap_entropy=float(cached_row["tflite_int8_center_heatmap_entropy"]),
        tip_heatmap_entropy=float(cached_row["tflite_int8_tip_heatmap_entropy"]),
        center_heatmap_spread_px=float(cached_row["tflite_int8_center_heatmap_spread_px"]),
        tip_heatmap_spread_px=float(cached_row["tflite_int8_tip_heatmap_spread_px"]),
        confidence=float(cached_row["tflite_int8_confidence"]),
    )
    guarded = SimpleNamespace(
        status=str(cached_row["tflite_int8_guardrail_status"]),
        rejection_reasons=[]
        if str(cached_row["tflite_int8_rejection_reasons"]) in {"", "none"}
        else str(cached_row["tflite_int8_rejection_reasons"]).split(";"),
        temperature_c=float(cached_row["tflite_int8_guarded_temperature_c"]),
        quality_features=SimpleNamespace(
            center_heatmap_entropy=float(cached_row["tflite_int8_center_heatmap_entropy"]),
            tip_heatmap_entropy=float(cached_row["tflite_int8_tip_heatmap_entropy"]),
            center_heatmap_spread_px=float(cached_row["tflite_int8_center_heatmap_spread_px"]),
            tip_heatmap_spread_px=float(cached_row["tflite_int8_tip_heatmap_spread_px"]),
        ),
    )
    return decoded, guarded


def _decode_candidate_row(
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
    """Decode the selected candidate variant and apply guardrails."""

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


def _prediction_row(
    *,
    sample: Any,
    keras_decoded: Any,
    keras_guarded: Any,
    int8_decoded: Any,
    int8_guarded: Any,
    candidate_decoded: Any,
    candidate_guarded: Any,
    candidate_variant_name: str,
    selected_decode_name: str,
    selected_decode_method: str,
    selected_decode_window_size: int,
    calibration_path: Path,
    thresholds_path: Path,
) -> dict[str, Any]:
    """Flatten one sample into a stable CSV row."""

    return {
        "variant_name": candidate_variant_name,
        "selected_decode_path": str(DEFAULT_SELECTED_DECODE_PATH),
        "split": str(sample.metadata["split"]),
        "image_path": str(sample.metadata["image_path"]),
        "source_manifest": str(sample.metadata.get("source_manifest", "")),
        "quality_flag": str(sample.metadata.get("quality_flag", "")),
        "preprocessing_mode": str(sample.metadata.get("preprocessing_mode", "")),
        "decode_name": selected_decode_name,
        "decode_method": selected_decode_method,
        "decode_window_size": int(selected_decode_window_size),
        "calibration_path": str(calibration_path),
        "guardrail_thresholds_path": str(thresholds_path),
        "model_variant_name": candidate_variant_name,
        "selected_decode_name": selected_decode_name,
        "selected_decode_method": selected_decode_method,
        "selected_decode_window_size": int(selected_decode_window_size),
        "resize_method": str(sample.metadata.get("resize_method", "")),
        "channel_strategy": str(sample.metadata.get("channel_strategy", "")),
        "normalization": str(sample.metadata.get("normalization", "")),
        "true_temperature_c": float(sample.metadata["temperature_c"]),
        "true_angle_degrees": float(keras_decoded.true_angle_degrees),
        "true_center_x_224": float(sample.metadata["center_x_224"]),
        "true_center_y_224": float(sample.metadata["center_y_224"]),
        "true_tip_x_224": float(sample.metadata["tip_x_224"]),
        "true_tip_y_224": float(sample.metadata["tip_y_224"]),
        "keras_guardrail_status": str(keras_guarded.status),
        "int8_guardrail_status": str(int8_guarded.status),
        "candidate_guardrail_status": str(candidate_guarded.status),
        "keras_rejection_reasons": ";".join(keras_guarded.rejection_reasons) if keras_guarded.rejection_reasons else "none",
        "int8_rejection_reasons": ";".join(int8_guarded.rejection_reasons) if int8_guarded.rejection_reasons else "none",
        "candidate_rejection_reasons": ";".join(candidate_guarded.rejection_reasons) if candidate_guarded.rejection_reasons else "none",
        "keras_predicted_temperature_c_calibrated": float(keras_decoded.predicted_temperature_c_calibrated),
        "int8_predicted_temperature_c_calibrated": float(int8_decoded.predicted_temperature_c_calibrated),
        "candidate_predicted_temperature_c_calibrated": float(candidate_decoded.predicted_temperature_c_calibrated),
        "keras_guarded_temperature_c": float(keras_guarded.temperature_c),
        "int8_guarded_temperature_c": float(int8_guarded.temperature_c),
        "candidate_guarded_temperature_c": float(candidate_guarded.temperature_c),
        "keras_vs_candidate_temperature_delta_c": float(
            abs(float(keras_decoded.predicted_temperature_c_calibrated) - float(candidate_decoded.predicted_temperature_c_calibrated))
        ),
        "int8_vs_candidate_temperature_delta_c": float(
            abs(float(int8_decoded.predicted_temperature_c_calibrated) - float(candidate_decoded.predicted_temperature_c_calibrated))
        ),
        "keras_vs_candidate_center_delta_px": float(
            math.hypot(
                float(keras_decoded.predicted_center_x_224) - float(candidate_decoded.predicted_center_x_224),
                float(keras_decoded.predicted_center_y_224) - float(candidate_decoded.predicted_center_y_224),
            )
        ),
        "keras_vs_candidate_tip_delta_px": float(
            math.hypot(
                float(keras_decoded.predicted_tip_x_224) - float(candidate_decoded.predicted_tip_x_224),
                float(keras_decoded.predicted_tip_y_224) - float(candidate_decoded.predicted_tip_y_224),
            )
        ),
        "keras_vs_candidate_angle_delta_degrees": float(
            abs(float(keras_decoded.predicted_angle_degrees) - float(candidate_decoded.predicted_angle_degrees))
        ),
        "candidate_center_heatmap_peak_value": float(candidate_decoded.center_heatmap_peak_value),
        "candidate_tip_heatmap_peak_value": float(candidate_decoded.tip_heatmap_peak_value),
        "candidate_center_heatmap_entropy": float(candidate_guarded.quality_features.center_heatmap_entropy),
        "candidate_tip_heatmap_entropy": float(candidate_guarded.quality_features.tip_heatmap_entropy),
        "candidate_center_heatmap_spread_px": float(candidate_guarded.quality_features.center_heatmap_spread_px),
        "candidate_tip_heatmap_spread_px": float(candidate_guarded.quality_features.tip_heatmap_spread_px),
        "candidate_confidence": float(candidate_decoded.confidence),
        "keras_confidence": float(keras_decoded.confidence),
        "int8_confidence": float(int8_decoded.confidence),
        "guardrail_disagreement": bool(keras_guarded.status != candidate_guarded.status or int8_guarded.status != candidate_guarded.status),
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one split's worth of candidate predictions."""

    accepted_errors = np.asarray(
        [
            abs(float(row["candidate_guarded_temperature_c"]) - float(row["true_temperature_c"]))
            for row in rows
            if row["candidate_guardrail_status"] != "rejected"
        ],
        dtype=np.float64,
    )
    keras_temp_deltas = np.asarray([float(row["keras_vs_candidate_temperature_delta_c"]) for row in rows], dtype=np.float64)
    keras_center_deltas = np.asarray([float(row["keras_vs_candidate_center_delta_px"]) for row in rows], dtype=np.float64)
    keras_tip_deltas = np.asarray([float(row["keras_vs_candidate_tip_delta_px"]) for row in rows], dtype=np.float64)
    disagreement_count = int(sum(1 for row in rows if row["guardrail_disagreement"]))
    gt20_failures = int(
        sum(
            1
            for row in rows
            if row["candidate_guardrail_status"] != "rejected"
            and abs(float(row["candidate_guarded_temperature_c"]) - float(row["true_temperature_c"])) > 20.0
        )
    )
    return {
        "count": int(len(rows)),
        "accepted_count": int(sum(_status_is_accepted(str(row["candidate_guardrail_status"])) for row in rows)),
        "acceptance_rate": float(np.mean([_status_is_accepted(str(row["candidate_guardrail_status"])) for row in rows])),
        "accepted_mae_c": float(np.mean(accepted_errors)) if accepted_errors.size else math.nan,
        "worst_accepted_error_c": float(np.max(accepted_errors)) if accepted_errors.size else math.nan,
        "accepted_gt20_failures": gt20_failures,
        "keras_vs_candidate_temperature_delta_mean": float(np.mean(keras_temp_deltas)),
        "keras_vs_candidate_temperature_delta_median": float(np.median(keras_temp_deltas)),
        "keras_vs_candidate_temperature_delta_p90": float(np.percentile(keras_temp_deltas, 90)),
        "keras_vs_candidate_center_delta_mean": float(np.mean(keras_center_deltas)),
        "keras_vs_candidate_center_delta_median": float(np.median(keras_center_deltas)),
        "keras_vs_candidate_center_delta_p90": float(np.percentile(keras_center_deltas, 90)),
        "keras_vs_candidate_tip_delta_mean": float(np.mean(keras_tip_deltas)),
        "keras_vs_candidate_tip_delta_median": float(np.median(keras_tip_deltas)),
        "keras_vs_candidate_tip_delta_p90": float(np.percentile(keras_tip_deltas, 90)),
        "guardrail_disagreement_count": disagreement_count,
        "top_rejection_reasons": Counter(
            reason
            for row in rows
            if row["candidate_guardrail_status"] == "rejected"
            for reason in str(row["candidate_rejection_reasons"]).split(";")
        ),
    }


def _format_contract_summary(contract: dict[str, Any]) -> list[str]:
    """Turn a summarized TFLite contract into report lines."""

    output_lines: list[str] = []
    input_contract = contract["input"]
    output_lines.append(
        f"- Input: dtype `{input_contract['dtype']}`, shape `{input_contract['shape']}`, quantized `{input_contract['quantized']}`"
    )
    output_lines.append(
        f"- Input quantization: scale `{input_contract['quantization']['scale']}`, zero_point `{input_contract['quantization']['zero_point']}`"
    )
    output_lines.append(f"- Requires output dequantization? `{contract['requires_dequantization']}`")
    output_lines.append("- Outputs:")
    for index, output in enumerate(contract["outputs"]):
        output_lines.append(
            f"  - `{index}`: name `{output['name']}`, dtype `{output['dtype']}`, shape `{output['shape']}`, quantized `{output['quantized']}`"
        )
    return output_lines


def main() -> None:
    """Evaluate one selected candidate variant on one split."""

    parser = argparse.ArgumentParser(description="Fast evaluation of one geometry_heatmap_v2 TFLite variant")
    parser.add_argument("--variant-root-dir", type=Path, default=DEFAULT_VARIANT_ROOT)
    parser.add_argument("--variant-name", type=str, default=None)
    parser.add_argument("--split", type=str, choices=SPLITS, default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--partial-write-every", type=int, default=10)
    parser.add_argument("--keras-replay-predictions-path", type=Path, default=DEFAULT_KERAS_REPLAY_PREDICTIONS_PATH)
    parser.add_argument("--current-int8-replay-predictions-path", type=Path, default=DEFAULT_CURRENT_INT8_REPLAY_PREDICTIONS_PATH)
    parser.add_argument("--selected-decode-path", type=Path, default=DEFAULT_SELECTED_DECODE_PATH)
    parser.add_argument("--variant-index-path", type=Path, default=DEFAULT_VARIANT_INDEX_PATH)
    parser.add_argument("--manifest-path", type=Path, default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"))
    parser.add_argument("--calibration-json-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--heatmap-size", type=int, default=DEFAULT_HEATMAP_SIZE)
    parser.add_argument("--sigma-pixels", type=float, default=5.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    variant_root = resolve_repo_path(repo_root, args.variant_root_dir)
    variant_index_path = resolve_repo_path(repo_root, args.variant_index_path)
    keras_replay_predictions_path = resolve_repo_path(repo_root, args.keras_replay_predictions_path)
    current_int8_replay_predictions_path = resolve_repo_path(repo_root, args.current_int8_replay_predictions_path)
    selected_decode_path = resolve_repo_path(repo_root, args.selected_decode_path)
    manifest_path = resolve_repo_path(repo_root, args.manifest_path)
    calibration_json_path = resolve_repo_path(repo_root, args.calibration_json_path)
    thresholds_path = resolve_repo_path(repo_root, args.thresholds_path)
    output_dir = resolve_repo_path(repo_root, args.output_dir)
    report_path = resolve_repo_path(repo_root, args.report_path)
    predictions_path = resolve_repo_path(repo_root, args.predictions_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_decode_name, selected_decode_method, selected_decode_window_size = _load_selected_decode_spec(selected_decode_path)
    variant_rows = _load_variant_index(variant_index_path)
    candidate_spec = _select_candidate_variant(variant_rows, args.variant_name)
    keras_replay_rows = _load_cached_rows(
        keras_replay_predictions_path,
        selected_decode_path=selected_decode_path,
        selected_decode_name=selected_decode_name,
        selected_decode_method=selected_decode_method,
        selected_decode_window_size=selected_decode_window_size,
        calibration_path=calibration_json_path,
        thresholds_path=thresholds_path,
    )
    current_int8_replay_rows = _load_cached_rows(
        current_int8_replay_predictions_path,
        selected_decode_path=selected_decode_path,
        selected_decode_name=selected_decode_name,
        selected_decode_method=selected_decode_method,
        selected_decode_window_size=selected_decode_window_size,
        calibration_path=calibration_json_path,
        thresholds_path=thresholds_path,
    )

    print(f"[FAST] Selected decode: {selected_decode_name} -> {selected_decode_method} w{selected_decode_window_size}", flush=True)
    print(f"[FAST] Candidate variant: {candidate_spec.variant_name}", flush=True)
    print(f"[FAST] Cached Keras replay rows: {len(keras_replay_rows)}", flush=True)
    print(f"[FAST] Cached current INT8 replay rows: {len(current_int8_replay_rows)}", flush=True)

    thresholds = _load_thresholds(thresholds_path)
    calibration_candidate, calibration_json = load_selected_calibration_candidate(calibration_json_path)
    _ = calibration_json  # Keep the loaded calibration payload available for future debugging.

    split_samples = load_split_samples(
        manifest_path,
        repo_root,
        split=args.split,
        mode=DEFAULT_PREPROCESSING_MODE,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
    ).samples
    if args.max_samples is not None:
        split_samples = split_samples[: int(args.max_samples)]
    if not split_samples:
        raise RuntimeError(f"No samples available for split {args.split!r}.")

    print(f"[FAST] Split {args.split}: {len(split_samples)} samples", flush=True)
    selected_split_keras_rows = {
        image_path: row for (split_name, image_path), row in keras_replay_rows.items() if split_name == args.split
    }
    selected_split_int8_rows = {
        image_path: row for (split_name, image_path), row in current_int8_replay_rows.items() if split_name == args.split
    }
    if len(selected_split_keras_rows) < len(split_samples):
        raise RuntimeError(f"Missing cached Keras replay rows for split {args.split!r} in {keras_replay_predictions_path}.")
    if len(selected_split_int8_rows) < len(split_samples):
        raise RuntimeError(f"Missing cached current INT8 replay rows for split {args.split!r} in {current_int8_replay_predictions_path}.")

    candidate_model_path = resolve_repo_path(repo_root, candidate_spec.model_path)
    candidate_contract_path = resolve_repo_path(repo_root, candidate_spec.contract_path)
    print(f"[FAST] Loading selected candidate model: {candidate_model_path}", flush=True)
    selected_variant_bundle = load_tflite_model(candidate_model_path)
    semantic_output_order_indices = load_semantic_output_order_indices(candidate_contract_path)
    contract_summary = summarize_tflite_contract(candidate_model_path)

    rows: list[dict[str, Any]] = []
    split_inputs = [sample.crop_image for sample in split_samples]
    print("[FAST] Running cached baseline replay and candidate inference", flush=True)
    candidate_center, candidate_tip, candidate_confidence = predict_tflite_outputs(
        selected_variant_bundle,
        split_inputs,
        semantic_output_order_indices=semantic_output_order_indices,
    )

    for index, sample in enumerate(split_samples, start=1):
        image_path = str(sample.metadata["image_path"])
        keras_baseline_row = selected_split_keras_rows.get(image_path)
        if keras_baseline_row is None:
            raise RuntimeError(f"Missing cached Keras replay row for {image_path!r}.")
        current_int8_baseline_row = selected_split_int8_rows.get(image_path)
        if current_int8_baseline_row is None:
            raise RuntimeError(f"Missing cached current INT8 replay row for {image_path!r}.")

        keras_decoded, keras_guarded = _decode_keras_cached_row(sample, keras_baseline_row)
        int8_decoded, int8_guarded = _decode_int8_cached_row(sample, current_int8_baseline_row)
        candidate_decoded, candidate_guarded = _decode_candidate_row(
            sample,
            candidate_center[index - 1],
            candidate_tip[index - 1],
            float(np.ravel(candidate_confidence[index - 1])[0]),
            calibration_candidate,
            thresholds,
            decode_method=selected_decode_method,
            window_size=selected_decode_window_size,
        )

        rows.append(
            _prediction_row(
                sample=sample,
                keras_decoded=keras_decoded,
                keras_guarded=keras_guarded,
                int8_decoded=int8_decoded,
                int8_guarded=int8_guarded,
                candidate_decoded=candidate_decoded,
                candidate_guarded=candidate_guarded,
                candidate_variant_name=candidate_spec.variant_name,
                selected_decode_name=selected_decode_name,
                selected_decode_method=selected_decode_method,
                selected_decode_window_size=selected_decode_window_size,
                calibration_path=calibration_json_path,
                thresholds_path=thresholds_path,
            )
        )

        if args.partial_write_every > 0 and (index % args.partial_write_every == 0 or index == len(split_samples)):
            _write_csv(rows, predictions_path)
            print(f"[FAST] Wrote {len(rows)} rows to {predictions_path}", flush=True)

    summary = _summarize_rows(rows)
    selected_rows = [row for row in rows if row["candidate_guardrail_status"] != "rejected"]
    selected_errors = [abs(float(row["candidate_guarded_temperature_c"]) - float(row["true_temperature_c"])) for row in selected_rows]
    selected_mae = float(np.mean(selected_errors)) if selected_errors else math.nan
    selected_acceptance_rate = float(np.mean([_status_is_accepted(str(row["candidate_guardrail_status"])) for row in rows]))
    selected_worst = float(np.max(selected_errors)) if selected_errors else math.nan
    selected_gt20_failures = int(summary["accepted_gt20_failures"])

    report_lines = [
        "# Geometry Heatmap v2 Selected Variant Fast Replay",
        "",
        "## Selection",
        "",
        f"- Selected decode method: `{selected_decode_name}`",
        f"- Decoded method: `{selected_decode_method}`",
        f"- Window size: `{selected_decode_window_size}`",
        f"- Selected variant: `{candidate_spec.variant_name}`",
        f"- Split: `{args.split}`",
        f"- Sample count: `{len(rows)}`",
        f"- Max samples: `{args.max_samples if args.max_samples is not None else 'none'}`",
        "",
        "## Contract",
        "",
        *_format_contract_summary(contract_summary),
        "",
        "## Metrics",
        "",
        f"- Candidate accepted MAE: `{selected_mae:.4f} C`",
        f"- Candidate acceptance rate: `{selected_acceptance_rate:.4f}`",
        f"- Candidate worst accepted error: `{selected_worst:.4f} C`",
        f"- Candidate accepted >20 C failures: `{selected_gt20_failures}`",
        f"- Keras-vs-candidate temperature delta mean: `{summary['keras_vs_candidate_temperature_delta_mean']:.4f} C`",
        f"- Keras-vs-candidate temperature delta median: `{summary['keras_vs_candidate_temperature_delta_median']:.4f} C`",
        f"- Keras-vs-candidate center delta mean: `{summary['keras_vs_candidate_center_delta_mean']:.4f} px`",
        f"- Keras-vs-candidate center delta median: `{summary['keras_vs_candidate_center_delta_median']:.4f} px`",
        f"- Keras-vs-candidate tip delta mean: `{summary['keras_vs_candidate_tip_delta_mean']:.4f} px`",
        f"- Keras-vs-candidate tip delta median: `{summary['keras_vs_candidate_tip_delta_median']:.4f} px`",
        f"- Guardrail disagreements: `{summary['guardrail_disagreement_count']}`",
        "",
        "## Baseline Context",
        "",
        f"- Cached Keras acceptance rate: `{float(np.mean([_status_is_accepted(str(selected_split_keras_rows[row['image_path']]['keras_guardrail_status'])) for row in rows])):.4f}`",
        f"- Cached current INT8 acceptance rate: `{float(np.mean([_status_is_accepted(str(selected_split_int8_rows[row['image_path']]['tflite_int8_guardrail_status'])) for row in rows])):.4f}`",
        "",
        "## Notes",
        "",
        "This fast path uses cached replay rows for Keras and the current INT8 baseline and only executes fresh inference for the selected candidate variant.",
    ]

    _write_csv(rows, predictions_path)
    _write_json(
        {
            "selected_decode_name": selected_decode_name,
            "selected_decode_method": selected_decode_method,
            "selected_decode_window_size": selected_decode_window_size,
            "selected_variant_name": candidate_spec.variant_name,
            "split": args.split,
            "sample_count": len(rows),
            "candidate_contract": contract_summary,
            "candidate_metrics": summary,
            "candidate_accepted_mae_c": selected_mae,
            "candidate_acceptance_rate": selected_acceptance_rate,
            "candidate_worst_accepted_error_c": selected_worst,
            "candidate_accepted_gt20_failures": selected_gt20_failures,
        },
        output_dir / "selected_variant_fast_summary.json",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[FAST] Wrote predictions to {predictions_path}", flush=True)
    print(f"[FAST] Wrote report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
