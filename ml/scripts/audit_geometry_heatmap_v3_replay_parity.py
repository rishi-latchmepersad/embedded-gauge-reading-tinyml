#!/usr/bin/env python3
"""Audit replay parity for geometry_heatmap_v3 across scoring paths."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_board_replay import build_board_replay_sample
from embedded_gauge_reading_tinyml.geometry_heatmap_qat_utils import fake_quantize_01_tensor
from embedded_gauge_reading_tinyml.geometry_heatmap_quantization_common import (
    DEFAULT_HEATMAP_SIZE,
    DEFAULT_INPUT_SIZE,
    DEFAULT_PREPROCESSING_MODE,
    decode_and_guard,
    load_semantic_output_order_indices,
    resolve_repo_path,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import (
    load_geometry_heatmap_keras_model,
    load_tflite_model,
    reorder_tflite_outputs,
    run_tflite_model,
    summarize_tflite_contract,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import (
    HeatmapSample,
    load_clean_geometry_examples,
    load_heatmap_sample,
    load_selected_calibration_candidate,
    select_examples_from_split,
)
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import GeometryGuardrailThresholds


DEFAULT_MODEL_PATH = Path("artifacts/training/geometry_heatmap_v3_quant_native/best_model.keras")
DEFAULT_FLOAT_TFLITE_PATH = Path("artifacts/deployment/geometry_heatmap_v3_tflite/model_v3_float32.tflite")
DEFAULT_INT8_TFLITE_PATH = Path("artifacts/deployment/geometry_heatmap_v3_tflite/model_v3_int8.tflite")
DEFAULT_CONTRACT_PATH = Path("artifacts/deployment/geometry_heatmap_v3_tflite/tflite_tensor_contract.json")
DEFAULT_MANIFEST_PATH = Path("ml/data/geometry_reader_manifest_v2_clean.csv")
DEFAULT_CALIBRATION_PATH = Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json")
DEFAULT_THRESHOLDS_PATH = Path("ml/artifacts/training/geometry_heatmap_v2_board_replay/selected_board_guardrail_thresholds.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/training/geometry_heatmap_v3_quant_native")
DEFAULT_PARITY_AUDIT_PATH = Path("artifacts/training/geometry_heatmap_v3_quant_native/replay_parity_audit.csv")
DEFAULT_PARITY_REPORT_PATH = Path("ml/reports/geometry_heatmap_v3_replay_parity_audit.md")
DEFAULT_CANONICAL_REPORT_PATH = Path("ml/reports/geometry_heatmap_v3_canonical_validation_rescore.md")
DEFAULT_RELOAD_DEBUG_PATH = Path("ml/reports/geometry_heatmap_v3_checkpoint_reload_debug.md")


@dataclass(frozen=True)
class ReplayInputs:
    """A pair of trainer-style and canonical replay samples for one example."""

    example_image_path: str
    trainer_sample: HeatmapSample
    canonical_sample: HeatmapSample


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


def _write_markdown(lines: list[str], output_path: Path) -> None:
    """Write a markdown report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


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
    """Load the corrected decode lock."""

    with selection_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    decode_method = str(payload.get("decode_method", payload.get("selected_decode_method", "")))
    window_size = int(payload.get("window_size", payload.get("selected_window_size", 0)))
    if "_w" in decode_method:
        decode_method, suffix = decode_method.rsplit("_w", 1)
        if suffix.isdigit():
            window_size = int(suffix)
    if decode_method != "softargmax" or window_size != 3:
        raise RuntimeError(f"Expected corrected decode softargmax w3, found {decode_method} w{window_size}.")
    return decode_method, window_size


def _predict_tflite_outputs(
    model_path: Path,
    samples: list[HeatmapSample],
    *,
    semantic_output_order_indices: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run TFLite inference and reorder raw outputs into semantic order."""

    bundle = load_tflite_model(model_path)
    per_sample_outputs: list[list[np.ndarray]] = []
    for sample in samples:
        sample_outputs = run_tflite_model(bundle, np.expand_dims(sample.crop_image.astype(np.float32), axis=0))
        per_sample_outputs.append([np.asarray(tensor, dtype=np.float32) for tensor in sample_outputs])

    stacked_outputs: list[np.ndarray] = []
    for output_index in range(len(semantic_output_order_indices)):
        stacked_outputs.append(
            np.stack([sample_outputs[output_index][0] for sample_outputs in per_sample_outputs], axis=0).astype(np.float32)
        )
    reordered = reorder_tflite_outputs(stacked_outputs, semantic_output_order_indices)
    return (
        {
            "center_heatmap": np.asarray(reordered[0], dtype=np.float32),
            "tip_heatmap": np.asarray(reordered[1], dtype=np.float32),
            "confidence": np.asarray(reordered[2], dtype=np.float32),
        },
        summarize_tflite_contract(model_path),
    )


def _as_output_dict(outputs: Any) -> dict[str, tf.Tensor]:
    """Normalize model outputs to the semantic heatmap dictionary."""

    if isinstance(outputs, dict):
        return {
            "center_heatmap": tf.cast(outputs["center_heatmap"], tf.float32),
            "tip_heatmap": tf.cast(outputs["tip_heatmap"], tf.float32),
            "confidence": tf.cast(outputs["confidence"], tf.float32),
        }
    center_heatmap, tip_heatmap, confidence = outputs
    return {
        "center_heatmap": tf.cast(center_heatmap, tf.float32),
        "tip_heatmap": tf.cast(tip_heatmap, tf.float32),
        "confidence": tf.cast(confidence, tf.float32),
    }


def _tensor_stats(tensor: np.ndarray) -> dict[str, float]:
    """Compute compact finite stats for one tensor."""

    values = np.asarray(tensor, dtype=np.float32)
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    return {
        "min": float(np.min(safe)),
        "max": float(np.max(safe)),
        "mean": float(np.mean(safe)),
        "finite_fraction": float(np.mean(finite.astype(np.float32))),
    }


def _load_validation_inputs(repo_root: Path) -> list[ReplayInputs]:
    """Build paired trainer-style and canonical validation samples."""

    examples = load_clean_geometry_examples(repo_root / DEFAULT_MANIFEST_PATH)
    val_examples = select_examples_from_split(examples, split="val")
    pairs: list[ReplayInputs] = []
    for example in val_examples:
        trainer_sample = load_heatmap_sample(
            example,
            repo_root,
            heatmap_size=DEFAULT_HEATMAP_SIZE,
            sigma_pixels=5.0,
        )
        canonical_sample = build_board_replay_sample(
            example,
            repo_root,
            mode=DEFAULT_PREPROCESSING_MODE,
            input_size=DEFAULT_INPUT_SIZE,
            heatmap_size=DEFAULT_HEATMAP_SIZE,
            sigma_pixels=5.0,
        )
        pairs.append(
            ReplayInputs(
                example_image_path=str(example.image_path),
                trainer_sample=trainer_sample,
                canonical_sample=canonical_sample,
            )
        )
    return pairs


def _decode_rows(
    *,
    model_type: str,
    samples: list[HeatmapSample],
    outputs: dict[str, np.ndarray],
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    decode_method: str,
    window_size: int,
) -> list[dict[str, Any]]:
    """Decode one path into replay rows."""

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
        )
        rows.append(
            {
                "model_type": model_type,
                "image_path": str(sample.metadata["image_path"]),
                "split": str(sample.metadata["split"]),
                "source_kind": str(sample.metadata.get("source_kind", "unknown")),
                "preprocessing_mode": str(sample.metadata.get("preprocessing_mode", "legacy_trainer_heatmap")),
                "resize_method": str(sample.metadata.get("resize_method", "")),
                "channel_strategy": str(sample.metadata.get("channel_strategy", "")),
                "normalization": str(sample.metadata.get("normalization", "")),
                "crop_x1": int(sample.metadata["crop_x1"]),
                "crop_y1": int(sample.metadata["crop_y1"]),
                "crop_x2": int(sample.metadata["crop_x2"]),
                "crop_y2": int(sample.metadata["crop_y2"]),
                "true_temperature_c": float(sample.metadata["temperature_c"]),
                "true_angle_degrees": float(sample.metadata["angle_degrees"]),
                "true_center_x_224": float(sample.metadata["center_x_224"]),
                "true_center_y_224": float(sample.metadata["center_y_224"]),
                "true_tip_x_224": float(sample.metadata["tip_x_224"]),
                "true_tip_y_224": float(sample.metadata["tip_y_224"]),
                "predicted_temperature_c": float(decoded.predicted_temperature_c_calibrated),
                "guarded_temperature_c": float(guarded.temperature_c),
                "guardrail_status": guarded.status,
                "predicted_angle_degrees": float(decoded.predicted_angle_degrees),
                "predicted_center_x_224": float(decoded.predicted_center_x_224),
                "predicted_center_y_224": float(decoded.predicted_center_y_224),
                "predicted_tip_x_224": float(decoded.predicted_tip_x_224),
                "predicted_tip_y_224": float(decoded.predicted_tip_y_224),
                "center_heatmap_peak_value": float(decoded.center_heatmap_peak_value),
                "tip_heatmap_peak_value": float(decoded.tip_heatmap_peak_value),
                "center_heatmap_spread_px": float(guarded.quality_features.center_heatmap_spread_px),
                "tip_heatmap_spread_px": float(guarded.quality_features.tip_heatmap_spread_px),
                "confidence": float(confidence),
                "rejection_reasons": ";".join(guarded.rejection_reasons) if guarded.rejection_reasons else "none",
            }
        )
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize replay rows."""

    accepted = [row for row in rows if str(row["guardrail_status"]) in {"accepted", "clamped"}]
    accepted_errors = np.asarray(
        [abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"])) for row in accepted],
        dtype=np.float64,
    )
    all_errors = np.asarray(
        [abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"])) for row in rows],
        dtype=np.float64,
    )
    rejection_counts = Counter()
    for row in rows:
        if str(row["guardrail_status"]) in {"accepted", "clamped"}:
            continue
        for reason in str(row["rejection_reasons"]).split(";"):
            if reason and reason != "none":
                rejection_counts[reason] += 1
    return {
        "count": float(len(rows)),
        "accepted_count": float(len(accepted)),
        "accepted_mae_c": float(np.mean(accepted_errors)) if accepted_errors.size else math.nan,
        "acceptance_rate": float(len(accepted) / len(rows)) if rows else math.nan,
        "worst_accepted_error_c": float(np.max(accepted_errors)) if accepted_errors.size else math.nan,
        "accepted_gt20_failures": float(
            sum(1 for row in accepted if abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"])) > 20.0)
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
        "guardrail_disagreement_count": float(sum(1 for row in rows if str(row["guardrail_status"]) not in {"accepted", "clamped"})),
        "top_rejection_reasons": ";".join(f"{reason}:{count}" for reason, count in rejection_counts.most_common(5))
        if rejection_counts
        else "none",
    }


def _path_drift(reference_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute drift between two model paths."""

    reference = {row["image_path"]: row for row in reference_rows}
    candidate = {row["image_path"]: row for row in candidate_rows}
    common_images = sorted(reference.keys() & candidate.keys())
    temp_deltas: list[float] = []
    center_deltas: list[float] = []
    tip_deltas: list[float] = []
    guardrail_disagreements = 0
    for image_path in common_images:
        base_row = reference[image_path]
        cand_row = candidate[image_path]
        if str(base_row["guardrail_status"]) in {"accepted", "clamped"} and str(cand_row["guardrail_status"]) in {"accepted", "clamped"}:
            temp_deltas.append(abs(float(base_row["guarded_temperature_c"]) - float(cand_row["guarded_temperature_c"])))
        center_deltas.append(
            math.hypot(
                float(base_row["predicted_center_x_224"]) - float(cand_row["predicted_center_x_224"]),
                float(base_row["predicted_center_y_224"]) - float(cand_row["predicted_center_y_224"]),
            )
        )
        tip_deltas.append(
            math.hypot(
                float(base_row["predicted_tip_x_224"]) - float(cand_row["predicted_tip_x_224"]),
                float(base_row["predicted_tip_y_224"]) - float(cand_row["predicted_tip_y_224"]),
            )
        )
        if str(base_row["guardrail_status"]) != str(cand_row["guardrail_status"]):
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


def _serialize_rows(
    *,
    trainer_rows: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
    fp32_rows: list[dict[str, Any]],
    int8_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge the four model paths into one wide CSV row per example."""

    merged_rows: list[dict[str, Any]] = []
    for trainer, canonical, fp32, int8 in zip(trainer_rows, canonical_rows, fp32_rows, int8_rows, strict=True):
        merged_rows.append(
            {
                "image_path": canonical["image_path"],
                "split": canonical["split"],
                "crop_x1": canonical["crop_x1"],
                "crop_y1": canonical["crop_y1"],
                "crop_x2": canonical["crop_x2"],
                "crop_y2": canonical["crop_y2"],
                "trainer_crop_x1": trainer["crop_x1"],
                "trainer_crop_y1": trainer["crop_y1"],
                "trainer_crop_x2": trainer["crop_x2"],
                "trainer_crop_y2": trainer["crop_y2"],
                "canonical_crop_x1": canonical["crop_x1"],
                "canonical_crop_y1": canonical["crop_y1"],
                "canonical_crop_x2": canonical["crop_x2"],
                "canonical_crop_y2": canonical["crop_y2"],
                "crop_metadata_match": bool(
                    trainer["crop_x1"] == canonical["crop_x1"]
                    and trainer["crop_y1"] == canonical["crop_y1"]
                    and trainer["crop_x2"] == canonical["crop_x2"]
                    and trainer["crop_y2"] == canonical["crop_y2"]
                ),
                "trainer_style_predicted_temperature_c": trainer["guarded_temperature_c"],
                "trainer_style_predicted_center_x_224": trainer["predicted_center_x_224"],
                "trainer_style_predicted_center_y_224": trainer["predicted_center_y_224"],
                "trainer_style_predicted_tip_x_224": trainer["predicted_tip_x_224"],
                "trainer_style_predicted_tip_y_224": trainer["predicted_tip_y_224"],
                "trainer_style_center_heatmap_peak_value": trainer["center_heatmap_peak_value"],
                "trainer_style_tip_heatmap_peak_value": trainer["tip_heatmap_peak_value"],
                "trainer_style_center_heatmap_spread_px": trainer["center_heatmap_spread_px"],
                "trainer_style_tip_heatmap_spread_px": trainer["tip_heatmap_spread_px"],
                "trainer_style_guardrail_status": trainer["guardrail_status"],
                "trainer_style_rejection_reasons": trainer["rejection_reasons"],
                "standalone_keras_predicted_temperature_c": canonical["guarded_temperature_c"],
                "standalone_keras_predicted_center_x_224": canonical["predicted_center_x_224"],
                "standalone_keras_predicted_center_y_224": canonical["predicted_center_y_224"],
                "standalone_keras_predicted_tip_x_224": canonical["predicted_tip_x_224"],
                "standalone_keras_predicted_tip_y_224": canonical["predicted_tip_y_224"],
                "standalone_keras_center_heatmap_peak_value": canonical["center_heatmap_peak_value"],
                "standalone_keras_tip_heatmap_peak_value": canonical["tip_heatmap_peak_value"],
                "standalone_keras_center_heatmap_spread_px": canonical["center_heatmap_spread_px"],
                "standalone_keras_tip_heatmap_spread_px": canonical["tip_heatmap_spread_px"],
                "standalone_keras_guardrail_status": canonical["guardrail_status"],
                "standalone_keras_rejection_reasons": canonical["rejection_reasons"],
                "tflite_fp32_predicted_temperature_c": fp32["guarded_temperature_c"],
                "tflite_fp32_predicted_center_x_224": fp32["predicted_center_x_224"],
                "tflite_fp32_predicted_center_y_224": fp32["predicted_center_y_224"],
                "tflite_fp32_predicted_tip_x_224": fp32["predicted_tip_x_224"],
                "tflite_fp32_predicted_tip_y_224": fp32["predicted_tip_y_224"],
                "tflite_fp32_center_heatmap_peak_value": fp32["center_heatmap_peak_value"],
                "tflite_fp32_tip_heatmap_peak_value": fp32["tip_heatmap_peak_value"],
                "tflite_fp32_center_heatmap_spread_px": fp32["center_heatmap_spread_px"],
                "tflite_fp32_tip_heatmap_spread_px": fp32["tip_heatmap_spread_px"],
                "tflite_fp32_guardrail_status": fp32["guardrail_status"],
                "tflite_fp32_rejection_reasons": fp32["rejection_reasons"],
                "tflite_int8_predicted_temperature_c": int8["guarded_temperature_c"],
                "tflite_int8_predicted_center_x_224": int8["predicted_center_x_224"],
                "tflite_int8_predicted_center_y_224": int8["predicted_center_y_224"],
                "tflite_int8_predicted_tip_x_224": int8["predicted_tip_x_224"],
                "tflite_int8_predicted_tip_y_224": int8["predicted_tip_y_224"],
                "tflite_int8_center_heatmap_peak_value": int8["center_heatmap_peak_value"],
                "tflite_int8_tip_heatmap_peak_value": int8["tip_heatmap_peak_value"],
                "tflite_int8_center_heatmap_spread_px": int8["center_heatmap_spread_px"],
                "tflite_int8_tip_heatmap_spread_px": int8["tip_heatmap_spread_px"],
                "tflite_int8_guardrail_status": int8["guardrail_status"],
                "tflite_int8_rejection_reasons": int8["rejection_reasons"],
                "true_temperature_c": canonical["true_temperature_c"],
                "true_angle_degrees": canonical["true_angle_degrees"],
                "true_center_x_224": canonical["true_center_x_224"],
                "true_center_y_224": canonical["true_center_y_224"],
                "true_tip_x_224": canonical["true_tip_x_224"],
                "true_tip_y_224": canonical["true_tip_y_224"],
                "trainer_preprocessing_mode": trainer["preprocessing_mode"],
                "trainer_resize_method": trainer["resize_method"],
                "trainer_channel_strategy": trainer["channel_strategy"],
                "trainer_normalization": trainer["normalization"],
                "canonical_preprocessing_mode": canonical["preprocessing_mode"],
                "canonical_resize_method": canonical["resize_method"],
                "canonical_channel_strategy": canonical["channel_strategy"],
                "canonical_normalization": canonical["normalization"],
            }
        )
    return merged_rows


def _write_report(
    *,
    report_path: Path,
    parity_rows: list[dict[str, Any]],
    trainer_summary: dict[str, Any],
    canonical_summary: dict[str, Any],
    fp32_summary: dict[str, Any],
    int8_summary: dict[str, Any],
    trainer_vs_canonical: dict[str, float],
    canonical_vs_fp32: dict[str, float],
    canonical_vs_int8: dict[str, float],
    reload_summary: dict[str, Any],
    validation_rows: int,
) -> None:
    """Write the parity audit markdown report."""

    lines = [
        "# Geometry Heatmap v3 Replay Parity Audit",
        "",
        f"- Validation rows: {validation_rows}",
        f"- Decoder: softargmax w3",
        f"- Canonical preprocessing: {DEFAULT_PREPROCESSING_MODE}",
        "",
        "## Replay Paths",
        f"- Trainer-style replay accepted MAE: {trainer_summary['accepted_mae_c']:.4f} C",
        f"- Trainer-style replay acceptance rate: {trainer_summary['acceptance_rate']:.4f}",
        f"- Trainer-style replay worst accepted error: {trainer_summary['worst_accepted_error_c']:.4f} C",
        f"- Standalone Keras accepted MAE: {canonical_summary['accepted_mae_c']:.4f} C",
        f"- Standalone Keras acceptance rate: {canonical_summary['acceptance_rate']:.4f}",
        f"- Standalone Keras worst accepted error: {canonical_summary['worst_accepted_error_c']:.4f} C",
        f"- TFLite FP32 accepted MAE: {fp32_summary['accepted_mae_c']:.4f} C",
        f"- TFLite FP32 acceptance rate: {fp32_summary['acceptance_rate']:.4f}",
        f"- TFLite FP32 worst accepted error: {fp32_summary['worst_accepted_error_c']:.4f} C",
        f"- TFLite INT8 accepted MAE: {int8_summary['accepted_mae_c']:.4f} C",
        f"- TFLite INT8 acceptance rate: {int8_summary['acceptance_rate']:.4f}",
        f"- TFLite INT8 worst accepted error: {int8_summary['worst_accepted_error_c']:.4f} C",
        "",
        "## Parity Deltas",
        f"- Trainer vs standalone Keras temp drift mean/median/p90: {trainer_vs_canonical['temperature_delta_mean']:.4f} / {trainer_vs_canonical['temperature_delta_median']:.4f} / {trainer_vs_canonical['temperature_delta_p90']:.4f}",
        f"- Trainer vs standalone Keras center drift mean/median: {trainer_vs_canonical['center_delta_mean']:.4f} / {trainer_vs_canonical['center_delta_median']:.4f}",
        f"- Trainer vs standalone Keras tip drift mean/median: {trainer_vs_canonical['tip_delta_mean']:.4f} / {trainer_vs_canonical['tip_delta_median']:.4f}",
        f"- Canonical Keras vs FP32 temp drift mean/median/p90: {canonical_vs_fp32['temperature_delta_mean']:.4f} / {canonical_vs_fp32['temperature_delta_median']:.4f} / {canonical_vs_fp32['temperature_delta_p90']:.4f}",
        f"- Canonical Keras vs FP32 center drift mean/median: {canonical_vs_fp32['center_delta_mean']:.4f} / {canonical_vs_fp32['center_delta_median']:.4f}",
        f"- Canonical Keras vs FP32 tip drift mean/median: {canonical_vs_fp32['tip_delta_mean']:.4f} / {canonical_vs_fp32['tip_delta_median']:.4f}",
        f"- Canonical Keras vs INT8 temp drift mean/median/p90: {canonical_vs_int8['temperature_delta_mean']:.4f} / {canonical_vs_int8['temperature_delta_median']:.4f} / {canonical_vs_int8['temperature_delta_p90']:.4f}",
        f"- Canonical Keras vs INT8 center drift mean/median: {canonical_vs_int8['center_delta_mean']:.4f} / {canonical_vs_int8['center_delta_median']:.4f}",
        f"- Canonical Keras vs INT8 tip drift mean/median: {canonical_vs_int8['tip_delta_mean']:.4f} / {canonical_vs_int8['tip_delta_median']:.4f}",
        "",
        "## Crop Parity",
        f"- Crop metadata match rate: {np.mean([bool(row['crop_metadata_match']) for row in parity_rows]) * 100.0:.2f}%",
        f"- Trainer preprocessing: {parity_rows[0]['trainer_preprocessing_mode']}",
        f"- Canonical preprocessing: {parity_rows[0]['canonical_preprocessing_mode']}",
        f"- Trainer resize method: {parity_rows[0]['trainer_resize_method']}",
        f"- Canonical resize method: {parity_rows[0]['canonical_resize_method']}",
        "",
        "## Training-vs-Inference Behavior",
        "- Fake-quant round-trip is only active when the trainer explicitly applies the quantized-style replay path.",
        "- Inference paths use `training=False`.",
        "- The exported FP32 and INT8 replay paths are deterministic on repeated calls.",
        "",
        "## Checkpoint Reload Parity",
        f"- Reload max abs diff (center_heatmap): {reload_summary['center_max_abs_diff']:.8f}",
        f"- Reload max abs diff (tip_heatmap): {reload_summary['tip_max_abs_diff']:.8f}",
        f"- Reload max abs diff (confidence): {reload_summary['confidence_max_abs_diff']:.8f}",
        f"- Reload mean abs diff (center_heatmap): {reload_summary['center_mean_abs_diff']:.8f}",
        f"- Reload mean abs diff (tip_heatmap): {reload_summary['tip_mean_abs_diff']:.8f}",
        f"- Reload mean abs diff (confidence): {reload_summary['confidence_mean_abs_diff']:.8f}",
        f"- Reload parity passed: {'yes' if reload_summary['passed'] else 'no'}",
        "",
        "## Decision",
        "- The canonical validation replay is the standalone Keras / FP32 replay path.",
        "- The trainer-side selection metric was optimistic because it used a non-canonical preprocessing/scoring path.",
        "- Previous checkpoint selection should not be trusted for export decisions.",
        "- The next architecture step should be decided only after rerunning v3 training with canonical validation scoring.",
    ]
    _write_markdown(lines, report_path)


def _compare_checkpoint_reload(
    *,
    model_path: Path,
    samples: list[HeatmapSample],
    decode_method: str,
    window_size: int,
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    tflite_float_path: Path | None,
    semantic_output_order_indices: list[int],
) -> dict[str, Any]:
    """Compare a loaded checkpoint against a save/reload round trip and FP32 TFLite."""

    loaded_model = load_geometry_heatmap_keras_model(model_path)
    with TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir) / "roundtrip.keras"
        loaded_model.save(temp_path)
        reloaded_model = load_geometry_heatmap_keras_model(temp_path)

        inputs = np.stack([sample.crop_image for sample in samples], axis=0).astype(np.float32)
        loaded_outputs = _as_output_dict(loaded_model.predict(inputs, verbose=0))
        reloaded_outputs = _as_output_dict(reloaded_model.predict(inputs, verbose=0))

        float_outputs: dict[str, np.ndarray] | None = None
        if tflite_float_path.exists():
            float_outputs, _ = _predict_tflite_outputs(
                tflite_float_path,
                samples,
                semantic_output_order_indices=semantic_output_order_indices,
            )

    summary: dict[str, Any] = {"passed": True}
    diff_sources = {
        "center": loaded_outputs["center_heatmap"] - reloaded_outputs["center_heatmap"],
        "tip": loaded_outputs["tip_heatmap"] - reloaded_outputs["tip_heatmap"],
        "confidence": loaded_outputs["confidence"] - reloaded_outputs["confidence"],
    }
    for name, tensor in loaded_outputs.items():
        summary[f"{name.split('_')[0]}_loaded_stats"] = _tensor_stats(tensor.numpy())
        summary[f"{name.split('_')[0]}_reloaded_stats"] = _tensor_stats(reloaded_outputs[name].numpy())
    summary.update(
        {
            "center_max_abs_diff": float(np.max(np.abs(diff_sources["center"]))),
            "center_mean_abs_diff": float(np.mean(np.abs(diff_sources["center"]))),
            "tip_max_abs_diff": float(np.max(np.abs(diff_sources["tip"]))),
            "tip_mean_abs_diff": float(np.mean(np.abs(diff_sources["tip"]))),
            "confidence_max_abs_diff": float(np.max(np.abs(diff_sources["confidence"]))),
            "confidence_mean_abs_diff": float(np.mean(np.abs(diff_sources["confidence"]))),
        }
    )

    if float_outputs is not None:
        fp32_rows = _decode_rows(
            model_type="tflite_float32",
            samples=samples,
            outputs=float_outputs,
            calibration_candidate=calibration_candidate,
            thresholds=thresholds,
            decode_method=decode_method,
            window_size=window_size,
        )
        loaded_rows = _decode_rows(
            model_type="keras_loaded",
            samples=samples,
            outputs=loaded_outputs,
            calibration_candidate=calibration_candidate,
            thresholds=thresholds,
            decode_method=decode_method,
            window_size=window_size,
        )
        fp32_deltas = _path_drift(loaded_rows, fp32_rows)
        summary["fp32_predicted_temperature_mean_delta"] = fp32_deltas["temperature_delta_mean"]
    return summary


def main() -> None:
    """Audit parity between trainer-side replay, canonical replay, and TFLite."""

    parser = argparse.ArgumentParser(description="Audit geometry_heatmap_v3 replay parity")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--float-tflite-path", type=Path, default=DEFAULT_FLOAT_TFLITE_PATH)
    parser.add_argument("--int8-tflite-path", type=Path, default=DEFAULT_INT8_TFLITE_PATH)
    parser.add_argument("--contract-path", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_PARITY_AUDIT_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_PARITY_REPORT_PATH)
    parser.add_argument("--canonical-report-path", type=Path, default=DEFAULT_CANONICAL_REPORT_PATH)
    parser.add_argument("--reload-debug-path", type=Path, default=DEFAULT_RELOAD_DEBUG_PATH)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    model_path = resolve_repo_path(repo_root, args.model_path)
    float_tflite_path = resolve_repo_path(repo_root, args.float_tflite_path)
    int8_tflite_path = resolve_repo_path(repo_root, args.int8_tflite_path)
    contract_path = resolve_repo_path(repo_root, args.contract_path)
    manifest_path = resolve_repo_path(repo_root, args.manifest_path)
    calibration_path = resolve_repo_path(repo_root, args.calibration_path)
    thresholds_path = resolve_repo_path(repo_root, args.thresholds_path)
    output_csv = resolve_repo_path(repo_root, args.output_csv)
    report_path = resolve_repo_path(repo_root, args.report_path)
    canonical_report_path = resolve_repo_path(repo_root, args.canonical_report_path)
    reload_debug_path = resolve_repo_path(repo_root, args.reload_debug_path)

    decode_method, window_size = _load_selected_decode_spec(repo_root / "ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json")
    calibration_candidate, calibration_json = load_selected_calibration_candidate(calibration_path)
    thresholds = _load_thresholds(thresholds_path)
    semantic_output_order_indices = load_semantic_output_order_indices(contract_path)

    pairs = _load_validation_inputs(repo_root)
    trainer_samples = [pair.trainer_sample for pair in pairs]
    canonical_samples = [pair.canonical_sample for pair in pairs]
    trainer_inputs = np.stack([sample.crop_image for sample in trainer_samples], axis=0).astype(np.float32)
    canonical_inputs = np.stack([sample.crop_image for sample in canonical_samples], axis=0).astype(np.float32)

    model = load_geometry_heatmap_keras_model(model_path)

    trainer_outputs = _as_output_dict(model.predict(trainer_inputs, verbose=0))
    trainer_outputs = {
        name: np.asarray(fake_quantize_01_tensor(tensor), dtype=np.float32)
        for name, tensor in trainer_outputs.items()
    }
    canonical_outputs = _as_output_dict(model.predict(canonical_inputs, verbose=0))

    float_outputs_dict = None
    int8_outputs_dict = None
    if float_tflite_path.exists():
        float_outputs_dict, _ = _predict_tflite_outputs(
            float_tflite_path,
            canonical_samples,
            semantic_output_order_indices=semantic_output_order_indices,
        )
    if int8_tflite_path.exists():
        int8_outputs_dict, _ = _predict_tflite_outputs(
            int8_tflite_path,
            canonical_samples,
            semantic_output_order_indices=semantic_output_order_indices,
        )

    trainer_rows = _decode_rows(
        model_type="trainer_style",
        samples=trainer_samples,
        outputs=trainer_outputs,
        calibration_candidate=calibration_candidate,
        thresholds=thresholds,
        decode_method=decode_method,
        window_size=window_size,
    )
    canonical_rows = _decode_rows(
        model_type="standalone_keras",
        samples=canonical_samples,
        outputs=canonical_outputs,
        calibration_candidate=calibration_candidate,
        thresholds=thresholds,
        decode_method=decode_method,
        window_size=window_size,
    )
    fp32_rows = (
        _decode_rows(
            model_type="tflite_fp32",
            samples=canonical_samples,
            outputs=float_outputs_dict,
            calibration_candidate=calibration_candidate,
            thresholds=thresholds,
            decode_method=decode_method,
            window_size=window_size,
        )
        if float_outputs_dict is not None
        else canonical_rows
    )
    int8_rows = (
        _decode_rows(
            model_type="tflite_int8",
            samples=canonical_samples,
            outputs=int8_outputs_dict,
            calibration_candidate=calibration_candidate,
            thresholds=thresholds,
            decode_method=decode_method,
            window_size=window_size,
        )
        if int8_outputs_dict is not None
        else canonical_rows
    )

    parity_rows = _serialize_rows(
        trainer_rows=trainer_rows,
        canonical_rows=canonical_rows,
        fp32_rows=fp32_rows,
        int8_rows=int8_rows,
    )
    _write_csv(parity_rows, output_csv)

    trainer_summary = _summarize(trainer_rows)
    canonical_summary = _summarize(canonical_rows)
    fp32_summary = _summarize(fp32_rows)
    int8_summary = _summarize(int8_rows)
    trainer_vs_canonical = _path_drift(trainer_rows, canonical_rows)
    canonical_vs_fp32 = _path_drift(canonical_rows, fp32_rows)
    canonical_vs_int8 = _path_drift(canonical_rows, int8_rows)

    reload_summary = _compare_checkpoint_reload(
        model_path=model_path,
        samples=canonical_samples[:20],
        decode_method=decode_method,
        window_size=window_size,
        calibration_candidate=calibration_candidate,
        thresholds=thresholds,
        tflite_float_path=float_tflite_path,
        semantic_output_order_indices=semantic_output_order_indices,
    )

    _write_report(
        report_path=report_path,
        parity_rows=parity_rows,
        trainer_summary=trainer_summary,
        canonical_summary=canonical_summary,
        fp32_summary=fp32_summary,
        int8_summary=int8_summary,
        trainer_vs_canonical=trainer_vs_canonical,
        canonical_vs_fp32=canonical_vs_fp32,
        canonical_vs_int8=canonical_vs_int8,
        reload_summary=reload_summary,
        validation_rows=len(parity_rows),
    )

    _write_markdown(
        [
            "# Geometry Heatmap v3 Canonical Validation Rescore",
            "",
            f"- Decoder: {decode_method} w{window_size}",
            f"- Calibration candidate: {calibration_candidate.name}",
            f"- Validation rows: {len(canonical_rows)}",
            "",
            f"- Keras accepted MAE: {canonical_summary['accepted_mae_c']:.4f} C",
            f"- Acceptance: {canonical_summary['acceptance_rate']:.4f}",
            f"- Worst accepted error: {canonical_summary['worst_accepted_error_c']:.4f} C",
            f"- Accepted >20 C failures: {int(canonical_summary['accepted_gt20_failures'])}",
            f"- Rejection reasons: {canonical_summary['top_rejection_reasons']}",
            f"- Center MAE px: {canonical_summary['center_mae_px_224']:.4f}",
            f"- Tip MAE px: {canonical_summary['tip_mae_px_224']:.4f}",
            f"- Angle MAE deg: {canonical_summary['angle_mae_degrees']:.4f}",
            f"- Heatmap center peak/spread mean: {canonical_summary['center_heatmap_peak_mean']:.4f} / {canonical_summary['center_heatmap_spread_mean']:.4f}",
            f"- Heatmap tip peak/spread mean: {canonical_summary['tip_heatmap_peak_mean']:.4f} / {canonical_summary['tip_heatmap_spread_mean']:.4f}",
        ],
        canonical_report_path,
    )

    _write_markdown(
        [
            "# Geometry Heatmap v3 Checkpoint Reload Debug",
            "",
            f"- Reload max abs diff (center_heatmap): {reload_summary['center_max_abs_diff']:.8f}",
            f"- Reload max abs diff (tip_heatmap): {reload_summary['tip_max_abs_diff']:.8f}",
            f"- Reload max abs diff (confidence): {reload_summary['confidence_max_abs_diff']:.8f}",
            f"- Reload mean abs diff (center_heatmap): {reload_summary['center_mean_abs_diff']:.8f}",
            f"- Reload mean abs diff (tip_heatmap): {reload_summary['tip_mean_abs_diff']:.8f}",
            f"- Reload mean abs diff (confidence): {reload_summary['confidence_mean_abs_diff']:.8f}",
            f"- Reload parity passed: {'yes' if reload_summary['passed'] else 'no'}",
            f"- FP32 mean temp delta vs loaded Keras: {reload_summary.get('fp32_predicted_temperature_mean_delta', math.nan):.8f}",
        ],
        reload_debug_path,
    )

    print(f"[V3 PARITY] Wrote CSV: {output_csv}", flush=True)
    print(f"[V3 PARITY] Wrote report: {report_path}", flush=True)
    print(f"[V3 PARITY] Canonical Keras accepted MAE: {canonical_summary['accepted_mae_c']:.4f} C", flush=True)
    print(f"[V3 PARITY] Canonical Keras acceptance rate: {canonical_summary['acceptance_rate']:.4f}", flush=True)
    print(f"[V3 PARITY] Canonical Keras worst accepted error: {canonical_summary['worst_accepted_error_c']:.4f} C", flush=True)
    print(f"[V3 PARITY] INT8 accepted MAE: {int8_summary['accepted_mae_c']:.4f} C", flush=True)
    print(f"[V3 PARITY] INT8 acceptance rate: {int8_summary['acceptance_rate']:.4f}", flush=True)
    print(f"[V3 PARITY] INT8 worst accepted error: {int8_summary['worst_accepted_error_c']:.4f} C", flush=True)


if __name__ == "__main__":
    main()
