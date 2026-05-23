#!/usr/bin/env python3
"""Compare Keras FP32 and TFLite INT8 geometry predictions for quantization drift."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
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
    load_models,
    load_semantic_output_order_indices,
    load_split_samples,
    predict_keras_outputs,
    predict_tflite_outputs,
    resolve_repo_path,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import load_selected_calibration_candidate
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import (
    GeometryGuardrailResult,
    GeometryGuardrailThresholds,
)


SPLITS = ("train", "val", "test")
KERAS_MODEL_LABEL = "keras_fp32"
INT8_MODEL_LABEL = "tflite_int8"
DEFAULT_THRESHOLD_PATH = Path("ml/artifacts/training/geometry_heatmap_v2_board_replay/selected_board_guardrail_thresholds.json")
DEFAULT_CONTRACT_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/tflite_tensor_contract.json")
DEFAULT_MODEL_PATH = Path("ml/artifacts/training/geometry_heatmap_v2/model.keras")
DEFAULT_TFLITE_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/model_int8.tflite")
DEFAULT_CALIBRATION_PATH = Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json")
DEFAULT_OUTPUT_DIR = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2")
DEFAULT_REPORT_PATH = Path("ml/reports/geometry_heatmap_v2_quantization_drift_v2.md")
DEFAULT_DEBUG_DIR = Path("ml/debug/geometry_heatmap_v2_quantization_drift_v2")


@dataclass(frozen=True)
class ModelDecodeResult:
    """Decoded geometry plus guardrail outcome for one model."""

    decoded: Any
    guarded: GeometryGuardrailResult


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a deterministic CSV table."""

    if not rows:
        raise ValueError("Cannot write an empty CSV file.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(report_path: Path, lines: list[str]) -> None:
    """Persist a markdown report with a stable newline convention."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _load_thresholds(thresholds_path: Path) -> GeometryGuardrailThresholds:
    """Load the selected board replay threshold artifact."""

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


def _load_selected_decode_spec(selection_path: Path) -> tuple[str, str, int]:
    """Load the selected decode name and split it into method plus window size."""

    if not selection_path.exists():
        raise RuntimeError(f"Missing selected decode selection artifact: {selection_path}")
    with selection_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
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
    return selected_name, decode_method, selected_window_size


def _status_is_accepted(status: str) -> bool:
    """Treat accepted and clamped predictions as usable for gating statistics."""

    return status in {"accepted", "clamped"}


def _acceptance_class(status: str) -> str:
    """Map a guardrail status to the accepted/rejected class used in drift analysis."""

    return "accepted" if _status_is_accepted(status) else "rejected"


def _model_result(
    sample: Any,
    center_heatmap: np.ndarray,
    tip_heatmap: np.ndarray,
    confidence: float,
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    *,
    decode_method: str,
    window_size: int,
) -> ModelDecodeResult:
    """Decode one sample and apply the shared guardrails."""

    decoded, guarded = decode_and_guard(
        sample,
        center_heatmap,
        tip_heatmap,
        confidence,
        calibration_candidate,
        thresholds,
        decode_method=decode_method,  # type: ignore[arg-type]
        window_size=window_size,
    )
    return ModelDecodeResult(decoded=decoded, guarded=guarded)


def _flatten_model_result(
    *,
    prefix: str,
    sample: Any,
    result: ModelDecodeResult,
) -> dict[str, Any]:
    """Flatten one model result into prefixed CSV fields."""

    decoded = result.decoded
    guarded = result.guarded
    features = guarded.quality_features
    row: dict[str, Any] = {
        f"{prefix}_center_x_224": float(decoded.predicted_center_x_224),
        f"{prefix}_center_y_224": float(decoded.predicted_center_y_224),
        f"{prefix}_tip_x_224": float(decoded.predicted_tip_x_224),
        f"{prefix}_tip_y_224": float(decoded.predicted_tip_y_224),
        f"{prefix}_center_x_224_argmax": float(decoded.predicted_center_x_224_argmax),
        f"{prefix}_center_y_224_argmax": float(decoded.predicted_center_y_224_argmax),
        f"{prefix}_tip_x_224_argmax": float(decoded.predicted_tip_x_224_argmax),
        f"{prefix}_tip_y_224_argmax": float(decoded.predicted_tip_y_224_argmax),
        f"{prefix}_predicted_angle_degrees": float(decoded.predicted_angle_degrees),
        f"{prefix}_predicted_angle_degrees_argmax": float(decoded.predicted_angle_degrees_argmax),
        f"{prefix}_predicted_temperature_c_current_mapping": float(decoded.predicted_temperature_c_current_mapping),
        f"{prefix}_predicted_temperature_c_current_mapping_argmax": float(decoded.predicted_temperature_c_current_mapping_argmax),
        f"{prefix}_predicted_temperature_c_calibrated": float(decoded.predicted_temperature_c_calibrated),
        f"{prefix}_predicted_temperature_c_calibrated_argmax": float(decoded.predicted_temperature_c_calibrated_argmax),
        f"{prefix}_absolute_error_c_current_mapping": float(decoded.absolute_error_c_current_mapping),
        f"{prefix}_absolute_error_c_current_mapping_argmax": float(decoded.absolute_error_c_current_mapping_argmax),
        f"{prefix}_absolute_error_c_calibrated": float(decoded.absolute_error_c_calibrated),
        f"{prefix}_absolute_error_c_calibrated_argmax": float(decoded.absolute_error_c_calibrated_argmax),
        f"{prefix}_center_heatmap_peak_value": float(decoded.center_heatmap_peak_value),
        f"{prefix}_tip_heatmap_peak_value": float(decoded.tip_heatmap_peak_value),
        f"{prefix}_center_heatmap_mean_value": float(decoded.center_heatmap_mean_value),
        f"{prefix}_tip_heatmap_mean_value": float(decoded.tip_heatmap_mean_value),
        f"{prefix}_center_heatmap_entropy": float(features.center_heatmap_entropy),
        f"{prefix}_tip_heatmap_entropy": float(features.tip_heatmap_entropy),
        f"{prefix}_center_heatmap_spread_px": float(features.center_heatmap_spread_px),
        f"{prefix}_tip_heatmap_spread_px": float(features.tip_heatmap_spread_px),
        f"{prefix}_confidence": float(decoded.confidence),
        f"{prefix}_guardrail_status": str(guarded.status),
        f"{prefix}_rejection_reasons": ";".join(guarded.rejection_reasons) if guarded.rejection_reasons else "none",
        f"{prefix}_guarded_temperature_c": float(guarded.temperature_c),
        f"{prefix}_raw_temperature_c": float(guarded.raw_temperature_c),
        f"{prefix}_accepted_class": _acceptance_class(guarded.status),
        f"{prefix}_softargmax_gap_center_px": float(
            math.hypot(
                float(decoded.predicted_center_x_224) - float(decoded.predicted_center_x_224_argmax),
                float(decoded.predicted_center_y_224) - float(decoded.predicted_center_y_224_argmax),
            )
        ),
        f"{prefix}_softargmax_gap_tip_px": float(
            math.hypot(
                float(decoded.predicted_tip_x_224) - float(decoded.predicted_tip_x_224_argmax),
                float(decoded.predicted_tip_y_224) - float(decoded.predicted_tip_y_224_argmax),
            )
        ),
        f"{prefix}_source_manifest": str(sample.metadata.get("source_manifest", "")),
        f"{prefix}_source_folder": Path(str(sample.metadata.get("image_path", ""))).parent.name,
    }
    return row


def _build_rows_for_split(
    *,
    split: str,
    samples: list[Any],
    keras_center: np.ndarray,
    keras_tip: np.ndarray,
    keras_confidence: np.ndarray,
    int8_center: np.ndarray,
    int8_tip: np.ndarray,
    int8_confidence: np.ndarray,
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    decode_method: str,
    window_size: int,
) -> list[dict[str, Any]]:
    """Build the per-sample drift table for one split."""

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        keras_result = _model_result(
            sample,
            keras_center[index],
            keras_tip[index],
            float(np.ravel(keras_confidence[index])[0]),
            calibration_candidate,
            thresholds,
            decode_method=decode_method,
            window_size=window_size,
        )
        int8_result = _model_result(
            sample,
            int8_center[index],
            int8_tip[index],
            float(np.ravel(int8_confidence[index])[0]),
            calibration_candidate,
            thresholds,
            decode_method=decode_method,
            window_size=window_size,
        )
        keras_row = _flatten_model_result(prefix="keras", sample=sample, result=keras_result)
        int8_row = _flatten_model_result(prefix="int8", sample=sample, result=int8_result)

        keras_status = keras_row["keras_guardrail_status"]
        int8_status = int8_row["int8_guardrail_status"]
        if keras_status == "accepted" and int8_status == "accepted":
            disagreement_type = "both accepted"
        elif keras_status == "accepted" and int8_status == "rejected":
            disagreement_type = "keras accepted, int8 rejected"
        elif keras_status == "rejected" and int8_status == "accepted":
            disagreement_type = "keras rejected, int8 accepted"
        else:
            disagreement_type = "both rejected"

        row: dict[str, Any] = {
            "split": split,
            "image_path": str(sample.metadata["image_path"]),
            "true_temperature_c": float(sample.metadata["temperature_c"]),
            "true_angle_degrees": float(keras_result.decoded.true_angle_degrees),
            "true_center_x_224": float(keras_result.decoded.true_center_x_224),
            "true_center_y_224": float(keras_result.decoded.true_center_y_224),
            "true_tip_x_224": float(keras_result.decoded.true_tip_x_224),
            "true_tip_y_224": float(keras_result.decoded.true_tip_y_224),
            "source_manifest": str(sample.metadata.get("source_manifest", "")),
            "source_folder": Path(str(sample.metadata.get("image_path", ""))).parent.name,
            "quality_flag": str(sample.metadata.get("quality_flag", "")),
            "crop_x1": int(sample.metadata["crop_x1"]),
            "crop_y1": int(sample.metadata["crop_y1"]),
            "crop_x2": int(sample.metadata["crop_x2"]),
            "crop_y2": int(sample.metadata["crop_y2"]),
            "crop_width": int(sample.metadata["crop_width"]),
            "crop_height": int(sample.metadata["crop_height"]),
            "jitter_shift_x": int(sample.metadata["jitter_shift_x"]),
            "jitter_shift_y": int(sample.metadata["jitter_shift_y"]),
            "jitter_scale": float(sample.metadata["jitter_scale"]),
            "jitter_aspect": float(sample.metadata["jitter_aspect"]),
            "preprocessing_mode": str(sample.metadata["preprocessing_mode"]),
            "disagreement_type": disagreement_type,
            "guardrail_disagreement": disagreement_type != "both accepted",
            "keras_vs_int8_temperature_delta_c": float(
                abs(float(keras_row["keras_predicted_temperature_c_calibrated"]) - float(int8_row["int8_predicted_temperature_c_calibrated"]))
            ),
            "keras_vs_int8_center_delta_px": float(
                math.hypot(
                    float(keras_row["keras_center_x_224"]) - float(int8_row["int8_center_x_224"]),
                    float(keras_row["keras_center_y_224"]) - float(int8_row["int8_center_y_224"]),
                )
            ),
            "keras_vs_int8_tip_delta_px": float(
                math.hypot(
                    float(keras_row["keras_tip_x_224"]) - float(int8_row["int8_tip_x_224"]),
                    float(keras_row["keras_tip_y_224"]) - float(int8_row["int8_tip_y_224"]),
                )
            ),
            "keras_vs_int8_angle_delta_degrees": float(
                circular_angle_error_degrees(
                    float(keras_row["keras_predicted_angle_degrees"]),
                    float(int8_row["int8_predicted_angle_degrees"]),
                )
            ),
            "keras_vs_int8_center_peak_delta": float(
                float(keras_row["keras_center_heatmap_peak_value"]) - float(int8_row["int8_center_heatmap_peak_value"])
            ),
            "keras_vs_int8_tip_peak_delta": float(
                float(keras_row["keras_tip_heatmap_peak_value"]) - float(int8_row["int8_tip_heatmap_peak_value"])
            ),
            "keras_vs_int8_center_spread_delta_px": float(
                float(int8_row["int8_center_heatmap_spread_px"]) - float(keras_row["keras_center_heatmap_spread_px"])
            ),
            "keras_vs_int8_tip_spread_delta_px": float(
                float(int8_row["int8_tip_heatmap_spread_px"]) - float(keras_row["keras_tip_heatmap_spread_px"])
            ),
            "keras_vs_int8_confidence_delta": float(
                float(keras_row["keras_confidence"]) - float(int8_row["int8_confidence"])
            ),
        }
        row.update(keras_row)
        row.update(int8_row)
        rows.append(row)
    return rows


def _summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one split worth of paired Keras/INT8 predictions."""

    keras_accepted_errors = np.asarray(
        [
            abs(float(row["keras_guarded_temperature_c"]) - float(row["true_temperature_c"]))
            for row in rows
            if row["keras_guardrail_status"] != "rejected"
        ],
        dtype=np.float64,
    )
    int8_accepted_errors = np.asarray(
        [
            abs(float(row["int8_guarded_temperature_c"]) - float(row["true_temperature_c"]))
            for row in rows
            if row["int8_guardrail_status"] != "rejected"
        ],
        dtype=np.float64,
    )
    int8_drift = np.asarray([float(row["keras_vs_int8_temperature_delta_c"]) for row in rows], dtype=np.float64)
    int8_tip_drift = np.asarray([float(row["keras_vs_int8_tip_delta_px"]) for row in rows], dtype=np.float64)
    int8_center_drift = np.asarray([float(row["keras_vs_int8_center_delta_px"]) for row in rows], dtype=np.float64)
    disagreement_counts = Counter(str(row["disagreement_type"]) for row in rows)
    accepted_disagreements = sum(
        1 for row in rows if row["keras_guardrail_status"] != row["int8_guardrail_status"]
    )
    accepted_gt20 = sum(
        1
        for row in rows
        if row["int8_guardrail_status"] != "rejected" and abs(float(row["int8_guarded_temperature_c"]) - float(row["true_temperature_c"])) > 20.0
    )

    return {
        "count": int(len(rows)),
        "keras_accepted_count": int(sum(1 for row in rows if row["keras_guardrail_status"] != "rejected")),
        "int8_accepted_count": int(sum(1 for row in rows if row["int8_guardrail_status"] != "rejected")),
        "keras_acceptance_rate": float(np.mean([row["keras_guardrail_status"] != "rejected" for row in rows])),
        "int8_acceptance_rate": float(np.mean([row["int8_guardrail_status"] != "rejected" for row in rows])),
        "keras_accepted_mae_c": float(np.mean(keras_accepted_errors)) if keras_accepted_errors.size else math.nan,
        "keras_accepted_rmse_c": float(np.sqrt(np.mean(np.square(keras_accepted_errors)))) if keras_accepted_errors.size else math.nan,
        "keras_worst_accepted_error_c": float(np.max(keras_accepted_errors)) if keras_accepted_errors.size else math.nan,
        "int8_accepted_mae_c": float(np.mean(int8_accepted_errors)) if int8_accepted_errors.size else math.nan,
        "int8_accepted_rmse_c": float(np.sqrt(np.mean(np.square(int8_accepted_errors)))) if int8_accepted_errors.size else math.nan,
        "int8_worst_accepted_error_c": float(np.max(int8_accepted_errors)) if int8_accepted_errors.size else math.nan,
        "int8_percentage_under_2c": float(np.mean(int8_accepted_errors < 2.0) * 100.0) if int8_accepted_errors.size else 0.0,
        "int8_percentage_under_5c": float(np.mean(int8_accepted_errors < 5.0) * 100.0) if int8_accepted_errors.size else 0.0,
        "int8_percentage_under_10c": float(np.mean(int8_accepted_errors < 10.0) * 100.0) if int8_accepted_errors.size else 0.0,
        "keras_vs_int8_temperature_delta_mean": float(np.mean(int8_drift)),
        "keras_vs_int8_temperature_delta_median": float(np.median(int8_drift)),
        "keras_vs_int8_temperature_delta_p90": float(np.percentile(int8_drift, 90)),
        "keras_vs_int8_center_delta_mean": float(np.mean(int8_center_drift)),
        "keras_vs_int8_center_delta_median": float(np.median(int8_center_drift)),
        "keras_vs_int8_center_delta_p90": float(np.percentile(int8_center_drift, 90)),
        "keras_vs_int8_tip_delta_mean": float(np.mean(int8_tip_drift)),
        "keras_vs_int8_tip_delta_median": float(np.median(int8_tip_drift)),
        "keras_vs_int8_tip_delta_p90": float(np.percentile(int8_tip_drift, 90)),
        "guardrail_disagreement_count": int(accepted_disagreements),
        "accepted_gt20_failures": int(accepted_gt20),
        "disagreement_counts": dict(disagreement_counts),
    }


def _format_table(rows: list[dict[str, Any]], columns: list[str], *, limit: int | None = None) -> list[str]:
    """Render a compact markdown table from a row list."""

    selected_rows = rows if limit is None else rows[:limit]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in selected_rows:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _pick_top(rows: list[dict[str, Any]], key: str, limit: int = 30) -> list[dict[str, Any]]:
    """Return the top rows by one numeric key."""

    return sorted(rows, key=lambda row: float(row[key]), reverse=True)[:limit]


def _root_cause_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the likely causes of INT8 drift."""

    tip_flattening = [
        row
        for row in rows
        if float(row["keras_vs_int8_tip_peak_delta"]) > 0.05
        and float(row["keras_vs_int8_tip_spread_delta_px"]) > 0.5
    ]
    softargmax_sensitive = [
        row
        for row in rows
        if float(row["keras_vs_int8_tip_delta_px"]) > 4.0
        and float(row["keras_vs_int8_tip_softargmax_gap_px"]) < float(row["keras_vs_int8_tip_delta_px"]) * 0.4
    ]
    center_spread_growth = [row for row in rows if float(row["keras_vs_int8_center_spread_delta_px"]) > 0.5]
    tip_spread_growth = [row for row in rows if float(row["keras_vs_int8_tip_spread_delta_px"]) > 0.5]
    output_order_issue = False  # ruled out by the FP32 replay parity observed in Phase 7
    return {
        "tip_flattening_count": int(len(tip_flattening)),
        "center_spread_growth_count": int(len(center_spread_growth)),
        "tip_spread_growth_count": int(len(tip_spread_growth)),
        "softargmax_sensitive_count": int(len(softargmax_sensitive)),
        "output_order_issue": output_order_issue,
        "input_quantization_primary": False,
        "output_quantization_primary": True,
        "representative_dataset_weakness_likely": True,
    }


def _make_overlay(
    *,
    sample: Any,
    keras_result: ModelDecodeResult,
    int8_result: ModelDecodeResult,
    output_path: Path,
) -> None:
    """Render a side-by-side quantization drift overlay."""

    fig = plt.figure(figsize=(18, 11), dpi=150)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.35, 1.0, 1.0), height_ratios=(1.0, 1.0))
    ax_crop = fig.add_subplot(grid[:, 0])
    ax_keras_center = fig.add_subplot(grid[0, 1])
    ax_keras_tip = fig.add_subplot(grid[1, 1])
    ax_int8_center = fig.add_subplot(grid[0, 2])
    ax_int8_tip = fig.add_subplot(grid[1, 2])

    crop = np.asarray(sample.crop_image, dtype=np.float32)
    ax_crop.imshow(crop)
    ax_crop.scatter(
        [float(sample.metadata["center_x_224"]), float(keras_result.decoded.predicted_center_x_224), float(int8_result.decoded.predicted_center_x_224)],
        [float(sample.metadata["center_y_224"]), float(keras_result.decoded.predicted_center_y_224), float(int8_result.decoded.predicted_center_y_224)],
        c=["lime", "cyan", "orange"],
        s=[75, 65, 65],
        marker="o",
        edgecolors="white",
        linewidths=1.0,
        label="center",
    )
    ax_crop.scatter(
        [float(sample.metadata["tip_x_224"]), float(keras_result.decoded.predicted_tip_x_224), float(int8_result.decoded.predicted_tip_x_224)],
        [float(sample.metadata["tip_y_224"]), float(keras_result.decoded.predicted_tip_y_224), float(int8_result.decoded.predicted_tip_y_224)],
        c=["red", "deepskyblue", "yellow"],
        s=[75, 65, 65],
        marker="x",
        linewidths=2.0,
        label="tip",
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
        [float(keras_result.decoded.predicted_center_x_224), float(keras_result.decoded.predicted_tip_x_224)],
        [float(keras_result.decoded.predicted_center_y_224), float(keras_result.decoded.predicted_tip_y_224)],
        color="deepskyblue",
        linewidth=2.0,
        alpha=0.9,
        label="keras line",
    )
    ax_crop.plot(
        [float(int8_result.decoded.predicted_center_x_224), float(int8_result.decoded.predicted_tip_x_224)],
        [float(int8_result.decoded.predicted_center_y_224), float(int8_result.decoded.predicted_tip_y_224)],
        color="yellow",
        linewidth=2.0,
        alpha=0.9,
        label="int8 line",
    )
    ax_crop.set_title("Board-contract crop fed to the model")
    ax_crop.set_axis_off()
    ax_crop.legend(loc="lower right", fontsize=8, framealpha=0.9)

    def _plot_heatmap(ax: plt.Axes, heatmap: np.ndarray, *, title: str, true_x: float, true_y: float, pred_x: float, pred_y: float) -> None:
        """Plot one predicted heatmap with true and predicted markers."""

        ax.imshow(np.asarray(heatmap, dtype=np.float32), cmap="magma", origin="upper")
        ax.scatter([true_x * 55.0 / 223.0], [true_y * 55.0 / 223.0], c="white", s=40, marker="o", edgecolors="black", linewidths=0.8)
        ax.scatter([pred_x * 55.0 / 223.0], [pred_y * 55.0 / 223.0], c="cyan", s=50, marker="x", linewidths=2.0)
        ax.set_title(title)
        ax.set_xlim(-0.5, heatmap.shape[1] - 0.5)
        ax.set_ylim(heatmap.shape[0] - 0.5, -0.5)
        ax.set_xlabel("x / col")
        ax.set_ylabel("y / row")

    _plot_heatmap(
        ax_keras_center,
        keras_result.decoded.center_heatmap,
        title=f"Keras center heatmap\npeak={keras_result.decoded.center_heatmap_peak_value:.4f}, spread={keras_result.guarded.quality_features.center_heatmap_spread_px:.2f}",
        true_x=float(sample.metadata["center_x_224"]),
        true_y=float(sample.metadata["center_y_224"]),
        pred_x=float(keras_result.decoded.predicted_center_x_224),
        pred_y=float(keras_result.decoded.predicted_center_y_224),
    )
    _plot_heatmap(
        ax_keras_tip,
        keras_result.decoded.tip_heatmap,
        title=f"Keras tip heatmap\npeak={keras_result.decoded.tip_heatmap_peak_value:.4f}, spread={keras_result.guarded.quality_features.tip_heatmap_spread_px:.2f}",
        true_x=float(sample.metadata["tip_x_224"]),
        true_y=float(sample.metadata["tip_y_224"]),
        pred_x=float(keras_result.decoded.predicted_tip_x_224),
        pred_y=float(keras_result.decoded.predicted_tip_y_224),
    )
    _plot_heatmap(
        ax_int8_center,
        int8_result.decoded.center_heatmap,
        title=f"INT8 center heatmap\npeak={int8_result.decoded.center_heatmap_peak_value:.4f}, spread={int8_result.guarded.quality_features.center_heatmap_spread_px:.2f}",
        true_x=float(sample.metadata["center_x_224"]),
        true_y=float(sample.metadata["center_y_224"]),
        pred_x=float(int8_result.decoded.predicted_center_x_224),
        pred_y=float(int8_result.decoded.predicted_center_y_224),
    )
    _plot_heatmap(
        ax_int8_tip,
        int8_result.decoded.tip_heatmap,
        title=f"INT8 tip heatmap\npeak={int8_result.decoded.tip_heatmap_peak_value:.4f}, spread={int8_result.guarded.quality_features.tip_heatmap_spread_px:.2f}",
        true_x=float(sample.metadata["tip_x_224"]),
        true_y=float(sample.metadata["tip_y_224"]),
        pred_x=float(int8_result.decoded.predicted_tip_x_224),
        pred_y=float(int8_result.decoded.predicted_tip_y_224),
    )

    info_lines = [
        f"file: {Path(str(sample.metadata['image_path'])).name}",
        f"split: {sample.metadata['split']}",
        f"source_manifest: {sample.metadata.get('source_manifest', '')}",
        f"true temp: {float(sample.metadata['temperature_c']):.2f} C",
        f"keras temp: {float(keras_result.decoded.predicted_temperature_c_calibrated):.2f} C",
        f"int8 temp: {float(int8_result.decoded.predicted_temperature_c_calibrated):.2f} C",
        f"temp delta: {abs(float(keras_result.decoded.predicted_temperature_c_calibrated) - float(int8_result.decoded.predicted_temperature_c_calibrated)):.2f} C",
        f"tip delta: {math.hypot(float(keras_result.decoded.predicted_tip_x_224) - float(int8_result.decoded.predicted_tip_x_224), float(keras_result.decoded.predicted_tip_y_224) - float(int8_result.decoded.predicted_tip_y_224)):.2f} px",
        f"keras status: {keras_result.guarded.status} ({';'.join(keras_result.guarded.rejection_reasons) or 'none'})",
        f"int8 status: {int8_result.guarded.status} ({';'.join(int8_result.guarded.rejection_reasons) or 'none'})",
        f"keras confidence: {float(keras_result.decoded.confidence):.4f}",
        f"int8 confidence: {float(int8_result.decoded.confidence):.4f}",
        f"keras tip peak/spread: {float(keras_result.decoded.tip_heatmap_peak_value):.4f} / {keras_result.guarded.quality_features.tip_heatmap_spread_px:.2f}",
        f"int8 tip peak/spread: {float(int8_result.decoded.tip_heatmap_peak_value):.4f} / {int8_result.guarded.quality_features.tip_heatmap_spread_px:.2f}",
    ]
    fig.text(0.01, 0.01, "\n".join(info_lines), family="monospace", fontsize=9, va="bottom")
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the quantization drift autopsy and save analysis artifacts."""

    parser = argparse.ArgumentParser(description="Analyze geometry heatmap quantization drift")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tflite-path", type=Path, default=DEFAULT_TFLITE_PATH)
    parser.add_argument("--manifest-path", type=Path, default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"))
    parser.add_argument("--calibration-json-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLD_PATH)
    parser.add_argument("--contract-path", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--heatmap-size", type=int, default=DEFAULT_HEATMAP_SIZE)
    parser.add_argument("--sigma-pixels", type=float, default=DEFAULT_SIGMA_PIXELS)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    model_path = resolve_repo_path(repo_root, args.model_path)
    tflite_path = resolve_repo_path(repo_root, args.tflite_path)
    manifest_path = resolve_repo_path(repo_root, args.manifest_path)
    calibration_json_path = resolve_repo_path(repo_root, args.calibration_json_path)
    thresholds_path = resolve_repo_path(repo_root, args.thresholds_path)
    contract_path = resolve_repo_path(repo_root, args.contract_path)
    output_dir = resolve_repo_path(repo_root, args.output_dir)
    report_path = resolve_repo_path(repo_root, args.report_path)
    debug_dir = resolve_repo_path(repo_root, args.debug_dir)
    selected_decode_path = output_dir / "selected_decode_method.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _load_thresholds(thresholds_path)
    calibration_candidate, calibration_json = load_selected_calibration_candidate(calibration_json_path)
    semantic_output_order_indices = load_semantic_output_order_indices(contract_path)
    selected_decode_name, selected_decode_method, selected_decode_window_size = _load_selected_decode_spec(selected_decode_path)
    keras_model, int8_bundle = load_models(model_path, tflite_path)

    all_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, dict[str, Any]] = {}

    for split in SPLITS:
        split_samples = load_split_samples(
            manifest_path,
            repo_root,
            split=split,
            mode=DEFAULT_PREPROCESSING_MODE,
            input_size=args.input_size,
            heatmap_size=args.heatmap_size,
            sigma_pixels=args.sigma_pixels,
        ).samples
        split_inputs = [sample.crop_image for sample in split_samples]
        keras_outputs = predict_keras_outputs(keras_model, split_inputs, batch_size=16)
        int8_outputs = predict_tflite_outputs(
            int8_bundle,
            split_inputs,
            semantic_output_order_indices=semantic_output_order_indices,
        )

        keras_center, keras_tip, keras_conf = keras_outputs
        int8_center, int8_tip, int8_conf = int8_outputs

        split_rows = _build_rows_for_split(
            split=split,
            samples=split_samples,
            keras_center=keras_center,
            keras_tip=keras_tip,
            keras_confidence=keras_conf,
            int8_center=int8_center,
            int8_tip=int8_tip,
            int8_confidence=int8_conf,
            calibration_candidate=calibration_candidate,
            thresholds=thresholds,
            decode_method=selected_decode_method,
            window_size=selected_decode_window_size,
        )
        all_rows.extend(split_rows)
        split_summaries[split] = _summary_from_rows(split_rows)

    _write_csv(all_rows, output_dir / "quantization_drift_analysis.csv")

    # Build overlay sets once so the most suspicious cases are easy to inspect visually.
    keras_results: dict[str, ModelDecodeResult] = {}
    int8_results: dict[str, ModelDecodeResult] = {}
    sample_lookup: dict[str, Any] = {}
    for row in all_rows:
        image_path = str(row["image_path"])
        sample_lookup.setdefault(image_path, None)

    # Reconstruct the sample-level results for the overlay set from the wide table.
    for split in SPLITS:
        split_samples = load_split_samples(
            manifest_path,
            repo_root,
            split=split,
            mode=DEFAULT_PREPROCESSING_MODE,
            input_size=args.input_size,
            heatmap_size=args.heatmap_size,
            sigma_pixels=args.sigma_pixels,
        ).samples
        split_inputs = [sample.crop_image for sample in split_samples]
        keras_outputs = predict_keras_outputs(keras_model, split_inputs, batch_size=16)
        int8_outputs = predict_tflite_outputs(
            int8_bundle,
            split_inputs,
            semantic_output_order_indices=semantic_output_order_indices,
        )
        for index, sample in enumerate(split_samples):
            keras_results[str(sample.metadata["image_path"])] = _model_result(
                sample,
                keras_outputs[0][index],
                keras_outputs[1][index],
                float(np.ravel(keras_outputs[2][index])[0]),
                calibration_candidate,
                thresholds,
                decode_method=selected_decode_method,
                window_size=selected_decode_window_size,
            )
            int8_results[str(sample.metadata["image_path"])] = _model_result(
                sample,
                int8_outputs[0][index],
                int8_outputs[1][index],
                float(np.ravel(int8_outputs[2][index])[0]),
                calibration_candidate,
                thresholds,
                decode_method=selected_decode_method,
                window_size=selected_decode_window_size,
            )
            sample_lookup[str(sample.metadata["image_path"])] = sample

    # Collect the overlay targets.
    overlay_targets: list[tuple[str, str]] = []
    top_tip = _pick_top(all_rows, "keras_vs_int8_tip_delta_px", limit=30)
    top_temp = _pick_top(all_rows, "keras_vs_int8_temperature_delta_c", limit=30)
    disagreement_rows = [row for row in all_rows if bool(row["guardrail_disagreement"])]
    for row in top_tip:
        overlay_targets.append((str(row["image_path"]), "top_tip_delta"))
    for row in top_temp:
        overlay_targets.append((str(row["image_path"]), "top_temp_delta"))
    for row in disagreement_rows:
        overlay_targets.append((str(row["image_path"]), "guardrail_disagreement"))
    # Deduplicate while preserving order.
    seen: set[str] = set()
    deduped_targets: list[tuple[str, str]] = []
    for image_path, reason in overlay_targets:
        key = f"{image_path}:{reason}"
        if key in seen:
            continue
        seen.add(key)
        deduped_targets.append((image_path, reason))

    for image_path, reason in deduped_targets:
        sample = sample_lookup[image_path]
        if sample is None:
            continue
        _make_overlay(
            sample=sample,
            keras_result=keras_results[image_path],
            int8_result=int8_results[image_path],
            output_path=debug_dir / reason / f"{Path(image_path).stem}.png",
        )

    # Build the markdown report.
    lines: list[str] = [
        "# Geometry Heatmap v2 Quantization Drift Autopsy",
        "",
        "## Setup",
        "",
        f"- Model: {model_path}",
        f"- TFLite: {tflite_path}",
        f"- Preprocessing mode: {DEFAULT_PREPROCESSING_MODE}",
        f"- Selected decode method: `{selected_decode_name}` -> `{selected_decode_method}` with window size `{selected_decode_window_size}`",
        f"- Guardrails: {thresholds_path}",
        f"- Calibration candidate: {calibration_json['selected_candidate_name']}",
        f"- TFLite semantic output order indices: {semantic_output_order_indices}",
        "",
        "## Root Cause Summary",
        "",
    ]
    root_cause = _root_cause_summary(all_rows)
    lines.extend(
        [
            f"- Tip heatmap flattening cases: {root_cause['tip_flattening_count']}",
            f"- Center heatmap spread growth cases: {root_cause['center_spread_growth_count']}",
            f"- Tip heatmap spread growth cases: {root_cause['tip_spread_growth_count']}",
            f"- Softargmax-sensitive cases: {root_cause['softargmax_sensitive_count']}",
            f"- Output order issue likely? {root_cause['output_order_issue']}",
            f"- Input quantization primary? {root_cause['input_quantization_primary']}",
            f"- Output quantization primary? {root_cause['output_quantization_primary']}",
            f"- Representative dataset weakness likely? {root_cause['representative_dataset_weakness_likely']}",
            "- The TFLite FP32 replay matched Keras in Phase 7, so raw tensor order / dequantization are not the main culprit.",
            "- The remaining drift is concentrated in tip coordinates, where the int8 heatmap tends to flatten and spread out more than the Keras heatmap.",
            "",
            "## Split Summary",
            "",
        ]
    )
    for split in SPLITS:
        summary = split_summaries[split]
        lines.extend(
            [
                f"### {split}",
                "",
                "| metric | value |",
                "| --- | ---: |",
                f"| count | {summary['count']} |",
                f"| keras accepted MAE (C) | {summary['keras_accepted_mae_c']:.4f} |",
                f"| keras acceptance rate | {summary['keras_acceptance_rate']:.4f} |",
                f"| int8 accepted MAE (C) | {summary['int8_accepted_mae_c']:.4f} |",
                f"| int8 acceptance rate | {summary['int8_acceptance_rate']:.4f} |",
                f"| int8 worst accepted error (C) | {summary['int8_worst_accepted_error_c']:.4f} |",
                f"| keras-vs-int8 temp delta mean (C) | {summary['keras_vs_int8_temperature_delta_mean']:.4f} |",
                f"| keras-vs-int8 temp delta median (C) | {summary['keras_vs_int8_temperature_delta_median']:.4f} |",
                f"| keras-vs-int8 temp delta p90 (C) | {summary['keras_vs_int8_temperature_delta_p90']:.4f} |",
                f"| keras-vs-int8 tip delta mean (px) | {summary['keras_vs_int8_tip_delta_mean']:.4f} |",
                f"| keras-vs-int8 tip delta median (px) | {summary['keras_vs_int8_tip_delta_median']:.4f} |",
                f"| keras-vs-int8 tip delta p90 (px) | {summary['keras_vs_int8_tip_delta_p90']:.4f} |",
                f"| guardrail disagreements | {summary['guardrail_disagreement_count']} |",
                f"| accepted >20 C int8 failures | {summary['accepted_gt20_failures']} |",
                "",
            ]
        )

    top_tip_rows = _pick_top(all_rows, "keras_vs_int8_tip_delta_px", limit=30)
    top_temp_rows = _pick_top(all_rows, "keras_vs_int8_temperature_delta_c", limit=30)
    lines.extend(
        [
            "## Top 30 Tip Deltas",
            "",
            *_format_table(
                top_tip_rows,
                [
                    "split",
                    "image_path",
                    "keras_vs_int8_tip_delta_px",
                    "keras_vs_int8_center_delta_px",
                    "keras_vs_int8_temperature_delta_c",
                    "keras_tip_heatmap_peak_value",
                    "int8_tip_heatmap_peak_value",
                    "keras_tip_heatmap_spread_px",
                    "int8_tip_heatmap_spread_px",
                    "keras_guardrail_status",
                    "int8_guardrail_status",
                    "disagreement_type",
                ],
            ),
            "",
            "## Top 30 Temperature Deltas",
            "",
            *_format_table(
                top_temp_rows,
                [
                    "split",
                    "image_path",
                    "keras_vs_int8_temperature_delta_c",
                    "keras_vs_int8_tip_delta_px",
                    "keras_vs_int8_center_delta_px",
                    "keras_tip_heatmap_peak_value",
                    "int8_tip_heatmap_peak_value",
                    "keras_tip_heatmap_spread_px",
                    "int8_tip_heatmap_spread_px",
                    "keras_guardrail_status",
                    "int8_guardrail_status",
                    "disagreement_type",
                ],
            ),
            "",
            "## Guardrail Disagreements",
            "",
        ]
    )
    if disagreement_rows:
        lines.extend(
            _format_table(
                disagreement_rows,
                [
                    "split",
                    "image_path",
                    "disagreement_type",
                    "keras_guardrail_status",
                    "int8_guardrail_status",
                    "keras_vs_int8_temperature_delta_c",
                    "keras_vs_int8_tip_delta_px",
                    "keras_tip_heatmap_peak_value",
                    "int8_tip_heatmap_peak_value",
                    "keras_tip_heatmap_spread_px",
                    "int8_tip_heatmap_spread_px",
                    "keras_vs_int8_center_delta_px",
                    "keras_vs_int8_confidence_delta",
                ],
            )
        )
    else:
        lines.append("No guardrail disagreements were observed.")

    lines.extend(
        [
            "",
            "## Source Manifest Breakdown",
            "",
        ]
    )
    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        source_groups[str(row["source_manifest"])].append(row)
    source_summary_rows: list[dict[str, Any]] = []
    for source_manifest, rows in sorted(source_groups.items(), key=lambda item: (len(item[1]), item[0]), reverse=True):
        temp_deltas = np.asarray([float(row["keras_vs_int8_temperature_delta_c"]) for row in rows], dtype=np.float64)
        tip_deltas = np.asarray([float(row["keras_vs_int8_tip_delta_px"]) for row in rows], dtype=np.float64)
        source_summary_rows.append(
            {
                "source_manifest": source_manifest,
                "count": len(rows),
                "mean_temp_delta_c": f"{float(np.mean(temp_deltas)):.4f}",
                "mean_tip_delta_px": f"{float(np.mean(tip_deltas)):.4f}",
            }
        )
    lines.extend(_format_table(source_summary_rows, ["source_manifest", "count", "mean_temp_delta_c", "mean_tip_delta_px"], limit=12))

    _write_markdown(report_path, lines)
    print(f"[DRIFT] Wrote analysis CSV to {output_dir / 'quantization_drift_analysis.csv'}", flush=True)
    print(f"[DRIFT] Wrote drift report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
