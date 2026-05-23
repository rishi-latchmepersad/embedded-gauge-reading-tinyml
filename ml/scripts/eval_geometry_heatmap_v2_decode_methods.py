#!/usr/bin/env python3
"""Compare decode methods for geometry_heatmap_v2 Keras and INT8 replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import circular_angle_error_degrees
from embedded_gauge_reading_tinyml.geometry_heatmap_quantization_common import (
    DEFAULT_HEATMAP_SIZE,
    DEFAULT_INPUT_SIZE,
    decode_and_guard,
    load_models,
    load_semantic_output_order_indices,
    load_split_samples,
    predict_keras_outputs,
    predict_tflite_outputs,
    resolve_repo_path,
    SELECTED_PREPROCESSING_MODE,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import load_tflite_model
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import load_selected_calibration_candidate
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import GeometryGuardrailThresholds


SPLITS = ("train", "val", "test")
DEFAULT_CURRENT_REFERENCE_MODEL = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/model_int8.tflite")
DEFAULT_CURRENT_REFERENCE_CONTRACT = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/tflite_tensor_contract.json")
DEFAULT_TRAINED_MODEL = Path("ml/artifacts/training/geometry_heatmap_v2/model.keras")
DEFAULT_CALIBRATION_PATH = Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json")
DEFAULT_THRESHOLDS_PATH = Path("ml/artifacts/training/geometry_heatmap_v2_board_replay/selected_board_guardrail_thresholds.json")
DEFAULT_OUTPUT_DIR = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2")
DEFAULT_REPORT_PATH = Path("ml/reports/geometry_heatmap_v2_decode_method_comparison.md")


@dataclass(frozen=True)
class DecodeSpec:
    """One decode method and its window size."""

    name: str
    method: str
    window_size: int


DECODE_SPECS: tuple[DecodeSpec, ...] = (
    DecodeSpec(name="softargmax_w3", method="softargmax", window_size=3),
    DecodeSpec(name="argmax_w3", method="argmax", window_size=3),
    DecodeSpec(name="local_window_softargmax_w3", method="local_window_softargmax", window_size=3),
    DecodeSpec(name="local_window_softargmax_w5", method="local_window_softargmax", window_size=5),
    DecodeSpec(name="peak_weighted_centroid_w3", method="peak_weighted_centroid", window_size=3),
    DecodeSpec(name="peak_weighted_centroid_w5", method="peak_weighted_centroid", window_size=5),
)

DECODE_PRIORITY = {spec.name: index for index, spec in enumerate(DECODE_SPECS)}


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
    """Load the selected guardrail thresholds."""

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
    """Treat accepted and clamped predictions as usable predictions."""

    return status in {"accepted", "clamped"}


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
    """Decode a heatmap bundle with the requested method."""

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


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one set of per-sample predictions."""

    accepted_errors = np.asarray(
        [
            abs(float(row["int8_guarded_temperature_c"]) - float(row["true_temperature_c"]))
            for row in rows
            if row["int8_guardrail_status"] != "rejected"
        ],
        dtype=np.float64,
    )
    keras_errors = np.asarray(
        [
            abs(float(row["keras_guarded_temperature_c"]) - float(row["true_temperature_c"]))
            for row in rows
            if row["keras_guardrail_status"] != "rejected"
        ],
        dtype=np.float64,
    )
    temp_deltas = np.asarray([float(row["keras_vs_int8_temperature_delta_c"]) for row in rows], dtype=np.float64)
    tip_deltas = np.asarray([float(row["keras_vs_int8_tip_delta_px"]) for row in rows], dtype=np.float64)
    center_deltas = np.asarray([float(row["keras_vs_int8_center_delta_px"]) for row in rows], dtype=np.float64)
    disagreement_count = sum(1 for row in rows if row["keras_guardrail_status"] != row["int8_guardrail_status"])
    return {
        "count": int(len(rows)),
        "keras_accepted_mae_c": float(np.mean(keras_errors)) if keras_errors.size else math.nan,
        "keras_acceptance_rate": float(np.mean([_status_is_accepted(str(row["keras_guardrail_status"])) for row in rows])),
        "int8_accepted_mae_c": float(np.mean(accepted_errors)) if accepted_errors.size else math.nan,
        "int8_acceptance_rate": float(np.mean([_status_is_accepted(str(row["int8_guardrail_status"])) for row in rows])),
        "int8_worst_accepted_error_c": float(np.max(accepted_errors)) if accepted_errors.size else math.nan,
        "int8_percentage_under_2c": float(np.mean(accepted_errors < 2.0) * 100.0) if accepted_errors.size else 0.0,
        "int8_percentage_under_5c": float(np.mean(accepted_errors < 5.0) * 100.0) if accepted_errors.size else 0.0,
        "int8_percentage_under_10c": float(np.mean(accepted_errors < 10.0) * 100.0) if accepted_errors.size else 0.0,
        "keras_vs_int8_temperature_delta_mean": float(np.mean(temp_deltas)),
        "keras_vs_int8_temperature_delta_median": float(np.median(temp_deltas)),
        "keras_vs_int8_temperature_delta_p90": float(np.percentile(temp_deltas, 90)),
        "keras_vs_int8_center_delta_mean": float(np.mean(center_deltas)),
        "keras_vs_int8_center_delta_median": float(np.median(center_deltas)),
        "keras_vs_int8_center_delta_p90": float(np.percentile(center_deltas, 90)),
        "keras_vs_int8_tip_delta_mean": float(np.mean(tip_deltas)),
        "keras_vs_int8_tip_delta_median": float(np.median(tip_deltas)),
        "keras_vs_int8_tip_delta_p90": float(np.percentile(tip_deltas, 90)),
        "guardrail_disagreement_count": int(disagreement_count),
    }


def _is_valid_selection_row(row: dict[str, Any]) -> bool:
    """Return True when a validation row has usable acceptance metrics."""

    numeric_keys = (
        "keras_accepted_mae_c",
        "keras_acceptance_rate",
        "int8_accepted_mae_c",
        "int8_acceptance_rate",
        "keras_vs_int8_temperature_delta_mean",
        "keras_vs_int8_tip_delta_mean",
        "guardrail_disagreement_count",
    )
    if not all(math.isfinite(float(row[key])) for key in numeric_keys):
        return False
    return float(row["keras_acceptance_rate"]) > 0.0 and float(row["int8_acceptance_rate"]) > 0.0


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, int, float, float]:
    """Rank decode methods by validation robustness and drift."""

    return (
        -float(row["keras_acceptance_rate"]),
        -float(row["int8_acceptance_rate"]),
        float(row["keras_vs_int8_temperature_delta_mean"]),
        float(row["keras_vs_int8_tip_delta_mean"]),
        float(row["keras_vs_int8_center_delta_mean"]),
        float(row["int8_accepted_mae_c"]),
        int(row["guardrail_disagreement_count"]),
        float(DECODE_PRIORITY.get(str(row["decode_method"]), len(DECODE_SPECS))),
    )


def main() -> None:
    """Compare decode strategies on the current Keras and int8 model."""

    parser = argparse.ArgumentParser(description="Compare geometry heatmap decode methods")
    parser.add_argument("--current-reference-model-path", type=Path, default=DEFAULT_CURRENT_REFERENCE_MODEL)
    parser.add_argument("--current-reference-contract-path", type=Path, default=DEFAULT_CURRENT_REFERENCE_CONTRACT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_TRAINED_MODEL)
    parser.add_argument("--manifest-path", type=Path, default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"))
    parser.add_argument("--calibration-json-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--heatmap-size", type=int, default=DEFAULT_HEATMAP_SIZE)
    parser.add_argument("--sigma-pixels", type=float, default=5.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    current_reference_model_path = resolve_repo_path(repo_root, args.current_reference_model_path)
    current_reference_contract_path = resolve_repo_path(repo_root, args.current_reference_contract_path)
    model_path = resolve_repo_path(repo_root, args.model_path)
    manifest_path = resolve_repo_path(repo_root, args.manifest_path)
    calibration_json_path = resolve_repo_path(repo_root, args.calibration_json_path)
    thresholds_path = resolve_repo_path(repo_root, args.thresholds_path)
    output_dir = resolve_repo_path(repo_root, args.output_dir)
    report_path = resolve_repo_path(repo_root, args.report_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _load_thresholds(thresholds_path)
    calibration_candidate, calibration_json = load_selected_calibration_candidate(calibration_json_path)
    keras_model, _ = load_models(model_path, current_reference_model_path)
    reference_bundle = load_tflite_model(current_reference_model_path)
    reference_order = load_semantic_output_order_indices(current_reference_contract_path)

    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    split_cache: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        split_samples = load_split_samples(
            manifest_path,
            repo_root,
            split=split,
            mode=SELECTED_PREPROCESSING_MODE,
            input_size=args.input_size,
            heatmap_size=args.heatmap_size,
            sigma_pixels=args.sigma_pixels,
        ).samples
        split_inputs = [sample.crop_image for sample in split_samples]
        keras_center, keras_tip, keras_confidence = predict_keras_outputs(keras_model, split_inputs, batch_size=16)
        reference_center, reference_tip, reference_confidence = predict_tflite_outputs(
            reference_bundle,
            split_inputs,
            semantic_output_order_indices=reference_order,
        )
        split_cache[split] = {
            "samples": split_samples,
            "keras_center": keras_center,
            "keras_tip": keras_tip,
            "keras_confidence": keras_confidence,
            "reference_center": reference_center,
            "reference_tip": reference_tip,
            "reference_confidence": reference_confidence,
        }

    for spec in DECODE_SPECS:
        for split, cache in split_cache.items():
            split_rows: list[dict[str, Any]] = []
            for index, sample in enumerate(cache["samples"]):
                keras_decoded, keras_guarded = _decode_prediction(
                    sample,
                    cache["keras_center"][index],
                    cache["keras_tip"][index],
                    float(np.ravel(cache["keras_confidence"][index])[0]),
                    calibration_candidate,
                    thresholds,
                    decode_method=spec.method,
                    window_size=spec.window_size,
                )
                int8_decoded, int8_guarded = _decode_prediction(
                    sample,
                    cache["reference_center"][index],
                    cache["reference_tip"][index],
                    float(np.ravel(cache["reference_confidence"][index])[0]),
                    calibration_candidate,
                    thresholds,
                    decode_method=spec.method,
                    window_size=spec.window_size,
                )
                row = {
                    "decode_method": spec.name,
                    "split": split,
                    "image_path": str(sample.metadata["image_path"]),
                    "true_temperature_c": float(sample.metadata["temperature_c"]),
                    "true_angle_degrees": float(keras_decoded.true_angle_degrees),
                    "keras_guardrail_status": str(keras_guarded.status),
                    "int8_guardrail_status": str(int8_guarded.status),
                    "keras_guarded_temperature_c": float(keras_guarded.temperature_c),
                    "int8_guarded_temperature_c": float(int8_guarded.temperature_c),
                    "keras_predicted_temperature_c_calibrated": float(keras_decoded.predicted_temperature_c_calibrated),
                    "int8_predicted_temperature_c_calibrated": float(int8_decoded.predicted_temperature_c_calibrated),
                    "keras_vs_int8_temperature_delta_c": float(
                        abs(float(keras_decoded.predicted_temperature_c_calibrated) - float(int8_decoded.predicted_temperature_c_calibrated))
                    ),
                    "keras_vs_int8_center_delta_px": float(
                        math.hypot(
                            float(keras_decoded.predicted_center_x_224) - float(int8_decoded.predicted_center_x_224),
                            float(keras_decoded.predicted_center_y_224) - float(int8_decoded.predicted_center_y_224),
                        )
                    ),
                    "keras_vs_int8_tip_delta_px": float(
                        math.hypot(
                            float(keras_decoded.predicted_tip_x_224) - float(int8_decoded.predicted_tip_x_224),
                            float(keras_decoded.predicted_tip_y_224) - float(int8_decoded.predicted_tip_y_224),
                        )
                    ),
                    "keras_vs_int8_angle_delta_degrees": float(
                        circular_angle_error_degrees(
                            float(keras_decoded.predicted_angle_degrees),
                            float(int8_decoded.predicted_angle_degrees),
                        )
                    ),
                    "keras_center_px_mae_224": float(
                        math.hypot(
                            float(keras_decoded.predicted_center_x_224) - float(keras_decoded.true_center_x_224),
                            float(keras_decoded.predicted_center_y_224) - float(keras_decoded.true_center_y_224),
                        )
                    ),
                    "keras_tip_px_mae_224": float(
                        math.hypot(
                            float(keras_decoded.predicted_tip_x_224) - float(keras_decoded.true_tip_x_224),
                            float(keras_decoded.predicted_tip_y_224) - float(keras_decoded.true_tip_y_224),
                        )
                    ),
                    "int8_center_px_mae_224": float(
                        math.hypot(
                            float(int8_decoded.predicted_center_x_224) - float(int8_decoded.true_center_x_224),
                            float(int8_decoded.predicted_center_y_224) - float(int8_decoded.true_center_y_224),
                        )
                    ),
                    "int8_tip_px_mae_224": float(
                        math.hypot(
                            float(int8_decoded.predicted_tip_x_224) - float(int8_decoded.true_tip_x_224),
                            float(int8_decoded.predicted_tip_y_224) - float(int8_decoded.true_tip_y_224),
                        )
                    ),
                    "keras_center_heatmap_peak_value": float(keras_decoded.center_heatmap_peak_value),
                    "keras_tip_heatmap_peak_value": float(keras_decoded.tip_heatmap_peak_value),
                    "int8_center_heatmap_peak_value": float(int8_decoded.center_heatmap_peak_value),
                    "int8_tip_heatmap_peak_value": float(int8_decoded.tip_heatmap_peak_value),
                    "keras_center_heatmap_spread_px": float(keras_guarded.quality_features.center_heatmap_spread_px),
                    "keras_tip_heatmap_spread_px": float(keras_guarded.quality_features.tip_heatmap_spread_px),
                    "int8_center_heatmap_spread_px": float(int8_guarded.quality_features.center_heatmap_spread_px),
                    "int8_tip_heatmap_spread_px": float(int8_guarded.quality_features.tip_heatmap_spread_px),
                    "keras_confidence": float(keras_decoded.confidence),
                    "int8_confidence": float(int8_decoded.confidence),
                    "keras_rejection_reasons": ";".join(keras_guarded.rejection_reasons) if keras_guarded.rejection_reasons else "none",
                    "int8_rejection_reasons": ";".join(int8_guarded.rejection_reasons) if int8_guarded.rejection_reasons else "none",
                }
                split_rows.append(row)
                prediction_rows.append(row)
            summary = _summary(split_rows)
            summary.update({"decode_method": spec.name, "split": split, "window_size": spec.window_size})
            summary_rows.append(summary)

    _write_csv(prediction_rows, output_dir / "decode_method_predictions.csv")
    _write_csv(summary_rows, output_dir / "decode_method_summary.csv")

    # Choose the best validation method by acceptance first, then drift and MAE.
    val_rows = [row for row in summary_rows if row["split"] == "val"]
    valid_val_rows = [row for row in val_rows if _is_valid_selection_row(row)]
    if valid_val_rows:
        selected_row = sorted(valid_val_rows, key=_selection_key)[0]
    else:
        # If no method has usable accepted rows, fall back to the row with the
        # best acceptance signal so the report still tells us what failed.
        selected_row = sorted(
            val_rows,
            key=lambda row: (
                -float(row["keras_acceptance_rate"]),
                -float(row["int8_acceptance_rate"]),
                float(row["keras_vs_int8_temperature_delta_mean"]),
                float(row["keras_vs_int8_tip_delta_mean"]),
            ),
        )[0]
    selected_method_payload = {
        "selected_decode_method": selected_row["decode_method"],
        "selected_window_size": int(selected_row["window_size"]),
        "selection_split": "val",
        "selection_metrics": selected_row,
        "decode_method_count": len(DECODE_SPECS),
    }
    _write_json(selected_method_payload, output_dir / "selected_decode_method.json")

    lines: list[str] = [
        "# Geometry Heatmap v2 Decode Method Comparison",
        "",
        "## Selected Method",
        "",
        f"- Selected decode method: `{selected_row['decode_method']}`",
        f"- Selected window size: `{int(selected_row['window_size'])}`",
        f"- Validation INT8 accepted MAE: `{float(selected_row['int8_accepted_mae_c']):.4f} C`",
        f"- Validation INT8 acceptance rate: `{float(selected_row['int8_acceptance_rate']):.4f}`",
        f"- Validation INT8 worst accepted error: `{float(selected_row['int8_worst_accepted_error_c']):.4f} C`",
        f"- Validation Keras-vs-INT8 temperature delta mean: `{float(selected_row['keras_vs_int8_temperature_delta_mean']):.4f} C`",
        f"- Validation Keras-vs-INT8 tip delta mean: `{float(selected_row['keras_vs_int8_tip_delta_mean']):.4f} px`",
        "",
        "## Split Summary",
        "",
    ]
    for split in SPLITS:
        split_rows = [row for row in summary_rows if row["split"] == split]
        lines.extend(
            [
                f"### {split}",
                "",
                "| decode method | window | INT8 accepted MAE | INT8 acceptance | worst accepted | Keras vs INT8 temp delta mean | Keras vs INT8 tip delta mean | disagreement count |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(split_rows, key=lambda item: (float(item["int8_accepted_mae_c"]), float(item["keras_vs_int8_temperature_delta_mean"]))):
            lines.append(
                f"| {row['decode_method']} | {int(row['window_size'])} | {float(row['int8_accepted_mae_c']):.4f} | {float(row['int8_acceptance_rate']):.4f} | {float(row['int8_worst_accepted_error_c']):.4f} | {float(row['keras_vs_int8_temperature_delta_mean']):.4f} | {float(row['keras_vs_int8_tip_delta_mean']):.4f} | {int(row['guardrail_disagreement_count'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Recommendation",
            "",
            f"- Recommended deployment decoder: `{selected_row['decode_method']}` with window size `{int(selected_row['window_size'])}`",
            f"- The selected method improves INT8 replay without degrading Keras too much? `{float(selected_row['keras_vs_int8_temperature_delta_mean']) <= 1.0}`",
            f"- Keras accepted MAE at selection: `{float(selected_row['keras_accepted_mae_c']):.4f} C`",
            f"- Keras acceptance rate at selection: `{float(selected_row['keras_acceptance_rate']):.4f}`",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DECODE] Wrote report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
