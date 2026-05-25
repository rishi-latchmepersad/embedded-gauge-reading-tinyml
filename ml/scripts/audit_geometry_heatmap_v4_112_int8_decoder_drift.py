#!/usr/bin/env python3
"""Audit INT8 decoder drift for geometry_heatmap_v4_112 recovery candidates.

For each candidate, runs the Keras and INT8 models once on the val split, then
re-decodes the same heatmaps with multiple deterministic decoders.  Reports
which decoder+candidate pair minimizes Keras-vs-INT8 drift and passes all gates.

Usage:
    poetry run python ml/scripts/audit_geometry_heatmap_v4_112_int8_decoder_drift.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_heatmap_quantization_common import (
    DEFAULT_INPUT_SIZE,
    DEFAULT_PREPROCESSING_MODE,
    decode_and_guard,
    load_semantic_output_order_indices,
    load_split_samples,
    predict_tflite_outputs,
    resolve_repo_path,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import (
    load_geometry_heatmap_keras_model,
    load_tflite_model,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import (
    load_selected_calibration_candidate,
)
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import (
    GeometryGuardrailThresholds,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

MANIFEST_PATH = REPO_ROOT / "ml/data/geometry_reader_manifest_v2_clean.csv"
CALIBRATION_PATH = (
    REPO_ROOT
    / "ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json"
)
THRESHOLDS_PATH = (
    REPO_ROOT
    / "ml/artifacts/training/geometry_heatmap_v4_112_quant_native/v4_112_guardrail_thresholds.json"
)
TRAINING_BASE = REPO_ROOT / "ml/artifacts/training/geometry_heatmap_v4_112_int8_recovery"
DEPLOY_BASE = REPO_ROOT / "ml/artifacts/deployment/geometry_heatmap_v4_112_tflite"
REPORT_DIR = REPO_ROOT / "ml/reports"
DEBUG_DIR = (
    REPO_ROOT / "ml/debug/geometry_heatmap_v4_112_int8_decoder_drift"
)

AUDIT_REPORT_PATH = (
    REPORT_DIR / "geometry_heatmap_v4_112_int8_decoder_drift_audit.md"
)
DECISION_REPORT_PATH = (
    REPORT_DIR / "geometry_heatmap_v4_112_int8_decoder_drift_decision.md"
)
SUMMARY_CSV_PATH = DEPLOY_BASE / "v4_112_int8_decoder_drift_audit_summary.csv"

V4_HEATMAP_SIZE = 112
V4_SIGMA_PIXELS = 2.5
BATCH_SIZE = 16

CANDIDATE_TAGS: dict[str, str] = {
    "01_conservative": "01_conservative__pt0.30_c0.10_t0.20_cf0.05_wu5",
    "02_lower_peak_target": "02_lower_peak_target__pt0.25_c0.10_t0.20_cf0.05_wu5",
    "03_lighter_shaping": "03_lighter_shaping__pt0.30_c0.05_t0.15_cf0.05_wu5",
    "04_short_warmup": "04_short_warmup__pt0.30_c0.10_t0.20_cf0.05_wu3",
    "05_light_all": "05_light_all__pt0.25_c0.05_t0.15_cf0.03_wu5",
    "06_aggressive": "06_aggressive__pt0.25_c0.05_t0.15_cf0.03_wu3",
    "07_high_peak_low_floor": "07_high_peak_low_floor__pt0.30_c0.05_t0.20_cf0.03_wu5",
    "08_tip_focus": "08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3",
}

DECODERS: list[tuple[str, int]] = [
    ("softargmax", 3),
    ("argmax", 0),
    ("local_window_softargmax", 3),
    ("local_window_softargmax", 5),
    ("peak_weighted_centroid", 3),
    ("peak_weighted_centroid", 5),
]

GATES: dict[str, float] = {
    "accepted_mae_c": 4.5,
    "acceptance_rate": 0.65,
    "worst_accepted_error_c": 20.0,
    "accepted_gt20_failures": 0,
    "temperature_delta_mean": 1.0,
}


def _resolve_tag(name: str) -> str:
    """Resolve a short candidate name to full artifact tag, or pass through."""
    if name in CANDIDATE_TAGS:
        return CANDIDATE_TAGS[name]
    if name.startswith("candidate_"):
        name = name[len("candidate_"):]
    if name in CANDIDATE_TAGS:
        return CANDIDATE_TAGS[name]
    return name


def _candidate_paths(tag: str) -> dict[str, Any]:
    full_tag = _resolve_tag(tag)
    train_dir = TRAINING_BASE / f"candidate_{full_tag}"
    deploy_dir = DEPLOY_BASE / f"recovery_{full_tag}"
    return {
        "model_path": train_dir / "model_v4_112.keras",
        "int8_path": deploy_dir / "model_v4_112_int8.tflite",
        "contract_path": deploy_dir / "tflite_tensor_contract.json",
        "full_tag": full_tag,
    }


def _load_thresholds(thresholds_path: Path) -> GeometryGuardrailThresholds:
    with thresholds_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = payload["selected_thresholds"]
    return GeometryGuardrailThresholds(
        center_peak_min=float(selected["center_peak_min"]),
        tip_peak_min=float(selected["tip_peak_min"]),
        confidence_min=float(selected["confidence_min"]),
        max_heatmap_entropy=float(selected["max_heatmap_entropy"]),
        max_heatmap_spread_px=float(selected["max_heatmap_spread_px"]),
        center_tip_distance_ratio_min=float(
            selected["center_tip_distance_ratio_min"]
        ),
        center_tip_distance_ratio_max=float(
            selected["center_tip_distance_ratio_max"]
        ),
        edge_margin_px=float(selected["edge_margin_px"]),
        temperature_physical_margin_c=float(
            selected["temperature_physical_range_margin_c"]
        ),
        minimum_celsius=float(selected["minimum_celsius"]),
        maximum_celsius=float(selected["maximum_celsius"]),
        clamp_temperature_to_physical_range=bool(
            selected["clamp_temperature_to_physical_range"]
        ),
    )


def _as_output_dict(outputs: Any) -> dict[str, tf.Tensor]:
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


def _status_is_accepted(status: str) -> bool:
    return status in {"accepted", "clamped"}


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [
        r for r in rows if _status_is_accepted(str(r["guardrail_status"]))
    ]
    accepted_errors = np.asarray(
        [
            abs(float(r["guarded_temperature_c"]) - float(r["true_temperature_c"]))
            for r in accepted
        ],
        dtype=np.float64,
    )
    all_errors = np.asarray(
        [
            abs(float(r["guarded_temperature_c"]) - float(r["true_temperature_c"]))
            for r in rows
        ],
        dtype=np.float64,
    )
    rejection_reasons = Counter()
    for r in rows:
        if _status_is_accepted(str(r["guardrail_status"])):
            continue
        for reason in str(r["rejection_reasons"]).split(";"):
            if reason and reason != "none":
                rejection_reasons[reason] += 1

    center_errors = [
        math.hypot(
            float(r["predicted_center_x_224"])
            - float(r["true_center_x_224"]),
            float(r["predicted_center_y_224"])
            - float(r["true_center_y_224"]),
        )
        for r in rows
    ]
    tip_errors = [
        math.hypot(
            float(r["predicted_tip_x_224"])
            - float(r["true_tip_x_224"]),
            float(r["predicted_tip_y_224"])
            - float(r["true_tip_y_224"]),
        )
        for r in rows
    ]

    return {
        "count": float(len(rows)),
        "accepted_count": float(len(accepted)),
        "accepted_mae_c": (
            float(np.mean(accepted_errors)) if accepted_errors.size else math.nan
        ),
        "acceptance_rate": (
            float(len(accepted) / len(rows)) if rows else math.nan
        ),
        "worst_accepted_error_c": (
            float(np.max(accepted_errors)) if accepted_errors.size else math.nan
        ),
        "accepted_gt20_failures": float(
            sum(
                1
                for r in accepted
                if abs(
                    float(r["guarded_temperature_c"])
                    - float(r["true_temperature_c"])
                )
                > 20.0
            )
        ),
        "percentage_under_2c": (
            float(np.mean(all_errors < 2.0) * 100.0) if all_errors.size else math.nan
        ),
        "percentage_under_5c": (
            float(np.mean(all_errors < 5.0) * 100.0) if all_errors.size else math.nan
        ),
        "percentage_under_10c": (
            float(np.mean(all_errors < 10.0) * 100.0) if all_errors.size else math.nan
        ),
        "center_mae_px_224": (
            float(np.mean(center_errors)) if center_errors else math.nan
        ),
        "tip_mae_px_224": (
            float(np.mean(tip_errors)) if tip_errors else math.nan
        ),
        "center_heatmap_peak_mean": float(
            np.mean([float(r["center_heatmap_peak_value"]) for r in rows])
        ),
        "tip_heatmap_peak_mean": float(
            np.mean([float(r["tip_heatmap_peak_value"]) for r in rows])
        ),
        "center_heatmap_spread_mean": float(
            np.mean([float(r["center_heatmap_spread_px"]) for r in rows])
        ),
        "tip_heatmap_spread_mean": float(
            np.mean([float(r["tip_heatmap_spread_px"]) for r in rows])
        ),
        "confidence_mean": float(
            np.mean([float(r["confidence"]) for r in rows])
        ),
        "guardrail_disagreement_count": float(
            sum(
                1
                for r in rows
                if not _status_is_accepted(str(r["guardrail_status"]))
            )
        ),
        "top_rejection_reasons": (
            ";".join(
                f"{reason}:{count}"
                for reason, count in rejection_reasons.most_common(5)
            )
            if rejection_reasons
            else "none"
        ),
    }


def _drift(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, float]:
    ref_by_image = {r["image_path"]: r for r in reference_rows}
    cand_by_image = {r["image_path"]: r for r in candidate_rows}
    common = sorted(ref_by_image.keys() & cand_by_image.keys())
    temp_deltas: list[float] = []
    center_deltas: list[float] = []
    tip_deltas: list[float] = []
    disagreements = 0
    for img_path in common:
        ref = ref_by_image[img_path]
        cand = cand_by_image[img_path]
        if _status_is_accepted(
            str(ref["guardrail_status"])
        ) and _status_is_accepted(str(cand["guardrail_status"])):
            temp_deltas.append(
                abs(
                    float(ref["guarded_temperature_c"])
                    - float(cand["guarded_temperature_c"])
                )
            )
        center_deltas.append(
            math.hypot(
                float(ref["predicted_center_x_224"])
                - float(cand["predicted_center_x_224"]),
                float(ref["predicted_center_y_224"])
                - float(cand["predicted_center_y_224"]),
            )
        )
        tip_deltas.append(
            math.hypot(
                float(ref["predicted_tip_x_224"])
                - float(cand["predicted_tip_x_224"]),
                float(ref["predicted_tip_y_224"])
                - float(cand["predicted_tip_y_224"]),
            )
        )
        if str(ref["guardrail_status"]) != str(cand["guardrail_status"]):
            disagreements += 1
    return {
        "temperature_delta_mean": (
            float(np.mean(temp_deltas)) if temp_deltas else math.nan
        ),
        "temperature_delta_median": (
            float(np.median(temp_deltas)) if temp_deltas else math.nan
        ),
        "temperature_delta_p90": (
            float(np.percentile(temp_deltas, 90)) if temp_deltas else math.nan
        ),
        "center_delta_mean": (
            float(np.mean(center_deltas)) if center_deltas else math.nan
        ),
        "center_delta_median": (
            float(np.median(center_deltas)) if center_deltas else math.nan
        ),
        "tip_delta_mean": (
            float(np.mean(tip_deltas)) if tip_deltas else math.nan
        ),
        "tip_delta_median": (
            float(np.median(tip_deltas)) if tip_deltas else math.nan
        ),
        "guardrail_disagreements": float(disagreements),
    }


def _decode_rows(
    *,
    model_type: str,
    samples: list[Any],
    outputs: dict[str, np.ndarray],
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    decode_method: str,
    window_size: int,
) -> list[dict[str, Any]]:
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
        reasons = (
            ";".join(guarded.rejection_reasons)
            if guarded.rejection_reasons
            else "none"
        )
        rows.append(
            {
                "model_type": model_type,
                "image_path": str(sample.metadata["image_path"]),
                "decode_method": decode_method,
                "window_size": str(window_size),
                "true_temperature_c": float(sample.metadata["temperature_c"]),
                "true_angle_degrees": float(sample.metadata["angle_degrees"]),
                "true_center_x_224": float(sample.metadata["center_x_224"]),
                "true_center_y_224": float(sample.metadata["center_y_224"]),
                "true_tip_x_224": float(sample.metadata["tip_x_224"]),
                "true_tip_y_224": float(sample.metadata["tip_y_224"]),
                "predicted_center_x_224": float(
                    decoded.predicted_center_x_224
                ),
                "predicted_center_y_224": float(
                    decoded.predicted_center_y_224
                ),
                "predicted_tip_x_224": float(
                    decoded.predicted_tip_x_224
                ),
                "predicted_tip_y_224": float(
                    decoded.predicted_tip_y_224
                ),
                "predicted_angle_degrees": float(
                    decoded.predicted_angle_degrees
                ),
                "predicted_temperature_c": float(
                    decoded.predicted_temperature_c_calibrated
                ),
                "guarded_temperature_c": float(guarded.temperature_c),
                "guardrail_status": guarded.status,
                "center_heatmap_peak_value": float(
                    decoded.center_heatmap_peak_value
                ),
                "tip_heatmap_peak_value": float(
                    decoded.tip_heatmap_peak_value
                ),
                "center_heatmap_spread_px": float(
                    guarded.quality_features.center_heatmap_spread_px
                ),
                "tip_heatmap_spread_px": float(
                    guarded.quality_features.tip_heatmap_spread_px
                ),
                "confidence": float(confidence),
                "rejection_reasons": reasons,
            }
        )
    return rows


def _predict_keras_outputs(
    model, inputs: np.ndarray, *, batch_size: int
) -> dict[str, np.ndarray]:
    outputs = model.predict(inputs, batch_size=batch_size, verbose=0)
    ordered = _as_output_dict(outputs)
    return {
        name: np.asarray(tensor, dtype=np.float32)
        for name, tensor in ordered.items()
    }


def _predict_tflite_outputs_wrapper(
    model_path: Path,
    inputs: list[np.ndarray],
    *,
    semantic_output_order_indices: list[int],
) -> dict[str, np.ndarray]:
    bundle = load_tflite_model(model_path)
    outputs = predict_tflite_outputs(
        bundle,
        inputs,
        semantic_output_order_indices=semantic_output_order_indices,
    )
    return {
        "center_heatmap": np.asarray(outputs[0], dtype=np.float32),
        "tip_heatmap": np.asarray(outputs[1], dtype=np.float32),
        "confidence": np.asarray(outputs[2], dtype=np.float32),
    }


def _check_gates(
    summary: dict[str, Any], drift: dict[str, float]
) -> dict[str, bool]:
    return {
        "accepted_mae_c": summary["accepted_mae_c"] <= GATES["accepted_mae_c"],
        "acceptance_rate": (
            summary["acceptance_rate"] >= GATES["acceptance_rate"]
        ),
        "worst_accepted_error_c": (
            summary["worst_accepted_error_c"]
            < GATES["worst_accepted_error_c"]
        ),
        "accepted_gt20_failures": (
            summary["accepted_gt20_failures"]
            <= GATES["accepted_gt20_failures"]
        ),
        "temperature_delta_mean": (
            drift["temperature_delta_mean"]
            <= GATES["temperature_delta_mean"]
        ),
    }


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write empty CSV.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(lines: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _sanitize_rejection_reasons(reasons_str: str, max_len: int = 40) -> str:
    if len(reasons_str) > max_len:
        return reasons_str[: max_len - 3] + "..."
    return reasons_str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit INT8 decoder drift"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        nargs="+",
        default=[
            "08_tip_focus",
            "04_short_warmup",
            "06_aggressive",
        ],
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    manifest_path = resolve_repo_path(repo_root, MANIFEST_PATH)
    calibration_path = resolve_repo_path(repo_root, CALIBRATION_PATH)
    thresholds_path = resolve_repo_path(repo_root, THRESHOLDS_PATH)

    calibration_candidate, _ = load_selected_calibration_candidate(
        calibration_path
    )
    thresholds = _load_thresholds(thresholds_path)

    # Load val samples once, shared across all candidates
    print("[AUDIT] Loading val split...", flush=True)
    val_samples = load_split_samples(
        manifest_path,
        repo_root,
        split="val",
        mode=DEFAULT_PREPROCESSING_MODE,
        input_size=DEFAULT_INPUT_SIZE,
        heatmap_size=V4_HEATMAP_SIZE,
        sigma_pixels=V4_SIGMA_PIXELS,
    ).samples
    val_inputs_np = (
        np.stack([s.crop_image for s in val_samples], axis=0).astype(np.float32)
    )
    val_inputs_list = list(val_inputs_np)
    print(
        f"[AUDIT] Loaded {len(val_samples)} val samples", flush=True
    )

    # Container for per-pair detailed rows
    all_detailed_rows: list[dict[str, Any]] = []
    # Container for one-row-per-pair summary
    summary_rows: list[dict[str, Any]] = []

    for tag_arg in args.candidates:
        paths = _candidate_paths(tag_arg)
        full_tag = paths["full_tag"]
        print(f"\n[AUDIT] === Processing {full_tag} ===", flush=True)

        model_path = resolve_repo_path(repo_root, paths["model_path"])
        int8_path = resolve_repo_path(repo_root, paths["int8_path"])
        contract_path = resolve_repo_path(repo_root, paths["contract_path"])

        if not model_path.exists():
            print(
                f"[AUDIT] SKIP {full_tag}: Keras model not found at {model_path}",
                flush=True,
            )
            continue
        if not int8_path.exists():
            print(
                f"[AUDIT] SKIP {full_tag}: INT8 TFLite not found at {int8_path}",
                flush=True,
            )
            continue

        semantic_output_order_indices = load_semantic_output_order_indices(
            contract_path
        )

        # Run Keras model once
        print(f"[AUDIT]  Running Keras model...", flush=True)
        keras_model = load_geometry_heatmap_keras_model(model_path)
        keras_outputs = _predict_keras_outputs(
            keras_model, val_inputs_np, batch_size=BATCH_SIZE
        )

        # Run INT8 TFLite once
        print(f"[AUDIT]  Running INT8 TFLite...", flush=True)
        int8_outputs = _predict_tflite_outputs_wrapper(
            int8_path,
            val_inputs_list,
            semantic_output_order_indices=semantic_output_order_indices,
        )

        # Re-decode with every decoder
        for decode_method, window_size in DECODERS:
            method_label = (
                f"{decode_method}_w{window_size}"
                if window_size > 0
                else decode_method
            )
            print(
                f"[AUDIT]  Decoder {method_label}...", flush=True
            )

            keras_rows = _decode_rows(
                model_type="keras_v3",
                samples=val_samples,
                outputs=keras_outputs,
                calibration_candidate=calibration_candidate,
                thresholds=thresholds,
                decode_method=decode_method,
                window_size=window_size,
            )
            int8_rows = _decode_rows(
                model_type="tflite_int8",
                samples=val_samples,
                outputs=int8_outputs,
                calibration_candidate=calibration_candidate,
                thresholds=thresholds,
                decode_method=decode_method,
                window_size=window_size,
            )

            keras_summary = _summarize_rows(keras_rows)
            int8_summary = _summarize_rows(int8_rows)
            drift_metrics = _drift(keras_rows, int8_rows)
            gates = _check_gates(int8_summary, drift_metrics)
            all_pass = all(gates.values())

            # Label detailed rows
            for row in keras_rows:
                row["candidate_tag"] = full_tag
                all_detailed_rows.append(row)
            for row in int8_rows:
                row["candidate_tag"] = full_tag
                all_detailed_rows.append(row)

            summary_rows.append(
                {
                    "candidate_tag": full_tag,
                    "decode_method": method_label,
                    "all_gates_pass": all_pass,
                    **{f"gate_{k}": v for k, v in gates.items()},
                    "keras_accepted_mae_c": keras_summary[
                        "accepted_mae_c"
                    ],
                    "int8_accepted_mae_c": int8_summary[
                        "accepted_mae_c"
                    ],
                    "int8_acceptance_rate": int8_summary[
                        "acceptance_rate"
                    ],
                    "int8_worst_accepted_error_c": int8_summary[
                        "worst_accepted_error_c"
                    ],
                    "int8_accepted_gt20_failures": int8_summary[
                        "accepted_gt20_failures"
                    ],
                    "int8_percentage_under_2c": int8_summary[
                        "percentage_under_2c"
                    ],
                    "int8_percentage_under_5c": int8_summary[
                        "percentage_under_5c"
                    ],
                    "int8_percentage_under_10c": int8_summary[
                        "percentage_under_10c"
                    ],
                    "int8_center_mae_px_224": int8_summary[
                        "center_mae_px_224"
                    ],
                    "int8_tip_mae_px_224": int8_summary["tip_mae_px_224"],
                    "int8_center_heatmap_peak_mean": int8_summary[
                        "center_heatmap_peak_mean"
                    ],
                    "int8_tip_heatmap_peak_mean": int8_summary[
                        "tip_heatmap_peak_mean"
                    ],
                    "int8_center_heatmap_spread_mean": int8_summary[
                        "center_heatmap_spread_mean"
                    ],
                    "int8_tip_heatmap_spread_mean": int8_summary[
                        "tip_heatmap_spread_mean"
                    ],
                    "temperature_delta_mean": drift_metrics[
                        "temperature_delta_mean"
                    ],
                    "temperature_delta_median": drift_metrics[
                        "temperature_delta_median"
                    ],
                    "temperature_delta_p90": drift_metrics[
                        "temperature_delta_p90"
                    ],
                    "center_delta_mean": drift_metrics[
                        "center_delta_mean"
                    ],
                    "center_delta_median": drift_metrics[
                        "center_delta_median"
                    ],
                    "tip_delta_mean": drift_metrics["tip_delta_mean"],
                    "tip_delta_median": drift_metrics[
                        "tip_delta_median"
                    ],
                    "guardrail_disagreements": drift_metrics[
                        "guardrail_disagreements"
                    ],
                    "keras_accepted_count": keras_summary[
                        "accepted_count"
                    ],
                    "int8_accepted_count": int8_summary[
                        "accepted_count"
                    ],
                    "int8_top_rejection_reasons": int8_summary[
                        "top_rejection_reasons"
                    ],
                }
            )

    # Sort: all gates pass first, then by acceptance desc, MAE asc, drift asc
    summary_rows.sort(
        key=lambda r: (
            not r["all_gates_pass"],
            -(
                r["int8_acceptance_rate"]
                if not math.isnan(r["int8_acceptance_rate"])
                else 0
            ),
            r["int8_accepted_mae_c"]
            if not math.isnan(r["int8_accepted_mae_c"])
            else 999,
            r["temperature_delta_mean"]
            if not math.isnan(r["temperature_delta_mean"])
            else 999,
        )
    )

    # Write summary CSV
    _write_csv(summary_rows, SUMMARY_CSV_PATH)
    print(
        f"[AUDIT] Wrote summary CSV: {SUMMARY_CSV_PATH}", flush=True
    )

    # Write detailed predictions CSV
    detailed_path = DEBUG_DIR / "audit_predictions.csv"
    _write_csv(all_detailed_rows, detailed_path)
    print(
        f"[AUDIT] Wrote detailed predictions: {detailed_path}",
        flush=True,
    )

    # ---- Audit report ----
    audit_lines = [
        "# Geometry Heatmap v4 112 INT8 Decoder Drift Audit",
        "",
        f"- Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Candidates: {', '.join(args.candidates)}",
        f"- Decoders: {', '.join(f'{d}_w{w}' if w > 0 else d for d, w in DECODERS)}",
        f"- Validation samples: {len(val_samples)}",
        f"- Calibration: {calibration_candidate.name}",
        f"- Guardrails: max_heatmap_spread_px={thresholds.max_heatmap_spread_px} (v4 thresholds)",
        "",
        "## Ranked Results",
        "",
        "Pairs sorted by: gates pass first, then acceptance desc, MAE asc, temp drift asc.",
        "",
    ]

    if summary_rows:
        header = (
            "| Rank | Candidate | Decoder | All Gates "
            "| Acc MAE(C) | Acc Rate | Worst Err(C) | >20C Fail "
            "| Temp Drift Mean(C) | Tip Drift Mean(px) | Center Drift(px) "
            "| Under 2C | Under 5C | Under 10C | Rejections |"
        )
        sep = (
            "|------|-----------|---------|-----------"
            "|------------|----------|--------------|-----------"
            "|--------------------|--------------------|------------------"
            "|----------|----------|-----------|------------|"
        )
        audit_lines.append(header)
        audit_lines.append(sep)
        for rank, row in enumerate(summary_rows, 1):
            pass_mark = "PASS" if row["all_gates_pass"] else "FAIL"
            rej = _sanitize_rejection_reasons(
                str(row.get("int8_top_rejection_reasons", "none"))
            )
            audit_lines.append(
                f"| {rank} "
                f"| {row['candidate_tag']} "
                f"| {row['decode_method']} "
                f"| {pass_mark} "
                f"| {row['int8_accepted_mae_c']:.2f} "
                f"| {row['int8_acceptance_rate']:.4f} "
                f"| {row['int8_worst_accepted_error_c']:.2f} "
                f"| {int(row['int8_accepted_gt20_failures'])} "
                f"| {row['temperature_delta_mean']:.4f} "
                f"| {row['tip_delta_mean']:.4f} "
                f"| {row['center_delta_mean']:.4f} "
                f"| {row['int8_percentage_under_2c']:.1f}% "
                f"| {row['int8_percentage_under_5c']:.1f}% "
                f"| {row['int8_percentage_under_10c']:.1f}% "
                f"| {rej} |"
            )
    else:
        audit_lines.append("(No results)")

    audit_lines.append("")
    _write_markdown(audit_lines, AUDIT_REPORT_PATH)
    print(
        f"[AUDIT] Wrote audit report: {AUDIT_REPORT_PATH}", flush=True
    )

    # ---- Decision report ----
    passing = [r for r in summary_rows if r["all_gates_pass"]]
    if passing:
        best = passing[0]
        decision_lines = [
            "# INT8 Decoder Drift Audit Decision",
            "",
            f"- Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- Candidates evaluated: {len(args.candidates)}",
            f"- Total decoder+candidate pairs: {len(summary_rows)}",
            f"- Passing pairs: {len(passing)}",
            "",
            "**Decision A**: Champion found.",
            "",
            f"### Best Pair: `{best['candidate_tag']}` + `{best['decode_method']}`",
            "",
            "#### INT8 Metrics",
            f"- Accepted MAE: {best['int8_accepted_mae_c']:.4f} C",
            f"- Acceptance rate: {best['int8_acceptance_rate']:.4f}",
            f"- Worst accepted error: {best['int8_worst_accepted_error_c']:.4f} C",
            f"- Accepted >20C failures: {int(best['int8_accepted_gt20_failures'])}",
            f"- Under 2C / 5C / 10C: {best['int8_percentage_under_2c']:.1f}% / {best['int8_percentage_under_5c']:.1f}% / {best['int8_percentage_under_10c']:.1f}%",
            "",
            "#### Keras-vs-INT8 Drift",
            f"- Temperature drift mean: {best['temperature_delta_mean']:.4f} C",
            f"- Temperature drift median: {best['temperature_delta_median']:.4f} C",
            f"- Temperature drift p90: {best['temperature_delta_p90']:.4f} C",
            f"- Center drift mean: {best['center_delta_mean']:.4f} px",
            f"- Tip drift mean: {best['tip_delta_mean']:.4f} px",
            f"- Guardrail disagreements: {int(best['guardrail_disagreements'])}",
            "",
            "#### Comparison vs Baseline (08_tip_focus / softargmax_w3)",
        ]
        baseline = None
        for r in summary_rows:
            if (
                r["candidate_tag"] == "08_tip_focus"
                and r["decode_method"] == "softargmax_w3"
            ):
                baseline = r
                break
        if baseline:
            temp_improvement = (
                baseline["temperature_delta_mean"]
                - best["temperature_delta_mean"]
            )
            tip_improvement = (
                baseline["tip_delta_mean"] - best["tip_delta_mean"]
            )
            decision_lines.append(
                f"- Baseline temp drift: {baseline['temperature_delta_mean']:.4f} C"
            )
            decision_lines.append(
                f"- Best temp drift: {best['temperature_delta_mean']:.4f} C"
            )
            decision_lines.append(
                f"- Temp drift improvement: {temp_improvement:+.4f} C"
            )
            decision_lines.append(
                f"- Baseline tip drift: {baseline['tip_delta_mean']:.4f} px"
            )
            decision_lines.append(
                f"- Best tip drift: {best['tip_delta_mean']:.4f} px"
            )
            decision_lines.append(
                f"- Tip drift improvement: {tip_improvement:+.4f} px"
            )

        decision_lines.extend(
            [
                "",
                "### Next Step",
                f"Freeze decoder to `{best['decode_method']}` and run one val "
                "confirmation replay with that exact pair.",
                "Do not run test until the confirmation replay reproduces the pass.",
                "",
            ]
        )
    else:
        failing_gates: dict[str, int] = {}
        for r in summary_rows:
            for k in GATES:
                gk = f"gate_{k}"
                if not r.get(gk, False):
                    failing_gates[k] = failing_gates.get(k, 0) + 1
        decision_lines = [
            "# INT8 Decoder Drift Audit Decision",
            "",
            f"- Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- Candidates evaluated: {len(args.candidates)}",
            f"- Total decoder+candidate pairs: {len(summary_rows)}",
            f"- Passing pairs: 0",
            "",
            "**Decision B**: Decoder-only recovery failed.",
            "",
            "No decoder+candidate pair passes all gates.",
            "",
            "### Dominant Failing Gates",
        ]
        for gate, count in sorted(
            failing_gates.items(), key=lambda x: -x[1]
        ):
            decision_lines.append(
                f"- **{gate}**: failed by {count}/{len(summary_rows)} pairs"
            )
        decision_lines.append("")

        # Top 3 near-miss pairs
        decision_lines.append("### Top 3 Near-Miss Pairs")
        decision_lines.append("")
        for r in summary_rows[:3]:
            gate_str = " | ".join(
                f"{k}={'PASS' if r.get(f'gate_{k}', False) else 'FAIL'}"
                for k in GATES
            )
            decision_lines.append(
                f"#### {r['candidate_tag']} / {r['decode_method']}"
            )
            decision_lines.append(f"- Gates: {gate_str}")
            decision_lines.append(
                f"- INT8 accepted MAE: {r['int8_accepted_mae_c']:.4f} C"
            )
            decision_lines.append(
                f"- INT8 acceptance rate: {r['int8_acceptance_rate']:.4f}"
            )
            decision_lines.append(
                f"- Temp drift mean: {r['temperature_delta_mean']:.4f} C"
            )
            decision_lines.append(
                f"- Tip drift mean: {r['tip_delta_mean']:.4f} px"
            )
            decision_lines.append("")

        decision_lines.append("### Next Step")
        decision_lines.append(
            "Decoder-only recovery failed. "
            "Next step should be a small architecture change: add an INT8-friendly "
            "coordinate/offset auxiliary head or replace pure heatmap decoding "
            "with a quantization-robust point head."
        )

    _write_markdown(decision_lines, DECISION_REPORT_PATH)
    print(
        f"[AUDIT] Wrote decision report: {DECISION_REPORT_PATH}",
        flush=True,
    )

    # Console summary
    print(
        f"\n[AUDIT] ===== Results =====", flush=True
    )
    if passing:
        print(
            f"[AUDIT] DECISION A: Champion: {passing[0]['candidate_tag']} / {passing[0]['decode_method']}",
            flush=True,
        )
    else:
        print(
            f"[AUDIT] DECISION B: No champion found", flush=True
        )
    for r in summary_rows:
        pass_str = "PASS" if r["all_gates_pass"] else "FAIL"
        print(
            f"  {pass_str}  {r['candidate_tag']:40s} {r['decode_method']:25s}  "
            f"MAE={r['int8_accepted_mae_c']:.2f}  "
            f"Acc={r['int8_acceptance_rate']:.3f}  "
            f"Drift={r['temperature_delta_mean']:.4f}C  "
            f"TipDrift={r['tip_delta_mean']:.2f}px",
            flush=True,
        )


if __name__ == "__main__":
    main()
