#!/usr/bin/env python3
"""Evaluate all 6 representative-dataset strategies on the validation split.

Compares each INT8 strategy against Keras (reference) and against the
Phase 10E baseline.  Produces a single comparison CSV + markdown report.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_heatmap_quantization_common import (
    decode_and_guard,
    load_semantic_output_order_indices,
    load_split_samples,
    predict_tflite_outputs,
    resolve_repo_path,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import (
    load_geometry_heatmap_keras_model,
    load_tflite_model,
    summarize_tflite_contract,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import load_selected_calibration_candidate
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import GeometryGuardrailThresholds

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = Path("ml/artifacts/training/geometry_heatmap_v4_112_quant_native_30epoch_smoke/model_v4_112.keras")
SWEEP_DIR = Path("ml/artifacts/deployment/geometry_heatmap_v4_112_tflite_rep_sweep")
PHASE10E_DIR = Path("ml/artifacts/deployment/geometry_heatmap_v4_112_tflite")
MANIFEST_PATH = Path("ml/data/geometry_reader_manifest_v2_clean.csv")
CALIBRATION_PATH = Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json")
THRESHOLDS_PATH = Path("ml/artifacts/training/geometry_heatmap_v4_112_quant_native/v4_112_guardrail_thresholds.json")
DECODER_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json")
OUTPUT_DIR = Path("ml/artifacts/deployment/geometry_heatmap_v4_112_tflite_rep_sweep")
REPORT_PATH = Path("ml/reports/geometry_heatmap_v4_112_int8_rep_sweep.md")
DECISION_PATH = Path("ml/reports/geometry_heatmap_v4_112_int8_rep_sweep_decision.md")
PREDICTIONS_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v4_112_tflite_rep_sweep/sweep_predictions.csv")
SUMMARY_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v4_112_tflite_rep_sweep/sweep_summary.csv")

STRATEGIES: dict[str, str] = {
    "baseline_identity": "A",
    "identity_mild": "B",
    "identity_mild_medium": "C",
    "stratified": "D",
    "spread_boundary": "E",
    "combined": "F",
}

QAT_DIR = Path("ml/artifacts/deployment/geometry_heatmap_v4_112_qat")

VALIDATION_GATE = {
    "accepted_mae_c": 4.5,
    "acceptance_rate": 0.65,
    "worst_accepted_error_c": 20.0,
    "accepted_gt20_failures": 0,
    "temperature_delta_mean": 1.0,
    "tip_delta_mean": 14.82,
}


def _status_is_accepted(status: str) -> bool:
    return status in {"accepted", "clamped"}


def _load_thresholds(thresholds_path: Path) -> GeometryGuardrailThresholds:
    with thresholds_path.open("r") as handle:
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


def _load_decode_spec(selection_path: Path) -> tuple[str, int]:
    with selection_path.open("r") as handle:
        payload = json.load(handle)
    decode_method = str(payload.get("decode_method", payload.get("selected_decode_method", "")))
    window_size = int(payload.get("window_size", payload.get("selected_window_size", 0)))
    if "_w" in decode_method:
        decode_method, suffix = decode_method.rsplit("_w", 1)
        if suffix.isdigit():
            window_size = int(suffix)
    if decode_method != "softargmax" or window_size != 3:
        raise RuntimeError(f"Expected softargmax w3, got {decode_method} w{window_size}")
    return decode_method, window_size


def _predict_keras_outputs(model: keras.Model, inputs: np.ndarray, *, batch_size: int) -> dict[str, np.ndarray]:
    outputs = model.predict(inputs, batch_size=batch_size, verbose=0)
    if isinstance(outputs, dict):
        return {name: np.asarray(tensor, dtype=np.float32) for name, tensor in outputs.items()}
    center, tip, conf = outputs
    return {
        "center_heatmap": np.asarray(center, dtype=np.float32),
        "tip_heatmap": np.asarray(tip, dtype=np.float32),
        "confidence": np.asarray(conf, dtype=np.float32),
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
        reasons = ";".join(guarded.rejection_reasons) if guarded.rejection_reasons else "none"
        rows.append({
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
        })
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
            sum(1 for row in accepted if abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"])) > 20.0)
        ),
        "tip_mae_px_224": float(
            np.mean([
                math.hypot(
                    float(row["predicted_tip_x_224"]) - float(row["true_tip_x_224"]),
                    float(row["predicted_tip_y_224"]) - float(row["true_tip_y_224"]),
                )
                for row in rows
            ])
        ),
        "angle_mae_degrees": float(
            np.mean([abs(float(row["predicted_angle_degrees"]) - float(row["true_angle_degrees"])) for row in rows])
        ),
        "tip_heatmap_spread_mean": float(np.mean([float(row["tip_heatmap_spread_px"]) for row in rows])),
        "center_heatmap_spread_mean": float(np.mean([float(row["center_heatmap_spread_px"]) for row in rows])),
        "guardrail_rejection_count": float(sum(1 for row in rows if not _status_is_accepted(str(row["guardrail_status"])))),
        "top_rejection_reasons": ";".join(
            f"{reason}:{count}" for reason, count in top_rejection_reasons.most_common(5)
        ) if top_rejection_reasons else "none",
    }


def _compare_against_reference(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, float]:
    ref_by_image = {row["image_path"]: row for row in reference_rows}
    cand_by_image = {row["image_path"]: row for row in candidate_rows}
    common = sorted(ref_by_image.keys() & cand_by_image.keys())
    temp_deltas: list[float] = []
    center_deltas: list[float] = []
    tip_deltas: list[float] = []
    guardrail_disagreements = 0
    for image_path in common:
        ref = ref_by_image[image_path]
        cand = cand_by_image[image_path]
        if _status_is_accepted(str(ref["guardrail_status"])) and _status_is_accepted(str(cand["guardrail_status"])):
            temp_deltas.append(abs(float(ref["guarded_temperature_c"]) - float(cand["guarded_temperature_c"])))
        center_deltas.append(
            math.hypot(
                float(ref["predicted_center_x_224"]) - float(cand["predicted_center_x_224"]),
                float(ref["predicted_center_y_224"]) - float(cand["predicted_center_y_224"]),
            )
        )
        tip_deltas.append(
            math.hypot(
                float(ref["predicted_tip_x_224"]) - float(cand["predicted_tip_x_224"]),
                float(ref["predicted_tip_y_224"]) - float(cand["predicted_tip_y_224"]),
            )
        )
        if str(ref["guardrail_status"]) != str(cand["guardrail_status"]):
            guardrail_disagreements += 1
    return {
        "temperature_delta_mean": float(np.mean(temp_deltas)) if temp_deltas else math.nan,
        "temperature_delta_median": float(np.median(temp_deltas)) if temp_deltas else math.nan,
        "center_delta_mean": float(np.mean(center_deltas)) if center_deltas else math.nan,
        "tip_delta_mean": float(np.mean(tip_deltas)) if tip_deltas else math.nan,
        "guardrail_disagreements": float(guardrail_disagreements),
    }


def _gate_result(summary: dict[str, Any], drift: dict[str, float]) -> dict[str, bool]:
    return {
        "mae_pass": summary["accepted_mae_c"] <= VALIDATION_GATE["accepted_mae_c"],
        "acceptance_pass": summary["acceptance_rate"] >= VALIDATION_GATE["acceptance_rate"],
        "worst_pass": summary["worst_accepted_error_c"] < VALIDATION_GATE["worst_accepted_error_c"],
        "gt20_pass": summary["accepted_gt20_failures"] <= VALIDATION_GATE["accepted_gt20_failures"],
        "temp_drift_pass": drift.get("temperature_delta_mean", math.nan) <= VALIDATION_GATE["temperature_delta_mean"],
        "tip_drift_pass": drift.get("tip_delta_mean", math.nan) < VALIDATION_GATE["tip_delta_mean"],
    }


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV file.")
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
        import csv
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    base = resolve_repo_path(REPO_ROOT, Path("."))
    manifest_path = resolve_repo_path(REPO_ROOT, MANIFEST_PATH)
    model_path = resolve_repo_path(REPO_ROOT, MODEL_PATH)
    sweep_dir = resolve_repo_path(REPO_ROOT, SWEEP_DIR)
    phase10e_dir = resolve_repo_path(REPO_ROOT, PHASE10E_DIR)
    decoder_path = resolve_repo_path(REPO_ROOT, DECODER_PATH)
    calibration_path = resolve_repo_path(REPO_ROOT, CALIBRATION_PATH)
    thresholds_path = resolve_repo_path(REPO_ROOT, THRESHOLDS_PATH)
    output_dir = resolve_repo_path(REPO_ROOT, OUTPUT_DIR)
    report_path = resolve_repo_path(REPO_ROOT, REPORT_PATH)
    decision_path = resolve_repo_path(REPO_ROOT, DECISION_PATH)
    predictions_path = resolve_repo_path(REPO_ROOT, PREDICTIONS_PATH)
    summary_path = resolve_repo_path(REPO_ROOT, SUMMARY_PATH)

    # Load shared artifacts
    decode_method, window_size = _load_decode_spec(decoder_path)
    calibration_candidate, _ = load_selected_calibration_candidate(calibration_path)
    thresholds = _load_thresholds(thresholds_path)

    # Load samples once
    print("[SWEEP-EVAL] Loading validation split...")
    samples = load_split_samples(
        manifest_path, base, split="val",
        mode="python_training_rgb_bilinear",
        input_size=224, heatmap_size=112, sigma_pixels=2.5,
    )
    inputs = [np.asarray(s.crop_image, dtype=np.float32) for s in samples.samples]
    keras_inputs = np.stack([inp * 2.0 - 1.0 for inp in inputs], axis=0)
    print(f"  {len(samples.samples)} validation samples")

    # Keras reference
    print("[SWEEP-EVAL] Running Keras reference...")
    model = load_geometry_heatmap_keras_model(model_path)
    keras_outputs = _predict_keras_outputs(model, keras_inputs, batch_size=16)
    keras_rows = _evaluate_model(
        model_type="keras_v3", samples=samples.samples, outputs=keras_outputs,
        calibration_candidate=calibration_candidate, thresholds=thresholds,
        decode_method=decode_method, window_size=window_size,
    )
    keras_summary = _summarize_rows(keras_rows)
    print(f"  Keras MAE: {keras_summary['accepted_mae_c']:.4f}, accept: {keras_summary['acceptance_rate']:.3f}")

    # Phase 10E baseline INT8
    phase10e_int8_path = phase10e_dir / "model_v4_112_int8.tflite"
    baseline_label = "baseline_phase10e"
    all_predictions: list[dict[str, Any]] = []

    print(f"[SWEEP-EVAL] Evaluating {baseline_label}...")
    if phase10e_int8_path.exists():
        contract = load_semantic_output_order_indices(phase10e_dir / "tflite_tensor_contract.json")
        bundle = load_tflite_model(phase10e_int8_path)
        outputs = predict_tflite_outputs(bundle, inputs, semantic_output_order_indices=contract)
        semantic_outputs = {
            "center_heatmap": np.asarray(outputs[0], dtype=np.float32),
            "tip_heatmap": np.asarray(outputs[1], dtype=np.float32),
            "confidence": np.asarray(outputs[2], dtype=np.float32),
        }
        rows = _evaluate_model(
            model_type=baseline_label, samples=samples.samples, outputs=semantic_outputs,
            calibration_candidate=calibration_candidate, thresholds=thresholds,
            decode_method=decode_method, window_size=window_size,
        )
        all_predictions.extend(rows)
        baseline_summary = _summarize_rows(rows)
        baseline_drift = _compare_against_reference(keras_rows, rows)
        print(f"  MAE: {baseline_summary['accepted_mae_c']:.4f}, accept: {baseline_summary['acceptance_rate']:.3f}, "
              f"drift: {baseline_drift['temperature_delta_mean']:.4f}")
    else:
        print(f"  {phase10e_int8_path} not found, skipping")
        baseline_summary = {}
        baseline_drift = {}

    # Sweep strategies
    strategy_results: dict[str, dict[str, Any]] = {}
    strategy_labels = sorted(STRATEGIES.keys(), key=lambda k: STRATEGIES[k])

    for strat_name in strategy_labels:
        label = f"int8_{strat_name}"
        int8_path = sweep_dir / strat_name / "model_v4_112_int8.tflite"
        contract_path_s = sweep_dir / strat_name / "tflite_tensor_contract.json"

        if not int8_path.exists():
            print(f"  SKIP (no model): {strat_name}")
            continue

        print(f"[SWEEP-EVAL] Evaluating {strat_name} (strategy {STRATEGIES[strat_name]})...")
        contract = load_semantic_output_order_indices(contract_path_s)
        bundle = load_tflite_model(int8_path)
        outputs = predict_tflite_outputs(bundle, inputs, semantic_output_order_indices=contract)
        semantic_outputs = {
            "center_heatmap": np.asarray(outputs[0], dtype=np.float32),
            "tip_heatmap": np.asarray(outputs[1], dtype=np.float32),
            "confidence": np.asarray(outputs[2], dtype=np.float32),
        }
        rows = _evaluate_model(
            model_type=label, samples=samples.samples, outputs=semantic_outputs,
            calibration_candidate=calibration_candidate, thresholds=thresholds,
            decode_method=decode_method, window_size=window_size,
        )
        all_predictions.extend(rows)
        summary = _summarize_rows(rows)
        drift = _compare_against_reference(keras_rows, rows)
        gates = _gate_result(summary, drift)
        strategy_results[strat_name] = {
            "summary": summary,
            "drift": drift,
            "gates": gates,
            "all_passed": all(gates.values()),
        }
        print(f"  MAE: {summary['accepted_mae_c']:.4f}, accept: {summary['acceptance_rate']:.3f}, "
              f"drift: {drift['temperature_delta_mean']:.4f}, gate: {'PASS' if strategy_results[strat_name]['all_passed'] else 'FAIL'}")

    # Write predictions CSV
    _write_csv(all_predictions, predictions_path)
    print(f"[SWEEP-EVAL] Predictions -> {predictions_path}")

    # Build summary table
    summary_rows: list[dict[str, Any]] = []
    for strat_name in strategy_labels:
        if strat_name not in strategy_results:
            continue
        sr = strategy_results[strat_name]
        s = sr["summary"]
        d = sr["drift"]
        g = sr["gates"]
        summary_rows.append({
            "strategy": f"{STRATEGIES[strat_name]}",
            "strategy_name": strat_name,
            "rep_count": s["count"],
            "accepted_count": int(s["accepted_count"]),
            "accepted_mae_c": round(s["accepted_mae_c"], 4),
            "acceptance_rate": round(s["acceptance_rate"], 4),
            "worst_accepted_error_c": round(s["worst_accepted_error_c"], 4),
            "accepted_gt20_failures": int(s["accepted_gt20_failures"]),
            "temp_drift_mean": round(d.get("temperature_delta_mean", math.nan), 4),
            "tip_drift_mean": round(d.get("tip_delta_mean", math.nan), 4),
            "guardrail_disagreements": int(d.get("guardrail_disagreements", 0)),
            "tip_heatmap_spread_mean": round(s["tip_heatmap_spread_mean"], 4),
            "gate_all_pass": "PASS" if sr["all_passed"] else "FAIL",
            "mae_pass": g["mae_pass"],
            "acceptance_pass": g["acceptance_pass"],
            "worst_pass": g["worst_pass"],
            "gt20_pass": g["gt20_pass"],
            "temp_drift_pass": g["temp_drift_pass"],
            "tip_drift_pass": g["tip_drift_pass"],
        })

    # Add baseline if present
    if baseline_summary:
        baseline_gates = _gate_result(baseline_summary, baseline_drift)
        summary_rows.insert(0, {
            "strategy": "Ref",
            "strategy_name": "Phase 10E",
            "rep_count": len(samples.samples),
            "accepted_count": int(baseline_summary.get("accepted_count", 0)),
            "accepted_mae_c": round(baseline_summary.get("accepted_mae_c", math.nan), 4),
            "acceptance_rate": round(baseline_summary.get("acceptance_rate", math.nan), 4),
            "worst_accepted_error_c": round(baseline_summary.get("worst_accepted_error_c", math.nan), 4),
            "accepted_gt20_failures": int(baseline_summary.get("accepted_gt20_failures", 0)),
            "temp_drift_mean": round(baseline_drift.get("temperature_delta_mean", math.nan), 4),
            "tip_drift_mean": round(baseline_drift.get("tip_delta_mean", math.nan), 4),
            "guardrail_disagreements": int(baseline_drift.get("guardrail_disagreements", 0)),
            "tip_heatmap_spread_mean": round(baseline_summary.get("tip_heatmap_spread_mean", math.nan), 4),
            "gate_all_pass": "FAIL",
            "mae_pass": baseline_gates["mae_pass"],
            "acceptance_pass": baseline_gates["acceptance_pass"],
            "worst_pass": baseline_gates["worst_pass"],
            "gt20_pass": baseline_gates["gt20_pass"],
            "temp_drift_pass": baseline_gates["temp_drift_pass"],
            "tip_drift_pass": baseline_gates["tip_drift_pass"],
        })

    _write_csv(summary_rows, summary_path)
    print(f"[SWEEP-EVAL] Summary -> {summary_path}")

    # Markdown report
    lines = [
        "# Geometry Heatmap v4 112 INT8 Representative-Dataset Sweep",
        "",
        f"- Validation samples: {len(samples.samples)}",
        f"- Decoder: {decode_method} w{window_size}",
        f"- Calibration: {calibration_candidate.name}",
        f"- Keras MAE: {keras_summary['accepted_mae_c']:.4f} C, accept rate: {keras_summary['acceptance_rate']:.3f}",
        "",
        "## Gate Criteria",
        f"- MAE ≤ {VALIDATION_GATE['accepted_mae_c']} C",
        f"- Acceptance ≥ {VALIDATION_GATE['acceptance_rate']:.0%}",
        f"- Worst error < {VALIDATION_GATE['worst_accepted_error_c']} C",
        f"- >20 C failures = {VALIDATION_GATE['accepted_gt20_failures']}",
        f"- Temp drift ≤ {VALIDATION_GATE['temperature_delta_mean']} C",
        f"- Tip drift < {VALIDATION_GATE['tip_delta_mean']} px",
        "",
        "## Results",
        "",
        "| Strat | Name | MAE | Accept | Worst | >20C | Drift | Tip Drift | Spread | Gate |",
        "|-------|------|-----|--------|-------|------|-------|-----------|--------|------|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['strategy']} | {row['strategy_name']} "
            f"| {row['accepted_mae_c']:.2f} "
            f"| {row['acceptance_rate']:.1%} "
            f"| {row['worst_accepted_error_c']:.2f} "
            f"| {row['accepted_gt20_failures']} "
            f"| {row['temp_drift_mean']:.2f} "
            f"| {row['tip_drift_mean']:.2f} "
            f"| {row['tip_heatmap_spread_mean']:.1f} "
            f"| {row['gate_all_pass']} |"
        )

    lines.append("")
    passing = [r for r in summary_rows if r["gate_all_pass"] == "PASS"]
    if passing:
        passing_strats = ", ".join(f"{r['strategy']} ({r['strategy_name']})" for r in passing)
        lines.append(f"**Passing strategies: {passing_strats}**")
    else:
        lines.append("**No strategy passed all gates.**")

    lines.extend([
        "",
        "## Per-Gate Breakdown",
        "",
        "| Strat | MAE Pass | Accept Pass | Worst Pass | >20C Pass | Drift Pass | Tip Drift Pass |",
        "|-------|----------|-------------|------------|-----------|------------|----------------|",
    ])
    for row in summary_rows:
        lines.append(
            f"| {row['strategy']} {row['strategy_name']} "
            f"| {'✅' if row['mae_pass'] else '❌'} "
            f"| {'✅' if row['acceptance_pass'] else '❌'} "
            f"| {'✅' if row['worst_pass'] else '❌'} "
            f"| {'✅' if row['gt20_pass'] else '❌'} "
            f"| {'✅' if row['temp_drift_pass'] else '❌'} "
            f"| {'✅' if row['tip_drift_pass'] else '❌'} |"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SWEEP-EVAL] Report -> {report_path}")

    # Decision document
    if passing:
        best = min(passing, key=lambda r: r["temp_drift_mean"])
        decision_lines = [
            "# Sweep Decision: Proceed to Test",
            "",
            f"Best strategy: {best['strategy']} ({best['strategy_name']}) — drift {best['temp_drift_mean']:.4f} C, MAE {best['accepted_mae_c']:.4f} C",
            "",
            "## Reasoning",
            f"- {len(passing)} strategy/strategies pass all gates.",
            f"- Selected `{best['strategy_name']}` (strategy {best['strategy']}) for lowest temp drift.",
            "- Next: run test-split replay with selected strategy, then check Cube.AI readiness.",
        ]
    else:
        failing_gates = [k for k, v in summary_rows[-1].items() if k.endswith("_pass") and not v]
        decision_lines = [
            "# Sweep Decision: No Strategy Passes All Gates",
            "",
            "## Failing Gates",
        ]
        for gate_key in ["mae_pass", "acceptance_pass", "worst_pass", "gt20_pass", "temp_drift_pass", "tip_drift_pass"]:
            for row in summary_rows:
                if not row.get(gate_key, True):
                    gate_name = gate_key.replace("_pass", "").replace("_", " ").title()
                    decision_lines.append(f"- {gate_name}: {row['strategy']} {row['strategy_name']} "
                                          f"({row.get(gate_key.replace('_pass', '').replace('_mae', '_mae_c').replace('_worst', '_worst_accepted_error_c'), '?')})")
        decision_lines.extend([
            "",
            "## Recommendations",
            "- Consider widening the representative dataset further (more jitter, more diversity).",
            "- Consider QAT (quantization-aware training) to bake quantization robustness into weights.",
            "- Or deploy the FP32 model instead (7.6 MB, exact Keras match).",
        ])

    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text("\n".join(decision_lines), encoding="utf-8")
    print(f"[SWEEP-EVAL] Decision -> {decision_path}")


if __name__ == "__main__":
    main()
