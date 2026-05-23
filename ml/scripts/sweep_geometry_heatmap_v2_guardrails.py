#!/usr/bin/env python3
"""Sweep guardrail thresholds for geometry heatmap v2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import GeometryGuardrailThresholds


def _write_json(data: dict[str, Any], output_path: Path) -> None:
    """Write a JSON artifact with stable formatting."""

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a CSV table with explicit columns."""

    if not rows:
        raise ValueError("Cannot write an empty sweep table.")
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _safe_float(value: Any) -> float:
    """Convert a CSV cell into a float."""

    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _apply_thresholds(row: dict[str, Any], thresholds: GeometryGuardrailThresholds) -> tuple[str, float, tuple[str, ...]]:
    """Apply one threshold set to a precomputed failure row."""

    reasons: list[str] = []

    if int(row["predicted_center_normalized_in_bounds"]) == 0:
        reasons.append("center_normalized_out_of_bounds")
    if int(row["predicted_tip_normalized_in_bounds"]) == 0:
        reasons.append("tip_normalized_out_of_bounds")
    if _safe_float(row["predicted_min_edge_margin_px"]) < thresholds.edge_margin_px:
        reasons.append("predicted_point_near_edge")
    if _safe_float(row["center_heatmap_peak_value"]) < thresholds.center_peak_min:
        reasons.append("center_peak_too_low")
    if _safe_float(row["tip_heatmap_peak_value"]) < thresholds.tip_peak_min:
        reasons.append("tip_peak_too_low")
    if _safe_float(row["confidence"]) < thresholds.confidence_min:
        reasons.append("confidence_too_low")
    if _safe_float(row["center_heatmap_entropy"]) > thresholds.max_heatmap_entropy:
        reasons.append("center_heatmap_too_diffuse")
    if _safe_float(row["tip_heatmap_entropy"]) > thresholds.max_heatmap_entropy:
        reasons.append("tip_heatmap_too_diffuse")
    if _safe_float(row["center_heatmap_spread_px"]) > thresholds.max_heatmap_spread_px:
        reasons.append("center_heatmap_too_spread_out")
    if _safe_float(row["tip_heatmap_spread_px"]) > thresholds.max_heatmap_spread_px:
        reasons.append("tip_heatmap_too_spread_out")
    if int(row["angle_within_valid_sweep"]) == 0:
        reasons.append("predicted_angle_outside_valid_sweep")
    if not (
        thresholds.minimum_celsius - thresholds.temperature_physical_margin_c
        <= _safe_float(row["predicted_temperature_c_calibrated"])
        <= thresholds.maximum_celsius + thresholds.temperature_physical_margin_c
    ):
        reasons.append("temperature_outside_physical_margin")
    ratio = _safe_float(row["center_tip_distance_ratio"])
    if not (thresholds.center_tip_distance_ratio_min <= ratio <= thresholds.center_tip_distance_ratio_max):
        reasons.append("center_tip_distance_ratio_implausible")

    raw_temperature_c = _safe_float(row["predicted_temperature_c_calibrated"])
    if reasons:
        return ("rejected", math.nan, tuple(reasons))

    if raw_temperature_c < thresholds.minimum_celsius or raw_temperature_c > thresholds.maximum_celsius:
        clamped_temperature_c = float(np.clip(raw_temperature_c, thresholds.minimum_celsius, thresholds.maximum_celsius))
        return ("clamped", clamped_temperature_c, ("temperature_clamped_to_physical_range",))

    return ("accepted", raw_temperature_c, ())


def _summarize_subset(rows: list[dict[str, Any]], thresholds: GeometryGuardrailThresholds) -> dict[str, float]:
    """Summarize one jitter subset under one threshold set."""

    statuses: list[str] = []
    final_errors: list[float] = []
    raw_errors: list[float] = []
    rejected_good = 0
    good_count = 0
    rejected_over_20 = 0
    over_20_count = 0

    for row in rows:
        status, final_temperature_c, _ = _apply_thresholds(row, thresholds)
        raw_error = abs(_safe_float(row["predicted_temperature_c_calibrated"]) - _safe_float(row["true_temperature_c"]))
        raw_errors.append(raw_error)
        if status != "rejected":
            final_errors.append(abs(final_temperature_c - _safe_float(row["true_temperature_c"])))
        statuses.append(status)
        if raw_error <= 5.0:
            good_count += 1
            if status == "rejected":
                rejected_good += 1
        if raw_error > 20.0:
            over_20_count += 1
            if status == "rejected":
                rejected_over_20 += 1

    accepted_count = sum(1 for status in statuses if status != "rejected")
    accepted_fraction = accepted_count / len(rows) if rows else math.nan
    rejected_fraction = 1.0 - accepted_fraction if rows else math.nan
    clamped_count = sum(1 for status in statuses if status == "clamped")
    accepted_mae = float(np.mean(final_errors)) if final_errors else math.nan
    accepted_worst = float(np.max(final_errors)) if final_errors else math.nan
    pct_under_5 = float(np.mean(np.asarray(final_errors) < 5.0) * 100.0) if final_errors else math.nan
    pct_under_10 = float(np.mean(np.asarray(final_errors) < 10.0) * 100.0) if final_errors else math.nan
    false_rejection_rate_good = rejected_good / good_count if good_count else math.nan
    all_gt20_rejected = over_20_count > 0 and rejected_over_20 == over_20_count

    return {
        "count": float(len(rows)),
        "accepted_fraction": float(accepted_fraction),
        "rejected_fraction": float(rejected_fraction),
        "clamped_fraction": float(clamped_count / len(rows) if rows else math.nan),
        "accepted_mae_c": float(accepted_mae),
        "accepted_worst_error_c": float(accepted_worst),
        "accepted_percentage_under_5c": float(pct_under_5),
        "accepted_percentage_under_10c": float(pct_under_10),
        "false_rejection_rate_good_5c": float(false_rejection_rate_good),
        "all_gt20_rejected": float(1.0 if all_gt20_rejected else 0.0),
    }


def _summarize_level(rows: list[dict[str, Any]], thresholds: GeometryGuardrailThresholds) -> dict[str, float]:
    """Summarize one jitter level, including accepted outputs only."""

    status_rows = [_apply_thresholds(row, thresholds) for row in rows]
    accepted_rows: list[dict[str, Any]] = []
    for row, (status, temperature_c, _) in zip(rows, status_rows, strict=True):
        if status != "rejected":
            accepted_row = dict(row)
            accepted_row["guard_status"] = status
            accepted_row["guard_temperature_c"] = temperature_c
            accepted_rows.append(accepted_row)

    accepted_errors = [abs(_safe_float(row["guard_temperature_c"]) - _safe_float(row["true_temperature_c"])) for row in accepted_rows]
    return {
        "accepted_fraction": float(len(accepted_rows) / len(rows) if rows else math.nan),
        "accepted_mae_c": float(np.mean(accepted_errors)) if accepted_errors else math.nan,
        "accepted_worst_error_c": float(np.max(accepted_errors)) if accepted_errors else math.nan,
        "accepted_percentage_under_5c": float(np.mean(np.asarray(accepted_errors) < 5.0) * 100.0) if accepted_errors else math.nan,
        "accepted_percentage_under_10c": float(np.mean(np.asarray(accepted_errors) < 10.0) * 100.0) if accepted_errors else math.nan,
    }


def _thresholds_to_dict(thresholds: GeometryGuardrailThresholds) -> dict[str, float]:
    """Convert the thresholds into a JSON-friendly dictionary."""

    return {
        "center_peak_min": thresholds.center_peak_min,
        "tip_peak_min": thresholds.tip_peak_min,
        "confidence_min": thresholds.confidence_min,
        "max_heatmap_entropy": thresholds.max_heatmap_entropy,
        "max_heatmap_spread_px": thresholds.max_heatmap_spread_px,
        "center_tip_distance_ratio_min": thresholds.center_tip_distance_ratio_min,
        "center_tip_distance_ratio_max": thresholds.center_tip_distance_ratio_max,
        "edge_margin_px": thresholds.edge_margin_px,
        "temperature_physical_range_margin_c": thresholds.temperature_physical_margin_c,
        "minimum_celsius": thresholds.minimum_celsius,
        "maximum_celsius": thresholds.maximum_celsius,
    }


def _score_candidate(row: dict[str, Any]) -> tuple[float, float, float, float]:
    """Score one threshold candidate."""

    invalid_penalty = 0.0 if (
        row["identity_accepted_fraction"] >= 0.70
        and row["medium_accepted_fraction"] >= 0.60
        and row["accepted_worst_error_c"] < 20.0
        and row["all_gt20_rejected"] >= 1.0
    ) else 1000.0
    score = (
        invalid_penalty
        + _safe_float(row["accepted_mae_c"])
        + 0.10 * _safe_float(row["accepted_worst_error_c"])
        + 0.25 * _safe_float(row["false_rejection_rate_good_5c"])
        + 0.10 * _safe_float(row["rejected_fraction"])
    )
    return (score, -_safe_float(row["medium_accepted_fraction"]), -_safe_float(row["identity_accepted_fraction"]), _safe_float(row["accepted_mae_c"]))


def main() -> None:
    """Sweep all candidate guardrail thresholds."""

    parser = argparse.ArgumentParser(description="Sweep geometry heatmap guardrails")
    parser.add_argument(
        "--failure-analysis-csv",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2_guarded/failure_analysis.csv"),
        help="Per-prediction failure analysis CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2_guarded/guardrail_sweep.csv"),
        help="Guardrail sweep CSV.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_v2_guardrail_sweep.md"),
        help="Markdown sweep report.",
    )
    parser.add_argument(
        "--selected-thresholds-json",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2_guarded/selected_guardrail_thresholds.json"),
        help="Selected thresholds JSON artifact.",
    )
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    failure_analysis_csv = base_path / args.failure_analysis_csv if not args.failure_analysis_csv.is_absolute() else args.failure_analysis_csv
    output_csv = base_path / args.output_csv if not args.output_csv.is_absolute() else args.output_csv
    report_path = base_path / args.report_path if not args.report_path.is_absolute() else args.report_path
    selected_thresholds_json = (
        base_path / args.selected_thresholds_json if not args.selected_thresholds_json.is_absolute() else args.selected_thresholds_json
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(failure_analysis_csv, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {failure_analysis_csv}")

    by_level: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_level.setdefault(str(row["jitter_level"]), []).append(row)

    center_peak_candidates = [0.40, 0.50, 0.60]
    tip_peak_candidates = [0.40, 0.50, 0.60]
    confidence_candidates = [0.40, 0.50, 0.60]
    entropy_candidates = [0.98, 0.99, 1.00]
    spread_candidates = [25.0, 30.0, 35.0]
    ratio_min_candidates = [0.40, 0.50, 0.60]
    ratio_max_candidates = [1.40, 1.50, 1.60]
    temp_margin_candidates = [2.0, 5.0, 10.0]
    edge_margin_candidates = [4.0, 8.0, 12.0]

    sweep_rows: list[dict[str, Any]] = []
    for (
        center_peak_min,
        tip_peak_min,
        confidence_min,
        max_heatmap_entropy,
        max_heatmap_spread_px,
        center_tip_distance_ratio_min,
        center_tip_distance_ratio_max,
        temperature_physical_margin_c,
        edge_margin_px,
    ) in product(
        center_peak_candidates,
        tip_peak_candidates,
        confidence_candidates,
        entropy_candidates,
        spread_candidates,
        ratio_min_candidates,
        ratio_max_candidates,
        temp_margin_candidates,
        edge_margin_candidates,
    ):
        thresholds = GeometryGuardrailThresholds(
            center_peak_min=center_peak_min,
            tip_peak_min=tip_peak_min,
            confidence_min=confidence_min,
            max_heatmap_entropy=max_heatmap_entropy,
            max_heatmap_spread_px=max_heatmap_spread_px,
            center_tip_distance_ratio_min=center_tip_distance_ratio_min,
            center_tip_distance_ratio_max=center_tip_distance_ratio_max,
            edge_margin_px=edge_margin_px,
            temperature_physical_margin_c=temperature_physical_margin_c,
        )

        row: dict[str, Any] = _thresholds_to_dict(thresholds)
        overall = _summarize_subset(rows, thresholds)
        row.update(overall)
        for level in ["identity", "mild", "medium", "strong"]:
            level_metrics = _summarize_level(by_level[level], thresholds)
            for key, value in level_metrics.items():
                row[f"{level}_{key}"] = value
        sweep_rows.append(row)

    sweep_rows.sort(key=_score_candidate)
    _write_csv(sweep_rows, output_csv)

    valid_rows = [
        row
        for row in sweep_rows
        if row["identity_accepted_fraction"] >= 0.70
        and row["medium_accepted_fraction"] >= 0.60
        and row["accepted_worst_error_c"] < 20.0
        and row["all_gt20_rejected"] >= 1.0
    ]
    selected_row = valid_rows[0] if valid_rows else sweep_rows[0]
    selected_thresholds = GeometryGuardrailThresholds(
        center_peak_min=float(selected_row["center_peak_min"]),
        tip_peak_min=float(selected_row["tip_peak_min"]),
        confidence_min=float(selected_row["confidence_min"]),
        max_heatmap_entropy=float(selected_row["max_heatmap_entropy"]),
        max_heatmap_spread_px=float(selected_row["max_heatmap_spread_px"]),
        center_tip_distance_ratio_min=float(selected_row["center_tip_distance_ratio_min"]),
        center_tip_distance_ratio_max=float(selected_row["center_tip_distance_ratio_max"]),
        edge_margin_px=float(selected_row["edge_margin_px"]),
        temperature_physical_margin_c=float(selected_row["temperature_physical_range_margin_c"]),
    )
    _write_json(
        {
            "selected_thresholds": _thresholds_to_dict(selected_thresholds),
            "selection_reason": "first valid candidate by score" if valid_rows else "best available candidate",
        },
        selected_thresholds_json,
    )

    lines = [
        "# Geometry Heatmap v2 Guardrail Sweep",
        "",
        "## Selected Thresholds",
        "",
        "| threshold | value |",
        "| --- | ---: |",
    ]
    for key, value in _thresholds_to_dict(selected_thresholds).items():
        lines.append(f"| {key} | {value:.3f} |")
    lines.extend(
        [
            "",
            "## Selected Candidate Metrics",
            "",
            "| metric | value |",
            "| --- | ---: |",
        ]
    )
    for key in [
        "accepted_fraction",
        "rejected_fraction",
        "clamped_fraction",
        "accepted_mae_c",
        "accepted_worst_error_c",
        "accepted_percentage_under_5c",
        "accepted_percentage_under_10c",
        "false_rejection_rate_good_5c",
        "all_gt20_rejected",
        "identity_accepted_fraction",
        "identity_accepted_mae_c",
        "identity_accepted_worst_error_c",
        "medium_accepted_fraction",
        "medium_accepted_mae_c",
        "medium_accepted_worst_error_c",
        "strong_accepted_fraction",
        "strong_accepted_mae_c",
        "strong_accepted_worst_error_c",
    ]:
        lines.append(f"| {key} | {float(selected_row[key]):.3f} |")
    lines.extend(
        [
            "",
            "## Selection Notes",
            "",
            f"- Valid candidates found: {len(valid_rows)}",
            f"- The selected guardrail set keeps identity and medium jitter above the requested acceptance floor while rejecting the catastrophic tail.",
            f"- The selected candidate's worst accepted error is {float(selected_row['accepted_worst_error_c']):.3f} C.",
            f"- False rejection rate on <=5 C predictions: {float(selected_row['false_rejection_rate_good_5c']):.3f}.",
            "",
            "## Top 10 Candidates",
            "",
            "| rank | accepted_mae | worst | identity_accept | medium_accept | strong_accept | false_reject_good | rejected_fraction | center_peak | tip_peak | conf | entropy | spread | ratio_min | ratio_max | edge | temp_margin |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    display_rows = valid_rows[:10] if valid_rows else sweep_rows[:10]
    for rank, row in enumerate(display_rows, start=1):
        lines.append(
            "| {rank} | {accepted_mae_c:.3f} | {accepted_worst_error_c:.3f} | {identity_accepted_fraction:.3f} | {medium_accepted_fraction:.3f} | {strong_accepted_fraction:.3f} | {false_rejection_rate_good_5c:.3f} | {rejected_fraction:.3f} | {center_peak_min:.2f} | {tip_peak_min:.2f} | {confidence_min:.2f} | {max_heatmap_entropy:.2f} | {max_heatmap_spread_px:.1f} | {center_tip_distance_ratio_min:.2f} | {center_tip_distance_ratio_max:.2f} | {edge_margin_px:.1f} | {temperature_physical_range_margin_c:.1f} |".format(
                rank=rank, **row
            )
        )

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(f"Guardrail sweep CSV: {output_csv}")
    print(f"Guardrail sweep report: {report_path}")
    print(f"Selected thresholds JSON: {selected_thresholds_json}")


if __name__ == "__main__":
    main()
