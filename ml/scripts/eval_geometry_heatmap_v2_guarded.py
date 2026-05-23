#!/usr/bin/env python3
"""Evaluate geometry heatmap v2 after applying guardrails."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import GeometryGuardrailThresholds


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a CSV table to disk."""

    if not rows:
        raise ValueError("Cannot write an empty CSV table.")
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _write_json(data: dict[str, Any], output_path: Path) -> None:
    """Write a JSON artifact with stable formatting."""

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _safe_float(value: Any) -> float:
    """Convert a CSV cell into a float."""

    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _load_thresholds(path: Path) -> GeometryGuardrailThresholds:
    """Load the selected threshold set from JSON."""

    if not path.exists():
        return GeometryGuardrailThresholds()
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = payload.get("selected_thresholds", payload)
    return GeometryGuardrailThresholds(
        center_peak_min=float(selected["center_peak_min"]),
        tip_peak_min=float(selected["tip_peak_min"]),
        confidence_min=float(selected["confidence_min"]),
        max_heatmap_entropy=float(selected["max_heatmap_entropy"]),
        max_heatmap_spread_px=float(selected["max_heatmap_spread_px"]),
        center_tip_distance_ratio_min=float(selected["center_tip_distance_ratio_min"]),
        center_tip_distance_ratio_max=float(selected["center_tip_distance_ratio_max"]),
        edge_margin_px=float(selected["edge_margin_px"]),
        temperature_physical_margin_c=float(
            selected.get("temperature_physical_range_margin_c", selected.get("temperature_physical_margin_c", 5.0))
        ),
    )


def _apply_thresholds(row: dict[str, Any], thresholds: GeometryGuardrailThresholds) -> tuple[str, float, tuple[str, ...]]:
    """Apply one threshold set to a failure-analysis row."""

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


def _summarize_level(rows: list[dict[str, Any]], thresholds: GeometryGuardrailThresholds) -> dict[str, float]:
    """Summarize one jitter level under the chosen policy."""

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    clamped_rows: list[dict[str, Any]] = []

    for row in rows:
        status, temperature_c, reasons = _apply_thresholds(row, thresholds)
        row = dict(row)
        row["guard_status"] = status
        row["guard_temperature_c"] = temperature_c
        row["guard_rejection_reasons"] = "|".join(reasons)
        if status == "rejected":
            rejected_rows.append(row)
        else:
            if status == "clamped":
                clamped_rows.append(row)
            accepted_rows.append(row)

    accepted_errors = [abs(_safe_float(row["guard_temperature_c"]) - _safe_float(row["true_temperature_c"])) for row in accepted_rows]
    rejected_errors = [abs(_safe_float(row["predicted_temperature_c_calibrated"]) - _safe_float(row["true_temperature_c"])) for row in rejected_rows]

    return {
        "total_predictions": float(len(rows)),
        "accepted_predictions": float(len(accepted_rows)),
        "rejected_predictions": float(len(rejected_rows)),
        "clamped_predictions": float(len(clamped_rows)),
        "acceptance_rate": float(len(accepted_rows) / len(rows) if rows else math.nan),
        "accepted_calibrated_mae_c": float(np.mean(accepted_errors)) if accepted_errors else math.nan,
        "accepted_worst_error_c": float(np.max(accepted_errors)) if accepted_errors else math.nan,
        "accepted_percentage_under_2c": float(np.mean(np.asarray(accepted_errors) < 2.0) * 100.0) if accepted_errors else math.nan,
        "accepted_percentage_under_5c": float(np.mean(np.asarray(accepted_errors) < 5.0) * 100.0) if accepted_errors else math.nan,
        "accepted_percentage_under_10c": float(np.mean(np.asarray(accepted_errors) < 10.0) * 100.0) if accepted_errors else math.nan,
        "rejected_case_mean_raw_error_c": float(np.mean(rejected_errors)) if rejected_errors else math.nan,
        "rejected_case_worst_raw_error_c": float(np.max(rejected_errors)) if rejected_errors else math.nan,
    }


def _write_report(
    *,
    report_path: Path,
    thresholds: GeometryGuardrailThresholds,
    metrics_by_level: dict[str, dict[str, float]],
    baseline_temperature_mae_c: float,
    baseline_center_mae_px: float,
    baseline_tip_mae_px: float,
    oracle_temperature_mae_c: float,
    worst_accepted_rows: list[dict[str, Any]],
    rejected_reason_counts: dict[str, int],
    catastrophic_total: int,
    catastrophic_rejected: int,
) -> None:
    """Write the guarded evaluation report."""

    identity = metrics_by_level["identity"]
    medium = metrics_by_level["medium"]
    strong = metrics_by_level["strong"]
    # The same policy is applied to every jitter level; this mirrors the test-set use case.
    # We still call out the identity, medium, and strong rows separately in the report.
    proceed = (
        identity["accepted_calibrated_mae_c"] <= 4.5
        and medium["accepted_calibrated_mae_c"] <= 5.5
        and max(identity["accepted_worst_error_c"], medium["accepted_worst_error_c"], strong["accepted_worst_error_c"]) < 20.0
        and medium["acceptance_rate"] >= 0.60
        and catastrophic_total > 0
        and catastrophic_rejected / catastrophic_total >= 0.50
    )

    lines = [
        "# Geometry Heatmap v2 Guarded Evaluation",
        "",
        "## Selected Guardrail Thresholds",
        "",
        "| threshold | value |",
        "| --- | ---: |",
        f"| center_peak_min | {thresholds.center_peak_min:.3f} |",
        f"| tip_peak_min | {thresholds.tip_peak_min:.3f} |",
        f"| confidence_min | {thresholds.confidence_min:.3f} |",
        f"| max_heatmap_entropy | {thresholds.max_heatmap_entropy:.3f} |",
        f"| max_heatmap_spread_px | {thresholds.max_heatmap_spread_px:.3f} |",
        f"| center_tip_distance_ratio_min | {thresholds.center_tip_distance_ratio_min:.3f} |",
        f"| center_tip_distance_ratio_max | {thresholds.center_tip_distance_ratio_max:.3f} |",
        f"| edge_margin_px | {thresholds.edge_margin_px:.3f} |",
        f"| temperature_physical_margin_c | {thresholds.temperature_physical_margin_c:.3f} |",
        "",
        "## Per-Level Metrics",
        "",
        "| level | total | accepted | rejected | clamped | acceptance_rate | accepted_mae | worst_accepted | under_2c | under_5c | under_10c | rejected_mean_raw | rejected_worst_raw |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for level in ["identity", "mild", "medium", "strong"]:
        metrics = metrics_by_level[level]
        lines.append(
            "| {level} | {total_predictions:.0f} | {accepted_predictions:.0f} | {rejected_predictions:.0f} | {clamped_predictions:.0f} | {acceptance_rate:.3f} | {accepted_calibrated_mae_c:.3f} | {accepted_worst_error_c:.3f} | {accepted_percentage_under_2c:.1f} | {accepted_percentage_under_5c:.1f} | {accepted_percentage_under_10c:.1f} | {rejected_case_mean_raw_error_c:.3f} | {rejected_case_worst_raw_error_c:.3f} |".format(
                level=level, **metrics
            )
        )

    lines.extend(
        [
            "",
            "## Baseline Comparison",
            "",
            f"- geometry_points_v1 test temperature MAE: {baseline_temperature_mae_c:.2f} C",
            f"- geometry_points_v1 test center MAE: {baseline_center_mae_px:.2f} px",
            f"- geometry_points_v1 test tip MAE: {baseline_tip_mae_px:.2f} px",
            f"- Oracle calibrated geometry ceiling: {oracle_temperature_mae_c:.3f} C",
            "",
            "## Rejected Reasons",
            "",
            f"- Rejected reason counts: {rejected_reason_counts}",
            "",
            "## Worst Remaining Accepted Predictions",
            "",
            "| rank | level | image | abs_err | temp_true | temp_guard | status | center_err | tip_err | ratio | confidence | reasons |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for rank, row in enumerate(worst_accepted_rows[:30], start=1):
        lines.append(
            f"| {rank} | {row['jitter_level']} | {Path(str(row['image_path'])).name} | {abs(_safe_float(row['guard_temperature_c']) - _safe_float(row['true_temperature_c'])):.3f} | {float(row['true_temperature_c']):.2f} | {float(row['guard_temperature_c']):.2f} | {row['guard_status']} | {float(row['center_px_mae_224']):.2f} | {float(row['tip_px_mae_224']):.2f} | {float(row['center_tip_distance_ratio']):.3f} | {float(row['confidence']):.3f} | {row['guard_rejection_reasons']} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Proceed to board-style replay: {'yes' if proceed else 'no'}",
            f"- Identity accepted MAE: {identity['accepted_calibrated_mae_c']:.3f} C",
            f"- Medium accepted MAE: {medium['accepted_calibrated_mae_c']:.3f} C",
            f"- Strong accepted MAE: {strong['accepted_calibrated_mae_c']:.3f} C",
            f"- Worst accepted error after gating: {max(identity['accepted_worst_error_c'], medium['accepted_worst_error_c'], strong['accepted_worst_error_c']):.3f} C",
            f"- Medium acceptance rate: {medium['acceptance_rate']:.3f}",
            f"- Rejection rate (identity/medium/strong): {1.0 - identity['acceptance_rate']:.3f} / {1.0 - medium['acceptance_rate']:.3f} / {1.0 - strong['acceptance_rate']:.3f}",
            "",
        ]
    )
    if proceed:
        lines.extend(
            [
            "- The guardrails are strict enough to remove the catastrophic tail while preserving enough coverage for board-style replay.",
            "- The remaining accepted predictions are materially better than the coordinate baseline.",
        ]
    )
    else:
        lines.extend(
        [
            "- The current policy is still too brittle for deployment.",
            "- Next recommended training change: heatmap_v3 with stronger jitter and heavier tip supervision.",
        ]
    )
    lines.extend(
        [
            "",
            "## Explicit Answers",
            "",
            f"- Does guarded heatmap_v2 reduce the catastrophic tail enough? {'yes' if proceed else 'no'}",
            f"- What is the accepted MAE on identity, mild, medium, and strong jitter? {identity['accepted_calibrated_mae_c']:.3f} / {metrics_by_level['mild']['accepted_calibrated_mae_c']:.3f} / {medium['accepted_calibrated_mae_c']:.3f} / {strong['accepted_calibrated_mae_c']:.3f} C",
            f"- What is the worst accepted error after gating? {max(identity['accepted_worst_error_c'], medium['accepted_worst_error_c'], strong['accepted_worst_error_c']):.3f} C",
            f"- What percentage of predictions are rejected? identity {1.0 - identity['acceptance_rate']:.3f}, mild {1.0 - metrics_by_level['mild']['acceptance_rate']:.3f}, medium {1.0 - medium['acceptance_rate']:.3f}, strong {1.0 - strong['acceptance_rate']:.3f}",
            f"- Are rejections reasonable, or too aggressive? {'reasonable' if proceed else 'too aggressive or insufficient'}",
            f"- Should we proceed to board-style replay? {'yes' if proceed else 'no'}",
            f"- Or should we train heatmap_v3 with stronger jitter and/or better tip supervision? {'no' if proceed else 'yes'}",
            "",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """Evaluate the selected guardrails on all jitter levels."""

    parser = argparse.ArgumentParser(description="Evaluate geometry heatmap v2 with guardrails")
    parser.add_argument(
        "--failure-analysis-csv",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2_guarded/failure_analysis.csv"),
        help="Per-prediction failure analysis CSV.",
    )
    parser.add_argument(
        "--thresholds-json",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2_guarded/selected_guardrail_thresholds.json"),
        help="Selected threshold JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2_guarded"),
        help="Output directory for guarded predictions.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_v2_guarded_eval.md"),
        help="Guarded evaluation report.",
    )
    parser.add_argument(
        "--decision-report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_v2_deployment_readiness_decision.md"),
        help="Deployment readiness decision report.",
    )
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    failure_analysis_csv = base_path / args.failure_analysis_csv if not args.failure_analysis_csv.is_absolute() else args.failure_analysis_csv
    thresholds_json = base_path / args.thresholds_json if not args.thresholds_json.is_absolute() else args.thresholds_json
    output_dir = base_path / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    report_path = base_path / args.report_path if not args.report_path.is_absolute() else args.report_path
    decision_report_path = (
        base_path / args.decision_report_path if not args.decision_report_path.is_absolute() else args.decision_report_path
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    decision_report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(failure_analysis_csv, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {failure_analysis_csv}")

    thresholds = _load_thresholds(thresholds_json)

    by_level: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_level.setdefault(str(row["jitter_level"]), []).append(row)

    guarded_rows: list[dict[str, Any]] = []
    metrics_by_level: dict[str, dict[str, float]] = {}
    rejected_reason_counts: dict[str, int] = {}
    catastrophic_total = 0
    catastrophic_rejected = 0

    for level in ["identity", "mild", "medium", "strong"]:
        level_rows = by_level[level]
        accepted_rows: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        clamped_rows: list[dict[str, Any]] = []
        for row in level_rows:
            status, temperature_c, reasons = _apply_thresholds(row, thresholds)
            guarded_row = dict(row)
            guarded_row["guard_status"] = status
            guarded_row["guard_temperature_c"] = temperature_c
            guarded_row["guard_rejection_reasons"] = "|".join(reasons)
            guarded_row["guard_absolute_error_c"] = abs(temperature_c - _safe_float(row["true_temperature_c"])) if status != "rejected" else math.nan
            guarded_rows.append(guarded_row)
            if _safe_float(row["absolute_error_c_calibrated"]) > 20.0:
                catastrophic_total += 1
                if status == "rejected":
                    catastrophic_rejected += 1
            if status == "rejected":
                rejected_rows.append(guarded_row)
                for reason in reasons:
                    rejected_reason_counts[reason] = rejected_reason_counts.get(reason, 0) + 1
            else:
                accepted_rows.append(guarded_row)
                if status == "clamped":
                    clamped_rows.append(guarded_row)

        accepted_errors = [float(row["guard_absolute_error_c"]) for row in accepted_rows]
        rejected_errors = [abs(_safe_float(row["predicted_temperature_c_calibrated"]) - _safe_float(row["true_temperature_c"])) for row in rejected_rows]
        metrics_by_level[level] = {
            "total_predictions": float(len(level_rows)),
            "accepted_predictions": float(len(accepted_rows)),
            "rejected_predictions": float(len(rejected_rows)),
            "clamped_predictions": float(len(clamped_rows)),
            "acceptance_rate": float(len(accepted_rows) / len(level_rows) if level_rows else math.nan),
            "accepted_calibrated_mae_c": float(np.mean(accepted_errors)) if accepted_errors else math.nan,
            "accepted_worst_error_c": float(np.max(accepted_errors)) if accepted_errors else math.nan,
            "accepted_percentage_under_2c": float(np.mean(np.asarray(accepted_errors) < 2.0) * 100.0) if accepted_errors else math.nan,
            "accepted_percentage_under_5c": float(np.mean(np.asarray(accepted_errors) < 5.0) * 100.0) if accepted_errors else math.nan,
            "accepted_percentage_under_10c": float(np.mean(np.asarray(accepted_errors) < 10.0) * 100.0) if accepted_errors else math.nan,
            "rejected_case_mean_raw_error_c": float(np.mean(rejected_errors)) if rejected_errors else math.nan,
            "rejected_case_worst_raw_error_c": float(np.max(rejected_errors)) if rejected_errors else math.nan,
        }

    _write_csv(guarded_rows, output_dir / "guarded_predictions.csv")
    accepted_rows_sorted = sorted(
        [row for row in guarded_rows if row["guard_status"] != "rejected"],
        key=lambda row: float(row["guard_absolute_error_c"]),
        reverse=True,
    )
    _write_csv(accepted_rows_sorted[:30], output_dir / "remaining_worst_accepted.csv")
    _write_json(
        {
            "thresholds": {
                "center_peak_min": thresholds.center_peak_min,
                "tip_peak_min": thresholds.tip_peak_min,
                "confidence_min": thresholds.confidence_min,
                "max_heatmap_entropy": thresholds.max_heatmap_entropy,
                "max_heatmap_spread_px": thresholds.max_heatmap_spread_px,
                "center_tip_distance_ratio_min": thresholds.center_tip_distance_ratio_min,
                "center_tip_distance_ratio_max": thresholds.center_tip_distance_ratio_max,
                "edge_margin_px": thresholds.edge_margin_px,
                "temperature_physical_margin_c": thresholds.temperature_physical_margin_c,
            },
            "metrics_by_level": metrics_by_level,
            "rejected_reason_counts": rejected_reason_counts,
        },
        output_dir / "guarded_summary.json",
    )

    _write_report(
        report_path=report_path,
        thresholds=thresholds,
        metrics_by_level=metrics_by_level,
        baseline_temperature_mae_c=7.91,
        baseline_center_mae_px=11.30,
        baseline_tip_mae_px=21.82,
        oracle_temperature_mae_c=1.195,
        worst_accepted_rows=accepted_rows_sorted,
        rejected_reason_counts=rejected_reason_counts,
        catastrophic_total=catastrophic_total,
        catastrophic_rejected=catastrophic_rejected,
    )

    # The decision report is intentionally derived from the same guarded metrics.
    _write_report(
        report_path=decision_report_path,
        thresholds=thresholds,
        metrics_by_level=metrics_by_level,
        baseline_temperature_mae_c=7.91,
        baseline_center_mae_px=11.30,
        baseline_tip_mae_px=21.82,
        oracle_temperature_mae_c=1.195,
        worst_accepted_rows=accepted_rows_sorted,
        rejected_reason_counts=rejected_reason_counts,
        catastrophic_total=catastrophic_total,
        catastrophic_rejected=catastrophic_rejected,
    )

    print(f"Guarded predictions CSV: {output_dir / 'guarded_predictions.csv'}")
    print(f"Remaining worst accepted CSV: {output_dir / 'remaining_worst_accepted.csv'}")
    print(f"Guarded evaluation report: {report_path}")
    print(f"Deployment readiness report: {decision_report_path}")


if __name__ == "__main__":
    main()
