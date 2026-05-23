#!/usr/bin/env python3
"""Micro-sweep board replay guardrails on train/val only.

The sweep consumes saved board-replay predictions for the selected preprocessing
mode so no model inference or test-set tuning is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_board_replay_guardrails import (
    board_replay_candidate_grid,
    candidate_relaxation_key,
    summarize_board_replay_rows,
)
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import GeometryGuardrailThresholds


SELECTED_PREPROCESSING_MODE = "python_training_rgb_bilinear"


def _resolve_path(base_path: Path, maybe_relative: Path) -> Path:
    """Resolve a path relative to the repo root when needed."""

    return maybe_relative if maybe_relative.is_absolute() else base_path / maybe_relative


def _load_rows(csv_path: Path, *, preprocessing_mode: str) -> list[dict[str, Any]]:
    """Load saved board replay rows for one preprocessing mode."""

    rows: list[dict[str, Any]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row["preprocessing_mode"]) != preprocessing_mode:
                continue
            rows.append(row)
    return rows


def _write_csv(rows: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Write a list of dictionaries to CSV."""

    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _candidate_to_dict(thresholds: GeometryGuardrailThresholds) -> dict[str, Any]:
    """Serialize one threshold set for JSON and CSV output."""

    return {
        "center_peak_min": float(thresholds.center_peak_min),
        "tip_peak_min": float(thresholds.tip_peak_min),
        "confidence_min": float(thresholds.confidence_min),
        "max_heatmap_entropy": float(thresholds.max_heatmap_entropy),
        "max_heatmap_spread_px": float(thresholds.max_heatmap_spread_px),
        "center_tip_distance_ratio_min": float(thresholds.center_tip_distance_ratio_min),
        "center_tip_distance_ratio_max": float(thresholds.center_tip_distance_ratio_max),
        "edge_margin_px": float(thresholds.edge_margin_px),
        "temperature_physical_range_margin_c": float(thresholds.temperature_physical_margin_c),
        "minimum_celsius": float(thresholds.minimum_celsius),
        "maximum_celsius": float(thresholds.maximum_celsius),
        "clamp_temperature_to_physical_range": bool(thresholds.clamp_temperature_to_physical_range),
    }


def _passes_validation(summary: dict[str, Any]) -> bool:
    """Return whether one candidate satisfies the validation gate."""

    return (
        float(summary["accepted_mae_c"]) <= 4.5
        and float(summary["acceptance_rate"]) >= 0.65
        and float(summary["accepted_worst_error_c"]) < 20.0
        and int(summary["accepted_gt20c_failures"]) == 0
    )


def _select_candidate(
    candidate_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the least-relaxed candidate that passes validation."""

    passing = [candidate for candidate in candidate_summaries if _passes_validation(candidate["val_summary"])]
    if not passing:
        raise RuntimeError("No candidate satisfied the validation gate.")

    passing.sort(
        key=lambda candidate: (
            candidate_relaxation_key(candidate["thresholds"]),
            float(candidate["val_summary"]["accepted_worst_error_c"]),
            float(candidate["val_summary"]["accepted_mae_c"]),
            -float(candidate["val_summary"]["acceptance_rate"]),
        )
    )
    return passing[0]


def main() -> None:
    """Run the micro-sweep on saved train/val board replay rows."""

    parser = argparse.ArgumentParser(description="Sweep board replay guardrails on train/val")
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2_board_replay/board_replay_predictions.csv"),
        help="Saved board replay predictions from the first replay pass.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2_board_replay"),
        help="Directory for sweep artifacts.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_v2_board_guardrail_micro_sweep.md"),
        help="Markdown report path.",
    )
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    predictions_csv = _resolve_path(base_path, args.predictions_csv)
    output_dir = _resolve_path(base_path, args.output_dir)
    report_path = _resolve_path(base_path, args.report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(predictions_csv, preprocessing_mode=SELECTED_PREPROCESSING_MODE)
    split_rows = {
        split: [row for row in rows if str(row["split"]) == split]
        for split in ("train", "val")
    }
    candidate_summaries: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []

    for thresholds in board_replay_candidate_grid():
        train_summary = summarize_board_replay_rows(split_rows["train"], thresholds)
        val_summary = summarize_board_replay_rows(split_rows["val"], thresholds)
        candidate_summary = {
            "thresholds": thresholds,
            "thresholds_json": _candidate_to_dict(thresholds),
            "train_summary": train_summary,
            "val_summary": val_summary,
            "passes_validation": _passes_validation(val_summary),
        }
        candidate_summaries.append(candidate_summary)
        output_rows.append(
            {
                **_candidate_to_dict(thresholds),
                "train_accepted_mae_c": float(train_summary["accepted_mae_c"]),
                "train_acceptance_rate": float(train_summary["acceptance_rate"]),
                "train_accepted_worst_error_c": float(train_summary["accepted_worst_error_c"]),
                "train_percentage_under_2c": float(train_summary["percentage_under_2c"]),
                "train_percentage_under_5c": float(train_summary["percentage_under_5c"]),
                "train_percentage_under_10c": float(train_summary["percentage_under_10c"]),
                "train_accepted_gt20c_failures": int(train_summary["accepted_gt20c_failures"]),
                "train_top_rejection_reasons": ";".join(
                    f"{reason}:{count}" for reason, count in train_summary["top_rejection_reasons"]
                ),
                "val_accepted_mae_c": float(val_summary["accepted_mae_c"]),
                "val_acceptance_rate": float(val_summary["acceptance_rate"]),
                "val_accepted_worst_error_c": float(val_summary["accepted_worst_error_c"]),
                "val_percentage_under_2c": float(val_summary["percentage_under_2c"]),
                "val_percentage_under_5c": float(val_summary["percentage_under_5c"]),
                "val_percentage_under_10c": float(val_summary["percentage_under_10c"]),
                "val_accepted_gt20c_failures": int(val_summary["accepted_gt20c_failures"]),
                "val_top_rejection_reasons": ";".join(
                    f"{reason}:{count}" for reason, count in val_summary["top_rejection_reasons"]
                ),
                "passes_validation": bool(candidate_summary["passes_validation"]),
            }
        )

    selected_candidate = _select_candidate(candidate_summaries)
    selected_thresholds = selected_candidate["thresholds"]
    selected_json = {
        "selected_thresholds": _candidate_to_dict(selected_thresholds),
        "selection_rule": "least-relaxed validation-passing candidate, tie-broken by worst error, MAE, then acceptance",
        "selected_mode": SELECTED_PREPROCESSING_MODE,
        "validation_summary": selected_candidate["val_summary"],
        "train_summary": selected_candidate["train_summary"],
    }
    with open(output_dir / "selected_board_guardrail_thresholds.json", "w", encoding="utf-8") as handle:
        json.dump(selected_json, handle, indent=2, sort_keys=True)

    _write_csv(output_rows, output_dir / "board_guardrail_micro_sweep.csv")

    report_lines = [
        "# Geometry Heatmap v2 Board Guardrail Micro Sweep",
        "",
        f"- Selected preprocessing mode: `{SELECTED_PREPROCESSING_MODE}`",
        f"- Predictions source: `{predictions_csv}`",
        "",
        "## Selection Rule",
        "",
        "- Search the train/val grid only.",
        "- Keep center peak, confidence, and edge margin fixed.",
        "- Choose the least-relaxed candidate that passes validation.",
        "- Tie-break by validation worst accepted error, then MAE, then acceptance rate.",
        "",
        "## Selected Thresholds",
        "",
        "| threshold | value |",
        "| --- | ---: |",
    ]
    for key, value in _candidate_to_dict(selected_thresholds).items():
        report_lines.append(f"| {key} | {value} |")

    report_lines.extend(
        [
            "",
            "## Validation Summary",
            "",
            f"- accepted MAE: {selected_candidate['val_summary']['accepted_mae_c']:.3f} C",
            f"- acceptance rate: {selected_candidate['val_summary']['acceptance_rate']:.3f}",
            f"- worst accepted error: {selected_candidate['val_summary']['accepted_worst_error_c']:.3f} C",
            f"- accepted >20 C failures: {int(selected_candidate['val_summary']['accepted_gt20c_failures'])}",
            "",
            "## Train Summary",
            "",
            f"- accepted MAE: {selected_candidate['train_summary']['accepted_mae_c']:.3f} C",
            f"- acceptance rate: {selected_candidate['train_summary']['acceptance_rate']:.3f}",
            f"- worst accepted error: {selected_candidate['train_summary']['accepted_worst_error_c']:.3f} C",
            f"- accepted >20 C failures: {int(selected_candidate['train_summary']['accepted_gt20c_failures'])}",
            "",
            "## Top Validation Rejection Reasons",
            "",
            "| reason | count |",
            "| --- | ---: |",
        ]
    )
    for reason, count in selected_candidate["val_summary"]["top_rejection_reasons"]:
        report_lines.append(f"| {reason} | {count} |")

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines))

    print(f"Sweep CSV written to {output_dir / 'board_guardrail_micro_sweep.csv'}")
    print(f"Selected thresholds written to {output_dir / 'selected_board_guardrail_thresholds.json'}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
