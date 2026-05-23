#!/usr/bin/env python3
"""Evaluate the best possible temperature error from perfect geometry labels."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from embedded_gauge_reading_tinyml.geometry_temperature_calibration import (
    DEFAULT_COLD_ANGLE_DEGREES,
    DEFAULT_MAXIMUM_CELSIUS,
    DEFAULT_MINIMUM_CELSIUS,
    DEFAULT_SWEEP_DEGREES,
    GeometryTemperatureRecord,
    evaluate_candidate,
    load_clean_geometry_records,
    make_candidate_current,
    summarize_mae,
)


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a list of dictionaries to CSV with stable field ordering."""

    if not rows:
        raise ValueError("Cannot write an empty CSV.")

    preferred_order = [
        "image_path",
        "split",
        "source_manifest",
        "temperature_c",
        "angle_degrees",
        "current_temperature_c",
        "current_absolute_error_c",
        "source_width",
        "source_height",
        "dial_radius_source",
        "center_x_source",
        "center_y_source",
        "tip_x_source",
        "tip_y_source",
        "center_tip_distance_pixels",
    ]
    fieldnames = [field for field in preferred_order if field in rows[0]]
    fieldnames.extend(field for field in rows[0].keys() if field not in fieldnames)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _bin_edges(values: Iterable[float], *, bins: int = 4) -> list[float]:
    """Compute quantile bin edges for the dial radius groups."""

    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return []

    quantiles = np.quantile(np.asarray(sorted_values, dtype=np.float64), np.linspace(0.0, 1.0, bins + 1))
    edges = [float(edge) for edge in quantiles]
    edges[0] = float(sorted_values[0])
    edges[-1] = float(sorted_values[-1])
    return edges


def _quantile_bin_label(value: float, edges: list[float]) -> str:
    """Convert a value into a human-readable bin label."""

    if len(edges) < 2:
        return "all"

    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        is_last = index == len(edges) - 2
        if lower <= value <= upper or (not is_last and lower <= value < upper):
            return f"[{lower:.1f}, {upper:.1f}]"
    return f"[{edges[-2]:.1f}, {edges[-1]:.1f}]"


def _group_summary_numeric_bins(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    label_name: str,
    edges: list[float],
) -> list[dict[str, Any]]:
    """Aggregate summary metrics for a numeric field binned by quantiles."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = _quantile_bin_label(float(row[value_key]), edges)
        grouped[label].append(row)

    summary_rows: list[dict[str, Any]] = []
    for label, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        summary_rows.append(
            {
                label_name: label,
                "count": len(group),
                "mae_c": float(sum(abs(float(row["current_absolute_error_c"])) for row in group) / len(group)),
                "mean_radius": float(sum(float(row[value_key]) for row in group) / len(group)),
            }
        )
    return summary_rows


def _format_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    """Render a compact markdown table."""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if index == 0 else "---:" for index, _ in enumerate(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _build_report(
    *,
    records: list[GeometryTemperatureRecord],
    evaluated_rows: list[dict[str, Any]],
    output_path: Path,
    worst_csv_path: Path,
) -> None:
    """Write the oracle report and summary tables."""

    overall_mae = summarize_mae(evaluated_rows)
    splits = ["train", "val", "test"]
    split_rows = []
    for split in splits:
        split_eval = [row for row in evaluated_rows if row["split"] == split]
        split_rows.append(
            {
                "split": split,
                "count": len(split_eval),
                "mae_c": summarize_mae(split_eval),
                "max_error_c": max(float(row["current_absolute_error_c"]) for row in split_eval),
            }
        )

    temp_rows = []
    for temp in sorted({float(record.temperature_c) for record in records}):
        group = [row for row in evaluated_rows if math.isclose(float(row["temperature_c"]), temp, abs_tol=1e-9)]
        temp_rows.append(
            {
                "temperature_c": temp,
                "count": len(group),
                "mae_c": summarize_mae(group),
                "max_error_c": max(float(row["current_absolute_error_c"]) for row in group),
            }
        )

    dimension_groups = defaultdict(list)
    for row in evaluated_rows:
        key = f"{int(row['source_width'])}x{int(row['source_height'])}"
        dimension_groups[key].append(row)
    dimension_rows = [
        {
            "image_dims": key,
            "count": len(group),
            "mae_c": summarize_mae(group),
            "max_error_c": max(float(row["current_absolute_error_c"]) for row in group),
        }
        for key, group in dimension_groups.items()
    ]

    source_manifest_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated_rows:
        source_manifest_groups[str(row["source_manifest"])].append(row)
    source_manifest_rows = [
        {
            "source_manifest": key,
            "count": len(group),
            "mae_c": summarize_mae(group),
            "max_error_c": max(float(row["current_absolute_error_c"]) for row in group),
        }
        for key, group in sorted(source_manifest_groups.items(), key=lambda item: (-len(item[1]), item[0]))
    ]

    radius_edges = _bin_edges([float(record.dial_radius_source) for record in records], bins=4)
    radius_rows = _group_summary_numeric_bins(
        evaluated_rows,
        value_key="dial_radius_source",
        label_name="dial_radius_bin",
        edges=radius_edges,
    )

    worst_rows = sorted(evaluated_rows, key=lambda row: float(row["current_absolute_error_c"]), reverse=True)[:50]
    _write_csv(worst_rows, worst_csv_path)

    lines = [
        "# Oracle Geometry Temperature v1",
        "",
        "## Run Summary",
        "",
        f"- Clean rows evaluated: {len(records)}",
        f"- Current mapping: cold_angle={DEFAULT_COLD_ANGLE_DEGREES:.1f}, sweep={DEFAULT_SWEEP_DEGREES:.1f}, min={DEFAULT_MINIMUM_CELSIUS:.1f}, max={DEFAULT_MAXIMUM_CELSIUS:.1f}",
        f"- Oracle MAE with perfect geometry: {overall_mae:.3f} C",
        "",
        "## Split Metrics",
        "",
        _format_table(["split", "count", "mae_c", "max_error_c"], split_rows),
        "",
        "## Temperature Label Metrics",
        "",
        _format_table(["temperature_c", "count", "mae_c", "max_error_c"], temp_rows),
        "",
        "## Image Dimension Metrics",
        "",
        _format_table(["image_dims", "count", "mae_c", "max_error_c"], dimension_rows),
        "",
        "## Source Batch Metrics",
        "",
        _format_table(["source_manifest", "count", "mae_c", "max_error_c"], source_manifest_rows),
        "",
        "## Dial Radius Bins",
        "",
        f"- Bin edges: {', '.join(f'{edge:.1f}' for edge in radius_edges)}",
        "",
        _format_table(["dial_radius_bin", "count", "mae_c", "mean_radius"], radius_rows),
        "",
        "## Worst 10 Mismatches",
        "",
        _format_table(
            ["image_path", "split", "temperature_c", "current_temperature_c", "current_absolute_error_c", "source_manifest"],
            worst_rows[:10],
        ),
        "",
        "## Interpretation",
        "",
        "- This is the irreducible temperature error from geometry alone under the current mapping.",
        "- Any model that predicts the same center/tip labels perfectly cannot beat this ceiling without a better angle-to-temperature calibration or cleaner labels.",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """Evaluate perfect geometry under the current angle-to-temperature mapping."""

    parser = argparse.ArgumentParser(description="Evaluate oracle geometry temperature error")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"),
        help="Path to the clean manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/training/oracle_geometry_temperature_v1"),
        help="Directory for oracle artifacts.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/oracle_geometry_temperature_v1.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--worst-csv-path",
        type=Path,
        default=Path("ml/artifacts/training/oracle_geometry_temperature_v1/worst_50_oracle_mismatches.csv"),
        help="CSV path for the worst mismatches.",
    )
    parser.add_argument(
        "--predictions-csv-path",
        type=Path,
        default=Path("ml/artifacts/training/oracle_geometry_temperature_v1/oracle_predictions.csv"),
        help="CSV path for all oracle predictions.",
    )
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    manifest_path = base_path / args.manifest_path if not args.manifest_path.is_absolute() else args.manifest_path
    output_dir = base_path / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    report_path = base_path / args.report_path if not args.report_path.is_absolute() else args.report_path
    worst_csv_path = base_path / args.worst_csv_path if not args.worst_csv_path.is_absolute() else args.worst_csv_path
    predictions_csv_path = (
        base_path / args.predictions_csv_path if not args.predictions_csv_path.is_absolute() else args.predictions_csv_path
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    worst_csv_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_csv_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_clean_geometry_records(manifest_path)
    current_candidate = make_candidate_current()
    evaluated_rows = evaluate_candidate(records, current_candidate)

    _write_csv(evaluated_rows, predictions_csv_path)
    _build_report(
        records=records,
        evaluated_rows=evaluated_rows,
        output_path=report_path,
        worst_csv_path=worst_csv_path,
    )

    print(f"Oracle predictions written to {predictions_csv_path}")
    print(f"Worst mismatches written to {worst_csv_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
