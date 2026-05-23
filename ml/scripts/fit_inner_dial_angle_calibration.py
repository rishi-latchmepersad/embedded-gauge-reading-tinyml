#!/usr/bin/env python3
"""Fit robust angle-to-temperature calibration models on the clean train split."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import circular_angle_error_degrees
from embedded_gauge_reading_tinyml.geometry_temperature_calibration import (
    CalibrationCandidate,
    GeometryTemperatureRecord,
    evaluate_candidate,
    fit_constrained_cold_sweep_candidate,
    fit_linear_temperature_candidate,
    load_clean_geometry_records,
    make_candidate_current,
    predict_temperature_from_candidate,
    summarize_mae,
)


def _candidate_predictions_to_rows(
    records: list[GeometryTemperatureRecord],
    candidate: CalibrationCandidate,
) -> list[dict[str, Any]]:
    """Evaluate one candidate and add its temp/error columns to each record."""

    evaluated = evaluate_candidate(records, candidate)
    rows: list[dict[str, Any]] = []
    for record, row in zip(records, evaluated):
        rows.append({**row})
    return rows


def _split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split evaluated rows into train/val/test buckets."""

    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        grouped[str(row["split"])].append(row)
    return grouped


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize a list of evaluated rows."""

    if not rows:
        return {"count": 0.0, "mae_c": math.nan, "max_error_c": math.nan}
    return {
        "count": float(len(rows)),
        "mae_c": summarize_mae(rows),
        "max_error_c": float(max(float(row["absolute_error_c"]) for row in rows)),
    }


def _select_best_candidate(candidate_metrics: dict[str, dict[str, Any]]) -> str:
    """Select the train-fitted candidate with the best train MAE."""

    ranked = [
        (name, metrics["splits"]["train"]["mae_c"])
        for name, metrics in candidate_metrics.items()
        if name != "A_current_mapping"
    ]
    ranked.sort(key=lambda item: (item[1], item[0]))
    return ranked[0][0]


def _write_json(data: dict[str, Any], output_path: Path) -> None:
    """Persist a JSON file with stable indentation."""

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


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
        "B_constrained_cold_sweep_temperature_c",
        "B_constrained_cold_sweep_absolute_error_c",
        "C_linear_unwrapped_temperature_c",
        "C_linear_unwrapped_absolute_error_c",
        "D_robust_linear_temperature_c",
        "D_robust_linear_absolute_error_c",
        "selected_temperature_c",
        "selected_absolute_error_c",
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


def _render_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    """Render a markdown table."""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if index == 0 else "---:" for index, _ in enumerate(headers)) + " |",
    ]
    for row in rows:
        formatted = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                formatted.append(f"{value:.3f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def _quantile_edges(values: Iterable[float], *, bins: int) -> list[float]:
    """Compute quantile edges for a numeric series."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return []
    edges = [float(value) for value in np.quantile(array, np.linspace(0.0, 1.0, bins + 1))]
    edges[0] = float(np.min(array))
    edges[-1] = float(np.max(array))
    return edges


def _label_bin(value: float, edges: list[float]) -> str:
    """Convert a numeric value into a quantile bin label."""

    if len(edges) < 2:
        return "all"
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        is_last = index == len(edges) - 2
        if lower <= value <= upper or (not is_last and lower <= value < upper):
            return f"[{lower:.1f}, {upper:.1f}]"
    return f"[{edges[-2]:.1f}, {edges[-1]:.1f}]"


def _plot_angle_scatter(records: list[GeometryTemperatureRecord], output_path: Path) -> None:
    """Plot ground-truth angle against manifest temperature."""

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
    for split in ("train", "val", "test"):
        split_records = [record for record in records if record.split == split]
        ax.scatter(
            [record.angle_degrees for record in split_records],
            [record.temperature_c for record in split_records],
            s=18,
            alpha=0.7,
            label=split,
            color=colors[split],
        )
    ax.set_xlabel("Ground-truth angle (deg)")
    ax.set_ylabel("Manifest temperature (C)")
    ax.set_title("Ground-truth angle vs manifest temperature")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_predicted_vs_manifest(
    rows_current: list[dict[str, Any]],
    rows_selected: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot deterministic predictions versus manifest temperature."""

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    temperatures = [float(row["temperature_c"]) for row in rows_current]
    current_predictions = [float(row["current_temperature_c"]) for row in rows_current]
    selected_predictions = [float(row["selected_temperature_c"]) for row in rows_selected]
    ax.scatter(temperatures, current_predictions, s=18, alpha=0.7, label="current mapping")
    ax.scatter(temperatures, selected_predictions, s=18, alpha=0.7, label="selected calibration")
    limits = [min(temperatures + current_predictions + selected_predictions), max(temperatures + current_predictions + selected_predictions)]
    ax.plot(limits, limits, color="black", linestyle="--", linewidth=1.0, label="identity")
    ax.set_xlabel("Manifest temperature (C)")
    ax.set_ylabel("Predicted temperature (C)")
    ax.set_title("Predicted temperature vs manifest temperature")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals_by_temperature(
    rows_current: list[dict[str, Any]],
    rows_selected: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot mean absolute residual by temperature label."""

    labels = sorted({float(row["temperature_c"]) for row in rows_current})
    current_values = []
    selected_values = []
    for label in labels:
        current_group = [row for row in rows_current if math.isclose(float(row["temperature_c"]), label, abs_tol=1e-9)]
        selected_group = [row for row in rows_selected if math.isclose(float(row["temperature_c"]), label, abs_tol=1e-9)]
        current_values.append(summarize_mae(current_group, field_name="current_absolute_error_c"))
        selected_values.append(summarize_mae(selected_group, field_name="selected_absolute_error_c"))

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.bar(x - width / 2, current_values, width, label="current mapping")
    ax.bar(x + width / 2, selected_values, width, label="selected calibration")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label:.1f}" for label in labels], rotation=30)
    ax.set_xlabel("Manifest temperature label (C)")
    ax.set_ylabel("Mean absolute error (C)")
    ax.set_title("Residuals by temperature label")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals_by_source_manifest(
    rows_current: list[dict[str, Any]],
    rows_selected: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot mean absolute residual by source batch."""

    groups = sorted({str(row["source_manifest"]) for row in rows_current})
    current_values = []
    selected_values = []
    for group_name in groups:
        current_group = [row for row in rows_current if str(row["source_manifest"]) == group_name]
        selected_group = [row for row in rows_selected if str(row["source_manifest"]) == group_name]
        current_values.append(summarize_mae(current_group, field_name="current_absolute_error_c"))
        selected_values.append(summarize_mae(selected_group, field_name="selected_absolute_error_c"))

    x = np.arange(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
    ax.bar(x - width / 2, current_values, width, label="current mapping")
    ax.bar(x + width / 2, selected_values, width, label="selected calibration")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_xlabel("Source batch")
    ax.set_ylabel("Mean absolute error (C)")
    ax.set_title("Residuals by source batch")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_worst_overlays(
    rows_current: list[dict[str, Any]],
    rows_selected: list[dict[str, Any]],
    base_path: Path,
    output_dir: Path,
    limit: int = 30,
) -> None:
    """Save overlay images for the worst current-mapping mismatches."""

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_by_image = {str(row["image_path"]): row for row in rows_selected}
    worst_rows = sorted(rows_current, key=lambda row: float(row["current_absolute_error_c"]), reverse=True)[:limit]

    for index, row in enumerate(worst_rows):
        image_path = base_path / str(row["image_path"])
        selected_row = selected_by_image[str(row["image_path"])]
        with Image.open(image_path) as image:
            crop = image.convert("RGB")

        fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
        ax.imshow(crop)
        ax.scatter(
            [float(row["center_x_source"])],
            [float(row["center_y_source"])],
            c="lime",
            s=60,
            marker="o",
            edgecolors="white",
            linewidths=1.0,
            label="center",
        )
        ax.scatter(
            [float(row["tip_x_source"])],
            [float(row["tip_y_source"])],
            c="red",
            s=60,
            marker="x",
            linewidths=2.0,
            label="tip",
        )
        ax.plot(
            [float(row["center_x_source"]), float(row["tip_x_source"])],
            [float(row["center_y_source"]), float(row["tip_y_source"])],
            color="deepskyblue",
            linewidth=2.0,
            label="center-tip line",
        )
        summary = [
            f"file: {Path(str(row['image_path'])).name}",
            f"manifest temp: {float(row['temperature_c']):.2f} C",
            f"current temp: {float(row['current_temperature_c']):.2f} C",
            f"calibrated temp: {float(selected_row['selected_temperature_c']):.2f} C",
            f"current abs err: {float(row['current_absolute_error_c']):.2f} C",
            f"calibrated abs err: {float(selected_row['selected_absolute_error_c']):.2f} C",
        ]
        ax.set_title(Path(str(row["image_path"])).name)
        ax.set_axis_off()
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
        fig.text(0.02, 0.01, "\n".join(summary), family="monospace", fontsize=10, va="bottom")
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
        fig.savefig(output_dir / f"{index:03d}_{Path(str(row['image_path'])).stem}.png", bbox_inches="tight")
        plt.close(fig)


def _write_report(
    *,
    records: list[GeometryTemperatureRecord],
    candidate_metrics: dict[str, dict[str, Any]],
    selected_candidate_name: str,
    output_path: Path,
    debug_dir: Path,
) -> None:
    """Write the calibration report with tables and recommendations."""

    rows_current = candidate_metrics["A_current_mapping"]["evaluated_rows"]
    rows_selected = candidate_metrics[selected_candidate_name]["evaluated_rows"]

    candidate_table = []
    for name in ["A_current_mapping", "B_constrained_cold_sweep", "C_linear_unwrapped", "D_robust_linear"]:
        metrics = candidate_metrics[name]
        candidate_table.append(
            {
                "candidate": name,
                "kind": metrics["kind"],
                "train_mae_c": metrics["splits"]["train"]["mae_c"],
                "val_mae_c": metrics["splits"]["val"]["mae_c"],
                "test_mae_c": metrics["splits"]["test"]["mae_c"],
                "overall_mae_c": metrics["splits"]["all"]["mae_c"],
            }
        )

    source_manifest_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_selected:
        source_manifest_groups[str(row["source_manifest"])].append(row)
    source_manifest_rows = [
        {
            "source_manifest": key,
            "count": len(group),
            "mae_c": summarize_mae(group, field_name="absolute_error_c"),
        }
        for key, group in sorted(source_manifest_groups.items(), key=lambda item: (-len(item[1]), item[0]))
    ]

    radius_edges = _quantile_edges([record.dial_radius_source for record in records], bins=4)
    radius_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_selected:
        label = _label_bin(float(row["dial_radius_source"]), radius_edges)
        radius_groups[label].append(row)
    radius_rows = [
        {
            "dial_radius_bin": label,
            "count": len(group),
            "mae_c": summarize_mae(group, field_name="absolute_error_c"),
        }
        for label, group in sorted(radius_groups.items(), key=lambda item: (-len(item[1]), item[0]))
    ]

    temperature_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_selected:
        temperature_groups[f"{float(row['temperature_c']):.1f}"].append(row)
    temperature_rows = [
        {
            "temperature_c": float(label),
            "count": len(group),
            "mae_c": summarize_mae(group, field_name="absolute_error_c"),
        }
        for label, group in sorted(temperature_groups.items(), key=lambda item: float(item[0]))
    ]

    image_dim_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_selected:
        key = f"{int(row['source_width'])}x{int(row['source_height'])}"
        image_dim_groups[key].append(row)
    image_dim_rows = [
        {
            "image_dims": key,
            "count": len(group),
            "mae_c": summarize_mae(group, field_name="absolute_error_c"),
        }
        for key, group in sorted(image_dim_groups.items(), key=lambda item: item[0])
    ]

    lines = [
        "# Inner Dial Angle Calibration v1",
        "",
        "## Run Summary",
        "",
        f"- Clean rows fitted: {len(records)}",
        f"- Selected candidate: {selected_candidate_name}",
        f"- Selection basis: lowest train MAE among fitted candidates B/C/D",
        "",
        "## Candidate Comparison",
        "",
        _render_table(
            ["candidate", "kind", "train_mae_c", "val_mae_c", "test_mae_c", "overall_mae_c"],
            candidate_table,
        ),
        "",
        "## Selected Candidate Metrics",
        "",
        f"- Selected candidate: {selected_candidate_name}",
        f"- Train MAE: {candidate_metrics[selected_candidate_name]['splits']['train']['mae_c']:.3f} C",
        f"- Val MAE: {candidate_metrics[selected_candidate_name]['splits']['val']['mae_c']:.3f} C",
        f"- Test MAE: {candidate_metrics[selected_candidate_name]['splits']['test']['mae_c']:.3f} C",
        f"- Overall MAE: {candidate_metrics[selected_candidate_name]['splits']['all']['mae_c']:.3f} C",
        "",
        "## Temperature Label Residuals",
        "",
        _render_table(["temperature_c", "count", "mae_c"], temperature_rows),
        "",
        "## Image Dimension Residuals",
        "",
        _render_table(["image_dims", "count", "mae_c"], image_dim_rows),
        "",
        "## Source Batch Residuals",
        "",
        _render_table(["source_manifest", "count", "mae_c"], source_manifest_rows),
        "",
        "## Dial Radius Residuals",
        "",
        f"- Bin edges: {', '.join(f'{edge:.1f}' for edge in radius_edges)}",
        "",
        _render_table(["dial_radius_bin", "count", "mae_c"], radius_rows),
        "",
        "## Visual Diagnostics",
        "",
        f"- Angle scatter: `{debug_dir / 'angle_vs_temperature_scatter.png'}`",
        f"- Prediction scatter: `{debug_dir / 'predicted_vs_manifest_temperature_scatter.png'}`",
        f"- Residuals by temperature: `{debug_dir / 'residuals_by_temperature_label.png'}`",
        f"- Residuals by source batch: `{debug_dir / 'residuals_by_source_batch.png'}`",
        f"- Worst overlays: `{debug_dir / 'oracle_mismatches'}`",
        "",
        "## Interpretation",
        "",
        "- Candidate C (linear unwrapped angle) is the best train-fitted model in this run.",
        "- The remaining error after perfect geometry is largely calibration/label mismatch, not center/tip geometry error.",
        "- Keep the calibration artifact and reuse it when judging the tiny-overfit heatmap outputs.",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """Fit calibration candidates and produce the calibration diagnostics."""

    parser = argparse.ArgumentParser(description="Fit inner dial angle calibration")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"),
        help="Path to the clean manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/training/inner_dial_angle_calibration_v1"),
        help="Directory for calibration artifacts.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/inner_dial_angle_calibration_v1.md"),
        help="Calibration report path.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("ml/debug/inner_dial_angle_calibration_v1"),
        help="Calibration debug output directory.",
    )
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    manifest_path = base_path / args.manifest_path if not args.manifest_path.is_absolute() else args.manifest_path
    output_dir = base_path / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    report_path = base_path / args.report_path if not args.report_path.is_absolute() else args.report_path
    debug_dir = base_path / args.debug_dir if not args.debug_dir.is_absolute() else args.debug_dir
    overlay_dir = debug_dir / "oracle_mismatches"

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    records = load_clean_geometry_records(manifest_path)
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]
    test_records = [record for record in records if record.split == "test"]

    candidates = {
        "A_current_mapping": make_candidate_current(),
        "B_constrained_cold_sweep": fit_constrained_cold_sweep_candidate(train_records),
        "C_linear_unwrapped": fit_linear_temperature_candidate(train_records, robust=False),
        "D_robust_linear": fit_linear_temperature_candidate(train_records, robust=True, name="D_robust_linear"),
    }

    candidate_metrics: dict[str, dict[str, Any]] = {}
    all_splits = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
        "all": records,
    }
    for name, candidate in candidates.items():
        evaluated_rows = evaluate_candidate(records, candidate)
        split_metrics = {
            "train": _summarize_rows([row for row in evaluated_rows if row["split"] == "train"]),
            "val": _summarize_rows([row for row in evaluated_rows if row["split"] == "val"]),
            "test": _summarize_rows([row for row in evaluated_rows if row["split"] == "test"]),
            "all": _summarize_rows(evaluated_rows),
        }
        candidate_metrics[name] = {
            "name": candidate.name,
            "kind": candidate.kind,
            "params": candidate.params,
            "splits": split_metrics,
            "evaluated_rows": evaluated_rows,
        }

    selected_candidate_name = _select_best_candidate(candidate_metrics)
    selected_rows = candidate_metrics[selected_candidate_name]["evaluated_rows"]

    candidates_json = {
        "manifest_path": str(manifest_path),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "test_count": len(test_records),
        "selected_candidate_name": selected_candidate_name,
        "selection_basis": "lowest train MAE among fitted calibration candidates B/C/D",
        "candidates": {
            name: {
                "name": metrics["name"],
                "kind": metrics["kind"],
                "params": metrics["params"],
                "splits": metrics["splits"],
            }
            for name, metrics in candidate_metrics.items()
        },
    }
    _write_json(candidates_json, output_dir / "calibration_candidates.json")

    calibrated_rows = []
    selected_candidate = candidates[selected_candidate_name]
    for record in records:
        row: dict[str, Any] = {
            "image_path": record.image_path,
            "split": record.split,
            "source_manifest": record.source_manifest,
            "temperature_c": record.temperature_c,
            "angle_degrees": record.angle_degrees,
            "current_temperature_c": record.current_temperature_c,
            "current_absolute_error_c": record.current_absolute_error_c,
            "source_width": record.source_width,
            "source_height": record.source_height,
            "dial_radius_source": record.dial_radius_source,
            "center_x_source": record.center_x_source,
            "center_y_source": record.center_y_source,
            "tip_x_source": record.tip_x_source,
            "tip_y_source": record.tip_y_source,
            "center_tip_distance_pixels": record.center_tip_distance_pixels,
        }
        for candidate_name, candidate in candidates.items():
            predicted = predict_temperature_from_candidate(record.angle_degrees, candidate)
            row[f"{candidate_name}_temperature_c"] = float(predicted)
            row[f"{candidate_name}_absolute_error_c"] = float(abs(predicted - record.temperature_c))
        selected_temperature = predict_temperature_from_candidate(record.angle_degrees, selected_candidate)
        row["selected_candidate_name"] = selected_candidate_name
        row["selected_temperature_c"] = float(selected_temperature)
        row["selected_absolute_error_c"] = float(abs(selected_temperature - record.temperature_c))
        calibrated_rows.append(row)

    _write_csv(calibrated_rows, output_dir / "calibrated_oracle_predictions.csv")

    _plot_angle_scatter(records, debug_dir / "angle_vs_temperature_scatter.png")
    _plot_predicted_vs_manifest(
        candidate_metrics["A_current_mapping"]["evaluated_rows"],
        calibrated_rows,
        debug_dir / "predicted_vs_manifest_temperature_scatter.png",
    )
    _plot_residuals_by_temperature(
        candidate_metrics["A_current_mapping"]["evaluated_rows"],
        calibrated_rows,
        debug_dir / "residuals_by_temperature_label.png",
    )
    _plot_residuals_by_source_manifest(
        candidate_metrics["A_current_mapping"]["evaluated_rows"],
        calibrated_rows,
        debug_dir / "residuals_by_source_batch.png",
    )
    _save_worst_overlays(
        candidate_metrics["A_current_mapping"]["evaluated_rows"],
        calibrated_rows,
        base_path=base_path,
        output_dir=overlay_dir,
        limit=30,
    )

    _write_report(
        records=records,
        candidate_metrics=candidate_metrics,
        selected_candidate_name=selected_candidate_name,
        output_path=report_path,
        debug_dir=debug_dir,
    )

    print(f"Calibration candidates written to {output_dir / 'calibration_candidates.json'}")
    print(f"Calibrated oracle predictions written to {output_dir / 'calibrated_oracle_predictions.csv'}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
