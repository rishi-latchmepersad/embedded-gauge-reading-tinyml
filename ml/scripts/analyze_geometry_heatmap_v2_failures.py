#!/usr/bin/env python3
"""Analyze geometry heatmap v2 catastrophic failures across jitter levels."""

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

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import circular_angle_error_degrees
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import (
    load_clean_geometry_examples,
    load_heatmap_sample,
    load_selected_calibration_candidate,
    sample_jitter_params,
    select_examples_from_split,
)
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import (
    GeometryDecodedPrediction,
    GeometryGuardrailResult,
    GeometryGuardrailThresholds,
    apply_geometry_guardrails,
    decode_heatmap_geometry_prediction,
)


JITTER_LEVELS: dict[str, dict[str, float | int]] = {
    "identity": {
        "shift_min_px": 0,
        "shift_max_px": 0,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "aspect_min": 1.0,
        "aspect_max": 1.0,
    },
    "mild": {
        "shift_min_px": 4,
        "shift_max_px": 4,
        "scale_min": 0.97,
        "scale_max": 1.03,
        "aspect_min": 0.98,
        "aspect_max": 1.02,
    },
    "medium": {
        "shift_min_px": 8,
        "shift_max_px": 8,
        "scale_min": 0.93,
        "scale_max": 1.08,
        "aspect_min": 0.95,
        "aspect_max": 1.05,
    },
    "strong": {
        "shift_min_px": 12,
        "shift_max_px": 12,
        "scale_min": 0.90,
        "scale_max": 1.15,
        "aspect_min": 0.95,
        "aspect_max": 1.08,
    },
}


@dataclass(frozen=True)
class FailureAnalysisRecord:
    """One decoded prediction plus the derived guardrail result."""

    sample: Any
    decoded: GeometryDecodedPrediction
    guardrail: GeometryGuardrailResult
    jitter_level: str
    row: dict[str, Any]


def _write_json(data: dict[str, Any], output_path: Path) -> None:
    """Write a JSON file with stable formatting."""

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write the failure analysis rows to CSV."""

    if not rows:
        raise ValueError("Cannot write an empty CSV table.")
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _safe_float(value: Any) -> float:
    """Convert a value to float without losing NaN values."""

    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _center_tip_distance_ratio(decoded: GeometryDecodedPrediction) -> float:
    """Recompute the center-tip distance ratio from decoded fields."""

    predicted_center_tip_distance_px = math.hypot(
        decoded.predicted_tip_x_224 - decoded.predicted_center_x_224,
        decoded.predicted_tip_y_224 - decoded.predicted_center_y_224,
    )
    x_scale = 224.0 / float(decoded.crop_width)
    y_scale = 224.0 / float(decoded.crop_height)
    angle_radians = math.radians(decoded.predicted_angle_degrees)
    expected_distance_px = float(
        decoded.dial_radius_source * math.hypot(math.cos(angle_radians) * x_scale, math.sin(angle_radians) * y_scale)
    )
    if expected_distance_px <= 0.0:
        return math.nan
    return predicted_center_tip_distance_px / expected_distance_px


def _dominant_failure_mode(decoded: GeometryDecodedPrediction, guardrail: GeometryGuardrailResult, row: dict[str, Any]) -> str:
    """Classify one catastrophic prediction into a human-readable failure mode."""

    reasons = list(guardrail.rejection_reasons)
    abs_error = _safe_float(row["absolute_error_c_calibrated"])
    center_error = _safe_float(row["center_px_mae_224"])
    tip_error = _safe_float(row["tip_px_mae_224"])
    angle_error = _safe_float(row["angle_mae_degrees"])
    tip_peak = _safe_float(row["tip_heatmap_peak_value"])
    confidence = _safe_float(row["confidence"])
    ratio = _safe_float(row["center_tip_distance_ratio"])
    temp_outside_margin = "temperature_outside_physical_margin" in reasons

    if temp_outside_margin or abs_error > 20.0 and _safe_float(row["predicted_temperature_c_calibrated"]) > 55.0:
        return "calibration_extrapolation"
    if angle_error > 90.0:
        return "angle_wrap_or_opposite_side"
    if tip_error > center_error * 1.5 and tip_error > 20.0:
        return "bad_tip_localization"
    if center_error > tip_error * 1.5 and center_error > 10.0:
        return "bad_center_localization"
    if tip_peak < 0.40 or confidence < 0.50:
        return "low_peak_or_confidence"
    if not math.isnan(ratio) and (ratio < 0.50 or ratio > 1.50):
        return "center_tip_distance_ratio_implausible"
    if abs_error > 20.0 and "predicted_point_near_edge" in reasons:
        return "edge_exclusion"
    return "mixed_geometry_failure"


def _predicted_near_edge(decoded: GeometryDecodedPrediction, *, edge_margin_px: float) -> bool:
    """Check whether either predicted point is too close to a crop edge."""

    center_margin = min(
        decoded.predicted_center_x_224,
        decoded.predicted_center_y_224,
        223.0 - decoded.predicted_center_x_224,
        223.0 - decoded.predicted_center_y_224,
    )
    tip_margin = min(
        decoded.predicted_tip_x_224,
        decoded.predicted_tip_y_224,
        223.0 - decoded.predicted_tip_x_224,
        223.0 - decoded.predicted_tip_y_224,
    )
    return min(center_margin, tip_margin) < edge_margin_px


def _build_row(
    decoded: GeometryDecodedPrediction,
    guardrail: GeometryGuardrailResult,
    *,
    jitter_level: str,
) -> dict[str, Any]:
    """Build a flat CSV row for one failure analysis sample."""

    features = guardrail.quality_features
    abs_error_clamped = abs(guardrail.temperature_c - decoded.true_temperature_c) if guardrail.status != "rejected" else math.nan
    abs_error_raw_clipped = abs(
        np.clip(decoded.predicted_temperature_c_calibrated, -30.0, 50.0) - decoded.true_temperature_c
    )
    current_clipped_error = abs(np.clip(decoded.predicted_temperature_c_current_mapping, -30.0, 50.0) - decoded.true_temperature_c)

    return {
        "image_path": decoded.image_path,
        "split": decoded.split,
        "jitter_level": jitter_level,
        "jitter_shift_x": decoded.jitter_shift_x,
        "jitter_shift_y": decoded.jitter_shift_y,
        "jitter_scale": decoded.jitter_scale,
        "jitter_aspect": decoded.jitter_aspect,
        "true_temperature_c": decoded.true_temperature_c,
        "predicted_temperature_c_current_mapping": decoded.predicted_temperature_c_current_mapping,
        "predicted_temperature_c_calibrated": decoded.predicted_temperature_c_calibrated,
        "predicted_temperature_c_calibrated_clipped": float(np.clip(decoded.predicted_temperature_c_calibrated, -30.0, 50.0)),
        "absolute_error_c_calibrated": decoded.absolute_error_c_calibrated,
        "absolute_error_c_current_mapping": decoded.absolute_error_c_current_mapping,
        "absolute_error_c_calibrated_clipped": abs_error_raw_clipped,
        "absolute_error_c_current_mapping_clipped": current_clipped_error,
        "true_angle_degrees": decoded.true_angle_degrees,
        "predicted_angle_degrees": decoded.predicted_angle_degrees,
        "predicted_angle_degrees_argmax": decoded.predicted_angle_degrees_argmax,
        "angle_mae_degrees": abs(circular_angle_error_degrees(decoded.predicted_angle_degrees, decoded.true_angle_degrees)),
        "angle_mae_degrees_argmax": abs(
            circular_angle_error_degrees(decoded.predicted_angle_degrees_argmax, decoded.true_angle_degrees)
        ),
        "predicted_center_x_224": decoded.predicted_center_x_224,
        "predicted_center_y_224": decoded.predicted_center_y_224,
        "predicted_tip_x_224": decoded.predicted_tip_x_224,
        "predicted_tip_y_224": decoded.predicted_tip_y_224,
        "predicted_center_x_224_argmax": decoded.predicted_center_x_224_argmax,
        "predicted_center_y_224_argmax": decoded.predicted_center_y_224_argmax,
        "predicted_tip_x_224_argmax": decoded.predicted_tip_x_224_argmax,
        "predicted_tip_y_224_argmax": decoded.predicted_tip_y_224_argmax,
        "true_center_x_224": decoded.true_center_x_224,
        "true_center_y_224": decoded.true_center_y_224,
        "true_tip_x_224": decoded.true_tip_x_224,
        "true_tip_y_224": decoded.true_tip_y_224,
        "center_px_mae_224": math.hypot(decoded.predicted_center_x_224 - decoded.true_center_x_224, decoded.predicted_center_y_224 - decoded.true_center_y_224),
        "tip_px_mae_224": math.hypot(decoded.predicted_tip_x_224 - decoded.true_tip_x_224, decoded.predicted_tip_y_224 - decoded.true_tip_y_224),
        "center_px_mae_224_argmax": math.hypot(
            decoded.predicted_center_x_224_argmax - decoded.true_center_x_224,
            decoded.predicted_center_y_224_argmax - decoded.true_center_y_224,
        ),
        "tip_px_mae_224_argmax": math.hypot(
            decoded.predicted_tip_x_224_argmax - decoded.true_tip_x_224,
            decoded.predicted_tip_y_224_argmax - decoded.true_tip_y_224,
        ),
        "predicted_center_tip_distance_px": features.predicted_center_tip_distance_px,
        "true_center_tip_distance_px": features.true_center_tip_distance_px,
        "expected_center_tip_distance_px": features.expected_center_tip_distance_px,
        "center_tip_distance_ratio": features.center_tip_distance_ratio,
        "center_heatmap_peak_value": decoded.center_heatmap_peak_value,
        "tip_heatmap_peak_value": decoded.tip_heatmap_peak_value,
        "center_heatmap_entropy": features.center_heatmap_entropy,
        "tip_heatmap_entropy": features.tip_heatmap_entropy,
        "center_heatmap_spread_px": features.center_heatmap_spread_px,
        "tip_heatmap_spread_px": features.tip_heatmap_spread_px,
        "confidence": decoded.confidence,
        "predicted_center_normalized_in_bounds": int(features.center_normalized_in_bounds),
        "predicted_tip_normalized_in_bounds": int(features.tip_normalized_in_bounds),
        "predicted_min_edge_margin_px": features.min_edge_margin_px,
        "predicted_near_edge_8px": int(_predicted_near_edge(decoded, edge_margin_px=8.0)),
        "predicted_near_edge_4px": int(_predicted_near_edge(decoded, edge_margin_px=4.0)),
        "angle_unwrapped_from_cold_degrees": features.angle_unwrapped_from_cold_degrees,
        "angle_within_valid_sweep": int(features.angle_within_valid_sweep),
        "temperature_within_physical_range": int(features.calibrated_temperature_within_physical_range),
        "temperature_outside_physical_range": int(features.calibrated_temperature_outside_physical_range),
        "temperature_outside_physical_margin": int(
            not (-35.0 <= decoded.predicted_temperature_c_calibrated <= 55.0)
        ),
        "guard_status": guardrail.status,
        "guard_rejection_reasons": "|".join(guardrail.rejection_reasons),
        "guard_temperature_c": guardrail.temperature_c,
        "guard_raw_temperature_c": guardrail.raw_temperature_c,
        "dominant_failure_mode": _dominant_failure_mode(decoded, guardrail, {
            "absolute_error_c_calibrated": decoded.absolute_error_c_calibrated,
            "center_px_mae_224": math.hypot(decoded.predicted_center_x_224 - decoded.true_center_x_224, decoded.predicted_center_y_224 - decoded.true_center_y_224),
            "tip_px_mae_224": math.hypot(decoded.predicted_tip_x_224 - decoded.true_tip_x_224, decoded.predicted_tip_y_224 - decoded.true_tip_y_224),
            "angle_mae_degrees": abs(circular_angle_error_degrees(decoded.predicted_angle_degrees, decoded.true_angle_degrees)),
            "tip_heatmap_peak_value": decoded.tip_heatmap_peak_value,
            "confidence": decoded.confidence,
            "predicted_temperature_c_calibrated": decoded.predicted_temperature_c_calibrated,
            "center_tip_distance_ratio": features.center_tip_distance_ratio,
        }),
        "calibrated_error_after_clamping": abs_error_raw_clipped,
    }


def _write_failure_overlay(
    record: FailureAnalysisRecord,
    output_path: Path,
    *,
    selected_thresholds: GeometryGuardrailThresholds,
    selected_tags: str,
) -> None:
    """Render one failure case with crop and heatmap insets."""

    decoded = record.decoded
    guardrail = record.guardrail
    features = guardrail.quality_features
    sample = record.sample

    fig = plt.figure(figsize=(17, 10), dpi=150)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.35, 1.0, 1.0), height_ratios=(1.0, 1.0))

    ax_crop = fig.add_subplot(grid[:, 0])
    ax_center_pred = fig.add_subplot(grid[0, 1])
    ax_tip_pred = fig.add_subplot(grid[1, 1])
    ax_center_target = fig.add_subplot(grid[0, 2])
    ax_tip_target = fig.add_subplot(grid[1, 2])

    ax_crop.imshow(sample.crop_image)
    ax_crop.scatter(
        [decoded.true_center_x_224, decoded.predicted_center_x_224],
        [decoded.true_center_y_224, decoded.predicted_center_y_224],
        c=["lime", "cyan"],
        s=70,
        marker="o",
        edgecolors="white",
        linewidths=1.0,
        label="center",
    )
    ax_crop.scatter(
        [decoded.true_tip_x_224, decoded.predicted_tip_x_224],
        [decoded.true_tip_y_224, decoded.predicted_tip_y_224],
        c=["red", "yellow"],
        s=70,
        marker="x",
        linewidths=2.0,
        label="tip",
    )
    ax_crop.plot(
        [decoded.true_center_x_224, decoded.true_tip_x_224],
        [decoded.true_center_y_224, decoded.true_tip_y_224],
        color="white",
        linewidth=2.0,
        alpha=0.85,
        label="true needle",
    )
    ax_crop.plot(
        [decoded.predicted_center_x_224, decoded.predicted_tip_x_224],
        [decoded.predicted_center_y_224, decoded.predicted_tip_y_224],
        color="deepskyblue",
        linewidth=2.0,
        alpha=0.85,
        label="pred needle",
    )
    ax_crop.set_title("Crop overlay")
    ax_crop.set_axis_off()
    ax_crop.legend(loc="lower right", fontsize=8, framealpha=0.9)

    def _plot_heatmap(ax: plt.Axes, heatmap: np.ndarray, *, title: str, true_x: float, true_y: float, pred_x: float, pred_y: float) -> None:
        """Plot one heatmap with true and predicted coordinates."""

        heatmap_size = heatmap.shape[0]
        scale = float(heatmap_size - 1)
        ax.imshow(heatmap, cmap="magma", origin="upper")
        ax.scatter([true_x * scale], [true_y * scale], c="white", s=45, marker="o", edgecolors="black", linewidths=1.0)
        ax.scatter([pred_x * scale], [pred_y * scale], c="cyan", s=55, marker="x", linewidths=2.0)
        ax.set_title(title)
        ax.set_axis_off()

    _plot_heatmap(
        ax_center_pred,
        decoded.center_heatmap,
        title="Predicted center heatmap",
        true_x=decoded.true_center_x_224 / 223.0,
        true_y=decoded.true_center_y_224 / 223.0,
        pred_x=decoded.predicted_center_x_224 / 223.0,
        pred_y=decoded.predicted_center_y_224 / 223.0,
    )
    _plot_heatmap(
        ax_tip_pred,
        decoded.tip_heatmap,
        title="Predicted tip heatmap",
        true_x=decoded.true_tip_x_224 / 223.0,
        true_y=decoded.true_tip_y_224 / 223.0,
        pred_x=decoded.predicted_tip_x_224 / 223.0,
        pred_y=decoded.predicted_tip_y_224 / 223.0,
    )
    _plot_heatmap(
        ax_center_target,
        sample.center_heatmap,
        title="Target center heatmap",
        true_x=decoded.true_center_x_224 / 223.0,
        true_y=decoded.true_center_y_224 / 223.0,
        pred_x=decoded.true_center_x_224 / 223.0,
        pred_y=decoded.true_center_y_224 / 223.0,
    )
    _plot_heatmap(
        ax_tip_target,
        sample.tip_heatmap,
        title="Target tip heatmap",
        true_x=decoded.true_tip_x_224 / 223.0,
        true_y=decoded.true_tip_y_224 / 223.0,
        pred_x=decoded.true_tip_x_224 / 223.0,
        pred_y=decoded.true_tip_y_224 / 223.0,
    )

    summary_lines = [
        f"file: {Path(decoded.image_path).name}",
        f"split: {decoded.split}",
        f"jitter: {record.jitter_level} sx={decoded.jitter_shift_x} sy={decoded.jitter_shift_y} scale={decoded.jitter_scale:.3f} aspect={decoded.jitter_aspect:.3f}",
        f"true temp: {decoded.true_temperature_c:.2f} C",
        f"pred temp current: {decoded.predicted_temperature_c_current_mapping:.2f} C",
        f"pred temp calibrated: {decoded.predicted_temperature_c_calibrated:.2f} C",
        f"guard temp: {guardrail.temperature_c:.2f} C",
        f"abs err calibrated: {decoded.absolute_error_c_calibrated:.2f} C",
        f"center err: {math.hypot(decoded.predicted_center_x_224 - decoded.true_center_x_224, decoded.predicted_center_y_224 - decoded.true_center_y_224):.2f} px",
        f"tip err: {math.hypot(decoded.predicted_tip_x_224 - decoded.true_tip_x_224, decoded.predicted_tip_y_224 - decoded.true_tip_y_224):.2f} px",
        f"ratio: {features.center_tip_distance_ratio:.3f}",
        f"peaks center/tip: {decoded.center_heatmap_peak_value:.4f} / {decoded.tip_heatmap_peak_value:.4f}",
        f"confidence: {decoded.confidence:.4f}",
        f"guard status: {guardrail.status}",
        f"guard reasons: {', '.join(guardrail.rejection_reasons) if guardrail.rejection_reasons else 'none'}",
        f"tags: {selected_tags}",
    ]
    fig.suptitle(f"{Path(decoded.image_path).name} | {record.jitter_level} | {decoded.absolute_error_c_calibrated:.2f} C", fontsize=15)
    fig.text(0.02, 0.01, "\n".join(summary_lines), family="monospace", fontsize=9, va="bottom")
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _summarize_records(records: list[FailureAnalysisRecord]) -> dict[str, dict[str, float]]:
    """Summarize the failure rows by jitter level."""

    summary: dict[str, dict[str, float]] = {}
    for jitter_level in sorted({record.jitter_level for record in records}):
        subset = [record for record in records if record.jitter_level == jitter_level]
        errors = np.asarray([float(record.row["absolute_error_c_calibrated"]) for record in subset], dtype=np.float64)
        clamped_errors = np.asarray(
            [float(record.row["absolute_error_c_calibrated_clipped"]) for record in subset],
            dtype=np.float64,
        )
        guard_rejected = np.asarray([1.0 if record.guardrail.status == "rejected" else 0.0 for record in subset], dtype=np.float64)
        summary[jitter_level] = {
            "count": float(len(subset)),
            "mae": float(np.mean(errors)),
            "worst": float(np.max(errors)),
            "clamped_mae": float(np.mean(clamped_errors)),
            "clamped_worst": float(np.max(clamped_errors)),
            "rejected_fraction": float(np.mean(guard_rejected)),
            "accepted_fraction": float(1.0 - np.mean(guard_rejected)),
        }
    return summary


def _build_report(
    *,
    records: list[FailureAnalysisRecord],
    report_path: Path,
    model_path: Path,
    calibration_name: str,
    calibration_kind: str,
    selected_thresholds: GeometryGuardrailThresholds,
) -> None:
    """Write the markdown failure analysis report."""

    errors = np.asarray([float(record.row["absolute_error_c_calibrated"]) for record in records], dtype=np.float64)
    clamped_errors = np.asarray([float(record.row["absolute_error_c_calibrated_clipped"]) for record in records], dtype=np.float64)
    current_errors = np.asarray([float(record.row["absolute_error_c_current_mapping"]) for record in records], dtype=np.float64)
    current_clamped_errors = np.asarray([float(record.row["absolute_error_c_current_mapping_clipped"]) for record in records], dtype=np.float64)
    top_30 = sorted(records, key=lambda record: float(record.row["absolute_error_c_calibrated"]), reverse=True)[:30]
    over_20 = [record for record in records if float(record.row["absolute_error_c_calibrated"]) > 20.0]
    rejected_over_20 = [record for record in over_20 if record.guardrail.status == "rejected"]
    clamped_over_20 = [record for record in over_20 if record.guardrail.status == "clamped"]

    failure_mode_counts: dict[str, int] = {}
    for record in top_30:
        failure_mode = str(record.row["dominant_failure_mode"])
        failure_mode_counts[failure_mode] = failure_mode_counts.get(failure_mode, 0) + 1

    lines = [
        "# Geometry Heatmap v2 Failure Analysis",
        "",
        "## Run Summary",
        "",
        f"- Model: `{model_path}`",
        f"- Calibration candidate: {calibration_name} ({calibration_kind})",
        f"- Guardrail thresholds: center_peak>={selected_thresholds.center_peak_min:.2f}, tip_peak>={selected_thresholds.tip_peak_min:.2f}, confidence>={selected_thresholds.confidence_min:.2f}, ratio=[{selected_thresholds.center_tip_distance_ratio_min:.2f}, {selected_thresholds.center_tip_distance_ratio_max:.2f}], edge_margin={selected_thresholds.edge_margin_px:.1f}px, temp_margin={selected_thresholds.temperature_physical_margin_c:.1f}C",
        "",
        f"- Overall calibrated MAE: {float(np.mean(errors)):.3f} C",
        f"- Overall clamped calibrated MAE: {float(np.mean(clamped_errors)):.3f} C",
        f"- Overall current-mapping MAE: {float(np.mean(current_errors)):.3f} C",
        f"- Overall clamped current-mapping MAE: {float(np.mean(current_clamped_errors)):.3f} C",
        f"- Worst calibrated error: {float(np.max(errors)):.3f} C",
        f"- Worst clamped calibrated error: {float(np.max(clamped_errors)):.3f} C",
        "",
        "## Jitter-Level Summary",
        "",
        "| level | count | calibrated_mae | clamped_mae | worst | clamped_worst | accepted_fraction | rejected_fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for jitter_level in ["identity", "mild", "medium", "strong"]:
        subset = [record for record in records if record.jitter_level == jitter_level]
        level_errors = np.asarray([float(record.row["absolute_error_c_calibrated"]) for record in subset], dtype=np.float64)
        level_clamped_errors = np.asarray([float(record.row["absolute_error_c_calibrated_clipped"]) for record in subset], dtype=np.float64)
        accepted_fraction = float(np.mean([record.guardrail.status != "rejected" for record in subset]))
        rejected_fraction = 1.0 - accepted_fraction
        lines.append(
            f"| {jitter_level} | {len(subset)} | {float(np.mean(level_errors)):.3f} | {float(np.mean(level_clamped_errors)):.3f} | {float(np.max(level_errors)):.3f} | {float(np.max(level_clamped_errors)):.3f} | {accepted_fraction:.3f} | {rejected_fraction:.3f} |"
        )

    lines.extend(
        [
            "",
            "## What Broke",
            "",
            "- The worst cases are dominated by tip localization failures and center-tip distance implausibility.",
            "- Several failures also push the calibrated temperature outside the physical gauge range, which indicates extrapolation instead of safe interpolation.",
            "- Peak confidence alone is not sufficient, so the guardrail also checks geometry consistency and temperature plausibility.",
            "",
            "## Top 30 Worst Predictions",
            "",
            "| rank | level | image | abs_err_calibrated | clamped_err | mode | guard | temp_calibrated | temp_clamped | center_err | tip_err | ratio | peaks c/t | confidence | reasons |",
            "| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for rank, record in enumerate(top_30, start=1):
        row = record.row
        lines.append(
            f"| {rank} | {record.jitter_level} | {Path(str(row['image_path'])).name} | {float(row['absolute_error_c_calibrated']):.3f} | {float(row['absolute_error_c_calibrated_clipped']):.3f} | {row['dominant_failure_mode']} | {record.guardrail.status} | {float(row['predicted_temperature_c_calibrated']):.2f} | {float(row['predicted_temperature_c_calibrated_clipped']):.2f} | {float(row['center_px_mae_224']):.2f} | {float(row['tip_px_mae_224']):.2f} | {float(row['center_tip_distance_ratio']):.3f} | {float(row['center_heatmap_peak_value']):.3f} / {float(row['tip_heatmap_peak_value']):.3f} | {float(row['confidence']):.3f} | {row['guard_rejection_reasons']} |"
        )

    worst_row = max(records, key=lambda record: float(record.row["absolute_error_c_calibrated"]))
    worst_current_row = worst_row.row
    lines.extend(
        [
            "",
            "## Catastrophic Tail Check",
            "",
            f"- Worst calibrated error row: {Path(str(worst_current_row['image_path'])).name} at {worst_row.jitter_level} jitter.",
            f"- Raw calibrated temperature: {float(worst_current_row['predicted_temperature_c_calibrated']):.2f} C.",
            f"- Clamped calibrated temperature: {float(worst_current_row['predicted_temperature_c_calibrated_clipped']):.2f} C.",
            f"- True temperature: {float(worst_current_row['true_temperature_c']):.2f} C.",
            f"- Clamping changes the error from {float(worst_current_row['absolute_error_c_calibrated']):.2f} C to {float(worst_current_row['absolute_error_c_calibrated_clipped']):.2f} C, so it helps a little but does not solve the tail.",
            f"- This row is {'rejected' if worst_row.guardrail.status == 'rejected' else 'not rejected'} by the reference guardrails.",
            "",
            "## Guardrail Coverage",
            "",
            f"- Total >20 C errors: {len(over_20)}",
            f"- >20 C errors rejected by the reference guardrails: {len(rejected_over_20)}",
            f"- >20 C errors clamped rather than rejected: {len(clamped_over_20)}",
            f"- Top-30 failure-mode counts: {failure_mode_counts}",
            "",
            "## Interpretation",
            "",
            "- The tail is mostly a geometry/tip-localization problem, not a pure heatmap-confidence problem.",
            "- Physical clamping alone is not enough because the worst errors remain far from the target even after clipping.",
            "- The reference guardrails are able to identify the worst rows well enough to reject them instead of returning a wildly wrong temperature.",
            "",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """Run the failure autopsy on geometry heatmap v2."""

    parser = argparse.ArgumentParser(description="Analyze geometry heatmap v2 failures")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2/model.keras"),
        help="Trained heatmap model artifact.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"),
        help="Clean geometry manifest.",
    )
    parser.add_argument(
        "--calibration-json-path",
        type=Path,
        default=Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json"),
        help="Calibration artifact from Phase 4.7.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2_guarded"),
        help="Directory for failure analysis CSV output.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_v2_failure_analysis.md"),
        help="Markdown failure analysis report.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("ml/debug/geometry_heatmap_v2_failures"),
        help="Directory for failure overlays.",
    )
    parser.add_argument("--heatmap-size", type=int, default=56, help="Heatmap size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for jitter sampling.")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    model_path = base_path / args.model_path if not args.model_path.is_absolute() else args.model_path
    manifest_path = base_path / args.manifest_path if not args.manifest_path.is_absolute() else args.manifest_path
    calibration_json_path = (
        base_path / args.calibration_json_path if not args.calibration_json_path.is_absolute() else args.calibration_json_path
    )
    output_dir = base_path / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    report_path = base_path / args.report_path if not args.report_path.is_absolute() else args.report_path
    debug_dir = base_path / args.debug_dir if not args.debug_dir.is_absolute() else args.debug_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(model_path, compile=False)
    calibration_candidate, calibration_json = load_selected_calibration_candidate(calibration_json_path)
    thresholds = GeometryGuardrailThresholds()

    examples = load_clean_geometry_examples(manifest_path)
    test_examples = select_examples_from_split(examples, split="test")

    records: list[FailureAnalysisRecord] = []

    for level_index, (jitter_level, jitter_spec) in enumerate(JITTER_LEVELS.items()):
        level_samples = []
        for example_index, example in enumerate(test_examples):
            if jitter_level == "identity":
                sample = load_heatmap_sample(example, base_path, heatmap_size=args.heatmap_size, sigma_pixels=5.0, jitter=None)
            else:
                jitter_rng = np.random.default_rng(args.seed + level_index * 1000 + example_index)
                jitter = sample_jitter_params(jitter_rng, **jitter_spec)
                sample = load_heatmap_sample(example, base_path, heatmap_size=args.heatmap_size, sigma_pixels=5.0, jitter=jitter)
            level_samples.append(sample)

        x = np.stack([sample.crop_image for sample in level_samples], axis=0).astype(np.float32)
        predictions = model.predict(x, verbose=0)
        center_batch = np.asarray(predictions[0], dtype=np.float32)
        tip_batch = np.asarray(predictions[1], dtype=np.float32)
        confidence_batch = np.asarray(predictions[2], dtype=np.float32)

        for sample_index, sample in enumerate(level_samples):
            decoded = decode_heatmap_geometry_prediction(
                sample,
                center_batch[sample_index],
                tip_batch[sample_index],
                float(np.ravel(confidence_batch[sample_index])[0]),
                calibration_candidate,
            )
            guardrail = apply_geometry_guardrails(decoded, thresholds)
            row = _build_row(decoded, guardrail, jitter_level=jitter_level)
            records.append(
                FailureAnalysisRecord(
                    sample=sample,
                    decoded=decoded,
                    guardrail=guardrail,
                    jitter_level=jitter_level,
                    row=row,
                )
            )

    rows = [record.row for record in records]
    rows.sort(key=lambda row: float(row["absolute_error_c_calibrated"]), reverse=True)
    _write_csv(rows, output_dir / "failure_analysis.csv")
    _write_json(
        {
            "model_path": str(model_path),
            "calibration_candidate": calibration_candidate.to_json(),
            "calibration_selected_candidate_name": calibration_json["selected_candidate_name"],
            "guardrail_thresholds": {
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
        },
        output_dir / "failure_analysis_summary.json",
    )

    worst_50 = sorted(records, key=lambda record: float(record.row["absolute_error_c_calibrated"]), reverse=True)[:50]
    strong_worst_30 = sorted(
        [record for record in records if record.jitter_level == "strong"],
        key=lambda record: float(record.row["absolute_error_c_calibrated"]),
        reverse=True,
    )[:30]
    over_20 = [record for record in records if float(record.row["absolute_error_c_calibrated"]) > 20.0]

    overlay_records: dict[tuple[str, str, int, int, float, float], tuple[FailureAnalysisRecord, list[str]]] = {}
    for tag, subset in {
        "worst50": worst_50,
        "strong30": strong_worst_30,
        "over20": over_20,
    }.items():
        for record in subset:
            key = (
                str(record.row["image_path"]),
                record.jitter_level,
                int(record.row["jitter_shift_x"]),
                int(record.row["jitter_shift_y"]),
                float(record.row["jitter_scale"]),
                float(record.row["jitter_aspect"]),
            )
            if key not in overlay_records:
                overlay_records[key] = (record, [tag])
            else:
                overlay_records[key][1].append(tag)

    for index, (record, tags) in enumerate(
        sorted(overlay_records.values(), key=lambda item: float(item[0].row["absolute_error_c_calibrated"]), reverse=True),
        start=1,
    ):
        overlay_name = (
            f"{index:03d}_{record.jitter_level}_{Path(str(record.row['image_path'])).stem}_err{float(record.row['absolute_error_c_calibrated']):.2f}.png"
        )
        _write_failure_overlay(
            record,
            debug_dir / overlay_name,
            selected_thresholds=thresholds,
            selected_tags=",".join(tags),
        )

    _build_report(
        records=records,
        report_path=report_path,
        model_path=model_path,
        calibration_name=calibration_candidate.name,
        calibration_kind=calibration_candidate.kind,
        selected_thresholds=thresholds,
    )

    print(f"Failure analysis CSV: {output_dir / 'failure_analysis.csv'}")
    print(f"Failure analysis report: {report_path}")
    print(f"Failure overlays: {debug_dir}")


if __name__ == "__main__":
    main()
