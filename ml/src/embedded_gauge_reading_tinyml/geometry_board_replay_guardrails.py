"""Pure guardrail helpers for board-replay geometry threshold sweeps.

This module evaluates saved board-replay prediction rows against a candidate
threshold set without rerunning the model.  The same helpers are reused by the
threshold micro-sweep and the selected-threshold test evaluation so both stages
apply identical logic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import math
import numpy as np

from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import (
    GeometryGuardrailThresholds,
)


@dataclass(frozen=True)
class BoardReplayGuardrailDecision:
    """The result of applying one threshold set to one replay row."""

    status: str
    temperature_c: float
    raw_temperature_c: float
    rejection_reasons: tuple[str, ...]


def build_board_replay_thresholds(
    *,
    tip_peak_min: float,
    max_heatmap_entropy: float,
    max_heatmap_spread_px: float,
    center_tip_distance_ratio_min: float,
    center_tip_distance_ratio_max: float,
    center_peak_min: float = 0.40,
    confidence_min: float = 0.40,
    edge_margin_px: float = 4.0,
    temperature_physical_margin_c: float = 2.0,
) -> GeometryGuardrailThresholds:
    """Build one board replay guardrail threshold set."""

    return GeometryGuardrailThresholds(
        center_peak_min=center_peak_min,
        tip_peak_min=tip_peak_min,
        confidence_min=confidence_min,
        max_heatmap_entropy=max_heatmap_entropy,
        max_heatmap_spread_px=max_heatmap_spread_px,
        center_tip_distance_ratio_min=center_tip_distance_ratio_min,
        center_tip_distance_ratio_max=center_tip_distance_ratio_max,
        edge_margin_px=edge_margin_px,
        temperature_physical_margin_c=temperature_physical_margin_c,
        clamp_temperature_to_physical_range=True,
    )


def candidate_relaxation_key(thresholds: GeometryGuardrailThresholds) -> tuple[float, float, float, float, float]:
    """Return a strictness key where smaller means less relaxed."""

    return (
        -float(thresholds.tip_peak_min),
        float(thresholds.max_heatmap_entropy),
        float(thresholds.max_heatmap_spread_px),
        -float(thresholds.center_tip_distance_ratio_min),
        float(thresholds.center_tip_distance_ratio_max),
    )


def board_replay_candidate_grid() -> list[GeometryGuardrailThresholds]:
    """Return the micro-sweep candidate grid in strictness order."""

    candidates = [
        build_board_replay_thresholds(
            tip_peak_min=tip_peak_min,
            max_heatmap_entropy=max_heatmap_entropy,
            max_heatmap_spread_px=max_heatmap_spread_px,
            center_tip_distance_ratio_min=center_tip_distance_ratio_min,
            center_tip_distance_ratio_max=center_tip_distance_ratio_max,
        )
        for tip_peak_min in (0.30, 0.35, 0.40)
        for max_heatmap_entropy in (1.00, 1.10, 1.20)
        for max_heatmap_spread_px in (25.0, 30.0, 35.0)
        for center_tip_distance_ratio_min in (0.35, 0.40)
        for center_tip_distance_ratio_max in (1.40, 1.50)
    ]
    return sorted(candidates, key=candidate_relaxation_key)


def _parse_bool(value: Any) -> bool:
    """Parse a CSV boolean field robustly."""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _parse_float(value: Any) -> float:
    """Parse a CSV float field robustly."""

    return float(value)


def _is_finite(value: float) -> bool:
    """Return whether one float is finite."""

    return math.isfinite(value)


def evaluate_board_replay_row(
    row: Mapping[str, Any],
    thresholds: GeometryGuardrailThresholds,
) -> BoardReplayGuardrailDecision:
    """Apply guardrails to one board replay prediction row."""

    true_temperature_c = _parse_float(row["true_temperature_c"])
    predicted_temperature_c = _parse_float(row["predicted_temperature_c_calibrated"])
    center_peak = _parse_float(row["center_heatmap_peak_value"])
    tip_peak = _parse_float(row["tip_heatmap_peak_value"])
    confidence = _parse_float(row["confidence"])
    center_entropy = _parse_float(row["center_heatmap_entropy"])
    tip_entropy = _parse_float(row["tip_heatmap_entropy"])
    center_spread = _parse_float(row["center_heatmap_spread_px"])
    tip_spread = _parse_float(row["tip_heatmap_spread_px"])
    min_edge_margin_px = _parse_float(row["min_edge_margin_px"])
    center_tip_distance_ratio = _parse_float(row["center_tip_distance_ratio"])
    angle_within_valid_sweep = _parse_bool(row["angle_within_valid_sweep"])
    center_normalized_in_bounds = _parse_bool(row["center_normalized_in_bounds"])
    tip_normalized_in_bounds = _parse_bool(row["tip_normalized_in_bounds"])

    reasons: list[str] = []
    if not center_normalized_in_bounds:
        reasons.append("center_normalized_out_of_bounds")
    if not tip_normalized_in_bounds:
        reasons.append("tip_normalized_out_of_bounds")
    if min_edge_margin_px < thresholds.edge_margin_px:
        reasons.append("predicted_point_near_edge")
    if center_peak < thresholds.center_peak_min:
        reasons.append("center_peak_too_low")
    if tip_peak < thresholds.tip_peak_min:
        reasons.append("tip_peak_too_low")
    if confidence < thresholds.confidence_min:
        reasons.append("confidence_too_low")
    if center_entropy > thresholds.max_heatmap_entropy:
        reasons.append("center_heatmap_too_diffuse")
    if tip_entropy > thresholds.max_heatmap_entropy:
        reasons.append("tip_heatmap_too_diffuse")
    if center_spread > thresholds.max_heatmap_spread_px:
        reasons.append("center_heatmap_too_spread_out")
    if tip_spread > thresholds.max_heatmap_spread_px:
        reasons.append("tip_heatmap_too_spread_out")
    if not angle_within_valid_sweep:
        reasons.append("predicted_angle_outside_valid_sweep")
    if not (
        thresholds.minimum_celsius - thresholds.temperature_physical_margin_c
        <= predicted_temperature_c
        <= thresholds.maximum_celsius + thresholds.temperature_physical_margin_c
    ):
        reasons.append("temperature_outside_physical_margin")
    if not (
        thresholds.center_tip_distance_ratio_min
        <= center_tip_distance_ratio
        <= thresholds.center_tip_distance_ratio_max
    ):
        reasons.append("center_tip_distance_ratio_implausible")

    if reasons:
        return BoardReplayGuardrailDecision(
            status="rejected",
            temperature_c=math.nan,
            raw_temperature_c=predicted_temperature_c,
            rejection_reasons=tuple(reasons),
        )

    if predicted_temperature_c < thresholds.minimum_celsius or predicted_temperature_c > thresholds.maximum_celsius:
        clamped_temperature_c = float(
            np.clip(predicted_temperature_c, thresholds.minimum_celsius, thresholds.maximum_celsius)
        )
        return BoardReplayGuardrailDecision(
            status="clamped",
            temperature_c=clamped_temperature_c,
            raw_temperature_c=predicted_temperature_c,
            rejection_reasons=("temperature_clamped_to_physical_range",),
        )

    return BoardReplayGuardrailDecision(
        status="accepted",
        temperature_c=predicted_temperature_c,
        raw_temperature_c=predicted_temperature_c,
        rejection_reasons=(),
    )


def summarize_board_replay_rows(
    rows: Sequence[Mapping[str, Any]],
    thresholds: GeometryGuardrailThresholds,
) -> dict[str, Any]:
    """Summarize a set of saved replay rows under one threshold set."""

    decisions = [evaluate_board_replay_row(row, thresholds) for row in rows]
    accepted_rows = [
        (row, decision)
        for row, decision in zip(rows, decisions, strict=True)
        if decision.status in {"accepted", "clamped"}
    ]
    rejected_rows = [
        (row, decision)
        for row, decision in zip(rows, decisions, strict=True)
        if decision.status == "rejected"
    ]
    if not accepted_rows:
        return {
            "count": int(len(rows)),
            "accepted_count": 0,
            "clamped_count": 0,
            "rejected_count": int(len(rows)),
            "acceptance_rate": 0.0,
            "accepted_mae_c": math.nan,
            "accepted_rmse_c": math.nan,
            "accepted_worst_error_c": math.nan,
            "percentage_under_2c": 0.0,
            "percentage_under_5c": 0.0,
            "percentage_under_10c": 0.0,
            "accepted_gt20c_failures": 0,
            "top_rejection_reasons": [],
        }

    accepted_errors = np.asarray(
        [
            abs(float(decision.temperature_c) - _parse_float(row["true_temperature_c"]))
            for row, decision in accepted_rows
        ],
        dtype=np.float64,
    )
    rejection_reasons = Counter(
        reason
        for _, decision in rejected_rows
        for reason in decision.rejection_reasons
        if reason
    )

    return {
        "count": int(len(rows)),
        "accepted_count": int(sum(1 for _, decision in accepted_rows if decision.status == "accepted")),
        "clamped_count": int(sum(1 for _, decision in accepted_rows if decision.status == "clamped")),
        "rejected_count": int(len(rejected_rows)),
        "acceptance_rate": float(len(accepted_rows) / len(rows)),
        "accepted_mae_c": float(np.mean(accepted_errors)),
        "accepted_rmse_c": float(np.sqrt(np.mean(np.square(accepted_errors)))),
        "accepted_worst_error_c": float(np.max(accepted_errors)),
        "percentage_under_2c": float(np.mean(accepted_errors < 2.0) * 100.0),
        "percentage_under_5c": float(np.mean(accepted_errors < 5.0) * 100.0),
        "percentage_under_10c": float(np.mean(accepted_errors < 10.0) * 100.0),
        "accepted_gt20c_failures": int(np.sum(accepted_errors > 20.0)),
        "top_rejection_reasons": rejection_reasons.most_common(5),
    }
