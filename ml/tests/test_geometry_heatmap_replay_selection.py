"""Tests for replay candidate selection in geometry heatmap training."""

from __future__ import annotations

import math

from embedded_gauge_reading_tinyml.geometry_heatmap_replay_selection import (
    build_replay_candidate,
    preferred_metric_value,
    select_best_replay_candidate,
)


def test_select_best_replay_candidate_prefers_relaxed_candidate_when_strict_has_no_accepts() -> None:
    """A shadow replay candidate should beat strict nan metrics early in training."""

    strict = build_replay_candidate(
        "strict",
        {
            "accepted_count": 0.0,
            "accepted_mae_c": math.nan,
            "raw_mae_c": 12.0,
            "worst_accepted_error_c": math.nan,
            "raw_worst_error_c": 18.0,
            "acceptance_rate": 0.0,
            "accepted_gt20_failures": 0.0,
            "temperature_delta_mean": math.nan,
            "temperature_delta_mean_all": 11.0,
            "tip_delta_mean": 9.0,
        },
    )
    shadow = build_replay_candidate(
        "shadow_spread_45",
        {
            "accepted_count": 3.0,
            "accepted_mae_c": 3.25,
            "raw_mae_c": 8.5,
            "worst_accepted_error_c": 5.1,
            "raw_worst_error_c": 14.0,
            "acceptance_rate": 0.0345,
            "accepted_gt20_failures": 0.0,
            "temperature_delta_mean": 0.9,
            "temperature_delta_mean_all": 0.9,
            "tip_delta_mean": 8.1,
        },
    )

    best = select_best_replay_candidate([strict, shadow])

    assert best.name == "shadow_spread_45"
    assert preferred_metric_value(best.metrics, "accepted_mae_c", "raw_mae_c") == 3.25


def test_select_best_replay_candidate_falls_back_to_raw_mae_when_every_candidate_rejects() -> None:
    """When nothing gets accepted, raw MAE should still rank the candidates."""

    candidate_a = build_replay_candidate(
        "candidate_a",
        {
            "accepted_count": 0.0,
            "accepted_mae_c": math.nan,
            "raw_mae_c": 9.0,
            "worst_accepted_error_c": math.nan,
            "raw_worst_error_c": 16.0,
            "acceptance_rate": 0.0,
            "accepted_gt20_failures": 0.0,
            "temperature_delta_mean": math.nan,
            "temperature_delta_mean_all": 1.8,
            "tip_delta_mean": 10.0,
        },
    )
    candidate_b = build_replay_candidate(
        "candidate_b",
        {
            "accepted_count": 0.0,
            "accepted_mae_c": math.nan,
            "raw_mae_c": 6.0,
            "worst_accepted_error_c": math.nan,
            "raw_worst_error_c": 14.0,
            "acceptance_rate": 0.0,
            "accepted_gt20_failures": 0.0,
            "temperature_delta_mean": math.nan,
            "temperature_delta_mean_all": 1.4,
            "tip_delta_mean": 10.0,
        },
    )

    best = select_best_replay_candidate([candidate_a, candidate_b])

    assert best.name == "candidate_b"
    assert preferred_metric_value(best.metrics, "accepted_mae_c", "raw_mae_c") == 6.0
