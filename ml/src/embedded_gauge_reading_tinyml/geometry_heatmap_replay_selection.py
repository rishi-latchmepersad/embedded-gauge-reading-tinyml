"""Selection helpers for geometry heatmap replay metrics.

The training loop evaluates the same checkpoint under a strict guardrail set
and several relaxed "shadow" variants.  This module keeps the checkpoint
ranking policy small, deterministic, and unit-testable so the trainer can save
the best useful replay candidate even when the strict guardrails are still too
harsh early in training.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ReplayCandidateMetrics:
    """One replay candidate plus its derived ranking score."""

    name: str
    metrics: dict[str, float]
    score: tuple[float, float, float, float, float, float, float, float]


def _numeric_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    """Keep only numeric metrics and coerce them to plain floats."""

    numeric: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, Real) and not isinstance(value, bool):
            numeric[key] = float(value)
    return numeric


def preferred_metric_value(metrics: Mapping[str, float], primary_key: str, fallback_key: str) -> float:
    """Return a metric, falling back when the preferred value is not finite."""

    primary = float(metrics.get(primary_key, math.nan))
    if math.isfinite(primary):
        return primary
    return float(metrics.get(fallback_key, math.nan))


def score_replay_candidate(metrics: Mapping[str, float]) -> tuple[float, float, float, float, float, float, float, float]:
    """Rank one replay candidate from most useful to least useful.

    Lower tuples are better.  The first terms prefer any candidate that accepts
    at least one sample, then prefer candidates that avoid large failures and
    improve the raw geometry quality.  The tail terms break ties using
    temperature and tip drift.
    """

    accepted_count = float(metrics.get("accepted_count", 0.0))
    accepted_mae_c = preferred_metric_value(metrics, "accepted_mae_c", "raw_mae_c")
    worst_error_c = preferred_metric_value(metrics, "worst_accepted_error_c", "raw_worst_error_c")
    acceptance_rate = float(metrics.get("acceptance_rate", math.inf))
    temperature_delta_mean = preferred_metric_value(
        metrics,
        "temperature_delta_mean",
        "temperature_delta_mean_all",
    )
    tip_delta_mean = float(metrics.get("tip_delta_mean", math.inf))

    if not math.isfinite(accepted_mae_c):
        accepted_mae_c = math.inf
    if not math.isfinite(worst_error_c):
        worst_error_c = math.inf
    if not math.isfinite(temperature_delta_mean):
        temperature_delta_mean = math.inf
    if not math.isfinite(tip_delta_mean):
        tip_delta_mean = math.inf

    return (
        0.0 if accepted_count > 0.0 else 1.0,
        0.0 if float(metrics.get("accepted_gt20_failures", math.inf)) <= 0.0 else 1.0,
        0.0 if worst_error_c < 20.0 else 1.0,
        0.0 if acceptance_rate >= 0.65 else 1.0,
        0.0 if accepted_mae_c <= 4.5 else 1.0,
        accepted_mae_c,
        temperature_delta_mean,
        tip_delta_mean,
    )


def build_replay_candidate(name: str, metrics: Mapping[str, object]) -> ReplayCandidateMetrics:
    """Build a ranked replay candidate from a mixed metrics dictionary."""

    numeric_metrics = _numeric_metrics(metrics)
    return ReplayCandidateMetrics(
        name=name,
        metrics=numeric_metrics,
        score=score_replay_candidate(numeric_metrics),
    )


def select_best_replay_candidate(candidates: Sequence[ReplayCandidateMetrics]) -> ReplayCandidateMetrics:
    """Return the candidate with the smallest ranking score."""

    if not candidates:
        raise ValueError("Cannot select from an empty replay candidate list.")
    return min(candidates, key=lambda candidate: candidate.score)
