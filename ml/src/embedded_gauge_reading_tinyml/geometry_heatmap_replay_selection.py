"""Helpers for choosing the best geometry replay candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ReplayCandidateMetrics:
    """A named candidate plus its summary metrics."""

    name: str
    metrics: dict[str, Any]


def build_replay_candidate(name: str, metrics: dict[str, Any]) -> ReplayCandidateMetrics:
    """Create a simple metrics bundle for replay selection."""

    return ReplayCandidateMetrics(name=name, metrics=dict(metrics))


def preferred_metric_value(metrics: dict[str, Any], primary_key: str, fallback_key: str) -> float:
    """Return the primary metric if present, otherwise the fallback metric."""

    value = metrics.get(primary_key, metrics.get(fallback_key, float("nan")))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def select_best_replay_candidate(candidates: list[ReplayCandidateMetrics]) -> ReplayCandidateMetrics:
    """Pick the candidate with the lowest accepted MAE, then lowest worst error."""

    if not candidates:
        raise ValueError("No replay candidates were provided.")

    def _score(candidate: ReplayCandidateMetrics) -> tuple[float, float, str]:
        metrics = candidate.metrics
        accepted = preferred_metric_value(metrics, "accepted_mae_c", "raw_mae_c")
        worst = preferred_metric_value(metrics, "worst_accepted_error_c", "raw_worst_error_c")
        return (accepted, worst, candidate.name)

    return min(candidates, key=_score)

