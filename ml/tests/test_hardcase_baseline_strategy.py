"""Regression checks for the hard-case classical-baseline strategy sweep."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
"""Repository root used to read the committed hard-case sweep artifact."""

SWEEP_JSON = (
    REPO_ROOT
    / "ml"
    / "artifacts"
    / "baseline"
    / "hardcase_strategy_sweep"
    / "strategy_sweep.json"
)
"""Committed strategy sweep summary generated from the hard-case manifests."""


def test_hardcase_sweep_keeps_hough_only_better_than_center_fallback() -> None:
    """The hard-case sweep should keep favoring the pure Hough geometry path."""
    payload = json.loads(SWEEP_JSON.read_text(encoding="utf-8"))
    per_strategy = {
        entry["strategy_name"]: entry
        for entry in payload["per_strategy"]
    }

    hough_only_mae = float(per_strategy["hough_only"]["combined_mae"])
    fallback_mae = float(per_strategy["hough_then_center_t4"]["combined_mae"])

    assert hough_only_mae < fallback_mae
    assert int(per_strategy["hough_only"]["combined_failed_samples"]) == 3
