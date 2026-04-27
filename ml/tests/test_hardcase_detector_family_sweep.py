"""Regression checks for the hard-case detector-family benchmark."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
"""Repository root used to read the committed benchmark artifact."""

BENCHMARK_JSON = (
    REPO_ROOT
    / "ml"
    / "artifacts"
    / "baseline"
    / "detector_family_sweep"
    / "benchmark.json"
)
"""Committed summary for the detector-family hard-case sweep."""


def test_gradient_polar_remains_the_best_hard_case_classical_family() -> None:
    """The hard-case sweep should keep the gradient-polar detector on top."""
    payload = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    per_detector = {
        entry["name"]: entry
        for entry in payload["detectors"]
    }

    gradient_mae = float(per_detector["gradient_polar"]["mae"])
    ray_score_mae = float(per_detector["ray_score"]["mae"])
    hough_lines_mae = float(per_detector["hough_lines"]["mae"])
    dark_polar_successful = int(per_detector["dark_polar"]["successful"])

    assert gradient_mae < ray_score_mae
    assert gradient_mae < hough_lines_mae
    assert dark_polar_successful == 0
    assert int(per_detector["gradient_polar"]["successful"]) == 28
