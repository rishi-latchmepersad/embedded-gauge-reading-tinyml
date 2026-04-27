"""Regression tests for the classical polar baseline in firmware."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
BASELINE_RUNTIME_FILE: Path = (
    REPO_ROOT / "firmware" / "stm32" / "n657" / "Appli" / "Src" / "app_baseline_runtime.c"
)
_FLOAT_RE: str = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?f?"


def _extract_float_constant(name: str, text: str) -> float:
    """Read one named float constant from the firmware source."""
    pattern = rf"{re.escape(name)}\s+({_FLOAT_RE})"
    match = re.search(pattern, text)
    if match is None:
        raise AssertionError(f"Could not find {name} in {BASELINE_RUNTIME_FILE}.")
    return float(match.group(1).rstrip("f"))


def test_baseline_confidence_gate_matches_snr_scale() -> None:
    """The firmware confidence gate should track the polar detector's SNR."""
    text = BASELINE_RUNTIME_FILE.read_text(encoding="utf-8")
    threshold = _extract_float_constant("APP_BASELINE_CONFIDENCE_THRESHOLD", text)

    # The live board needs a slightly more permissive gate so real captures can
    # seed history without waiting for perfect SNR.
    assert 1.0 < threshold < 2.0
    assert "baseline-polar-warming" in text
    assert "baseline-polar-smoothed" in text
    assert "baseline-polar-held" in text


def test_baseline_uses_polar_vote_histogram() -> None:
    """The firmware should vote in polar space, not along a single ray."""
    text = BASELINE_RUNTIME_FILE.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", text)

    assert "AppBaselineRuntime_EstimatePolarNeedle" in text
    assert "AppBaselineRuntime_EstimateDialRadiusPixels" in text
    assert "AppBaselineRuntime_EstimateDialCenterFromRimVotes" in text
    assert "APP_BASELINE_DIAL_RADIUS_FROM_CROP_HEIGHT_RATIO" in text
    assert "center_prior" in normalized
    assert "rim_weight = rim_bias * rim_bias" in normalized
    assert "alignment_weight = radial_alignment * radial_alignment" in normalized
    assert "angle_votes[APP_BASELINE_ANGLE_BINS]" in normalized
    assert "smoothed_votes[APP_BASELINE_ANGLE_BINS]" in normalized
    assert "tangential_weight" in normalized
    assert "vote = edge_mag * tangential_weight" in normalized
    assert "edge_mag <= 8.0f" in normalized
    assert "gradient_x_out, float *gradient_y_out" in normalized
    assert "AppBaselineRuntime_ReadEdgeMagnitude(" in text
    assert "training-crop-hough" not in text
    assert "baseline-hough" not in text


def test_baseline_selection_prefers_sharper_peak_for_every_candidate() -> None:
    """All classical candidates should compete on a blended sharpness/support score."""
    text = BASELINE_RUNTIME_FILE.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", text)

    assert "AppBaselineRuntime_IsBetterEstimate" in normalized
    assert "AppBaselineRuntime_ComputeEstimateQuality" in normalized
    assert "blended quality score" in text
    assert "runner_up_score <= 0.0f" in normalized
    assert "selected_is_fixed_crop" not in normalized
    assert "estimate_out->confidence < APP_BASELINE_CONFIDENCE_THRESHOLD" in normalized
    assert "estimate_out->best_score < APP_BASELINE_MIN_ACCEPT_SCORE" in normalized
    assert "((estimate_out->runner_up_score > 0.0f) && ((estimate_out->best_score / estimate_out->runner_up_score) < APP_BASELINE_MIN_PEAK_RATIO))" in normalized
    assert "APP_BASELINE_MIN_ACCEPT_SCORE" in text
    assert "0.15f * estimate->confidence" in normalized
    assert "dial_radius_px * 0.75f" in normalized
    assert "rim_geometry_hypothesis" in normalized
    assert "rim-center-polar" in text


def test_baseline_history_rejects_weak_near_ties() -> None:
    """Weak near-tie reads should not be allowed to overwrite the history."""
    text = BASELINE_RUNTIME_FILE.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", text)

    assert "AppBaselineRuntime_IsStableEstimateForHistory" in normalized
    assert "Holding last stable estimate after an unstable frame." in text
    assert "selected_is_fixed_crop" not in normalized
    assert "estimate->best_score < APP_BASELINE_MIN_ACCEPT_SCORE" in normalized
    assert "estimate->runner_up_score > 0.0f" in normalized
    assert "estimate->confidence < APP_BASELINE_CONFIDENCE_THRESHOLD" in normalized


def test_baseline_candidate_selection_uses_dominant_geometry() -> None:
    """The selector should compare refined candidates using peak sharpness."""
    text = BASELINE_RUNTIME_FILE.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", text)

    assert "fixed_crop_ok = AppBaselineRuntime_EstimateFromTrainingCropHypothesis" in normalized
    assert "AppBaselineRuntime_ComputeEstimateQuality" in normalized
    assert "AppBaselineRuntime_IsBetterEstimate" in normalized
    assert "peak separation" in text
    assert "bright_hypothesis" in normalized
    assert "fixed_crop_hypothesis" in normalized
    assert "center_hypothesis" in normalized
    assert "AppBaselineRuntime_RefineEstimateAroundSeed" in normalized
    assert "APP_BASELINE_GEOMETRY_SEARCH_RADIUS_PIXELS" in text
    assert "offset_values[] = { -8L, -4L, 0L, 4L, 8L }" in normalized
    assert "APP_BASELINE_MIN_PEAK_RATIO" in text
