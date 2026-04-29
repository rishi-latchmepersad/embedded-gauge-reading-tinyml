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
    assert "APP_BASELINE_BORDERLINE_PEAK_RATIO" in text
    assert "APP_BASELINE_BORDERLINE_MIN_CONFIDENCE" in text
    assert "APP_BASELINE_BORDERLINE_MAX_TEMP_DELTA_C" in text


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
    assert "const float tangential =" in normalized
    assert "vote = edge_mag * fabsf(tangential)" in normalized
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
    assert "AppBaselineRuntime_PassesAcceptanceGate" in normalized
    assert "AppBaselineRuntime_GeometryPriority" not in normalized
    assert "peak_excess = peak_ratio - 1.0f" in normalized
    assert "confidence * peak_excess" in normalized
    assert "peak-separation quality score" in text
    assert "runner_up_score > 0.0f" in normalized
    assert "selected_is_fixed_crop" not in normalized
    assert "APP_BASELINE_MIN_ACCEPT_SCORE" in text
    assert "dial_radius_px * 0.75f" in normalized
    assert "rim_geometry_hypothesis" in normalized
    assert "rim-center-polar" in text
    assert "candidate_quality > incumbent_quality" in normalized
    assert "candidate_peak_ratio > incumbent_peak_ratio" in normalized


def test_baseline_history_rejects_weak_near_ties() -> None:
    """Weak near-tie reads should not be allowed to overwrite the history."""
    text = BASELINE_RUNTIME_FILE.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", text)

    assert "AppBaselineRuntime_IsStableEstimateForHistory" in normalized
    assert "AppBaselineRuntime_HasAcceptablePeakSeparation" in normalized
    assert "AppBaselineRuntime_IsBorderlineContinuityEstimate" in normalized
    assert "Holding last stable estimate after an unstable frame." in text
    assert "selected_is_fixed_crop" not in normalized
    assert "estimate->best_score < APP_BASELINE_MIN_ACCEPT_SCORE" in normalized
    assert "estimate->runner_up_score > 0.0f" in normalized
    assert "estimate->confidence < APP_BASELINE_CONFIDENCE_THRESHOLD" in normalized


def test_baseline_candidate_selection_defaults_to_fixed_crop_then_center() -> None:
    """The firmware selector should stay conservative by default."""
    text = BASELINE_RUNTIME_FILE.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", text)

    assert "APP_BASELINE_ENABLE_LOCAL_GEOMETRY_SWEEP" in text
    assert "APP_BASELINE_ENABLE_LOCAL_GEOMETRY_SWEEP 0U" in text
    assert "Keep the live selector conservative by default" in text
    assert "#if APP_BASELINE_ENABLE_LOCAL_GEOMETRY_SWEEP" in text
    assert "#else" in text
    assert "selected_estimate = &fixed_crop_hypothesis;" in normalized
    assert "else if (center_ok)" in normalized
    assert "selected_estimate = &center_hypothesis;" in normalized
    assert "bright_hypothesis" in normalized
    assert "rim_geometry_hypothesis" in normalized
    assert "APP_BASELINE_MIN_PEAK_RATIO" in text
