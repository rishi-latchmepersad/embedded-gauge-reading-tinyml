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
    peak_ratio = _extract_float_constant("APP_BASELINE_MIN_PEAK_RATIO", text)

    # The live board needs a slightly more permissive gate so real captures can
    # seed history without waiting for perfect SNR.
    assert 1.0 < threshold < 2.0
    assert 1.0 <= peak_ratio <= 1.05
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


def test_baseline_selection_penalizes_spiky_peak_outliers() -> None:
    """Classical candidates should favor stable support over ratio explosions."""
    text = BASELINE_RUNTIME_FILE.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", text)

    assert "AppBaselineRuntime_IsBetterEstimate" in normalized
    assert "AppBaselineRuntime_ComputeEstimateQuality" in normalized
    assert "AppBaselineRuntime_PassesAcceptanceGate" in normalized
    assert "AppBaselineRuntime_GeometryPriority" not in normalized
    assert "confidence / peak_ratio" in normalized
    assert "spiky false geometry" in text
    assert "runner_up_score > 0.0f" in normalized
    assert "selected_is_fixed_crop" not in normalized
    assert "APP_BASELINE_MIN_ACCEPT_SCORE" in text
    assert "dial_radius_px * 0.75f" in normalized
    assert "rim_geometry_hypothesis" in normalized
    assert "rim-center-polar" in text
    assert "candidate_quality > incumbent_quality" in normalized
    assert "candidate_peak_ratio > incumbent_peak_ratio" in normalized
    assert "APP_BASELINE_CONSENSUS_MIN_QUALITY_RATIO" in text


def test_baseline_source_priority_keeps_fixed_crop_ahead_of_board_prior() -> None:
    """The live firmware should rank the stable crop ahead of the board prior."""
    text = BASELINE_RUNTIME_FILE.read_text(encoding="utf-8")
    start = text.index("static int AppBaselineRuntime_SourcePriority(const char *source_label)")
    end = text.index(
        "/**\n * @brief Check whether one estimate clears the live acceptance gate.",
        start,
    )
    source_priority_block = text[start:end]

    assert 'strcmp(source_label, "fixed-crop-polar") == 0' in source_priority_block
    assert 'return 5;' in source_priority_block
    assert 'strcmp(source_label, "image-center-polar") == 0' in source_priority_block
    assert 'return 4;' in source_priority_block
    assert 'strcmp(source_label, "board-prior-polar") == 0' in source_priority_block
    assert 'return 3;' in source_priority_block
    assert source_priority_block.index('fixed-crop-polar') < source_priority_block.index(
        'board-prior-polar'
    )


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


def test_baseline_candidate_selection_enables_local_refinement_sweep() -> None:
    """The firmware selector should use the narrow local refinement sweep."""
    text = BASELINE_RUNTIME_FILE.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", text)
    sweep_branch_start = text.index("#if APP_BASELINE_ENABLE_LOCAL_GEOMETRY_SWEEP")
    sweep_branch_end = text.index("#else", sweep_branch_start)
    sweep_branch = text[sweep_branch_start:sweep_branch_end]

    assert "APP_BASELINE_ENABLE_LOCAL_GEOMETRY_SWEEP" in text
    assert "APP_BASELINE_ENABLE_LOCAL_GEOMETRY_SWEEP 1U" in text
    assert "Run the narrow local geometry sweep by default" in text
    assert "#if APP_BASELINE_ENABLE_LOCAL_GEOMETRY_SWEEP" in text
    assert "#else" in text
    assert "const AppBaselineRuntime_Estimate_t *seed_candidates[5]" in sweep_branch
    assert "AppBaselineRuntime_Estimate_t *refined_candidates[5]" in sweep_branch
    assert "AppBaselineRuntime_RefineEstimateAroundSeed(" in sweep_branch
    assert "AppBaselineRuntime_SelectConsensusEstimate(" in sweep_branch
    assert "&fixed_crop_hypothesis" in normalized
    assert "&board_prior_hypothesis" in normalized
    assert "&center_hypothesis" in normalized
    assert "bright_hypothesis" in normalized
    assert "rim_geometry_hypothesis" in normalized
    assert "APP_BASELINE_MIN_PEAK_RATIO" in text
    assert "APP_BASELINE_CONSENSUS_MIN_QUALITY_RATIO" in text
    assert "selected_estimate = AppBaselineRuntime_SelectConsensusEstimate(" in sweep_branch


def test_baseline_consensus_respects_source_priority_before_quality() -> None:
    """Consensus should keep the stronger anchor family ahead of rim clutter."""
    text = BASELINE_RUNTIME_FILE.read_text(encoding="utf-8")
    consensus_start = text.index(
        "static const AppBaselineRuntime_Estimate_t *AppBaselineRuntime_SelectConsensusEstimate("
    )
    consensus_end = text.index(
        "/**\n * @brief Return a smoothed estimate from the tiny baseline history.",
        consensus_start,
    )
    consensus_block = text[consensus_start:consensus_end]

    assert "best_priority" in consensus_block
    assert "fallback_priority" in consensus_block
    assert "AppBaselineRuntime_SourcePriority(candidate->source_label)" in consensus_block
    assert "AppBaselineRuntime_SourcePriority(fallback_estimate->source_label)" in consensus_block
    assert "if (best_priority < fallback_priority)" in consensus_block
    assert "if (best_priority > fallback_priority)" in consensus_block
    assert "APP_BASELINE_CONSENSUS_MIN_QUALITY_RATIO" in consensus_block
