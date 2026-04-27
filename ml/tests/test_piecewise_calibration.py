"""Regression tests for the firmware's affine scalar calibration."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
FIRMWARE_CALIBRATION_FILE: Path = (
    REPO_ROOT / "firmware" / "stm32" / "n657" / "Appli" / "Src" / "app_inference_calibration.c"
)
AFFINE_P5_METRICS_FILE: Path = (
    REPO_ROOT
    / "ml"
    / "artifacts"
    / "training"
    / "scalar_full_finetune_from_best_affine_calibrated_p5"
    / "metrics.json"
)
_FLOAT_RE: str = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?f?"


def _extract_float(name: str, text: str) -> float:
    """Extract one named float constant from the firmware source."""
    pattern = rf"{re.escape(name)}\s*=\s*({_FLOAT_RE});"
    match = re.search(pattern, text)
    if match is None:
        raise AssertionError(f"Could not find {name} in {FIRMWARE_CALIBRATION_FILE}.")
    return float(match.group(1).rstrip("f"))


def _evaluate_affine(raw_value: float, scale: float, bias: float) -> float:
    """Evaluate the affine correction used by the firmware."""
    return bias + (scale * raw_value)


def test_firmware_affine_calibration_matches_hard_case_p5_fit() -> None:
    """The firmware should match the hard-case affine p5 calibration payload."""
    text = FIRMWARE_CALIBRATION_FILE.read_text(encoding="utf-8")
    metrics = json.loads(AFFINE_P5_METRICS_FILE.read_text(encoding="utf-8"))

    scale = _extract_float("kCalibrationAffineScale", text)
    bias = _extract_float("kCalibrationAffineBias", text)

    assert "kCalibrationPiecewiseWeights" not in text
    assert "APP_INFERENCE_USE_PIECEWISE_CALIBRATION" not in text
    assert metrics["mode"] == "affine"
    assert metrics["calibrated_mae"] < metrics["raw_mae"]
    assert scale == pytest.approx(metrics["scale"], abs=1e-6)
    assert bias == pytest.approx(metrics["bias"], abs=1e-6)

    # A few spot checks keep the affine math honest and make the output trend
    # easy to reason about in the board logs.
    expectations = {
        -30.0: -34.15068119764328,
        -10.0: -10.888690650463104,
        20.0: 24.00429517030716,
        45.0: 53.08178335428238,
        -22.547728: -25.482946972846985,
        -22.2549: -25.142358887195587,
    }
    for raw_value, expected in expectations.items():
        calibrated = _evaluate_affine(raw_value, scale, bias)
        assert calibrated == pytest.approx(expected, abs=1e-6)
