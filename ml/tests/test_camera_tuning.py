"""Regression tests for the live-board camera tuning knobs."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
CAMERA_CONFIG_FILE: Path = (
    REPO_ROOT / "firmware" / "stm32" / "n657" / "Appli" / "Inc" / "app_camera_config.h"
)
APP_AI_FILE: Path = (
    REPO_ROOT / "firmware" / "stm32" / "n657" / "Appli" / "Src" / "app_ai.c"
)
GAUGE_GEOMETRY_FILE: Path = (
    REPO_ROOT / "firmware" / "stm32" / "n657" / "Appli" / "Inc" / "app_gauge_geometry.h"
)
_FLOAT_RE: str = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?f?"
_UINT_RE: str = r"\d+U?"


def _extract_float_constant(name: str, text: str) -> float:
    """Extract one named float constant from a firmware header/source file."""
    pattern = rf"{re.escape(name)}\s+({_FLOAT_RE})"
    match = re.search(pattern, text)
    if match is None:
        raise AssertionError(f"Could not find {name}.")
    return float(match.group(1).rstrip("f"))


def _extract_uint_constant(name: str, text: str) -> int:
    """Extract one named unsigned integer constant from firmware source."""
    pattern = rf"{re.escape(name)}\s+({_UINT_RE})"
    match = re.search(pattern, text)
    if match is None:
        raise AssertionError(f"Could not find {name}.")
    return int(match.group(1).rstrip("U"))


def test_camera_brightness_nudges_use_fractional_steps() -> None:
    """The brightness gate should use a small proportional step, not a 2x jump."""
    text = CAMERA_CONFIG_FILE.read_text(encoding="utf-8")

    exposure_shift = _extract_uint_constant(
        "CAMERA_CAPTURE_BRIGHTNESS_EXPOSURE_STEP_FRACTION_SHIFT",
        text,
    )
    gain_shift = _extract_uint_constant(
        "CAMERA_CAPTURE_BRIGHTNESS_GAIN_STEP_FRACTION_SHIFT",
        text,
    )

    assert exposure_shift == 2
    assert gain_shift == 2


def test_camera_brightness_gate_uses_a_bright_pixel_ratio() -> None:
    """The bright-frame gate should notice broad overexposure, not just min-Y."""
    text = CAMERA_CONFIG_FILE.read_text(encoding="utf-8")

    bright_mean_threshold = _extract_uint_constant(
        "CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_MEAN_THRESHOLD",
        text,
    )
    bright_pixel_threshold = _extract_uint_constant(
        "CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_PIXEL_LEVEL_THRESHOLD",
        text,
    )
    bright_ratio_percent = _extract_uint_constant(
        "CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_RATIO_PERCENT",
        text,
    )
    bright_solid_mean_threshold = _extract_uint_constant(
        "CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_SOLID_MEAN_THRESHOLD",
        text,
    )

    assert bright_mean_threshold == 180
    assert bright_pixel_threshold == 220
    assert bright_ratio_percent == 50
    assert bright_solid_mean_threshold == 220


def test_obb_crop_window_keeps_the_current_live_crop_in_family() -> None:
    """The OBB crop window should keep the current live crop on the fast path."""
    ai_text = APP_AI_FILE.read_text(encoding="utf-8")
    gauge_text = GAUGE_GEOMETRY_FILE.read_text(encoding="utf-8")

    crop_min_ratio = _extract_float_constant("APP_AI_OBB_TRAINING_CROP_MIN_RATIO", ai_text)
    crop_max_ratio = _extract_float_constant("APP_AI_OBB_TRAINING_CROP_MAX_RATIO", ai_text)
    training_x_min = _extract_float_constant("APP_GAUGE_TRAINING_CROP_X_MIN_RATIO", gauge_text)
    training_y_min = _extract_float_constant("APP_GAUGE_TRAINING_CROP_Y_MIN_RATIO", gauge_text)
    training_x_max = _extract_float_constant("APP_GAUGE_TRAINING_CROP_X_MAX_RATIO", gauge_text)
    training_y_max = _extract_float_constant("APP_GAUGE_TRAINING_CROP_Y_MAX_RATIO", gauge_text)

    frame_width = 224.0
    frame_height = 224.0
    training_crop_width = round(frame_width * (training_x_max - training_x_min))
    training_crop_height = round(frame_height * (training_y_max - training_y_min))

    # Current live OBB crops were clustering around 152x159 in the UART logs.
    live_crop_width = 152.0
    live_crop_height = 159.0
    width_ratio = live_crop_width / float(training_crop_width)
    height_ratio = live_crop_height / float(training_crop_height)

    assert math.isfinite(width_ratio)
    assert math.isfinite(height_ratio)
    assert crop_min_ratio == pytest.approx(0.60)
    assert crop_max_ratio == pytest.approx(1.40)
    assert crop_min_ratio <= width_ratio <= crop_max_ratio
    assert crop_min_ratio <= height_ratio <= crop_max_ratio
