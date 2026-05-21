"""Tests for the firmware-parity board pipeline helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embedded_gauge_reading_tinyml.board_pipeline import (
    BoardCalibration,
    OBB_CROP_SCALE,
    OBB_HEIGHT_SCALE,
    OBB_TRAINING_CROP_MIN_RATIO,
    OBB_TRAINING_CROP_MAX_RATIO,
    OBB_WIDTH_SCALE,
    RECTIFIER_CROP_SCALE,
    InferenceBurstHistory,
    _apply_board_calibration,
    _load_board_calibration,
    decode_obb_crop_box,
    decode_rectifier_crop_box,
    load_capture_image,
    _training_crop_box,
)


def test_yuv422_loader_repeats_luma(tmp_path: Path) -> None:
    """YUV422 replay should expose luma as a grayscale RGB image."""
    capture_path = tmp_path / "capture.yuv422"
    pair = bytes([25, 128, 50, 128])
    capture_path.write_bytes(pair * (224 * 224 // 2))

    image, kind = load_capture_image(capture_path, image_width=224, image_height=224)
    assert kind == "yuv422"
    assert image.shape == (224, 224, 3)
    assert int(image[0, 0, 0]) == 25
    assert int(image[0, 0, 1]) == 25
    assert int(image[0, 1, 2]) == 50


def test_rectifier_scales_crop_even_when_center_runs_away() -> None:
    """Rectifier crop decode should stay geometry-aware even when the center drifts."""
    rectifier_params = np.array([0.95, 0.50, 0.40, 0.40], dtype=np.float32)
    result = decode_rectifier_crop_box(
        rectifier_params,
        source_width=224,
        source_height=224,
    )

    assert result.accepted is True
    assert result.fallback_reason is None
    expected_width = pytest.approx(0.40 * 224.0 * RECTIFIER_CROP_SCALE)
    expected_height = pytest.approx(0.40 * 224.0 * RECTIFIER_CROP_SCALE)
    assert result.crop_box_xyxy[2] - result.crop_box_xyxy[0] == expected_width
    assert result.crop_box_xyxy[3] - result.crop_box_xyxy[1] == expected_height


def test_rectifier_uses_the_same_crop_scale_for_a_plausible_center() -> None:
    """A plausible rectifier center should keep the same crop geometry."""
    rectifier_params = np.array([0.60, 0.55, 0.40, 0.40], dtype=np.float32)
    result = decode_rectifier_crop_box(
        rectifier_params,
        source_width=224,
        source_height=224,
    )

    assert result.accepted is True
    assert result.fallback_reason is None
    assert result.crop_box_xyxy[2] - result.crop_box_xyxy[0] == pytest.approx(
        0.40 * 224.0 * RECTIFIER_CROP_SCALE
    )
    assert result.crop_box_xyxy[3] - result.crop_box_xyxy[1] == pytest.approx(
        0.40 * 224.0 * RECTIFIER_CROP_SCALE
    )


def test_obb_keeps_tiny_crops_usable_with_the_min_size_floor() -> None:
    """Tiny OBB boxes should still expand to a usable crop instead of collapsing."""
    obb_params = np.array([0.50, 0.50, 0.05, 0.05, 1.0, 0.0], dtype=np.float32)
    result = decode_obb_crop_box(
        obb_params,
        source_width=224,
        source_height=224,
    )

    assert result.accepted is True
    assert result.fallback_reason is None
    assert OBB_CROP_SCALE == pytest.approx(0.83)
    assert OBB_TRAINING_CROP_MIN_RATIO == pytest.approx(0.15)
    assert OBB_TRAINING_CROP_MAX_RATIO == pytest.approx(1.60)
    assert result.crop_box_xyxy[2] - result.crop_box_xyxy[0] == pytest.approx(48.0)
    assert result.crop_box_xyxy[3] - result.crop_box_xyxy[1] == pytest.approx(48.0)
    assert result.details["crop_width_ratio"] >= OBB_TRAINING_CROP_MIN_RATIO
    assert result.details["crop_height_ratio"] >= OBB_TRAINING_CROP_MIN_RATIO


def test_obb_center_bias_shifts_the_crop_window() -> None:
    """The OBB decoder should expose a controllable center bias for replay tuning."""
    obb_params = np.array([0.50, 0.50, 0.50, 0.40, 1.0, 0.0], dtype=np.float32)
    baseline = decode_obb_crop_box(
        obb_params,
        source_width=224,
        source_height=224,
    )
    biased = decode_obb_crop_box(
        obb_params,
        source_width=224,
        source_height=224,
        obb_center_x_bias_pixels=5.0,
        obb_center_y_bias_pixels=-5.0,
    )

    assert biased.crop_box_xyxy[0] == pytest.approx(baseline.crop_box_xyxy[0] + 5.0)
    assert biased.crop_box_xyxy[1] == pytest.approx(baseline.crop_box_xyxy[1] - 5.0)
    assert biased.details["center_x_bias_pixels"] == pytest.approx(5.0)
    assert biased.details["center_y_bias_pixels"] == pytest.approx(-5.0)


def test_obb_aspect_scales_adjust_the_crop_shape() -> None:
    """The OBB decoder should allow separate width and height tuning."""
    obb_params = np.array([0.50, 0.50, 0.50, 0.40, 1.0, 0.0], dtype=np.float32)
    baseline = decode_obb_crop_box(
        obb_params,
        source_width=224,
        source_height=224,
    )
    shaped = decode_obb_crop_box(
        obb_params,
        source_width=224,
        source_height=224,
        obb_width_scale=0.94,
        obb_height_scale=1.18,
    )

    baseline_width = baseline.crop_box_xyxy[2] - baseline.crop_box_xyxy[0]
    baseline_height = baseline.crop_box_xyxy[3] - baseline.crop_box_xyxy[1]
    shaped_width = shaped.crop_box_xyxy[2] - shaped.crop_box_xyxy[0]
    shaped_height = shaped.crop_box_xyxy[3] - shaped.crop_box_xyxy[1]

    assert OBB_WIDTH_SCALE == pytest.approx(1.0)
    assert OBB_HEIGHT_SCALE == pytest.approx(1.0)
    assert shaped_width == pytest.approx(baseline_width * 0.94)
    assert shaped_height == pytest.approx(baseline_height * 1.18)


def test_obb_source_bias_shifts_the_final_crop_box() -> None:
    """The OBB decoder should allow direct source-space crop correction."""
    obb_params = np.array([0.50, 0.50, 0.50, 0.40, 1.0, 0.0], dtype=np.float32)
    baseline = decode_obb_crop_box(
        obb_params,
        source_width=224,
        source_height=224,
    )
    shifted = decode_obb_crop_box(
        obb_params,
        source_width=224,
        source_height=224,
        obb_source_x_bias_pixels=-10.0,
        obb_source_y_bias_pixels=6.0,
    )

    assert shifted.crop_box_xyxy[0] == pytest.approx(baseline.crop_box_xyxy[0] - 10.0)
    assert shifted.crop_box_xyxy[1] == pytest.approx(baseline.crop_box_xyxy[1] + 6.0)
    assert shifted.details["source_x_bias_pixels"] == pytest.approx(-10.0)
    assert shifted.details["source_y_bias_pixels"] == pytest.approx(6.0)


def test_obb_source_shape_scales_adjust_the_final_crop_box() -> None:
    """The OBB decoder should allow post-projection width and height tuning."""
    obb_params = np.array([0.50, 0.50, 0.50, 0.40, 1.0, 0.0], dtype=np.float32)
    baseline = decode_obb_crop_box(
        obb_params,
        source_width=224,
        source_height=224,
    )
    shaped = decode_obb_crop_box(
        obb_params,
        source_width=224,
        source_height=224,
        obb_source_width_scale=0.90,
        obb_source_height_scale=0.80,
    )

    baseline_width = baseline.crop_box_xyxy[2] - baseline.crop_box_xyxy[0]
    baseline_height = baseline.crop_box_xyxy[3] - baseline.crop_box_xyxy[1]
    shaped_width = shaped.crop_box_xyxy[2] - shaped.crop_box_xyxy[0]
    shaped_height = shaped.crop_box_xyxy[3] - shaped.crop_box_xyxy[1]

    assert shaped_width == pytest.approx(baseline_width * 0.90)
    assert shaped_height == pytest.approx(baseline_height * 0.80)
    assert shaped.details["source_width_scale"] == pytest.approx(0.90)
    assert shaped.details["source_height_scale"] == pytest.approx(0.80)


def test_burst_history_matches_firmware_median_and_reset() -> None:
    """The burst history should behave like the firmware's 3-sample smoother."""
    history = InferenceBurstHistory()

    value, reset, count = history.update(10.0)
    assert value == 10.0
    assert reset is False
    assert count == 1

    value, reset, count = history.update(12.0)
    assert value == 11.0
    assert reset is False
    assert count == 2

    value, reset, count = history.update(11.0)
    assert value == 11.0
    assert reset is False
    assert count == 3

    value, reset, count = history.update(30.0)
    assert value == 30.0
    assert reset is True
    assert count == 1


def test_affine_board_calibration_applies_expected_transform(tmp_path: Path) -> None:
    """Affine calibration should scale and shift the raw scalar prediction."""
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "selected_mode": "affine",
                "affine": {"scale": 1.25, "bias": -2.5},
            }
        ),
        encoding="utf-8",
    )

    calibration = _load_board_calibration(str(calibration_path))
    assert calibration == BoardCalibration(mode="affine", scale=1.25, bias=-2.5)
    assert _apply_board_calibration(10.0, calibration) == pytest.approx(10.0)


def test_piecewise_board_calibration_applies_expected_transform(tmp_path: Path) -> None:
    """Piecewise calibration should match the stored hinge basis parameters."""
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "selected_mode": "piecewise",
                "piecewise": {
                    "bias": 1.0,
                    "weights": [2.0, 0.5, -0.25],
                    "knots": [4.0, 8.0],
                },
            }
        ),
        encoding="utf-8",
    )

    calibration = _load_board_calibration(str(calibration_path))
    assert calibration.mode == "piecewise"
    assert calibration.bias == pytest.approx(1.0)
    assert calibration.weights == (2.0, 0.5, -0.25)
    assert calibration.knots == (4.0, 8.0)

    # 1.0 + 2.0 * 10 + 0.5 * (10 - 4) - 0.25 * (10 - 8) = 23.5
    assert _apply_board_calibration(10.0, calibration) == pytest.approx(23.5)
