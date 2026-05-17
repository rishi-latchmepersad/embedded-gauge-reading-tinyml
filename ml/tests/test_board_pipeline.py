"""Tests for the firmware-parity board pipeline helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embedded_gauge_reading_tinyml.board_pipeline import (
    BoardCalibration,
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


def test_rectifier_falls_back_to_training_crop_when_center_runs_away() -> None:
    """Rectifier center values outside the trust window should use the fixed crop."""
    rectifier_params = np.array([0.95, 0.50, 0.40, 0.40], dtype=np.float32)
    result = decode_rectifier_crop_box(
        rectifier_params,
        source_width=224,
        source_height=224,
    )

    assert result.accepted is True
    assert result.fallback_reason == "centre out of range"
    assert result.crop_box_xyxy == _training_crop_box(224, 224)


def test_rectifier_uses_blended_training_crop_when_center_is_plausible() -> None:
    """A plausible rectifier center should keep the fixed training crop size."""
    rectifier_params = np.array([0.60, 0.55, 0.40, 0.40], dtype=np.float32)
    result = decode_rectifier_crop_box(
        rectifier_params,
        source_width=224,
        source_height=224,
    )

    training_x0, training_y0, training_x1, training_y1 = _training_crop_box(224, 224)
    expected_width = training_x1 - training_x0
    expected_height = training_y1 - training_y0

    assert result.accepted is True
    assert result.fallback_reason is None
    assert result.crop_box_xyxy[2] - result.crop_box_xyxy[0] == expected_width
    assert result.crop_box_xyxy[3] - result.crop_box_xyxy[1] == expected_height


def test_obb_rejects_crops_outside_the_training_window() -> None:
    """Very large OBB boxes should be rejected and sent to the rectifier."""
    obb_params = np.array([0.50, 0.50, 0.95, 0.95, 1.0, 0.0], dtype=np.float32)
    result = decode_obb_crop_box(
        obb_params,
        source_width=224,
        source_height=224,
    )

    assert result.accepted is False
    assert result.fallback_reason == "crop outside training window"
    assert (
        result.details["crop_width_ratio"] < 0.60
        or result.details["crop_width_ratio"] > 1.40
    )
    assert (
        result.details["crop_height_ratio"] < 0.60
        or result.details["crop_height_ratio"] > 1.40
    )


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
