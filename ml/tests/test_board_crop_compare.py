"""Unit tests for the board-vs-training crop comparison helpers."""

from __future__ import annotations

from pathlib import Path
import sys

# Add `ml/src` to sys.path so this test can run without an editable install.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pytest

from embedded_gauge_reading_tinyml.board_crop_compare import (
    compute_training_crop_box,
    estimate_board_crop_from_rgb,
    load_yuv422_capture_as_rgb,
    resize_with_pad_rgb,
)
from embedded_gauge_reading_tinyml.dataset import EllipseLabel, PointLabel, Sample


def _make_sample() -> Sample:
    """Build a small synthetic sample with a simple dial ellipse."""
    return Sample(
        image_path=Path("synthetic.png"),
        dial=EllipseLabel(
            cx=120.0,
            cy=110.0,
            rx=40.0,
            ry=30.0,
            rotation=0.0,
            label="temp_dial",
        ),
        center=PointLabel(x=120.0, y=110.0, label="temp_center"),
        tip=PointLabel(x=150.0, y=110.0, label="temp_tip"),
    )


def _make_yuv422_bytes(
    *,
    width: int = 224,
    height: int = 224,
    background_luma: int = 10,
    bright_luma: int = 220,
) -> bytes:
    """Build a packed YUYV buffer with one bright rectangular region."""
    # Each pair of pixels uses four bytes: Y0, U, Y1, V.
    pairs = np.zeros((height, width // 2, 4), dtype=np.uint8)
    pairs[:, :, 0] = background_luma
    pairs[:, :, 2] = background_luma
    pairs[88:120, 48:80, 0] = bright_luma
    pairs[88:120, 48:80, 2] = bright_luma
    return pairs.tobytes()


def test_estimate_board_crop_from_rgb_uses_bright_centroid() -> None:
    """The crop estimator should anchor on the bright dial-like region."""
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[88:120, 96:128] = 220

    estimate = estimate_board_crop_from_rgb(image)

    assert estimate is not None
    assert estimate.crop_box.width == 156
    assert estimate.crop_box.height == 123
    assert estimate.crop_box.centroid_x == 111
    assert estimate.crop_box.centroid_y == 103
    assert estimate.crop_box.x_min == 33
    assert estimate.crop_box.y_min == 42
    assert estimate.center_luma == 220
    assert estimate.mean_luma > 0.0
    assert estimate.min_luma == 0
    assert estimate.max_luma == 220


def test_estimate_board_crop_from_rgb_returns_none_for_blank_image() -> None:
    """Blank images should not produce a bogus crop estimate."""
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    assert estimate_board_crop_from_rgb(image) is None


def test_load_yuv422_capture_as_rgb_decodes_luma_to_grayscale(
    tmp_path: Path,
) -> None:
    """Raw board captures should decode to a grayscale RGB array."""
    raw_path = tmp_path / "synthetic_capture.yuv422"
    raw_bytes = _make_yuv422_bytes()
    raw_path.write_bytes(raw_bytes)

    image = load_yuv422_capture_as_rgb(raw_path)

    assert image.shape == (224, 224, 3)
    assert int(image[0, 0, 0]) == 10
    assert int(image[100, 100, 0]) == 220
    assert np.all(image[100, 100] == image[100, 100, 0])


def test_resize_with_pad_rgb_outputs_expected_shape_and_signal() -> None:
    """The crop-and-pad path should preserve shape and keep bright content alive."""
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[88:120, 96:128] = 220
    estimate = estimate_board_crop_from_rgb(image)
    assert estimate is not None

    cropped = resize_with_pad_rgb(
        image,
        (
            float(estimate.crop_box.x_min),
            float(estimate.crop_box.y_min),
            float(estimate.crop_box.x_max),
            float(estimate.crop_box.y_max),
        ),
    )

    assert cropped.shape == (224, 224, 3)
    assert int(cropped[112, 112, 0]) > 0
    assert np.all(cropped[112, 112] == cropped[112, 112, 0])


def test_compute_training_crop_box_matches_ellipse_padding_math() -> None:
    """The training crop helper should keep the expected ellipse padding."""
    sample = _make_sample()

    crop_box = compute_training_crop_box(sample, 0.10)

    assert crop_box == pytest.approx((76.0, 77.0, 164.0, 143.0))
