"""Tests for the board-style geometry heatmap replay helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from embedded_gauge_reading_tinyml.geometry_board_replay import (
    build_board_replay_sample,
    load_board_replay_image,
    preprocess_board_replay_image,
)
from embedded_gauge_reading_tinyml.geometry_crop_dataset import SourceGeometryExample


def _make_example(image_path: Path) -> SourceGeometryExample:
    """Build a tiny manifest-like example with an in-bounds loose crop."""

    return SourceGeometryExample(
        image_path=str(image_path),
        temperature_c=12.5,
        source_width=8,
        source_height=6,
        loose_crop_x1=1,
        loose_crop_y1=1,
        loose_crop_x2=7,
        loose_crop_y2=5,
        center_x_source=3.0,
        center_y_source=3.0,
        tip_x_source=5.0,
        tip_y_source=3.0,
        dial_radius_source=2.0,
        split="test",
        quality_flag="clean",
    )


def test_load_board_replay_image_decodes_yuv422_as_grayscale_rgb(tmp_path: Path) -> None:
    """YUV422 captures should decode to grayscale RGB for replay."""

    capture_path = tmp_path / "capture.yuv422"
    pairs = np.array(
        [
            [10, 128, 20, 128],
            [30, 128, 40, 128],
        ],
        dtype=np.uint8,
    )
    capture_path.write_bytes(pairs.tobytes())

    image, kind = load_board_replay_image(capture_path, image_width=4, image_height=1)

    assert kind == "yuv422"
    assert image.shape == (1, 4, 3)
    assert int(image[0, 0, 0]) == 10
    assert int(image[0, 1, 0]) == 20
    assert np.all(image[0, 0] == image[0, 0, 0])
    assert np.all(image[0, 1] == image[0, 1, 0])


def test_preprocess_board_replay_image_supports_bilinear_nearest_and_luma_modes() -> None:
    """The replay helper should expose all three supported preprocessing modes."""

    image = np.array(
        [
            [[0, 0, 0], [32, 64, 96], [64, 96, 128], [96, 128, 160]],
            [[16, 32, 48], [48, 80, 112], [80, 112, 144], [112, 144, 176]],
            [[32, 48, 64], [64, 96, 128], [96, 128, 160], [128, 160, 192]],
        ],
        dtype=np.uint8,
    )
    crop_box = (0.0, 0.0, 4.0, 3.0)

    bilinear, bilinear_meta = preprocess_board_replay_image(
        image,
        crop_box_xyxy=crop_box,
        mode="python_training_rgb_bilinear",
        input_size=8,
    )
    nearest, nearest_meta = preprocess_board_replay_image(
        image,
        crop_box_xyxy=crop_box,
        mode="board_like_rgb_nearest",
        input_size=8,
    )
    luma, luma_meta = preprocess_board_replay_image(
        image,
        crop_box_xyxy=crop_box,
        mode="board_like_luma_nearest_if_supported",
        input_size=8,
    )

    assert bilinear.shape == (8, 8, 3)
    assert nearest.shape == (8, 8, 3)
    assert luma.shape == (8, 8, 3)
    assert bilinear_meta["resize_method"] == "rgb_bilinear"
    assert nearest_meta["resize_method"] == "rgb_nearest"
    assert luma_meta["resize_method"] == "luma_nearest"
    assert luma_meta["luma_supported"] is True
    assert np.allclose(luma[..., 0], luma[..., 1])
    assert np.allclose(luma[..., 1], luma[..., 2])
    assert not np.allclose(bilinear, nearest)


def test_build_board_replay_sample_populates_crop_metadata(tmp_path: Path) -> None:
    """Identity board replay samples should carry the crop and label metadata."""

    image_path = tmp_path / "sample.png"
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    image[1:5, 1:7] = np.array([160, 120, 80], dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(image_path)

    example = _make_example(image_path)
    sample = build_board_replay_sample(
        example,
        tmp_path,
        mode="board_like_rgb_nearest",
        input_size=8,
        heatmap_size=4,
        sigma_pixels=1.5,
    )

    assert sample.crop_image.shape == (8, 8, 3)
    assert sample.metadata["preprocessing_mode"] == "board_like_rgb_nearest"
    assert sample.metadata["crop_x1"] == 1
    assert sample.metadata["crop_y1"] == 1
    assert sample.metadata["crop_x2"] == 7
    assert sample.metadata["crop_y2"] == 5
    assert sample.metadata["center_x_224"] == pytest.approx(74.6666666667)
    assert sample.metadata["tip_x_224"] == pytest.approx(149.3333333333)
    assert sample.center_heatmap.shape == (4, 4)
    assert sample.tip_heatmap.shape == (4, 4)
