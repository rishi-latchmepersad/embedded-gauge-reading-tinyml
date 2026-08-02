"""Fresh-checkout smoke tests for the WSL/ML workspace contract."""

from __future__ import annotations

from pathlib import Path
import tomllib

import numpy as np

from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import (
    decode_heatmap_point_xy,
)


ML_ROOT = Path(__file__).resolve().parents[2]


def test_workspace_manifest_declares_current_contract() -> None:
    """The checked-in manifest must identify the current board and research paths."""

    with (ML_ROOT / "workspace_manifest.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    assert manifest["workspace"]["active_board_contract"] == "obb_then_tip_focus_v18"
    assert manifest["workspace"]["research_candidate"] == "ellipse_then_keypoint_temperature"
    assert manifest["required_inputs"]
    assert manifest["optional_artifacts"]


def test_heatmap_decoder_maps_peak_to_input_pixels() -> None:
    """The shared decoder must preserve the center/tip input-space contract."""

    heatmap = np.zeros((56, 56), dtype=np.float32)
    heatmap[14, 28] = 1.0
    x_pixel, y_pixel = decode_heatmap_point_xy(
        heatmap,
        method="argmax",
        heatmap_size=56,
        input_size=224,
    )
    assert x_pixel == 28.0 * 223.0 / 55.0
    assert y_pixel == 14.0 * 223.0 / 55.0
