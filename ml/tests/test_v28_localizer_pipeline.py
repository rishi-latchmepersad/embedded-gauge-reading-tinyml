"""Tests for localizer-to-crop helpers used by the exact V28 replay."""

from __future__ import annotations

import numpy as np

from embedded_gauge_reading_tinyml.v28_localizer_pipeline import (
    decode_heatmap_crop_box,
    decode_keypoint_crop_box,
    decode_source_crop_box,
)


def test_decode_keypoint_crop_box_centers_on_mean_keypoint() -> None:
    """The crop should center on the mean predicted keypoint position."""
    coords = np.array([[112.0, 112.0], [120.0, 104.0]], dtype=np.float32)
    decision = decode_keypoint_crop_box(
        coords,
        source_width=2592,
        source_height=1944,
        keypoint_crop_scale=0.83,
        min_crop_size=48.0,
    )
    x0, y0, x1, y1 = decision.crop_box_xyxy
    assert decision.crop_source == "keypoint_coords"
    assert decision.accepted
    assert x1 > x0 and y1 > y0
    assert 1000.0 < (x1 - x0) < 2000.0
    assert 700.0 < (y1 - y0) < 1400.0


def test_decode_heatmap_crop_box_accepts_heatmaps() -> None:
    """Heatmaps should decode to the same crop helper as explicit keypoints."""
    heatmaps = np.zeros((28, 28, 2), dtype=np.float32)
    heatmaps[14, 14, 0] = 10.0
    heatmaps[15, 15, 1] = 10.0
    decision = decode_heatmap_crop_box(
        heatmaps,
        source_width=2592,
        source_height=1944,
        keypoint_crop_scale=0.83,
        min_crop_size=48.0,
    )
    assert decision.crop_source == "keypoint_coords"
    assert decision.accepted


def test_decode_source_crop_box_accepts_xyxy_predictions() -> None:
    """Source-space box predictions should decode directly to a crop."""
    box = np.array([0.10, 0.20, 0.80, 0.90], dtype=np.float32)
    decision = decode_source_crop_box(
        box,
        source_width=2592,
        source_height=1944,
    )
    x0, y0, x1, y1 = decision.crop_box_xyxy
    assert decision.crop_source == "source_crop_box"
    assert decision.accepted
    assert x1 > x0 and y1 > y0
    assert 0.0 <= x0 < x1 <= 2592.0
    assert 0.0 <= y0 < y1 <= 1944.0
