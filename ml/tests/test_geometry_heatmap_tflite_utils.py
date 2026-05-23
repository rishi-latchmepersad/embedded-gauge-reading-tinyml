"""Tests for the shared geometry heatmap TFLite helper utilities."""

from __future__ import annotations

import numpy as np
import pytest

from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import (
    decode_heatmap_point,
    decode_heatmap_point_xy,
    reorder_tflite_outputs,
)


def _single_peak_heatmap(row: int, col: int, *, size: int = 56) -> np.ndarray:
    """Build a deterministic heatmap with one exact peak."""

    heatmap = np.zeros((size, size), dtype=np.float32)
    heatmap[row, col] = 1.0
    return heatmap


def test_reorder_tflite_outputs_uses_semantic_mapping() -> None:
    """The raw TFLite output order should be remapped deterministically."""

    outputs = [
        np.array([[1.0]], dtype=np.float32),
        np.array([[2.0]], dtype=np.float32),
        np.array([[3.0]], dtype=np.float32),
    ]

    reordered = reorder_tflite_outputs(outputs, [1, 0, 2])

    assert [tensor.item() for tensor in reordered] == [2.0, 1.0, 3.0]


def test_decode_heatmap_point_local_window_softargmax_tracks_peak() -> None:
    """Local-window softargmax should stay near a sharp peak even with diffuse background."""

    heatmap = np.full((7, 7), 0.01, dtype=np.float32)
    heatmap[5, 1] = 2.0
    heatmap[5, 2] = 1.0
    heatmap[4, 1] = 0.6

    row, col = decode_heatmap_point(heatmap, method="local_window_softargmax", window_size=3)

    assert row == pytest.approx(5.0, abs=0.25)
    assert col == pytest.approx(1.5, abs=0.6)


def test_decode_heatmap_point_peak_weighted_centroid_tracks_peak_window() -> None:
    """The centroid decoder should average only the local peak neighborhood."""

    heatmap = np.zeros((7, 7), dtype=np.float32)
    heatmap[1, 5] = 0.4
    heatmap[1, 6] = 1.0
    heatmap[2, 6] = 0.6

    row, col = decode_heatmap_point(heatmap, method="peak_weighted_centroid", window_size=3)

    assert row == pytest.approx(1.375, abs=0.25)
    assert col == pytest.approx(5.75, abs=0.25)


def test_decode_heatmap_point_xy_uses_heatmap_size() -> None:
    """Decoded coordinates should scale from heatmap pixels to crop pixels."""

    heatmap = np.zeros((56, 56), dtype=np.float32)
    heatmap[14, 28] = 1.0

    x_pixel, y_pixel = decode_heatmap_point_xy(heatmap, method="argmax", heatmap_size=56, input_size=224)

    assert x_pixel == pytest.approx(112.0, abs=3.0)
    assert y_pixel == pytest.approx(56.0, abs=3.0)


@pytest.mark.parametrize(
    ("method", "window_size"),
    [
        ("argmax", 3),
        ("softargmax", 3),
        ("local_window_softargmax", 5),
        ("peak_weighted_centroid", 5),
    ],
)
def test_decode_heatmap_point_xy_preserves_center_peak(method: str, window_size: int) -> None:
    """All decoders should map the center peak to the middle of crop space."""

    heatmap = _single_peak_heatmap(28, 28)

    row, col = decode_heatmap_point(heatmap, method=method, window_size=window_size)
    x_pixel, y_pixel = decode_heatmap_point_xy(
        heatmap,
        method=method,
        window_size=window_size,
        heatmap_size=56,
        input_size=224,
    )

    assert row == pytest.approx(28.0, abs=0.25)
    assert col == pytest.approx(28.0, abs=0.25)
    assert x_pixel == pytest.approx(112.0, abs=3.0)
    assert y_pixel == pytest.approx(112.0, abs=3.0)


@pytest.mark.parametrize(
    ("method", "window_size"),
    [
        ("argmax", 3),
        ("softargmax", 3),
        ("local_window_softargmax", 5),
        ("peak_weighted_centroid", 5),
    ],
)
def test_decode_heatmap_point_xy_preserves_off_center_peak(method: str, window_size: int) -> None:
    """All decoders should preserve a known off-center peak and x/y ordering."""

    heatmap = _single_peak_heatmap(14, 42)
    expected_x = 42.0 * 223.0 / 55.0
    expected_y = 14.0 * 223.0 / 55.0

    row, col = decode_heatmap_point(heatmap, method=method, window_size=window_size)
    x_pixel, y_pixel = decode_heatmap_point_xy(
        heatmap,
        method=method,
        window_size=window_size,
        heatmap_size=56,
        input_size=224,
    )

    assert row == pytest.approx(14.0, abs=0.25)
    assert col == pytest.approx(42.0, abs=0.25)
    assert x_pixel == pytest.approx(expected_x, abs=1.0)
    assert y_pixel == pytest.approx(expected_y, abs=1.0)
