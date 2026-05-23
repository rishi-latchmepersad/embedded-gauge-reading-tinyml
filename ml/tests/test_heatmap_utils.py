#!/usr/bin/env python3
"""Tests for heatmap utilities."""

import pytest
import numpy as np
from embedded_gauge_reading_tinyml.heatmap_utils import (
    HeatmapConfig,
    make_gaussian_heatmap,
    argmax_2d,
    softargmax_2d,
    normalized_point_from_heatmap,
    pixel_mae_between_points,
    generate_center_tip_heatmaps,
    decode_heatmap_to_pixel_coords,
)


class TestMakeGaussianHeatmap:
    """Tests for Gaussian heatmap generation."""

    def test_peak_at_expected_coordinate(self):
        """Gaussian peak should land near the expected coordinate."""
        heatmap = make_gaussian_heatmap(
            height=56,
            width=56,
            x_normalized=0.5,
            y_normalized=0.5,
            sigma_pixels=2.5,
        )
        row, col = argmax_2d(heatmap)
        assert row == 27 or row == 28
        assert col == 27 or col == 28
        # Peak value may be slightly < 1.0 due to discrete grid
        assert heatmap[row, col] > 0.95

    def test_peak_at_corner(self):
        """Test heatmap with peak at corner."""
        heatmap = make_gaussian_heatmap(
            height=56,
            width=56,
            x_normalized=0.0,
            y_normalized=0.0,
            sigma_pixels=2.5,
        )
        row, col = argmax_2d(heatmap)
        assert row == 0
        assert col == 0

    def test_invalid_x_normalized(self):
        """Should reject x_normalized outside [0, 1]."""
        with pytest.raises(ValueError):
            make_gaussian_heatmap(56, 56, -0.1, 0.5, 2.5)
        with pytest.raises(ValueError):
            make_gaussian_heatmap(56, 56, 1.1, 0.5, 2.5)

    def test_invalid_sigma(self):
        """Should reject non-positive sigma."""
        with pytest.raises(ValueError):
            make_gaussian_heatmap(56, 56, 0.5, 0.5, 0)
        with pytest.raises(ValueError):
            make_gaussian_heatmap(56, 56, 0.5, 0.5, -1.0)


class TestArgmax2d:
    """Tests for argmax operation."""

    def test_argmax_recovers_peak(self):
        """Argmax should recover known peak location."""
        heatmap = np.zeros((10, 10))
        heatmap[3, 7] = 1.0
        row, col = argmax_2d(heatmap)
        assert row == 3
        assert col == 7

    def test_argmax_invalid_input(self):
        """Should reject non2D input."""
        with pytest.raises(ValueError):
            argmax_2d(np.zeros((10,)))
        with pytest.raises(ValueError):
            argmax_2d(np.zeros((5, 5, 3)))


class TestSoftargmax2d:
    """Tests for softargmax operation."""

    def test_softargmax_on_single_peak(self):
        """Softargmax on single-pixel peak should recover location."""
        heatmap = np.zeros((10, 10))
        heatmap[5, 5] = 1.0
        row, col = softargmax_2d(heatmap)
        assert row == pytest.approx(5.0)
        assert col == pytest.approx(5.0)

    def test_softargmax_on_gaussian(self):
        """Softargmax on Gaussian should be near center."""
        heatmap = make_gaussian_heatmap(56, 56, 0.3, 0.7, 3.0)
        row, col = softargmax_2d(heatmap)
        expected_row = 0.7 * 55
        expected_col = 0.3 * 55
        assert abs(row - expected_row) < 1.0
        assert abs(col - expected_col) < 1.0

    def test_softargmax_all_zeros(self):
        """Should reject all-zero heatmap."""
        with pytest.raises(ValueError):
            softargmax_2d(np.zeros((10, 10)))


class TestNormalizedPointFromHeatmap:
    """Tests for normalized coordinate extraction."""

    def test_normalized_coords_correct(self):
        """Normalized coordinate conversion should be correct."""
        heatmap = make_gaussian_heatmap(56, 56, 0.25, 0.75, 2.0)
        x_norm, y_norm = normalized_point_from_heatmap(heatmap, method="argmax")
        assert abs(x_norm - 0.25) < 0.05
        assert abs(y_norm - 0.75) < 0.05

    def test_softargmax_method(self):
        """Softargmax should work correctly."""
        heatmap = make_gaussian_heatmap(56, 56, 0.5, 0.5, 2.0)
        x_norm, y_norm = normalized_point_from_heatmap(heatmap, method="softargmax")
        assert abs(x_norm - 0.5) < 0.05
        assert abs(y_norm - 0.5) < 0.05

    def test_invalid_method(self):
        """Should reject invalid method."""
        heatmap = np.ones((10, 10))
        with pytest.raises(ValueError):
            normalized_point_from_heatmap(heatmap, method="invalid")


class TestPixelMaeBetweenPoints:
    """Tests for pixel MAE computation."""

    def test_mae_computation(self):
        """MAE should be average of x and y errors."""
        mae = pixel_mae_between_points(10.0, 20.0, 12.0, 18.0, 224, 224)
        assert mae == pytest.approx(2.0)

    def test_mae_zero(self):
        """MAE should be zero for identical points."""
        mae = pixel_mae_between_points(10.0, 20.0, 10.0, 20.0, 224, 224)
        assert mae == pytest.approx(0.0)


class TestGenerateCenterTipHeatmaps:
    """Tests for generating both heatmaps."""

    def test_both_heatmaps_generated(self):
        """Should generate both center and tip heatmaps."""
        center, tip = generate_center_tip_heatmaps(0.5, 0.5, 0.3, 0.7)
        assert center.shape == (56, 56)
        assert tip.shape == (56, 56)
        assert center.max() > 0.95
        assert tip.max() > 0.95


class TestDecodeHeatmapToPixelCoords:
    """Tests for decoding to pixel coordinates."""

    def test_decode_to_pixel_coords(self):
        """Should decode to correct pixel coordinates."""
        heatmap = make_gaussian_heatmap(56, 56, 0.5, 0.5, 2.0)
        x_pixel, y_pixel = decode_heatmap_to_pixel_coords(heatmap, input_size=224)
        assert abs(x_pixel - 111.5) < 5.0
        assert abs(y_pixel - 111.5) < 5.0
