#!/usr/bin/env python3
"""Heatmap utilities for geometry-based gauge reading."""

from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation and decoding."""
    heatmap_height: int = 56
    heatmap_width: int = 56
    input_height: int = 224
    input_width: int = 224
    sigma_pixels: float = 2.5
    method: str = "softargmax"


def make_gaussian_heatmap(height, width, x_normalized, y_normalized, sigma_pixels):
    """Generate a 2D Gaussian heatmap with peak at normalized coordinate."""
    if not (0.0 <= x_normalized <= 1.0):
        raise ValueError(f"x_normalized must be in [0, 1], got {x_normalized}")
    if not (0.0 <= y_normalized <= 1.0):
        raise ValueError(f"y_normalized must be in [0, 1], got {y_normalized}")
    if sigma_pixels <= 0:
        raise ValueError(f"sigma_pixels must be positive, got {sigma_pixels}")
    center_x = x_normalized * (width - 1)
    center_y = y_normalized * (height - 1)
    y_coords, x_coords = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    dx = x_coords - center_x
    dy = y_coords - center_y
    squared_distance = dx**2 + dy**2
    heatmap = np.exp(-squared_distance / (2.0 * sigma_pixels**2))
    return heatmap


def argmax_2d(heatmap):
    """Find the row col coordinates of the maximum value."""
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap must be 2D, got {heatmap.ndim}D")
    if heatmap.size == 0:
        raise ValueError("Heatmap cannot be empty")
    flat_idx = np.argmax(heatmap)
    row = flat_idx // heatmap.shape[1]
    col = flat_idx % heatmap.shape[1]
    return (row, col)


def softargmax_2d(heatmap):
    """Compute the soft-argmax expected value of a 2D heatmap."""
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap must be 2D, got {heatmap.ndim}D")
    if heatmap.size == 0:
        raise ValueError("Heatmap cannot be empty")
    heatmap_sum = np.sum(heatmap)
    if heatmap_sum <= 0:
        raise ValueError("Heatmap must have positive values for softargmax")
    normalized_heatmap = heatmap / heatmap_sum
    height, width = heatmap.shape
    y_coords, x_coords = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    expected_row = np.sum(normalized_heatmap * y_coords)
    expected_col = np.sum(normalized_heatmap * x_coords)
    return (expected_row, expected_col)


def normalized_point_from_heatmap(heatmap, method="softargmax", height=None, width=None):
    """Extract normalized x y coordinates from a heatmap."""
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap must be 2D, got {heatmap.ndim}D")
    h = height if height is not None else heatmap.shape[0]
    w = width if width is not None else heatmap.shape[1]
    if heatmap.shape != (h, w):
        raise ValueError("Heatmap shape mismatch")
    if method == "softargmax":
        row, col = softargmax_2d(heatmap)
    elif method == "argmax":
        row, col = argmax_2d(heatmap)
    else:
        raise ValueError("Unknown method, use softargmax or argmax")
    x_normalized = col / (w - 1) if w > 1 else 0.5
    y_normalized = row / (h - 1) if h > 1 else 0.5
    return (x_normalized, y_normalized)


def pixel_mae_between_points(pred_x, pred_y, true_x, true_y, image_width, image_height):
    """Compute MAE in pixels between predicted and true points."""
    mae_x = abs(pred_x - true_x)
    mae_y = abs(pred_y - true_y)
    return (mae_x + mae_y) / 2.0


def generate_center_tip_heatmaps(center_x_norm, center_y_norm, tip_x_norm, tip_y_norm, config=None):
    """Generate both center and tip heatmaps from normalized coordinates."""
    if config is None:
        config = HeatmapConfig()
    center_heatmap = make_gaussian_heatmap(height=config.heatmap_height, width=config.heatmap_width, x_normalized=center_x_norm, y_normalized=center_y_norm, sigma_pixels=config.sigma_pixels)
    tip_heatmap = make_gaussian_heatmap(height=config.heatmap_height, width=config.heatmap_width, x_normalized=tip_x_norm, y_normalized=tip_y_norm, sigma_pixels=config.sigma_pixels)
    return (center_heatmap, tip_heatmap)


def decode_heatmap_to_pixel_coords(heatmap, method="softargmax", input_size=224):
    """Decode a heatmap to pixel coordinates in the input crop space."""
    h, w = heatmap.shape
    x_norm, y_norm = normalized_point_from_heatmap(heatmap, method=method)
    x_pixel = x_norm * (input_size - 1)
    y_pixel = y_norm * (input_size - 1)
    return (x_pixel, y_pixel)
