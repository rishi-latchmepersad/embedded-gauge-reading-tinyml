"""Tests for the polar projection angle decode helpers."""

from __future__ import annotations

import numpy as np

from embedded_gauge_reading_tinyml.polar_projection import (
    angle_from_polar_prediction,
    needle_mask_from_polar,
)


def _circular_distance_deg(a_deg: float, b_deg: float) -> float:
    """Return the smallest wrap-around distance between two angles."""

    delta = abs(a_deg - b_deg) % 360.0
    return min(delta, 360.0 - delta)


def test_angle_from_polar_prediction_recovers_a_noisy_peak() -> None:
    """The robust decode should stay close to the true angle under light noise."""

    polar = np.zeros((160, 160, 3), dtype=np.float32)
    mask = needle_mask_from_polar(polar, needle_angle_deg=123.0, mask_sigma=2.0)
    noise = np.random.default_rng(123).normal(loc=0.0, scale=0.02, size=mask.shape).astype(np.float32)
    noisy_mask = np.clip(mask + noise, 0.0, 1.0)

    decoded_angle = angle_from_polar_prediction(noisy_mask)
    assert _circular_distance_deg(decoded_angle, 123.0) < 1.5


def test_angle_from_polar_prediction_handles_wraparound_near_zero() -> None:
    """Wrap-around angles near 0/360 degrees should decode cleanly."""

    polar = np.zeros((160, 160, 3), dtype=np.float32)
    mask = needle_mask_from_polar(polar, needle_angle_deg=359.0, mask_sigma=2.0)

    decoded_angle = angle_from_polar_prediction(mask)
    assert _circular_distance_deg(decoded_angle, 359.0) < 1.5
