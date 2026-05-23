"""Tests for the geometry_heatmap_v3 quantization-native helpers."""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf

from embedded_gauge_reading_tinyml.geometry_heatmap_v3_quant_native_utils import (
    angle_degrees_from_center_to_tip_tf,
    circular_angle_difference_degrees_tf,
    circular_angle_loss_tf,
    linear_temperature_from_angle_tf,
    normalize_scalar_tf,
    normalized_softargmax_coordinates_tf,
    normalized_temperature_huber_loss_tf,
    softargmax_coordinates_tf,
    temperature_from_coords_tf,
)


def test_softargmax_coordinates_tf_tracks_a_center_peak() -> None:
    """A single central peak should decode to the middle heatmap coordinate."""

    heatmap = np.zeros((1, 3, 3), dtype=np.float32)
    heatmap[0, 1, 1] = 1.0
    coords = softargmax_coordinates_tf(tf.convert_to_tensor(heatmap)).numpy()
    assert coords.shape == (1, 2)
    assert math.isclose(float(coords[0, 0]), 1.0, abs_tol=1e-5)
    assert math.isclose(float(coords[0, 1]), 1.0, abs_tol=1e-5)


def test_normalized_softargmax_coordinates_tf_tracks_an_off_center_peak() -> None:
    """Normalized coordinates should preserve x/y ordering and scale."""

    heatmap = np.zeros((1, 5, 5), dtype=np.float32)
    heatmap[0, 1, 4] = 1.0
    coords = normalized_softargmax_coordinates_tf(tf.convert_to_tensor(heatmap)).numpy()
    assert coords.shape == (1, 2)
    assert math.isclose(float(coords[0, 0]), 1.0, abs_tol=1e-5)
    assert math.isclose(float(coords[0, 1]), 0.25, abs_tol=1e-5)


def test_angle_helpers_handle_wraparound() -> None:
    """Angles should preserve image-space ordering and circular error."""

    angle = angle_degrees_from_center_to_tip_tf(
        tf.constant([0.0]),
        tf.constant([0.0]),
        tf.constant([1.0]),
        tf.constant([0.0]),
    ).numpy()[0]
    assert math.isclose(float(angle), 0.0, abs_tol=1e-6)

    diff = circular_angle_difference_degrees_tf(
        tf.constant([359.0]),
        tf.constant([1.0]),
    ).numpy()[0]
    assert math.isclose(float(diff), 2.0, abs_tol=1e-6)


def test_circular_angle_loss_tf_is_finite_near_wraparound() -> None:
    """A near-wraparound mismatch should remain finite and small."""

    loss = circular_angle_loss_tf(
        tf.constant([359.0], dtype=tf.float32),
        tf.constant([1.0], dtype=tf.float32),
    ).numpy()
    assert np.isfinite(loss).all()
    assert float(loss) < 0.001


def test_temperature_mapping_from_angle_is_linear() -> None:
    """The calibration helper should apply a simple affine transform."""

    temp = linear_temperature_from_angle_tf(
        tf.constant([135.0]),
        slope=0.5,
        intercept=-30.0,
        cold_angle_degrees=135.0,
    ).numpy()[0]
    assert math.isclose(float(temp), -30.0, abs_tol=1e-6)

    mapped = temperature_from_coords_tf(
        tf.constant([0.0]),
        tf.constant([0.0]),
        tf.constant([0.0]),
        tf.constant([1.0]),
        slope=0.5,
        intercept=-30.0,
        cold_angle_degrees=135.0,
    ).numpy()[0]
    assert isinstance(mapped, np.floating)


def test_normalize_scalar_tf_maps_physical_ranges_to_unit_interval() -> None:
    """Temperature-style normalization should stay within the unit interval."""

    normalized = normalize_scalar_tf(
        tf.constant([20.0, 60.0], dtype=tf.float32),
        minimum=20.0,
        maximum=60.0,
    ).numpy()
    assert np.allclose(normalized, np.asarray([0.0, 1.0], dtype=np.float32))


def test_normalized_temperature_huber_loss_tf_is_finite() -> None:
    """A normalized temperature loss should stay finite even for a moderate miss."""

    loss = normalized_temperature_huber_loss_tf(
        tf.constant([55.0], dtype=tf.float32),
        tf.constant([50.0], dtype=tf.float32),
        minimum_celsius=20.0,
        maximum_celsius=60.0,
    ).numpy()
    assert np.isfinite(loss).all()
    assert float(loss) > 0.0
