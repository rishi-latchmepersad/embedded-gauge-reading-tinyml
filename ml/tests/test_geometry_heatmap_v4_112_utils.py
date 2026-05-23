"""Tests for the geometry_heatmap_v4_112 quantization-native helpers."""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf

from embedded_gauge_reading_tinyml.geometry_heatmap_qat_utils import fake_quantize_01_tensor
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import decode_heatmap_point_xy
from embedded_gauge_reading_tinyml.geometry_heatmap_v3_quant_native_utils import (
    circular_angle_loss_tf,
    normalized_softargmax_coordinates_tf,
    normalized_temperature_huber_loss_tf,
    softargmax_coordinates_tf,
)
from embedded_gauge_reading_tinyml.heatmap_utils import make_gaussian_heatmap


def test_112_center_heatmap_decodes_to_the_middle_of_the_crop() -> None:
    """A centered 112x112 Gaussian should decode near the 224px crop center."""

    heatmap = make_gaussian_heatmap(
        height=112,
        width=112,
        x_normalized=0.5,
        y_normalized=0.5,
        sigma_pixels=1.5,
    )
    x_pixel, y_pixel = decode_heatmap_point_xy(heatmap, method="softargmax", heatmap_size=112, input_size=224)
    assert math.isclose(float(x_pixel), 112.0, abs_tol=1.5)
    assert math.isclose(float(y_pixel), 112.0, abs_tol=1.5)


def test_112_off_center_heatmap_decodes_with_explicit_x_y_order() -> None:
    """An off-center peak should preserve x/y ordering after 112->224 scaling."""

    x_index = 84
    y_index = 28
    heatmap = np.zeros((112, 112), dtype=np.float32)
    heatmap[y_index, x_index] = 1.0

    argmax_x, argmax_y = decode_heatmap_point_xy(heatmap, method="argmax", heatmap_size=112, input_size=224)
    soft_x, soft_y = decode_heatmap_point_xy(heatmap, method="softargmax", heatmap_size=112, input_size=224)
    expected_x = float(x_index) * 223.0 / 111.0
    expected_y = float(y_index) * 223.0 / 111.0

    assert math.isclose(float(argmax_x), expected_x, abs_tol=1e-5)
    assert math.isclose(float(argmax_y), expected_y, abs_tol=1e-5)
    assert math.isclose(float(soft_x), expected_x, abs_tol=1e-3)
    assert math.isclose(float(soft_y), expected_y, abs_tol=1e-3)


def test_112_fake_quant_and_losses_remain_finite() -> None:
    """Quantization-aware helper losses should stay finite on synthetic 112 heatmaps."""

    heatmap = make_gaussian_heatmap(
        height=112,
        width=112,
        x_normalized=0.62,
        y_normalized=0.37,
        sigma_pixels=2.0,
    ).astype(np.float32)
    batch = tf.convert_to_tensor(heatmap[None, ..., None], dtype=tf.float32)
    quantized = fake_quantize_01_tensor(batch)
    coords = softargmax_coordinates_tf(quantized)
    coords_norm = normalized_softargmax_coordinates_tf(quantized)

    assert np.isfinite(quantized.numpy()).all()
    assert np.isfinite(coords.numpy()).all()
    assert np.isfinite(coords_norm.numpy()).all()

    angle_loss = circular_angle_loss_tf(tf.constant([359.0], dtype=tf.float32), tf.constant([1.0], dtype=tf.float32))
    temp_loss = normalized_temperature_huber_loss_tf(
        tf.constant([55.0], dtype=tf.float32),
        tf.constant([50.0], dtype=tf.float32),
        minimum_celsius=20.0,
        maximum_celsius=60.0,
    )
    assert np.isfinite(angle_loss.numpy()).all()
    assert np.isfinite(temp_loss.numpy()).all()
    assert float(angle_loss.numpy()) >= 0.0
    assert float(temp_loss.numpy()) >= 0.0

