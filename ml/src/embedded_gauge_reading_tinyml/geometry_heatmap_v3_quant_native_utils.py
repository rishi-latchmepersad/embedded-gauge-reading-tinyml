"""Small tensor helpers used by the quant-native geometry heatmap trainer."""

from __future__ import annotations

import math

import tensorflow as tf


def normalize_scalar_tf(values: tf.Tensor, *, minimum: float, maximum: float) -> tf.Tensor:
    """Map a scalar tensor into the [0, 1] range."""

    values = tf.cast(values, tf.float32)
    return tf.clip_by_value((values - minimum) / max(maximum - minimum, 1e-6), 0.0, 1.0)


def linear_temperature_from_angle_tf(
    angle_degrees: tf.Tensor,
    *,
    cold_angle_degrees: float = 135.0,
    sweep_degrees: float = 270.0,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> tf.Tensor:
    """Convert an angle to a temperature on the gauge's linear scale."""

    angle = tf.cast(angle_degrees, tf.float32)
    normalized = tf.math.floormod(angle - cold_angle_degrees, 360.0) / max(sweep_degrees, 1e-6)
    normalized = tf.clip_by_value(normalized, 0.0, 1.0)
    return value_min + (value_max - value_min) * normalized


def angle_degrees_from_center_to_tip_tf(
    center_x: tf.Tensor,
    center_y: tf.Tensor,
    tip_x: tf.Tensor,
    tip_y: tf.Tensor,
) -> tf.Tensor:
    """Compute the needle angle in degrees from center to tip coordinates."""

    dx = tf.cast(tip_x, tf.float32) - tf.cast(center_x, tf.float32)
    dy = tf.cast(tip_y, tf.float32) - tf.cast(center_y, tf.float32)
    angle = tf.math.atan2(-dy, dx) * (180.0 / math.pi)
    return tf.math.floormod(angle + 360.0, 360.0)


def normalized_softargmax_coordinates_tf(heatmap: tf.Tensor) -> tf.Tensor:
    """Decode a heatmap into normalized [0, 1] x/y coordinates."""

    hm = tf.cast(tf.squeeze(heatmap), tf.float32)
    hm_shape = tf.shape(hm)
    height = tf.cast(hm_shape[0], tf.float32)
    width = tf.cast(hm_shape[1], tf.float32)
    flat = tf.reshape(hm, [-1])
    weights = tf.nn.softmax(flat)
    ys = tf.repeat(tf.range(hm_shape[0], dtype=tf.float32), hm_shape[1])
    xs = tf.tile(tf.range(hm_shape[1], dtype=tf.float32), [hm_shape[0]])
    x = tf.reduce_sum(weights * xs) / tf.maximum(width - 1.0, 1.0)
    y = tf.reduce_sum(weights * ys) / tf.maximum(height - 1.0, 1.0)
    return tf.stack([x, y], axis=0)


def temperature_from_coords_tf(
    center_x: tf.Tensor,
    center_y: tf.Tensor,
    tip_x: tf.Tensor,
    tip_y: tf.Tensor,
    *,
    cold_angle_degrees: float = 135.0,
    sweep_degrees: float = 270.0,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> tf.Tensor:
    """Convert center/tip coordinates directly into a temperature tensor."""

    angle = angle_degrees_from_center_to_tip_tf(center_x, center_y, tip_x, tip_y)
    return linear_temperature_from_angle_tf(
        angle,
        cold_angle_degrees=cold_angle_degrees,
        sweep_degrees=sweep_degrees,
        value_min=value_min,
        value_max=value_max,
    )


def circular_angle_loss_tf(predicted: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    """A small circular-distance loss for angles in degrees."""

    predicted = tf.cast(predicted, tf.float32)
    target = tf.cast(target, tf.float32)
    delta = tf.math.floormod(predicted - target + 180.0, 360.0) - 180.0
    return tf.abs(delta) / 180.0


def normalized_temperature_huber_loss_tf(
    predicted: tf.Tensor,
    target: tf.Tensor,
    *,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> tf.Tensor:
    """Huber loss on normalized temperature predictions."""

    pred_norm = normalize_scalar_tf(predicted, minimum=value_min, maximum=value_max)
    target_norm = normalize_scalar_tf(target, minimum=value_min, maximum=value_max)
    return tf.keras.losses.huber(target_norm, pred_norm)

