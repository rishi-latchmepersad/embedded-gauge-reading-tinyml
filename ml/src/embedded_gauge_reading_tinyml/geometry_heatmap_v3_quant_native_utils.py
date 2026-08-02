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
    slope: float | None = None,
    intercept: float | None = None,
) -> tf.Tensor:
    """Convert an angle to a temperature on the gauge's linear scale."""

    angle = tf.cast(angle_degrees, tf.float32)
    normalized = tf.math.floormod(angle - cold_angle_degrees, 360.0) / max(sweep_degrees, 1e-6)
    normalized = tf.clip_by_value(normalized, 0.0, 1.0)
    if slope is not None or intercept is not None:
        # why: retain the old affine-test spelling while the production API
        # remains expressed in physical minimum/maximum values.
        return (value_min if slope is None else slope) * normalized + (
            value_max if intercept is None else intercept
        )
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


def _batched_heatmap_weights(heatmap: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return sharp positive weights and the heatmap height and width."""

    values = tf.cast(heatmap, tf.float32)
    if values.shape.rank == 4:
        values = tf.squeeze(values, axis=-1)
    if values.shape.rank == 2:
        values = values[None, ...]
    shape = tf.shape(values)
    height = tf.cast(shape[1], tf.float32)
    width = tf.cast(shape[2], tf.float32)
    # why: a fourth-power positive contrast keeps a quantized heatmap's peak
    # local instead of averaging it toward the background floor.
    weights = tf.pow(tf.nn.relu(values - tf.reduce_min(values, axis=[1, 2], keepdims=True)), 4.0)
    weights /= tf.maximum(tf.reduce_sum(weights, axis=[1, 2], keepdims=True), 1e-6)
    return weights, height, width


def softargmax_coordinates_tf(heatmap: tf.Tensor) -> tf.Tensor:
    """Decode batched heatmaps into raw ``(x, y)`` pixel coordinates."""

    weights, height, width = _batched_heatmap_weights(heatmap)
    shape = tf.shape(weights)
    ys = tf.cast(tf.range(shape[1]), tf.float32)[None, :, None]
    xs = tf.cast(tf.range(shape[2]), tf.float32)[None, None, :]
    x = tf.reduce_sum(weights * xs, axis=[1, 2])
    y = tf.reduce_sum(weights * ys, axis=[1, 2])
    return tf.stack([x, y], axis=1)


def normalized_softargmax_coordinates_tf(heatmap: tf.Tensor) -> tf.Tensor:
    """Decode batched heatmaps into normalized ``(x, y)`` coordinates."""

    coords = softargmax_coordinates_tf(heatmap)
    _, height, width = _batched_heatmap_weights(heatmap)
    scale = tf.stack([tf.maximum(width - 1.0, 1.0), tf.maximum(height - 1.0, 1.0)])
    return coords / scale[None, :]


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
    slope: float | None = None,
    intercept: float | None = None,
) -> tf.Tensor:
    """Convert center/tip coordinates directly into a temperature tensor."""

    angle = angle_degrees_from_center_to_tip_tf(center_x, center_y, tip_x, tip_y)
    return linear_temperature_from_angle_tf(
        angle,
        cold_angle_degrees=cold_angle_degrees,
        sweep_degrees=sweep_degrees,
        value_min=value_min,
        value_max=value_max,
        slope=slope,
        intercept=intercept,
    )


def circular_angle_difference_degrees_tf(
    predicted: tf.Tensor,
    target: tf.Tensor,
) -> tf.Tensor:
    """Return the absolute shortest angular difference in degrees."""

    predicted = tf.cast(predicted, tf.float32)
    target = tf.cast(target, tf.float32)
    delta = tf.math.floormod(predicted - target + 180.0, 360.0) - 180.0
    return tf.abs(delta)


def circular_angle_loss_tf(predicted: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    """A small circular-distance loss for angles in degrees."""

    predicted = tf.cast(predicted, tf.float32)
    target = tf.cast(target, tf.float32)
    delta = tf.math.floormod(predicted - target + 180.0, 360.0) - 180.0
    return tf.square(delta / 180.0)


def normalized_temperature_huber_loss_tf(
    predicted: tf.Tensor,
    target: tf.Tensor,
    *,
    value_min: float = -30.0,
    value_max: float = 50.0,
    minimum_celsius: float | None = None,
    maximum_celsius: float | None = None,
) -> tf.Tensor:
    """Huber loss on normalized temperature predictions."""

    minimum = value_min if minimum_celsius is None else minimum_celsius
    maximum = value_max if maximum_celsius is None else maximum_celsius
    pred_norm = normalize_scalar_tf(predicted, minimum=minimum, maximum=maximum)
    target_norm = normalize_scalar_tf(target, minimum=minimum, maximum=maximum)
    return tf.keras.losses.huber(target_norm, pred_norm)
