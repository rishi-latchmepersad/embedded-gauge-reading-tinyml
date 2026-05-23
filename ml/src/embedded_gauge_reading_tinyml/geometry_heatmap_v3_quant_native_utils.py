"""Pure helpers for geometry_heatmap_v3 quantization-native training.

These helpers keep the v3 training objective aligned with the deployment math:
soft-argmax decoding, circular angle error, and calibrated temperature mapping.
"""

from __future__ import annotations

from typing import Final

import tensorflow as tf


_EPSILON: Final[float] = 1e-7
_PI: Final[float] = 3.141592653589793


def _prepare_heatmap_batch(heatmaps: tf.Tensor) -> tf.Tensor:
    """Return heatmaps as a float32 tensor with shape (batch, height, width)."""

    tensor = tf.cast(tf.convert_to_tensor(heatmaps), tf.float32)
    rank = tensor.shape.rank

    if rank == 2:
        tensor = tf.expand_dims(tensor, axis=0)
    elif rank == 3 and tensor.shape[-1] == 1:
        tensor = tf.squeeze(tensor, axis=-1)
        tensor = tf.expand_dims(tensor, axis=0)
    elif rank == 4 and tensor.shape[-1] == 1:
        tensor = tf.squeeze(tensor, axis=-1)
    elif rank not in (3, 4):
        raise ValueError("Heatmaps must have rank 2, 3, or 4 with a singleton channel dimension.")

    if tensor.shape.rank != 3:
        raise ValueError("Heatmaps must resolve to shape (batch, height, width).")

    return tensor


def softargmax_coordinates_tf(heatmaps: tf.Tensor) -> tf.Tensor:
    """Compute soft-argmax x/y coordinates for a batch of heatmaps."""

    batch_heatmaps = _prepare_heatmap_batch(heatmaps)
    height = tf.shape(batch_heatmaps)[1]
    width = tf.shape(batch_heatmaps)[2]
    x_coords = tf.cast(tf.range(width), tf.float32)[tf.newaxis, tf.newaxis, :]
    y_coords = tf.cast(tf.range(height), tf.float32)[tf.newaxis, :, tf.newaxis]

    heatmap_sum = tf.reduce_sum(batch_heatmaps, axis=[1, 2], keepdims=True)
    heatmap_sum = tf.maximum(heatmap_sum, _EPSILON)
    normalized = batch_heatmaps / heatmap_sum

    expected_x = tf.reduce_sum(normalized * x_coords, axis=[1, 2])
    expected_y = tf.reduce_sum(normalized * y_coords, axis=[1, 2])
    return tf.stack([expected_x, expected_y], axis=-1)


def normalized_softargmax_coordinates_tf(heatmaps: tf.Tensor) -> tf.Tensor:
    """Return soft-argmax coordinates normalized to the [0, 1] interval."""

    batch_heatmaps = _prepare_heatmap_batch(heatmaps)
    coords = softargmax_coordinates_tf(batch_heatmaps)
    height = tf.cast(tf.maximum(tf.shape(batch_heatmaps)[1] - 1, 1), tf.float32)
    width = tf.cast(tf.maximum(tf.shape(batch_heatmaps)[2] - 1, 1), tf.float32)
    x_norm = coords[:, 0] / width
    y_norm = coords[:, 1] / height
    return tf.stack([x_norm, y_norm], axis=-1)


def angle_degrees_from_center_to_tip_tf(
    center_x_pixels: tf.Tensor,
    center_y_pixels: tf.Tensor,
    tip_x_pixels: tf.Tensor,
    tip_y_pixels: tf.Tensor,
) -> tf.Tensor:
    """Compute the image-space angle from center to tip in degrees."""

    dx = tf.cast(tip_x_pixels, tf.float32) - tf.cast(center_x_pixels, tf.float32)
    dy = tf.cast(tip_y_pixels, tf.float32) - tf.cast(center_y_pixels, tf.float32)
    angle_radians = tf.math.atan2(dy, dx)
    angle_degrees = angle_radians * (180.0 / tf.constant(_PI, dtype=tf.float32))
    return tf.where(angle_degrees < 0.0, angle_degrees + 360.0, angle_degrees)


def circular_angle_difference_radians_tf(
    predicted_angle_degrees: tf.Tensor,
    true_angle_degrees: tf.Tensor,
) -> tf.Tensor:
    """Compute the signed shortest angular difference in radians."""

    predicted = tf.cast(predicted_angle_degrees, tf.float32) * (tf.constant(_PI, dtype=tf.float32) / 180.0)
    true = tf.cast(true_angle_degrees, tf.float32) * (tf.constant(_PI, dtype=tf.float32) / 180.0)
    delta = predicted - true
    return tf.math.atan2(tf.sin(delta), tf.cos(delta))


def circular_angle_difference_degrees_tf(
    predicted_angle_degrees: tf.Tensor,
    true_angle_degrees: tf.Tensor,
) -> tf.Tensor:
    """Compute the shortest angular distance between two tensors of angles."""

    predicted = tf.math.floormod(tf.cast(predicted_angle_degrees, tf.float32), 360.0)
    true = tf.math.floormod(tf.cast(true_angle_degrees, tf.float32), 360.0)
    diff = tf.abs(predicted - true)
    return tf.where(diff > 180.0, 360.0 - diff, diff)


def circular_angle_loss_tf(
    predicted_angle_degrees: tf.Tensor,
    true_angle_degrees: tf.Tensor,
) -> tf.Tensor:
    """Return a bounded circular loss in the [0, 1] interval."""

    delta_radians = circular_angle_difference_radians_tf(predicted_angle_degrees, true_angle_degrees)
    return tf.reduce_mean(0.5 * (1.0 - tf.cos(delta_radians)))


def normalize_scalar_tf(value: tf.Tensor, *, minimum: float, maximum: float) -> tf.Tensor:
    """Normalize a scalar tensor to [0, 1] using a fixed physical range."""

    span = tf.maximum(tf.cast(maximum, tf.float32) - tf.cast(minimum, tf.float32), _EPSILON)
    normalized = (tf.cast(value, tf.float32) - tf.cast(minimum, tf.float32)) / span
    return tf.clip_by_value(normalized, 0.0, 1.0)


def normalized_temperature_huber_loss_tf(
    predicted_temperature_c: tf.Tensor,
    true_temperature_c: tf.Tensor,
    *,
    minimum_celsius: float,
    maximum_celsius: float,
    delta: float = 0.05,
) -> tf.Tensor:
    """Compute a bounded Huber loss on normalized temperature values."""

    pred_norm = normalize_scalar_tf(predicted_temperature_c, minimum=minimum_celsius, maximum=maximum_celsius)
    true_norm = normalize_scalar_tf(true_temperature_c, minimum=minimum_celsius, maximum=maximum_celsius)
    abs_error = tf.abs(pred_norm - true_norm)
    delta_tensor = tf.cast(delta, tf.float32)
    loss = tf.where(
        abs_error <= delta_tensor,
        0.5 * tf.square(abs_error),
        delta_tensor * (abs_error - 0.5 * delta_tensor),
    )
    return tf.reduce_mean(loss)


def linear_temperature_from_angle_tf(
    angle_degrees: tf.Tensor,
    *,
    slope: float,
    intercept: float,
    cold_angle_degrees: float = 135.0,
) -> tf.Tensor:
    """Apply a linear calibrated temperature mapping to a needle angle."""

    angle = tf.cast(angle_degrees, tf.float32)
    delta = tf.math.floormod(angle - float(cold_angle_degrees), 360.0)
    return tf.cast(slope, tf.float32) * delta + tf.cast(intercept, tf.float32)


def temperature_from_coords_tf(
    center_x_pixels: tf.Tensor,
    center_y_pixels: tf.Tensor,
    tip_x_pixels: tf.Tensor,
    tip_y_pixels: tf.Tensor,
    *,
    slope: float,
    intercept: float,
    cold_angle_degrees: float = 135.0,
) -> tf.Tensor:
    """Predict calibrated temperature from predicted center/tip coordinates."""

    angle = angle_degrees_from_center_to_tip_tf(center_x_pixels, center_y_pixels, tip_x_pixels, tip_y_pixels)
    return linear_temperature_from_angle_tf(angle, slope=slope, intercept=intercept, cold_angle_degrees=cold_angle_degrees)
