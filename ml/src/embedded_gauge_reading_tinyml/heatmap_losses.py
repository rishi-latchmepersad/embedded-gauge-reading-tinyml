"""Losses and metrics for geometry heatmap training."""

from __future__ import annotations

from typing import Final

import tensorflow as tf
from tensorflow import keras


_EPSILON: Final[float] = 1e-7
_DEFAULT_POSITIVE_THRESHOLD: Final[float] = 0.1
_DEFAULT_POSITIVE_WEIGHT: Final[float] = 10.0
_DEFAULT_BACKGROUND_WEIGHT: Final[float] = 0.2
_DEFAULT_COORDINATE_WEIGHT: Final[float] = 0.5
_DEFAULT_FOCAL_ALPHA: Final[float] = 0.25
_DEFAULT_FOCAL_GAMMA: Final[float] = 2.0


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
        raise ValueError(
            "Heatmaps must have rank 2, 3, or 4 with a singleton channel dimension."
        )

    if tensor.shape.rank != 3:
        raise ValueError("Heatmaps must resolve to shape (batch, height, width).")

    return tensor


def _build_pixel_grids(height: tf.Tensor, width: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Build broadcastable x/y coordinate grids in heatmap pixel space."""

    x_coords = tf.cast(tf.range(width), tf.float32)[tf.newaxis, tf.newaxis, :]
    y_coords = tf.cast(tf.range(height), tf.float32)[tf.newaxis, :, tf.newaxis]
    return x_coords, y_coords


def _softargmax_coordinates(heatmaps: tf.Tensor) -> tf.Tensor:
    """Compute expected x/y coordinates for each heatmap in pixel space."""

    batch_heatmaps = _prepare_heatmap_batch(heatmaps)
    height = tf.shape(batch_heatmaps)[1]
    width = tf.shape(batch_heatmaps)[2]
    x_coords, y_coords = _build_pixel_grids(height, width)

    heatmap_sum = tf.reduce_sum(batch_heatmaps, axis=[1, 2], keepdims=True)
    heatmap_sum = tf.maximum(heatmap_sum, _EPSILON)
    normalized = batch_heatmaps / heatmap_sum

    expected_x = tf.reduce_sum(normalized * x_coords, axis=[1, 2])
    expected_y = tf.reduce_sum(normalized * y_coords, axis=[1, 2])
    return tf.stack([expected_x, expected_y], axis=-1)


def _weighted_pixel_reduce(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    pixel_loss: tf.Tensor,
    *,
    positive_threshold: float = _DEFAULT_POSITIVE_THRESHOLD,
    positive_weight: float = _DEFAULT_POSITIVE_WEIGHT,
    background_weight: float = _DEFAULT_BACKGROUND_WEIGHT,
) -> tf.Tensor:
    """Apply foreground/background weights and reduce a per-pixel loss tensor."""

    true_heatmaps = _prepare_heatmap_batch(y_true)
    per_pixel_loss = tf.cast(pixel_loss, tf.float32)
    weights = tf.where(
        true_heatmaps > positive_threshold,
        tf.cast(positive_weight, tf.float32),
        tf.cast(background_weight, tf.float32),
    )
    weighted_loss = tf.reduce_sum(weights * per_pixel_loss, axis=[1, 2])
    weight_sum = tf.reduce_sum(weights, axis=[1, 2])
    reduced = weighted_loss / tf.maximum(weight_sum, _EPSILON)
    return tf.reduce_mean(reduced)


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def weighted_heatmap_mse_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    positive_threshold: float = _DEFAULT_POSITIVE_THRESHOLD,
    positive_weight: float = _DEFAULT_POSITIVE_WEIGHT,
    background_weight: float = _DEFAULT_BACKGROUND_WEIGHT,
) -> tf.Tensor:
    """Weighted MSE that emphasizes pixels near the target peak."""

    true_heatmaps = _prepare_heatmap_batch(y_true)
    pred_heatmaps = _prepare_heatmap_batch(y_pred)
    per_pixel_mse = tf.square(pred_heatmaps - true_heatmaps)
    return _weighted_pixel_reduce(
        true_heatmaps,
        pred_heatmaps,
        per_pixel_mse,
        positive_threshold=positive_threshold,
        positive_weight=positive_weight,
        background_weight=background_weight,
    )


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def weighted_heatmap_bce_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    positive_threshold: float = _DEFAULT_POSITIVE_THRESHOLD,
    positive_weight: float = _DEFAULT_POSITIVE_WEIGHT,
    background_weight: float = _DEFAULT_BACKGROUND_WEIGHT,
) -> tf.Tensor:
    """Weighted binary cross-entropy for soft heatmap supervision."""

    true_heatmaps = _prepare_heatmap_batch(y_true)
    pred_heatmaps = tf.clip_by_value(_prepare_heatmap_batch(y_pred), _EPSILON, 1.0 - _EPSILON)
    per_pixel_bce = keras.losses.binary_crossentropy(true_heatmaps, pred_heatmaps)
    return _weighted_pixel_reduce(
        true_heatmaps,
        pred_heatmaps,
        per_pixel_bce,
        positive_threshold=positive_threshold,
        positive_weight=positive_weight,
        background_weight=background_weight,
    )


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def focal_heatmap_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    positive_threshold: float = _DEFAULT_POSITIVE_THRESHOLD,
    positive_weight: float = _DEFAULT_POSITIVE_WEIGHT,
    background_weight: float = _DEFAULT_BACKGROUND_WEIGHT,
    alpha: float = _DEFAULT_FOCAL_ALPHA,
    gamma: float = _DEFAULT_FOCAL_GAMMA,
) -> tf.Tensor:
    """Simple focal-style heatmap loss that keeps peak pixels important."""

    true_heatmaps = _prepare_heatmap_batch(y_true)
    pred_heatmaps = tf.clip_by_value(_prepare_heatmap_batch(y_pred), _EPSILON, 1.0 - _EPSILON)
    base_bce = keras.losses.binary_crossentropy(true_heatmaps, pred_heatmaps)
    pt = true_heatmaps * pred_heatmaps + (1.0 - true_heatmaps) * (1.0 - pred_heatmaps)
    focal_factor = tf.pow(1.0 - pt, gamma)
    focal_loss = alpha * focal_factor * base_bce
    return _weighted_pixel_reduce(
        true_heatmaps,
        pred_heatmaps,
        focal_loss,
        positive_threshold=positive_threshold,
        positive_weight=positive_weight,
        background_weight=background_weight,
    )


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def softargmax_coordinate_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean squared error between soft-argmax coordinates in heatmap pixels."""

    true_coords = _softargmax_coordinates(y_true)
    pred_coords = _softargmax_coordinates(y_pred)
    return tf.reduce_mean(tf.square(true_coords - pred_coords))


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def softargmax_coordinate_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean absolute error between soft-argmax coordinates in heatmap pixels."""

    true_coords = _softargmax_coordinates(y_true)
    pred_coords = _softargmax_coordinates(y_pred)
    return tf.reduce_mean(tf.abs(true_coords - pred_coords))


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def mean_predicted_heatmap_peak(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Average maximum predicted heatmap value across the batch."""

    del y_true
    batch_heatmaps = _prepare_heatmap_batch(y_pred)
    return tf.reduce_mean(tf.reduce_max(batch_heatmaps, axis=[1, 2]))


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def combined_heatmap_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Recommended v1 heatmap objective: weighted MSE plus coordinate loss."""

    return weighted_heatmap_mse_loss(y_true, y_pred) + (
        _DEFAULT_COORDINATE_WEIGHT * softargmax_coordinate_loss(y_true, y_pred)
    )


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def weighted_center_heatmap_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Center heatmap objective used by the v2 full training run."""

    return weighted_heatmap_mse_loss(y_true, y_pred) + softargmax_coordinate_loss(y_true, y_pred)


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def weighted_tip_heatmap_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Tip heatmap objective used by the v2 full training run."""

    return weighted_heatmap_mse_loss(y_true, y_pred) + softargmax_coordinate_loss(y_true, y_pred)


def _coordinate_weighted_heatmap_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    coordinate_weight: float,
) -> tf.Tensor:
    """Combine weighted heatmap MSE with a tunable coordinate penalty."""

    return weighted_heatmap_mse_loss(y_true, y_pred) + (
        tf.cast(coordinate_weight, tf.float32) * softargmax_coordinate_loss(y_true, y_pred)
    )


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def center_priority_heatmap_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Heatmap objective tuned to make the center branch easier to fit."""

    return _coordinate_weighted_heatmap_loss(y_true, y_pred, coordinate_weight=1.0)


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def tip_priority_heatmap_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Heatmap objective tuned to keep the tip branch stable without overpowering it."""

    return _coordinate_weighted_heatmap_loss(y_true, y_pred, coordinate_weight=0.5)
