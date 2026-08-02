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


def _heatmap_spread_pixels(heatmaps: tf.Tensor) -> tf.Tensor:
    """Compute the spatial standard deviation of each heatmap in pixel units."""

    batch_heatmaps = _prepare_heatmap_batch(heatmaps)
    batch_heatmaps = tf.maximum(batch_heatmaps, 0.0)
    height = tf.shape(batch_heatmaps)[1]
    width = tf.shape(batch_heatmaps)[2]
    x_coords, y_coords = _build_pixel_grids(height, width)

    heatmap_sum = tf.reduce_sum(batch_heatmaps, axis=[1, 2], keepdims=True)
    heatmap_sum = tf.maximum(heatmap_sum, _EPSILON)
    normalized = batch_heatmaps / heatmap_sum

    mean_x = tf.reduce_sum(normalized * x_coords, axis=[1, 2], keepdims=True)
    mean_y = tf.reduce_sum(normalized * y_coords, axis=[1, 2], keepdims=True)
    spread_sq = tf.reduce_sum(
        normalized
        * (
            tf.square(x_coords - mean_x)
            + tf.square(y_coords - mean_y)
        ),
        axis=[1, 2],
    )
    return tf.sqrt(tf.maximum(spread_sq, 0.0))


def _prepare_polar_profile_batch(heatmaps: tf.Tensor) -> tf.Tensor:
    """Collapse polar heatmaps into 1D angular profiles by summing rows."""

    batch_heatmaps = _prepare_heatmap_batch(heatmaps)
    return tf.reduce_sum(batch_heatmaps, axis=1)


def _normalize_profile_distribution(profile: tf.Tensor) -> tf.Tensor:
    """Normalize a 1D profile into a proper probability distribution."""

    tensor = tf.cast(tf.convert_to_tensor(profile), tf.float32)
    tensor = tf.maximum(tensor, 0.0)
    profile_sum = tf.reduce_sum(tensor, axis=-1, keepdims=True)
    return tensor / tf.maximum(profile_sum, _EPSILON)


def _softargmax_profile_coordinates(profiles: tf.Tensor) -> tf.Tensor:
    """Compute the expected angle-bin index for each 1D profile."""

    tensor = tf.cast(tf.convert_to_tensor(profiles), tf.float32)
    rank = tensor.shape.rank

    if rank == 1:
        tensor = tf.expand_dims(tensor, axis=0)
    elif rank != 2:
        raise ValueError("Profiles must have rank 1 or 2.")

    width = tf.shape(tensor)[1]
    coords = tf.cast(tf.range(width), tf.float32)[tf.newaxis, :]
    profile_sum = tf.reduce_sum(tensor, axis=-1, keepdims=True)
    profile_sum = tf.maximum(profile_sum, _EPSILON)
    normalized = tensor / profile_sum
    return tf.reduce_sum(normalized * coords, axis=-1)


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


def _elementwise_binary_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute pixelwise binary cross-entropy without collapsing spatial axes."""

    true_heatmaps = _prepare_heatmap_batch(y_true)
    pred_heatmaps = tf.clip_by_value(_prepare_heatmap_batch(y_pred), _EPSILON, 1.0 - _EPSILON)
    return -(
        true_heatmaps * tf.math.log(pred_heatmaps)
        + (1.0 - true_heatmaps) * tf.math.log(1.0 - pred_heatmaps)
    )


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
    pred_heatmaps = _prepare_heatmap_batch(y_pred)
    per_pixel_bce = _elementwise_binary_crossentropy(true_heatmaps, pred_heatmaps)
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
    pred_heatmaps = _prepare_heatmap_batch(y_pred)
    clipped_pred_heatmaps = tf.clip_by_value(pred_heatmaps, _EPSILON, 1.0 - _EPSILON)
    base_bce = _elementwise_binary_crossentropy(true_heatmaps, clipped_pred_heatmaps)
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
def heatmap_spread_hinge_loss(y_pred: tf.Tensor, *, max_spread_px: float = 30.0) -> tf.Tensor:
    """Penalize heatmaps whose spatial spread exceeds a target bound."""

    if max_spread_px <= 0.0:
        raise ValueError("max_spread_px must be positive.")

    spread_px = _heatmap_spread_pixels(y_pred)
    return tf.reduce_mean(tf.square(tf.maximum(spread_px - float(max_spread_px), 0.0)))


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def combined_heatmap_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Recommended v1 heatmap objective: weighted MSE plus coordinate loss."""

    return weighted_heatmap_mse_loss(y_true, y_pred) + (
        _DEFAULT_COORDINATE_WEIGHT * softargmax_coordinate_loss(y_true, y_pred)
    )


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def polar_profile_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Angle-first objective for polar needle masks.

    The polar decoder only needs the horizontal angular profile, so this loss
    collapses the mask vertically and focuses training on the angle column
    instead of penalizing irrelevant vertical placement.
    """

    true_profile = _prepare_polar_profile_batch(y_true)
    pred_profile = _prepare_polar_profile_batch(y_pred)
    true_dist = _normalize_profile_distribution(true_profile)
    pred_dist = _normalize_profile_distribution(pred_profile)

    profile_kl = tf.reduce_mean(
        tf.reduce_sum(
            true_dist
            * (
                tf.math.log(true_dist + _EPSILON)
                - tf.math.log(pred_dist + _EPSILON)
            ),
            axis=-1,
        )
    )
    coord_loss = tf.reduce_mean(
        tf.square(
            _softargmax_profile_coordinates(true_profile)
            - _softargmax_profile_coordinates(pred_profile)
        )
    )
    peak_loss = tf.reduce_mean(
        tf.square(
            tf.reduce_max(true_dist, axis=-1) - tf.reduce_max(pred_dist, axis=-1)
        )
    )
    loss = profile_kl + (0.05 * coord_loss) + (0.1 * peak_loss)
    return tf.maximum(loss, 0.0)


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


# ---------------------------------------------------------------------------
# Wing Loss for keypoint heatmap regression
# Feng et al., CVPR 2018 -- designed for sub-pixel keypoint localization
# ---------------------------------------------------------------------------

@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def wing_heatmap_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    omega: float = 10.0,
    epsilon: float = 2.0,
    positive_weight: float = 10.0,
    background_weight: float = 0.2,
    positive_threshold: float = 0.1,
) -> tf.Tensor:
    """Wing loss for heatmap regression (Feng et al., CVPR 2018).

    Has a linear region near zero that gives constant gradients for small
    errors, and a log region beyond that which gives stronger gradients
    for medium errors.  Ideal for keypoint localization where the model
    needs to learn precise sub-pixel positions.

    Handles multi-channel heatmaps (batch, H, W, C) by averaging the
    per-channel Wing loss.

    Args:
        omega: Width of the linear region (in heatmap pixel units).
        epsilon: Controls curvature of the log region.  Larger = more linear.
        positive_weight: Loss multiplier for heatmap peak pixels.
        background_weight: Loss multiplier for background pixels.
        positive_threshold: Heatmap value above which a pixel is "positive".
    """
    y_true_f = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
    y_pred_f = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)

    # C = omega * (1 - log(1 + omega/epsilon))
    c = omega * (1.0 - tf.math.log(1.0 + omega / epsilon))

    def _per_channel_loss(true_ch, pred_ch):
        """Wing loss on a single (batch, H, W) channel."""
        diff = tf.abs(pred_ch - true_ch)
        per_pixel = tf.where(
            diff < omega,
            omega * tf.math.log(1.0 + diff / epsilon),
            diff - c,
        )
        weights = tf.where(
            true_ch > positive_threshold,
            tf.cast(positive_weight, tf.float32),
            tf.cast(background_weight, tf.float32),
        )
        weighted = tf.reduce_sum(weights * per_pixel, axis=[1, 2])
        w_sum = tf.maximum(tf.reduce_sum(weights, axis=[1, 2]), 1.0)
        return tf.reduce_mean(weighted / w_sum)

    # Handle multi-channel (batch, H, W, C) by averaging per-channel loss
    if y_true_f.shape.rank == 4 and y_true_f.shape[-1] > 1:
        num_ch = y_true_f.shape[-1]
        losses = [_per_channel_loss(y_true_f[..., i], y_pred_f[..., i]) for i in range(num_ch)]
        # Weight tip channel more (channel 1)
        channel_weights = [1.0, 1.5] if num_ch == 2 else [1.0] * num_ch
        total = sum(w * l for w, l in zip(channel_weights, losses)) / sum(channel_weights)
        return total
    else:
        return _per_channel_loss(
            tf.squeeze(y_true_f, axis=-1) if y_true_f.shape.rank == 4 else y_true_f,
            tf.squeeze(y_pred_f, axis=-1) if y_pred_f.shape.rank == 4 else y_pred_f,
        )


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def adaptive_wing_heatmap_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    omega: float = 14.0,
    theta: float = 0.5,
    epsilon: float = 1.0,
    alpha: float = 2.1,
) -> tf.Tensor:
    """Adaptive Wing Loss (Wang et al., ICCV 2019).

    Adapts its shape based on ground truth value: near peaks the loss is
    tighter (smaller omega, larger gradient), away from peaks it relaxes.
    Handles multi-channel heatmaps by averaging per-channel loss.
    """
    y_true_f = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
    y_pred_f = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)

    def _per_channel(true_ch, pred_ch):
        diff = tf.abs(pred_ch - true_ch)
        w = tf.where(
            true_ch < theta,
            1.0,
            alpha * tf.pow(true_ch + 1e-7, 1.0 / alpha),
        )
        c = omega * (1.0 / (1.0 + tf.pow(theta / epsilon, alpha - true_ch)))
        awing = tf.where(
            diff < omega,
            w * omega * tf.math.log(1.0 + diff / epsilon),
            w * (diff - c),
        )
        return tf.reduce_mean(awing)

    if y_true_f.shape.rank == 4 and y_true_f.shape[-1] > 1:
        num_ch = y_true_f.shape[-1]
        losses = [_per_channel(y_true_f[..., i], y_pred_f[..., i]) for i in range(num_ch)]
        return tf.add_n(losses) / float(num_ch)
    else:
        return _per_channel(
            tf.squeeze(y_true_f, axis=-1) if y_true_f.shape.rank == 4 else y_true_f,
            tf.squeeze(y_pred_f, axis=-1) if y_pred_f.shape.rank == 4 else y_pred_f,
        )
