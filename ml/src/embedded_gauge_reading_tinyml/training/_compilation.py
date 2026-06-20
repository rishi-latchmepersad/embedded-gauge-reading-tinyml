"""Model compilation helpers for training.
"""

from __future__ import annotations

from typing import Any

import keras
import numpy as np
import tensorflow as tf

from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec
from embedded_gauge_reading_tinyml.presets import (
    DEFAULT_GEOMETRY_UNCERTAINTY_HIGH_QUANTILE,
    DEFAULT_GEOMETRY_UNCERTAINTY_LOSS_WEIGHT,
    DEFAULT_GEOMETRY_UNCERTAINTY_LOW_QUANTILE,
    DEFAULT_GEOMETRY_VALUE_LOSS_WEIGHT,
    DEFAULT_INTERPOLATION_PAIR_SCALE,
    DEFAULT_INTERVAL_LOSS_WEIGHT,
    DEFAULT_KEYPOINT_COORD_LOSS_WEIGHT,
    DEFAULT_KEYPOINT_HEATMAP_LOSS_WEIGHT,
    DEFAULT_ORDINAL_LOSS_WEIGHT,
    DEFAULT_SWEEP_FRACTION_LOSS_WEIGHT,
)


def _make_scalar_regression_loss(
    *,
    monotonic_pair_strength: float = 0.0,
    monotonic_pair_margin: float = 0.0,
    interpolation_pair_strength: float = 0.0,
    interpolation_pair_scale: float = DEFAULT_INTERPOLATION_PAIR_SCALE,
):
    """Create the scalar regression loss used by the training heads."""

    def monotonic_pair_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Penalize local ordering violations inside a batch."""
        y_true_flat: tf.Tensor = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred_flat: tf.Tensor = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        order: tf.Tensor = tf.argsort(y_true_flat, stable=True)
        sorted_pred: tf.Tensor = tf.gather(y_pred_flat, order)
        count: tf.Tensor = tf.shape(sorted_pred)[0]

        def _compute() -> tf.Tensor:
            diffs: tf.Tensor = sorted_pred[1:] - sorted_pred[:-1]
            violations: tf.Tensor = tf.nn.relu(monotonic_pair_margin - diffs)
            return tf.reduce_mean(violations)

        return tf.cond(count < 2, lambda: tf.constant(0.0, dtype=tf.float32), _compute)

    def interpolation_pair_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Penalize local slope mismatches so nearby temperatures interpolate smoothly."""
        y_true_flat: tf.Tensor = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred_flat: tf.Tensor = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        count: tf.Tensor = tf.shape(y_true_flat)[0]

        def _compute() -> tf.Tensor:
            true_i: tf.Tensor = tf.expand_dims(y_true_flat, axis=1)
            true_j: tf.Tensor = tf.expand_dims(y_true_flat, axis=0)
            pred_i: tf.Tensor = tf.expand_dims(y_pred_flat, axis=1)
            pred_j: tf.Tensor = tf.expand_dims(y_pred_flat, axis=0)

            true_diff: tf.Tensor = true_i - true_j
            pred_diff: tf.Tensor = pred_i - pred_j
            abs_true_diff: tf.Tensor = tf.abs(true_diff)
            pair_weights: tf.Tensor = tf.exp(
                -abs_true_diff / tf.constant(interpolation_pair_scale, dtype=tf.float32)
            )
            pair_mask: tf.Tensor = 1.0 - tf.eye(count, dtype=tf.float32)
            pair_error: tf.Tensor = tf.abs(pred_diff - true_diff)
            weighted_error: tf.Tensor = pair_weights * pair_mask * pair_error
            normalizer: tf.Tensor = tf.reduce_sum(pair_weights * pair_mask)
            return tf.math.divide_no_nan(tf.reduce_sum(weighted_error), normalizer)

        return tf.cond(count < 2, lambda: tf.constant(0.0, dtype=tf.float32), _compute)

    def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Blend pointwise regression loss with interpolation and ordering penalties."""
        mse: tf.Tensor = tf.reduce_mean(
            tf.square(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32))
        )
        total_loss: tf.Tensor = mse
        if monotonic_pair_strength > 0.0:
            total_loss = total_loss + monotonic_pair_strength * monotonic_pair_loss(
                y_true, y_pred
            )
        if interpolation_pair_strength > 0.0:
            total_loss = (
                total_loss
                + interpolation_pair_strength * interpolation_pair_loss(y_true, y_pred)
            )
        return total_loss

    return combined_loss



def _make_pinball_loss(quantile: float):
    """Create a pinball loss for a scalar quantile head."""
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0, 1).")

    quantile_const: tf.Tensor = tf.constant(quantile, dtype=tf.float32)

    def pinball_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Penalize under- and over-estimates according to the target quantile."""
        error: tf.Tensor = tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)
        return tf.reduce_mean(
            tf.maximum(
                quantile_const * error,
                (quantile_const - 1.0) * error,
            )
        )

    pinball_loss.__name__ = f"pinball_q{int(round(quantile * 100.0)):02d}"
    return pinball_loss



def _compile_regression_model(
    model: keras.Model,
    *,
    learning_rate: float,
    monotonic_pair_strength: float = 0.0,
    monotonic_pair_margin: float = 0.0,
    interpolation_pair_strength: float = 0.0,
    interpolation_pair_scale: float = DEFAULT_INTERPOLATION_PAIR_SCALE,
) -> None:
    """Compile a scalar regression model with standard losses and metrics."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss=_make_scalar_regression_loss(
            monotonic_pair_strength=monotonic_pair_strength,
            monotonic_pair_margin=monotonic_pair_margin,
            interpolation_pair_strength=interpolation_pair_strength,
            interpolation_pair_scale=interpolation_pair_scale,
        ),
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )



def _compile_fraction_model(
    model: keras.Model,
    *,
    learning_rate: float,
    fraction_loss_weight: float = DEFAULT_SWEEP_FRACTION_LOSS_WEIGHT,
) -> None:
    """Compile a sweep-fraction model with scalar and fraction losses."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "gauge_value": keras.losses.MeanSquaredError(),
            "sweep_fraction": keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "gauge_value": 1.0,
            "sweep_fraction": fraction_loss_weight,
        },
        metrics={
            "gauge_value": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "sweep_fraction": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
        },
    )



def _compile_keypoint_model(
    model: keras.Model,
    *,
    learning_rate: float,
    heatmap_loss_weight: float = DEFAULT_KEYPOINT_HEATMAP_LOSS_WEIGHT,
    value_loss_weight: float = 1.0,
) -> None:
    """Compile a keypoint-auxiliary model with scalar and heatmap losses."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "gauge_value": keras.losses.MeanSquaredError(),
            "keypoint_heatmaps": keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "gauge_value": value_loss_weight,
            "keypoint_heatmaps": heatmap_loss_weight,
        },
        metrics={
            "gauge_value": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "keypoint_heatmaps": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
        },
    )



def _compile_geometry_model(
    model: keras.Model,
    *,
    learning_rate: float,
    heatmap_loss_weight: float = DEFAULT_KEYPOINT_HEATMAP_LOSS_WEIGHT,
    coord_loss_weight: float = DEFAULT_KEYPOINT_COORD_LOSS_WEIGHT,
    value_loss_weight: float = DEFAULT_GEOMETRY_VALUE_LOSS_WEIGHT,
) -> None:
    """Compile a geometry detector with heatmap, coordinate, and value losses."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "gauge_value": keras.losses.MeanSquaredError(),
            "keypoint_heatmaps": keras.losses.MeanSquaredError(),
            "keypoint_coords": keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "gauge_value": value_loss_weight,
            "keypoint_heatmaps": heatmap_loss_weight,
            "keypoint_coords": coord_loss_weight,
        },
        metrics={
            "gauge_value": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "keypoint_heatmaps": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
            "keypoint_coords": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                _make_keypoint_angle_mae_metric(),
            ],
        },
    )



def _compile_geometry_uncertainty_model(
    model: keras.Model,
    *,
    learning_rate: float,
    heatmap_loss_weight: float = DEFAULT_KEYPOINT_HEATMAP_LOSS_WEIGHT,
    coord_loss_weight: float = DEFAULT_KEYPOINT_COORD_LOSS_WEIGHT,
    value_loss_weight: float = DEFAULT_GEOMETRY_VALUE_LOSS_WEIGHT,
    uncertainty_loss_weight: float = DEFAULT_GEOMETRY_UNCERTAINTY_LOSS_WEIGHT,
    low_quantile: float = DEFAULT_GEOMETRY_UNCERTAINTY_LOW_QUANTILE,
    high_quantile: float = DEFAULT_GEOMETRY_UNCERTAINTY_HIGH_QUANTILE,
) -> None:
    """Compile a geometry model with symmetric uncertainty bounds."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "gauge_value": keras.losses.MeanSquaredError(),
            "keypoint_heatmaps": keras.losses.MeanSquaredError(),
            "keypoint_coords": keras.losses.MeanSquaredError(),
            "gauge_value_lower": _make_pinball_loss(low_quantile),
            "gauge_value_upper": _make_pinball_loss(high_quantile),
        },
        loss_weights={
            "gauge_value": value_loss_weight,
            "keypoint_heatmaps": heatmap_loss_weight,
            "keypoint_coords": coord_loss_weight,
            "gauge_value_lower": uncertainty_loss_weight,
            "gauge_value_upper": uncertainty_loss_weight,
        },
        metrics={
            "gauge_value": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "keypoint_heatmaps": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
            "keypoint_coords": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                _make_keypoint_angle_mae_metric(),
            ],
            "gauge_value_lower": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
            "gauge_value_upper": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
        },
    )





def _source_crop_box_v2_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    giou_weight: float = 2.0,
    huber_weight: float = 1.0,
    aspect_weight: float = 1.0,
    center_weight: float = 0.5,
    eps: float = 1e-7,
) -> tf.Tensor:
    """Combined GIoU + Huber + aspect-ratio + center loss for source crop boxes.

    Targets and predictions are ordered normalized xyxy tensors in [0, 1].
    """
    yt = tf.cast(y_true, tf.float32)
    yp = tf.cast(y_pred, tf.float32)

    # Split corners
    xt0, yt0, xt1, yt1 = tf.split(yt, 4, axis=-1)
    xp0, yp0, xp1, yp1 = tf.split(yp, 4, axis=-1)

    # Widths and heights
    w_true = tf.maximum(xt1 - xt0, 0.0)
    h_true = tf.maximum(yt1 - yt0, 0.0)
    w_pred = tf.maximum(xp1 - xp0, 0.0)
    h_pred = tf.maximum(yp1 - yp0, 0.0)

    # Areas
    area_true = w_true * h_true
    area_pred = w_pred * h_pred

    # Intersection
    xi0 = tf.maximum(xt0, xp0)
    yi0 = tf.maximum(yt0, yp0)
    xi1 = tf.minimum(xt1, xp1)
    yi1 = tf.minimum(yt1, yp1)
    wi = tf.maximum(xi1 - xi0, 0.0)
    hi = tf.maximum(yi1 - yi0, 0.0)
    intersection = wi * hi

    union = area_true + area_pred - intersection
    iou = intersection / (union + eps)

    # Enclosing box for GIoU
    xc0 = tf.minimum(xt0, xp0)
    yc0 = tf.minimum(yt0, yp0)
    xc1 = tf.maximum(xt1, xp1)
    yc1 = tf.maximum(yt1, yp1)
    wc = tf.maximum(xc1 - xc0, 0.0)
    hc = tf.maximum(yc1 - yc0, 0.0)
    area_c = wc * hc

    giou = iou - (area_c - union) / (area_c + eps)
    giou_loss = tf.reduce_mean(1.0 - giou)

    # Huber for direct corner accuracy
    huber_loss = tf.reduce_mean(
        keras.losses.huber(yt, yp, delta=0.05)
    )

    # Aspect ratio loss: match arctan(w/h)
    ar_true = tf.math.atan(w_true / (h_true + eps))
    ar_pred = tf.math.atan(w_pred / (h_pred + eps))
    aspect_loss = tf.reduce_mean(tf.square(ar_true - ar_pred))

    # Center loss: keep predicted box centered on target
    cx_true = (xt0 + xt1) * 0.5
    cy_true = (yt0 + yt1) * 0.5
    cx_pred = (xp0 + xp1) * 0.5
    cy_pred = (yp0 + yp1) * 0.5
    center_loss = tf.reduce_mean(
        tf.square(cx_true - cx_pred) + tf.square(cy_true - cy_pred)
    )

    total = (
        giou_weight * giou_loss
        + huber_weight * huber_loss
        + aspect_weight * aspect_loss
        + center_weight * center_loss
    )
    return total



def _compile_source_crop_box_v2_model(
    model: keras.Model,
    *,
    learning_rate: float,
) -> None:
    """Compile a source-space crop-box v2 model with GIoU + aspect + center loss."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "source_crop_box": _source_crop_box_v2_loss,
        },
        metrics={
            "source_crop_box": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        },
    )


def _compile_rectifier_model(
    model: keras.Model,
    *,
    learning_rate: float,
) -> None:
    """Compile a rectifier model that regresses the normalized crop box."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "rectifier_box": keras.losses.Huber(delta=0.05),
        },
        metrics={
            "rectifier_box": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        },
    )



def _compile_source_crop_box_model(
    model: keras.Model,
    *,
    learning_rate: float,
) -> None:
    """Compile a source-space crop-box model that regresses xyxy corners."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "source_crop_box": keras.losses.Huber(delta=0.05),
        },
        metrics={
            "source_crop_box": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        },
    )



def _compile_source_crop_corner_model(
    model: keras.Model,
    *,
    learning_rate: float,
    heatmap_loss_weight: float = DEFAULT_KEYPOINT_HEATMAP_LOSS_WEIGHT,
    coord_loss_weight: float = DEFAULT_KEYPOINT_COORD_LOSS_WEIGHT,
    box_loss_weight: float = 1.0,
) -> None:
    """Compile the corner-localizer with heatmap, coordinate, and box losses."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "source_crop_canvas_box": keras.losses.Huber(delta=0.05),
            "keypoint_heatmaps": keras.losses.MeanSquaredError(),
            "keypoint_coords": keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "source_crop_canvas_box": box_loss_weight,
            "keypoint_heatmaps": heatmap_loss_weight,
            "keypoint_coords": coord_loss_weight,
        },
        metrics={
            "source_crop_canvas_box": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "keypoint_heatmaps": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
            "keypoint_coords": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
        },
    )



def _compile_obb_model(
    model: keras.Model,
    *,
    learning_rate: float,
) -> None:
    """Compile an oriented-box localizer with a compact regression loss."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "obb_params": keras.losses.Huber(delta=0.05),
        },
        metrics={
            "obb_params": [
                keras.metrics.MeanAbsoluteError(name="obb_params_mae"),
                keras.metrics.RootMeanSquaredError(name="obb_params_rmse"),
            ],
        },
    )



def _compile_obb_geometry_model(
    model: keras.Model,
    *,
    learning_rate: float,
    obb_loss_weight: float = 0.35,
    heatmap_loss_weight: float = DEFAULT_KEYPOINT_HEATMAP_LOSS_WEIGHT,
    coord_loss_weight: float = DEFAULT_KEYPOINT_COORD_LOSS_WEIGHT,
    value_loss_weight: float = 1.0,
) -> None:
    """Compile a detector-plus-geometry model with OBB and keypoint losses."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "gauge_value": keras.losses.MeanSquaredError(),
            "keypoint_heatmaps": keras.losses.MeanSquaredError(),
            "keypoint_coords": keras.losses.MeanSquaredError(),
            "obb_params": keras.losses.Huber(delta=0.05),
        },
        loss_weights={
            "gauge_value": value_loss_weight,
            "keypoint_heatmaps": heatmap_loss_weight,
            "keypoint_coords": coord_loss_weight,
            "obb_params": obb_loss_weight,
        },
        metrics={
            "gauge_value": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "keypoint_heatmaps": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
            "keypoint_coords": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                _make_keypoint_angle_mae_metric(),
            ],
            "obb_params": [
                keras.metrics.MeanAbsoluteError(name="obb_params_mae"),
                keras.metrics.RootMeanSquaredError(name="obb_params_rmse"),
            ],
        },
    )



def _compile_obb_mask_geometry_model(
    model: keras.Model,
    *,
    learning_rate: float,
    heatmap_loss_weight: float = DEFAULT_KEYPOINT_HEATMAP_LOSS_WEIGHT,
    coord_loss_weight: float = DEFAULT_KEYPOINT_COORD_LOSS_WEIGHT,
    mask_loss_weight: float = DEFAULT_KEYPOINT_HEATMAP_LOSS_WEIGHT,
    value_loss_weight: float = DEFAULT_GEOMETRY_VALUE_LOSS_WEIGHT,
    obb_loss_weight: float = 0.35,
) -> None:
    """Compile an OBB-plus-mask model with segmentation and geometry losses."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "gauge_value": keras.losses.MeanSquaredError(),
            "keypoint_heatmaps": keras.losses.MeanSquaredError(),
            "keypoint_coords": keras.losses.MeanSquaredError(),
            "pointer_mask": keras.losses.BinaryCrossentropy(),
            "obb_params": keras.losses.Huber(delta=0.05),
        },
        loss_weights={
            "gauge_value": value_loss_weight,
            "keypoint_heatmaps": heatmap_loss_weight,
            "keypoint_coords": coord_loss_weight,
            "pointer_mask": mask_loss_weight,
            "obb_params": obb_loss_weight,
        },
        metrics={
            "gauge_value": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "keypoint_heatmaps": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
            "keypoint_coords": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                _make_keypoint_angle_mae_metric(),
            ],
            "pointer_mask": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
            "obb_params": [
                keras.metrics.MeanAbsoluteError(name="obb_params_mae"),
                keras.metrics.RootMeanSquaredError(name="obb_params_rmse"),
            ],
        },
    )



def _compile_interval_model(
    model: keras.Model,
    *,
    learning_rate: float,
    monotonic_pair_strength: float = 0.0,
    monotonic_pair_margin: float = 0.0,
    interval_loss_weight: float = DEFAULT_INTERVAL_LOSS_WEIGHT,
) -> None:
    """Compile the hybrid interval model with scalar and coarse-bin losses."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )
    scalar_loss = _make_scalar_regression_loss(
        monotonic_pair_strength=monotonic_pair_strength,
        monotonic_pair_margin=monotonic_pair_margin,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "gauge_value": scalar_loss,
            "interval_logits": keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
        },
        loss_weights={
            "gauge_value": 1.0,
            "interval_logits": interval_loss_weight,
        },
        metrics={
            "gauge_value": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "interval_logits": [
                keras.metrics.SparseCategoricalAccuracy(name="acc"),
            ],
        },
    )



def _compile_ordinal_model(
    model: keras.Model,
    *,
    learning_rate: float,
    ordinal_loss_weight: float = DEFAULT_ORDINAL_LOSS_WEIGHT,
) -> None:
    """Compile an ordinal-threshold model with scalar and ordinal losses."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "gauge_value": keras.losses.MeanSquaredError(),
            "ordinal_logits": keras.losses.BinaryCrossentropy(from_logits=True),
        },
        loss_weights={
            "gauge_value": 1.0,
            "ordinal_logits": ordinal_loss_weight,
        },
        metrics={
            "gauge_value": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        },
    )



def _direction_cosine_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Optimize angular agreement between unit needle vectors."""
    dot: tf.Tensor = tf.reduce_sum(y_true * y_pred, axis=-1)
    clipped: tf.Tensor = tf.clip_by_value(dot, -1.0, 1.0)
    return tf.reduce_mean(1.0 - clipped)



def _compile_direction_model(
    model: keras.Model,
    *,
    learning_rate: float,
    spec: GaugeSpec,
) -> None:
    """Compile a direction-regression model with value-space metrics."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )
    model.compile(
        optimizer=optimizer,
        loss=_direction_cosine_loss,
        metrics=[
            _make_angle_mae_metric(),
            _make_value_mae_metric(spec),
            _make_value_rmse_metric(spec),
        ],
    )



def _compile_direction_geometry_model(
    model: keras.Model,
    *,
    learning_rate: float,
    spec: GaugeSpec,
) -> None:
    """Compile a geometry-bottleneck direction model with scalar supervision."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "gauge_value": keras.losses.MeanSquaredError(),
            "needle_xy": _direction_cosine_loss,
        },
        loss_weights={
            "gauge_value": 1.0,
            # Keep geometric direction as an auxiliary signal only.
            # Hard-case manifests often contain value-only labels, so forcing a
            # strong direction loss can overpower the scalar objective and hurt
            # generalization on unseen board captures.
            "needle_xy": 0.0,
        },
        metrics={
            "gauge_value": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "needle_xy": [
                _make_angle_mae_metric(),
            ],
        },
    )


