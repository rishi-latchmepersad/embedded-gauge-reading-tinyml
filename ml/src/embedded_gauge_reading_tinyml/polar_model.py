"""Polar needle-segmentation model for analog gauge reading.

This model treats gauge reading as a needle-detection problem in polar
space rather than direct temperature regression. The key insight is that
in a polar projection of the gauge face, the needle becomes a vertical
line — much easier for a CNN to localize than inferring angle from raw
Cartesian pixels.

Architecture:
  1. Input: polar-projected image (angle vs radius).
  2. Lightweight UNet-style encoder-decoder for needle segmentation.
  3. Angle extraction via soft argmax on the predicted mask.
  4. Temperature conversion using the known gauge calibration.

The polar projection is done in the data pipeline, not inside the model,
so the model itself is pure TensorFlow/Keras and fully differentiable.
"""

from __future__ import annotations

import math

import keras
import numpy as np
import tensorflow as tf


def _conv_bn_relu(
    x: keras.KerasTensor,
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
) -> keras.KerasTensor:
    """Conv2D + BatchNorm + ReLU block."""
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    return x


def _separable_conv_bn_relu(
    x: keras.KerasTensor,
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
) -> keras.KerasTensor:
    """SeparableConv2D + BatchNorm + ReLU block."""
    x = keras.layers.SeparableConv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    return x


def _build_polar_encoder(
    x: keras.KerasTensor,
    base_filters: int = 32,
    depth: int = 4,
) -> tuple[list[keras.KerasTensor], keras.KerasTensor]:
    """Build a lightweight encoder stack with skip connections.

    Returns:
        (skip_connections, bottleneck)
    """
    skips: list[keras.KerasTensor] = []
    filters = base_filters

    for level in range(depth):
        x = _separable_conv_bn_relu(x, filters, 3, 1)
        x = _separable_conv_bn_relu(x, filters, 3, 1)
        skips.append(x)
        # Downsample: use stride 2 conv instead of maxpool for smoother features.
        if level < depth - 1:
            x = _separable_conv_bn_relu(x, filters * 2, 3, 2)
            filters *= 2

    return skips, x


def _build_polar_decoder(
    x: keras.KerasTensor,
    skip_connections: list[keras.KerasTensor],
    base_filters: int = 32,
    depth: int = 4,
) -> keras.KerasTensor:
    """Build a lightweight decoder with skip connections."""
    filters = base_filters * (2 ** (depth - 2))

    for level in range(depth - 2, -1, -1):
        # Upsample.
        x = keras.layers.Conv2DTranspose(
            filters // 2,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        # Concatenate skip.
        skip = skip_connections[level]
        x = keras.layers.Concatenate()([x, skip])

        # Refine.
        x = _separable_conv_bn_relu(x, filters // 2, 3, 1)
        x = _separable_conv_bn_relu(x, filters // 2, 3, 1)
        filters //= 2

    return x


@keras.saving.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class PolarAngleToTemperature(keras.layers.Layer):
    """Convert a polar needle mask into a calibrated temperature value.

    This layer is differentiable so the whole pipeline can be trained end-to-end
    with only temperature supervision (no mask labels required during training).
    """

    def __init__(
        self,
        *,
        value_min: float = -30.0,
        value_max: float = 50.0,
        min_angle_deg: float = 135.0,
        sweep_deg: float = 270.0,
        temperature: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.min_angle_deg = float(min_angle_deg)
        self.sweep_deg = float(sweep_deg)
        self.temperature = float(temperature)

    def build(self, input_shape: tf.TensorShape) -> None:
        """Precompute the angle coordinate grid."""
        # Polar image width corresponds to 0-360 degrees.
        width = int(input_shape[2]) if input_shape[2] is not None else 224
        coords = tf.range(width, dtype=tf.float32)
        # Each column corresponds to an angle fraction of 360deg.
        angles_deg = (coords / tf.maximum(tf.cast(width, tf.float32), 1.0)) * 360.0
        # Map angle to gauge value.
        # Normalize angle into [0, sweep] relative to min_angle.
        two_pi_equiv = 360.0
        shifted = tf.math.floormod(angles_deg - self.min_angle_deg, two_pi_equiv)
        fractions = tf.clip_by_value(shifted / self.sweep_deg, 0.0, 1.0)
        span = self.value_max - self.value_min
        values = self.value_min + fractions * span
        # Store as a constant weight vector.
        self._value_weights = tf.constant(values, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Compute temperature from needle mask via soft expectation."""
        mask = tf.cast(inputs, tf.float32)  # (B, H, W, 1)
        # Sum over vertical axis to get 1D angular profile.
        profile = tf.reduce_sum(mask, axis=[1, 3])  # (B, W)
        # Softmax for differentiable peak selection.
        weights = tf.nn.softmax(profile * self.temperature, axis=-1)
        # Weighted sum of value bins.
        values = self._value_weights  # (W,)
        predicted = tf.reduce_sum(weights * values, axis=-1, keepdims=True)  # (B, 1)
        return predicted

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "value_min": self.value_min,
                "value_max": self.value_max,
                "min_angle_deg": self.min_angle_deg,
                "sweep_deg": self.sweep_deg,
                "temperature": self.temperature,
            }
        )
        return config


def build_polar_needle_segmentation_model(
    polar_size: int = 224,
    base_filters: int = 32,
    depth: int = 4,
    dropout_rate: float = 0.1,
    value_min: float = -30.0,
    value_max: float = 50.0,
    min_angle_deg: float = 135.0,
    sweep_deg: float = 270.0,
) -> keras.Model:
    """Build a polar needle-segmentation model.

    The model takes a polar-projected image as input and predicts a needle mask.
    The temperature is derived from the mask via soft argmax.

    Args:
        polar_size: Size of the polar projection (square).
        base_filters: Base filter count for the UNet.
        depth: Encoder depth (number of resolution levels).
        dropout_rate: Dropout rate applied in the bottleneck.
        value_min: Minimum gauge temperature.
        value_max: Maximum gauge temperature.
        min_angle_deg: Angle corresponding to value_min.
        sweep_deg: Total sweep angle of the gauge.

    Returns:
        A Keras Model with outputs:
          - "needle_mask": Predicted needle mask in polar space (H, W, 1).
          - "gauge_value": Derived temperature in Celsius.
    """
    # Input: polar-projected RGB image.
    inputs = keras.Input(
        shape=(polar_size, polar_size, 3),
        name="polar_image",
        dtype=tf.float32,
    )

    # Encoder-decoder for needle segmentation.
    skips, bottleneck = _build_polar_encoder(inputs, base_filters, depth)

    if dropout_rate > 0.0:
        bottleneck = keras.layers.Dropout(dropout_rate)(bottleneck)

    decoded = _build_polar_decoder(bottleneck, skips, base_filters, depth)

    # Final segmentation head: 1 channel sigmoid mask.
    needle_mask = keras.layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        name="needle_mask",
    )(decoded)

    # Temperature head: soft argmax on the mask to get angle, then convert.
    gauge_value = PolarAngleToTemperature(
        value_min=value_min,
        value_max=value_max,
        min_angle_deg=min_angle_deg,
        sweep_deg=sweep_deg,
        name="gauge_value",
    )(needle_mask)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "needle_mask": needle_mask,
        },
        name="polar_needle_segmentation",
    )
    return model


def build_polar_tiny_model(
    polar_size: int = 224,
    value_min: float = -30.0,
    value_max: float = 50.0,
    min_angle_deg: float = 135.0,
    sweep_deg: float = 270.0,
) -> keras.Model:
    """Build the smallest viable polar needle model for STM32N6.

    Uses base_filters=16 and depth=3 for minimal parameter count.
    """
    return build_polar_needle_segmentation_model(
        polar_size=polar_size,
        base_filters=16,
        depth=3,
        dropout_rate=0.05,
        value_min=value_min,
        value_max=value_max,
        min_angle_deg=min_angle_deg,
        sweep_deg=sweep_deg,
    )
