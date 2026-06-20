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


def _residual_separable_block(
    x: keras.KerasTensor,
    filters: int,
    kernel_size: int = 3,
) -> keras.KerasTensor:
    """Residual refinement block for the deeper parts of the polar model."""

    residual = x
    x = keras.layers.SeparableConv2D(
        filters,
        kernel_size,
        strides=1,
        padding="same",
        use_bias=False,
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv2D(
        filters,
        kernel_size,
        strides=1,
        padding="same",
        use_bias=False,
    )(x)
    x = keras.layers.BatchNormalization()(x)

    residual_channels = residual.shape[-1]
    if residual_channels is None or int(residual_channels) != filters:
        residual = keras.layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
        )(residual)
        residual = keras.layers.BatchNormalization()(residual)

    x = keras.layers.Add()([x, residual])
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
        coords = np.arange(width, dtype=np.float32)
        # Each column corresponds to an angle fraction of 360deg.
        angles_deg = (coords / max(float(width), 1.0)) * 360.0
        # Map angle to gauge value.
        # Normalize angle into [0, sweep] relative to min_angle.
        two_pi_equiv = 360.0
        shifted = np.mod(angles_deg - self.min_angle_deg, two_pi_equiv)
        fractions = np.clip(shifted / self.sweep_deg, 0.0, 1.0)
        span = self.value_max - self.value_min
        values = np.asarray(self.value_min + fractions * span, dtype=np.float32)
        # Store as a non-trainable weight vector so tracing/export stays graph-safe.
        self._value_weights = self.add_weight(
            name="value_weights",
            shape=(width,),
            initializer=tf.constant_initializer(values),
            trainable=False,
            dtype=tf.float32,
        )
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


def _build_polar_board_encoder(
    x: keras.KerasTensor,
    *,
    stem_filters: int = 16,
    base_filters: int = 16,
    bridge_filters: int | None = None,
    bridge_blocks: int = 2,
) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    """Build the aggressive stride-4 encoder used by the board-sized model."""

    bridge_filters = int(bridge_filters) if bridge_filters is not None else max(base_filters * 3, stem_filters * 2)

    # The first layer downsamples hard so the largest feature map stays NPU-friendly.
    x = _conv_bn_relu(x, stem_filters, kernel_size=5, strides=4)
    x = _separable_conv_bn_relu(x, base_filters, 3, 1)
    skip = _separable_conv_bn_relu(x, base_filters, 3, 1)

    # The 20x20 bridge is where we can spend extra capacity without increasing
    # the board-side activation peak.  This is the safest place to widen.
    x = _separable_conv_bn_relu(skip, base_filters * 2, 3, 2)
    x = _separable_conv_bn_relu(x, bridge_filters, 3, 1)
    for _ in range(max(1, int(bridge_blocks))):
        x = _residual_separable_block(x, bridge_filters, 3)

    # Project back to the decoder width so the later 160x160 stage still fits
    # the SRAM budget while carrying richer deep features.
    x = _conv_bn_relu(x, base_filters * 2, kernel_size=1, strides=1)
    return skip, x


def _build_polar_board_decoder(
    x: keras.KerasTensor,
    skip: keras.KerasTensor,
    *,
    base_filters: int = 16,
    mid_filters: int | None = None,
    mid_blocks: int = 2,
    dropout_rate: float = 0.0,
) -> keras.KerasTensor:
    """Build the compact decoder for the board-sized polar mask model.

    The 40x40 stage is widened first, then compressed back to the board-width
    feature map before the final 80x80 and 160x160 upsampling steps. That gives
    us more representational capacity without increasing the peak activation
    tensor that has to fit in SRAM.
    """

    mid_filters = int(mid_filters) if mid_filters is not None else max(base_filters * 2, 64)
    mid_filters = max(int(mid_filters), int(base_filters))

    # First expand back to the stem resolution and fuse the high-resolution skip.
    x = keras.layers.UpSampling2D(size=2, interpolation="nearest")(x)
    x = keras.layers.Concatenate()([x, skip])
    x = _separable_conv_bn_relu(x, mid_filters, 3, 1)
    for _ in range(max(1, int(mid_blocks))):
        x = _residual_separable_block(x, mid_filters, 3)
    x = _conv_bn_relu(x, base_filters, kernel_size=1, strides=1)
    x = _separable_conv_bn_relu(x, base_filters, 3, 1)
    x = _residual_separable_block(x, base_filters, 3)

    if dropout_rate > 0.0:
        x = keras.layers.Dropout(dropout_rate)(x)

    # Finish the 4x decoder pyramid with light, int8-friendly refinement blocks.
    x = keras.layers.UpSampling2D(size=2, interpolation="nearest")(x)
    x = _separable_conv_bn_relu(x, base_filters, 3, 1)
    x = _residual_separable_block(x, base_filters, 3)
    x = keras.layers.UpSampling2D(size=2, interpolation="nearest")(x)
    x = _separable_conv_bn_relu(x, base_filters, 3, 1)
    x = _residual_separable_block(x, base_filters, 3)
    return x


def build_polar_board_friendly_mask_model(
    polar_size: int = 160,
    *,
    input_channels: int = 3,
    stem_filters: int = 16,
    base_filters: int = 16,
    bridge_filters: int | None = None,
    bridge_blocks: int = 2,
    decoder_mid_filters: int | None = None,
    decoder_mid_blocks: int = 2,
    dropout_rate: float = 0.0,
) -> keras.Model:
    """Build a board-sized polar mask model with a stride-4 stem.

    The model stays deliberately small so the largest activation maps remain
    under the STM32N6 streaming-engine comfort zone while still predicting a
    full-resolution needle mask for downstream geometric fitting.
    """

    if polar_size % 4 != 0:
        raise ValueError("polar_size must be divisible by 4 for the board-friendly polar model.")

    if input_channels <= 0:
        raise ValueError("input_channels must be positive.")

    inputs = keras.Input(
        shape=(polar_size, polar_size, input_channels),
        name="polar_image",
        dtype=tf.float32,
    )

    skip, bottleneck = _build_polar_board_encoder(
        inputs,
        stem_filters=stem_filters,
        base_filters=base_filters,
        bridge_filters=bridge_filters,
        bridge_blocks=bridge_blocks,
    )
    decoded = _build_polar_board_decoder(
        bottleneck,
        skip,
        base_filters=base_filters,
        mid_filters=decoder_mid_filters,
        mid_blocks=decoder_mid_blocks,
        dropout_rate=dropout_rate,
    )
    needle_mask = keras.layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        name="needle_mask",
    )(decoded)

    return keras.Model(
        inputs=inputs,
        outputs=needle_mask,
        name="polar_board_friendly_mask",
    )
