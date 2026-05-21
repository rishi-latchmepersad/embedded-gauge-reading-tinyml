"""Exact Keras builder for the polar-vote V28 architecture.

The training run that produced ``polar_vote_circular_v28`` uses a 7-channel
polar input, two Conv2D/SeparableConv2D feature blocks, radial mean/max pooling,
and a 1D vote head.  This module keeps that architecture in one reusable place
so offline evaluators can load the original weights without depending on the
serialized Lambda layers from the saved ``.keras`` archive.
"""

from __future__ import annotations

from typing import Any

import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class ReduceMeanAxis(keras.layers.Layer):
    """Reduce a tensor along one axis using ``tf.reduce_mean``."""

    def __init__(self, axis: int = -1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.axis = int(axis)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(inputs, axis=self.axis)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@keras.saving.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class ReduceMaxAxis(keras.layers.Layer):
    """Reduce a tensor along one axis using ``tf.reduce_max``."""

    def __init__(self, axis: int = -1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.axis = int(axis)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.reduce_max(inputs, axis=self.axis)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


def build_polar_vote_v28_model(
    *,
    polar_size: int = 224,
    input_channels: int = 7,
    base_filters: int = 32,
    head_units: int = 128,
    dropout: float = 0.2,
) -> keras.Model:
    """Build the exact V28 polar-vote network used by the training artifacts."""
    if polar_size <= 0:
        raise ValueError("polar_size must be positive.")
    if input_channels <= 0:
        raise ValueError("input_channels must be positive.")

    inputs = keras.Input(
        shape=(polar_size, polar_size, input_channels),
        name="polar_image",
    )

    x = keras.layers.Conv2D(
        base_filters,
        kernel_size=3,
        strides=(2, 1),
        padding="same",
        use_bias=False,
        name="vote_conv2d_1",
    )(inputs)
    x = keras.layers.BatchNormalization(name="vote_bn2d_1")(x)
    x = keras.layers.Activation("swish", name="vote_act2d_1")(x)

    x = keras.layers.SeparableConv2D(
        base_filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="vote_sepconv2d_2",
    )(x)
    x = keras.layers.BatchNormalization(name="vote_bn2d_2")(x)
    x = keras.layers.Activation("swish", name="vote_act2d_2")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 1), name="vote_pool2d_1")(x)

    x = keras.layers.SeparableConv2D(
        base_filters * 2,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="vote_sepconv2d_3",
    )(x)
    x = keras.layers.BatchNormalization(name="vote_bn2d_3")(x)
    x = keras.layers.Activation("swish", name="vote_act2d_3")(x)
    x = keras.layers.SeparableConv2D(
        base_filters * 2,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="vote_sepconv2d_4",
    )(x)
    x = keras.layers.BatchNormalization(name="vote_bn2d_4")(x)
    x = keras.layers.Activation("swish", name="vote_act2d_4")(x)

    radial_mean = ReduceMeanAxis(axis=1, name="vote_radial_mean")(x)
    radial_max = ReduceMaxAxis(axis=1, name="vote_radial_max")(x)
    x = keras.layers.Concatenate(axis=-1, name="vote_radial_fuse")(
        [radial_mean, radial_max]
    )

    x = keras.layers.Conv1D(
        head_units,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="vote_conv1d_1",
    )(x)
    x = keras.layers.BatchNormalization(name="vote_bn1d_1")(x)
    x = keras.layers.Activation("swish", name="vote_act1d_1")(x)
    x = keras.layers.Dropout(dropout, name="vote_dropout")(x)

    angle_logits = keras.layers.Conv1D(
        1,
        kernel_size=1,
        padding="same",
        name="angle_logits",
    )(x)
    angle_logits_flat = keras.layers.Flatten(name="angle_logits_flat")(angle_logits)
    return keras.Model(inputs=inputs, outputs=angle_logits_flat, name="polar_angle_vote")
