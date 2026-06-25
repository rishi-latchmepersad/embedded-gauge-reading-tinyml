"""
Model utilities for geometry points prediction.

This module provides model architectures for predicting dial center and needle tip
coordinates from cropped gauge images.

The model predicts normalized coordinates in [0, 1] range:
- center_x_normalized
- center_y_normalized
- tip_x_normalized
- tip_y_normalized
- confidence

All coordinate outputs are sigmoid-constrained to [0, 1].
"""

import math
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _scaled_width(channels: int, width_multiplier: float) -> int:
    """Scale a channel count and round it to a multiple of 8."""

    scaled = int(round(float(channels) * float(width_multiplier) / 8.0) * 8)
    return max(8, scaled)


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class Identity3x3Initializer(keras.initializers.Initializer):
    """A 3x3 conv kernel initializer that approximates identity.

    The centre pixel copies each input channel to the matching output channel
    (weight=1.0) while all other spatial positions and cross-channel entries
    are zero.  Together with bias=0 this makes the layer a near-pass-through
    so the model starts from a sane state when a new refinement block is added
    on top of a pre-trained decoder.
    """
    def __call__(self, shape, dtype=None):
        kernel_np = np.zeros(shape, dtype=dtype or np.float32)
        cy = shape[0] // 2
        cx = shape[1] // 2
        in_c = shape[2]
        out_c = shape[3]
        common = min(in_c, out_c)
        for c in range(common):
            kernel_np[cy, cx, c, c] = 1.0
        return tf.constant(kernel_np, dtype=dtype or tf.float32)

    def get_config(self):
        return {}


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class BilinearUpsamplingInitializer(keras.initializers.Initializer):
    """Initialize a transposed-conv kernel to behave like bilinear upsampling.

    This gives the model a learnable final resize stage without starting from
    a random checkerboard-prone kernel.  The initializer copies the same
    bilinear kernel into the matching input/output channel pairs so the
    upsampling layer begins as a stable 2x resize and can then sharpen the
    features during training.
    """

    def __init__(self, scale: int = 2) -> None:
        self.scale = int(scale)

    def __call__(self, shape, dtype=None):
        if len(shape) != 4:
            raise ValueError(
                "BilinearUpsamplingInitializer expects a Conv2DTranspose kernel "
                f"shape, got {shape!r}."
            )
        kernel_height, kernel_width, out_channels, in_channels = shape
        if kernel_height != kernel_width:
            raise ValueError(
                "BilinearUpsamplingInitializer requires a square kernel, "
                f"got {shape!r}."
            )
        if kernel_height < 2:
            raise ValueError(
                "BilinearUpsamplingInitializer requires a kernel size of at least 2."
            )

        # Build the standard 2x bilinear filter for a transpose-conv resize.
        factor = (kernel_height + 1) // 2 if kernel_height % 2 == 1 else kernel_height // 2
        center = factor - 1 if kernel_height % 2 == 1 else factor - 0.5
        grid_y, grid_x = np.ogrid[:kernel_height, :kernel_width]
        bilinear = (1.0 - np.abs(grid_y - center) / factor) * (
            1.0 - np.abs(grid_x - center) / factor
        )

        kernel = np.zeros(shape, dtype=np.float32)
        common_channels = min(int(out_channels), int(in_channels))
        for channel_index in range(common_channels):
            kernel[:, :, channel_index, channel_index] = bilinear
        return tf.convert_to_tensor(kernel, dtype=dtype or tf.float32)

    def get_config(self):
        return {"scale": self.scale}


def _identity_3x3_initializer(channels: int):
    """Return a kernel initializer that approximates an identity 3x3 conv.

    The centre pixel copies each input channel to the matching output channel
    (weight=1.0) while all other spatial positions and cross-channel entries
    are zero.  Together with bias=0 this makes the layer a near-pass-through
    so the model starts from a sane state when a new refinement block is added
    on top of a pre-trained decoder.
    """
    # Use the serializable class so saved .keras files can be reloaded.
    return Identity3x3Initializer()


def _input_channel_count(x: tf.Tensor) -> int:
    """Return the channel count of a tensor, handling both eager and graph builds."""
    input_shape = x.shape
    try:
        rank = input_shape.rank  # type: ignore[union-attr]
        if rank is not None and input_shape[-1] is not None:
            return int(input_shape[-1])
    except AttributeError:
        if len(input_shape) >= 1 and input_shape[-1] is not None:
            return int(input_shape[-1])
    return int(tf.shape(x)[-1])


def _conv_bn_relu(
    x: tf.Tensor,
    *,
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    name: str,
) -> tf.Tensor:
    """Apply a simple Conv2D -> BatchNorm -> ReLU block.

    The block keeps the geometry model flat so tfmot can clone it for QAT
    without running into nested-Model limitations.
    """

    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    return x


def set_heatmap_encoder_trainable(model: keras.Model, trainable: bool) -> None:
    """Toggle the flat heatmap encoder layers while keeping the decoder active."""

    for layer in model.layers:
        if layer.name.startswith("heatmap_encoder_"):
            layer.trainable = bool(trainable)
        else:
            layer.trainable = True


def set_needle_direction_encoder_trainable(model: keras.Model, trainable: bool) -> None:
    """Toggle the compact direction encoder while leaving the head trainable."""

    for layer in model.layers:
        if layer.name.startswith("needle_direction_encoder_"):
            layer.trainable = bool(trainable)
        else:
            layer.trainable = True


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class NeedleValueFromDirection(keras.layers.Layer):
    """Map a unit needle-direction vector to the calibrated gauge value."""

    def __init__(
        self,
        *,
        value_min: float = -30.0,
        value_max: float = 50.0,
        cold_angle_degrees: float = 135.0,
        sweep_degrees: float = 270.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if value_max <= value_min:
            raise ValueError("value_max must be > value_min.")
        if sweep_degrees <= 0.0:
            raise ValueError("sweep_degrees must be > 0.")
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.cold_angle_degrees = float(cold_angle_degrees)
        self.sweep_degrees = float(sweep_degrees)
        self._two_pi = float(2.0 * math.pi)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Convert needle direction vectors into calibrated Celsius readings."""

        needle_xy = tf.math.l2_normalize(tf.cast(inputs, tf.float32), axis=-1)
        raw_angle = tf.atan2(needle_xy[..., 1], needle_xy[..., 0])
        cold_angle = tf.constant(math.radians(self.cold_angle_degrees), dtype=tf.float32)
        sweep = tf.constant(math.radians(self.sweep_degrees), dtype=tf.float32)
        shifted = tf.math.floormod(raw_angle - cold_angle, self._two_pi)
        fraction = tf.clip_by_value(shifted / sweep, 0.0, 1.0)
        value_span = tf.constant(self.value_max - self.value_min, dtype=tf.float32)
        return self.value_min + fraction * value_span

    def get_config(self) -> dict[str, object]:
        """Serialize the calibration constants with the layer."""

        config = super().get_config()
        config.update(
            {
                "value_min": self.value_min,
                "value_max": self.value_max,
                "cold_angle_degrees": self.cold_angle_degrees,
                "sweep_degrees": self.sweep_degrees,
            }
        )
        return config


def build_qat_friendly_heatmap_angle_model(
    input_shape=(224, 224, 3),
    *,
    heatmap_size: int = 112,
    encoder_width_multiplier: float = 1.0,
    decoder_width_multiplier: float = 1.0,
    model_name: str = "qat_friendly_heatmap_angle",
) -> keras.Model:
    """Build a flat heatmap model that is friendly to QAT cloning.

    The encoder-decoder stays entirely inside a single Functional graph so
    tfmot.quantize_model() can clone it.  It trades ImageNet transfer for a
    cleaner export path, which is a better fit for the current deployment
    constraint.
    """

    if heatmap_size != 112:
        raise ValueError(f"Unsupported heatmap_size={heatmap_size}; expected 112.")

    enc32 = _scaled_width(32, encoder_width_multiplier)
    enc48 = _scaled_width(48, encoder_width_multiplier)
    enc64 = _scaled_width(64, encoder_width_multiplier)
    enc96 = _scaled_width(96, encoder_width_multiplier)
    enc128 = _scaled_width(128, encoder_width_multiplier)
    dec96 = _scaled_width(96, decoder_width_multiplier)
    dec64 = _scaled_width(64, decoder_width_multiplier)
    dec48 = _scaled_width(48, decoder_width_multiplier)
    dec32 = _scaled_width(32, decoder_width_multiplier)

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = inputs

    # Encoder: downsample from 224x224 to 7x7 with named blocks.
    x = _conv_bn_relu(x, filters=enc32, strides=2, name="heatmap_encoder_1")
    x = _conv_bn_relu(x, filters=enc32, name="heatmap_encoder_1b")
    skip_112 = x

    x = _conv_bn_relu(x, filters=enc48, strides=2, name="heatmap_encoder_2")
    x = _conv_bn_relu(x, filters=enc48, name="heatmap_encoder_2b")
    skip_56 = x

    x = _conv_bn_relu(x, filters=enc64, strides=2, name="heatmap_encoder_3")
    x = _conv_bn_relu(x, filters=enc64, name="heatmap_encoder_3b")
    skip_28 = x

    x = _conv_bn_relu(x, filters=enc96, strides=2, name="heatmap_encoder_4")
    x = _conv_bn_relu(x, filters=enc96, name="heatmap_encoder_4b")
    skip_14 = x

    x = _conv_bn_relu(x, filters=enc128, strides=2, name="heatmap_encoder_5")
    x = _conv_bn_relu(x, filters=enc128, name="heatmap_encoder_5b")

    # Decoder: progressively recover spatial detail with flat skip connections.
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="heatmap_decoder_up_1")(x)
    x = layers.Concatenate(name="heatmap_decoder_concat_14")([x, skip_14])
    x = _conv_bn_relu(x, filters=dec96, name="heatmap_decoder_1")
    x = _conv_bn_relu(x, filters=dec96, name="heatmap_decoder_1b")

    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="heatmap_decoder_up_2")(x)
    x = layers.Concatenate(name="heatmap_decoder_concat_28")([x, skip_28])
    x = _conv_bn_relu(x, filters=dec64, name="heatmap_decoder_2")
    x = _conv_bn_relu(x, filters=dec64, name="heatmap_decoder_2b")

    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="heatmap_decoder_up_3")(x)
    x = layers.Concatenate(name="heatmap_decoder_concat_56")([x, skip_56])
    x = _conv_bn_relu(x, filters=dec48, name="heatmap_decoder_3")
    x = _conv_bn_relu(x, filters=dec48, name="heatmap_decoder_3b")

    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="heatmap_decoder_up_4")(x)
    x = layers.Concatenate(name="heatmap_decoder_concat_112")([x, skip_112])
    x = _conv_bn_relu(x, filters=dec32, name="heatmap_decoder_4")
    x = _conv_bn_relu(x, filters=dec32, name="heatmap_decoder_4b")

    center_heatmap = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="center_heatmap")(x)
    tip_heatmap = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="tip_heatmap")(x)
    confidence_features = layers.GlobalAveragePooling2D(name="heatmap_confidence_gap")(x)
    confidence = layers.Dense(1, activation="sigmoid", name="confidence")(confidence_features)

    return keras.Model(
        inputs=inputs,
        outputs=[center_heatmap, tip_heatmap, confidence],
        name=model_name,
    )


def build_qat_friendly_needle_direction_model(
    input_shape=(224, 224, 3),
    *,
    encoder_width_multiplier: float = 1.0,
    head_units: int = 64,
    head_dropout: float = 0.15,
    model_name: str = "qat_friendly_needle_direction",
) -> keras.Model:
    """Build a compact needle-direction regressor that is friendly to QAT.

    The network predicts a 2D unit vector ``(dx, dy)`` for the needle
    direction. That direct target is simpler than heatmaps, easier to quantize,
    and maps cleanly to the downstream polar-vote angle.
    """

    enc24 = _scaled_width(24, encoder_width_multiplier)
    enc32 = _scaled_width(32, encoder_width_multiplier)
    enc48 = _scaled_width(48, encoder_width_multiplier)
    enc64 = _scaled_width(64, encoder_width_multiplier)

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = keras.layers.Rescaling(1.0 / 255.0, name="needle_direction_rescale")(inputs)

    # Early stages keep the model tiny so it stays well under the board SRAM budget.
    x = _conv_bn_relu(x, filters=enc24, strides=2, name="needle_direction_encoder_1")
    x = _conv_bn_relu(x, filters=enc24, name="needle_direction_encoder_1b")
    x = keras.layers.MaxPooling2D(pool_size=2, name="needle_direction_pool_1")(x)

    x = _conv_bn_relu(x, filters=enc32, name="needle_direction_encoder_2")
    x = _conv_bn_relu(x, filters=enc32, name="needle_direction_encoder_2b")
    x = keras.layers.MaxPooling2D(pool_size=2, name="needle_direction_pool_2")(x)

    x = _conv_bn_relu(x, filters=enc48, name="needle_direction_encoder_3")
    x = _conv_bn_relu(x, filters=enc48, name="needle_direction_encoder_3b")
    x = keras.layers.MaxPooling2D(pool_size=2, name="needle_direction_pool_3")(x)

    x = _conv_bn_relu(x, filters=enc64, name="needle_direction_encoder_4")
    x = _conv_bn_relu(x, filters=enc64, name="needle_direction_encoder_4b")

    x = keras.layers.GlobalAveragePooling2D(name="needle_direction_gap")(x)
    x = keras.layers.Dense(head_units, activation="swish", name="needle_direction_dense")(x)
    x = keras.layers.Dropout(head_dropout, name="needle_direction_dropout")(x)
    needle_xy = keras.layers.Dense(2, name="needle_xy")(x)

    return keras.Model(inputs=inputs, outputs=needle_xy, name=model_name)


def build_qat_friendly_needle_direction_geometry_model(
    input_shape=(224, 224, 3),
    *,
    encoder_width_multiplier: float = 1.0,
    head_units: int = 64,
    head_dropout: float = 0.15,
    value_min: float = -30.0,
    value_max: float = 50.0,
    cold_angle_degrees: float = 135.0,
    sweep_degrees: float = 270.0,
    model_name: str = "qat_friendly_needle_direction_geometry",
) -> keras.Model:
    """Build a compact direction model with an auxiliary temperature head."""

    enc24 = _scaled_width(24, encoder_width_multiplier)
    enc32 = _scaled_width(32, encoder_width_multiplier)
    enc48 = _scaled_width(48, encoder_width_multiplier)
    enc64 = _scaled_width(64, encoder_width_multiplier)

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = keras.layers.Rescaling(1.0 / 255.0, name="needle_direction_rescale")(inputs)

    x = _conv_bn_relu(x, filters=enc24, strides=2, name="needle_direction_encoder_1")
    x = _conv_bn_relu(x, filters=enc24, name="needle_direction_encoder_1b")
    x = keras.layers.MaxPooling2D(pool_size=2, name="needle_direction_pool_1")(x)

    x = _conv_bn_relu(x, filters=enc32, name="needle_direction_encoder_2")
    x = _conv_bn_relu(x, filters=enc32, name="needle_direction_encoder_2b")
    x = keras.layers.MaxPooling2D(pool_size=2, name="needle_direction_pool_2")(x)

    x = _conv_bn_relu(x, filters=enc48, name="needle_direction_encoder_3")
    x = _conv_bn_relu(x, filters=enc48, name="needle_direction_encoder_3b")
    x = keras.layers.MaxPooling2D(pool_size=2, name="needle_direction_pool_3")(x)

    x = _conv_bn_relu(x, filters=enc64, name="needle_direction_encoder_4")
    x = _conv_bn_relu(x, filters=enc64, name="needle_direction_encoder_4b")

    x = keras.layers.GlobalAveragePooling2D(name="needle_direction_gap")(x)
    x = keras.layers.Dense(head_units, activation="swish", name="needle_direction_dense")(x)
    x = keras.layers.Dropout(head_dropout, name="needle_direction_dropout")(x)
    needle_xy = keras.layers.Dense(2, name="needle_xy")(x)
    gauge_value = NeedleValueFromDirection(
        value_min=value_min,
        value_max=value_max,
        cold_angle_degrees=cold_angle_degrees,
        sweep_degrees=sweep_degrees,
        name="gauge_value",
    )(needle_xy)

    return keras.Model(
        inputs=inputs,
        outputs={"needle_xy": needle_xy, "gauge_value": gauge_value},
        name=model_name,
    )


def _full_resolution_refine_block(
    x: tf.Tensor,
    *,
    name: str,
    filters: int | None = None,
    widen_filters: int = 0,
) -> tf.Tensor:
    """Add a full-resolution spatial refinement block before heatmap heads.

    By default keeps the same channel count as the input so that downstream
    1x1 heatmap projections can transfer their weights.  When widen_filters
    > 0, a zero-init residual bottleneck is added in parallel to the identity
    branch, giving the model more capacity without breaking transfer safety.
    """

    if filters is None:
        input_shape = x.shape
        try:
            rank = input_shape.rank  # type: ignore[union-attr]
            if rank is not None and input_shape[-1] is not None:
                filters = int(input_shape[-1])
        except AttributeError:
            if len(input_shape) >= 1 and input_shape[-1] is not None:
                filters = int(input_shape[-1])
        if filters is None:
            filters = tf.shape(x)[-1]

    # Identity branch: 3x3 conv that starts as a pass-through.
    identity_branch = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu",
        kernel_initializer=_identity_3x3_initializer(filters),
        bias_initializer="zeros",
        name=name,
    )(x)

    if widen_filters <= 0:
        return identity_branch

    # Widened residual branch: deeper representation, zero-initialised so
    # the model starts exactly where the source checkpoint left off.
    residual = layers.Conv2D(
        widen_filters,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        name=f"{name}_widen",
    )(x)
    residual = layers.Conv2D(
        filters,
        1,
        padding="same",
        activation=None,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name=f"{name}_widen_proj",
    )(residual)
    return layers.Add(name=f"{name}_fused")([identity_branch, residual])


def _full_resolution_residual_sharpen_block(
    x: tf.Tensor,
    *,
    name: str,
    filters: int | None = None,
    residual_scale: float = 0.1,
    depth: int = 2,
) -> tf.Tensor:
    """Add a zero-start residual sharpening branch at full resolution.

    The branch keeps the input tensor shape unchanged so it is safe to add on
    top of a source-transfer checkpoint. It begins as a no-op, then can learn
    a small correction that sharpens heatmap peaks without disturbing the
    v14-compatible decoder weights.

    depth: number of 3x3 conv layers inside the residual branch.  The first
    (depth-1) layers use identity-initialized kernels with relu so they
    start as near-pass-throughs.  The final layer is zero-initialized with
    no activation, guaranteeing the branch starts as a no-op regardless of
    depth.  Default 2 preserves the original behaviour.
    """

    if filters is None:
        input_shape = x.shape
        try:
            rank = input_shape.rank  # type: ignore[union-attr]
            if rank is not None and input_shape[-1] is not None:
                filters = int(input_shape[-1])
        except AttributeError:
            if len(input_shape) >= 1 and input_shape[-1] is not None:
                filters = int(input_shape[-1])
        if filters is None:
            filters = tf.shape(x)[-1]

    depth = max(int(depth), 2)
    residual = x
    # Identity-initialised layers: start as pass-through, learn spatial detail.
    for i in range(depth - 1):
        residual = layers.Conv2D(
            filters,
            3,
            padding="same",
            activation="relu",
            kernel_initializer=_identity_3x3_initializer(filters),
            bias_initializer="zeros",
            name=f"{name}_conv_{i + 1}",
        )(residual)
    # Final zero-initialised layer keeps the branch a no-op at start.
    residual = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation=None,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name=f"{name}_conv_{depth}",
    )(residual)
    residual = layers.Rescaling(residual_scale, name=f"{name}_scale")(residual)
    return layers.Add(name=name)([x, residual])


def _upsample_2x(
    x: tf.Tensor,
    *,
    filters: int,
    name: str,
    mode: str,
    hybrid_residual_scale: float = 0.2,
) -> tf.Tensor:
    """Upsample a feature map by 2x using a named decoder strategy."""

    if mode == "bilinear":
        return layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=name)(x)
    if mode == "transpose":
        # Start from a bilinear resize so the layer can learn sharper edges
        # without the random checkerboard artifacts of an uninitialized deconv.
        return layers.Conv2DTranspose(
            filters,
            kernel_size=4,
            strides=2,
            padding="same",
            activation="relu",
            kernel_initializer=BilinearUpsamplingInitializer(),
            bias_initializer="zeros",
            name=name,
        )(x)
    if mode == "hybrid_residual":
        # Keep the stable bilinear path as the anchor, then learn a small
        # residual correction with a zero-initialized transpose conv.
        bilinear = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"{name}_bilinear")(x)
        residual = layers.Conv2DTranspose(
            filters,
            kernel_size=4,
            strides=2,
            padding="same",
            activation=None,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name=f"{name}_residual",
        )(x)
        residual = layers.Rescaling(hybrid_residual_scale, name=f"{name}_residual_scale")(residual)
        return layers.Add(name=name)([bilinear, residual])
    raise ValueError(f"Unsupported upsample mode: {mode!r}")


def _fuse_decoder_skip(
    x: tf.Tensor,
    skip: tf.Tensor,
    *,
    filters: int,
    name: str,
) -> tf.Tensor:
    """Fuse a backbone skip tensor into the decoder at the same resolution.

    The skip path is projected to the decoder width before being added back in.
    That keeps the fusion cheap while still giving the model a direct
    multi-scale signal to sharpen the needle geometry.
    """

    skip = layers.Conv2D(
        filters,
        1,
        padding="same",
        activation=None,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        name=f"{name}_skip_proj",
    )(skip)
    fused = layers.Add(name=f"{name}_add")([x, skip])
    return layers.Activation("relu", name=f"{name}_relu")(fused)


def build_mobilenetv2_geometry_points_v1(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    alpha: float = 0.35,
    backbone_frozen: bool = True,
    dense_units: int = 96,
    dropout_rate: float = 0.15,
    num_outputs: int = 5,
) -> keras.Model:
    """
    Build a MobileNetV2-based model for geometry points prediction.

    Architecture:
    - Input: 224x224x3 RGB image
    - Backbone: MobileNetV2 (alpha=0.35, frozen)
    - Global average pooling
    - Dense head: dense_units (96 default)
    - Dropout: dropout_rate (0.15 default)
    - Output: num_outputs (5) values with sigmoid activation

    The 5 outputs are:
    - center_x_normalized [0, 1]
    - center_y_normalized [0, 1]
    - tip_x_normalized [0, 1]
    - tip_y_normalized [0, 1]
    - confidence [0, 1]

    Why sigmoid constraints:
    - Normalized coordinates must be in [0, 1] range
    - Sigmoid ensures outputs are bounded, preventing extreme predictions
    - This matches the coordinate transformation in geometry_crop_dataset.py

    Args:
        input_shape: Input image shape (height, width, channels)
        alpha: MobileNetV2 width multiplier (0.35 for tiny model)
        backbone_frozen: Whether to freeze backbone weights
        dense_units: Number of units in dense head
        dropout_rate: Dropout rate for regularization
        num_outputs: Number of output values (default 5)

    Returns:
        keras.Model ready for compilation and training
    """
    # Input layer
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Backbone: MobileNetV2
    # alpha=0.35 gives ~1.2M parameters for backbone
    # include_top=False removes the classification head
    # pooling='avg' gives us global average pooling directly
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet",  # Use ImageNet weights if available
        pooling="avg",
    )

    # Freeze backbone for first version
    # This prevents catastrophic forgetting and leverages pretrained features
    backbone.trainable = not backbone_frozen

    # Get backbone features
    x = backbone(inputs)

    # Dense head for coordinate regression
    # Using a moderate size head to avoid overfitting on small dataset
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_initializer="he_normal",
        name="geometry_head_dense",
    )(x)

    # Dropout for regularization
    x = layers.Dropout(dropout_rate, name="geometry_head_dropout")(x)

    # Optional: Add batch normalization for stability
    x = layers.BatchNormalization(name="geometry_head_bn")(x)

    # Second dense layer for feature refinement
    x = layers.Dense(
        dense_units // 2,
        activation="relu",
        kernel_initializer="he_normal",
        name="geometry_head_dense_2",
    )(x)

    # Output layer with sigmoid activation
    # Sigmoid constrains all outputs to [0, 1] range
    outputs = layers.Dense(
        num_outputs,
        activation="sigmoid",
        name="geometry_outputs",
    )(x)

    # Build model
    model = keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_geometry_points_v1")

    return model


def compile_geometry_model(
    model: keras.Model,
    learning_rate: float = 1e-4,
    coordinate_loss_weight: float = 1.0,
    confidence_loss_weight: float = 0.1,
) -> None:
    """
    Compile the geometry model with appropriate loss and metrics.

    Loss function:
    - MSE for coordinate outputs (first 4 values)
    - Binary crossentropy for confidence (5th value)
    - Weighted combination of both

    Metrics:
    - MAE for coordinates (interpretable in pixel space)
    - Binary accuracy for confidence

    Args:
        model: The geometry model to compile
        learning_rate: Initial learning rate
        coordinate_loss_weight: Weight for coordinate MSE loss
        confidence_loss_weight: Weight for confidence BCE loss
    """

    # Custom loss function
    def geometry_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute combined geometry loss.

        Args:
            y_true: Ground truth [center_x, center_y, tip_x, tip_y, confidence]
            y_pred: Predicted [center_x, center_y, tip_x, tip_y, confidence]

        Returns:
            Scalar loss value
        """
        # Split coordinates and confidence
        true_coords = y_true[:, :4]  # center_x, center_y, tip_x, tip_y
        pred_coords = y_pred[:, :4]
        true_conf = y_true[:, 4]  # confidence
        pred_conf = y_pred[:, 4]

        # Coordinate loss: MSE
        coord_loss = keras.losses.mean_squared_error(true_coords, pred_coords)
        coord_loss = tf.reduce_mean(coord_loss)

        # Confidence loss: Binary crossentropy
        conf_loss = keras.losses.binary_crossentropy(true_conf, pred_conf)
        conf_loss = tf.reduce_mean(conf_loss)

        # Combined loss
        total_loss = (
            coordinate_loss_weight * coord_loss +
            confidence_loss_weight * conf_loss
        )

        return total_loss

    # Compile with Adam optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=geometry_loss,
        metrics=[
            keras.metrics.MeanAbsoluteError(name="coord_mae"),
            keras.metrics.MeanSquaredError(name="coord_mse"),
        ],
    )


def create_learning_rate_schedule(
    initial_lr: float = 1e-4,
    warmup_epochs: int = 5,
    decay_epochs: int = 40,
) -> keras.callbacks.LearningRateScheduler:
    """
    Create a learning rate schedule with warmup and decay.

    Schedule:
    - Epochs 0-5: Linear warmup from 0 to initial_lr
    - Epochs 5-45: Constant at initial_lr
    - Epochs 45+: Exponential decay

    Args:
        initial_lr: Peak learning rate after warmup
        warmup_epochs: Number of warmup epochs
        decay_epochs: Epoch at which decay starts

    Returns:
        LearningRateScheduler callback
    """
    def lr_schedule(epoch: int, current_lr: float) -> float:
        if epoch < warmup_epochs:
            # Linear warmup
            return initial_lr * (epoch + 1) / warmup_epochs
        elif epoch < decay_epochs:
            # Constant
            return initial_lr
        else:
            # Exponential decay
            decay_rate = 0.95
            decay_epoch = epoch - decay_epochs
            return initial_lr * (decay_rate ** decay_epoch)

    return keras.callbacks.LearningRateScheduler(lr_schedule, verbose=False)


def get_model_summary(model: keras.Model) -> str:
    """
    Get a summary of the model architecture.

    Args:
        model: The model to summarize

    Returns:
        String summary of model architecture
    """
    summary_lines = []

    # Count parameters
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params

    summary_lines.append(f"Model: {model.name}")
    summary_lines.append(f"Total parameters: {total_params:,}")
    summary_lines.append(f"Trainable parameters: {trainable_params:,}")
    summary_lines.append(f"Non-trainable parameters: {non_trainable_params:,}")
    summary_lines.append("")

    # Layer summary
    summary_lines.append("Layer summary:")
    summary_lines.append("-" * 60)
    for layer in model.layers:
        layer_params = layer.count_params()
        trainable = "trainable" if layer.trainable else "frozen"
        summary_lines.append(
            f"  {layer.name:30s} {str(layer.output_shape):20s} {layer_params:>10,} ({trainable})"
        )

    return "\n".join(summary_lines)


def build_mobilenetv2_geometry_heatmap_v1(
    input_shape=(224, 224, 3),
    alpha=0.35,
    backbone_frozen=True,
    heatmap_size=56,
    learning_rate=1e-4,
):
    """Build a MobileNetV2-based model for heatmap-based geometry prediction."""
    inputs = keras.Input(shape=input_shape, name='input_image')

    # Backbone: MobileNetV2
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet" if pretrained else None,
        pooling=None,
    )
    backbone.trainable = not backbone_frozen

    x = backbone(inputs)

    # Decoder: Upsample to heatmap size
    x = layers.Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)

    center_heatmap = layers.Conv2D(1, 1, padding='same', activation='sigmoid', name='center_heatmap')(x)
    tip_heatmap = layers.Conv2D(1, 1, padding='same', activation='sigmoid', name='tip_heatmap')(x)

    conf_features = layers.GlobalAveragePooling2D()(x)
    confidence = layers.Dense(1, activation='sigmoid', name='confidence')(conf_features)

    model = keras.Model(
        inputs=inputs,
        outputs=[center_heatmap, tip_heatmap, confidence],
        name='mobilenetv2_geometry_heatmap_v1',
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'center_heatmap': keras.losses.MeanSquaredError(),
            'tip_heatmap': keras.losses.MeanSquaredError(),
            'confidence': keras.losses.BinaryCrossentropy(),
        },
        loss_weights={
            'center_heatmap': 1.0,
            'tip_heatmap': 1.0,
            'confidence': 0.1,
        },
        metrics={
            'center_heatmap': [keras.metrics.MeanAbsoluteError(name='mae')],
            'tip_heatmap': [keras.metrics.MeanAbsoluteError(name='mae')],
        },
    )

    return model


def _build_mobilenetv2_geometry_heatmap_decoder(
    *,
    input_shape=(224, 224, 3),
    alpha=0.35,
    backbone_frozen=True,
    heatmap_size=56,
    decoder_channels: tuple[int, ...] = (128, 64, 32),
    model_name: str = "mobilenetv2_geometry_heatmap",
    pretrained: bool = True,
):
    """Build a compact MobileNetV2 heatmap decoder with a configurable output size.

    For 112x112 outputs, the decoder adds a tiny full-resolution refinement
    block before the final heatmap heads so the model can sharpen peaks
    without widening the whole decoder.
    """

    if heatmap_size not in (56, 112):
        raise ValueError(f"Unsupported heatmap_size={heatmap_size}; expected 56 or 112.")

    inputs = keras.Input(shape=input_shape, name="input_image")

    # Keep the backbone shallow and reusable so we can transfer weights across v2/v3/v4.
    # Honor the caller's initialization choice so "random" runs do not spend
    # time loading ImageNet weights and can train from a true scratch start.
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet" if pretrained else None,
        pooling=None,
    )
    backbone.trainable = not backbone_frozen

    x = backbone(inputs)
    current_size = 7  # MobileNetV2 reduces 224x224 to 7x7 at the penultimate stage.

    # Build a tiny progressive decoder so the output resolution stays friendly for INT8 export.
    for stage, channels in enumerate(decoder_channels, start=1):
        x = layers.Conv2D(
            channels,
            3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name=f"geometry_decoder_conv_{stage}",
        )(x)
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"geometry_decoder_up_{stage}")(x)
        current_size *= 2

    if heatmap_size == 112 and current_size != 112:
        x = layers.Conv2D(
            32,
            3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name="geometry_decoder_refine_112",
        )(x)
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="geometry_decoder_up_4")(x)
        current_size *= 2

    if current_size != heatmap_size:
        raise RuntimeError(f"Decoder reached {current_size}x{current_size}, expected {heatmap_size}x{heatmap_size}.")

    if heatmap_size == 112:
        # Give the 112x112 tensor one learned spatial pass before the
        # final 1x1 heatmap projections.
        x = _full_resolution_refine_block(
            x,
            name="geometry_decoder_refine_fullres",
        )

    center_heatmap = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="center_heatmap")(x)
    tip_heatmap = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="tip_heatmap")(x)

    confidence_features = layers.GlobalAveragePooling2D(name="geometry_confidence_gap")(x)
    confidence = layers.Dense(1, activation="sigmoid", name="confidence")(confidence_features)

    return keras.Model(inputs=inputs, outputs=[center_heatmap, tip_heatmap, confidence], name=model_name)


def build_mobilenetv2_geometry_heatmap_v4_112(
    input_shape=(224, 224, 3),
    alpha=0.35,
    backbone_frozen=True,
    include_aux_coords=False,
    aux_head_size="small",
    aux_head_type="none",
    decoder_width_multiplier: float = 1.0,
    decoder_upsample_mode: str = "bilinear",
    decoder_multiscale_fusion: bool = False,
    decoder_fullres_residual_block: bool = False,
    decoder_fullres_residual_scale: float = 0.1,
    decoder_fullres_residual_depth: int = 2,
    hybrid_residual_scale: float = 0.2,
    decoder_refine_widen_filters: int = 0,
    decoder_fullres_dw_residual: bool = False,
    decoder_fullres_dw_residual_scale: float = 0.05,
    subpixel_refinement_head: bool = False,
    pretrained: bool = True,
):
    """Build the 112x112 geometry heatmap model for tip-stable INT8 deployment.

    The 112x112 head keeps the v3-style decoder blocks so we can still transfer
    compatible weights from the canonical v3 checkpoint, and it can optionally
    fuse 14x14 / 28x28 / 56x56 MobileNetV2 skips into the decoder. A tiny
    two-layer full-resolution refinement block then sharpens the 112x112
    tensor before the final heatmap projections, giving the decoder a cleaner
    spatial signal than a plain bilinear 56->112 wrapper while keeping the
    model compact enough for embedded deployment.

    Auxiliary head types (aux_head_type):
      - "none": no aux head (default). include_aux_coords is ignored.
      - "gap": GAP-based aux coords regression head predicting
        [center_x_norm, center_y_norm, tip_x_norm, tip_y_norm] from pooled
        decoder features (same as include_aux_coords=True).
        aux_head_size: "small" (Dense(64)->Dense(4)) or "large" (Dense(128)->Dense(64)->Dense(4))
      - "local_offset": spatially-aware offset head that branches from the
        112x112 decoder tensor and predicts per-pixel dx/dy offsets via a
        small conv head. Output is 112x112x4 with channels
        [center_dx, center_dy, tip_dx, tip_dy] in tanh range [-1, 1].
      - decoder_width_multiplier: scales the decoder and auxiliary head widths
        while keeping the spatial resolution fixed. This lets us spend more
        capacity on richer geometry features without changing the heatmap size.
      - decoder_upsample_mode: final 56x56 -> 112x112 resize strategy. Use
        "bilinear" for the existing deterministic resize or "transpose" for a
        learnable Conv2DTranspose stage initialized to bilinear weights, or
        "hybrid_residual" for bilinear upsampling plus a small learnable
        residual transpose-conv correction.
      - decoder_multiscale_fusion: if True, fuse 14x14, 28x28, and 56x56
        MobileNetV2 skip features into the decoder with residual adds. This
        gives the network a more UNet-like multi-scale path for sharper
        geometry and better locality.
      - decoder_fullres_residual_block: if True, add a small zero-start
        residual sharpening branch after the existing full-resolution refine
        block. This keeps v14 transfer compatibility while giving the model an
        extra place to learn fine peak corrections.

    For backward compat, include_aux_coords=True is equivalent to aux_head_type="gap".
    """

    # Backward compat: include_aux_coords=True maps to aux_head_type="gap"
    if include_aux_coords and aux_head_type == "none":
        aux_head_type = "gap"

    inputs = keras.Input(shape=input_shape, name="input_image")

    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet" if pretrained else None,
        pooling=None,
    )
    backbone.trainable = not backbone_frozen

    feature_extractor = keras.Model(
        inputs=backbone.input,
        outputs=[
            backbone.get_layer("block_13_expand_relu").output,
            backbone.get_layer("block_6_expand_relu").output,
            backbone.get_layer("block_3_expand_relu").output,
            backbone.output,
        ],
        name="mobilenetv2_geometry_heatmap_v4_112_backbone",
    )

    skip_14, skip_28, skip_56, x = feature_extractor(inputs)

    decoder_conv_1_filters = _scaled_width(128, decoder_width_multiplier)
    decoder_conv_2_filters = _scaled_width(64, decoder_width_multiplier)
    decoder_conv_3_filters = _scaled_width(32, decoder_width_multiplier)
    skip_56_filters = _scaled_width(16, decoder_width_multiplier)
    decoder_refine_filters = _scaled_width(32, decoder_width_multiplier)

    x = layers.Conv2D(
        decoder_conv_1_filters,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="geometry_decoder_conv_1",
    )(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="geometry_decoder_up_1")(x)
    if decoder_multiscale_fusion:
        x = _fuse_decoder_skip(
            x,
            skip_14,
            filters=decoder_conv_1_filters,
            name="geometry_decoder_fuse_14",
        )

    x = layers.Conv2D(
        decoder_conv_2_filters,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="geometry_decoder_conv_2",
    )(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="geometry_decoder_up_2")(x)
    if decoder_multiscale_fusion:
        x = _fuse_decoder_skip(
            x,
            skip_28,
            filters=decoder_conv_2_filters,
            name="geometry_decoder_fuse_28",
        )

    x = layers.Conv2D(
        decoder_conv_3_filters,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="geometry_decoder_conv_3",
    )(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="geometry_decoder_up_3")(x)
    if decoder_multiscale_fusion:
        x = _fuse_decoder_skip(
            x,
            skip_56,
            filters=decoder_conv_3_filters,
            name="geometry_decoder_fuse_56",
        )
    else:
        skip_56 = layers.Conv2D(
            skip_56_filters,
            1,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name="geometry_decoder_skip_56",
        )(skip_56)
        x = layers.Concatenate(name="geometry_decoder_concat_56")([x, skip_56])
    x = layers.Conv2D(
        decoder_refine_filters,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="geometry_decoder_refine_112",
    )(x)
    x = _upsample_2x(
        x,
        filters=decoder_refine_filters,
        name="geometry_decoder_up_4",
        mode=decoder_upsample_mode,
        hybrid_residual_scale=hybrid_residual_scale,
    )

    # Sharpen the full-resolution feature map before the 1x1 heatmap heads.
    x = _full_resolution_refine_block(
        x,
        name="geometry_decoder_refine_fullres",
        widen_filters=decoder_refine_widen_filters,
    )
    if decoder_fullres_residual_block:
        x = _full_resolution_residual_sharpen_block(
            x,
            name="geometry_decoder_refine_fullres_residual",
            residual_scale=decoder_fullres_residual_scale,
            depth=decoder_fullres_residual_depth,
        )

    if decoder_fullres_dw_residual:
        # Lightweight HRNet-lite style detail branch: zero-start depthwise
        # separable residual that learns spatial corrections without adding
        # meaningful activation overhead.  Zero-init across the board so it
        # starts as a no-op (transfer-safe).
        input_filters = _input_channel_count(x)
        dw_residual = layers.DepthwiseConv2D(
            3, padding="same", activation=None,
            depthwise_initializer="zeros", bias_initializer="zeros",
            name="geometry_decoder_dw_residual_depthwise",
        )(x)
        dw_residual = layers.Conv2D(
            input_filters, 1, padding="same", activation=None,
            kernel_initializer="zeros", bias_initializer="zeros",
            name="geometry_decoder_dw_residual_pointwise",
        )(dw_residual)
        dw_residual = layers.Rescaling(
            decoder_fullres_dw_residual_scale,
            name="geometry_decoder_dw_residual_scale",
        )(dw_residual)
        x = layers.Add(name="geometry_decoder_dw_residual_add")([x, dw_residual])

    center_heatmap = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="center_heatmap")(x)
    tip_heatmap = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="tip_heatmap")(x)

    confidence_features = layers.GlobalAveragePooling2D(name="geometry_confidence_gap")(x)
    confidence = layers.Dense(1, activation="sigmoid", name="confidence")(confidence_features)

    outputs: list[keras.layers.Layer] = [center_heatmap, tip_heatmap, confidence]

    if subpixel_refinement_head:
        # Tiny Dense head predicting residual offsets for center and tip.
        # Trained against the difference between soft-argmax peak and ground
        # truth, so the heatmaps learn coarse position and this head learns
        # the sub-pixel correction.  Output is tanh [-1, 1], scaled by a
        # configurable factor during loss computation.
        offset_features = layers.Dense(
            _scaled_width(32, decoder_width_multiplier),
            activation="relu",
            kernel_initializer="he_normal",
            name="subpixel_refinement_dense",
        )(confidence_features)
        subpixel_offsets = layers.Dense(
            4,
            activation="tanh",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="subpixel_offsets",
        )(offset_features)
        outputs.append(subpixel_offsets)

    if aux_head_type == "gap":
        if aux_head_size == "large":
            aux_coords = layers.Dense(
                _scaled_width(128, decoder_width_multiplier),
                activation="relu",
                kernel_initializer="he_normal",
                name="aux_coords_dense_1",
            )(confidence_features)
            aux_coords = layers.Dense(
                _scaled_width(64, decoder_width_multiplier),
                activation="relu",
                kernel_initializer="he_normal",
                name="aux_coords_dense_2",
            )(aux_coords)
        else:
            aux_coords = layers.Dense(
                _scaled_width(64, decoder_width_multiplier),
                activation="relu",
                kernel_initializer="he_normal",
                name="aux_coords_dense",
            )(confidence_features)
        aux_coords = layers.Dense(
            4,
            activation="sigmoid",
            name="aux_coords",
        )(aux_coords)
        outputs.append(aux_coords)

    if aux_head_type == "local_offset":
        # Spatially-aware local offset head branched from the 112x112 decoder
        # tensor x (before the per-pixel heatmap heads).  Produces per-pixel
        # dx/dy offsets in tanh range [-1, 1] for both center and tip keypoints.
        # Output channels: [center_dx, center_dy, tip_dx, tip_dy].
        aux_offset = layers.Conv2D(
            decoder_refine_filters,
            3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name="aux_offset_conv_1",
        )(x)
        aux_offset = layers.Conv2D(
            skip_56_filters,
            3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name="aux_offset_conv_2",
        )(aux_offset)
        aux_offset_map = layers.Conv2D(
            4,
            1,
            padding="same",
            activation="tanh",
            name="aux_offset_map",
        )(aux_offset)
        outputs.append(aux_offset_map)

    if aux_head_type == "axis_simcc":
        # SimCC-style axis-marginal logit head.  Branches from the decoder
        # GAP features (*not* spatial Conv2D + mean/max pool) to avoid gradient
        # dilution.  Dense layers provide dense gradient flow to every weight.
        # Outputs 1D logit vectors for center/tip x/y coordinates.
        # Shape: (batch, 4, 112) with semantic order [center_x, center_y, tip_x, tip_y].
        axis_feat = layers.Dense(
            _scaled_width(128, decoder_width_multiplier),
            activation="relu",
            kernel_initializer="he_normal",
            name="axis_simcc_dense_1",
        )(confidence_features)
        axis_raw = layers.Dense(
            4 * 112,
            activation="linear",
            kernel_initializer="he_normal",
            name="axis_simcc_logits_raw",
        )(axis_feat)
        axis_logits = layers.Reshape((4, 112), name="axis_logits")(axis_raw)
        outputs.append(axis_logits)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mobilenetv2_geometry_heatmap_v4_112",
    )


def build_heatmap_angle_model(
    input_shape=(224, 224, 3),
    alpha=0.35,
    backbone_frozen=False,
    heatmap_size=112,
) -> keras.Model:
    """Build a heatmap model for angle prediction from cropped gauge images.

    This model predicts center and tip heatmaps from which the needle angle
    is derived via soft-argmax decoding and atan2. The architecture is based
    on the proven v4 112x112 heatmap model but simplified by removing the
    auxiliary heads that did not improve INT8 robustness in Phase 11
    experiments. It also keeps the tiny two-layer full-resolution refinement
    block before the final heatmap heads so peak sharpening stays consistent
    with the main geometry family.

    Architecture:
    - Backbone: MobileNetV2 (alpha=0.35, optionally ImageNet pretrained)
    - Decoder: Progressive upsampling with 56x56 skip connection
    - Heads:
        - center_heatmap (112x112x1, sigmoid) — Gaussian peak at dial center
        - tip_heatmap (112x112x1, sigmoid) — Gaussian peak at needle tip
        - confidence (scalar, sigmoid) — whether needle is visible

    Inference pipeline:
    1. Predict heatmaps from 224x224 cropped input
    2. Decode via soft-argmax: center_x, center_y, tip_x, tip_y
    3. Compute angle: atan2(tip_y - center_y, tip_x - center_x)
    4. Map to temperature: celsius_from_inner_dial_angle_degrees(angle)

    Why heatmap approach:
    - Spatial heatmaps provide explicit needle geometry supervision
    - Soft-argmax decoding is differentiable, enabling end-to-end training
    - Angle is derived geometrically, ensuring circular consistency
    - More interpretable than direct angle regression

    Args:
        input_shape: Input image shape (height, width, channels)
        alpha: MobileNetV2 width multiplier (0.35 for compact model)
        backbone_frozen: Whether to freeze backbone weights
            (False = fine-tune, True = use pretrained features only)
        heatmap_size: Output heatmap resolution (default 112)

    Returns:
        keras.Model with outputs [center_heatmap, tip_heatmap, confidence]
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Backbone: MobileNetV2 with ImageNet weights
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    backbone.trainable = not backbone_frozen

    # Extract features at multiple scales for skip connection
    feature_extractor = keras.Model(
        inputs=backbone.input,
        outputs=[
            backbone.get_layer("block_3_expand_relu").output,  # 56x56 features
            backbone.output,  # 7x7 features
        ],
        name="mobilenetv2_angle_backbone",
    )

    skip_56, x = feature_extractor(inputs)

    # Progressive decoder with bilinear upsampling
    # Block 1: 7x7 -> 14x14
    x = layers.Conv2D(
        128,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="angle_decoder_conv_1",
    )(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="angle_decoder_up_1")(x)

    # Block 2: 14x14 -> 28x28
    x = layers.Conv2D(
        64,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="angle_decoder_conv_2",
    )(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="angle_decoder_up_2")(x)

    # Block 3: 28x28 -> 56x56
    x = layers.Conv2D(
        32,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="angle_decoder_conv_3",
    )(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="angle_decoder_up_3")(x)

    # Skip connection from block_3_expand_relu (56x56)
    # Project to matching channels and concatenate
    skip_56 = layers.Conv2D(
        16,
        1,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="angle_decoder_skip_56",
    )(skip_56)
    x = layers.Concatenate(name="angle_decoder_concat_56")([x, skip_56])

    # Refine and upsample to 112x112
    x = layers.Conv2D(
        32,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="angle_decoder_refine_112",
    )(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="angle_decoder_up_4")(x)

    # Add a compact full-resolution sharpening stage before the
    # heatmap heads.
    x = _full_resolution_refine_block(
        x,
        name="angle_decoder_refine_fullres",
    )

    # Heatmap heads
    center_heatmap = layers.Conv2D(
        1, 1, padding="same", activation="sigmoid", name="center_heatmap"
    )(x)
    tip_heatmap = layers.Conv2D(
        1, 1, padding="same", activation="sigmoid", name="tip_heatmap"
    )(x)

    # Confidence head from pooled features
    confidence_features = layers.GlobalAveragePooling2D(name="angle_confidence_gap")(x)
    confidence = layers.Dense(1, activation="sigmoid", name="confidence")(confidence_features)

    return keras.Model(
        inputs=inputs,
        outputs=[center_heatmap, tip_heatmap, confidence],
        name="mobilenetv2_heatmap_angle",
    )
