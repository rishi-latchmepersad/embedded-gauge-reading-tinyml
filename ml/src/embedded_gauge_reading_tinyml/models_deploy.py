"""
TFLite-compatible deployment models for gauge geometry prediction.

All layers use only standard Keras ops that survive TFLite conversion:
  - Conv2D, DepthwiseConv2D, BatchNormalization, ReLU
  - GlobalAveragePooling2D, Dense, Softmax, Sigmoid
  - Add, Rescaling
  - NO Conv2DTranspose, Lambda, custom initializers, or custom layers
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_simcc_gauge_model(
    input_shape=(224, 224, 3),
    alpha=0.35,
    num_bins=112,
    bottleneck_units=96,
    pretrained=True,
):
    """TFLite-safe SimCC-style gauge geometry model.

    Architecture:
      MobileNetV2 backbone -> GAP -> bottleneck Dense -> 4 SimCC classification
      heads (1D distributions over num_bins for center_x, center_y, tip_x, tip_y)
      plus a scalar confidence head.

    Each SimCC head outputs a softmax distribution over `num_bins` bins.
    The expected bin index (soft-argmax) is converted to a normalised [0,1]
    coordinate by dividing by (num_bins - 1).  This replaces the entire
    decoder upsampling chain with a simple classification head and is
    structurally similar to SimCC / RTMPose.

    Args:
        input_shape: Image shape (H, W, C).
        alpha: MobileNetV2 width multiplier.
        num_bins: Number of 1D classification bins per SimCC head.
        bottleneck_units: Units in the shared Dense before the SimCC heads.
        pretrained: Whether to load ImageNet weights for the backbone.

    Returns:
        A Keras Model with 5 outputs:
            [center_x_simcc, center_y_simcc, tip_x_simcc, tip_y_simcc, confidence]
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # ------------------------------------------------------------------
    # Backbone: MobileNetV2 (standard Keras application, TFLite-safe)
    # ------------------------------------------------------------------
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet" if pretrained else None,
        pooling="avg",  # GAP built-in
    )
    backbone.trainable = True
    x = backbone(inputs)

    # ------------------------------------------------------------------
    # Shared bottleneck
    # ------------------------------------------------------------------
    x = layers.Dense(
        bottleneck_units,
        activation="relu",
        kernel_initializer="he_normal",
        name="simcc_bottleneck",
    )(x)

    # ------------------------------------------------------------------
    # SimCC classification heads: 1D softmax per coordinate
    # ------------------------------------------------------------------
    center_x_logits = layers.Dense(
        num_bins, activation="softmax", kernel_initializer="he_normal",
        name="center_x_simcc",
    )(x)
    center_y_logits = layers.Dense(
        num_bins, activation="softmax", kernel_initializer="he_normal",
        name="center_y_simcc",
    )(x)
    tip_x_logits = layers.Dense(
        num_bins, activation="softmax", kernel_initializer="he_normal",
        name="tip_x_simcc",
    )(x)
    tip_y_logits = layers.Dense(
        num_bins, activation="softmax", kernel_initializer="he_normal",
        name="tip_y_simcc",
    )(x)

    # ------------------------------------------------------------------
    # Scalar confidence
    # ------------------------------------------------------------------
    confidence = layers.Dense(1, activation="sigmoid", name="confidence")(x)

    return keras.Model(
        inputs=inputs,
        outputs=[center_x_logits, center_y_logits, tip_x_logits, tip_y_logits, confidence],
        name="simcc_gauge_v1",
    )


def _build_axis_simcc_head(
    features: tf.Tensor,
    *,
    axis: str,
    num_bins: int,
    head_channels: int,
    name_prefix: str,
) -> tf.Tensor:
    """Project a shared spatial map into one 1D SimCC coordinate head.

    The head keeps the 14x14 spatial trunk intact as long as possible:
    first collapse the orthogonal axis with average pooling, then expand the
    remaining axis to 112 bins with a lightweight resize + conv stack.
    """
    if axis == "x":
        x = layers.AveragePooling2D(
            pool_size=(14, 1),
            strides=(14, 1),
            padding="valid",
            name=f"{name_prefix}_collapse_height",
        )(features)
        x = layers.UpSampling2D(
            size=(1, 8),
            interpolation="bilinear",
            name=f"{name_prefix}_expand_width",
        )(x)
    elif axis == "y":
        x = layers.AveragePooling2D(
            pool_size=(1, 14),
            strides=(1, 14),
            padding="valid",
            name=f"{name_prefix}_collapse_width",
        )(features)
        x = layers.UpSampling2D(
            size=(8, 1),
            interpolation="bilinear",
            name=f"{name_prefix}_expand_height",
        )(x)
    else:
        raise ValueError(f"Unsupported axis '{axis}'.")

    # Give each axis a tiny amount of local context before the final logits.
    x = layers.Conv2D(
        head_channels,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_conv_1",
    )(x)
    x = layers.Conv2D(
        head_channels,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_conv_2",
    )(x)
    x = layers.Conv2D(
        1,
        1,
        padding="same",
        activation=None,
        kernel_initializer="he_normal",
        name=f"{name_prefix}_logits_2d",
    )(x)
    x = layers.Flatten(name=f"{name_prefix}_flatten")(x)
    return layers.Softmax(name=f"{name_prefix}_simcc")(x)


def build_spatial_simcc_gauge_model(
    input_shape=(224, 224, 3),
    alpha=0.35,
    num_bins=112,
    spatial_channels=96,
    pretrained=True,
):
    """TFLite-safe SimCC model with an explicit 14x14 spatial trunk.

    This variant keeps a small shared feature map alive before the SimCC
    heads instead of collapsing everything with GAP.  The x/y heads each
    preserve one spatial axis, resize it to 112 bins, and then emit a 1D
    softmax distribution.  The model stays fully TFLite-friendly because it
    uses only standard Conv2D, AveragePooling2D, UpSampling2D, Flatten, Dense,
    Softmax, and Sigmoid layers.
    """
    # ------------------------------------------------------------------
    # Backbone: keep the final feature map spatial instead of pooling it.
    # We use backbone.layers[-1].output directly (NOT backbone(inputs))
    # so the backbone's internal layers become direct children of the
    # outer functional graph, avoiding the nested-sub-model issue that
    # breaks tfmot.quantization.keras.quantize_model().
    # ------------------------------------------------------------------
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet" if pretrained else None,
        pooling=None,
    )
    backbone.trainable = True
    x = backbone.layers[-1].output

    # ------------------------------------------------------------------
    # Lightweight 14x14 spatial trunk.
    # ------------------------------------------------------------------
    x = layers.Conv2D(
        spatial_channels,
        1,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="spatial_trunk_proj",
    )(x)
    x = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        name="spatial_trunk_up_1",
    )(x)
    x = layers.Conv2D(
        spatial_channels,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="spatial_trunk_conv_1",
    )(x)
    x = layers.Conv2D(
        spatial_channels,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="spatial_trunk_conv_2",
    )(x)

    # ------------------------------------------------------------------
    # SimCC heads: preserve one axis at a time instead of collapsing space.
    # ------------------------------------------------------------------
    center_x_logits = _build_axis_simcc_head(
        x,
        axis="x",
        num_bins=num_bins,
        head_channels=spatial_channels,
        name_prefix="center_x",
    )
    center_y_logits = _build_axis_simcc_head(
        x,
        axis="y",
        num_bins=num_bins,
        head_channels=spatial_channels,
        name_prefix="center_y",
    )
    tip_x_logits = _build_axis_simcc_head(
        x,
        axis="x",
        num_bins=num_bins,
        head_channels=spatial_channels,
        name_prefix="tip_x",
    )
    tip_y_logits = _build_axis_simcc_head(
        x,
        axis="y",
        num_bins=num_bins,
        head_channels=spatial_channels,
        name_prefix="tip_y",
    )

    # ------------------------------------------------------------------
    # Scalar confidence from the shared spatial trunk.
    # ------------------------------------------------------------------
    confidence = layers.GlobalAveragePooling2D(name="confidence_gap")(x)
    confidence = layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer="he_normal",
        name="confidence",
    )(confidence)

    return keras.Model(
        inputs=backbone.input,
        outputs=[center_x_logits, center_y_logits, tip_x_logits, tip_y_logits, confidence],
        name="simcc_gauge_v2_spatial",
    )
