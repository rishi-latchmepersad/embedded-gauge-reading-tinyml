"""QAT-friendly combined OBB and center + SimCC models for gauge localization.

This module stays in the ``tf_keras`` stack so the resulting Functional models
can be wrapped with ``tfmot.quantization.keras.quantize_model()`` during QAT.
The shared backbone is MobileNetV2 with a small 14x14 spatial trunk so the
model can learn both a coarse oriented box and finer center/tip geometry from
the same 224x224 frame without blowing the activation budget.

The center-detector variant keeps the same SimCC heads but replaces the OBB
regression branch with a compact gauge-face center regressor that is easier to
quantize and easier to distill from a teacher model.

Inputs are expected to be preprocessed to the MobileNetV2 ``[-1, 1]`` range
outside the graph so that the model remains cloneable for QAT.
"""

from __future__ import annotations

from typing import Final

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
from tf_keras.src.applications import imagenet_utils


_DEFAULT_NUM_BINS: Final[int] = 112


def _make_divisible(value: float, divisor: int, min_value: int | None = None) -> int:
    """Match the filter rounding used by the official MobileNetV2 builder."""

    if min_value is None:
        min_value = divisor
    rounded = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return rounded


def _conv_bn_relu(
    inputs: tf.Tensor,
    filters: int,
    *,
    kernel_size: int,
    strides: tuple[int, int],
    name: str,
    created_layers: dict[str, keras.layers.Layer],
) -> tf.Tensor:
    """Apply Conv2D -> BN -> ReLU6 as a reusable MobileNetV2 stem block."""

    conv = layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=name,
    )
    bn = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-3,
        momentum=0.999,
        name=f"{name}_bn",
    )
    relu = layers.ReLU(6.0, name=f"{name}_relu")
    created_layers[conv.name] = conv
    created_layers[bn.name] = bn
    created_layers[relu.name] = relu
    x = conv(inputs)
    x = bn(x)
    x = relu(x)
    return x


def _inverted_res_block(
    inputs: tf.Tensor,
    *,
    filters: int,
    alpha: float,
    stride: int,
    expansion: int,
    block_id: int,
    created_layers: dict[str, keras.layers.Layer],
) -> tf.Tensor:
    """Replicate the official MobileNetV2 inverted residual block."""

    in_channels = int(inputs.shape[-1])
    pointwise_filters = _make_divisible(filters * alpha, 8)
    x = inputs
    prefix = f"block_{block_id}_"

    if block_id:
        expand = layers.Conv2D(
            expansion * in_channels,
            kernel_size=1,
            padding="same",
            use_bias=False,
            activation=None,
            name=prefix + "expand",
        )
        expand_bn = layers.BatchNormalization(
            axis=-1,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "expand_BN",
        )
        expand_relu = layers.ReLU(6.0, name=prefix + "expand_relu")
        created_layers[expand.name] = expand
        created_layers[expand_bn.name] = expand_bn
        created_layers[expand_relu.name] = expand_relu
        x = expand(x)
        x = expand_bn(x)
        x = expand_relu(x)
    else:
        prefix = "expanded_conv_"

    if stride == 2:
        pad = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, 3),
            name=prefix + "pad",
        )
        created_layers[pad.name] = pad
        x = pad(x)

    depthwise = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        activation=None,
        use_bias=False,
        padding="same" if stride == 1 else "valid",
        name=prefix + "depthwise",
    )
    depthwise_bn = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "depthwise_BN",
    )
    depthwise_relu = layers.ReLU(6.0, name=prefix + "depthwise_relu")
    project = layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        activation=None,
        name=prefix + "project",
    )
    project_bn = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "project_BN",
    )
    created_layers[depthwise.name] = depthwise
    created_layers[depthwise_bn.name] = depthwise_bn
    created_layers[depthwise_relu.name] = depthwise_relu
    created_layers[project.name] = project
    created_layers[project_bn.name] = project_bn

    x = depthwise(x)
    x = depthwise_bn(x)
    x = depthwise_relu(x)
    x = project(x)
    x = project_bn(x)

    if in_channels == pointwise_filters and stride == 1:
        add = layers.Add(name=prefix + "add")
        created_layers[add.name] = add
        return add([inputs, x])
    return x


def _build_mobilenetv2_feature_tensor(
    inputs: tf.Tensor,
    *,
    alpha: float,
    created_layers: dict[str, keras.layers.Layer],
) -> tf.Tensor:
    """Build a flat MobileNetV2 feature extractor directly on ``inputs``."""

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = _conv_bn_relu(
        inputs,
        first_block_filters,
        kernel_size=3,
        strides=(2, 2),
        name="Conv1",
        created_layers=created_layers,
    )

    x = _inverted_res_block(
        x,
        filters=16,
        alpha=alpha,
        stride=1,
        expansion=1,
        block_id=0,
        created_layers=created_layers,
    )
    x = _inverted_res_block(
        x,
        filters=24,
        alpha=alpha,
        stride=2,
        expansion=6,
        block_id=1,
        created_layers=created_layers,
    )
    x = _inverted_res_block(
        x,
        filters=24,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=2,
        created_layers=created_layers,
    )
    x = _inverted_res_block(
        x,
        filters=32,
        alpha=alpha,
        stride=2,
        expansion=6,
        block_id=3,
        created_layers=created_layers,
    )
    x = _inverted_res_block(
        x,
        filters=32,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=4,
        created_layers=created_layers,
    )
    x = _inverted_res_block(
        x,
        filters=32,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=5,
        created_layers=created_layers,
    )
    x = _inverted_res_block(
        x,
        filters=64,
        alpha=alpha,
        stride=2,
        expansion=6,
        block_id=6,
        created_layers=created_layers,
    )
    x = _inverted_res_block(
        x,
        filters=64,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=7,
        created_layers=created_layers,
    )
    x = _inverted_res_block(
        x,
        filters=64,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=8,
        created_layers=created_layers,
    )
    x = _inverted_res_block(
        x,
        filters=64,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=9,
        created_layers=created_layers,
    )
    x = _inverted_res_block(
        x,
        filters=96,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=10,
        created_layers=created_layers,
    )
    x = _inverted_res_block(
        x,
        filters=96,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=11,
        created_layers=created_layers,
    )
    x = _inverted_res_block(
        x,
        filters=96,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=12,
        created_layers=created_layers,
    )
    x = _inverted_res_block(
        x,
        filters=160,
        alpha=alpha,
        stride=2,
        expansion=6,
        block_id=13,
        created_layers=created_layers,
    )
    x = _inverted_res_block(
        x,
        filters=160,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=14,
        created_layers=created_layers,
    )
    x = _inverted_res_block(
        x,
        filters=160,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=15,
        created_layers=created_layers,
    )
    x = _inverted_res_block(
        x,
        filters=320,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=16,
        created_layers=created_layers,
    )

    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    conv_1 = layers.Conv2D(
        last_block_filters,
        kernel_size=1,
        use_bias=False,
        name="Conv_1",
    )
    conv_1_bn = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-3,
        momentum=0.999,
        name="Conv_1_bn",
    )
    out_relu = layers.ReLU(6.0, name="out_relu")
    created_layers[conv_1.name] = conv_1
    created_layers[conv_1_bn.name] = conv_1_bn
    created_layers[out_relu.name] = out_relu
    x = conv_1(x)
    x = conv_1_bn(x)
    x = out_relu(x)
    return x


def _transfer_mobilenetv2_weights(
    created_layers: dict[str, keras.layers.Layer],
    *,
    image_shape: tuple[int, int, int],
    alpha: float,
) -> None:
    """Copy ImageNet MobileNetV2 weights into the flat backbone layers."""

    source_backbone = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=image_shape,
        alpha=alpha,
        pooling=None,
    )
    source_layers = {layer.name: layer for layer in source_backbone.layers}
    for layer_name, target_layer in created_layers.items():
        source_layer = source_layers.get(layer_name)
        if source_layer is None:
            continue
        source_weights = source_layer.get_weights()
        if source_weights:
            target_layer.set_weights(source_weights)


def _build_axis_simcc_head(
    features: tf.Tensor,
    *,
    axis: str,
    num_bins: int,
    head_channels: int,
    name_prefix: str,
) -> tf.Tensor:
    """Project a 14x14 shared feature map into one SimCC 1D distribution."""

    if num_bins % 14 != 0:
        raise ValueError("num_bins must be a multiple of 14 for the SimCC trunk.")

    expansion = num_bins // 14
    input_channels = features.shape[-1]
    if input_channels is None:
        raise ValueError("SimCC heads require a statically known channel count.")
    channels = int(input_channels)
    collapse_initializer = keras.initializers.Constant(1.0 / 14.0)

    if axis == "x":
        x = layers.Conv2D(
            channels,
            kernel_size=(14, 1),
            strides=(14, 1),
            padding="valid",
            use_bias=False,
            groups=channels,
            kernel_initializer=collapse_initializer,
            name=f"{name_prefix}_collapse_height",
        )(features)
        x = layers.UpSampling2D(
            size=(1, expansion),
            interpolation="bilinear",
            name=f"{name_prefix}_expand_width",
        )(x)
    elif axis == "y":
        x = layers.Conv2D(
            channels,
            kernel_size=(1, 14),
            strides=(1, 14),
            padding="valid",
            use_bias=False,
            groups=channels,
            kernel_initializer=collapse_initializer,
            name=f"{name_prefix}_collapse_width",
        )(features)
        x = layers.UpSampling2D(
            size=(expansion, 1),
            interpolation="bilinear",
            name=f"{name_prefix}_expand_height",
        )(x)
    else:
        raise ValueError(f"Unsupported axis '{axis}'.")

    # Keep the head compact: a little local context, then a single logit plane.
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


def build_mobilenetv2_obb_simcc_model(
    image_shape: tuple[int, int, int] = (224, 224, 3),
    *,
    alpha: float = 0.35,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    num_bins: int = _DEFAULT_NUM_BINS,
    spatial_channels: int = 64,
    head_units: int = 96,
    head_dropout: float = 0.15,
    include_temperature_head: bool = False,
) -> keras.Model:
    """Build a joint OBB + SimCC gauge localizer for 224x224 IMX frames.

    Outputs:
      - obb_center_xy: normalized [0, 1] center coordinates
      - obb_size_wh: normalized [0, 1] width/height
      - obb_angle_sincos: tanh-bounded [cos(angle), sin(angle)]
      - center_x_simcc, center_y_simcc, tip_x_simcc, tip_y_simcc
      - gauge_value: optional auxiliary Celsius head
    """

    inputs = keras.Input(shape=image_shape, name="input_image")

    backbone_layers: dict[str, keras.layers.Layer] = {}
    backbone_features = _build_mobilenetv2_feature_tensor(
        inputs,
        alpha=alpha,
        created_layers=backbone_layers,
    )

    if pretrained:
        _transfer_mobilenetv2_weights(
            backbone_layers,
            image_shape=image_shape,
            alpha=alpha,
        )

    for layer in backbone_layers.values():
        layer.trainable = backbone_trainable

    # A tiny spatial trunk keeps the geometry heads expressive without turning
    # the model into a large decoder.
    shared = layers.Conv2D(
        spatial_channels,
        1,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="shared_spatial_proj",
    )(backbone_features)
    shared = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        name="shared_spatial_up",
    )(shared)
    shared = layers.Conv2D(
        spatial_channels,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="shared_spatial_conv_1",
    )(shared)
    shared = layers.Conv2D(
        spatial_channels,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="shared_spatial_conv_2",
    )(shared)

    # The OBB branch uses global context to predict the dial center, size, and
    # orientation.  The separate heads make it easy to apply partial labels.
    obb_features = layers.GlobalAveragePooling2D(name="obb_gap")(shared)
    obb_features = layers.Dense(
        head_units,
        activation="relu",
        kernel_initializer="he_normal",
        name="obb_dense",
    )(obb_features)
    obb_features = layers.Dropout(head_dropout, name="obb_dropout")(obb_features)

    obb_center_xy = layers.Dense(
        2,
        activation="sigmoid",
        kernel_initializer="he_normal",
        name="obb_center_xy",
    )(obb_features)
    obb_size_wh = layers.Dense(
        2,
        activation="sigmoid",
        kernel_initializer="he_normal",
        name="obb_size_wh",
    )(obb_features)
    obb_angle_sincos = layers.Dense(
        2,
        activation="tanh",
        kernel_initializer="he_normal",
        name="obb_angle_sincos",
    )(obb_features)

    # SimCC heads keep the x and y coordinates explicit instead of collapsing
    # them into a single heatmap decoder.  That makes the geometry easier to
    # quantize and much cheaper to decode on the firmware side.
    center_x_simcc = _build_axis_simcc_head(
        shared,
        axis="x",
        num_bins=num_bins,
        head_channels=spatial_channels,
        name_prefix="center_x",
    )
    center_y_simcc = _build_axis_simcc_head(
        shared,
        axis="y",
        num_bins=num_bins,
        head_channels=spatial_channels,
        name_prefix="center_y",
    )
    tip_x_simcc = _build_axis_simcc_head(
        shared,
        axis="x",
        num_bins=num_bins,
        head_channels=spatial_channels,
        name_prefix="tip_x",
    )
    tip_y_simcc = _build_axis_simcc_head(
        shared,
        axis="y",
        num_bins=num_bins,
        head_channels=spatial_channels,
        name_prefix="tip_y",
    )

    outputs: dict[str, keras.KerasTensor] = {
        "obb_center_xy": obb_center_xy,
        "obb_size_wh": obb_size_wh,
        "obb_angle_sincos": obb_angle_sincos,
        "center_x_simcc": center_x_simcc,
        "center_y_simcc": center_y_simcc,
        "tip_x_simcc": tip_x_simcc,
        "tip_y_simcc": tip_y_simcc,
    }

    if include_temperature_head:
        # This is an auxiliary training head.  The board can still compute the
        # final Celsius reading with the existing CPU polar-voting model.
        temp_features = layers.Dense(
            head_units,
            activation="relu",
            kernel_initializer="he_normal",
            name="temperature_dense",
        )(obb_features)
        temp_features = layers.Dropout(
            head_dropout,
            name="temperature_dropout",
        )(temp_features)
        outputs["gauge_value"] = layers.Dense(
            1,
            activation="linear",
            kernel_initializer="he_normal",
            name="gauge_value",
        )(temp_features)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mobilenetv2_obb_simcc_gauge",
    )
    setattr(model, "_mobilenet_backbone_layers", backbone_layers)
    return model


def build_mobilenetv2_center_simcc_model(
    image_shape: tuple[int, int, int] = (224, 224, 3),
    *,
    alpha: float = 0.35,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    num_bins: int = _DEFAULT_NUM_BINS,
    spatial_channels: int = 64,
    head_units: int = 96,
    head_dropout: float = 0.15,
    include_temperature_head: bool = False,
) -> keras.Model:
    """Build a joint center-detector + SimCC gauge localizer for 224x224 IMX frames.

    Outputs:
      - center_xy: normalized [0, 1] gauge-face center coordinates
      - center_x_simcc, center_y_simcc, tip_x_simcc, tip_y_simcc
      - gauge_value: optional auxiliary Celsius head
    """

    inputs = keras.Input(shape=image_shape, name="input_image")

    backbone_layers: dict[str, keras.layers.Layer] = {}
    backbone_features = _build_mobilenetv2_feature_tensor(
        inputs,
        alpha=alpha,
        created_layers=backbone_layers,
    )

    if pretrained:
        _transfer_mobilenetv2_weights(
            backbone_layers,
            image_shape=image_shape,
            alpha=alpha,
        )

    for layer in backbone_layers.values():
        layer.trainable = backbone_trainable

    # Keep a compact spatial trunk so the center head and SimCC heads can share
    # low-resolution geometry without adding a heavy decoder.
    shared = layers.Conv2D(
        spatial_channels,
        1,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="shared_spatial_proj",
    )(backbone_features)
    shared = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        name="shared_spatial_up",
    )(shared)
    shared = layers.Conv2D(
        spatial_channels,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="shared_spatial_conv_1",
    )(shared)
    shared = layers.Conv2D(
        spatial_channels,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="shared_spatial_conv_2",
    )(shared)

    # The center detector predicts the gauge-face center directly.  That keeps
    # the first stage quantization-friendly while still giving the crop stage a
    # strong geometric prior.
    center_features = layers.GlobalAveragePooling2D(name="center_gap")(shared)
    center_features = layers.Dense(
        head_units,
        activation="relu",
        kernel_initializer="he_normal",
        name="center_dense",
    )(center_features)
    center_features = layers.Dropout(head_dropout, name="center_dropout")(center_features)
    center_xy = layers.Dense(
        2,
        activation="sigmoid",
        kernel_initializer="he_normal",
        name="center_xy",
    )(center_features)

    # SimCC heads keep the x and y coordinates explicit instead of collapsing
    # them into a single heatmap decoder.  That makes the geometry easier to
    # quantize and much cheaper to decode on the firmware side.
    center_x_simcc = _build_axis_simcc_head(
        shared,
        axis="x",
        num_bins=num_bins,
        head_channels=spatial_channels,
        name_prefix="center_x",
    )
    center_y_simcc = _build_axis_simcc_head(
        shared,
        axis="y",
        num_bins=num_bins,
        head_channels=spatial_channels,
        name_prefix="center_y",
    )
    tip_x_simcc = _build_axis_simcc_head(
        shared,
        axis="x",
        num_bins=num_bins,
        head_channels=spatial_channels,
        name_prefix="tip_x",
    )
    tip_y_simcc = _build_axis_simcc_head(
        shared,
        axis="y",
        num_bins=num_bins,
        head_channels=spatial_channels,
        name_prefix="tip_y",
    )

    outputs: dict[str, keras.KerasTensor] = {
        "center_xy": center_xy,
        "center_x_simcc": center_x_simcc,
        "center_y_simcc": center_y_simcc,
        "tip_x_simcc": tip_x_simcc,
        "tip_y_simcc": tip_y_simcc,
    }

    if include_temperature_head:
        # Keep the auxiliary value head off the quantization-critical path.
        temp_features = layers.Dense(
            head_units,
            activation="relu",
            kernel_initializer="he_normal",
            name="temperature_dense",
        )(center_features)
        temp_features = layers.Dropout(
            head_dropout,
            name="temperature_dropout",
        )(temp_features)
        outputs["gauge_value"] = layers.Dense(
            1,
            activation="linear",
            kernel_initializer="he_normal",
            name="gauge_value",
        )(temp_features)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mobilenetv2_center_simcc_gauge",
    )
    setattr(model, "_mobilenet_backbone_layers", backbone_layers)
    return model


def build_mobilenetv2_obb_box_model(
    image_shape: tuple[int, int, int] = (224, 224, 3),
    *,
    alpha: float = 0.35,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    spatial_channels: int = 64,
    head_units: int = 96,
    head_dropout: float = 0.15,
) -> keras.Model:
    """Build a full-frame OBB box model for deployable crop generation.

    Outputs:
      - conf: a lightweight confidence score in [0, 1]
      - box: normalized box parameters ``[cx, cy, w, h]`` in [0, 1]

    The box output is intentionally axis-aligned. That keeps the first-stage
    crop decoder simple and makes the model easier to quantize and deploy on
    embedded hardware while still letting the camera and gauge move around in
    the field.
    """

    inputs = keras.Input(shape=image_shape, name="input_image")

    backbone_layers: dict[str, keras.layers.Layer] = {}
    backbone_features = _build_mobilenetv2_feature_tensor(
        inputs,
        alpha=alpha,
        created_layers=backbone_layers,
    )

    if pretrained:
        _transfer_mobilenetv2_weights(
            backbone_layers,
            image_shape=image_shape,
            alpha=alpha,
        )

    for layer in backbone_layers.values():
        layer.trainable = backbone_trainable

    shared = layers.Conv2D(
        spatial_channels,
        1,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="shared_spatial_proj",
    )(backbone_features)
    shared = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        name="shared_spatial_up",
    )(shared)
    shared = layers.Conv2D(
        spatial_channels,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="shared_spatial_conv_1",
    )(shared)
    shared = layers.Conv2D(
        spatial_channels,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="shared_spatial_conv_2",
    )(shared)

    obb_features = layers.GlobalAveragePooling2D(name="obb_gap")(shared)
    obb_features = layers.Dense(
        head_units,
        activation="relu",
        kernel_initializer="he_normal",
        name="obb_dense",
    )(obb_features)
    obb_features = layers.Dropout(head_dropout, name="obb_dropout")(obb_features)

    conf = layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer="he_normal",
        name="conf",
    )(obb_features)
    box = layers.Dense(
        4,
        activation="sigmoid",
        kernel_initializer="he_normal",
        name="box",
    )(obb_features)

    model = keras.Model(
        inputs=inputs,
        outputs={"conf": conf, "box": box},
        name="mobilenetv2_obb_box",
    )
    setattr(model, "_mobilenet_backbone_layers", backbone_layers)
    return model
