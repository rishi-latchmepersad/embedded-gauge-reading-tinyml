# Archived dead model builders from models.py.
# Kept for reference; not imported by the active package.
from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
import keras


def build_mobilenetv2_angle_vote_model(
    image_height: int,
    image_width: int,
    *,
    num_angle_bins: int = 36,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 0.35,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a gauge-agnostic angle-vote model with MobileNetV2 backbone.

    Predicts a distribution over 36 angle bins (10° resolution) covering the
    full 360° circle. The angle is converted to temperature at inference via
    GaugeSpec, making this model reusable across gauges.
    """
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    x = keras.layers.GlobalAveragePooling2D(name="angle_vote_gap")(x)
    x = keras.layers.Dropout(head_dropout, name="angle_vote_dropout_1")(x)
    x = keras.layers.Dense(
        head_units,
        activation="swish",
        name="angle_vote_dense",
    )(x)
    x = keras.layers.Dropout(head_dropout, name="angle_vote_dropout_2")(x)
    angle_logits = keras.layers.Dense(
        num_angle_bins,
        activation="linear",
        name="angle_logits",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=angle_logits,
        name=_mobilenetv2_model_name(
            regression_kind="angle_vote",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    setattr(model, "_num_angle_bins", num_angle_bins)
    return model


def build_mobilenetv2_bluraware_reader_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
) -> keras.Model:
    """Build a narrower MobileNetV2 regressor sized for STM32N6 deployment."""
    return build_mobilenetv2_regression_model(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=0.35,
        head_units=64,
        head_dropout=0.15,
    )


def build_mobilenetv2_bluraware_reader_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
    linear_output: bool = False,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> keras.Model:
    """Build a blur-aware reader that fuses raw and unsharp-masked views.

    The recent gauge-reading papers consistently favor geometry-aware readers,
    but our board path already has a strong OBB localizer. This reader keeps the
    crop fixed and spends its capacity on low-contrast detail recovery by fusing
    the raw crop with a lightweight unsharp-mask branch before the scalar head.
    """
    inputs, _, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    raw_branch = keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1.0,
        name="bluraware_reader_raw_preprocess",
    )(inputs)
    enhanced_branch = _build_unsharp_mask_branch(
        raw_branch,
        name_prefix="bluraware_reader",
    )

    raw_maps = base_model(raw_branch, training=backbone_trainable)
    enhanced_maps = base_model(enhanced_branch, training=backbone_trainable)
    x = keras.layers.Average(name="bluraware_reader_feature_average")(
        [raw_maps, enhanced_maps]
    )

    x = keras.layers.GlobalAveragePooling2D(name="bluraware_reader_pooled_features")(x)
    x = keras.layers.Dropout(head_dropout, name="bluraware_reader_dropout_1")(x)
    x = keras.layers.Dense(
        head_units,
        activation="swish",
        name="bluraware_reader_dense",
    )(x)
    x = keras.layers.Dropout(head_dropout, name="bluraware_reader_dropout_2")(x)

    span = value_max - value_min
    if linear_output:
        gauge_value = keras.layers.Dense(
            1,
            activation="linear",
            name="gauge_value_linear",
        )(x)
    else:
        gauge_value = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="gauge_value_sigmoid",
        )(x)

    gauge_value = keras.layers.Rescaling(
        scale=span,
        offset=value_min,
        name="gauge_value",
    )(gauge_value)

    model = keras.Model(
        inputs=inputs,
        outputs=gauge_value,
        name=_mobilenetv2_model_name(
            regression_kind="bluraware_reader",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model

