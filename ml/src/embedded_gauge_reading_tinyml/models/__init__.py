"""Model builders for gauge-reading networks.

Public builders are exported from this package.  Helper utilities live in
``_layers.py`` (custom Keras layers), ``_backbones.py`` (backbone factories),
and ``_heads.py`` (head builders).
"""

from __future__ import annotations

import math

import keras
import numpy as np
import tensorflow as tf

from ._layers import (
    CBAMBlock,
    CenterCropResize,
    CoordinateAttention,
    CornerKeypointsToBox,
    GaugeValueFromKeypoints,
    GaugeValueFromNeedleDirection,
    GaugeValueFromRelationKeypoints,
    GaugeValueFromSweepDistribution,
    OrderedCornerBox,
    PolarEvidenceLayer,
    SpatialSoftArgmax2D,
)
from ._backbones import (
    _build_compact_geometry_backbone,
    _build_feature_backbone,
    _build_mobilenetv2_backbone,
    _build_mobilenetv2_dual_resolution_backbone,
    _build_mobilenetv2_polar_backbone,
    _build_mobilenetv2_polar_dualview_backbone,
    _conv_norm_swish,
    _mobilenetv2_model_name,
    _norm,
    _residual_separable_block,
)
from ._heads import (
    _build_interval_expectation_head,
    _build_keypoint_heatmap_head,
    _build_mobilenetv2_multi_scale_backbone,
    _build_ordinal_expectation_head,
    _build_pointer_mask_head,
    _build_sweep_fraction_head,
    _build_unsharp_mask_branch,
    _cbam_refine,
)

__all__ = [
    # Layers
    "CBAMBlock",
    "CenterCropResize",
    "CoordinateAttention",
    "CornerKeypointsToBox",
    "GaugeValueFromKeypoints",
    "GaugeValueFromNeedleDirection",
    "GaugeValueFromRelationKeypoints",
    "GaugeValueFromSweepDistribution",
    "OrderedCornerBox",
    "PolarEvidenceLayer",
    "SpatialSoftArgmax2D",
    # Backbones
    "_build_feature_backbone",
    "_build_mobilenetv2_backbone",
    # Model builders
    "build_regression_model",
    "build_needle_direction_model",
    "build_compact_interval_model",
    "build_compact_geometry_model",
    "build_compact_source_crop_box_model",
    "build_mobilenetv2_regression_model",
    "build_compact_obb_model",
    "build_mobilenetv2_angle_sincos_model",
    "build_mobilenetv2_direction_model",
    "build_mobilenetv2_direction_geometry_model",
    "build_mobilenetv2_interval_model",
    "build_mobilenetv2_ordinal_model",
    "build_mobilenetv2_fraction_model",
    "build_mobilenetv2_dual_resolution_regression_model",
    "build_mobilenetv2_dual_resolution_interval_model",
    "build_mobilenetv2_polar_dualview_regression_model",
    "build_mobilenetv2_polar_regression_model",
    "build_mobilenetv2_polar_sweep_distribution_model",
    "build_mobilenetv2_sweep_distribution_model",
    "build_mobilenetv2_keypoint_model",
    "build_mobilenetv2_detector_model",
    "build_mobilenetv2_geometry_model",
    "build_mobilenetv2_polar_evidence_model",
    "build_mobilenetv2_center_selector",
    "build_mobilenetv2_geometry_reader",
    "build_mobilenetv2_geometry_uncertainty_model",
    "build_mobilenetv2_obb_geometry_model",
    "build_mobilenetv2_obb_mask_geometry_model",
    "build_mobilenetv2_obb_relation_geometry_model",
    "build_mobilenetv2_bluraware_obb_geometry_model",
    "build_mobilenetv2_bluraware_obb_relation_geometry_model",
    "build_mobilenetv2_bluraware_obb_sequence_geometry_model",
    "build_mobilenetv2_bluraware_reader_model",
    "build_mobilenetv2_rectifier_model",
    "build_mobilenetv2_source_crop_box_model",
    "build_mobilenetv2_source_crop_box_v2_model",
    "build_mobilenetv2_source_crop_corner_model",
    "build_mobilenetv2_obb_model",
    "build_center_detection_model",
    "build_mobilenetv2_enhanced_regression_model",
    "build_mobilenetv2_enhanced_ensemble_model",
    "build_mobilenetv2_sota_multiscale_model",
    "build_mobilenetv2_sota_ensemble_model",
    "build_mobilenetv2_uncertainty_model",
    "build_mobilenetv2_heatmap_center_model",
    "build_mobilenetv2_obb_center_model",
    "build_mobilenetv3_obb_center_model",
]


def build_regression_model(image_height: int, image_width: int) -> keras.Model:
    """Build a compact residual CNN regressor for scalar gauge value output."""
    inputs, x = _build_feature_backbone(image_height, image_width)

    x = keras.layers.Dense(192, activation="swish")(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(1, name="gauge_value")(x)

    return keras.Model(inputs=inputs, outputs=output, name="gauge_value_regressor")


def build_needle_direction_model(image_height: int, image_width: int) -> keras.Model:
    """Build a compact residual CNN that predicts unit needle direction (dx, dy)."""
    inputs, x = _build_feature_backbone(image_height, image_width)

    x = keras.layers.Dense(192, activation="swish")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(2, name="needle_xy_raw")(x)
    output = keras.layers.UnitNormalization(axis=-1, name="needle_xy")(x)

    return keras.Model(inputs=inputs, outputs=output, name="needle_direction_regressor")


def build_compact_interval_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    bin_width: float = 5.0,
) -> keras.Model:
    """Build a compact CNN that predicts coarse bins plus a local residual."""
    inputs, x = _build_feature_backbone(image_height, image_width)

    x = keras.layers.Dense(192, activation="swish")(x)
    x = keras.layers.Dropout(0.2)(x)

    span = value_max - value_min
    num_bins = int(math.ceil(span / bin_width))
    if num_bins < 2:
        raise ValueError("Compact interval model needs at least two bins.")

    interval_logits = keras.layers.Dense(
        num_bins,
        name="interval_logits",
    )(x)
    gauge_value, interval_probs = _build_interval_expectation_head(
        interval_logits,
        value_min=value_min,
        value_max=value_max,
        bin_width=bin_width,
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "interval_logits": interval_logits,
        },
        name="compact_interval_gauge_regressor",
    )
    setattr(model, "_compact_interval_probs", interval_probs)
    return model


def build_compact_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
) -> keras.Model:
    """Build a compact CNN with explicit keypoint heatmaps and geometry value."""
    inputs, spatial_features, _pooled_features = _build_compact_geometry_backbone(
        image_height,
        image_width,
    )
    heatmaps = _build_keypoint_heatmap_head(
        spatial_features,
        heatmap_size=heatmap_size,
        num_keypoints=2,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=2,
        name="keypoint_coords",
    )(heatmaps)
    gauge_value = GaugeValueFromKeypoints(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
        },
        name="compact_geometry_gauge_regressor",
    )
    return model


def build_compact_source_crop_box_model(
    image_height: int,
    image_width: int,
    *,
    head_units: int = 96,
    head_dropout: float = 0.15,
) -> keras.Model:
    """Build a compact CNN that regresses a normalized source-space crop box."""
    inputs, x = _build_feature_backbone(image_height, image_width)

    x = keras.layers.Dense(
        head_units,
        activation="swish",
        name="source_crop_box_dense",
    )(x)
    x = keras.layers.Dropout(head_dropout, name="source_crop_box_dropout")(x)

    raw_box = keras.layers.Dense(
        4,
        activation="sigmoid",
        name="source_crop_box_raw",
    )(x)
    source_crop_box = OrderedCornerBox(name="source_crop_box")(raw_box)

    return keras.Model(
        inputs=inputs,
        outputs={"source_crop_box": source_crop_box},
        name="compact_source_crop_box_regressor",
    )


def build_mobilenetv2_regression_model(
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
    """Build a transfer-learning regressor on top of MobileNetV2 features.

    Args:
        linear_output: If True, use a linear output head on the normalized target.
        value_min: Minimum gauge value for output scaling.
        value_max: Maximum gauge value for output scaling.
    """
    if linear_output:
        pass  # linear output enabled for range-compression-free regression
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    span = value_max - value_min
    if linear_output:
        # Predict a normalized fraction directly and then map it back to Celsius.
        x = keras.layers.Dense(1, activation="linear", name="gauge_value_linear")(x)
        output = keras.layers.Rescaling(
            scale=span,
            offset=value_min,
            name="gauge_value",
        )(x)
    else:
        # Sigmoid output bounded to [0,1], then rescaled to value range.
        x = keras.layers.Dense(1, activation="sigmoid", name="gauge_value_sigmoid")(x)
        # Rescale sigmoid [0,1] output to [value_min, value_max].
        output = keras.layers.Rescaling(
            scale=span,
            offset=value_min,
            name="gauge_value",
        )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name=_mobilenetv2_model_name(
            regression_kind="gauge",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    # Store the backbone so training can run staged freeze/unfreeze schedules.
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_compact_obb_model(
    image_height: int = 224,
    image_width: int = 224,
    *,
    head_units: int = 96,
    head_dropout: float = 0.15,
) -> keras.Model:
    """Build a compact CNN that predicts oriented ellipse parameters (6 floats).

    Uses the custom separable-conv backbone (~430K params) instead of
    MobileNetV2 (~2.3M).  The 5x parameter reduction makes it train faster,
    harder to overfit a small dataset, and easier to deploy on embedded targets.

    Output: [cx, cy, w, h, cos(2θ), sin(2θ)] in [0,1]/[-1,1] normalized.
    """
    inputs, x = _build_feature_backbone(image_height, image_width)

    x = keras.layers.LayerNormalization(name="obb_pooled_norm")(x)
    x = keras.layers.Dense(head_units, activation="swish", name="obb_dense")(x)
    x = keras.layers.Dropout(head_dropout, name="obb_dropout")(x)

    center_xy = keras.layers.Dense(
        2, activation="sigmoid", name="obb_center_xy",
    )(x)
    size_wh = keras.layers.Dense(
        2, activation="sigmoid", name="obb_size_wh",
    )(x)
    angle_raw = keras.layers.Dense(
        2, name="obb_angle_raw",
    )(x)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1, name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy, size_wh, angle_sincos],
    )

    model = keras.Model(
        inputs=inputs,
        outputs={"obb_params": obb_params},
        name=f"compact_obb_localizer_h{head_units}",
    )
    return model


def build_mobilenetv2_angle_sincos_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 0.35,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a gauge-agnostic angle sin/cos regression model with MobileNetV2 backbone.

    Predicts (sin(angle), cos(angle)) of the needle direction. The angle is
    converted to temperature at inference via GaugeSpec. This is a 2-output
    regression task (much simpler than 36-bin classification) and works well
    with limited training data.
    """
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    x = keras.layers.GlobalAveragePooling2D(name="sincos_gap")(x)
    x = keras.layers.Dropout(head_dropout, name="sincos_dropout_1")(x)
    x = keras.layers.Dense(
        head_units,
        activation="swish",
        name="sincos_dense",
    )(x)
    x = keras.layers.Dropout(head_dropout, name="sincos_dropout_2")(x)
    outputs = keras.layers.Dense(
        2,
        activation="linear",
        name="angle_sincos",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=_mobilenetv2_model_name(
            regression_kind="angle_sincos",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_direction_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a transfer-learning model that predicts unit needle direction."""
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(2, name="needle_xy_raw")(x)
    output = keras.layers.UnitNormalization(axis=-1, name="needle_xy")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name=_mobilenetv2_model_name(
            regression_kind="needle_direction",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_direction_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a MobileNetV2 model that predicts needle direction then value.

    The backbone learns a compact needle direction embedding, the unit-vector
    head keeps the geometry explicit, and the final scalar value is computed
    from that direction using the calibrated sweep mapping.
    """
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(2, name="needle_xy_raw")(x)
    needle_xy = keras.layers.UnitNormalization(axis=-1, name="needle_xy")(x)
    gauge_value = GaugeValueFromNeedleDirection(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(needle_xy)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "needle_xy": needle_xy,
        },
        name=_mobilenetv2_model_name(
            regression_kind="needle_geometry",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_interval_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    bin_width: float = 5.0,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    span = value_max - value_min
    num_bins = int(math.ceil(span / bin_width))
    if num_bins < 2:
        raise ValueError("Interval model needs at least two bins.")

    interval_logits = keras.layers.Dense(
        num_bins,
        name="interval_logits",
    )(x)
    gauge_value, interval_probs = _build_interval_expectation_head(
        interval_logits,
        value_min=value_min,
        value_max=value_max,
        bin_width=bin_width,
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "interval_logits": interval_logits,
        },
        name=_mobilenetv2_model_name(
            regression_kind="interval",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    setattr(model, "_mobilenet_interval_probs", interval_probs)
    return model


def build_mobilenetv2_ordinal_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    threshold_step: float = 2.5,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a transfer-learning regressor with an ordinal threshold head."""
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    span = value_max - value_min
    num_thresholds = int(math.ceil(span / threshold_step))
    if num_thresholds < 2:
        raise ValueError("Need at least two thresholds for ordinal regression.")

    ordinal_logits = keras.layers.Dense(
        num_thresholds,
        name="ordinal_logits",
    )(x)
    gauge_value, ordinal_probs = _build_ordinal_expectation_head(
        ordinal_logits,
        value_min=value_min,
        value_max=value_max,
        threshold_step=threshold_step,
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "ordinal_logits": ordinal_logits,
        },
        name=_mobilenetv2_model_name(
            regression_kind="ordinal",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    setattr(model, "_mobilenet_ordinal_probs", ordinal_probs)
    return model


def build_mobilenetv2_fraction_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a transfer-learning regressor with a normalized sweep-fraction head."""
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    sweep_fraction_logits = keras.layers.Dense(
        1,
        name="sweep_fraction_logits",
    )(x)
    gauge_value, sweep_fraction = _build_sweep_fraction_head(
        sweep_fraction_logits,
        value_min=value_min,
        value_max=value_max,
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "sweep_fraction": sweep_fraction,
        },
        name=_mobilenetv2_model_name(
            regression_kind="fraction",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_dual_resolution_regression_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 0.35,
    head_units: int = 96,
    head_dropout: float = 0.2,
    crop_ratio: float = 0.78,
    linear_output: bool = True,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> keras.Model:
    """Build a dual-resolution MobileNetV2 regressor for gauge value."""
    inputs, features, base_model = _build_mobilenetv2_dual_resolution_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
        crop_ratio=crop_ratio,
    )

    x = keras.layers.Dense(head_units * 2, activation="swish")(features)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    span = value_max - value_min
    if linear_output:
        # Predict an unrestricted normalized scalar and map it back to Celsius.
        x = keras.layers.Dense(
            1,
            activation="linear",
            name="gauge_value_linear",
        )(x)
    else:
        # Keep a bounded head when the caller wants the older sigmoid behavior.
        x = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="gauge_value_sigmoid",
        )(x)

    output = keras.layers.Rescaling(
        scale=span,
        offset=value_min,
        name="gauge_value",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name=_mobilenetv2_model_name(
            regression_kind="dualres",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_dual_resolution_interval_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    bin_width: float = 5.0,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 0.35,
    head_units: int = 96,
    head_dropout: float = 0.2,
    crop_ratio: float = 0.78,
) -> keras.Model:
    """Build a dual-resolution MobileNetV2 model with interval supervision."""
    inputs, features, base_model = _build_mobilenetv2_dual_resolution_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
        crop_ratio=crop_ratio,
    )

    x = keras.layers.Dense(head_units * 2, activation="swish")(features)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    span = value_max - value_min
    num_bins = int(math.ceil(span / bin_width))
    if num_bins < 2:
        raise ValueError("Interval model needs at least two bins.")

    interval_logits = keras.layers.Dense(
        num_bins,
        name="interval_logits",
    )(x)
    gauge_value, interval_probs = _build_interval_expectation_head(
        interval_logits,
        value_min=value_min,
        value_max=value_max,
        bin_width=bin_width,
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "interval_logits": interval_logits,
        },
        name=_mobilenetv2_model_name(
            regression_kind="dualres_interval",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    setattr(model, "_mobilenet_interval_probs", interval_probs)
    return model


def build_mobilenetv2_polar_dualview_regression_model(
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
    """Build a polar-dualview MobileNetV2 regressor for gauge value.

    The full-frame branch keeps the broader scene context while the polar branch
    turns the circular dial into a flattened angle/radius image. This is a
    better fit for the gauge geometry than another center-crop-only variant.
    """
    inputs, features, base_model = _build_mobilenetv2_polar_dualview_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    x = keras.layers.Dense(head_units, activation="swish", name="polar_dense")(features)
    x = keras.layers.Dropout(head_dropout, name="polar_dropout")(x)

    span = value_max - value_min
    if linear_output:
        # Predict an unrestricted normalized scalar and map it back to Celsius.
        x = keras.layers.Dense(
            1,
            activation="linear",
            name="gauge_value_linear",
        )(x)
    else:
        # Keep a bounded head when the caller wants the older sigmoid behavior.
        x = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="gauge_value_sigmoid",
        )(x)

    output = keras.layers.Rescaling(
        scale=span,
        offset=value_min,
        name="gauge_value",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name=_mobilenetv2_model_name(
            regression_kind="polar_dualview",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_polar_regression_model(
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
    """Build a single-input polar-unwrapped MobileNetV2 regressor.

    Compared with the dual-view branch, this version keeps one input tensor and
    one shared MobileNetV2 trunk. It is less expressive, but it is much easier
    to train, debug, and export on the constrained WSL/GPU setup we have here.
    """
    inputs, features, base_model = _build_mobilenetv2_polar_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    x = keras.layers.Dense(head_units, activation="swish", name="polar_dense")(features)
    x = keras.layers.Dropout(head_dropout, name="polar_dropout")(x)

    span = value_max - value_min
    if linear_output:
        # Predict a normalized scalar directly and then map it back to Celsius.
        x = keras.layers.Dense(
            1,
            activation="linear",
            name="gauge_value_linear",
        )(x)
    else:
        # Keep the older bounded form if the caller wants a sigmoid output.
        x = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="gauge_value_sigmoid",
        )(x)

    output = keras.layers.Rescaling(
        scale=span,
        offset=value_min,
        name="gauge_value",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name=_mobilenetv2_model_name(
            regression_kind="polar",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_polar_sweep_distribution_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    num_bins: int = 81,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a polar gauge reader with a distributional sweep head.

    The polar warp flattens the circular dial into an angle-versus-radius view,
    while the sweep distribution head preserves ordinal structure better than a
    plain scalar regressor. This keeps the model compact but more geometry-aware
    than the earlier polar-only baseline.
    """
    inputs, features, base_model = _build_mobilenetv2_polar_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    # The polar backbone already returns a pooled 2D feature vector, so the
    # distribution head can work directly from that compact representation.
    sweep_features = keras.layers.Dropout(head_dropout)(features)
    sweep_features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="polar_sweep_dense",
    )(sweep_features)
    sweep_features = keras.layers.Dropout(head_dropout)(sweep_features)
    sweep_distribution_logits = keras.layers.Dense(
        num_bins,
        name="sweep_distribution_logits",
    )(sweep_features)
    gauge_value = GaugeValueFromSweepDistribution(
        value_min=value_min,
        value_max=value_max,
        num_bins=num_bins,
        name="gauge_value",
    )(sweep_distribution_logits)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "sweep_distribution_logits": sweep_distribution_logits,
        },
        name=_mobilenetv2_model_name(
            regression_kind="polar_sweep_distribution",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_sweep_distribution_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    num_bins: int = 81,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a gauge reader that predicts a smooth sweep distribution.

    The network learns a value distribution over the known Celsius range and a
    deterministic expectation layer turns that distribution back into the final
    scalar temperature. This is a more geometry-aware alternative to direct
    scalar regression.
    """
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    features = keras.layers.GlobalAveragePooling2D(name="sweep_distribution_gap")(x)
    features = keras.layers.Dropout(head_dropout)(features)
    features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="sweep_distribution_dense",
    )(features)
    features = keras.layers.Dropout(head_dropout)(features)
    sweep_distribution_logits = keras.layers.Dense(
        num_bins,
        name="sweep_distribution_logits",
    )(features)
    gauge_value = GaugeValueFromSweepDistribution(
        value_min=value_min,
        value_max=value_max,
        num_bins=num_bins,
        name="gauge_value",
    )(sweep_distribution_logits)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "sweep_distribution_logits": sweep_distribution_logits,
        },
        name=_mobilenetv2_model_name(
            regression_kind="sweep_distribution",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_keypoint_model(
    image_height: int,
    image_width: int,
    *,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a transfer-learning model with a scalar head plus keypoint heatmaps."""
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    scalar_features = keras.layers.GlobalAveragePooling2D()(x)
    scalar_features = keras.layers.Dropout(head_dropout)(scalar_features)
    scalar_features = keras.layers.Dense(head_units, activation="swish")(
        scalar_features
    )
    scalar_features = keras.layers.Dropout(head_dropout)(scalar_features)
    gauge_value = keras.layers.Dense(1, name="gauge_value")(scalar_features)

    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=2,
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
        },
        name=_mobilenetv2_model_name(
            regression_kind="keypoint",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_detector_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a detector-first model that turns keypoints into gauge value."""
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=2,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=2,
        name="keypoint_coords",
    )(heatmaps)
    gauge_value = GaugeValueFromKeypoints(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
        },
        name=_mobilenetv2_model_name(
            regression_kind="detector",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a geometry-first model with explicit keypoint and value outputs."""
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=2,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=2,
        name="keypoint_coords",
    )(heatmaps)
    gauge_value = GaugeValueFromKeypoints(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
        },
        name=_mobilenetv2_model_name(
            regression_kind="geometry",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_polar_evidence_model(
    image_height: int,
    image_width: int,
    *,
    num_angles: int = 180,
    decoder_channels: tuple[int, ...] = (128, 64, 32),
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 0.35,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a classical-inspired polar evidence CNN with three learned heads.

    Architecture (mimics the firmware classical pipeline):
      1. Center head: predicts the dial center (cx, cy) in [0, 1] normalized coords.
      2. Polar evidence head: samples backbone features radially around the
         predicted center and produces per-angle evidence logits.
      3. Confidence head: scalar [0, 1] confidence in the predicted geometry.

    All gauge decode (angle → temperature) happens outside the model using
    GaugeSpec, so this model is gauge-agnostic.

    Key design choices:
    - Local, center-conditioned polar sampling (not old full-frame polar warp).
    - Differentiable bilinear sampling for end-to-end center + evidence training.
    - No Celsius or angle computation inside the graph, keeping it TFLite-friendly.
    - Backbone frozen by default.

    Args:
        image_height: Input image height.
        image_width: Input image width.
        num_angles: Number of angular bins for the polar evidence head.
        decoder_channels: Progressive decoder channel sizes.
        heatmap_size: Intermediate feature resolution for polar sampling.
        pretrained: Load ImageNet backbone weights.
        backbone_trainable: Whether the backbone is trainable.
        alpha: MobileNetV2 width multiplier.
        head_dropout: Dropout rate on dense heads.

    Returns:
        keras.Model with outputs:
            center: (B, 2) — (cx, cy) sigmoid-normalized.
            polar_evidence: (B, num_angles) — per-angle evidence logits.
            confidence: (B, 1) — scalar sigmoid confidence.
    """
    inputs = keras.Input(
        shape=(image_height, image_width, 3), name="image"
    )
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(inputs)

    backbone = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(image_height, image_width, 3),
        alpha=alpha,
    )
    backbone.trainable = backbone_trainable

    # Shared backbone features at 7x7
    backbone_features = backbone(x, training=backbone_trainable)

    # ---- Head 1: Center prediction ----
    pooled = keras.layers.GlobalAveragePooling2D(name="center_gap")(
        backbone_features
    )
    center_feat = keras.layers.Dropout(head_dropout)(pooled)
    center_feat = keras.layers.Dense(64, activation="swish", name="center_dense")(
        center_feat
    )
    center_feat = keras.layers.Dropout(head_dropout)(center_feat)
    center = keras.layers.Dense(2, activation="sigmoid", name="center")(center_feat)

    # ---- Head 2: Polar evidence ----
    # Build a small progressive decoder to get higher-res features for sampling
    x_dec = backbone_features
    for stage, ch in enumerate(decoder_channels, start=1):
        x_dec = keras.layers.Conv2D(
            ch, 3, padding="same", use_bias=False, name=f"polar_decoder_conv_{stage}"
        )(x_dec)
        x_dec = keras.layers.BatchNormalization(
            momentum=0.9, epsilon=1e-3, name=f"polar_decoder_norm_{stage}"
        )(x_dec)
        x_dec = keras.layers.Activation("swish")(x_dec)
        x_dec = keras.layers.UpSampling2D(
            size=2, interpolation="bilinear", name=f"polar_decoder_up_{stage}"
        )(x_dec)

    # Ensure we reach the target resolution
    if heatmap_size != x_dec.shape[1]:
        x_dec = keras.layers.Resizing(
            heatmap_size,
            heatmap_size,
            interpolation="bilinear",
            name="polar_decoder_resize",
        )(x_dec)

    # Polar evidence sampling
    polar_evidence_logits = PolarEvidenceLayer(
        num_angles=num_angles,
        sigma_angle=0.04,
        radius_mean=0.35,
        radius_sigma=0.25,
        name="polar_evidence",
    )(x_dec, center)

    # ---- Head 3: Confidence ----
    conf_feat = keras.layers.Dropout(head_dropout)(pooled)
    conf_feat = keras.layers.Dense(16, activation="swish", name="confidence_dense")(
        conf_feat
    )
    conf_feat = keras.layers.Dropout(head_dropout)(conf_feat)
    confidence = keras.layers.Dense(
        1, activation="sigmoid", name="confidence"
    )(conf_feat)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "center": center,
            "polar_evidence": polar_evidence_logits,
            "confidence": confidence,
        },
        name=f"mobilenetv2_polar_evidence_a{alpha}",
    )
    setattr(model, "_mobilenet_backbone", backbone)
    return model


def build_mobilenetv2_center_selector(
    image_height: int,
    image_width: int,
    *,
    num_hypotheses: int = 5,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 0.35,
    head_units: int = 64,
    head_dropout: float = 0.3,
) -> keras.Model:
    """Build a small MobileNetV2-tiny hybrid localizer.

    Learns two things:
      1. center_logits (num_hypotheses) — which of the classical firmware
         center hypotheses is best for this image.
      2. center_offset (2, tanh) — sub-pixel refinement from the argmax
         hypothesis to the true center.

    The classical hypotheses (computed outside the model) are:
      - bright centroid, crop center, board prior, rim center, image center.

    Decode: argmax(center_logits) → pick hypothesis → add center_offset
    → feed refined center into classical polar spoke vote → GaugeSpec.

    Args:
        image_height: Input image height.
        image_width: Input image width.
        num_hypotheses: Number of classical center hypotheses (default 4).
        pretrained: Load ImageNet backbone weights.
        backbone_trainable: Whether the backbone is trainable.
        alpha: MobileNetV2 width multiplier.
        head_units: Dense layer units.
        head_dropout: Dropout rate.

    Returns:
        keras.Model with outputs:
          center_logits: (B, num_hypotheses) — unnormalized hypothesis scores.
          center_offset: (B, 2, tanh) — residual from chosen hypothesis.
    """
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    x = keras.layers.GlobalAveragePooling2D(name="selector_gap")(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish", name="selector_dense")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    center_logits = keras.layers.Dense(
        num_hypotheses, name="center_logits"
    )(x)
    center_offset = keras.layers.Dense(
        2, activation="tanh", name="center_offset"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "center_logits": center_logits,
            "center_offset": center_offset,
        },
        name=f"mobilenetv2_center_selector_a{alpha}",
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_geometry_reader(
    image_height: int,
    image_width: int,
    *,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 0.35,
) -> keras.Model:
    """Build a gauge-agnostic geometry reader producing raw keypoint heatmaps.

    MobileNetV2 alpha=0.35 → small conv decoder → 2 heatmap outputs
    (center, tip). No gauge-specific parameters are embedded in the model;
    all gauge decode (angle → temperature) happens outside using GaugeSpec
    from embedded_gauge_reading_tinyml.gauge.processing.

    The two heatmap outputs are:
    - center_heatmap (heatmap_size x heatmap_size x 1, sigmoid): Gaussian peak at dial center
    - tip_heatmap (heatmap_size x heatmap_size x 1, sigmoid): Gaussian peak at needle tip

    Key design rationale:
    - Gauge-agnostic: the same model can decode any gauge by swapping the GaugeSpec
    - No Celsius or angle computation inside the model graph, keeping it TFLite-friendly
    - Backbone frozen by default to leverage pretrained ImageNet features
      without catastrophic overfitting on the small gauge dataset

    Args:
        image_height: Input image height in pixels.
        image_width: Input image width in pixels.
        heatmap_size: Output heatmap spatial resolution (default 28).
        pretrained: Whether to load ImageNet backbone weights.
        backbone_trainable: Whether the MobileNetV2 backbone is trainable.
        alpha: MobileNetV2 width multiplier (0.35 for tiny NPU-friendly model).

    Returns:
        keras.Model with outputs {'center_heatmap': ..., 'tip_heatmap': ...}.
    """
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=2,
    )
    center_heatmap = heatmaps[..., 0:1]
    tip_heatmap = heatmaps[..., 1:2]

    model = keras.Model(
        inputs=inputs,
        outputs={
            "center_heatmap": center_heatmap,
            "tip_heatmap": tip_heatmap,
        },
        name=f"mobilenetv2_geometry_reader_a{alpha}_hm{heatmap_size}",
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_geometry_uncertainty_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a geometry model with symmetric uncertainty bounds around the value."""
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    pooled_features = keras.layers.GlobalAveragePooling2D(
        name="geometry_pooled_features"
    )(x)
    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=2,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=2,
        name="keypoint_coords",
    )(heatmaps)
    gauge_value = GaugeValueFromKeypoints(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(keypoints)

    uncertainty_features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="geometry_uncertainty_dense",
    )(pooled_features)
    uncertainty_features = keras.layers.Dropout(head_dropout)(uncertainty_features)
    interval_radius = keras.layers.Dense(
        1,
        activation="softplus",
        name="gauge_value_interval_radius",
    )(uncertainty_features)
    half_interval = keras.layers.Rescaling(
        0.5,
        name="gauge_value_half_interval",
    )(interval_radius)
    gauge_value_lower = keras.layers.Subtract(name="gauge_value_lower")(
        [gauge_value, half_interval]
    )
    gauge_value_upper = keras.layers.Add(name="gauge_value_upper")(
        [gauge_value, half_interval]
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
            "gauge_value_lower": gauge_value_lower,
            "gauge_value_upper": gauge_value_upper,
        },
        name=_mobilenetv2_model_name(
            regression_kind="geometry_uncertainty",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_obb_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a joint OBB-plus-geometry model for detector-style gauge reading.

    The OBB branch encourages the network to stay aware of the full dial extent,
    while the keypoint branch turns that localized view into an explicit pointer
    geometry and the final Celsius value. This keeps the architecture closer to
    the recent literature than a pure scalar head.
    """
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    pooled_features = keras.layers.GlobalAveragePooling2D(
        name="obb_geometry_pooled_features"
    )(x)
    pooled_features = keras.layers.Dropout(head_dropout)(pooled_features)
    obb_features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="obb_geometry_dense",
    )(pooled_features)
    obb_features = keras.layers.Dropout(head_dropout)(obb_features)

    # Keep the OBB branch compact and directly supervised so it stays useful as
    # a localizer instead of competing with the geometry head for capacity.
    center_xy = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_center_xy",
    )(obb_features)
    size_wh = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_size_wh",
    )(obb_features)
    angle_raw = keras.layers.Dense(
        2,
        name="obb_angle_raw",
    )(obb_features)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1,
        name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy, size_wh, angle_sincos]
    )

    # The geometry branch reuses the dense feature map and predicts heatmaps
    # for the needle center/tip, which the soft-argmax layer turns into explicit
    # coordinates. That gives the final value a structural path rather than a
    # pure scalar shortcut.
    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=2,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=2,
        name="keypoint_coords",
    )(heatmaps)
    gauge_value = GaugeValueFromKeypoints(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
            "obb_params": obb_params,
        },
        name=_mobilenetv2_model_name(
            regression_kind="obb_geometry",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_obb_mask_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
    num_keypoints: int = 2,
) -> keras.Model:
    """Build an OBB-plus-mask model that combines localization and segmentation."""
    if num_keypoints < 2:
        raise ValueError("num_keypoints must be >= 2.")

    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    pooled_features = keras.layers.GlobalAveragePooling2D(
        name="obb_mask_pooled_features"
    )(x)
    pooled_features = keras.layers.Dropout(head_dropout)(pooled_features)
    obb_features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="obb_mask_dense",
    )(pooled_features)
    obb_features = keras.layers.Dropout(head_dropout)(obb_features)

    center_xy = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_center_xy",
    )(obb_features)
    size_wh = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_size_wh",
    )(obb_features)
    angle_raw = keras.layers.Dense(
        2,
        name="obb_angle_raw",
    )(obb_features)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1,
        name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy, size_wh, angle_sincos]
    )

    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=num_keypoints,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=num_keypoints,
        name="keypoint_coords",
    )(heatmaps)
    pointer_mask = _build_pointer_mask_head(
        x,
        mask_size=heatmap_size,
    )
    gauge_value = GaugeValueFromKeypoints(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
            "pointer_mask": pointer_mask,
            "obb_params": obb_params,
        },
        name=_mobilenetv2_model_name(
            regression_kind="obb_mask_geometry",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_obb_relation_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a blur-aware OBB-plus-relation model.

    This keeps the proven OBB relation family but adds a lightweight unsharp
    branch so low-contrast crops and preview frames have a better chance of
    surfacing the pointer/scale structure the reader needs.
    """
    if heatmap_size < 4:
        raise ValueError("heatmap_size must be >= 4.")

    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    raw_branch = keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1.0,
        name="obb_relation_raw_preprocess",
    )(inputs)
    enhanced_branch = _build_unsharp_mask_branch(
        raw_branch,
        name_prefix="obb_relation",
    )

    raw_maps = base_model(raw_branch, training=backbone_trainable)
    enhanced_maps = base_model(enhanced_branch, training=backbone_trainable)
    x = keras.layers.Average(name="obb_relation_feature_average")(
        [raw_maps, enhanced_maps]
    )

    pooled_features = keras.layers.GlobalAveragePooling2D(
        name="obb_relation_pooled_features"
    )(x)
    pooled_features = keras.layers.Dropout(head_dropout)(pooled_features)
    obb_features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="obb_relation_dense",
    )(pooled_features)
    obb_features = keras.layers.Dropout(head_dropout)(obb_features)

    center_xy = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_center_xy",
    )(obb_features)
    size_wh = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_size_wh",
    )(obb_features)
    angle_raw = keras.layers.Dense(
        2,
        name="obb_angle_raw",
    )(obb_features)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1,
        name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy, size_wh, angle_sincos]
    )

    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=4,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=4,
        name="keypoint_coords",
    )(heatmaps)
    pointer_mask = _build_pointer_mask_head(
        x,
        mask_size=heatmap_size,
    )
    gauge_value = GaugeValueFromRelationKeypoints(
        value_min=value_min,
        value_max=value_max,
        head_units=head_units,
        dropout=head_dropout,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
            "pointer_mask": pointer_mask,
            "obb_params": obb_params,
        },
        name=_mobilenetv2_model_name(
            regression_kind="obb_relation_geometry",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_bluraware_obb_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a blur-aware OBB-geometry model with raw and sharpened views."""
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
        name="bluraware_raw_preprocess",
    )(inputs)
    enhanced_branch = _build_unsharp_mask_branch(
        raw_branch,
        name_prefix="bluraware",
    )

    raw_maps = base_model(raw_branch, training=backbone_trainable)
    enhanced_maps = base_model(enhanced_branch, training=backbone_trainable)
    x = keras.layers.Average(name="bluraware_feature_average")(
        [raw_maps, enhanced_maps]
    )

    pooled_features = keras.layers.GlobalAveragePooling2D(
        name="obb_geometry_pooled_features"
    )(x)
    pooled_features = keras.layers.Dropout(head_dropout)(pooled_features)
    obb_features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="obb_geometry_dense",
    )(pooled_features)
    obb_features = keras.layers.Dropout(head_dropout)(obb_features)

    center_xy = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_center_xy",
    )(obb_features)
    size_wh = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_size_wh",
    )(obb_features)
    angle_raw = keras.layers.Dense(
        2,
        name="obb_angle_raw",
    )(obb_features)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1,
        name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy, size_wh, angle_sincos]
    )

    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=2,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=2,
        name="keypoint_coords",
    )(heatmaps)
    gauge_value = GaugeValueFromKeypoints(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
            "obb_params": obb_params,
        },
        name=_mobilenetv2_model_name(
            regression_kind="bluraware_obb_geometry",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_bluraware_obb_relation_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a blur-aware OBB reader with relation keypoints and mask output.

    This variant keeps the OBB backbone intact, adds a fixed unsharp-mask
    branch for low-contrast crops, and then learns pointer-mask plus relation
    geometry outputs on top. The design mirrors the recent literature trend:
    localize first, then reason about the pointer/scale relation explicitly.
    """
    if heatmap_size < 4:
        raise ValueError("heatmap_size must be >= 4.")

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
        name="bluraware_relation_raw_preprocess",
    )(inputs)
    enhanced_branch = _build_unsharp_mask_branch(
        raw_branch,
        name_prefix="bluraware_relation",
    )

    raw_maps = base_model(raw_branch, training=backbone_trainable)
    enhanced_maps = base_model(enhanced_branch, training=backbone_trainable)
    x = keras.layers.Average(name="bluraware_relation_feature_average")(
        [raw_maps, enhanced_maps]
    )

    pooled_features = keras.layers.GlobalAveragePooling2D(
        name="obb_relation_pooled_features"
    )(x)
    pooled_features = keras.layers.Dropout(head_dropout)(pooled_features)
    obb_features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="obb_relation_dense",
    )(pooled_features)
    obb_features = keras.layers.Dropout(head_dropout)(obb_features)

    center_xy = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_center_xy",
    )(obb_features)
    size_wh = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_size_wh",
    )(obb_features)
    angle_raw = keras.layers.Dense(
        2,
        name="obb_angle_raw",
    )(obb_features)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1,
        name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy, size_wh, angle_sincos]
    )

    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=4,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=4,
        name="keypoint_coords",
    )(heatmaps)
    pointer_mask = _build_pointer_mask_head(
        x,
        mask_size=heatmap_size,
    )
    gauge_value = GaugeValueFromRelationKeypoints(
        value_min=value_min,
        value_max=value_max,
        head_units=head_units,
        dropout=head_dropout,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
            "pointer_mask": pointer_mask,
            "obb_params": obb_params,
        },
        name=_mobilenetv2_model_name(
            regression_kind="bluraware_obb_relation_geometry",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_bluraware_obb_sequence_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a blur-aware OBB sequence-geometry model.

    This keeps the OBB front-end fixed, adds a cheap unsharp-mask branch for
    preview-like crops, and learns the pointer sequence with four keypoints
    plus a pointer mask. The value head still converts keypoints into the final
    gauge reading, which matches the literature-backed geometry-first pattern.
    """
    if heatmap_size < 4:
        raise ValueError("heatmap_size must be >= 4.")

    inputs, _, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    # Blend the raw crop with a lightly sharpened view so low-contrast frames
    # still expose thin pointer structure to the shared MobileNet backbone.
    raw_branch = keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1.0,
        name="bluraware_sequence_raw_preprocess",
    )(inputs)
    enhanced_branch = _build_unsharp_mask_branch(
        raw_branch,
        name_prefix="bluraware_sequence",
    )

    raw_maps = base_model(raw_branch, training=backbone_trainable)
    enhanced_maps = base_model(enhanced_branch, training=backbone_trainable)
    x = keras.layers.Average(name="bluraware_sequence_feature_average")(
        [raw_maps, enhanced_maps]
    )

    pooled_features = keras.layers.GlobalAveragePooling2D(
        name="obb_sequence_pooled_features"
    )(x)
    pooled_features = keras.layers.Dropout(head_dropout)(pooled_features)
    obb_features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="obb_sequence_dense",
    )(pooled_features)
    obb_features = keras.layers.Dropout(head_dropout)(obb_features)

    center_xy = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_center_xy",
    )(obb_features)
    size_wh = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_size_wh",
    )(obb_features)
    angle_raw = keras.layers.Dense(
        2,
        name="obb_angle_raw",
    )(obb_features)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1,
        name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy, size_wh, angle_sincos]
    )

    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=4,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=4,
        name="keypoint_coords",
    )(heatmaps)
    pointer_mask = _build_pointer_mask_head(
        x,
        mask_size=heatmap_size,
    )
    gauge_value = GaugeValueFromKeypoints(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
            "pointer_mask": pointer_mask,
            "obb_params": obb_params,
        },
        name=_mobilenetv2_model_name(
            regression_kind="bluraware_obb_sequence_geometry",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


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
    """Build a blur-aware scalar reader with raw and sharpened crop branches."""
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
        x = keras.layers.Dense(
            1,
            activation="linear",
            name="gauge_value_linear",
        )(x)
    else:
        x = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="gauge_value_sigmoid",
        )(x)

    output = keras.layers.Rescaling(
        scale=span,
        offset=value_min,
        name="gauge_value",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name=_mobilenetv2_model_name(
            regression_kind="bluraware_reader",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_rectifier_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a compact rectifier model that predicts a normalized dial crop box."""
    _ = pretrained
    _ = backbone_trainable
    _ = alpha

    # Keep the rectifier lightweight so the two-stage STM32 build stays within
    # the board flash budget while still learning a useful crop proposal.
    inputs, x = _build_feature_backbone(image_height, image_width)
    x = keras.layers.LayerNormalization(name="rectifier_pooled_features")(x)
    x = keras.layers.Dense(
        head_units,
        activation="swish",
        name="rectifier_dense",
    )(x)
    x = keras.layers.Dropout(head_dropout, name="rectifier_dropout")(x)

    # Predict normalized center-x, center-y, width, height in [0, 1].
    box = keras.layers.Dense(
        4,
        activation="sigmoid",
        name="rectifier_box",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs={"rectifier_box": box},
        name=_mobilenetv2_model_name(
            regression_kind="rectifier",
            alpha=1.0,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", None)
    return model


def build_mobilenetv2_source_crop_box_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a MobileNetV2 crop-box regressor over source-image coordinates.

    The head predicts two x corners and two y corners directly in normalized
    source-frame coordinates. This keeps the learning target closer to the
    offline rectified oracle than the older center/size rectifier.
    """
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    x = keras.layers.GlobalAveragePooling2D(name="source_crop_box_gap")(x)
    x = keras.layers.Dropout(head_dropout, name="source_crop_box_dropout_1")(x)
    x = keras.layers.Dense(
        head_units,
        activation="swish",
        name="source_crop_box_dense",
    )(x)
    x = keras.layers.Dropout(head_dropout, name="source_crop_box_dropout_2")(x)

    raw_box = keras.layers.Dense(
        4,
        activation="sigmoid",
        name="source_crop_box_raw",
    )(x)
    source_crop_box = OrderedCornerBox(name="source_crop_box")(raw_box)

    model = keras.Model(
        inputs=inputs,
        outputs={"source_crop_box": source_crop_box},
        name=_mobilenetv2_model_name(
            regression_kind="source_crop_box",
            alpha=1.0,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_source_crop_box_v2_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 256,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a MobileNetV2 crop-box regressor v2 with attention and stronger head.

    Changes from v1:
    - CoordinateAttention before GAP to preserve spatial structure.
    - Wider head (256 units default) for richer box regression.
    - Same output head (sigmoid + OrderedCornerBox) so warm-start from
      rectifier or v1 weights is possible for shared backbone layers.
    """
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    # Coordinate Attention to keep spatial structure before GAP
    x = CoordinateAttention(reduction_ratio=16, name="source_crop_box_v2_coord_attn")(x)

    x = keras.layers.GlobalAveragePooling2D(name="source_crop_box_v2_gap")(x)
    x = keras.layers.Dropout(head_dropout, name="source_crop_box_v2_dropout_1")(x)
    x = keras.layers.Dense(
        head_units,
        activation="swish",
        name="source_crop_box_v2_dense",
    )(x)
    x = keras.layers.Dropout(head_dropout, name="source_crop_box_v2_dropout_2")(x)

    raw_box = keras.layers.Dense(
        4,
        activation="sigmoid",
        name="source_crop_box_v2_raw",
    )(x)
    source_crop_box = OrderedCornerBox(name="source_crop_box")(raw_box)

    model = keras.Model(
        inputs=inputs,
        outputs={"source_crop_box": source_crop_box},
        name=_mobilenetv2_model_name(
            regression_kind="source_crop_box_v2",
            alpha=1.0,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_source_crop_corner_model(
    image_height: int,
    image_width: int,
    *,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a corner-localizer that decodes four heatmaps into a crop box.

    This candidate keeps the localization problem geometric: it learns the
    four crop corners with heatmaps, converts those to corner coordinates, and
    then reduces the coordinates to an axis-aligned source crop box.
    """
    _ = head_units
    _ = head_dropout

    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=4,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=4,
        name="keypoint_coords",
    )(heatmaps)
    source_crop_canvas_box = CornerKeypointsToBox(
        heatmap_size=heatmap_size,
        name="source_crop_canvas_box",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "source_crop_canvas_box": source_crop_canvas_box,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
        },
        name=_mobilenetv2_model_name(
            regression_kind="source_crop_corner",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_obb_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a MobileNetV2 localizer that predicts oriented ellipse parameters.

    The head predicts a normalized center and size plus a unit angle vector.
    That gives us a more explicit detector-style target than the earlier
    keypoint heatmap proxies while still staying compact enough for embedded
    experiments.
    """
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    center_xy = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_center_xy",
    )(x)
    size_wh = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_size_wh",
    )(x)
    angle_raw = keras.layers.Dense(
        2,
        name="obb_angle_raw",
    )(x)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1,
        name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy, size_wh, angle_sincos]
    )

    model = keras.Model(
        inputs=inputs,
        outputs={"obb_params": obb_params},
        name=_mobilenetv2_model_name(
            regression_kind="obb",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_center_detection_model(
    image_height: int = 224,
    image_width: int = 224,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a MobileNetV2-based center detector with an auxiliary needle-colour head.

    The primary output ``center_xy`` predicts a normalized (cx, cy) in [0, 1]
    that locates the dial center / needle pivot.  The auxiliary head predicts
    whether the needle is dark or light (used as a weak regulariser and to
    support future multi-colour gauges).
    """
    inputs = keras.Input(
        shape=(image_height, image_width, 3), name="image"
    )

    # The pipeline emits [0, 1] floats; MobileNetV2 expects [-1, 1].
    x = keras.layers.Rescaling(
        1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess"
    )(inputs)

    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_height, image_width, 3),
        alpha=alpha,
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="center_detection_gap")(x)
    x = keras.layers.Dense(
        head_units, activation="swish", name="center_detection_dense"
    )(x)
    x = keras.layers.Dropout(head_dropout, name="center_detection_dropout")(x)

    center_xy = keras.layers.Dense(
        2, activation="sigmoid", name="center_xy"
    )(x)
    needle_colour_head = keras.layers.Dense(
        2, activation="softmax", name="needle_colour_head"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "center_xy": center_xy,
            "needle_colour_head": needle_colour_head,
        },
        name="center_detector",
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_enhanced_regression_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = True,
    alpha: float = 1.0,
    head_units: int = 256,
    head_dropout: float = 0.3,
    value_min: float = -30.0,
    value_max: float = 50.0,
    use_multi_scale: bool = True,
    use_coord_attention: bool = False,
) -> keras.Model:
    """Build an enhanced MobileNetV2 regressor with multi-scale features and attention.

    Architecture improvements over the baseline mobilenet_v2:
    1. Multi-scale feature fusion — combines early/mid/late MobileNetV2 features
       so the head sees both fine needle detail and global dial context.
    2. CBAM attention — channel + spatial attention at each scale to suppress
       background clutter and highlight the needle region.
    3. Wider head — 256 units with 0.3 dropout for more capacity.
    4. LayerNorm + residual connections in the head for training stability.
    5. Linear output — no sigmoid saturation at temperature extremes.
    6. Auxiliary sweep-fraction head — provides geometric supervision.

    Reference: Woo et al. "CBAM" (ECCV 2018), Hou et al. "Coordinate Attention" (CVPR 2021)
    """
    span = value_max - value_min

    if use_multi_scale:
        # Multi-scale backbone with feature pyramid
        inputs, multi_scale_features, base_model = (
            _build_mobilenetv2_multi_scale_backbone(
                image_height,
                image_width,
                pretrained=pretrained,
                backbone_trainable=backbone_trainable,
                alpha=alpha,
            )
        )
        # Fuse multi-scale features with CBAM
        spatial_features = _cbam_refine(multi_scale_features, base_channels=64)
        # Global pooling for regression head
        x = keras.layers.GlobalAveragePooling2D(name="enhanced_gap")(spatial_features)
        # Also keep max-pooled features for sharper responses
        x_max = keras.layers.GlobalMaxPooling2D(name="enhanced_gmp")(spatial_features)
        x = keras.layers.Concatenate(name="enhanced_pool_fusion")([x, x_max])
    else:
        # Standard backbone with coordinate attention on final features
        inputs, final_features, base_model = _build_mobilenetv2_backbone(
            image_height,
            image_width,
            pretrained=pretrained,
            backbone_trainable=backbone_trainable,
            alpha=alpha,
        )
        if use_coord_attention:
            final_features = CoordinateAttention(
                reduction_ratio=16, name="enhanced_coord_attn"
            )(final_features)
        x = keras.layers.GlobalAveragePooling2D(name="enhanced_gap")(final_features)
        x_max = keras.layers.GlobalMaxPooling2D(name="enhanced_gmp")(final_features)
        x = keras.layers.Concatenate(name="enhanced_pool_fusion")([x, x_max])

    # Wider, deeper regression head with LayerNorm for stability
    x = keras.layers.LayerNormalization(name="enhanced_head_norm_0")(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish", name="enhanced_dense_1")(x)
    x = keras.layers.LayerNormalization(name="enhanced_head_norm_1")(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(
        head_units // 2, activation="swish", name="enhanced_dense_2"
    )(x)
    x = keras.layers.LayerNormalization(name="enhanced_head_norm_2")(x)
    x = keras.layers.Dropout(head_dropout * 0.5)(x)

    # Main scalar output — linear (no sigmoid) to avoid saturation at extremes
    main_output = keras.layers.Dense(1, name="gauge_value_raw")(x)
    gauge_value = keras.layers.Rescaling(
        scale=span, offset=value_min, name="gauge_value"
    )(main_output)

    # Auxiliary sweep-fraction head for geometric supervision
    fraction_logit = keras.layers.Dense(1, name="sweep_fraction_raw")(x)
    sweep_fraction = keras.layers.Activation("sigmoid", name="sweep_fraction")(
        fraction_logit
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "sweep_fraction": sweep_fraction,
        },
        name=f"mobilenetv2_enhanced_gauge_regressor",
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_enhanced_ensemble_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = True,
    alpha: float = 1.0,
    head_units: int = 256,
    head_dropout: float = 0.3,
    value_min: float = -30.0,
    value_max: float = 50.0,
    num_heads: int = 3,
) -> keras.Model:
    """Build an enhanced MobileNetV2 with multi-head ensemble.

    Uses a shared backbone with multiple independent regression heads.
    At inference, heads are averaged for a more robust prediction.
    Each head gets its own dropout pattern for diversity.
    """
    span = value_max - value_min

    inputs, multi_scale_features, base_model = _build_mobilenetv2_multi_scale_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    spatial_features = _cbam_refine(multi_scale_features, base_channels=64)
    x = keras.layers.GlobalAveragePooling2D(name="ensemble_gap")(spatial_features)
    x_max = keras.layers.GlobalMaxPooling2D(name="ensemble_gmp")(spatial_features)
    shared = keras.layers.Concatenate(name="ensemble_pool_fusion")([x, x_max])
    shared = keras.layers.LayerNormalization(name="ensemble_shared_norm")(shared)

    head_outputs: list[keras.KerasTensor] = []
    for i in range(num_heads):
        h = keras.layers.Dropout(head_dropout, name=f"ensemble_head_{i}_drop_1")(shared)
        h = keras.layers.Dense(head_units, use_bias=False, name=f"ensemble_dense_{i}")(
            h
        )
        h = keras.layers.LayerNormalization(name=f"ensemble_ln_{i}")(h)
        h = keras.layers.Activation("swish")(h)
        h = keras.layers.Dropout(head_dropout)(h)

        h_linear = keras.layers.Dense(
            1, activation="linear", name=f"gauge_value_head_{i}_linear"
        )(h)
        h_out = keras.layers.Rescaling(
            scale=span, offset=value_min, name=f"gauge_value_head_{i}"
        )(h_linear)
        head_outputs.append(h_out)

    # Average all heads
    if num_heads > 1:
        avg_raw = keras.layers.Average(name="ensemble_avg")(head_outputs)
    else:
        avg_raw = head_outputs[0]

    gauge_value = keras.layers.Rescaling(
        scale=span, offset=value_min, name="gauge_value"
    )(avg_raw)

    model = keras.Model(
        inputs=inputs,
        outputs={"gauge_value": gauge_value},
        name=f"mobilenetv2_enhanced_ensemble_{num_heads}h_gauge_regressor",
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_sota_multiscale_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = True,
    alpha: float = 1.0,
    head_units: int = 256,
    head_dropout: float = 0.3,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> keras.Model:
    """Build SOTA model with multi-scale features and dual attention.

    This model incorporates the best practices from gauge-reading literature:
    1. Multi-scale feature fusion (FPN-style)
    2. CBAM + Coordinate Attention
    3. Wide head with LayerNorm
    4. Linear output (no saturation)
    5. Auxiliary sweep fraction head

    Args:
        image_height: Input image height
        image_width: Input image width
        pretrained: Use ImageNet weights
        backbone_trainable: Fine-tune backbone
        alpha: MobileNetV2 width multiplier
        head_units: Head FC layer units
        head_dropout: Dropout rate
        value_min: Minimum temperature value
        value_max: Maximum temperature value

    Returns:
        Keras model with gauge_value output
    """
    span = value_max - value_min

    # Build multi-scale backbone
    inputs, multi_scale_features, base_model = _build_mobilenetv2_multi_scale_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    # Fuse multi-scale features with CBAM
    spatial_features = _cbam_refine(multi_scale_features, base_channels=64)

    # Global pooling fusion
    x = keras.layers.GlobalAveragePooling2D(name="sota_gap")(spatial_features)
    x_max = keras.layers.GlobalMaxPooling2D(name="sota_gmp")(spatial_features)
    x = keras.layers.Concatenate(name="sota_pool_fusion")([x, x_max])

    # Wide head with LayerNorm
    x = keras.layers.Dense(head_units, use_bias=False, name="sota_dense_1")(x)
    x = keras.layers.LayerNormalization(name="sota_ln_1")(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    x = keras.layers.Dense(head_units // 2, use_bias=False, name="sota_dense_2")(x)
    x = keras.layers.LayerNormalization(name="sota_ln_2")(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.Dropout(head_dropout * 0.5)(x)

    # Linear output head
    gauge_value_linear = keras.layers.Dense(
        1, activation="linear", name="gauge_value_linear"
    )(x)
    gauge_value = keras.layers.Rescaling(
        scale=span, offset=value_min, name="gauge_value"
    )(gauge_value_linear)

    # Auxiliary sweep fraction head for geometric supervision
    sweep_fraction_logits = keras.layers.Dense(1, name="sweep_fraction_logits")(x)
    sweep_fraction = keras.layers.Activation("sigmoid", name="sweep_fraction")(
        sweep_fraction_logits
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "sweep_fraction": sweep_fraction,
        },
        name="mobilenetv2_sota_multiscale",
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_sota_ensemble_model(
    image_height: int,
    image_width: int,
    *,
    num_heads: int = 3,
    pretrained: bool = True,
    backbone_trainable: bool = True,
    alpha: float = 1.0,
    head_units: int = 192,
    head_dropout: float = 0.25,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> keras.Model:
    """Build ensemble model with multiple independent heads.

    Ensemble averaging reduces variance and improves generalization,
    especially on hard cases. Each head sees the same multi-scale
    features but learns independent weights.
    """
    span = value_max - value_min

    # Build shared multi-scale backbone
    inputs, multi_scale_features, base_model = _build_mobilenetv2_multi_scale_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    # Fuse features
    spatial_features = _cbam_refine(multi_scale_features, base_channels=64)
    x = keras.layers.GlobalAveragePooling2D(name="ensemble_gap")(spatial_features)

    # Create multiple independent heads
    head_outputs = []
    for i in range(num_heads):
        h = keras.layers.Dense(head_units, use_bias=False, name=f"ensemble_dense_{i}")(
            x
        )
        h = keras.layers.LayerNormalization(name=f"ensemble_ln_{i}")(h)
        h = keras.layers.Activation("swish")(h)
        h = keras.layers.Dropout(head_dropout)(h)

        h_linear = keras.layers.Dense(
            1, activation="linear", name=f"gauge_value_head_{i}_linear"
        )(h)
        h_out = keras.layers.Rescaling(
            scale=span, offset=value_min, name=f"gauge_value_head_{i}"
        )(h_linear)
        head_outputs.append(h_out)

    # Average ensemble predictions
    if len(head_outputs) > 1:
        gauge_value = keras.layers.Average(name="gauge_value_ensemble")(head_outputs)
    else:
        gauge_value = head_outputs[0]

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            **{f"gauge_value_head_{i}": head_outputs[i] for i in range(num_heads)},
        },
        name=f"mobilenetv2_sota_ensemble_{num_heads}h",
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_uncertainty_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = True,
    alpha: float = 1.0,
    head_units: int = 256,
    head_dropout: float = 0.3,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> keras.Model:
    """Build model with uncertainty estimation via quantile regression.

    Predicts median, lower, and upper quantiles to estimate
    prediction uncertainty. Useful for identifying hard cases
    and low-confidence predictions.

    Args:
        image_height: Input image height
        image_width: Input image width
        pretrained: Use ImageNet weights
        backbone_trainable: Fine-tune backbone
        alpha: MobileNetV2 width multiplier
        head_units: Head FC layer units
        head_dropout: Dropout rate
        value_min: Minimum temperature value
        value_max: Maximum temperature value

    Returns:
        Keras model with gauge_value and uncertainty bounds
    """
    span = value_max - value_min

    # Build multi-scale backbone
    inputs, multi_scale_features, base_model = _build_mobilenetv2_multi_scale_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    # Fuse features
    spatial_features = _cbam_refine(multi_scale_features, base_channels=64)
    x = keras.layers.GlobalAveragePooling2D(name="uncertainty_gap")(spatial_features)
    x_max = keras.layers.GlobalMaxPooling2D(name="uncertainty_gmp")(spatial_features)
    x = keras.layers.Concatenate(name="uncertainty_pool_fusion")([x, x_max])

    # Wide head
    x = keras.layers.Dense(head_units, use_bias=False, name="uncertainty_dense_1")(x)
    x = keras.layers.LayerNormalization(name="uncertainty_ln_1")(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    x = keras.layers.Dense(head_units // 2, use_bias=False, name="uncertainty_dense_2")(
        x
    )
    x = keras.layers.LayerNormalization(name="uncertainty_ln_2")(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.Dropout(head_dropout * 0.5)(x)

    # Predict median (main value)
    median_linear = keras.layers.Dense(1, activation="linear", name="median_linear")(x)
    gauge_value = keras.layers.Rescaling(
        scale=span, offset=value_min, name="gauge_value"
    )(median_linear)

    # Predict uncertainty bounds (learned offsets)
    # Use softplus to ensure positive intervals
    lower_offset = keras.layers.Dense(1, activation="softplus", name="lower_offset")(x)
    upper_offset = keras.layers.Dense(1, activation="softplus", name="upper_offset")(x)

    # Compute bounds
    gauge_value_lower = keras.layers.Subtract(name="gauge_value_lower")(
        [gauge_value, lower_offset]
    )
    gauge_value_upper = keras.layers.Add(name="gauge_value_upper")(
        [gauge_value, upper_offset]
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "gauge_value_lower": gauge_value_lower,
            "gauge_value_upper": gauge_value_upper,
        },
        name="mobilenetv2_sota_uncertainty",
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_heatmap_center_model(
    image_height: int = 224,
    image_width: int = 224,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 0.35,
    heatmap_size: int = 56,
    temperature: float = 10.0,
) -> keras.Model:
    """Build a U-Net-style center detector that predicts a 2D heatmap then
    extracts (cx, cy) via softargmax.

    The MobileNetV2 encoder stays frozen (based on our ablation finding that
    finetuning destroys CD generalisation).  Skip connections from three
    intermediate resolutions feed a lightweight decoder that predicts a single-
    channel heatmap at ``heatmap_size`` resolution.  ``SpatialSoftArgmax2D``
    converts that heatmap into differentiable (cx, cy) coordinates.

    Outputs:
        center_xy: (B, 2) — normalised (cx, cy) in [0, 1].
        heatmap:   (B, H, H, 1) — sigmoid heatmap for auxiliary supervision.
    """
    inputs = keras.Input(
        shape=(image_height, image_width, 3), name="image"
    )

    # The training pipeline emits [0, 1] floats; MobileNetV2 expects [-1, 1].
    x = keras.layers.Rescaling(
        1.0 / 127.5, offset=-1.0, name="preprocess"
    )(inputs)

    backbone = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(image_height, image_width, 3),
        alpha=alpha,
    )
    backbone.trainable = backbone_trainable

    # Extract intermediate feature maps for U-Net skip connections.
    #   block_2_add   → 56×56   (stride 4)
    #   block_4_add   → 28×28   (stride 8)
    #   block_12_add  → 14×14   (stride 16, before stride-32 downsampling)
    skip_layer_names = [
        "block_2_add",
        "block_4_add",
        "block_12_add",
    ]
    skip_outputs = [
        backbone.get_layer(name).output for name in skip_layer_names
    ]

    encoder = keras.Model(
        inputs=backbone.inputs,
        outputs=skip_outputs + [backbone.output],
        name="mobilenetv2_heatmap_encoder",
    )

    encoded = encoder(x, training=backbone_trainable)
    s_56 = encoded[0]   # 56×56 × C1  (8 channels at alpha=0.35)
    s_28 = encoded[1]   # 28×28 × C2  (16 channels)
    s_14 = encoded[2]   # 14×14 × C3  (32 channels)
    bottlneck = encoded[3]  # 7×7 × 1280

    # ---- Decoder: progressive upsampling with skip connections ----
    # Stage 1:  7×7 → 14×14
    d = keras.layers.Conv2D(
        256, 3, padding="same", use_bias=False, name="dec_conv1"
    )(bottlneck)
    d = keras.layers.BatchNormalization(
        momentum=0.9, epsilon=1e-3, name="dec_bn1"
    )(d)
    d = keras.layers.Activation("swish")(d)
    d = keras.layers.UpSampling2D(2, interpolation="bilinear", name="dec_up1")(
        d
    )
    d = keras.layers.Concatenate(name="dec_cat1")([d, s_14])
    d = keras.layers.Conv2D(
        128, 3, padding="same", use_bias=False, name="dec_conv2"
    )(d)
    d = keras.layers.BatchNormalization(
        momentum=0.9, epsilon=1e-3, name="dec_bn2"
    )(d)
    d = keras.layers.Activation("swish")(d)

    # Stage 2: 14×14 → 28×28
    d = keras.layers.UpSampling2D(2, interpolation="bilinear", name="dec_up2")(
        d
    )
    d = keras.layers.Concatenate(name="dec_cat2")([d, s_28])
    d = keras.layers.Conv2D(
        64, 3, padding="same", use_bias=False, name="dec_conv3"
    )(d)
    d = keras.layers.BatchNormalization(
        momentum=0.9, epsilon=1e-3, name="dec_bn3"
    )(d)
    d = keras.layers.Activation("swish")(d)

    # Stage 3: 28×28 → 56×56
    d = keras.layers.UpSampling2D(2, interpolation="bilinear", name="dec_up3")(
        d
    )
    d = keras.layers.Concatenate(name="dec_cat3")([d, s_56])
    d = keras.layers.Conv2D(
        32, 3, padding="same", use_bias=False, name="dec_conv4"
    )(d)
    d = keras.layers.BatchNormalization(
        momentum=0.9, epsilon=1e-3, name="dec_bn4"
    )(d)
    d = keras.layers.Activation("swish")(d)

    # Final refinement conv
    d = keras.layers.Conv2D(
        32, 3, padding="same", use_bias=False, name="dec_conv5"
    )(d)
    d = keras.layers.BatchNormalization(
        momentum=0.9, epsilon=1e-3, name="dec_bn5"
    )(d)
    d = keras.layers.Activation("swish")(d)

    # Single-channel sigmoid heatmap
    heatmap = keras.layers.Conv2D(
        1, 1, activation="sigmoid", name="heatmap"
    )(d)

    # Differentiable soft-argmax → (cx, cy) in heatmap-pixel coords
    center_xy_raw = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=1,
        temperature=temperature,
        name="center_xy_raw",
    )(heatmap)
    # Normalise to [0, 1] so the MSE loss has a stable scale matching the
    # metadata labels (which are also in normalised [0, 1]).
    center_xy_vec = keras.layers.Reshape((2,), name="center_xy_vec")(
        center_xy_raw
    )
    center_xy = keras.layers.Rescaling(
        1.0 / float(max(heatmap_size - 1, 1)),
        name="center_xy",
    )(center_xy_vec)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "center_xy": center_xy,
            "heatmap": heatmap,
        },
        name=f"mobilenetv2_heatmap_cd_a{int(round(alpha*100)):03d}_hm{heatmap_size}",
    )
    setattr(model, "_mobilenet_backbone", backbone)
    return model


def build_mobilenetv2_obb_center_model(
    image_height: int = 320,
    image_width: int = 320,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 0.50,
    head_units_1: int = 256,
    head_units_2: int = 96,
    head_dropout_1: float = 0.25,
    head_dropout_2: float = 0.15,
) -> keras.Model:
    """Build a MobileNetV2 + CoordConv model that jointly predicts OBB and center.

    Uses a shared MobileNetV2 backbone with CoordConv positional encoding so the
    model always knows absolute pixel position — critical for precise center
    regression after the bottleneck GAP layer discards most spatial information.
    Two lightweight prediction heads share the same features:
      - ``obb_params``: [cx, cy, w, h, cos(2θ), sin(2θ)] in normalised coords.
      - ``center_xy``:  [cx, cy] in normalised [0,1] coords (needle pivot).

    Args:
        image_height: Input image height (pixels).
        image_width: Input image width (pixels).
        pretrained: Whether to load ImageNet weights.
        backbone_trainable: Whether the backbone is trainable (False for
            warm-up, True for fine-tuning).
        alpha: MobileNetV2 width multiplier.
        head_units_1: Units in the first dense layer after GAP.
        head_units_2: Units in the second dense layer.
        head_dropout_1: Dropout rate after the first dense layer.
        head_dropout_2: Dropout rate after the second dense layer.

    Returns:
        A ``keras.Model`` with dict outputs ``{"obb_params", "center_xy"}``.
    """
    inputs = keras.Input(shape=(image_height, image_width, 3), name="image")

    # --- CoordConv: positional encoding channels -----------------------------
    # Generate normalised x/y coordinate grids in [-1, 1] so the model
    # always knows absolute position — GAP on a 10×10 feature map discards
    # almost all spatial information, so we inject it here.
    y_grid = tf.linspace(-1.0, 1.0, image_height)
    x_grid = tf.linspace(-1.0, 1.0, image_width)
    yy, xx = tf.meshgrid(y_grid, x_grid, indexing="ij")
    coords = tf.stack([xx, yy], axis=-1)  # [H, W, 2]
    coords = coords[tf.newaxis, ...]  # [1, H, W, 2]
    coord_input = keras.layers.Lambda(
        lambda img: tf.tile(
            coords, [tf.shape(img)[0], 1, 1, 1],
        ),
        output_shape=(image_height, image_width, 2),
        name="coord_channels",
    )(inputs)

    # Concatenate RGB + coordinate channels → 5-channel input
    x = keras.layers.Concatenate(name="rgb_coords")([inputs, coord_input])

    # 1×1 conv to map 5 channels back to 3 so MobileNetV2 can consume them.
    # This learns a small per-pixel projection from (colour + position) to
    # a colour-like 3-channel representation.
    x = keras.layers.Conv2D(
        3, kernel_size=1, use_bias=False, name="coord_proj",
    )(x)

    # Preprocessing: [0, 1] → [-1, 1] for MobileNetV2.
    x = keras.layers.Rescaling(
        2.0, offset=-1.0, name="mobilenetv2_preprocess",
    )(x)

    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(image_height, image_width, 3),
        alpha=alpha,
    )
    base_model.trainable = backbone_trainable
    features = base_model(x, training=backbone_trainable)

    # --- Deeper shared head for better representational capacity -------------
    x = keras.layers.GlobalAveragePooling2D(name="dual_gap")(features)
    x = keras.layers.LayerNormalization(name="dual_ln")(x)
    x = keras.layers.Dense(
        head_units_1, activation="swish", name="dual_dense_1",
    )(x)
    x = keras.layers.Dropout(head_dropout_1, name="dual_dropout_1")(x)
    x = keras.layers.Dense(
        head_units_2, activation="swish", name="dual_dense_2",
    )(x)
    x = keras.layers.Dropout(head_dropout_2, name="dual_dropout_2")(x)

    # OBB head: 6-parameter oriented bounding box.
    center_xy_obb = keras.layers.Dense(
        2, activation="sigmoid", name="obb_center_xy",
    )(x)
    size_wh = keras.layers.Dense(
        2, activation="sigmoid", name="obb_size_wh",
    )(x)
    angle_raw = keras.layers.Dense(2, name="obb_angle_raw")(x)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1, name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy_obb, size_wh, angle_sincos],
    )

    # Center head: direct (cx, cy) prediction for the needle pivot.
    center_xy = keras.layers.Dense(
        2, activation="sigmoid", name="center_xy",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs={"obb_params": obb_params, "center_xy": center_xy},
        name=(
            f"mobilenetv2_obb_center_cc_a{int(round(alpha*100)):03d}"
            f"_h{head_units_1}-{head_units_2}"
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv3_obb_center_model(
    image_height: int = 320,
    image_width: int = 320,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 0.75,
    head_units_1: int = 256,
    head_units_2: int = 96,
    head_dropout_1: float = 0.25,
    head_dropout_2: float = 0.15,
) -> keras.Model:
    """Build a MobileNetV3-Small + CoordConv model that jointly predicts OBB and center.

    Uses a shared MobileNetV3-Small backbone with CoordConv positional encoding so the
    model always knows absolute pixel position — critical for precise center
    regression after the bottleneck GAP layer discards most spatial information.
    Two lightweight prediction heads share the same features:
      - ``obb_params``: [cx, cy, w, h, cos(2θ), sin(2θ)] in normalised coords.
      - ``center_xy``:  [cx, cy] in normalised [0,1] coords (needle pivot).

    MobileNetV3-Small expects [0, 1] input (unlike V2 which uses [-1, 1]).

    Args:
        image_height: Input image height (pixels).
        image_width: Input image width (pixels).
        pretrained: Whether to load ImageNet weights.
        backbone_trainable: Whether the backbone is trainable (False for
            warm-up, True for fine-tuning).
        alpha: MobileNetV3-Small width multiplier.
        head_units_1: Units in the first dense layer after GAP.
        head_units_2: Units in the second dense layer.
        head_dropout_1: Dropout rate after the first dense layer.
        head_dropout_2: Dropout rate after the second dense layer.

    Returns:
        A ``keras.Model`` with dict outputs ``{"obb_params", "center_xy"}``.
    """
    inputs = keras.Input(shape=(image_height, image_width, 3), name="image")

    # --- CoordConv: positional encoding channels -----------------------------
    # Generate normalised x/y coordinate grids in [-1, 1] so the model
    # always knows absolute position — GAP on a 10×10 feature map discards
    # almost all spatial information, so we inject it here.
    y_grid = tf.linspace(-1.0, 1.0, image_height)
    x_grid = tf.linspace(-1.0, 1.0, image_width)
    yy, xx = tf.meshgrid(y_grid, x_grid, indexing="ij")
    coords = tf.stack([xx, yy], axis=-1)  # [H, W, 2]
    coords = coords[tf.newaxis, ...]  # [1, H, W, 2]
    coord_input = keras.layers.Lambda(
        lambda img: tf.tile(
            coords, [tf.shape(img)[0], 1, 1, 1],
        ),
        output_shape=(image_height, image_width, 2),
        name="coord_channels",
    )(inputs)

    # Concatenate RGB + coordinate channels → 5-channel input
    x = keras.layers.Concatenate(name="rgb_coords")([inputs, coord_input])

    # 1×1 conv to map 5 channels back to 3 so MobileNetV3 can consume them.
    # This learns a small per-pixel projection from (colour + position) to
    # a colour-like 3-channel representation.
    x = keras.layers.Conv2D(
        3, kernel_size=1, use_bias=False, name="coord_proj",
    )(x)

    # MobileNetV3-Small expects [0, 1] input — no rescaling needed.
    base_model = keras.applications.MobileNetV3Small(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(image_height, image_width, 3),
        alpha=alpha,
    )
    base_model.trainable = backbone_trainable
    features = base_model(x, training=backbone_trainable)

    # --- Shared head for better representational capacity --------------------
    x = keras.layers.GlobalAveragePooling2D(name="dual_gap")(features)
    x = keras.layers.LayerNormalization(name="dual_ln")(x)
    x = keras.layers.Dense(
        head_units_1, activation="swish", name="dual_dense_1",
    )(x)
    x = keras.layers.Dropout(head_dropout_1, name="dual_dropout_1")(x)
    x = keras.layers.Dense(
        head_units_2, activation="swish", name="dual_dense_2",
    )(x)
    x = keras.layers.Dropout(head_dropout_2, name="dual_dropout_2")(x)

    # OBB head: 6-parameter oriented bounding box.
    center_xy_obb = keras.layers.Dense(
        2, activation="sigmoid", name="obb_center_xy",
    )(x)
    size_wh = keras.layers.Dense(
        2, activation="sigmoid", name="obb_size_wh",
    )(x)
    angle_raw = keras.layers.Dense(2, name="obb_angle_raw")(x)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1, name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy_obb, size_wh, angle_sincos],
    )

    # Center head: direct (cx, cy) prediction for the needle pivot.
    center_xy = keras.layers.Dense(
        2, activation="sigmoid", name="center_xy",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs={"obb_params": obb_params, "center_xy": center_xy},
        name=(
            f"mobilenetv3_small_obb_center_cc_a{int(round(alpha*100)):03d}"
            f"_h{head_units_1}-{head_units_2}"
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


