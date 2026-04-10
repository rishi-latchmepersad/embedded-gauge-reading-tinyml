"""Model builders for gauge-reading networks."""

from __future__ import annotations

import math

import keras


def _norm(x: keras.KerasTensor) -> keras.KerasTensor:
    """Apply convolution-friendly normalization for image features."""
    return keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(x)


def _conv_norm_swish(
    x: keras.KerasTensor,
    filters: int,
    *,
    kernel_size: int = 3,
    strides: int = 1,
) -> keras.KerasTensor:
    """Apply Conv2D + normalization + swish."""
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
    )(x)
    x = _norm(x)
    x = keras.layers.Activation("swish")(x)
    return x


def _residual_separable_block(
    x: keras.KerasTensor,
    filters: int,
    *,
    dropout_rate: float = 0.0,
) -> keras.KerasTensor:
    """Apply a lightweight residual block with separable convolutions."""
    shortcut: keras.KerasTensor = x

    x = keras.layers.SeparableConv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
    )(x)
    x = _norm(x)
    x = keras.layers.Activation("swish")(x)

    x = keras.layers.SeparableConv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
    )(x)
    x = _norm(x)

    if shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
        )(shortcut)
        shortcut = _norm(shortcut)

    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation("swish")(x)

    if dropout_rate > 0.0:
        x = keras.layers.Dropout(dropout_rate)(x)

    return x


def _build_feature_backbone(
    image_height: int, image_width: int
) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    """Build the shared CNN backbone and return (inputs, pooled_features)."""
    inputs = keras.Input(shape=(image_height, image_width, 3), name="image")

    x = _conv_norm_swish(inputs, 32, strides=2)
    x = _residual_separable_block(x, 32, dropout_rate=0.02)
    x = keras.layers.MaxPool2D(pool_size=2)(x)

    x = _residual_separable_block(x, 64, dropout_rate=0.04)
    x = keras.layers.MaxPool2D(pool_size=2)(x)

    x = _residual_separable_block(x, 96, dropout_rate=0.06)
    x = keras.layers.MaxPool2D(pool_size=2)(x)

    x = _residual_separable_block(x, 128, dropout_rate=0.08)
    x = keras.layers.GlobalAveragePooling2D()(x)
    return inputs, x


def _build_mobilenetv2_backbone(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool,
    backbone_trainable: bool,
    alpha: float,
) -> tuple[keras.KerasTensor, keras.KerasTensor, keras.Model]:
    """Build a MobileNetV2 feature backbone and return the pooled feature map."""
    inputs = keras.Input(shape=(image_height, image_width, 3), name="image")

    # The pipeline emits [0, 1] floats; MobileNetV2 expects [-1, 1].
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(
        inputs
    )

    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(image_height, image_width, 3),
        alpha=alpha,
    )
    base_model.trainable = backbone_trainable

    x = base_model(x, training=backbone_trainable)
    return inputs, x, base_model


def _mobilenetv2_model_name(
    *,
    regression_kind: str,
    alpha: float,
    head_units: int,
) -> str:
    """Generate a stable model name for a MobileNetV2 variant."""
    if math.isclose(alpha, 1.0, rel_tol=0.0, abs_tol=1e-6) and head_units == 128:
        return f"mobilenetv2_{regression_kind}_regressor"

    alpha_tag: int = int(round(alpha * 100.0))
    return f"mobilenetv2_{regression_kind}_regressor_a{alpha_tag:03d}_h{head_units:03d}"


def _build_interval_expectation_head(
    logits: keras.KerasTensor,
    *,
    value_min: float,
    value_max: float,
    bin_width: float,
) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    """Convert interval logits into a scalar expected temperature value."""
    if bin_width <= 0.0:
        raise ValueError("bin_width must be > 0.")
    if value_max <= value_min:
        raise ValueError("value_max must be > value_min.")

    span = value_max - value_min
    num_bins = int(math.ceil(span / bin_width))
    if num_bins < 2:
        raise ValueError("Need at least two bins for interval regression.")

    # Place one center per fixed-width interval, anchored at the calibrated range.
    bin_centers = [
        value_min + (index + 0.5) * bin_width for index in range(num_bins)
    ]
    centers = keras.layers.Dense(
        1,
        use_bias=False,
        trainable=False,
        name="interval_value_projection",
        kernel_initializer=keras.initializers.Constant(
            [[center] for center in bin_centers]
        ),
    )
    probs = keras.layers.Softmax(name="interval_probs")(logits)
    value = centers(probs)
    return value, probs


def _build_interval_hybrid_head(
    features: keras.KerasTensor,
    logits: keras.KerasTensor,
    *,
    value_min: float,
    value_max: float,
    bin_width: float,
) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    """Combine coarse bin expectation with a bounded residual correction."""
    coarse_value, interval_probs = _build_interval_expectation_head(
        logits,
        value_min=value_min,
        value_max=value_max,
        bin_width=bin_width,
    )

    # Let the network learn a bounded local correction inside the coarse bin.
    half_bin_width = 0.5 * bin_width
    residual = keras.layers.Dense(
        1,
        activation="tanh",
        name="interval_residual_raw",
    )(features)
    residual = keras.layers.Rescaling(
        half_bin_width,
        offset=0.0,
        name="interval_residual",
    )(residual)

    value = keras.layers.Add(name="gauge_value")([coarse_value, residual])
    return value, interval_probs


def _build_ordinal_expectation_head(
    logits: keras.KerasTensor,
    *,
    value_min: float,
    value_max: float,
    threshold_step: float,
) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    """Convert ordinal threshold logits into a scalar expected value."""
    if threshold_step <= 0.0:
        raise ValueError("threshold_step must be > 0.")
    if value_max <= value_min:
        raise ValueError("value_max must be > value_min.")

    span = value_max - value_min
    num_thresholds = int(math.ceil(span / threshold_step))
    if num_thresholds < 2:
        raise ValueError("Need at least two thresholds for ordinal regression.")

    probs = keras.layers.Activation("sigmoid", name="ordinal_probs")(logits)
    threshold_count = keras.layers.Dense(
        1,
        use_bias=False,
        trainable=False,
        name="ordinal_threshold_count",
        kernel_initializer=keras.initializers.Constant([[1.0]] * num_thresholds),
    )(probs)
    value = keras.layers.Rescaling(
        threshold_step,
        offset=value_min,
        name="gauge_value",
    )(threshold_count)
    return value, probs


def _build_sweep_fraction_head(
    logits: keras.KerasTensor,
    *,
    value_min: float,
    value_max: float,
) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    """Convert a sweep-fraction logit into both fraction and calibrated value."""
    if value_max <= value_min:
        raise ValueError("value_max must be > value_min.")

    fraction = keras.layers.Activation("sigmoid", name="sweep_fraction")(logits)
    span = value_max - value_min
    value = keras.layers.Rescaling(
        span,
        offset=value_min,
        name="gauge_value",
    )(fraction)
    return value, fraction


def _build_keypoint_heatmap_head(
    features: keras.KerasTensor,
    *,
    heatmap_size: int,
    num_keypoints: int = 2,
) -> keras.KerasTensor:
    """Build a small decoder that predicts keypoint heatmaps."""
    if heatmap_size < 4:
        raise ValueError("heatmap_size must be >= 4.")
    if num_keypoints < 1:
        raise ValueError("num_keypoints must be >= 1.")

    x = keras.layers.Conv2D(
        256,
        3,
        padding="same",
        use_bias=False,
    )(features)
    x = _norm(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)

    x = keras.layers.Conv2D(
        128,
        3,
        padding="same",
        use_bias=False,
    )(x)
    x = _norm(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)

    x = keras.layers.Conv2D(
        64,
        3,
        padding="same",
        use_bias=False,
    )(x)
    x = _norm(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.Resizing(
        heatmap_size,
        heatmap_size,
        interpolation="bilinear",
        name="keypoint_heatmap_resize",
    )(x)
    heatmaps = keras.layers.Conv2D(
        num_keypoints,
        1,
        activation="sigmoid",
        name="keypoint_heatmaps",
    )(x)
    return heatmaps


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


def build_mobilenetv2_regression_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a transfer-learning regressor on top of MobileNetV2 features."""
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
    output = keras.layers.Dense(1, name="gauge_value")(x)

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


def build_mobilenetv2_tiny_regression_model(
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
    """Build a transfer-learning hybrid regressor with a coarse interval head."""
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
    gauge_value, interval_probs = _build_interval_hybrid_head(
        x,
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
