"""Model builders for gauge-reading networks."""

from __future__ import annotations

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
) -> keras.Model:
    """Build a transfer-learning regressor on top of MobileNetV2 features."""
    inputs = keras.Input(shape=(image_height, image_width, 3), name="image")

    # Training pipeline emits [0,1] floats; MobileNetV2 preprocessing expects [0,255].
    x = keras.layers.Rescaling(255.0, name="to_255")(inputs)
    x = keras.layers.Lambda(
        keras.applications.mobilenet_v2.preprocess_input,
        name="mobilenetv2_preprocess",
    )(x)

    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(image_height, image_width, 3),
    )
    base_model.trainable = backbone_trainable

    x = base_model(x, training=backbone_trainable)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(128, activation="swish")(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(1, name="gauge_value")(x)

    model = keras.Model(inputs=inputs, outputs=output, name="mobilenetv2_gauge_regressor")
    # Store the backbone so training can run staged freeze/unfreeze schedules.
    setattr(model, "_mobilenet_backbone", base_model)
    return model
