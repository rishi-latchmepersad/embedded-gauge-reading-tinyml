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

from typing import Tuple, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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
        weights='imagenet',
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
):
    """Build a compact MobileNetV2 heatmap decoder with a configurable output size."""

    if heatmap_size not in (56, 112):
        raise ValueError(f"Unsupported heatmap_size={heatmap_size}; expected 56 or 112.")

    inputs = keras.Input(shape=input_shape, name="input_image")

    # Keep the backbone shallow and reusable so we can transfer weights across v2/v3/v4.
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet",
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

    center_heatmap = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="center_heatmap")(x)
    tip_heatmap = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="tip_heatmap")(x)

    confidence_features = layers.GlobalAveragePooling2D(name="geometry_confidence_gap")(x)
    confidence = layers.Dense(1, activation="sigmoid", name="confidence")(confidence_features)

    return keras.Model(inputs=inputs, outputs=[center_heatmap, tip_heatmap, confidence], name=model_name)


def build_mobilenetv2_geometry_heatmap_v4_112(
    input_shape=(224, 224, 3),
    alpha=0.35,
    backbone_frozen=True,
):
    """Build the 112x112 geometry heatmap model for tip-stable INT8 deployment.

    The 112x112 head keeps the v3-style decoder blocks so we can still transfer
    compatible weights from the canonical v3 checkpoint, but it adds a shallow
    56x56 skip before the final upsampling stage. That gives the decoder a
    cleaner spatial signal than a plain bilinear 56->112 wrapper while keeping
    the model compact enough for embedded deployment.
    """

    inputs = keras.Input(shape=input_shape, name="input_image")

    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    backbone.trainable = not backbone_frozen

    # Build a small feature extractor so the skip and bottleneck tensors are
    # connected to the same functional graph as the external model input.
    feature_extractor = keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer("block_3_expand_relu").output, backbone.output],
        name="mobilenetv2_geometry_heatmap_v4_112_backbone",
    )

    # Pull a shallow skip from the 56x56 backbone stage so the final 112x112
    # heatmaps can preserve a bit more localization detail than a pure upsample.
    skip_56, x = feature_extractor(inputs)

    x = layers.Conv2D(
        128,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="geometry_decoder_conv_1",
    )(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="geometry_decoder_up_1")(x)

    x = layers.Conv2D(
        64,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="geometry_decoder_conv_2",
    )(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="geometry_decoder_up_2")(x)

    x = layers.Conv2D(
        32,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="geometry_decoder_conv_3",
    )(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="geometry_decoder_up_3")(x)

    skip_56 = layers.Conv2D(
        16,
        1,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="geometry_decoder_skip_56",
    )(skip_56)
    x = layers.Concatenate(name="geometry_decoder_concat_56")([x, skip_56])
    x = layers.Conv2D(
        32,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="geometry_decoder_refine_112",
    )(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="geometry_decoder_up_4")(x)

    center_heatmap = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="center_heatmap")(x)
    tip_heatmap = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="tip_heatmap")(x)

    confidence_features = layers.GlobalAveragePooling2D(name="geometry_confidence_gap")(x)
    confidence = layers.Dense(1, activation="sigmoid", name="confidence")(confidence_features)

    return keras.Model(
        inputs=inputs,
        outputs=[center_heatmap, tip_heatmap, confidence],
        name="mobilenetv2_geometry_heatmap_v4_112",
    )
