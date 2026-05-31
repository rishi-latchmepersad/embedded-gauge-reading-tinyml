"""
MobileNetV2-based coordinate regression model with transfer learning.
"""

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_mobilenetv2_coord_regression(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    alpha: float = 0.35,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.2,
) -> keras.Model:
    """
    Build MobileNetV2-based coordinate regression model.
    
    Architecture:
    - MobileNetV2 backbone (ImageNet pretrained optional)
    - Global average pooling
    - Dense layer with dropout
    - 4-unit sigmoid output (normalized coordinates)
    
    Args:
        input_shape: Input image shape
        alpha: MobileNetV2 width multiplier
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone during training
        dropout_rate: Dropout for regularization
    
    Returns:
        keras.Model for coordinate regression
    """
    inputs = keras.Input(shape=input_shape)
    
    # MobileNetV2 backbone
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet" if pretrained else None,
        pooling="avg",
    )
    backbone.trainable = not freeze_backbone
    
    x = backbone(inputs)
    
    # Dense head
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # 4 normalized coordinates [center_x, center_y, tip_x, tip_y]
    outputs = layers.Dense(4, activation="sigmoid")(x)
    
    model = keras.Model(inputs, outputs)
    return model


def build_two_phase_coord_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    alpha: float = 0.35,
) -> keras.Model:
    """
    Build coordinate regression model optimized for two-phase training.
    
    Phase 1: Train head with frozen backbone (pretrained on ImageNet)
    Phase 2: Unfreeze backbone and fine-tune with lower LR
    
    Returns:
        keras.Model with backbone marked for fine-tuning
    """
    inputs = keras.Input(shape=input_shape)
    
    # MobileNetV2 backbone with pretrained weights
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    backbone.trainable = False  # Start frozen
    
    x = backbone(inputs)
    
    # Head
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(4, activation="sigmoid")(x)
    
    model = keras.Model(inputs, outputs)
    return model
