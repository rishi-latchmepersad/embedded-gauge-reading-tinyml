#!/usr/bin/env python3
"""
Transfer learning training for gauge temperature regression with luma crop pipeline.

This script trains a single model to achieve <5°C MAE on board captures using:
- Luma-based bright-centroid crop detection (matches firmware pipeline)
- Transfer learning with MobileNetV2 or EfficientNet backbones
- Mixed training data: labelled phone photos + board captures
- Heavy augmentation for domain adaptation
- Board capture oversampling to reduce domain shift

Output: TFLite INT8 model ready for STM32 deployment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
)

# Import luma crop detector
from luma_crop_detector import (
    estimate_bright_centroid,
    compute_dynamic_crop,
    crop_and_resize,
    BrightCentroidResult,
    CropBox,
)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model architecture
    backbone: str = "mobilenetv2"  # mobilenetv2, efficientnetb0
    image_size: int = 224
    input_channels: int = 3

    # Training hyperparameters
    batch_size: int = 32
    epochs_phase1: int = 30  # Head training (backbone frozen)
    epochs_phase2: int = 50  # Full fine-tuning
    initial_lr: float = 1e-3
    fine_tune_lr: float = 1e-5

    # Data configuration
    board_weight_multiplier: float = 10.0  # Oversample board captures
    val_split: float = 0.15

    # Augmentation
    use_augmentation: bool = True
    aug_rotation_range: float = 15.0
    aug_brightness_range: tuple[float, float] = (0.7, 1.3)
    aug_contrast_range: tuple[float, float] = (0.7, 1.3)
    aug_zoom_range: float = 0.15

    # Regularization
    dropout_rate: float = 0.3
    weight_decay: float = 1e-4

    # Output
    output_dir: str = "/tmp/gauge_transfer_learning"
    experiment_name: str = "exp1"

    # Quantization
    quantize_int8: bool = True


# ============================================================================
# Data Loading
# ============================================================================


def load_metadata(manifest_path: Path) -> list[dict[str, Any]]:
    """Load training metadata from JSON manifest."""
    import json

    with open(manifest_path, "r") as f:
        return json.load(f)


def load_board_captures_manifest(csv_path: Path) -> list[dict[str, Any]]:
    """Load board captures from CSV manifest."""
    import csv

    samples = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                {
                    "image_path": str(row["image_path"]),
                    "temperature_c": float(row["temperature_c"]),
                    "center_x_norm": float(row["center_x"]) / float(row["source_width"]),
                    "center_y_norm": float(row["center_y"]) / float(row["source_height"]),
                    "tip_x_norm": float(row["tip_x"]) / float(row["source_width"]),
                    "tip_y_norm": float(row["tip_y"]) / float(row["source_height"]),
                    "angle_degrees": float(row["angle_degrees"]),
                    "is_board": True,
                }
            )
    return samples


def apply_luma_crop(image_path: Path, target_size: int = 224) -> Optional[np.ndarray]:
    """Load image and apply luma-based crop detection."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_arr = np.asarray(img, dtype=np.uint8)

        # Luma centroid detection
        centroid = estimate_bright_centroid(img_arr)

        # Compute crop box
        height, width = img_arr.shape[:2]
        crop_box = compute_dynamic_crop(
            width=width,
            height=height,
            center_x=centroid.center_x,
            center_y=centroid.center_y,
        )

        if crop_box is None:
            return None

        # Crop and resize
        crop = crop_and_resize(img_arr, crop_box, target_size=target_size)
        return crop

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def build_dataset(
    samples: list[dict[str, Any]],
    image_size: int = 224,
    batch_size: int = 32,
    augment: bool = False,
    board_weight: float = 10.0,
) -> tf.data.Dataset:
    """Build TensorFlow dataset from samples."""

    def generator():
        for sample in samples:
            image_path = Path(sample["image_path"])
            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path

            if not image_path.exists():
                continue

            # Apply luma crop
            crop = apply_luma_crop(image_path, target_size=image_size)
            if crop is None:
                continue

            # Normalize to [0, 1]
            img_float = crop.astype(np.float32) / 255.0

            # Get temperature label
            temp = sample.get("temperature_c")
            if temp is None:
                # Compute from coordinates
                cx = sample.get("center_x_norm", 0.5)
                cy = sample.get("center_y_norm", 0.5)
                tx = sample.get("tip_x_norm", 0.5)
                ty = sample.get("tip_y_norm", 0.5)
                angle = angle_degrees_from_center_to_tip(
                    cx * image_size, cy * image_size, tx * image_size, ty * image_size
                )
                temp = celsius_from_inner_dial_angle_degrees(angle)

            # Sample weight (oversample board captures)
            weight = board_weight if sample.get("is_board", False) else 1.0

            yield img_float, temp, weight

    # Create dataset
    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    # Shuffle and batch
    if augment:
        dataset = dataset.shuffle(buffer_size=len(samples) // 2)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_augmentation_layer(image_size: int, config: TrainingConfig) -> keras.Sequential:
    """Build Keras preprocessing augmentation layer."""
    return keras.Sequential(
        [
            layers.RandomRotation(
                factor=config.aug_rotation_range / 360.0,
                fill_mode="reflect",
            ),
            layers.RandomZoom(
                height_factor=config.aug_zoom_range,
                width_factor=config.aug_zoom_range,
                fill_mode="reflect",
            ),
            layers.RandomContrast(factor=0.3),
            layers.RandomBrightness(factor=0.3),
        ],
        name="augmentation",
    )


# ============================================================================
# Model Building
# ============================================================================


def build_transfer_learning_model(config: TrainingConfig) -> keras.Model:
    """Build transfer learning model for temperature regression."""

    # Input
    inputs = keras.Input(shape=(config.image_size, config.image_size, config.input_channels))

    # Augmentation (only during training)
    if config.use_augmentation:
        x = build_augmentation_layer(config.image_size, config)(inputs)
    else:
        x = inputs

    # Backbone
    if config.backbone == "mobilenetv2":
        backbone = keras.applications.MobileNetV2(
            input_shape=(config.image_size, config.image_size, config.input_channels),
            alpha=1.0,
            include_top=False,
            weights=None,  # Start from scratch
            pooling="avg",
        )
    elif config.backbone == "efficientnetb0":
        backbone = keras.applications.EfficientNetB0(
            input_shape=(config.image_size, config.image_size, config.input_channels),
            include_top=False,
            weights=None,  # Start from scratch
            pooling="avg",
        )
    else:
        raise ValueError(f"Unknown backbone: {config.backbone}")

    x = backbone(x)

    # Head
    x = layers.Dropout(config.dropout_rate)(x)
    x = layers.Dense(
        64,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(config.weight_decay),
    )(x)
    x = layers.Dropout(config.dropout_rate / 2)(x)

    # Output: single temperature value
    outputs = layers.Dense(1, activation="linear", name="temperature")(x)

    model = keras.Model(inputs, outputs, name=f"{config.backbone}_gauge_reader")

    return model, backbone


def build_imagenet_model(config: TrainingConfig) -> keras.Model:
    """Build model with ImageNet pretrained weights."""

    # Input
    inputs = keras.Input(shape=(config.image_size, config.image_size, config.input_channels))

    # Augmentation (only during training)
    if config.use_augmentation:
        x = build_augmentation_layer(config.image_size, config)(inputs)
    else:
        x = inputs

    # Backbone with ImageNet weights
    if config.backbone == "mobilenetv2":
        backbone = keras.applications.MobileNetV2(
            input_shape=(config.image_size, config.image_size, config.input_channels),
            alpha=1.0,
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
    elif config.backbone == "efficientnetb0":
        backbone = keras.applications.EfficientNetB0(
            input_shape=(config.image_size, config.image_size, config.input_channels),
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
    else:
        raise ValueError(f"Unknown backbone: {config.backbone}")

    x = backbone(x)

    # Head
    x = layers.Dropout(config.dropout_rate)(x)
    x = layers.Dense(
        64,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(config.weight_decay),
    )(x)
    x = layers.Dropout(config.dropout_rate / 2)(x)

    # Output
    outputs = layers.Dense(1, activation="linear", name="temperature")(x)

    model = keras.Model(inputs, outputs, name=f"{config.backbone}_imagenet_gauge")

    return model, backbone


# ============================================================================
# Training
# ============================================================================


def train_model(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    config: TrainingConfig,
    use_imagenet: bool = False,
) -> keras.Model:
    """Train model in two phases."""

    # Build model
    if use_imagenet:
        model, backbone = build_imagenet_model(config)
    else:
        model, backbone = build_transfer_learning_model(config)

    # Phase 1: Train head only (backbone frozen)
    print("\n" + "=" * 60)
    print("PHASE 1: Training head (backbone frozen)")
    print("=" * 60)

    backbone.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.initial_lr),
        loss=keras.losses.Huber(delta=5.0),  # Robust to outliers
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    # Custom weighted loss
    def weighted_mae(y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true, axis=-1)
        error = tf.abs(y_true - y_pred)
        if sample_weight is not None:
            error = error * sample_weight
        return tf.reduce_mean(error)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_mae",
        patience=10,
        restore_best_weights=True,
        mode="min",
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_mae",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        mode="min",
    )

    history_phase1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs_phase1,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    # Phase 2: Fine-tune entire model
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning entire model")
    print("=" * 60)

    backbone.trainable = True

    # Unfreeze gradually (optional - for now unfreeze all)
    for layer in backbone.layers:
        layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.fine_tune_lr),
        loss=keras.losses.Huber(delta=5.0),
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    early_stop_phase2 = keras.callbacks.EarlyStopping(
        monitor="val_mae",
        patience=15,
        restore_best_weights=True,
        mode="min",
    )

    history_phase2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs_phase2,
        callbacks=[early_stop_phase2, reduce_lr],
        verbose=1,
    )

    # Save model
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(output_dir / "best.keras")
    print(f"\nModel saved to {output_dir / 'best.keras'}")

    # Save history
    history = {
        "phase1": {k: [float(v) for v in hist] for k, hist in history_phase1.history.items()},
        "phase2": {k: [float(v) for v in hist] for k, hist in history_phase2.history.items()},
        "config": {
            "backbone": config.backbone,
            "image_size": config.image_size,
            "batch_size": config.batch_size,
            "epochs_phase1": config.epochs_phase1,
            "epochs_phase2": config.epochs_phase2,
            "initial_lr": config.initial_lr,
            "fine_tune_lr": config.fine_tune_lr,
            "board_weight": config.board_weight_multiplier,
            "dropout": config.dropout_rate,
            "use_augmentation": config.use_augmentation,
            "use_imagenet": use_imagenet,
        },
    }

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model


# ============================================================================
# Quantization
# ============================================================================


def quantize_to_int8(model: keras.Model, config: TrainingConfig, representative_dataset: tf.data.Dataset):
    """Quantize model to INT8 using full integer quantization."""

    def representative_data_gen():
        for images, _, _ in representative_dataset.take(100):
            yield [images]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.int8
    converter.inference_output_type = tf.float32  # Keep output as float for temperature

    tflite_model = converter.convert()

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "model_int8.tflite", "wb") as f:
        f.write(tflite_model)

    print(f"INT8 model saved to {output_dir / 'model_int8.tflite'}")
    print(f"Model size: {len(tflite_model) / 1024:.1f} KB")

    return tflite_model


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_on_board_captures(
    model: keras.Model,
    board_samples: list[dict[str, Any]],
    image_size: int = 224,
) -> dict[str, float]:
    """Evaluate model on board captures."""
    errors = []

    for sample in board_samples:
        image_path = Path(sample["image_path"])
        if not image_path.exists():
            continue

        crop = apply_luma_crop(image_path, target_size=image_size)
        if crop is None:
            continue

        img_float = crop.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_float, 0)

        pred_temp = float(model.predict(img_batch, verbose=0)[0, 0])
        gt_temp = sample["temperature_c"]

        error = abs(pred_temp - gt_temp)
        errors.append(error)

    if not errors:
        return {"mae": float("inf"), "count": 0}

    errors = np.array(errors)
    return {
        "mae": float(errors.mean()),
        "median": float(np.median(errors)),
        "std": float(errors.std()),
        "max": float(errors.max()),
        "min": float(errors.min()),
        "count": len(errors),
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train gauge reader with transfer learning")
    parser.add_argument("--backbone", type=str, default="mobilenetv2", choices=["mobilenetv2", "efficientnetb0"])
    parser.add_argument("--imagenet", action="store_true", help="Use ImageNet pretrained weights")
    parser.add_argument("--board-weight", type=float, default=10.0, help="Board capture weight multiplier")
    parser.add_argument("--epochs-phase1", type=int, default=30)
    parser.add_argument("--epochs-phase2", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--experiment", type=str, default="exp1")
    parser.add_argument("--output-dir", type=str, default="/tmp/gauge_transfer_learning")
    parser.add_argument("--quantize", action="store_true", help="Quantize to INT8 after training")

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        backbone=args.backbone,
        board_weight_multiplier=args.board_weight,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        batch_size=args.batch_size,
        dropout_rate=args.dropout,
        use_augmentation=not args.no_augment,
        experiment_name=args.experiment,
        output_dir=args.output_dir,
        quantize_int8=args.quantize,
    )

    print("=" * 60)
    print("GAUGE TRANSFER LEARNING TRAINING")
    print("=" * 60)
    print(f"Backbone: {config.backbone}")
    print(f"ImageNet init: {args.imagenet}")
    print(f"Board weight: {config.board_weight_multiplier}x")
    print(f"Augmentation: {config.use_augmentation}")
    print(f"Output: {Path(config.output_dir) / config.experiment_name}")

    # Load data
    print("\nLoading data...")

    # Load preprocessed crops metadata
    metadata_path = Path(__file__).resolve().parent.parent / "data" / "preprocessed_crops" / "metadata.json"
    if metadata_path.exists():
        all_samples = load_metadata(metadata_path)
        print(f"  Loaded {len(all_samples)} samples from metadata.json")
    else:
        all_samples = []
        print("  metadata.json not found")

    # Load board captures
    board_csv_path = Path(__file__).resolve().parent.parent / "data" / "board_captures_labeled_v2.csv"
    if board_csv_path.exists():
        board_samples = load_board_captures_manifest(board_csv_path)
        all_samples.extend(board_samples)
        print(f"  Loaded {len(board_samples)} board captures")
    else:
        board_samples = []
        print("  board_captures_labeled_v2.csv not found")

    if not all_samples:
        print("ERROR: No data loaded!")
        return

    # Split train/val
    np.random.seed(42)
    indices = np.random.permutation(len(all_samples))
    val_size = int(len(all_samples) * config.val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]

    print(f"\nTrain: {len(train_samples)}, Val: {len(val_samples)}")

    # Build datasets
    print("\nBuilding datasets...")
    train_dataset = build_dataset(
        train_samples,
        image_size=config.image_size,
        batch_size=config.batch_size,
        augment=config.use_augmentation,
        board_weight=config.board_weight_multiplier,
    )

    val_dataset = build_dataset(
        val_samples,
        image_size=config.image_size,
        batch_size=config.batch_size,
        augment=False,
        board_weight=1.0,
    )

    # Train
    print("\nStarting training...")
    model = train_model(train_dataset, val_dataset, config, use_imagenet=args.imagenet)

    # Evaluate on board captures
    print("\nEvaluating on board captures...")
    board_eval = evaluate_on_board_captures(model, board_samples, image_size=config.image_size)
    print(f"Board MAE: {board_eval['mae']:.2f}°C")
    print(f"Board Median: {board_eval['median']:.2f}°C")

    # Save board eval results
    output_dir = Path(config.output_dir) / config.experiment_name
    with open(output_dir / "board_eval.json", "w") as f:
        json.dump(board_eval, f, indent=2)

    # Quantize
    if config.quantize_int8:
        print("\nQuantizing to INT8...")
        quantize_to_int8(model, config, val_dataset)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Board MAE: {board_eval['mae']:.2f}°C")
    print(f"Target: <5.0°C")
    if board_eval["mae"] < 5.0:
        print("✓ TARGET ACHIEVED!")
    else:
        print("✗ Target not achieved - try different hyperparameters")


if __name__ == "__main__":
    main()
