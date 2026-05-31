#!/usr/bin/env python3
"""
Geometric gauge reader training with luma crop pipeline.

This script trains a model to predict gauge GEOMETRY (center + tip coordinates),
not temperature. The angle is computed from coordinates, then mapped to temperature
using a gauge spec JSON file. This makes the model gauge-agnostic.

Pipeline:
1. Luma bright-centroid crop detection (224x224)
2. Model predicts: [center_x, center_y, tip_x, tip_y] (normalized 0-1)
3. Compute angle: atan2(tip_y - center_y, tip_x - center_x)
4. Map angle → temperature using gauge spec (done in post-processing)

This approach:
- Works for ANY circular gauge with known min/max angles and temperature range
- Outputs are gauge-independent (normalized coordinates)
- Same model can be deployed to different gauges by changing gauge spec JSON
"""

from __future__ import annotations

import argparse
import json
import math
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
    circular_angle_error_degrees,
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
    num_outputs: int = 4  # center_x, center_y, tip_x, tip_y

    # Training hyperparameters
    batch_size: int = 32
    epochs_phase1: int = 30
    epochs_phase2: int = 50
    initial_lr: float = 1e-3
    fine_tune_lr: float = 1e-5

    # Data configuration
    board_weight_multiplier: float = 10.0
    val_split: float = 0.15

    # Augmentation
    use_augmentation: bool = True
    aug_rotation_range: float = 10.0  # Keep small - gauge orientation matters!
    aug_zoom_range: float = 0.15
    aug_brightness_range: float = 0.2
    aug_contrast_range: float = 0.2

    # Regularization
    dropout_rate: float = 0.3
    weight_decay: float = 1e-4

    # Output
    output_dir: str = "/tmp/gauge_geometry"
    experiment_name: str = "exp1"

    # Quantization
    quantize_int8: bool = True


# ============================================================================
# Data Loading
# ============================================================================


def load_metadata(manifest_path: Path) -> list[dict[str, Any]]:
    """Load training metadata from JSON manifest."""
    with open(manifest_path, "r") as f:
        return json.load(f)


def load_board_captures_manifest(csv_path: Path) -> list[dict[str, Any]]:
    """Load board captures from CSV manifest."""
    import csv

    samples = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize coordinates to [0, 1]
            source_width = float(row["source_width"])
            source_height = float(row["source_height"])
            
            samples.append(
                {
                    "image_path": str(row["image_path"]),
                    "temperature_c": float(row["temperature_c"]),
                    "center_x_norm": float(row["center_x"]) / source_width,
                    "center_y_norm": float(row["center_y"]) / source_height,
                    "tip_x_norm": float(row["tip_x"]) / source_width,
                    "tip_y_norm": float(row["tip_y"]) / source_height,
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

        centroid = estimate_bright_centroid(img_arr)
        height, width = img_arr.shape[:2]
        crop_box = compute_dynamic_crop(
            width=width,
            height=height,
            center_x=centroid.center_x,
            center_y=centroid.center_y,
        )

        if crop_box is None:
            return None

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
    preprocessed_dir: Optional[Path] = None,
) -> tuple[tf.data.Dataset, int]:
    """Build TensorFlow dataset from samples.
    
    Returns:
        Tuple of (dataset, num_samples)
    """
    # Pre-load all images
    images = []
    coords = []
    weights = []
    
    for sample in samples:
        image_path = Path(sample["image_path"])
        if not image_path.is_absolute():
            # Try CWD first, then preprocessed_dir
            cwd_path = Path.cwd() / image_path
            if cwd_path.exists():
                image_path = cwd_path
            elif preprocessed_dir is not None:
                image_path = preprocessed_dir / image_path
            else:
                continue

        if not image_path.exists():
            continue

        crop = apply_luma_crop(image_path, target_size=image_size)
        if crop is None:
            continue

        img_float = crop.astype(np.float32) / 255.0
        images.append(img_float)

        # Labels: normalized coordinates [cx, cy, tx, ty]
        coord = np.array([
            sample["center_x_norm"],
            sample["center_y_norm"],
            sample["tip_x_norm"],
            sample["tip_y_norm"],
        ], dtype=np.float32)
        coords.append(coord)

        # Sample weight
        weight = board_weight if sample.get("is_board", False) else 1.0
        weights.append(weight)

    if not images:
        raise ValueError("No images loaded!")

    # Convert to tensors
    images_tensor = np.stack(images)
    coords_tensor = np.stack(coords)
    weights_tensor = np.array(weights, dtype=np.float32)

    num_samples = len(images)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images_tensor, coords_tensor, weights_tensor))

    if augment:
        dataset = dataset.shuffle(buffer_size=max(num_samples // 2, 100))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, num_samples


def build_augmentation_layer(image_size: int, config: TrainingConfig) -> keras.Sequential:
    """Build Keras preprocessing augmentation layer."""
    return keras.Sequential(
        [
            # Small rotations only - gauge orientation is meaningful
            layers.RandomRotation(
                factor=config.aug_rotation_range / 360.0,
                fill_mode="reflect",
            ),
            layers.RandomZoom(
                height_factor=config.aug_zoom_range,
                width_factor=config.aug_zoom_range,
                fill_mode="reflect",
            ),
            layers.RandomContrast(factor=config.aug_contrast_range),
            layers.RandomBrightness(factor=config.aug_brightness_range),
        ],
        name="augmentation",
    )


# ============================================================================
# Model Building
# ============================================================================


def build_geometry_model(config: TrainingConfig, use_imagenet: bool = False) -> keras.Model:
    """Build geometric model for coordinate prediction."""

    inputs = keras.Input(shape=(config.image_size, config.image_size, config.input_channels))

    # Augmentation
    if config.use_augmentation:
        x = build_augmentation_layer(config.image_size, config)(inputs)
    else:
        x = inputs

    # Backbone
    weights = "imagenet" if use_imagenet else None
    
    if config.backbone == "mobilenetv2":
        backbone = keras.applications.MobileNetV2(
            input_shape=(config.image_size, config.image_size, config.input_channels),
            alpha=1.0,
            include_top=False,
            weights=weights,
            pooling="avg",
        )
    elif config.backbone == "efficientnetb0":
        backbone = keras.applications.EfficientNetB0(
            input_shape=(config.image_size, config.image_size, config.input_channels),
            include_top=False,
            weights=weights,
            pooling="avg",
        )
    else:
        raise ValueError(f"Unknown backbone: {config.backbone}")

    x = backbone(x)

    # Head for coordinate regression
    x = layers.Dropout(config.dropout_rate)(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(config.weight_decay),
    )(x)
    x = layers.Dropout(config.dropout_rate / 2)(x)
    x = layers.Dense(
        64,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(config.weight_decay),
    )(x)
    x = layers.Dropout(config.dropout_rate / 4)(x)

    # Output: 4 coordinates with sigmoid activation (constrained to [0, 1])
    outputs = layers.Dense(
        config.num_outputs,
        activation="sigmoid",  # Constrain to [0, 1]
        name="coordinates",
    )(x)

    model = keras.Model(inputs, outputs, name=f"{config.backbone}_geometry")

    return model, backbone


# ============================================================================
# Loss Functions
# ============================================================================


def coordinate_mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """MSE loss for coordinate regression."""
    return tf.reduce_mean(tf.square(y_true - y_pred))


def coordinate_mae_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """MAE loss for coordinate regression."""
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def angle_aware_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    image_size: int = 224,
    coord_weight: float = 1.0,
    angle_weight: float = 0.5,
) -> tf.Tensor:
    """
    Combined loss: coordinate MSE + angular error.
    
    This encourages the model to learn geometrically meaningful representations.
    """
    # Coordinate MSE
    coord_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Convert to angles
    cx_true, cy_true = y_true[:, 0] * image_size, y_true[:, 1] * image_size
    tx_true, ty_true = y_true[:, 2] * image_size, y_true[:, 3] * image_size
    
    cx_pred, cy_pred = y_pred[:, 0] * image_size, y_pred[:, 1] * image_size
    tx_pred, ty_pred = y_pred[:, 2] * image_size, y_pred[:, 3] * image_size

    # Compute angles
    dx_true = tx_true - cx_true
    dy_true = ty_true - cy_true
    angle_true = tf.atan2(dy_true, dx_true)

    dx_pred = tx_pred - cx_pred
    dy_pred = ty_pred - cy_pred
    angle_pred = tf.atan2(dy_pred, dx_pred)

    # Angular loss (circular)
    angle_diff = tf.abs(angle_true - angle_pred)
    angle_diff = tf.minimum(angle_diff, 2 * np.pi - angle_diff)  # Wrap around
    angle_loss = tf.reduce_mean(angle_diff)

    return coord_weight * coord_loss + angle_weight * angle_loss


# ============================================================================
# Training
# ============================================================================


def train_model(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    config: TrainingConfig,
    train_size: int,
    val_size: int,
    use_imagenet: bool = False,
) -> keras.Model:
    """Train model in two phases."""

    model, backbone = build_geometry_model(config, use_imagenet=use_imagenet)

    # Phase 1: Train head only
    print("\n" + "=" * 60)
    print("PHASE 1: Training head (backbone frozen)")
    print("=" * 60)

    backbone.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.initial_lr),
        loss=coordinate_mse_loss,
        metrics=[
            coordinate_mae_loss,
        ],
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_coordinate_mae_loss",
        patience=10,
        restore_best_weights=True,
        mode="min",
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_coordinate_mae_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        mode="min",
    )

    steps_per_epoch = max(1, train_size // config.batch_size)
    validation_steps = max(1, val_size // config.batch_size)

    history_phase1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs_phase1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    # Phase 2: Fine-tune entire model
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning entire model")
    print("=" * 60)

    backbone.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.fine_tune_lr),
        loss=coordinate_mse_loss,
        metrics=[coordinate_mae_loss],
    )

    early_stop_phase2 = keras.callbacks.EarlyStopping(
        monitor="val_coordinate_mae_loss",
        patience=15,
        restore_best_weights=True,
        mode="min",
    )

    history_phase2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs_phase2,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
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
# Evaluation
# ============================================================================


def coords_to_angle(cx: float, cy: float, tx: float, ty: float, image_size: int = 224) -> float:
    """Convert normalized coordinates to angle in degrees."""
    return angle_degrees_from_center_to_tip(
        cx * image_size, cy * image_size, tx * image_size, ty * image_size
    )


def angle_to_celsius(angle: float) -> float:
    """Convert angle to temperature using gauge spec."""
    return celsius_from_inner_dial_angle_degrees(angle)


def evaluate_geometry_model(
    model: keras.Model,
    board_samples: list[dict[str, Any]],
    image_size: int = 224,
) -> dict[str, float]:
    """Evaluate model on board captures using angle → temperature pipeline."""
    angle_errors = []
    temp_errors = []

    for sample in board_samples:
        image_path = Path(sample["image_path"])
        if not image_path.exists():
            continue

        crop = apply_luma_crop(image_path, target_size=image_size)
        if crop is None:
            continue

        img_float = crop.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_float, 0)

        # Predict coordinates
        coords = model.predict(img_batch, verbose=0)[0]
        cx, cy, tx, ty = coords[0], coords[1], coords[2], coords[3]

        # Convert to angle
        pred_angle = coords_to_angle(cx, cy, tx, ty, image_size)
        gt_angle = sample["angle_degrees"]

        # Angular error (circular)
        angle_error = circular_angle_error_degrees(pred_angle, gt_angle)
        angle_errors.append(angle_error)

        # Convert to temperature
        pred_temp = angle_to_celsius(pred_angle)
        gt_temp = sample["temperature_c"]
        temp_error = abs(pred_temp - gt_temp)
        temp_errors.append(temp_error)

    if not temp_errors:
        return {"angle_mae": float("inf"), "temp_mae": float("inf"), "count": 0}

    angle_errors = np.array(angle_errors)
    temp_errors = np.array(temp_errors)

    return {
        "angle_mae": float(angle_errors.mean()),
        "angle_median": float(np.median(angle_errors)),
        "temp_mae": float(temp_errors.mean()),
        "temp_median": float(np.median(temp_errors)),
        "temp_std": float(temp_errors.std()),
        "temp_max": float(temp_errors.max()),
        "count": len(temp_errors),
    }


# ============================================================================
# Quantization
# ============================================================================


def quantize_to_int8(model: keras.Model, config: TrainingConfig, representative_dataset: tf.data.Dataset):
    """Quantize model to INT8."""

    def representative_data_gen():
        for images, _, _ in representative_dataset.take(100):
            yield [images]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "model_int8.tflite", "wb") as f:
        f.write(tflite_model)

    print(f"INT8 model saved to {output_dir / 'model_int8.tflite'}")
    print(f"Model size: {len(tflite_model) / 1024:.1f} KB")

    return tflite_model


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train geometric gauge reader")
    parser.add_argument("--backbone", type=str, default="mobilenetv2", choices=["mobilenetv2", "efficientnetb0"])
    parser.add_argument("--imagenet", action="store_true", help="Use ImageNet pretrained weights")
    parser.add_argument("--board-weight", type=float, default=10.0)
    parser.add_argument("--epochs-phase1", type=int, default=30)
    parser.add_argument("--epochs-phase2", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--experiment", type=str, default="exp1")
    parser.add_argument("--output-dir", type=str, default="/tmp/gauge_geometry")
    parser.add_argument("--quantize", action="store_true")

    args = parser.parse_args()

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
    print("GEOMETRIC GAUGE READER TRAINING")
    print("=" * 60)
    print(f"Backbone: {config.backbone}")
    print(f"ImageNet init: {args.imagenet}")
    print(f"Board weight: {config.board_weight_multiplier}x")
    print(f"Output: {Path(config.output_dir) / config.experiment_name}")

    # Load data
    print("\nLoading data...")

    metadata_path = Path(__file__).resolve().parent.parent / "data" / "preprocessed_crops" / "metadata.json"
    if metadata_path.exists():
        all_samples = load_metadata(metadata_path)
        print(f"  Loaded {len(all_samples)} samples from metadata.json")
    else:
        all_samples = []
        print("  metadata.json not found")

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

    # Store board samples for evaluation
    eval_board_samples = board_samples.copy()

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
    preprocessed_dir = Path(__file__).resolve().parent.parent / "data" / "preprocessed_crops"
    
    train_dataset, train_size = build_dataset(
        train_samples,
        image_size=config.image_size,
        batch_size=config.batch_size,
        augment=config.use_augmentation,
        board_weight=config.board_weight_multiplier,
        preprocessed_dir=preprocessed_dir,
    )

    val_dataset, val_size = build_dataset(
        val_samples,
        image_size=config.image_size,
        batch_size=config.batch_size,
        augment=False,
        board_weight=1.0,
        preprocessed_dir=preprocessed_dir,
    )

    print(f"Train dataset: {train_size} samples, Val dataset: {val_size} samples")

    # Train
    print("\nStarting training...")
    model = train_model(train_dataset, val_dataset, config, train_size, val_size, use_imagenet=args.imagenet)

    # Evaluate on board captures
    print("\nEvaluating on board captures...")
    if eval_board_samples:
        board_eval = evaluate_geometry_model(model, eval_board_samples, image_size=config.image_size)
        print(f"\nBoard Angle MAE: {board_eval['angle_mae']:.2f}°")
        print(f"Board Temp MAE: {board_eval['temp_mae']:.2f}°C")
        print(f"Board Temp Median: {board_eval['temp_median']:.2f}°C")

        output_dir = Path(config.output_dir) / config.experiment_name
        with open(output_dir / "board_eval.json", "w") as f:
            json.dump(board_eval, f, indent=2)
    else:
        print("No board samples for evaluation")
        board_eval = None

    # Quantize
    if config.quantize_int8:
        print("\nQuantizing to INT8...")
        quantize_to_int8(model, config, val_dataset)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    if board_eval:
        print(f"Board Temp MAE: {board_eval['temp_mae']:.2f}°C")
        print(f"Target: <5.0°C")
        if board_eval["temp_mae"] < 5.0:
            print("✓ TARGET ACHIEVED!")
        else:
            print("✗ Target not achieved")


if __name__ == "__main__":
    main()
