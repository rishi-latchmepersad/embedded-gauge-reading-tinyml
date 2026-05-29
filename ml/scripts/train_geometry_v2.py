#!/usr/bin/env python3
"""
Geometric gauge reader v2 - fixes double-cropping and uses appropriate model size.

Key fixes from v1:
1. Preprocessed crops are already 224x224 - DON'T re-crop them, just resize
2. Board captures need luma crop to center the gauge
3. Use smaller model (alpha=0.35 MobileNetV2) to prevent overfitting
4. Convert coordinates properly for already-cropped vs raw images
5. Weighted sampling: oversample board captures in the dataset
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
    circular_angle_error_degrees,
)

from luma_crop_detector import (
    estimate_bright_centroid,
    compute_dynamic_crop,
    crop_and_resize,
    CropBox,
)


@dataclass
class TrainingConfig:
    backbone: str = "mobilenetv2"
    alpha: float = 0.35  # Width multiplier - 0.35 gives ~0.4M params
    image_size: int = 224
    input_channels: int = 3
    num_outputs: int = 4

    batch_size: int = 32
    epochs_phase1: int = 40
    epochs_phase2: int = 60
    initial_lr: float = 1e-3
    fine_tune_lr: float = 1e-5

    board_weight_multiplier: float = 10.0
    val_split: float = 0.15

    use_augmentation: bool = True
    aug_rotation_range: float = 10.0
    aug_zoom_range: float = 0.1
    aug_brightness_range: float = 0.15
    aug_contrast_range: float = 0.15

    dropout_rate: float = 0.3
    weight_decay: float = 1e-4

    output_dir: str = "/tmp/gauge_geometry_v2"
    experiment_name: str = "exp1"

    quantize_int8: bool = True


def load_metadata(manifest_path: Path) -> list[dict[str, Any]]:
    with open(manifest_path, "r") as f:
        return json.load(f)


def load_board_captures_manifest(csv_path: Path) -> list[dict[str, Any]]:
    import csv

    samples = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_width = float(row["source_width"])
            source_height = float(row["source_height"])
            samples.append({
                "image_path": str(row["image_path"]),
                "temperature_c": float(row["temperature_c"]),
                "center_x_norm": float(row["center_x"]) / source_width,
                "center_y_norm": float(row["center_y"]) / source_height,
                "tip_x_norm": float(row["tip_x"]) / source_width,
                "tip_y_norm": float(row["tip_y"]) / source_height,
                "angle_degrees": float(row["angle_degrees"]),
                "is_board": True,
            })
    return samples


def load_preprocessed_crop(image_path: Path, target_size: int = 224) -> Optional[np.ndarray]:
    """Load a preprocessed crop image (already 224x224 centered gauge).
    
    These images are already cropped to the gauge face. Just resize if needed.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        # Preprocessed crops should already be 224x224, but resize just in case
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)
    except Exception as e:
        print(f"  Error loading preprocessed {image_path}: {e}")
        return None


def load_board_capture_with_luma_crop(image_path: Path, target_size: int = 224) -> Optional[np.ndarray]:
    """Load a raw board capture and apply luma crop detection.
    
    Board captures are full-frame images from the STM32 camera.
    We need luma crop to center the gauge.
    """
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
            # Fallback: center crop
            min_dim = min(height, width)
            cx, cy = width // 2, height // 2
            crop_box = crop_and_resize(
                img_arr,
                CropBox(x_min=max(0, cx - min_dim // 2),
                        y_min=max(0, cy - min_dim // 2),
                        width=min_dim,
                        height=min_dim),
                target_size=target_size,
            )
            return crop_box.astype(np.uint8)

        crop = crop_and_resize(img_arr, crop_box, target_size=target_size)
        return np.asarray(crop, dtype=np.uint8)

    except Exception as e:
        print(f"  Error processing board capture {image_path}: {e}")
        return None


def build_dataset(
    samples: list[dict[str, Any]],
    image_size: int = 224,
    batch_size: int = 32,
    augment: bool = False,
    board_weight: float = 10.0,
    preprocessed_dir: Optional[Path] = None,
) -> tuple[tf.data.Dataset, int]:
    """Build dataset with proper handling of both preprocessed and board capture images."""
    images = []
    coords = []
    weights = []

    for sample in samples:
        image_path = Path(sample["image_path"])

        # Resolve path
        if not image_path.is_absolute():
            if preprocessed_dir is not None:
                # Try preprocessed_dir first
                alt_path = preprocessed_dir / image_path
                if alt_path.exists():
                    image_path = alt_path
                else:
                    alt_path2 = Path.cwd() / image_path
                    if alt_path2.exists():
                        image_path = alt_path2
                    else:
                        continue
            else:
                alt_path = Path.cwd() / image_path
                if alt_path.exists():
                    image_path = alt_path
                else:
                    continue

        if not image_path.exists():
            continue

        # Load image - different handling for preprocessed vs board
        is_board = sample.get("is_board", False)
        if is_board:
            crop = load_board_capture_with_luma_crop(image_path, target_size=image_size)
        else:
            crop = load_preprocessed_crop(image_path, target_size=image_size)

        if crop is None:
            continue

        # Normalize to [0, 1]
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

        # Weight: oversample board captures
        weight = board_weight if is_board else 1.0
        weights.append(weight)

    if not images:
        raise ValueError("No images loaded!")

    images_tensor = np.stack(images)
    coords_tensor = np.stack(coords)
    weights_tensor = np.array(weights, dtype=np.float32)

    num_samples = len(images)
    dataset = tf.data.Dataset.from_tensor_slices((images_tensor, coords_tensor, weights_tensor))

    if augment:
        dataset = dataset.shuffle(buffer_size=max(num_samples // 2, 100))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, num_samples


def build_augmentation_layer(config: TrainingConfig) -> keras.Sequential:
    return keras.Sequential([
        layers.RandomRotation(factor=config.aug_rotation_range / 360.0, fill_mode="reflect"),
        layers.RandomZoom(height_factor=config.aug_zoom_range, width_factor=config.aug_zoom_range, fill_mode="reflect"),
        layers.RandomContrast(factor=config.aug_contrast_range),
        layers.RandomBrightness(factor=config.aug_brightness_range),
    ], name="augmentation")


def build_geometry_model(config: TrainingConfig, use_imagenet: bool = False) -> tuple[keras.Model, keras.Model]:
    """Build smaller geometric model to prevent overfitting."""
    inputs = keras.Input(shape=(config.image_size, config.image_size, config.input_channels))

    if config.use_augmentation:
        x = build_augmentation_layer(config)(inputs)
    else:
        x = inputs

    # Use alpha=0.35 MobileNetV2 for ~0.4M params (prevents overfitting on 494 samples)
    weights = "imagenet" if use_imagenet else None

    if config.backbone == "mobilenetv2":
        backbone = keras.applications.MobileNetV2(
            input_shape=(config.image_size, config.image_size, config.input_channels),
            alpha=config.alpha,
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

    # Small head to prevent overfitting
    x = layers.Dropout(config.dropout_rate)(x)
    x = layers.Dense(64, activation="relu",
                     kernel_initializer="he_normal",
                     kernel_regularizer=keras.regularizers.l2(config.weight_decay))(x)
    x = layers.Dropout(config.dropout_rate / 2)(x)
    x = layers.Dense(32, activation="relu",
                     kernel_initializer="he_normal",
                     kernel_regularizer=keras.regularizers.l2(config.weight_decay))(x)

    # Sigmoid outputs constrained to [0, 1]
    outputs = layers.Dense(config.num_outputs, activation="sigmoid", name="coordinates")(x)

    model = keras.Model(inputs, outputs, name=f"{config.backbone}_geometry")
    return model, backbone


def coordinate_mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def coordinate_mae_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def angle_aware_loss(y_true, y_pred, image_size=224, coord_weight=1.0, angle_weight=0.5):
    """Combined coordinate MSE + angular error."""
    coord_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    cx_true, cy_true = y_true[:, 0] * image_size, y_true[:, 1] * image_size
    tx_true, ty_true = y_true[:, 2] * image_size, y_true[:, 3] * image_size
    cx_pred, cy_pred = y_pred[:, 0] * image_size, y_pred[:, 1] * image_size
    tx_pred, ty_pred = y_pred[:, 2] * image_size, y_pred[:, 3] * image_size

    dx_true = tx_true - cx_true
    dy_true = ty_true - cy_true
    angle_true = tf.atan2(dy_true, dx_true)

    dx_pred = tx_pred - cx_pred
    dy_pred = ty_pred - cy_pred
    angle_pred = tf.atan2(dy_pred, dx_pred)

    angle_diff = tf.abs(angle_true - angle_pred)
    angle_diff = tf.minimum(angle_diff, 2 * np.pi - angle_diff)
    angle_loss = tf.reduce_mean(angle_diff)

    return coord_weight * coord_loss + angle_weight * angle_loss


def train_model(train_dataset, val_dataset, config, train_size, val_size, use_imagenet=False):
    model, backbone = build_geometry_model(config, use_imagenet=use_imagenet)

    steps_per_epoch = max(1, train_size // config.batch_size)
    validation_steps = max(1, val_size // config.batch_size)

    # Phase 1: Train head only
    print("\n" + "=" * 60)
    print("PHASE 1: Training head (backbone frozen)")
    print("=" * 60)

    backbone.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.initial_lr),
        loss=coordinate_mse_loss,
        metrics=[coordinate_mae_loss],
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_coordinate_mae_loss", patience=10, restore_best_weights=True, mode="min")
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_coordinate_mae_loss", factor=0.5, patience=5, min_lr=1e-6, mode="min")

    history_phase1 = model.fit(
        train_dataset, validation_data=val_dataset,
        epochs=config.epochs_phase1,
        steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
        callbacks=[early_stop, reduce_lr], verbose=1)

    # Phase 2: Fine-tune
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning")
    print("=" * 60)

    backbone.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.fine_tune_lr),
        loss=angle_aware_loss,
        metrics=[coordinate_mae_loss],
    )

    early_stop_phase2 = keras.callbacks.EarlyStopping(
        monitor="val_coordinate_mae_loss", patience=15, restore_best_weights=True, mode="min")

    history_phase2 = model.fit(
        train_dataset, validation_data=val_dataset,
        epochs=config.epochs_phase2,
        steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
        callbacks=[early_stop_phase2, reduce_lr], verbose=1)

    # Save
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "best.keras")
    print(f"\nModel saved to {output_dir / 'best.keras'}")

    history = {
        "phase1": {k: [float(v) for v in hist] for k, hist in history_phase1.history.items()},
        "phase2": {k: [float(v) for v in hist] for k, hist in history_phase2.history.items()},
        "config": {
            "backbone": config.backbone, "alpha": config.alpha,
            "board_weight": config.board_weight_multiplier,
            "dropout": config.dropout_rate,
            "use_augmentation": config.use_augmentation,
            "use_imagenet": use_imagenet,
        },
    }
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model


def coords_to_temp(cx, cy, tx, ty, image_size=224):
    """Converts coordinates to temperature, handling edge cases."""
    # Clamp to valid range
    cx = max(0.01, min(0.99, cx))
    cy = max(0.01, min(0.99, cy))
    tx = max(0.01, min(0.99, tx))
    ty = max(0.01, min(0.99, ty))

    angle = angle_degrees_from_center_to_tip(
        cx * image_size, cy * image_size, tx * image_size, ty * image_size)
    temp = celsius_from_inner_dial_angle_degrees(angle)
    # Clamp to valid temp range
    return max(-30.0, min(50.0, temp))


def evaluate_geometry_model(model, board_samples, image_size=224):
    """Evaluate model on board captures."""
    temp_errors = []

    for sample in board_samples:
        image_path = Path(sample["image_path"])
        if not image_path.exists():
            continue

        crop = load_board_capture_with_luma_crop(image_path, target_size=image_size)
        if crop is None:
            continue

        img_float = crop.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_float, 0)

        coords = model.predict(img_batch, verbose=0)[0]
        cx, cy, tx, ty = coords[0], coords[1], coords[2], coords[3]

        pred_temp = coords_to_temp(cx, cy, tx, ty)
        gt_temp = sample["temperature_c"]
        temp_errors.append(abs(pred_temp - gt_temp))

    if not temp_errors:
        return {"temp_mae": float("inf"), "count": 0}

    temp_errors = np.array(temp_errors)
    return {
        "temp_mae": float(temp_errors.mean()),
        "temp_median": float(np.median(temp_errors)),
        "temp_std": float(temp_errors.std()),
        "temp_max": float(temp_errors.max()),
        "temp_min": float(temp_errors.min()),
        "count": len(temp_errors),
    }


def quantize_to_int8(model, config, representative_dataset):
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
    print(f"INT8 model: {output_dir / 'model_int8.tflite'} ({len(tflite_model) / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Geometric gauge reader v2")
    parser.add_argument("--backbone", type=str, default="mobilenetv2", choices=["mobilenetv2", "efficientnetb0"])
    parser.add_argument("--alpha", type=float, default=0.35, help="Width multiplier")
    parser.add_argument("--imagenet", action="store_true")
    parser.add_argument("--board-weight", type=float, default=10.0)
    parser.add_argument("--epochs-phase1", type=int, default=40)
    parser.add_argument("--epochs-phase2", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--experiment", type=str, default="exp1")
    parser.add_argument("--output-dir", type=str, default="/tmp/gauge_geometry_v2")
    parser.add_argument("--quantize", action="store_true")

    args = parser.parse_args()

    config = TrainingConfig(
        backbone=args.backbone,
        alpha=args.alpha,
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
    print("GEOMETRIC GAUGE READER V2")
    print("=" * 60)
    print(f"Backbone: {config.backbone} (alpha={config.alpha})")
    print(f"ImageNet: {args.imagenet}")
    print(f"Board weight: {config.board_weight_multiplier}x")
    print(f"Dropout: {config.dropout_rate}")
    print(f"Output: {Path(config.output_dir) / config.experiment_name}")

    # Load data
    print("\nLoading data...")
    metadata_path = Path(__file__).resolve().parent.parent / "data" / "preprocessed_crops" / "metadata.json"
    preprocessed_dir = Path(__file__).resolve().parent.parent / "data" / "preprocessed_crops"

    if metadata_path.exists():
        all_samples = load_metadata(metadata_path)
        print(f"  {len(all_samples)} preprocessed crops")
    else:
        all_samples = []

    board_csv_path = Path(__file__).resolve().parent.parent / "data" / "board_captures_labeled_v2.csv"
    if board_csv_path.exists():
        board_samples = load_board_captures_manifest(board_csv_path)
        all_samples.extend(board_samples)
        print(f"  {len(board_samples)} board captures")
    else:
        board_samples = []

    if not all_samples:
        print("ERROR: No data!")
        return

    eval_board_samples = board_samples.copy()

    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(all_samples))
    val_size = int(len(all_samples) * config.val_split)
    train_samples = [all_samples[i] for i in indices[val_size:]]
    val_samples = [all_samples[i] for i in indices[:val_size]]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Build datasets
    print("\nBuilding datasets...")
    train_dataset, train_sz = build_dataset(
        train_samples, config.image_size, config.batch_size,
        augment=config.use_augmentation, board_weight=config.board_weight_multiplier,
        preprocessed_dir=preprocessed_dir)
    val_dataset, val_sz = build_dataset(
        val_samples, config.image_size, config.batch_size,
        augment=False, board_weight=1.0,
        preprocessed_dir=preprocessed_dir)
    print(f"Dataset: {train_sz} train, {val_sz} val")

    # Train
    print("\nStarting training...")
    model = train_model(train_dataset, val_dataset, config, train_sz, val_sz, use_imagenet=args.imagenet)

    # Evaluate on board captures
    print("\nEvaluating on board captures...")
    if eval_board_samples:
        result = evaluate_geometry_model(model, eval_board_samples, config.image_size)
        print(f"  Board Temp MAE: {result['temp_mae']:.2f}°C")
        print(f"  Board Temp Median: {result['temp_median']:.2f}°C")

        output_dir = Path(config.output_dir) / config.experiment_name
        with open(output_dir / "board_eval.json", "w") as f:
            json.dump(result, f, indent=2)

        if result["temp_mae"] < 5.0:
            print("✓ TARGET ACHIEVED!")
        else:
            print(f"✗ Need <5°C, got {result['temp_mae']:.2f}°C")

    # Quantize
    if config.quantize_int8:
        print("\nQuantizing to INT8...")
        quantize_to_int8(model, config, val_dataset)


if __name__ == "__main__":
    main()
