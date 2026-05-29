#!/usr/bin/env python3
"""
Geometric gauge reader v3 - uses proven simple CNN architecture.

Uses the compact CNN feature backbone from models.py that achieved
val MAE 0.048 in initial experiments. No BatchNorm to avoid domain shift issues.

Key differences from v2:
- Simple CNN backbone (~65k params, no BN)
- Train from scratch directly (no frozen phase - the simple CNN is small enough)
- Proper handling of preprocessed crops vs board captures
- Coordinate outputs [cx, cy, tx, ty] with sigmoid constraint
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
    image_size: int = 224
    input_channels: int = 3
    num_outputs: int = 4

    batch_size: int = 32
    epochs: int = 100
    initial_lr: float = 1e-3
    min_lr: float = 1e-6

    board_weight_multiplier: float = 10.0
    val_split: float = 0.15

    use_augmentation: bool = True
    augment_rotation: float = 10.0
    augment_zoom: float = 0.1
    augment_brightness: float = 0.15
    augment_contrast: float = 0.15

    dropout_rate: float = 0.2

    output_dir: str = "/tmp/gauge_geometry_v3"
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
            sw = float(row["source_width"])
            sh = float(row["source_height"])
            samples.append({
                "image_path": str(row["image_path"]),
                "temperature_c": float(row["temperature_c"]),
                "center_x_norm": float(row["center_x"]) / sw,
                "center_y_norm": float(row["center_y"]) / sh,
                "tip_x_norm": float(row["tip_x"]) / sw,
                "tip_y_norm": float(row["tip_y"]) / sh,
                "angle_degrees": float(row["angle_degrees"]),
                "is_board": True,
            })
    return samples


def _conv_norm_swish(x, filters, strides=1, kernel_size=3):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=-1, epsilon=1e-6)(x)  # GroupNorm instead of BatchNorm
    x = layers.Activation("swish")(x)
    return x


def _residual_separable_block(x, filters, dropout_rate=0.0):
    shortcut = x
    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.Activation("swish")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=-1)(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, use_bias=False)(shortcut)
        shortcut = layers.GroupNormalization(groups=-1)(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("swish")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    return x


def build_simple_cnn(config: TrainingConfig) -> keras.Model:
    """Build compact CNN for coordinate regression.
    
    Architecture: ~65k params, no BatchNorm (uses GroupNorm instead).
    Matches the successful initial experiments.
    """
    inputs = keras.Input(shape=(config.image_size, config.image_size, config.input_channels))

    if config.use_augmentation:
        x = keras.Sequential([
            layers.RandomRotation(config.augment_rotation / 360.0, fill_mode="reflect"),
            layers.RandomZoom(config.augment_zoom, config.augment_zoom, fill_mode="reflect"),
            layers.RandomContrast(config.augment_contrast),
            layers.RandomBrightness(config.augment_brightness),
        ])(inputs)
    else:
        x = inputs

    # Feature backbone (same as _build_feature_backbone)
    x = _conv_norm_swish(x, 32, strides=2)
    x = _residual_separable_block(x, 32, dropout_rate=0.02)
    x = layers.MaxPool2D(pool_size=2)(x)

    x = _residual_separable_block(x, 64, dropout_rate=0.04)
    x = layers.MaxPool2D(pool_size=2)(x)

    x = _residual_separable_block(x, 96, dropout_rate=0.06)
    x = layers.MaxPool2D(pool_size=2)(x)

    x = _residual_separable_block(x, 128, dropout_rate=0.08)
    x = layers.GlobalAveragePooling2D()(x)

    # Regression head
    x = layers.Dense(192, activation="swish")(x)
    x = layers.Dropout(config.dropout_rate)(x)

    # Output: 4 coordinates with sigmoid [0, 1] constraint
    outputs = layers.Dense(config.num_outputs, activation="sigmoid", name="coordinates")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="simple_cnn_geometry")
    return model


def load_both_types(image_path: Path, is_board: bool, target_size: int = 224,
                    preprocessed_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    """Load image with proper handling for preprocessed crops vs board captures."""
    # Resolve path
    if not image_path.exists():
        if is_board:
            return None
        # Try preprocessed_dir
        if preprocessed_dir:
            alt = preprocessed_dir / image_path
            if alt.exists():
                image_path = alt
            else:
                return None
        else:
            return None

    try:
        img = Image.open(image_path).convert("RGB")
        img_arr = np.asarray(img, dtype=np.uint8)

        if is_board:
            # Apply luma crop to center the gauge
            centroid = estimate_bright_centroid(img_arr)
            h, w = img_arr.shape[:2]
            crop_box = compute_dynamic_crop(width=w, height=h,
                                            center_x=centroid.center_x,
                                            center_y=centroid.center_y)
            if crop_box is None:
                return None
            crop = crop_and_resize(img_arr, crop_box, target_size=target_size)
            return np.asarray(crop, dtype=np.uint8)
        else:
            # Preprocessed crops - just resize if needed
            if img.size != (target_size, target_size):
                img = img.resize((target_size, target_size), Image.BILINEAR)
            return np.asarray(img, dtype=np.uint8)

    except Exception as e:
        print(f"  Error loading {image_path}: {e}")
        return None


def build_dataset(samples, config: TrainingConfig, augment: bool,
                  preprocessed_dir: Optional[Path] = None) -> tuple[tf.data.Dataset, int]:
    images, coords = [], []

    for sample in samples:
        path = Path(sample["image_path"])
        if not path.is_absolute() and preprocessed_dir:
            full_path = preprocessed_dir / path
        else:
            full_path = path

        crop = load_both_types(full_path, sample.get("is_board", False),
                               config.image_size, preprocessed_dir)
        if crop is None:
            continue

        images.append(crop.astype(np.float32) / 255.0)
        coords.append(np.array([
            sample["center_x_norm"],
            sample["center_y_norm"],
            sample["tip_x_norm"],
            sample["tip_y_norm"],
        ], dtype=np.float32))

    if not images:
        raise ValueError("No images loaded!")

    images_ten = np.stack(images)
    coords_ten = np.stack(coords)
    n = len(images)

    ds = tf.data.Dataset.from_tensor_slices((images_ten, coords_ten))
    if augment:
        ds = ds.shuffle(max(n // 2, 100))
    ds = ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, n


def coords_to_temp_fn(cx, cy, tx, ty):
    cx = max(0.01, min(0.99, cx))
    cy = max(0.01, min(0.99, cy))
    tx = max(0.01, min(0.99, tx))
    ty = max(0.01, min(0.99, ty))
    angle = angle_degrees_from_center_to_tip(cx * 224, cy * 224, tx * 224, ty * 224)
    temp = celsius_from_inner_dial_angle_degrees(angle)
    return max(-30.0, min(50.0, temp))


def evaluate_board(model, board_samples, config):
    temp_errors = []
    repo_root = Path(__file__).resolve().parent.parent  # ml/
    
    for sample in board_samples:
        path = Path(sample["image_path"])
        if not path.exists():
            # Try repo root
            alt = repo_root / path
            if alt.exists():
                path = alt
            else:
                continue
        
        crop = load_both_types(path, True, config.image_size)
        if crop is None:
            continue
        img_f = crop.astype(np.float32) / 255.0
        batch = np.expand_dims(img_f, 0)
        c = model.predict(batch, verbose=0)[0]
        pt = coords_to_temp_fn(c[0], c[1], c[2], c[3])
        temp_errors.append(abs(pt - sample["temperature_c"]))

    if not temp_errors:
        return {"temp_mae": float("inf"), "count": 0}
    err = np.array(temp_errors)
    return {
        "temp_mae": float(err.mean()),
        "temp_median": float(np.median(err)),
        "temp_std": float(err.std()),
        "temp_max": float(err.max()),
        "temp_min": float(err.min()),
        "count": len(err),
    }


def train_model(train_ds, val_ds, config: TrainingConfig):
    model = build_simple_cnn(config)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.initial_lr),
        loss="mse",
        metrics=["mae"],
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=20, restore_best_weights=True, mode="min")
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_mae", factor=0.5, patience=8, min_lr=config.min_lr, mode="min")
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        str(Path(config.output_dir) / config.experiment_name / "checkpoint.keras"),
        monitor="val_mae", save_best_only=True, mode="min")

    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.epochs,
        callbacks=[early_stop, reduce_lr, model_checkpoint],
        verbose=1,
    )

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "best.keras")
    print(f"Saved: {output_dir / 'best.keras'}")

    # Save history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_dict["config"] = {
        "epochs": config.epochs, "batch_size": config.batch_size,
        "board_weight": config.board_weight_multiplier, "dropout": config.dropout_rate,
        "augmentation": config.use_augmentation,
    }
    with open(output_dir / "history.json", "w") as f:
        json.dump(hist_dict, f, indent=2)

    return model


def quantize_int8(model, config, val_ds):
    def rep_data():
        for imgs, _ in val_ds.take(100):
            yield [imgs]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32

    tflite = converter.convert()
    out = Path(config.output_dir) / config.experiment_name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "model_int8.tflite", "wb") as f:
        f.write(tflite)
    print(f"INT8: {out / 'model_int8.tflite'} ({len(tflite) / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-weight", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--experiment", type=str, default="exp1")
    parser.add_argument("--output-dir", type=str, default="/tmp/gauge_geometry_v3")
    parser.add_argument("--quantize", action="store_true")

    args = parser.parse_args()

    config = TrainingConfig(
        board_weight_multiplier=args.board_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout_rate=args.dropout,
        initial_lr=args.lr,
        use_augmentation=not args.no_augment,
        experiment_name=args.experiment,
        output_dir=args.output_dir,
        quantize_int8=args.quantize,
    )

    print("=" * 60)
    print("SIMPLE CNN GEOMETRIC GAUGE READER")
    print("=" * 60)
    print(f"Board weight: {config.board_weight_multiplier}x")
    print(f"Epochs: {config.epochs}")
    print(f"Dropout: {config.dropout_rate}")
    print(f"Augmentation: {config.use_augmentation}")
    print(f"Output: {Path(config.output_dir) / config.experiment_name}")

    # Load data
    root = Path(__file__).resolve().parent.parent
    preprocessed_dir = root / "data" / "preprocessed_crops"

    all_samples = []
    meta_path = preprocessed_dir / "metadata.json"
    if meta_path.exists():
        all_samples.extend(load_metadata(meta_path))
        print(f"  {len(all_samples)} preprocessed crops")

    board_csv = root / "data" / "board_captures_labeled_v2.csv"
    board_samples = []
    if board_csv.exists():
        board_samples = load_board_captures_manifest(board_csv)
        # Oversample board captures by weight multiplier
        for s in board_samples:
            all_samples.append(s)
        print(f"  {len(board_samples)} board captures")

    if not all_samples:
        print("No data!")
        return

    print(f"  Total: {len(all_samples)}")

    # Split
    np.random.seed(42)
    idx = np.random.permutation(len(all_samples))
    vs = int(len(all_samples) * config.val_split)
    train_s = [all_samples[i] for i in idx[vs:]]
    val_s = [all_samples[i] for i in idx[:vs]]

    # Weight train samples by board_weight_multiplier
    weighted_train = []
    for s in train_s:
        weighted_train.append(s)
        if s.get("is_board", False):
            for _ in range(int(config.board_weight_multiplier) - 1):
                weighted_train.append(s)
    print(f"  Train: {len(weighted_train)} (weighted), Val: {len(val_s)}")

    # Build datasets
    train_ds, _ = build_dataset(weighted_train, config, augment=config.use_augmentation,
                                preprocessed_dir=preprocessed_dir)
    val_ds, _ = build_dataset(val_s, config, augment=False, preprocessed_dir=preprocessed_dir)

    # Train
    model = train_model(train_ds, val_ds, config)

    # Evaluate on board captures
    print("\nEvaluating on board captures...")
    if board_samples:
        result = evaluate_board(model, board_samples, config)
        print(f"  Board Temp MAE: {result['temp_mae']:.2f}°C")
        print(f"  Board Temp Median: {result['temp_median']:.2f}°C")

        out = Path(config.output_dir) / config.experiment_name
        with open(out / "board_eval.json", "w") as f:
            json.dump(result, f, indent=2)

        if result["temp_mae"] < 5.0:
            print("  ✓ TARGET ACHIEVED!")
        else:
            print(f"  ✗ Need <5°C (got {result['temp_mae']:.2f}°C)")

    # Quantize
    if config.quantize_int8:
        quantize_int8(model, config, val_ds)


if __name__ == "__main__":
    main()
