#!/usr/bin/env python3
"""
Geometric gauge reader v5 — pretrained backbone + angle regression.

Combines ImageNet pretrained MobileNetV2 features with sin/cos angle output.
Frozen backbone first, then fine-tune last layers.

Key differences from v4:
- MobileNetV2 backbone with ImageNet weights (stronger features)
- Two-phase training: frozen backbone → fine-tune top layers
- BatchNorm in frozen phase uses running stats (OK for feature extraction)
"""

from __future__ import annotations

import argparse
import csv
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
)

from luma_crop_detector import (
    estimate_bright_centroid,
    compute_dynamic_crop,
    crop_and_resize,
)


@dataclass
class TrainingConfig:
    image_size: int = 224
    input_channels: int = 3

    batch_size: int = 32
    epochs_frozen: int = 60
    epochs_finetune: int = 40
    initial_lr: float = 1e-3
    finetune_lr: float = 1e-5
    min_lr: float = 1e-6

    board_weight_multiplier: float = 10.0
    val_split: float = 0.15

    backbone: str = "mobilenetv2"  # mobilenetv2 or simple
    use_augmentation: bool = True
    dropout_rate: float = 0.3

    output_dir: str = "/tmp/gauge_geometry_v5"
    experiment_name: str = "exp1"
    quantize_int8: bool = True


def load_metadata(manifest_path: Path) -> list[dict[str, Any]]:
    with open(manifest_path, "r") as f:
        raw = json.load(f)
    samples = []
    for entry in raw:
        p = Path(entry["image_path"])
        cx, cy = float(entry["center_x_norm"]), float(entry["center_y_norm"])
        tx, ty = float(entry["tip_x_norm"]), float(entry["tip_y_norm"])
        angle = angle_degrees_from_center_to_tip(cx * 224, cy * 224, tx * 224, ty * 224)
        angle_rad = math.radians(angle)
        samples.append({
            "image_path": str(p),
            "temperature_c": float(entry["temperature_c"]),
            "sin_angle": math.sin(angle_rad),
            "cos_angle": math.cos(angle_rad),
            "angle_degrees": angle,
            "is_board": False,
        })
    return samples


def load_board_captures_manifest(csv_path: Path) -> list[dict[str, Any]]:
    samples = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sw, sh = float(row["source_width"]), float(row["source_height"])
            cx = float(row["center_x"]) / sw
            cy = float(row["center_y"]) / sh
            tx = float(row["tip_x"]) / sw
            ty = float(row["tip_y"]) / sh
            angle = angle_degrees_from_center_to_tip(cx * 224, cy * 224, tx * 224, ty * 224)
            angle_rad = math.radians(angle)
            samples.append({
                "image_path": str(row["image_path"]),
                "temperature_c": float(row["temperature_c"]),
                "sin_angle": math.sin(angle_rad),
                "cos_angle": math.cos(angle_rad),
                "angle_degrees": angle,
                "is_board": True,
            })
    return samples


def _conv_norm_swish(x, filters, strides=1, kernel_size=3):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=-1, epsilon=1e-6)(x)
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


def _build_simple_cnn(inputs, config: TrainingConfig):
    """Simple CNN backbone from v3/v4 (~65k params), trained from scratch."""
    x = _conv_norm_swish(inputs, 32, strides=2)
    x = _residual_separable_block(x, 32, dropout_rate=0.02)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = _residual_separable_block(x, 64, dropout_rate=0.04)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = _residual_separable_block(x, 96, dropout_rate=0.06)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = _residual_separable_block(x, 128, dropout_rate=0.08)
    x = layers.GlobalAveragePooling2D()(x)
    return x


def _build_mobilenetv2_backbone(inputs, config: TrainingConfig):
    """Pretrained MobileNetV2 backbone with GroupNorm head."""
    base = keras.applications.MobileNetV2(
        input_shape=(config.image_size, config.image_size, config.input_channels),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    x = base(inputs)

    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    return x


def _build_angle_head(x, config: TrainingConfig):
    """Shared regression head for sin/cos output."""
    x = layers.Dense(256, activation="swish")(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.Dropout(config.dropout_rate)(x)
    x = layers.Dense(128, activation="swish")(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.Dropout(config.dropout_rate)(x)
    outputs = layers.Dense(2, activation="tanh", name="angle_sincos")(x)
    return outputs


def build_angle_model(config: TrainingConfig) -> keras.Model:
    inputs = keras.Input(shape=(config.image_size, config.image_size, config.input_channels))

    if config.use_augmentation:
        x = keras.Sequential([
            layers.RandomRotation(10.0 / 360.0, fill_mode="reflect"),
            layers.RandomZoom(0.1, 0.1, fill_mode="reflect"),
            layers.RandomContrast(0.15),
            layers.RandomBrightness(0.15),
        ])(inputs)
    else:
        x = inputs

    if config.backbone == "mobilenetv2":
        features = _build_mobilenetv2_backbone(x, config)
    else:
        features = _build_simple_cnn(x, config)

    outputs = _build_angle_head(features, config)
    model = keras.Model(inputs=inputs, outputs=outputs, name="angle_regressor")
    return model


def load_both_types(image_path: Path, is_board: bool, target_size: int = 224,
                    preprocessed_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    if not image_path.exists():
        if is_board:
            return None
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
            if img.size != (target_size, target_size):
                img = img.resize((target_size, target_size), Image.BILINEAR)
            return np.asarray(img, dtype=np.uint8)
    except Exception as e:
        print(f"  Error loading {image_path}: {e}")
        return None


def build_dataset(samples, config: TrainingConfig, augment: bool,
                  preprocessed_dir: Optional[Path] = None) -> tuple[tf.data.Dataset, int]:
    images, targets = [], []
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
        targets.append(np.array([sample["sin_angle"], sample["cos_angle"]], dtype=np.float32))
    if not images:
        raise ValueError("No images loaded!")
    n = len(images)
    ds = tf.data.Dataset.from_tensor_slices((np.stack(images), np.stack(targets)))
    if augment:
        ds = ds.shuffle(max(n // 2, 100))
    ds = ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, n


def sincos_to_temp_fn(sin_val, cos_val):
    angle_rad = math.atan2(sin_val, cos_val)
    angle_deg = math.degrees(angle_rad) % 360
    temp = celsius_from_inner_dial_angle_degrees(angle_deg)
    return max(-30.0, min(50.0, temp))


def evaluate_board(model, board_samples, config):
    temp_errors = []
    repo_root = Path(__file__).resolve().parent.parent
    for sample in board_samples:
        path = Path(sample["image_path"])
        if not path.exists():
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
        out = model.predict(batch, verbose=0)[0]
        pt = sincos_to_temp_fn(float(out[0]), float(out[1]))
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


def _angle_mae_metric(y_true, y_pred):
    dot = y_true[:, 0] * y_pred[:, 0] + y_true[:, 1] * y_pred[:, 1]
    dot = tf.clip_by_value(dot, -1.0, 1.0)
    return tf.reduce_mean(tf.acos(dot))


def train_model(train_ds, val_ds, config: TrainingConfig):
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Frozen backbone, train head only
    model = build_angle_model(config)
    print(f"\nPhase 1 — Frozen backbone ({config.backbone})")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.initial_lr),
        loss="mse",
        metrics=[_angle_mae_metric, "mae"],
    )

    cb_frozen = [
        keras.callbacks.EarlyStopping(
            monitor="val_angle_mae_metric", patience=15, restore_best_weights=True, mode="min"),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_angle_mae_metric", factor=0.5, patience=6, min_lr=config.min_lr, mode="min"),
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "checkpoint_frozen.keras"),
            monitor="val_angle_mae_metric", save_best_only=True, mode="min"),
    ]

    history_frozen = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.epochs_frozen,
        callbacks=cb_frozen,
        verbose=1,
    )

    model.save(output_dir / "best_frozen.keras")
    print(f"Saved: {output_dir / 'best_frozen.keras'}")

    # Phase 2: Fine-tune last layers of backbone
    if config.backbone == "mobilenetv2":
        print(f"\nPhase 2 — Fine-tuning last 20 layers")
        for layer in model.layers:
            if hasattr(layer, "_name") and layer._name == "mobilenetv2_1.00_224":
                base = layer
                # Unfreeze last 20 layers
                for l in base.layers[-20:]:
                    l.trainable = True
                break

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.finetune_lr),
            loss="mse",
            metrics=[_angle_mae_metric, "mae"],
        )

        cb_finetune = [
            keras.callbacks.EarlyStopping(
                monitor="val_angle_mae_metric", patience=10, restore_best_weights=True, mode="min"),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_angle_mae_metric", factor=0.5, patience=5, min_lr=config.min_lr, mode="min"),
            keras.callbacks.ModelCheckpoint(
                str(output_dir / "checkpoint.keras"),
                monitor="val_angle_mae_metric", save_best_only=True, mode="min"),
        ]

        history_ft = model.fit(
            train_ds, validation_data=val_ds,
            epochs=config.epochs_finetune,
            callbacks=cb_finetune,
            verbose=1,
        )

        model.save(output_dir / "best.keras")
        print(f"Saved: {output_dir / 'best.keras'}")

        # Combine histories
        combined = {}
        for k in history_frozen.history:
            combined[k + "_frozen"] = [float(v) for v in history_frozen.history[k]]
        for k in history_ft.history:
            combined[k] = [float(v) for v in history_ft.history[k]]
        combined["config"] = {
            "epochs_frozen": config.epochs_frozen,
            "epochs_finetune": config.epochs_finetune,
            "batch_size": config.batch_size,
            "board_weight": config.board_weight_multiplier,
            "backbone": config.backbone,
            "dropout": config.dropout_rate,
        }
        with open(output_dir / "history.json", "w") as f:
            json.dump(combined, f, indent=2)

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
    parser.add_argument("--epochs-frozen", type=int, default=60)
    parser.add_argument("--epochs-finetune", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--backbone", choices=["mobilenetv2", "simple"], default="mobilenetv2")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--experiment", type=str, default="exp1")
    parser.add_argument("--output-dir", type=str, default="/tmp/gauge_geometry_v5")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    config = TrainingConfig(
        board_weight_multiplier=args.board_weight,
        epochs_frozen=args.epochs_frozen,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        dropout_rate=args.dropout,
        initial_lr=args.lr,
        finetune_lr=args.finetune_lr,
        backbone=args.backbone,
        use_augmentation=not args.no_augment,
        experiment_name=args.experiment,
        output_dir=args.output_dir,
        quantize_int8=args.quantize,
    )

    print("=" * 60)
    print(f"ANGLE REGRESSION v5 — backbone={config.backbone}")
    print("=" * 60)
    print(f"Board weight: {config.board_weight_multiplier}x")
    print(f"Epochs: {config.epochs_frozen} frozen + {config.epochs_finetune} finetune")
    print(f"Dropout: {config.dropout_rate}")
    print(f"Output: {Path(config.output_dir) / config.experiment_name}")

    root = Path(__file__).resolve().parent.parent
    preprocessed_dir = root / "data" / "preprocessed_crops"

    all_samples = []
    meta_path = preprocessed_dir / "metadata.json"
    if meta_path.exists():
        all_samples.extend(load_metadata(meta_path))
    print(f"  {sum(1 for s in all_samples if not s.get('is_board'))} preprocessed crops")

    board_csv = root / "data" / "board_captures_labeled_v2.csv"
    board_samples = []
    if board_csv.exists():
        board_samples = load_board_captures_manifest(board_csv)
        for s in board_samples:
            all_samples.append(s)
    print(f"  {len(board_samples)} board captures")
    print(f"  Total: {len(all_samples)}")

    if not all_samples:
        print("No data!")
        return

    np.random.seed(42)
    idx = np.random.permutation(len(all_samples))
    vs = int(len(all_samples) * config.val_split)
    train_s = [all_samples[i] for i in idx[vs:]]
    val_s = [all_samples[i] for i in idx[:vs]]

    weighted_train = []
    for s in train_s:
        weighted_train.append(s)
        if s.get("is_board", False):
            for _ in range(int(config.board_weight_multiplier) - 1):
                weighted_train.append(s)
    print(f"  Train: {len(weighted_train)} (weighted), Val: {len(val_s)}")

    train_ds, _ = build_dataset(weighted_train, config, augment=config.use_augmentation,
                                preprocessed_dir=preprocessed_dir)
    val_ds, _ = build_dataset(val_s, config, augment=False, preprocessed_dir=preprocessed_dir)

    model = train_model(train_ds, val_ds, config)

    print("\nEvaluating on board captures...")
    if board_samples:
        result = evaluate_board(model, board_samples, config)
        print(f"  Board Temp MAE: {result['temp_mae']:.2f}°C")
        if result["temp_mae"] != float("inf"):
            print(f"  Board Temp Median: {result['temp_median']:.2f}°C")
        with open(Path(config.output_dir) / config.experiment_name / "board_eval.json", "w") as f:
            json.dump(result, f, indent=2)
        if result["temp_mae"] < 5.0:
            print("  ✓ TARGET ACHIEVED!")
        else:
            print(f"  ✗ Need <5°C (got {result['temp_mae']:.2f}°C)")

    if config.quantize_int8:
        quantize_int8(model, config, val_ds)


if __name__ == "__main__":
    main()
