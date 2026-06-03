#!/usr/bin/env python3
"""Train GaugeCenterDetectorTransferV1.

Architecture:
  Input: 224×224×3
  Backbone: MobileNetV2 alpha=0.35 (ImageNet pretrained, classification head removed)
  Decoder:
    - 1×1 conv to reduce channels
    - Upsample 2×
    - Depthwise separable conv
    - Upsample 2×
    - Depthwise separable conv
    - Final 1×1 conv → 32×32×1 heatmap
  Decode: softargmax → (cx, cy) in [0, 1]
  Loss: heatmap MSE + coordinate Huber
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
HEATMAP_SIZE = 32
BATCH_SIZE = 16
EPOCHS_WARMUP = 10
EPOCHS_FINETUNE = 80
ALPHA = 0.35
LEARNING_RATE = 3e-4
HUBER_DELTA = 0.02
GAUSSIAN_SIGMA = 1.5
LOSS_HEATMAP_WEIGHT = 1.0
LOSS_COORD_WEIGHT = 10.0

DATA_DIR = PROJECT_ROOT / "data" / "heatmap_training"
METADATA_PATH = str(DATA_DIR / "metadata.json")
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "training"
RUN_NAME = f"gauge_center_detector_transfer_v1_{datetime.now():%Y%m%d_%H%M%S}"
RUN_DIR = ARTIFACTS_DIR / RUN_NAME


class SoftArgmaxLayer(keras.layers.Layer):
    """Differentiable softargmax: converts heatmap to (x, y) coordinates in [0, 1]."""

    def __init__(self, h: int = HEATMAP_SIZE, w: int = HEATMAP_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.h = h
        self.w = w

    def call(self, heatmap: tf.Tensor) -> tf.Tensor:
        # heatmap: (batch, h, w, 1)
        flat = tf.reshape(heatmap, [-1, self.h * self.w])
        probs = tf.nn.softmax(flat, axis=-1)
        probs = tf.reshape(probs, [-1, self.h, self.w, 1])

        # Create coordinate grids
        y_pos = tf.range(self.h, dtype=tf.float32)
        x_pos = tf.range(self.w, dtype=tf.float32)
        y_pos = tf.reshape(y_pos, [1, self.h, 1, 1])
        x_pos = tf.reshape(x_pos, [1, 1, self.w, 1])

        # Weighted sum over spatial dimensions
        y_pred = tf.reduce_sum(probs * y_pos, axis=[1, 2])  # (batch, 1)
        x_pred = tf.reduce_sum(probs * x_pos, axis=[1, 2])  # (batch, 1)

        # Normalize to [0, 1]
        y_norm = y_pred / (self.h - 1)
        x_norm = x_pred / (self.w - 1)

        return tf.concat([x_norm, y_norm], axis=-1)  # (batch, 2)

    def get_config(self):
        config = super().get_config()
        config.update({"h": self.h, "w": self.w})
        return config


def load_metadata(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def generate_gaussian_heatmap(cx_norm: float, cy_norm: float, size: int = HEATMAP_SIZE, sigma: float = GAUSSIAN_SIGMA) -> np.ndarray:
    """Generate a 2D Gaussian heatmap centered at (cx_norm, cy_norm)."""
    cx = cx_norm * (size - 1)
    cy = cy_norm * (size - 1)
    y = np.arange(size)
    x = np.arange(size)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    return heatmap.astype(np.float32)


def parse_label(entry: dict) -> tuple[str, np.ndarray, np.ndarray]:
    """Return (image_path, heatmap_target, coord_target)."""
    img_path = os.path.join(DATA_DIR, entry["image_path"])
    cx = float(entry["cx_norm"])
    cy = float(entry["cy_norm"])
    heatmap = generate_gaussian_heatmap(cx, cy)
    heatmap = np.expand_dims(heatmap, axis=-1)  # (H, W, 1)
    coords = np.array([cx, cy], dtype=np.float32)
    return img_path, heatmap, coords


def build_dataset(
    entries: list[dict],
    augment: bool = False,
    shuffle: bool = False,
) -> tf.data.Dataset:
    paths = []
    heatmaps = []
    coords = []
    for e in entries:
        p, hm, c = parse_label(e)
        paths.append(p)
        heatmaps.append(hm)
        coords.append(c)

    def _gen():
        for p, hm, c in zip(paths, heatmaps, coords):
            yield p, (hm, c)

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_types=(tf.string, (tf.float32, tf.float32)),
        output_shapes=((), ((HEATMAP_SIZE, HEATMAP_SIZE, 1), (2,))),
    )

    def _load(path, label):
        hm, c = label
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img = tf.cast(img, tf.float32)
        return img, (hm, c)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        def _augment_fn(img, label):
            hm, c = label
            img = tf.image.random_brightness(img, max_delta=0.20)
            img = tf.image.random_contrast(img, lower=0.6, upper=1.4)
            img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
            img = tf.image.random_hue(img, max_delta=0.08)
            img = tf.clip_by_value(img, 0.0, 255.0)
            return img, (hm, c)
        ds = ds.map(_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(512)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model() -> keras.Model:
    """GaugeCenterDetectorTransferV1: MobileNetV2 + heatmap decoder + softargmax."""
    inputs = keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), name="image")

    # Preprocess: scale to [-1, 1] for MobileNetV2
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)

    # Backbone: MobileNetV2 (classification head removed)
    backbone = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        alpha=ALPHA,
    )
    backbone.trainable = False  # Freeze during warmup
    features = backbone(x, training=False)
    # features shape: (batch, 7, 7, C) where C depends on alpha

    # ---- Decoder ----
    # 1x1 conv to reduce channels
    x = keras.layers.Conv2D(128, 1, padding="same", activation="relu", name="decoder_reduce")(features)

    # Upsample 1: 7 → 14
    x = keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="upsample1")(x)
    x = keras.layers.SeparableConv2D(64, 3, padding="same", activation="relu", name="sepconv1")(x)

    # Upsample 2: 14 → 28
    x = keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="upsample2")(x)
    x = keras.layers.SeparableConv2D(32, 3, padding="same", activation="relu", name="sepconv2")(x)

    # Resize to 32×32 (from 28×28)
    x = keras.layers.Resizing(HEATMAP_SIZE, HEATMAP_SIZE, interpolation="bilinear", name="resize_to_32")(x)

    # Final 1×1 conv → single-channel heatmap
    heatmap = keras.layers.Conv2D(1, 1, padding="same", name="heatmap_output")(x)

    # Softargmax layer to get coordinates
    coords = SoftArgmaxLayer(name="coords_output")(heatmap)

    model = keras.Model(inputs=inputs, outputs=[heatmap, coords], name="GaugeCenterDetectorTransferV1")
    return model


def main():
    os.makedirs(str(RUN_DIR), exist_ok=True)

    all_entries = load_metadata(METADATA_PATH)
    train_entries = [e for e in all_entries if e["split"] == "train"]
    val_entries = [e for e in all_entries if e["split"] == "val"]
    test_entries = [e for e in all_entries if e["split"] == "test"]

    print(f"Train: {len(train_entries)}  Val: {len(val_entries)}  Test: {len(test_entries)}")

    train_ds = build_dataset(train_entries, augment=True, shuffle=True)
    val_ds = build_dataset(val_entries, augment=False)
    test_ds = build_dataset(test_entries, augment=False)

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss={
            "heatmap_output": keras.losses.MeanSquaredError(),
            "coords_output": keras.losses.Huber(delta=HUBER_DELTA),
        },
        loss_weights={
            "heatmap_output": LOSS_HEATMAP_WEIGHT,
            "coords_output": LOSS_COORD_WEIGHT,
        },
        metrics={
            "coords_output": [keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )
    
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(RUN_DIR / "best_model.keras"),
            monitor="val_coords_output_mae",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_coords_output_mae",
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_coords_output_mae",
            mode="min",
            patience=20,
            restore_best_weights=False,
            verbose=1,
        ),
    ]

    # Phase 1: Warmup with frozen backbone
    print("\n--- Phase 1: Warmup (backbone frozen) ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_WARMUP,
        callbacks=callbacks,
        verbose=2,
    )

    # Phase 2: Fine-tune with backbone unfrozen
    print("\n--- Phase 2: Fine-tune (backbone unfrozen) ---")
    for layer in model.layers:
        if hasattr(layer, "trainable"):
            layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE * 0.1),
        loss={
            "heatmap_output": keras.losses.MeanSquaredError(),
            "coords_output": keras.losses.Huber(delta=HUBER_DELTA),
        },
        loss_weights={
            "heatmap_output": LOSS_HEATMAP_WEIGHT,
            "coords_output": LOSS_COORD_WEIGHT,
        },
        metrics={
            "coords_output": [keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=EPOCHS_WARMUP,
        epochs=EPOCHS_WARMUP + EPOCHS_FINETUNE,
        callbacks=callbacks,
        verbose=2,
    )

    # Load best and evaluate
    model = keras.models.load_model(
        str(RUN_DIR / "best_model.keras"),
        custom_objects={"SoftArgmaxLayer": SoftArgmaxLayer},
    )

    test_results = model.evaluate(test_ds, verbose=0)
    print(f"\n--- Test set results ---")
    for name, val in zip(model.metrics_names, test_results):
        print(f"  {name}: {val:.6f}")

    # Per-sample errors on test set
    all_pred_coords = []
    all_true_coords = []
    
    for batch_x, batch_y in test_ds:
        pred_hm, pred_coords = model.predict(batch_x, verbose=0)
        all_pred_coords.append(pred_coords)
        _, batch_coords = batch_y
        all_true_coords.append(batch_coords.numpy())
    
    all_pred_coords = np.concatenate(all_pred_coords, axis=0)
    all_true_coords = np.concatenate(all_true_coords, axis=0)
    
    errors_px = np.abs(all_pred_coords - all_true_coords) * IMAGE_WIDTH
    print(f"\nPer-axis errors (pixels):")
    print(f"  cx MAE: {errors_px[:, 0].mean():.2f} px  (std={errors_px[:, 0].std():.2f})")
    print(f"  cy MAE: {errors_px[:, 1].mean():.2f} px  (std={errors_px[:, 1].std():.2f})")
    print(f"  Euclidean MAE: {np.mean(np.sqrt(np.sum(errors_px**2, axis=1))):.2f} px")

    # Save final
    model.save(str(RUN_DIR / "final_model.keras"))
    
    # Save summary
    summary = {
        "run_name": RUN_NAME,
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "test_samples": len(test_entries),
        "alpha": ALPHA,
        "huber_delta": HUBER_DELTA,
        "gaussian_sigma": GAUSSIAN_SIGMA,
        "heatmap_weight": LOSS_HEATMAP_WEIGHT,
        "coord_weight": LOSS_COORD_WEIGHT,
        "test_cx_mae_px": float(errors_px[:, 0].mean()),
        "test_cy_mae_px": float(errors_px[:, 1].mean()),
        "test_euclidean_mae_px": float(np.mean(np.sqrt(np.sum(errors_px**2, axis=1)))),
    }
    with open(str(RUN_DIR / "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {RUN_DIR}")


if __name__ == "__main__":
    main()
