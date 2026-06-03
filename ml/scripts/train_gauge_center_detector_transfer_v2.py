#!/usr/bin/env python3
"""Train GaugeCenterDetectorTransferV2.

Architecture:
  Input: 224×224×3
  Backbone: MobileNetV2 alpha=0.5, ImageNet pretrained
  Mid-level features: block_5_add (28×28×16)
  Decoder:
    - 1×1 conv 16→64
    - Upsample 2× → 56×56
    - Depthwise separable conv 3×3 64→32
    - 1×1 conv 32→1 → 56×56×1 heatmap
  Decode: softargmax → (cx, cy) in [0, 1]
  Loss: heatmap MSE (σ=2.0) + coordinate Huber (δ=0.02)
  Augmentation: colour jitter + rotation (±2.3°)
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

IMAGE_H = 224
IMAGE_W = 224
HEATMAP_SIZE = 56
GAUSSIAN_SIGMA = 2.0
HUBER_DELTA = 0.02
BATCH_SIZE = 16
EPOCHS_WARMUP = 15
EPOCHS_FINETUNE = 100
ALPHA = 0.5
LR = 3e-4
LOSS_HM_W = 1.0
LOSS_COORD_W = 10.0

DATA_DIR = PROJECT_ROOT / "data" / "heatmap_training"
METADATA_PATH = str(DATA_DIR / "metadata.json")
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "training"
RUN_NAME = f"gauge_center_detector_transfer_v2_{datetime.now():%Y%m%d_%H%M%S}"
RUN_DIR = ARTIFACTS_DIR / RUN_NAME

# Pre-create heatmap coordinate grid (constant, reused)
_hm_y = tf.range(HEATMAP_SIZE, dtype=tf.float32)
_hm_x = tf.range(HEATMAP_SIZE, dtype=tf.float32)
_hm_yy, _hm_xx = tf.meshgrid(_hm_y, _hm_x, indexing="ij")
_hm_yy = tf.expand_dims(tf.expand_dims(_hm_yy, 0), -1)  # 1×H×W×1
_hm_xx = tf.expand_dims(tf.expand_dims(_hm_xx, 0), -1)  # 1×H×W×1


class SoftArgmaxLayer(keras.layers.Layer):
    """Differentiable softargmax: heatmap → (cx, cy) in [0, 1]."""

    def __init__(self, h: int = HEATMAP_SIZE, w: int = HEATMAP_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.h = h
        self.w = w

    def call(self, heatmap: tf.Tensor) -> tf.Tensor:
        flat = tf.reshape(heatmap, [-1, self.h * self.w])
        probs = tf.nn.softmax(flat, axis=-1)
        probs = tf.reshape(probs, [-1, self.h, self.w, 1])

        y_pos = tf.range(self.h, dtype=tf.float32)
        x_pos = tf.range(self.w, dtype=tf.float32)
        y_pos = tf.reshape(y_pos, [1, self.h, 1, 1])
        x_pos = tf.reshape(x_pos, [1, 1, self.w, 1])

        y_pred = tf.reduce_sum(probs * y_pos, axis=[1, 2])
        x_pred = tf.reduce_sum(probs * x_pos, axis=[1, 2])

        y_norm = y_pred / (self.h - 1)
        x_norm = x_pred / (self.w - 1)
        return tf.concat([x_norm, y_norm], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"h": self.h, "w": self.w})
        return config


def _generate_heatmap_tf(cx_norm: tf.Tensor, cy_norm: tf.Tensor) -> tf.Tensor:
    """Generate 56×56 Gaussian heatmap from normalized coordinates."""
    denom = 2.0 * GAUSSIAN_SIGMA ** 2
    cx = cx_norm * (HEATMAP_SIZE - 1)
    cy = cy_norm * (HEATMAP_SIZE - 1)
    hm = tf.exp(-((_hm_xx - cx) ** 2 + (_hm_yy - cy) ** 2) / denom)
    return hm  # 1×H×W×1 (first dim is batch for broadcasting)


def load_metadata(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_dataset(
    entries: list[dict],
    augment: bool = False,
    shuffle: bool = False,
) -> tf.data.Dataset:
    paths = []
    coords = []
    for e in entries:
        paths.append(os.path.join(DATA_DIR, e["image_path"]))
        coords.append(np.array([float(e["cx_norm"]), float(e["cy_norm"])], dtype=np.float32))

    def _gen():
        for p, c in zip(paths, coords):
            yield p, c

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_types=(tf.string, tf.float32),
        output_shapes=((), (2,)),
    )

    def _load(path, c):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [IMAGE_H, IMAGE_W])
        img = tf.cast(img, tf.float32)

        cx, cy = c[0], c[1]
        heatmap = _generate_heatmap_tf(cx, cy)
        heatmap = tf.squeeze(heatmap, axis=0)  # H×W×1 (remove batch dim from broadcasting)
        return img, (heatmap, c)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        def _augment(img, label):
            heatmap, c = label
            cx, cy = c[0], c[1]

            # Colour jitter
            img = tf.image.random_brightness(img, max_delta=0.20)
            img = tf.image.random_contrast(img, lower=0.6, upper=1.4)
            img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
            img = tf.image.random_hue(img, max_delta=0.08)

            # Rotation (±2.3°) — image + label
            angle = tf.random.uniform([], -0.04, 0.04)
            c_rot = tf.cos(angle)
            s_rot = tf.sin(angle)
            cx_i = tf.constant(IMAGE_W / 2.0, tf.float32)
            cy_i = tf.constant(IMAGE_H / 2.0, tf.float32)
            tx = (1.0 - c_rot) * cx_i + s_rot * cy_i
            ty = -s_rot * cx_i + (1.0 - c_rot) * cy_i
            transform = [c_rot, -s_rot, tx, s_rot, c_rot, ty, 0.0, 0.0]
            img = tf.raw_ops.ImageProjectiveTransformV3(
                images=tf.expand_dims(img, 0),
                transforms=tf.expand_dims(transform, 0),
                output_shape=[IMAGE_H, IMAGE_W],
                fill_value=128,
                interpolation="BILINEAR",
            )[0]
            img = tf.clip_by_value(img, 0.0, 255.0)

            # Rotate coordinates
            lx = cx * IMAGE_W - cx_i
            ly = cy * IMAGE_H - cy_i
            lx_r = c_rot * lx - s_rot * ly
            ly_r = s_rot * lx + c_rot * ly
            cx_r = (lx_r + cx_i) / tf.cast(IMAGE_W, tf.float32)
            cy_r = (ly_r + cy_i) / tf.cast(IMAGE_H, tf.float32)
            c_new = tf.stack([cx_r, cy_r])

            # Regenerate heatmap from rotated coords
            heatmap = _generate_heatmap_tf(cx_r, cy_r)
            heatmap = tf.squeeze(heatmap, axis=0)
            return img, (heatmap, c_new)

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(512)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model() -> keras.Model:
    """V2 model: MobileNetV2 alpha=0.5 + mid-level features + 56×56 heatmap."""
    inputs = keras.Input(shape=(IMAGE_H, IMAGE_W, 3), name="image")
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)

    # Backbone
    backbone = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        input_shape=(IMAGE_H, IMAGE_W, 3),
        alpha=ALPHA,
    )
    backbone.trainable = False  # frozen during warmup

    # Grab mid-level 28×28 features
    mid_features = backbone.get_layer("block_5_add").output  # 28×28×16

    # ---- Decoder ----
    # 1×1 conv to expand channels (16→64)
    x = keras.layers.Conv2D(64, 1, padding="same", activation="relu", name="decoder_expand")(mid_features)
    # Upsample 2× → 56×56
    x = keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="decoder_upsample")(x)
    # Depthwise separable conv
    x = keras.layers.SeparableConv2D(32, 3, padding="same", activation="relu", name="decoder_sepconv")(x)
    # Final 1×1 conv → heatmap
    heatmap = keras.layers.Conv2D(1, 1, padding="same", name="heatmap_output")(x)
    # Softargmax → coordinates
    coords = SoftArgmaxLayer(name="coords_output")(heatmap)

    model = keras.Model(inputs=inputs, outputs=[heatmap, coords], name="GaugeCenterDetectorTransferV2")
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
        optimizer=keras.optimizers.Adam(LR),
        loss={
            "heatmap_output": keras.losses.MeanSquaredError(),
            "coords_output": keras.losses.Huber(delta=HUBER_DELTA),
        },
        loss_weights={
            "heatmap_output": LOSS_HM_W,
            "coords_output": LOSS_COORD_W,
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
            patience=25,
            restore_best_weights=False,
            verbose=1,
        ),
    ]

    # Phase 1: Warmup (backbone frozen)
    print("\n--- Phase 1: Warmup (backbone frozen) ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_WARMUP,
        callbacks=callbacks,
        verbose=2,
    )

    # Phase 2: Fine-tune
    print("\n--- Phase 2: Fine-tune (backbone unfrozen) ---")
    for layer in model.layers:
        if hasattr(layer, "trainable"):
            layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(LR * 0.1),
        loss={
            "heatmap_output": keras.losses.MeanSquaredError(),
            "coords_output": keras.losses.Huber(delta=HUBER_DELTA),
        },
        loss_weights={
            "heatmap_output": LOSS_HM_W,
            "coords_output": LOSS_COORD_W,
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

    # Evaluate best model
    model = keras.models.load_model(
        str(RUN_DIR / "best_model.keras"),
        custom_objects={"SoftArgmaxLayer": SoftArgmaxLayer},
    )

    test_results = model.evaluate(test_ds, verbose=0)
    print(f"\n--- Test set results ---")
    for name, val in zip(model.metrics_names, test_results):
        print(f"  {name}: {val:.6f}")

    # Per-sample errors
    all_pred_coords = []
    all_true_coords = []
    for batch_x, batch_y in test_ds:
        _, pred_coords = model.predict(batch_x, verbose=0)
        all_pred_coords.append(pred_coords)
        _, true_coords = batch_y
        all_true_coords.append(true_coords.numpy())

    all_pred_coords = np.concatenate(all_pred_coords, axis=0)
    all_true_coords = np.concatenate(all_true_coords, axis=0)

    errors_px = np.abs(all_pred_coords - all_true_coords) * IMAGE_W
    print(f"\nPer-axis errors (pixels):")
    print(f"  cx MAE: {errors_px[:, 0].mean():.2f} px  (std={errors_px[:, 0].std():.2f})")
    print(f"  cy MAE: {errors_px[:, 1].mean():.2f} px  (std={errors_px[:, 1].std():.2f})")
    print(f"  Euclidean MAE: {np.mean(np.sqrt(np.sum(errors_px**2, axis=1))):.2f} px")

    model.save(str(RUN_DIR / "final_model.keras"))

    summary = {
        "run_name": RUN_NAME,
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "test_samples": len(test_entries),
        "alpha": ALPHA,
        "heatmap_size": HEATMAP_SIZE,
        "gaussian_sigma": GAUSSIAN_SIGMA,
        "huber_delta": HUBER_DELTA,
        "test_cx_mae_px": float(errors_px[:, 0].mean()),
        "test_cy_mae_px": float(errors_px[:, 1].mean()),
        "test_euclidean_mae_px": float(np.mean(np.sqrt(np.sum(errors_px**2, axis=1)))),
    }
    with open(str(RUN_DIR / "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {RUN_DIR}")


if __name__ == "__main__":
    main()
