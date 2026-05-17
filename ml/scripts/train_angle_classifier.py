#!/usr/bin/env python3
"""Train angle-classification CNN for gauge reading."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import (
    load_gauge_specs,
    needle_angle_clockwise_rad,
    needle_value,
    GaugeSpec,
)

IMAGE_SIZE = 224
NUM_ANGLE_BINS = 90
SOFT_TARGET_SIGMA = 2.0

def _compute_crop_box(sample, crop_pad_ratio=0.25):
    cx, cy = sample.dial.cx, sample.dial.cy
    rx, ry = sample.dial.rx, sample.dial.ry
    pad = max(rx, ry) * crop_pad_ratio
    x1 = max(0.0, cx - rx - pad)
    y1 = max(0.0, cy - ry - pad)
    x2 = cx + rx + pad
    y2 = cy + ry + pad
    return (x1, y1, x2, y2)

def _load_image(path, crop_box):
    image = tf.io.read_file(str(path))
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.ensure_shape(image, [None, None, 3])
    image = tf.cast(image, tf.float32) / 255.0
    x1, y1, x2, y2 = crop_box
    x1 = int(x1)
    y1 = int(y1)
    crop_w = int(x2) - x1
    crop_h = int(y2) - y1
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    x1 = tf.clip_by_value(x1, 0, w - 1)
    y1 = tf.clip_by_value(y1, 0, h - 1)
    crop_w = tf.clip_by_value(crop_w, 1, w - x1)
    crop_h = tf.clip_by_value(crop_h, 1, h - y1)
    image = tf.image.crop_to_bounding_box(image, y1, x1, crop_h, crop_w)
    image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    return image

def _angle_to_soft_target(angle_rad, spec, num_bins=NUM_ANGLE_BINS, sigma=SOFT_TARGET_SIGMA):
    raw = (angle_rad - spec.min_angle_rad) % (2.0 * math.pi)
    if raw > spec.sweep_rad:
        raw = spec.sweep_rad
    bin_idx = raw / spec.sweep_rad * (num_bins - 1)
    indices = np.arange(num_bins, dtype=np.float64)
    dist = np.minimum(np.abs(indices - bin_idx), num_bins - np.abs(indices - bin_idx))
    target = np.exp(-0.5 * (dist / sigma) ** 2)
    target = target / target.sum()
    return target.astype(np.float32)

def _build_model(num_bins=NUM_ANGLE_BINS, alpha=0.35):
    base = keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        alpha=alpha,
    )
    base.trainable = False
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = base(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(128, activation="swish")(x)
    x = keras.layers.Dropout(0.2)(x)
    logits = keras.layers.Dense(num_bins, name="angle_logits")(x)
    model = keras.Model(inputs=inputs, outputs=logits, name="angle_classifier")
    return model

def _expected_angle(logits, spec, num_bins):
    probs = tf.nn.softmax(logits, axis=-1)
    angles = spec.min_angle_rad + tf.linspace(0.0, spec.sweep_rad, num_bins)
    sin_sum = tf.reduce_sum(probs * tf.sin(angles), axis=-1)
    cos_sum = tf.reduce_sum(probs * tf.cos(angles), axis=-1)
    mean_angle = tf.atan2(sin_sum, cos_sum)
    return mean_angle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--bins", type=int, default=NUM_ANGLE_BINS)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    args = parser.parse_args()

    print("[ANGLE] Loading dataset...")
    samples = load_dataset()
    spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]

    rng = np.random.default_rng(seed=21)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    n_val = int(len(samples) * 0.15)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    print(f"[ANGLE] Train={len(train_samples)} Val={len(val_samples)}")

    print("[ANGLE] Building model...")
    model = _build_model(num_bins=args.bins, alpha=args.alpha)
    if args.unfreeze_backbone:
        model.layers[1].trainable = True
        print("[ANGLE] Backbone unfrozen.")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=["accuracy"],
    )

    train_images = []
    train_targets = []
    train_temps = []
    for sample in train_samples:
        angle = needle_angle_clockwise_rad(sample)
        temp = needle_value(sample, spec, strict=False)
        target = _angle_to_soft_target(angle, spec, args.bins)
        img = _load_image(sample.image_path, _compute_crop_box(sample))
        train_images.append(img)
        train_targets.append(target)
        train_temps.append(temp)

    val_images = []
    val_targets = []
    val_temps = []
    for sample in val_samples:
        angle = needle_angle_clockwise_rad(sample)
        temp = needle_value(sample, spec, strict=False)
        target = _angle_to_soft_target(angle, spec, args.bins)
        img = _load_image(sample.image_path, _compute_crop_box(sample))
        val_images.append(img)
        val_targets.append(target)
        val_temps.append(temp)

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_targets, train_temps))
    train_ds = train_ds.shuffle(buffer_size=len(train_images), reshuffle_each_iteration=True)
    train_ds = train_ds.batch(args.batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_targets, val_temps))
    val_ds = val_ds.batch(args.batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    print("[ANGLE] Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
    )

    print("[ANGLE] Evaluating...")
    preds = []
    trues = []
    for img, target, temp in val_ds:
        logits = model.predict(img, verbose=0)
        pred_angle = _expected_angle(logits, spec, args.bins)
        pred_temp = spec.min_value + ((pred_angle - spec.min_angle_rad) / spec.sweep_rad) * (spec.max_value - spec.min_value)
        preds.extend(pred_temp.numpy().tolist())
        trues.extend(temp.numpy().tolist())

    errs = np.array(preds) - np.array(trues)
    mae = np.abs(errs).mean()
    rmse = np.sqrt((errs ** 2).mean())
    print(f"[ANGLE] Val MAE={mae:.2f}C RMSE={rmse:.2f}C")

    out_dir = PROJECT_ROOT / "artifacts" / "training" / "angle_classifier_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.keras")
    print(f"[ANGLE] Saved to {out_dir}")

if __name__ == "__main__":
    main()