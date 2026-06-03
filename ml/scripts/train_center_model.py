#!/usr/bin/env python3
"""Train a MobileNetV2 regression model to predict the gauge dial centre
from a 224×224 input frame.

Uses the preprocessed_crops dataset (418 images with centre annotations)
and data augmentation to produce a model robust to live board captures.
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
BATCH_SIZE = 16
EPOCHS_WARMUP = 8
EPOCHS_FINETUNE = 60
ALPHA = 0.50           # MobileNetV2 width multiplier — bigger = more capacity
HEAD_UNITS = 128
HEAD_DROPOUT = 0.30
LEARNING_RATE = 3e-4


DATA_DIR = PROJECT_ROOT / "data" / "preprocessed_crops"
METADATA_PATH = str(DATA_DIR / "metadata.json")
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "training"
RUN_NAME = f"center_model_{datetime.now():%Y%m%d_%H%M%S}"
RUN_DIR = ARTIFACTS_DIR / RUN_NAME


def load_metadata(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def parse_label(entry: dict) -> tuple[str, np.ndarray]:
    """Return (image_path, (center_x_norm, center_y_norm))."""
    img_path = os.path.join(DATA_DIR, entry["image_path"])
    cx = float(entry["center_x_norm"])
    cy = float(entry["center_y_norm"])
    return img_path, np.array([cx, cy], dtype=np.float32)


def _augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Heavy data augmentation to generalise to board captures.

    WARNING: any geometric transform (crop, translation, rotation) that
    shifts the gauge centre MUST also shift the label.  The colour jitter
    below is safe — it does not move pixels.
    """
    # Colour jitter
    image = tf.image.random_brightness(image, max_delta=0.20)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.image.random_hue(image, max_delta=0.08)

    # Small random rotation — adjust label to match.
    # Rotate around the image centre (112, 112).  A point (cx, cy) in
    # the original image moves to (cx', cy') in the rotated-then-cropped
    # output.  We rotate the image then apply the same transform to the
    # normalised label coordinates.
    angle_rad = tf.random.uniform([], -0.04, 0.04)   # ≈ ±2.3°
    # Build the affine transform matrix for ImageProjectiveTransformV3
    # which expects [a0 a1 a2; b0 b1 b2; 0 0 1] flattened row-major.
    c = tf.cos(angle_rad)
    s = tf.sin(angle_rad)
    # Centre of rotation at image centre (cx_img, cy_img) in pixels.
    cx_img = tf.constant(IMAGE_WIDTH / 2.0, tf.float32)
    cy_img = tf.constant(IMAGE_HEIGHT / 2.0, tf.float32)
    # Translation component to keep the image centre stationary:
    tx = (1.0 - c) * cx_img + s * cy_img
    ty = -s * cx_img + (1.0 - c) * cy_img
    transform = [c, -s, tx, s, c, ty, 0.0, 0.0]
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=[IMAGE_HEIGHT, IMAGE_WIDTH],
        fill_value=128,
        interpolation="BILINEAR",
    )[0]

    # Apply the same rotation to the label (convert to pixel coords,
    # rotate around image centre, convert back to norm).
    lx = label[0] * tf.constant(IMAGE_WIDTH, tf.float32) - cx_img
    ly = label[1] * tf.constant(IMAGE_HEIGHT, tf.float32) - cy_img
    lx_rot = c * lx - s * ly
    ly_rot = s * lx + c * ly
    label = tf.stack([
        (lx_rot + cx_img) / tf.constant(IMAGE_WIDTH, tf.float32),
        (ly_rot + cy_img) / tf.constant(IMAGE_HEIGHT, tf.float32),
    ])

    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def build_dataset(
    entries: list[dict],
    augment: bool = False,
    shuffle: bool = False,
) -> tf.data.Dataset:
    paths = []
    labels = []
    for e in entries:
        p, lbl = parse_label(e)
        paths.append(p)
        labels.append(lbl)

    def _gen():
        for p, lbl in zip(paths, labels):
            yield p, lbl

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_types=(tf.string, tf.float32),
        output_shapes=((), (2,)),
    )

    def _load(path, lbl):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img = tf.cast(img, tf.float32)
        return img, lbl

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(512)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model() -> keras.Model:
    """MobileNetV2 regression model predicting (cx, cy) in [0, 1]."""
    inputs = keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), name="image")

    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    backbone = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        alpha=ALPHA,
    )
    backbone.trainable = False
    x = backbone(x, training=False)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(HEAD_UNITS, activation="swish")(x)
    x = keras.layers.Dropout(HEAD_DROPOUT)(x)
    outputs = keras.layers.Dense(2, activation="sigmoid", name="center_xy")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="center_model")
    return model


def main():
    os.makedirs(str(RUN_DIR), exist_ok=True)

    all_entries = load_metadata(METADATA_PATH)
    # Merge train + val to maximise training data; keep test for eval
    train_entries = [e for e in all_entries if e["split"] in ("train", "val")]
    test_entries = [e for e in all_entries if e["split"] == "test"]

    print(f"Train (train+val): {len(train_entries)}  Test: {len(test_entries)}")

    train_ds = build_dataset(train_entries, augment=True, shuffle=True)
    test_ds = build_dataset(test_entries, augment=False)

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    model.summary()

    # No validation set — monitor training loss instead
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(RUN_DIR / "best_model.keras"),
            monitor="mae",
            mode="min",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="mae",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
        ),
    ]

    # Phase 1: warmup with frozen backbone
    print("\n--- Phase 1: Warmup (backbone frozen) ---")
    model.fit(
        train_ds,
        epochs=EPOCHS_WARMUP,
        callbacks=callbacks,
        verbose=2,
    )

    # Phase 2: fine-tune with backbone unfrozen
    print("\n--- Phase 2: Fine-tune (backbone unfrozen) ---")
    for layer in model.layers:
        if hasattr(layer, "trainable"):
            layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE * 0.1),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    model.fit(
        train_ds,
        initial_epoch=EPOCHS_WARMUP,
        epochs=EPOCHS_WARMUP + EPOCHS_FINETUNE,
        callbacks=callbacks,
        verbose=2,
    )

    # Load best weights and evaluate on test set
    model = keras.models.load_model(str(RUN_DIR / "best_model.keras"))

    test_loss, test_mae = model.evaluate(test_ds, verbose=0)
    print(f"\n--- Test set results ---")
    print(f"Test loss (MSE): {test_loss:.6f}")
    print(f"Test MAE (norm): {test_mae:.5f}")
    print(f"Test MAE (pixels): {test_mae * IMAGE_WIDTH:.2f} px")

    # Per-sample errors on test set
    all_preds = model.predict(test_ds)
    all_labels = np.stack([lbl.numpy() for _, lbl in test_ds.unbatch()])
    errors_px = np.abs(all_preds - all_labels) * IMAGE_WIDTH
    print(f"\nPer-axis errors (pixels):")
    print(f"  cx MAE: {errors_px[:, 0].mean():.2f} px  (std={errors_px[:, 0].std():.2f})")
    print(f"  cy MAE: {errors_px[:, 1].mean():.2f} px  (std={errors_px[:, 1].std():.2f})")
    print(f"  Euclidean MAE: {np.mean(np.sqrt(np.sum(errors_px**2, axis=1))):.2f} px")

    # Compare: how often does the model beat a fixed inner-dial centre?
    inner_cx_norm = 112.0 / IMAGE_WIDTH
    inner_cy_norm = 100.0 / IMAGE_HEIGHT
    fixed_errors_px = np.abs(all_labels - np.array([inner_cx_norm, inner_cy_norm])) * IMAGE_WIDTH
    fixed_mean_eucl = np.mean(np.sqrt(np.sum(fixed_errors_px**2, axis=1)))
    model_eucl = np.mean(np.sqrt(np.sum(errors_px**2, axis=1)))
    print(f"\nvs fixed inner-dial centre ({112}, {100}):")
    print(f"  Fixed centre mean Eucl. err: {fixed_mean_eucl:.2f} px")
    print(f"  Model    mean Eucl. err: {model_eucl:.2f} px")
    improvement = (fixed_mean_eucl - model_eucl) / fixed_mean_eucl * 100
    print(f"  Improvement: {improvement:.1f}%")

    # Save final model and run summary
    model.save(str(RUN_DIR / "final_model.keras"))
    summary = {
        "run_name": RUN_NAME,
        "train_samples": len(train_entries),
        "test_samples": len(test_entries),
        "test_mse": float(test_loss),
        "test_mae_norm": float(test_mae),
        "test_mae_px": float(test_mae * IMAGE_WIDTH),
        "test_cx_mae_px": float(errors_px[:, 0].mean()),
        "test_cy_mae_px": float(errors_px[:, 1].mean()),
        "test_euclidean_mae_px": float(model_eucl),
        "fixed_centre_euclidean_mae_px": float(fixed_mean_eucl),
        "improvement_vs_fixed_pct": float(improvement),
        "alpha": ALPHA,
        "head_units": HEAD_UNITS,
    }
    with open(str(RUN_DIR / "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RUN_DIR}")


if __name__ == "__main__":
    main()
