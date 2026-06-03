#!/usr/bin/env python3
"""Fine-tune the centre model on board captures only, then export int8."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 8
EPOCHS = 150
LEARNING_RATE = 1e-5
PATIENCE = 20

BASE_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "center_model_20260531_193518"
    / "best_model.keras"
)
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed_crops"
METADATA_PATH = DATA_DIR / "metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "deployment" / "center_model_v2_boardft_int8"
RUN_DIR = PROJECT_ROOT / "artifacts" / "training" / "center_model_v2_boardft"
DEPLOY_TFLITE = OUTPUT_DIR / "model_int8.tflite"


def _is_board(entry: dict) -> bool:
    return entry["image_path"].startswith("images/capture_")


def _augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Augment — same rotation + colour as the original training script."""
    image = tf.image.random_brightness(image, max_delta=0.20)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.image.random_hue(image, max_delta=0.08)

    angle_rad = tf.random.uniform([], -0.04, 0.04)
    c = tf.cos(angle_rad)
    s = tf.sin(angle_rad)
    cx_img = tf.constant(IMAGE_WIDTH / 2.0, tf.float32)
    cy_img = tf.constant(IMAGE_HEIGHT / 2.0, tf.float32)
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
    paths = [str(DATA_DIR / e["image_path"]) for e in entries]
    labels_arr = np.array([
        [float(e["center_x_norm"]), float(e["center_y_norm"])]
        for e in entries
    ], dtype=np.float32)

    def _gen():
        for p, lbl in zip(paths, labels_arr):
            yield p, lbl

    ds = tf.data.Dataset.from_generator(
        _gen, output_types=(tf.string, tf.float32), output_shapes=((), (2,)),
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
        ds = ds.shuffle(128)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def main() -> None:
    os.makedirs(str(RUN_DIR), exist_ok=True)
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    # Load metadata
    with open(METADATA_PATH) as f:
        all_entries = json.load(f)

    board_entries = [e for e in all_entries if _is_board(e)]
    print(f"Board captures: {len(board_entries)}")

    # Split: 85% train, 15% val
    np.random.seed(42)
    indices = np.random.permutation(len(board_entries))
    split = int(0.85 * len(board_entries))
    train_idx = indices[:split]
    val_idx = indices[split:]
    train_entries = [board_entries[i] for i in train_idx]
    val_entries = [board_entries[i] for i in val_idx]
    print(f"  Train: {len(train_entries)}  Val: {len(val_entries)}")

    train_ds = build_dataset(train_entries, augment=True, shuffle=True)
    val_ds = build_dataset(val_entries, augment=False)

    # Load base model
    model = keras.models.load_model(str(BASE_MODEL_PATH))
    print(f"Loaded base model from {BASE_MODEL_PATH}")

    # Freeze backbone, only train head
    for layer in model.layers:
        if "mobilenetv2" in layer.name:
            layer.trainable = False
            print(f"  Frozen: {layer.name}")

    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(RUN_DIR / "best_model.keras"),
            monitor="val_mae",
            mode="min",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_mae",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=8,
            min_lr=1e-7,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    # Evaluate on board test set
    val_loss, val_mae = model.evaluate(val_ds, verbose=0)
    print(f"\nVal MAE (norm): {val_mae:.5f}  (pixels): {val_mae * IMAGE_WIDTH:.2f} px")

    # Save final model
    model.save(str(RUN_DIR / "final_model.keras"))

    # --- Export to TFLite int8 ---
    print("\n--- Exporting int8 TFLite ---")

    # Representative dataset from board captures (train portion)
    def _rep_data():
        for e in train_entries:
            path = str(DATA_DIR / e["image_path"])
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
            img = tf.cast(img, tf.float32)
            yield [np.expand_dims(img.numpy(), axis=0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(str(DEPLOY_TFLITE), "wb") as f:
        f.write(tflite_model)
    print(f"Saved: {DEPLOY_TFLITE} ({len(tflite_model)} bytes)")

    # Inspect Q-params
    interpreter = tf.lite.Interpreter(model_path=str(DEPLOY_TFLITE), num_threads=1)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    metadata = {
        "source": "center_model_v2_boardft",
        "base_model": str(BASE_MODEL_PATH),
        "board_train": len(train_entries),
        "board_val": len(val_entries),
        "val_mae_norm": float(val_mae),
        "val_mae_px": float(val_mae * IMAGE_WIDTH),
        "input_shape": [int(v) for v in in_det["shape"]],
        "output_shape": [int(v) for v in out_det["shape"]],
        "input_scale": float(in_det["quantization"][0]),
        "input_zero_point": int(in_det["quantization"][1]),
        "output_scale": float(out_det["quantization"][0]),
        "output_zero_point": int(out_det["quantization"][1]),
    }
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(str(meta_path), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
