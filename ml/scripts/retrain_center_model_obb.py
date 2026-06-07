"""Retrain centre detector: fine-tune old checkpoint on ALL 374 images × 5 crops.

Strategy:
- Use the existing tuned checkpoint (trained on 312 synthetic + 154 real CD-crops).
- Feed it all 1870 CD-crops (374 images × 5) with mild augmentation.
- Keep BGR data ordering (matching the checkpoint's training distribution).
- Two-phase: head first, then last 20% backbone.
"""

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
BATCH_SIZE = 32
EPOCHS_HEAD = 300
EPOCHS_FULL = 0  # disable — consistently regresses
LEARNING_RATE = 3e-4
PATIENCE = 60

BASE_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "center_model_20260531_193518"
    / "best_model.keras"
)
DATA_DIR = PROJECT_ROOT / "data" / "center_training_crops"
RUN_DIR = PROJECT_ROOT / "artifacts" / "training" / "center_model_v4_cdcrop"


def _augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
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
        fill_value=128, interpolation="BILINEAR",
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
        ds = ds.shuffle(2048)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def main() -> None:
    os.makedirs(str(RUN_DIR), exist_ok=True)

    with open(DATA_DIR / "metadata.json") as f:
        all_entries = json.load(f)

    train_entries = [e for e in all_entries if e["split"] == "train"]
    val_entries = [e for e in all_entries if e["split"] == "val"]

    print(f"Train: {len(train_entries)}")
    print(f"Val:   {len(val_entries)}")

    train_ds = build_dataset(train_entries, augment=True, shuffle=True)
    val_ds = build_dataset(val_entries)

    model = keras.models.load_model(str(BASE_MODEL_PATH))
    print(f"Loaded base model: {BASE_MODEL_PATH}")

    for layer in model.layers:
        if "mobilenetv2" in layer.name:
            layer.trainable = False
            print(f"  Frozen: {layer.name}")

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=EPOCHS_HEAD * (len(train_entries) // BATCH_SIZE),
        alpha=1e-3,
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(RUN_DIR / "best_model.keras"),
            monitor="val_mae", mode="min", save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_mae", patience=PATIENCE, restore_best_weights=True,
        ),
    ]

    print("\n--- Training (backbone frozen, cosine decay 3e-4→3e-7) ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD + EPOCHS_FULL,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(str(RUN_DIR / "final_model.keras"))
    val_loss, val_mae = model.evaluate(val_ds, verbose=0)
    px_mae = val_mae * IMAGE_WIDTH
    target_met = px_mae < 3.0
    print(f"\nVal MAE: {val_mae:.5f} norm = {px_mae:.2f} px")
    print(f"Target < 3 px: {'YES!' if target_met else f'NO ({px_mae:.1f} px)'}")

    # --- Export to TFLite int8 ---
    print("\n--- Exporting int8 TFLite ---")

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
    deploy_dir = PROJECT_ROOT / "artifacts" / "deployment" / "center_model_v4_cdcrop_int8"
    os.makedirs(str(deploy_dir), exist_ok=True)
    tflite_path = deploy_dir / "model_int8.tflite"
    with open(str(tflite_path), "wb") as f:
        f.write(tflite_model)
    print(f"Saved: {tflite_path} ({len(tflite_model)} bytes)")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    metadata = {
        "source": str(RUN_DIR / "best_model.keras"),
        "base_model": str(BASE_MODEL_PATH),
        "tflite_path": str(tflite_path.relative_to(PROJECT_ROOT)),
        "input_shape": [int(v) for v in in_det["shape"]],
        "output_shape": [int(v) for v in out_det["shape"]],
        "input_scale": float(in_det["quantization"][0]),
        "input_zero_point": int(in_det["quantization"][1]),
        "output_scale": float(out_det["quantization"][0]),
        "output_zero_point": int(out_det["quantization"][1]),
        "val_mae_norm": float(val_mae),
        "val_mae_px": float(px_mae),
        "target_met": bool(target_met),
        "train_entries": len(train_entries),
        "val_entries": len(val_entries),
        "data_source": "center_training_crops (374 images × 5 crops, mild aug)",
    }
    meta_path = deploy_dir / "metadata.json"
    with open(str(meta_path), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
