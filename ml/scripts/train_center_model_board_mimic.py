#!/usr/bin/env python3
"""Train a MobileNetV2 center regressor on the board-mimic crop dataset."""

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
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedded_gauge_reading_tinyml.training import (  # noqa: E402
    _load_crop_with_weight_maybe_board_style,
)

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 16
WARMUP_EPOCHS = 4
FINETUNE_EPOCHS = 12
ALPHA = 0.35
HEAD_UNITS = 128
HEAD_DROPOUT = 0.30
LEARNING_RATE_WARMUP = 3e-4
LEARNING_RATE_FINETUNE = 3e-5
BOARD_CAPTURE_WEIGHT = 3.0
REVIEW_WEIGHT = 0.8
BOARD_STYLE_PROB = 0.5

DATA_DIR = PROJECT_ROOT / "data" / "center_training_board_mimic"
METADATA_PATH = DATA_DIR / "metadata.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "training"
DEPLOY_DIR = PROJECT_ROOT / "artifacts" / "deployment" / "center_model_board_mimic_int8"
RUN_NAME = f"center_model_board_mimic_{datetime.now():%Y%m%d_%H%M%S}"
RUN_DIR = ARTIFACTS_DIR / RUN_NAME


def _log(message: str) -> None:
    """Print a progress message immediately so long startup steps stay visible."""
    print(message, flush=True)


def load_metadata(path: Path) -> list[dict]:
    """Load the generated board-mimic metadata."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_label(entry: dict) -> tuple[str, np.ndarray, float, np.ndarray]:
    """Return the source path, center label, weight, and crop box for one row."""
    source_path = str(REPO_ROOT / entry["source_path"])
    center_xy = np.array(
        [float(entry["center_x_norm"]), float(entry["center_y_norm"])],
        dtype=np.float32,
    )
    crop_x_min = float(entry["crop_x_min"])
    crop_y_min = float(entry["crop_y_min"])
    crop_width = float(entry["crop_width"])
    crop_height = float(entry["crop_height"])
    crop_box_xyxy = np.array(
        [crop_x_min, crop_y_min, crop_x_min + crop_width, crop_y_min + crop_height],
        dtype=np.float32,
    )
    source_kind = str(entry["source_kind"])
    quality_flag = str(entry["quality_flag"])
    weight = BOARD_CAPTURE_WEIGHT if source_kind == "capture" else 1.0
    if quality_flag == "review":
        weight *= REVIEW_WEIGHT
    return source_path, center_xy, float(weight), crop_box_xyxy


def _augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply light photometric and rotation augmentation.

    The board-mimic crops already contain the OBB-centered geometry, so the
    augmentation stays modest and only changes the image content or rotates the
    crop around its center while keeping the label aligned.
    """
    image = tf.image.random_brightness(image, max_delta=0.18)
    image = tf.image.random_contrast(image, lower=0.70, upper=1.30)
    image = tf.image.random_saturation(image, lower=0.80, upper=1.20)
    image = tf.image.random_hue(image, max_delta=0.05)

    angle_rad = tf.random.uniform([], -0.035, 0.035)
    cos_theta = tf.cos(angle_rad)
    sin_theta = tf.sin(angle_rad)
    center_x = tf.constant(IMAGE_WIDTH / 2.0, tf.float32)
    center_y = tf.constant(IMAGE_HEIGHT / 2.0, tf.float32)
    translate_x = (1.0 - cos_theta) * center_x + sin_theta * center_y
    translate_y = -sin_theta * center_x + (1.0 - cos_theta) * center_y
    transform = [cos_theta, -sin_theta, translate_x, sin_theta, cos_theta, translate_y, 0.0, 0.0]
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=[IMAGE_HEIGHT, IMAGE_WIDTH],
        fill_value=0.0,
        interpolation="BILINEAR",
    )[0]

    label_x = label[0] * tf.constant(IMAGE_WIDTH, tf.float32) - center_x
    label_y = label[1] * tf.constant(IMAGE_HEIGHT, tf.float32) - center_y
    rot_x = cos_theta * label_x - sin_theta * label_y
    rot_y = sin_theta * label_x + cos_theta * label_y
    label = tf.stack(
        [
            (rot_x + center_x) / tf.constant(IMAGE_WIDTH, tf.float32),
            (rot_y + center_y) / tf.constant(IMAGE_HEIGHT, tf.float32),
        ]
    )

    return tf.clip_by_value(image, 0.0, 255.0), label


def build_dataset(
    entries: list[dict],
    *,
    augment: bool = False,
    shuffle: bool = False,
    use_sample_weights: bool = False,
) -> tf.data.Dataset:
    """Build a tf.data pipeline for the board-mimic center dataset."""
    paths: list[str] = []
    labels: list[np.ndarray] = []
    weights: list[float] = []
    crop_boxes: list[np.ndarray] = []
    for entry in entries:
        path, label, weight, crop_box_xyxy = parse_label(entry)
        paths.append(path)
        labels.append(label)
        weights.append(weight)
        crop_boxes.append(crop_box_xyxy)

    ds = tf.data.Dataset.from_tensor_slices(
        (
            np.asarray(paths, dtype=np.str_),
            np.asarray(labels, dtype=np.float32),
            np.asarray(weights, dtype=np.float32),
            np.asarray(crop_boxes, dtype=np.float32),
        )
    )

    def _load(path: tf.Tensor, label: tf.Tensor, weight: tf.Tensor, crop_box_xyxy: tf.Tensor):
        image, label, weight = _load_crop_with_weight_maybe_board_style(
            path,
            label,
            crop_box_xyxy,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            weight,
            BOARD_STYLE_PROB,
        )
        # Keep the existing augmentation and normalization contract, which
        # expects float images in the 0..255 range before the model rescaling.
        image = tf.cast(image * 255.0, tf.float32)
        if use_sample_weights:
            return image, label, weight
        return image, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        if use_sample_weights:
            ds = ds.map(
                lambda image, label, weight: (*_augment(image, label), weight),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(entries), 1), reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model() -> keras.Model:
    """Build the MobileNetV2 center regressor used for the board-mimic run."""
    inputs = keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), name="image")
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(inputs)

    backbone = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        alpha=ALPHA,
    )
    backbone.trainable = False
    x = backbone(x, training=False)

    x = keras.layers.GlobalAveragePooling2D(name="center_gap")(x)
    x = keras.layers.Dense(HEAD_UNITS, activation="swish", name="center_dense")(x)
    x = keras.layers.Dropout(HEAD_DROPOUT, name="center_dropout")(x)
    outputs = keras.layers.Dense(2, activation="sigmoid", name="center_xy")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="center_model_board_mimic")
    setattr(model, "_mobilenet_backbone", backbone)
    return model


def _evaluate_by_source_kind(
    predictions: np.ndarray,
    labels: np.ndarray,
    entries: list[dict],
) -> dict[str, float]:
    """Compute per-source-type pixel MAE on the holdout split."""
    errors_px = np.abs(predictions - labels) * float(IMAGE_WIDTH)
    summary: dict[str, float] = {}
    for source_kind in ("capture", "pxl"):
        mask = np.array([str(entry["source_kind"]) == source_kind for entry in entries], dtype=bool)
        if not np.any(mask):
            continue
        summary[f"{source_kind}_count"] = float(mask.sum())
        summary[f"{source_kind}_center_mae_px"] = float(np.mean(errors_px[mask]))
    summary["overall_center_mae_px"] = float(np.mean(errors_px))
    summary["overall_center_cx_mae_px"] = float(np.mean(errors_px[:, 0]))
    summary["overall_center_cy_mae_px"] = float(np.mean(errors_px[:, 1]))
    return summary


def main() -> None:
    """Run the warmup + fine-tune board-mimic training job."""
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(DEPLOY_DIR, exist_ok=True)

    _log("Loading board-mimic metadata...")
    all_entries = load_metadata(METADATA_PATH)
    train_entries = [entry for entry in all_entries if str(entry["split"]) == "train"]
    val_entries = [entry for entry in all_entries if str(entry["split"]) == "val"]
    test_entries = [entry for entry in all_entries if str(entry["split"]) == "test"]

    _log("=" * 60)
    _log("Board-Mimic Center Training")
    _log(f"  Run dir: {RUN_DIR}")
    _log(f"  Train: {len(train_entries)}  Val: {len(val_entries)}  Test: {len(test_entries)}")
    _log("=" * 60)

    _log("Building tf.data pipelines...")
    train_ds = build_dataset(train_entries, augment=True, shuffle=True, use_sample_weights=True)
    val_ds = build_dataset(val_entries, augment=False, use_sample_weights=False)
    test_ds = build_dataset(test_entries, augment=False, use_sample_weights=False)

    _log("Building MobileNetV2 center model...")
    model = build_model()
    backbone: keras.Model = getattr(model, "_mobilenet_backbone")
    model.summary(print_fn=lambda line: _log(line))

    warmup_callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(RUN_DIR / "best_model.keras"),
            monitor="val_mae",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(RUN_DIR / "training_log.csv")),
    ]

    _log("--- Phase 1: warmup (backbone frozen) ---")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_WARMUP),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=WARMUP_EPOCHS,
        callbacks=warmup_callbacks,
        verbose=2,
    )
    _log("Warmup fit complete.")

    _log("--- Phase 2: fine-tune (backbone unfrozen) ---")
    backbone.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINETUNE),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    finetune_callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(RUN_DIR / "best_model.keras"),
            monitor="val_mae",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_mae",
            mode="min",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(RUN_DIR / "training_log_finetune.csv")),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=WARMUP_EPOCHS,
        epochs=WARMUP_EPOCHS + FINETUNE_EPOCHS,
        callbacks=finetune_callbacks,
        verbose=2,
    )
    _log("Fine-tune complete.")

    _log("Evaluating holdout set...")
    best_model = keras.models.load_model(str(RUN_DIR / "best_model.keras"))
    test_loss, test_mae = best_model.evaluate(test_ds, verbose=0)
    _log("--- Holdout results ---")
    _log(f"Test loss (MSE): {test_loss:.6f}")
    _log(f"Test MAE (norm): {test_mae:.6f}")
    _log(f"Test MAE (px): {test_mae * IMAGE_WIDTH:.2f}")

    _log("Computing holdout source summary...")
    test_predictions = best_model.predict(test_ds, verbose=0)
    test_labels = np.stack([label.numpy() for _, label in test_ds.unbatch()], axis=0)
    holdout_summary = _evaluate_by_source_kind(test_predictions, test_labels, test_entries)
    _log("Holdout source summary:")
    for key, value in holdout_summary.items():
        _log(f"  {key}: {value:.4f}")

    _log("Saving final Keras model...")
    best_model.save(str(RUN_DIR / "final_model.keras"))

    summary = {
        "run_name": RUN_NAME,
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "test_samples": len(test_entries),
        "warmup_epochs": WARMUP_EPOCHS,
        "finetune_epochs": FINETUNE_EPOCHS,
        "alpha": ALPHA,
        "head_units": HEAD_UNITS,
        "head_dropout": HEAD_DROPOUT,
        "learning_rate_warmup": LEARNING_RATE_WARMUP,
        "learning_rate_finetune": LEARNING_RATE_FINETUNE,
        "test_mse": float(test_loss),
        "test_mae_norm": float(test_mae),
        "test_mae_px": float(test_mae * IMAGE_WIDTH),
        **holdout_summary,
    }
    with (RUN_DIR / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _log("--- Exporting int8 TFLite ---")

    def _representative_dataset():
        for entry in train_entries[: min(len(train_entries), 200)]:
            image_path = DATA_DIR / entry["image_path"]
            image = tf.io.read_file(str(image_path))
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
            image = tf.cast(image, tf.float32)
            yield [np.expand_dims(image.numpy(), axis=0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    tflite_path = DEPLOY_DIR / "model_int8.tflite"
    with tflite_path.open("wb") as handle:
        handle.write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    deploy_metadata = {
        "source_model": str(RUN_DIR / "best_model.keras"),
        "tflite_path": str(tflite_path.relative_to(PROJECT_ROOT)),
        "input_shape": [int(v) for v in input_details["shape"]],
        "output_shape": [int(v) for v in output_details["shape"]],
        "input_scale": float(input_details["quantization"][0]),
        "input_zero_point": int(input_details["quantization"][1]),
        "output_scale": float(output_details["quantization"][0]),
        "output_zero_point": int(output_details["quantization"][1]),
        "test_mae_px": float(test_mae * IMAGE_WIDTH),
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "test_samples": len(test_entries),
    }
    with (DEPLOY_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(deploy_metadata, handle, indent=2)

    _log(f"Saved TFLite model: {tflite_path}")
    _log(f"Saved deployment metadata: {DEPLOY_DIR / 'metadata.json'}")
    _log("Done.")


if __name__ == "__main__":
    main()
