#!/usr/bin/env python3
"""Train a QAT-friendly spatial face-mask model across all ellipse domains."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from eval_ellipse_all_test_sets import _load_zip
from train_ellipse_robust_384 import (
    BOARD_TRAIN_ZIPS,
    IMAGE_SIZE,
    SEED,
    _block,
    load_zips,
    make_scale_augmented_training_set,
)


MODEL_IMAGE_SIZE = 384
MASK_SIZE = 96


def configure_gpu() -> None:
    """Limit the training GPU to 15 GB so WSL retains desktop headroom."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def build_model() -> keras.Model:
    """Build an encoder-decoder whose output keeps face location explicit."""
    layers = keras.layers
    inputs = keras.Input((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 1), name="image")
    skips: list[tf.Tensor] = []
    x = inputs
    # why: skip features retain absolute position, while the bottleneck sees
    # the whole frame and handles gauges whose size varies by domain.
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = _block(x, filters, 2, f"enc{stage}_down")
        x = _block(x, filters, 1, f"enc{stage}_refine")
        skips.append(x)
    for stage, filters, skip_index in ((0, 48, 3), (1, 32, 2), (2, 24, 1)):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"up{stage}")(x)
        x = layers.Concatenate(name=f"join{stage}")([x, skips[skip_index]])
        x = _block(x, filters, 1, f"dec{stage}")
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="face_mask")(x)
    return keras.Model(inputs, outputs, name="ellipse_mask_all_domains_384")


def make_masks(targets: np.ndarray) -> np.ndarray:
    """Rasterize normalized ellipse labels onto the 96x96 output grid."""
    coords = (np.arange(MASK_SIZE, dtype=np.float32) + 0.5) / MASK_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    cx = targets[:, 0, None, None]
    cy = targets[:, 1, None, None]
    rx = np.maximum(targets[:, 2, None, None], 1e-3)
    ry = np.maximum(targets[:, 3, None, None], 1e-3)
    distance = ((xx[None] - cx) / rx) ** 2 + ((yy[None] - cy) / ry) ** 2
    return (distance <= 1.0).astype(np.float32)[..., None]


def mask_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Combine foreground-weighted BCE and Dice loss for small faces."""
    weights = 1.0 + 5.0 * y_true[..., 0]
    # why: tf_keras binary_crossentropy removes the singleton channel axis.
    bce = tf.reduce_mean(weights * keras.losses.binary_crossentropy(y_true, y_pred))
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    dice = tf.reduce_mean(1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0))
    return bce + dice


def box_from_masks(masks: np.ndarray) -> np.ndarray:
    """Decode thresholded masks into normalized center and radii."""
    boxes: list[np.ndarray] = []
    for mask in masks[..., 0]:
        active = mask >= 0.5
        if not np.any(active):
            active.flat[int(np.argmax(mask))] = True
        yy, xx = np.where(active)
        low = np.asarray([xx.min(), yy.min()], dtype=np.float32) / MASK_SIZE
        high = np.asarray([xx.max() + 1, yy.max() + 1], dtype=np.float32) / MASK_SIZE
        boxes.append(np.concatenate(((low + high) * 0.5, (high - low) * 0.5)))
    return np.asarray(boxes, dtype=np.float32)


def soft_box_from_masks(masks: np.ndarray) -> np.ndarray:
    """Decode masks with intensity-weighted moments for small or faint faces."""
    coords = (np.arange(MASK_SIZE, dtype=np.float32) + 0.5) / MASK_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    boxes: list[np.ndarray] = []
    for mask in masks[..., 0]:
        # why: subtracting a small background floor prevents dark predictions
        # around a tiny face from shifting the centroid toward frame center.
        weights = np.maximum(mask - 0.10, 0.0)
        total = float(weights.sum())
        if total <= 1e-6:
            boxes.append(box_from_masks(mask[None, ..., None])[0])
            continue
        cx = float((weights * xx).sum() / total)
        cy = float((weights * yy).sum() / total)
        rx = float(2.0 * np.sqrt(max((weights * (xx - cx) ** 2).sum() / total, 1e-6)))
        ry = float(2.0 * np.sqrt(max((weights * (yy - cy) ** 2).sum() / total, 1e-6)))
        boxes.append(np.asarray([cx, cy, rx, ry], dtype=np.float32))
    return np.asarray(boxes, dtype=np.float32)


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export a full-integer TFLite model using varied training frames."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield representative frames for activation calibration."""
        rng = np.random.default_rng(SEED)
        for index in rng.choice(len(images), min(512, len(images)), replace=False):
            yield [images[index : index + 1].astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model_path: Path, images: np.ndarray) -> np.ndarray:
    """Run the integer model and return dequantized masks."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    predictions = []
    for image in images:
        quantized = np.clip(np.round(image / input_scale + input_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"]).astype(np.float32)
        predictions.append((raw - output_zero) * output_scale)
    return np.concatenate(predictions, axis=0)


def main() -> None:
    """Train, quantize, export, and score the all-domain mask candidate."""
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--qat-epochs", type=int, default=12)
    parser.add_argument("--tiny-repeats", type=int, default=100)
    parser.add_argument("--board-repeats", type=int, default=3)
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    generic_images, generic_targets = load_zips(["train_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(BOARD_TRAIN_ZIPS, labels=("temp_dial",))
    # why: a bounded generic subset keeps CPU fallback runs practical while
    # retaining every tiny and board-domain example for the deployment target.
    generic_images = generic_images[:3000]
    generic_targets = generic_targets[:3000]
    images = np.concatenate(
        [generic_images, np.repeat(tiny_images, args.tiny_repeats, axis=0), np.repeat(board_images, args.board_repeats, axis=0)]
    )
    targets = np.concatenate(
        [generic_targets, np.repeat(tiny_targets, args.tiny_repeats, axis=0), np.repeat(board_targets, args.board_repeats, axis=0)]
    )
    images, targets = make_scale_augmented_training_set(images, targets)
    images = tf.image.resize(images, [MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE]).numpy()
    masks = make_masks(targets)
    print("training", images.shape, masks.shape)

    dataset = tf.data.Dataset.from_tensor_slices((images, masks)).shuffle(len(images), seed=SEED).batch(32).prefetch(tf.data.AUTOTUNE)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=mask_loss)
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=mask_loss)
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images, args.output / "model_int8.tflite")

    report: dict[str, object] = {"train_samples": int(len(images)), "mask_size": MASK_SIZE, "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        test_images = tf.image.resize(test_images, [MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE]).numpy()
        masks = predict_int8(args.output / "model_int8.tflite", test_images)
        predicted = soft_box_from_masks(masks)
        center_error = np.linalg.norm((predicted[:, :2] - test_targets[:, :2]) * 640.0, axis=1)
        radius_error = np.linalg.norm((predicted[:, 2:4] - test_targets[:, 2:4]) * 640.0, axis=1)
        report["tests"][zip_name] = {
            "n": int(len(test_targets)),
            "center_mae_px": float(center_error.mean()),
            "center_pct_le_8px": float(np.mean(center_error <= 8.0)),
            "center_pct_le_16px": float(np.mean(center_error <= 16.0)),
            "radius_mae_px": float(radius_error.mean()),
            "radius_pct_le_8px": float(np.mean(radius_error <= 8.0)),
        }
        print(zip_name, report["tests"][zip_name])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
