#!/usr/bin/env python3
"""Train a 640-pixel QAT center heatmap plus scalar radius model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_ellipse_robust_384 import BOARD_TRAIN_ZIPS, load_zips, make_scale_augmented_training_set
from train_ellipse_scalar_640 import IMAGE_SIZE, resize_cpu


HEATMAP_SIZE = 160
HEATMAP_VALUES = HEATMAP_SIZE * HEATMAP_SIZE


def configure_gpu() -> None:
    """Limit TensorFlow to 15 GB on the host GPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def build_model() -> keras.Model:
    """Build an encoder-decoder with a high-resolution center heatmap."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = inputs
    skips: list[tf.Tensor] = []
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        for block, stride in enumerate((2, 1)):
            x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"enc{stage}_{block}_conv")(x)
            x = layers.BatchNormalization(epsilon=1e-3, name=f"enc{stage}_{block}_bn")(x)
            x = layers.ReLU(name=f"enc{stage}_{block}_relu")(x)
        skips.append(x)
    geometry = layers.GlobalAveragePooling2D(name="geometry_gap")(x)
    geometry = layers.Dense(64, activation="relu", name="geometry_shared")(geometry)
    geometry = layers.Dense(4, activation="sigmoid", name="geometry")(geometry)
    for stage, filters, skip_index in ((0, 48, 3), (1, 32, 2), (2, 24, 1)):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"up{stage}")(x)
        x = layers.Concatenate(name=f"join{stage}")([x, skips[skip_index]])
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"dec{stage}_conv")(x)
        x = layers.BatchNormalization(epsilon=1e-3, name=f"dec{stage}_bn")(x)
        x = layers.ReLU(name=f"dec{stage}_relu")(x)
    heatmap = layers.Conv2D(1, 1, activation="sigmoid", name="center_heatmap")(x)
    heatmap = layers.Flatten(name="heatmap_flatten")(heatmap)
    outputs = layers.Concatenate(name="ellipse_contract")([heatmap, geometry])
    return keras.Model(inputs, outputs, name="ellipse_center_heatmap_640")


def make_heatmaps(targets: np.ndarray) -> np.ndarray:
    """Rasterize Gaussian center targets on the 160x160 grid."""
    coords = (np.arange(HEATMAP_SIZE, dtype=np.float32) + 0.5) / HEATMAP_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    cx = targets[:, 0, None, None]
    cy = targets[:, 1, None, None]
    # why: a fixed normalized sigma gives tiny and large faces equal center
    # localization supervision while keeping soft-argmax subpixel-friendly.
    sigma = 0.018
    heatmaps = np.exp(-((xx[None] - cx) ** 2 + (yy[None] - cy) ** 2) / (2.0 * sigma**2))
    return heatmaps.astype(np.float32).reshape(len(targets), HEATMAP_VALUES)


class HeatmapLoss(keras.losses.Loss):
    """Balance center heatmap BCE with scalar geometry supervision."""

    def __init__(self, geometry_weight: float = 5.0, **kwargs: object) -> None:
        """Initialize the scalar branch weight."""
        super().__init__(**kwargs)
        self.geometry_weight = geometry_weight

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return weighted heatmap BCE and robust geometry error."""
        true_heatmap = y_true[:, :HEATMAP_VALUES]
        pred_heatmap = y_pred[:, :HEATMAP_VALUES]
        geometry_true = y_true[:, HEATMAP_VALUES:]
        geometry_pred = y_pred[:, HEATMAP_VALUES:]
        weights = 1.0 + 15.0 * true_heatmap
        # why: Keras treats a flat heatmap as one multi-class vector; use
        # elementwise BCE so every spatial cell receives a foreground weight.
        clipped = tf.clip_by_value(pred_heatmap, 1e-6, 1.0 - 1e-6)
        elementwise_bce = -(true_heatmap * tf.math.log(clipped) + (1.0 - true_heatmap) * tf.math.log(1.0 - clipped))
        heatmap = tf.reduce_mean(weights * elementwise_bce, axis=-1)
        error = tf.abs(geometry_true - geometry_pred)
        quadratic = tf.minimum(error, 0.05)
        linear = error - quadratic
        geometry = tf.reduce_sum(0.5 * tf.square(quadratic) + 0.05 * linear, axis=-1)
        return heatmap + self.geometry_weight * geometry

    def get_config(self) -> dict[str, object]:
        """Return serializable loss configuration."""
        return {**super().get_config(), "geometry_weight": self.geometry_weight}


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export a fully integer heatmap model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield representative images for activation calibration."""
        rng = np.random.default_rng(42)
        for index in rng.choice(len(images), min(512, len(images)), replace=False):
            yield [images[index : index + 1].astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model_path: Path, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run int8 inference and return heatmaps plus scalar geometry."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    values = []
    for image in images:
        quantized = np.clip(np.round(image / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        values.append((raw - out_zero) * out_scale)
    result = np.asarray(values, dtype=np.float32)
    return result[:, :HEATMAP_VALUES].reshape(-1, HEATMAP_SIZE, HEATMAP_SIZE, 1), result[:, HEATMAP_VALUES:]


def decode_centers(heatmaps: np.ndarray) -> np.ndarray:
    """Decode heatmaps with a background-subtracted soft centroid."""
    coords = (np.arange(HEATMAP_SIZE, dtype=np.float32) + 0.5) / HEATMAP_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    centers = []
    for heatmap in heatmaps[..., 0]:
        weights = np.maximum(heatmap - 0.10, 0.0)
        total = max(float(weights.sum()), 1e-6)
        centers.append([(weights * xx).sum() / total, (weights * yy).sum() / total])
    return np.asarray(centers, dtype=np.float32)


def main() -> None:
    """Train, QAT-finetune, export, and evaluate the center-heatmap model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--qat-epochs", type=int, default=6)
    parser.add_argument("--tiny-repeats", type=int, default=40)
    parser.add_argument("--board-repeats", type=int, default=3)
    parser.add_argument("--tiny-only", action="store_true", help="Train the spatial branch only on tiny-domain images.")
    parser.add_argument("--board-only", action="store_true", help="Train the spatial branch only on labeled board captures.")
    args = parser.parse_args()
    configure_gpu()
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(BOARD_TRAIN_ZIPS, labels=("temp_dial",))
    generic_images, generic_targets = generic_images[:2000], generic_targets[:2000]
    if args.board_only:
        # why: board captures have a distinct optical/background domain; keeping
        # generic gradients out lets the high-resolution heatmap specialize in
        # the board center instead of learning the generic frame midpoint.
        images = np.repeat(board_images, args.board_repeats, axis=0)
        targets = np.repeat(board_targets, args.board_repeats, axis=0)
    elif args.tiny_only:
        # why: the routed high-resolution branch only serves tiny faces, so
        # remove large-domain gradients that can dilute its spatial signal.
        images = np.repeat(tiny_images, args.tiny_repeats, axis=0)
        targets = np.repeat(tiny_targets, args.tiny_repeats, axis=0)
    else:
        images = np.concatenate([generic_images, np.repeat(tiny_images, args.tiny_repeats, axis=0), np.repeat(board_images, args.board_repeats, axis=0)])
        targets = np.concatenate([generic_targets, np.repeat(tiny_targets, args.tiny_repeats, axis=0), np.repeat(board_targets, args.board_repeats, axis=0)])
    images, targets = make_scale_augmented_training_set(images, targets)
    images = resize_cpu(images)
    contract_targets = np.concatenate([make_heatmaps(targets), targets[:, :4]], axis=1).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((images, contract_targets)).shuffle(len(images), seed=42).batch(8).prefetch(tf.data.AUTOTUNE)
    print("training", images.shape, contract_targets.shape)

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=HeatmapLoss())
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=HeatmapLoss())
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images, args.output / "model_int8.tflite")

    report: dict[str, object] = {"train_samples": int(len(images)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        heatmaps, geometry = predict_int8(args.output / "model_int8.tflite", resize_cpu(test_images))
        centers = decode_centers(heatmaps)
        predictions = np.concatenate([centers, geometry[:, 2:4], np.ones((len(geometry), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
