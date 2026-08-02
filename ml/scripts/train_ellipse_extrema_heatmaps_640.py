#!/usr/bin/env python3
"""Train a QAT four-extrema heatmap model for direct ellipse geometry."""

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


GRID_SIZE = 160
GRID_VALUES = GRID_SIZE * GRID_SIZE
HEATMAP_VALUES = 4 * GRID_VALUES


def configure_gpu() -> None:
    """Reserve at most 15 GB of the host GPU for training."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])


def build_model() -> keras.Model:
    """Build a shared encoder with four extrema heatmaps and geometry scalars."""
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
    heatmaps = layers.Conv2D(4, 1, activation="sigmoid", name="extrema_heatmaps")(x)
    heatmaps = layers.Flatten(name="extrema_flatten")(heatmaps)
    return keras.Model(inputs, layers.Concatenate(name="ellipse_extrema_contract")([heatmaps, geometry]), name="ellipse_extrema_heatmaps_640")


def make_heatmaps(targets: np.ndarray) -> np.ndarray:
    """Rasterize Gaussian left/right/top/bottom extrema targets."""
    coords = (np.arange(GRID_SIZE, dtype=np.float32) + 0.5) / GRID_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    points = np.stack(
        [
            targets[:, 0] - targets[:, 2],
            targets[:, 0] + targets[:, 2],
            targets[:, 0],
            targets[:, 0],
        ],
        axis=1,
    )
    point_y = np.stack([targets[:, 1], targets[:, 1], targets[:, 1] - targets[:, 3], targets[:, 1] + targets[:, 3]], axis=1)
    sigma = 0.012
    values = np.exp(-((xx[None, None] - points[:, :, None, None]) ** 2 + (yy[None, None] - point_y[:, :, None, None]) ** 2) / (2.0 * sigma**2))
    # why: Keras Flatten serializes the Conv2D output as [row, col, channel],
    # so pack the NumPy [sample, channel, row, col] tensor in that same order.
    return values.astype(np.float32).transpose(0, 2, 3, 1).reshape(len(targets), HEATMAP_VALUES)


class ExtremaLoss(keras.losses.Loss):
    """Balance four foreground-sparse heatmaps with scalar ellipse supervision."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return foreground-weighted heatmap BCE and robust geometry loss."""
        true_map = y_true[:, :HEATMAP_VALUES]
        pred_map = y_pred[:, :HEATMAP_VALUES]
        true_geometry = y_true[:, HEATMAP_VALUES:]
        pred_geometry = y_pred[:, HEATMAP_VALUES:]
        weights = 1.0 + 20.0 * true_map
        clipped = tf.clip_by_value(pred_map, 1e-6, 1.0 - 1e-6)
        bce = -(true_map * tf.math.log(clipped) + (1.0 - true_map) * tf.math.log(1.0 - clipped))
        heatmap = tf.reduce_mean(weights * bce, axis=-1)
        error = tf.abs(true_geometry - pred_geometry)
        quadratic = tf.minimum(error, 0.05)
        linear = error - quadratic
        geometry = tf.reduce_sum(0.5 * tf.square(quadratic) + 0.05 * linear, axis=-1)
        return heatmap + 5.0 * geometry


def export_int8(model: keras.Model, images: np.ndarray, destination: Path) -> None:
    """Export a fully integer TFLite model using representative frames."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield deterministic activation-calibration frames."""
        rng = np.random.default_rng(42)
        for index in rng.choice(len(images), min(512, len(images)), replace=False):
            yield [images[index : index + 1].astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    destination.write_bytes(converter.convert())


def predict_int8(model_path: Path, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the exported model and return four heatmaps plus scalar geometry."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    input_scale, input_zero = inp["quantization"]
    output_scale, output_zero = out["quantization"]
    values = []
    for image in images:
        quantized = np.clip(np.round(image / input_scale + input_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        values.append((raw - output_zero) * output_scale)
    values = np.asarray(values, dtype=np.float32)
    # why: undo Flatten's pixel-major layout before decoding each channel.
    maps = values[:, :HEATMAP_VALUES].reshape(-1, GRID_SIZE, GRID_SIZE, 4).transpose(0, 3, 1, 2)
    return maps, values[:, HEATMAP_VALUES:]


def decode(heatmaps: np.ndarray, floor: float = 0.20, power: float = 2.0) -> np.ndarray:
    """Decode extrema heatmaps into normalized center and radii."""
    coords = (np.arange(GRID_SIZE, dtype=np.float32) + 0.5) / GRID_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    result = []
    for maps in heatmaps:
        points = []
        for values in maps:
            weights = np.maximum(values - floor, 0.0) ** power
            total = max(float(weights.sum()), 1e-6)
            points.append([(weights * xx).sum() / total, (weights * yy).sum() / total])
        left, right, top, bottom = np.asarray(points, dtype=np.float32)
        result.append([(left[0] + right[0]) / 2.0, (top[1] + bottom[1]) / 2.0, (right[0] - left[0]) / 2.0, (bottom[1] - top[1]) / 2.0])
    return np.asarray(result, dtype=np.float32)


def main() -> None:
    """Train, QAT-finetune, export, and evaluate the extrema model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--qat-epochs", type=int, default=4)
    parser.add_argument("--tiny-repeats", type=int, default=40)
    parser.add_argument("--board-repeats", type=int, default=5)
    args = parser.parse_args()
    configure_gpu()
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(BOARD_TRAIN_ZIPS, labels=("temp_dial",))
    generic_images, generic_targets = generic_images[:2000], generic_targets[:2000]
    images = np.concatenate([generic_images, np.repeat(tiny_images, args.tiny_repeats, axis=0), np.repeat(board_images, args.board_repeats, axis=0)])
    targets = np.concatenate([generic_targets, np.repeat(tiny_targets, args.tiny_repeats, axis=0), np.repeat(board_targets, args.board_repeats, axis=0)])
    images, targets = make_scale_augmented_training_set(images, targets)
    images = resize_cpu(images)
    contract_targets = np.concatenate([make_heatmaps(targets), targets[:, :4]], axis=1).astype(np.float32)
    # why: four 160x160 heatmaps make the target tensor about 3.2 GB; keep
    # the source dataset on CPU so only small batches cross the GPU boundary.
    with tf.device("/CPU:0"):
        dataset = tf.data.Dataset.from_tensor_slices((images, contract_targets)).shuffle(len(images), seed=42).batch(2).prefetch(tf.data.AUTOTUNE)
    print("training", images.shape, contract_targets.shape)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=ExtremaLoss())
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=ExtremaLoss())
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images, args.output / "model_int8.tflite")
    report: dict[str, object] = {"train_samples": int(len(images)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        heatmaps, geometry = predict_int8(args.output / "model_int8.tflite", resize_cpu(test_images))
        decoded = decode(heatmaps)
        predictions = np.concatenate([decoded[:, :2], geometry[:, 2:4], np.ones((len(geometry), 1), np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
