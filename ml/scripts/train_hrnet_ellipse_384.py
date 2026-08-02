#!/usr/bin/env python3
"""Train a QAT HRNet-style multi-resolution gauge-face center model.

Unlike an encoder-decoder that reconstructs spatial detail after a bottleneck,
this model keeps a 96x96 high-resolution branch alive and repeatedly fuses it
with lower-resolution context before predicting the center heatmap.
"""

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

IMAGE_SIZE = 384
GRID_SIZE = 96
GRID_VALUES = GRID_SIZE * GRID_SIZE
SEED = 42


def configure_gpu() -> None:
    """Limit TensorFlow to the approved 15 GB GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])


def conv_block(x: tf.Tensor, filters: int, stride: int, name: str) -> tf.Tensor:
    """Apply a QAT/NPU-safe convolution, batch normalization, and ReLU."""
    layers = keras.layers
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(epsilon=1e-3, name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def fuse(high: tf.Tensor, low: tf.Tensor, name: str) -> tuple[tf.Tensor, tf.Tensor]:
    """Fuse 96x96 and 48x48 branches in both directions."""
    layers = keras.layers
    high_to_low = conv_block(high, 48, 2, f"{name}_high_to_low")
    low_out = layers.Add(name=f"{name}_low_add")([low, high_to_low])
    low_out = layers.ReLU(name=f"{name}_low_relu")(low_out)
    low_to_high = conv_block(low_out, 32, 1, f"{name}_low_to_high")
    low_to_high = layers.UpSampling2D(2, interpolation="nearest", name=f"{name}_up")(low_to_high)
    high_out = layers.Add(name=f"{name}_high_add")([high, low_to_high])
    high_out = layers.ReLU(name=f"{name}_high_relu")(high_out)
    return high_out, low_out


def build_model() -> keras.Model:
    """Build the two-level HRNet-style center and geometry predictor."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    # why: this stem reaches 96x96 once, then the high-resolution branch is
    # never discarded, preventing center precision from depending on decoder
    # upsampling after a 12x12 bottleneck.
    x = conv_block(inputs, 24, 2, "stem_192")
    high = conv_block(x, 32, 2, "stem_high_96")
    low = conv_block(high, 48, 2, "stem_low_48")
    high, low = fuse(high, low, "fuse0")
    high, low = fuse(high, low, "fuse1")
    high, low = fuse(high, low, "fuse2")
    heatmap = layers.Conv2D(1, 1, activation="sigmoid", name="center_heatmap")(high)
    heatmap = layers.Flatten(name="heatmap_flatten")(heatmap)
    geometry = layers.Concatenate(name="geometry_features")([
        layers.GlobalAveragePooling2D(name="high_gap")(high),
        layers.GlobalAveragePooling2D(name="low_gap")(low),
    ])
    geometry = layers.Dense(64, activation="relu", name="geometry_shared")(geometry)
    geometry = layers.Dense(4, activation="sigmoid", name="geometry")(geometry)
    return keras.Model(inputs, layers.Concatenate(name="contract")([heatmap, geometry]), name="hrnet_ellipse_384")


def make_targets(targets: np.ndarray) -> np.ndarray:
    """Create subpixel Gaussian center targets and append ellipse geometry."""
    coords = (np.arange(GRID_SIZE, dtype=np.float32) + 0.5) / GRID_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    cx, cy = targets[:, 0, None, None], targets[:, 1, None, None]
    # why: sigma spans about 1.5 output cells and keeps gradients around the
    # exact center even when the face center is between grid locations.
    heatmaps = np.exp(-((xx[None] - cx) ** 2 + (yy[None] - cy) ** 2) / (2.0 * 0.016**2))
    return np.concatenate([heatmaps.reshape(len(targets), GRID_VALUES), targets[:, :4]], axis=1).astype(np.float32)


class HRLoss(keras.losses.Loss):
    """Weight sparse center heatmaps and robustly supervise ellipse geometry."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return center localization plus scalar geometry loss."""
        true_map, pred_map = y_true[:, :GRID_VALUES], y_pred[:, :GRID_VALUES]
        true_geom, pred_geom = y_true[:, GRID_VALUES:], y_pred[:, GRID_VALUES:]
        weights = 1.0 + 18.0 * true_map
        pred_map = tf.clip_by_value(pred_map, 1e-5, 1.0 - 1e-5)
        bce = -(true_map * tf.math.log(pred_map) + (1.0 - true_map) * tf.math.log(1.0 - pred_map))
        heatmap = tf.reduce_mean(weights * bce, axis=-1)
        error = tf.abs(true_geom - pred_geom)
        geometry = tf.reduce_mean(tf.where(error < 0.04, 0.5 * tf.square(error) / 0.04, error - 0.02), axis=-1)
        return heatmap + 8.0 * geometry


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export the QAT graph as a fully integer TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield representative images for activation calibration."""
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
    """Run integer inference and dequantize the concatenated output."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    input_scale, input_zero = inp["quantization"]
    output_scale, output_zero = out["quantization"]
    values: list[np.ndarray] = []
    for image in images:
        quantized = np.clip(np.round(image / input_scale + input_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        values.append((raw - output_zero) * output_scale)
    return np.asarray(values, dtype=np.float32)


def decode(contract: np.ndarray) -> np.ndarray:
    """Decode the center heatmap using local soft-argmax plus scalar radii."""
    coords = (np.arange(GRID_SIZE, dtype=np.float32) + 0.5) / GRID_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    centers: list[list[float]] = []
    for row in contract:
        heatmap = np.clip(row[:GRID_VALUES].reshape(GRID_SIZE, GRID_SIZE), 0.0, 1.0)
        # why: a local window around the peak prevents weak background logits
        # from dragging the center toward the image mean.
        py, px = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
        local = np.zeros_like(heatmap)
        radius = 8
        local[max(0, py - radius):min(GRID_SIZE, py + radius + 1), max(0, px - radius):min(GRID_SIZE, px + radius + 1)] = 1.0
        weights = np.maximum(heatmap - 0.10, 0.0) * local
        total = max(float(weights.sum()), 1e-6)
        centers.append([float((weights * xx).sum() / total), float((weights * yy).sum() / total)])
    return np.concatenate([np.asarray(centers, dtype=np.float32), contract[:, GRID_VALUES:]], axis=1)


def main() -> None:
    """Train, QAT-finetune, export, and independently score all test sets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--qat-epochs", type=int, default=6)
    parser.add_argument("--tiny-repeats", type=int, default=60)
    parser.add_argument("--board-repeats", type=int, default=3)
    args = parser.parse_args()
    configure_gpu(); random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    generic_images, generic_targets = load_zips(["train_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(BOARD_TRAIN_ZIPS, labels=("temp_dial",))
    generic_images, generic_targets = generic_images[:3000], generic_targets[:3000]
    images = np.concatenate([generic_images, np.repeat(tiny_images, args.tiny_repeats, axis=0), np.repeat(board_images, args.board_repeats, axis=0)])
    targets = np.concatenate([generic_targets, np.repeat(tiny_targets, args.tiny_repeats, axis=0), np.repeat(board_targets, args.board_repeats, axis=0)])
    images, targets = make_scale_augmented_training_set(images, targets)
    contract_targets = make_targets(targets)
    dataset = tf.data.Dataset.from_tensor_slices((images, contract_targets)).shuffle(len(images), seed=SEED).batch(8).prefetch(tf.data.AUTOTUNE)
    print("training", images.shape, contract_targets.shape, flush=True)
    model = build_model(); model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=HRLoss()); model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model); qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=HRLoss()); qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True); qat.save_weights(args.output / "model_qat.weights.h5"); export_int8(qat, images, args.output / "model_int8.tflite")
    report: dict[str, object] = {"train_samples": int(len(images)), "grid_size": GRID_SIZE, "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        predictions = np.concatenate([decode(predict_int8(args.output / "model_int8.tflite", test_images)), np.ones((len(test_targets), 1), np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
