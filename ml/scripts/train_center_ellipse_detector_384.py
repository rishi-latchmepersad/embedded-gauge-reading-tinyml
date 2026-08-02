#!/usr/bin/env python3
"""Train a dense CenterNet-style scene-level ellipse detector.

The detector predicts one gauge objectness heatmap and per-cell ellipse
parameters on a stride-4 grid.  Unlike a global regression head, every cell
can nominate a gauge, so translation and scale changes do not have to be
compressed into one global feature vector.
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
from train_ellipse_robust_384 import (
    SEED,
    _block,
    load_zips,
    make_scale_augmented_training_set,
)

IMAGE_SIZE = 384
GRID_SIZE = 96
GRID_VALUES = GRID_SIZE * GRID_SIZE
PARAM_VALUES = 6 * GRID_VALUES


def configure_gpu() -> None:
    """Limit TensorFlow to the project's 15 GB GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def build_model() -> keras.Model:
    """Build a compact stride-4 dense detector with no global pooling."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    skips: list[tf.Tensor] = []
    x = inputs
    # why: retain shallow spatial features because the tiny test gauges can
    # occupy only a small fraction of the scene.
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = _block(x, filters, 2, f"enc{stage}_down")
        x = _block(x, filters, 1, f"enc{stage}_refine")
        skips.append(x)
    for stage, filters, skip_index in ((0, 48, 3), (1, 32, 2), (2, 24, 1)):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"up{stage}")(x)
        x = layers.Concatenate(name=f"join{stage}")([x, skips[skip_index]])
        x = _block(x, filters, 1, f"dec{stage}")
    # why: the shared dense map keeps candidate selection spatially explicit;
    # the six channels are objectness, dx, dy, rx, ry, and a face quality.
    outputs = layers.Conv2D(6, 1, activation="sigmoid", name="ellipse_dense")(x)
    return keras.Model(inputs, layers.Flatten(name="ellipse_contract")(outputs), name="center_ellipse_detector_384")


def make_targets(targets: np.ndarray) -> np.ndarray:
    """Rasterize a Gaussian objectness target and center-cell ellipse values."""
    result = np.zeros((len(targets), GRID_SIZE, GRID_SIZE, 6), dtype=np.float32)
    yy, xx = np.mgrid[:GRID_SIZE, :GRID_SIZE]
    for index, (cx, cy, rx, ry) in enumerate(targets[:, :4]):
        gx, gy = cx * GRID_SIZE - 0.5, cy * GRID_SIZE - 0.5
        # why: a Gaussian target gives nearby cells useful gradients while the
        # focal loss still forces a single sharp winning location.
        result[index, ..., 0] = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * 1.5**2))
        cell_x, cell_y = int(np.floor(gx)), int(np.floor(gy))
        if 0 <= cell_x < GRID_SIZE and 0 <= cell_y < GRID_SIZE:
            # why: the positive-cell mask must be exact even when the target
            # center lies near a cell corner and the Gaussian peak is lower.
            result[index, cell_y, cell_x, 0] = 1.0
            result[index, cell_y, cell_x, 1:] = [gx - cell_x, gy - cell_y, rx, ry, 1.0]
    return result.reshape(len(targets), -1)


def _focal_heatmap_loss(true_heat: tf.Tensor, pred_heat: tf.Tensor) -> tf.Tensor:
    """Apply CenterNet-style positive/negative focal weighting."""
    positive = tf.cast(true_heat >= 0.95, tf.float32)
    negative = 1.0 - positive
    negative_weight = tf.pow(1.0 - true_heat, 4.0)
    pred_heat = tf.clip_by_value(pred_heat, 1e-4, 1.0 - 1e-4)
    positive_loss = -tf.pow(1.0 - pred_heat, 2.0) * tf.math.log(pred_heat) * positive
    negative_loss = -tf.pow(pred_heat, 2.0) * tf.math.log(1.0 - pred_heat) * negative_weight * negative
    normalizer = tf.reduce_sum(positive, axis=(1, 2)) + 1.0
    return tf.reduce_mean(tf.reduce_sum(positive_loss + negative_loss, axis=(1, 2)) / normalizer)


def detector_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Train heatmap selection strongly and regress parameters at positives."""
    true_values = tf.reshape(y_true, (-1, GRID_SIZE, GRID_SIZE, 6))
    pred_values = tf.reshape(y_pred, (-1, GRID_SIZE, GRID_SIZE, 6))
    heat_loss = _focal_heatmap_loss(true_values[..., 0], pred_values[..., 0])
    positive = tf.cast(true_values[..., 0] >= 0.95, tf.float32)[..., None]
    parameter_loss = tf.reduce_sum(
        positive * tf.abs(true_values[..., 1:5] - pred_values[..., 1:5]),
        axis=(1, 2, 3),
    ) / (tf.reduce_sum(positive, axis=(1, 2, 3)) + 1e-6)
    # why: masked L1 avoids a framework-dependent singleton-axis expansion
    # while still teaching the winning cell that it contains a face.
    quality_loss = tf.reduce_sum(
        positive[..., 0] * tf.abs(true_values[..., 5] - pred_values[..., 5]), axis=(1, 2)
    ) / (tf.reduce_sum(positive[..., 0], axis=(1, 2)) + 1e-6)
    return heat_loss + 8.0 * tf.reduce_mean(parameter_loss) + tf.reduce_mean(quality_loss)


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export the detector as a fully integer TFLite graph."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield bounded calibration samples at the deployment resolution."""
        rng = np.random.default_rng(SEED)
        for index in rng.choice(len(images), min(512, len(images)), replace=False):
            sample = tf.image.resize(images[index : index + 1], (IMAGE_SIZE, IMAGE_SIZE))
            yield [sample.numpy().astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model_path: Path, images: np.ndarray) -> np.ndarray:
    """Run the int8 detector and return normalized dense predictions."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    predictions: list[np.ndarray] = []
    for image in images:
        quantized = np.clip(np.round(image / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        predictions.append((raw - out_zero) * out_scale)
    return np.asarray(predictions, dtype=np.float32).reshape(-1, GRID_SIZE, GRID_SIZE, 6)


def decode(predictions: np.ndarray) -> np.ndarray:
    """Decode each dense map by selecting its highest objectness cell."""
    decoded: list[list[float]] = []
    for values in predictions:
        y, x = np.unravel_index(np.argmax(values[..., 0]), values[..., 0].shape)
        dx, dy, rx, ry = values[y, x, 1:5]
        decoded.append([(x + 0.5 + dx) / GRID_SIZE,
                        (y + 0.5 + dy) / GRID_SIZE,
                        max(float(rx), 1e-3), max(float(ry), 1e-3)])
    return np.asarray(decoded, dtype=np.float32)


def main() -> None:
    """Train, quantize, export, and evaluate the dense detector."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--qat-epochs", type=int, default=2)
    parser.add_argument("--tiny-repeats", type=int, default=50)
    parser.add_argument("--board-repeats", type=int, default=5)
    parser.add_argument("--generic-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    generic_images, generic_targets = load_zips(["train_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(["initial_temp_gauge/board_captures_1.zip"], labels=("temp_dial",))
    images = np.concatenate([generic_images[:args.generic_count],
                             np.repeat(tiny_images, args.tiny_repeats, axis=0),
                             np.repeat(board_images, args.board_repeats, axis=0)])
    targets = np.concatenate([generic_targets[:args.generic_count],
                              np.repeat(tiny_targets, args.tiny_repeats, axis=0),
                              np.repeat(board_targets, args.board_repeats, axis=0)])
    images, targets = make_scale_augmented_training_set(images, targets)
    contract_targets = make_targets(targets)
    dataset = (
        tf.data.Dataset.from_tensor_slices((images, contract_targets))
        .shuffle(len(images), seed=SEED)
        .map(lambda x, y: (tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE)), y), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=detector_loss)
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=detector_loss)
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images[:256], args.output / "model_int8.tflite")
    report: dict[str, object] = {"train_samples": int(len(images)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        resized = tf.image.resize(test_images, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
        predictions = np.concatenate([decode(predict_int8(args.output / "model_int8.tflite", resized)),
                                      np.ones((len(test_targets), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
