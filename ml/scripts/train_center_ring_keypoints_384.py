#!/usr/bin/env python3
"""Train a shared center-plus-rim heatmap model for ellipse localization."""

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
from train_ellipse_robust_384 import SEED, _block, load_zips, make_scale_augmented_training_set

IMAGE_SIZE = 384
GRID_SIZE = 96
POINTS = 9  # center followed by eight synthetic points sampled on the ellipse.
VALUES = GRID_SIZE * GRID_SIZE


def configure_gpu() -> None:
    """Apply the 15 GB GPU allocation limit before TensorFlow builds graphs."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])


def build_model() -> keras.Model:
    """Build a compact U-Net-like shared spatial keypoint detector."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    skips: list[tf.Tensor] = []
    x = inputs
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = _block(x, filters, 2, f"ring_enc{stage}_down")
        x = _block(x, filters, 1, f"ring_enc{stage}_refine")
        skips.append(x)
    for stage, filters, skip_index in ((0, 48, 3), (1, 32, 2), (2, 24, 1)):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"ring_up{stage}")(x)
        x = layers.Concatenate(name=f"ring_join{stage}")([x, skips[skip_index]])
        x = _block(x, filters, 1, f"ring_dec{stage}")
    heat = layers.Conv2D(POINTS, 1, activation="sigmoid", name="center_ring_heatmaps")(x)
    center_offset = layers.Conv2D(2, 1, activation="sigmoid", name="center_offset")(x)
    return keras.Model(inputs, layers.Flatten(name="center_ring_contract")(layers.Concatenate()([heat, center_offset])), name="center_ring_keypoints_384")


def make_targets(targets: np.ndarray) -> np.ndarray:
    """Rasterize center/rim Gaussian targets and a center-cell offset."""
    result = np.zeros((len(targets), GRID_SIZE, GRID_SIZE, POINTS + 2), dtype=np.float32)
    yy, xx = np.mgrid[:GRID_SIZE, :GRID_SIZE]
    for index, (cx, cy, rx, ry) in enumerate(targets[:, :4]):
        points = [(cx, cy)]
        # why: structural rim points make the center channel compete with the
        # actual face boundary, discouraging shortcuts on cluttered frames.
        for angle in np.linspace(0.0, 2.0 * np.pi, 9)[:-1]:
            points.append((cx + rx * np.cos(angle), cy + ry * np.sin(angle)))
        for point_index, (px, py) in enumerate(points):
            gx, gy = px * GRID_SIZE - 0.5, py * GRID_SIZE - 0.5
            sigma = 1.5 if point_index == 0 else 1.25
            result[index, ..., point_index] = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * sigma**2))
            cell_x, cell_y = int(np.floor(gx)), int(np.floor(gy))
            if point_index == 0 and 0 <= cell_x < GRID_SIZE and 0 <= cell_y < GRID_SIZE:
                result[index, cell_y, cell_x, point_index] = 1.0
                result[index, cell_y, cell_x, POINTS:] = [gx - cell_x, gy - cell_y]
    return result.reshape(len(targets), -1)


def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Use focal heatmap loss for nine points and masked center offset loss."""
    true = tf.reshape(y_true, (-1, GRID_SIZE, GRID_SIZE, POINTS + 2))
    pred = tf.reshape(y_pred, (-1, GRID_SIZE, GRID_SIZE, POINTS + 2))
    true_heat, pred_heat = true[..., :POINTS], tf.clip_by_value(pred[..., :POINTS], 1e-4, 1.0 - 1e-4)
    positive = tf.cast(true_heat >= 0.95, tf.float32)
    negative_weight = tf.pow(1.0 - true_heat, 4.0)
    focal = -tf.reduce_mean(tf.pow(1.0 - pred_heat, 2.0) * tf.math.log(pred_heat) * positive + tf.pow(pred_heat, 2.0) * tf.math.log(1.0 - pred_heat) * negative_weight * (1.0 - positive))
    center_positive = positive[..., :1]
    offset = tf.reduce_sum(center_positive * tf.abs(true[..., POINTS:] - pred[..., POINTS:])) / (tf.reduce_sum(center_positive) * 2.0 + 1e-6)
    return focal + 8.0 * offset


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export the QAT model as an int8-only TFLite graph."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([image[None].astype(np.float32)] for image in images[:512])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model: Path, images: np.ndarray) -> np.ndarray:
    """Run the integer model and restore channel-last heatmaps/offsets."""
    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    values = []
    for image in images:
        q = np.clip(np.round(image / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], q[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        values.append((raw - out_zero) * out_scale)
    return np.asarray(values, dtype=np.float32).reshape(-1, GRID_SIZE, GRID_SIZE, POINTS + 2)


def decode(values: np.ndarray) -> np.ndarray:
    """Decode center peak and estimate radii from the eight rim peaks."""
    outputs = []
    for value in values:
        cy, cx = np.unravel_index(np.argmax(value[..., 0]), value[..., 0].shape)
        dx, dy = value[cy, cx, POINTS:]
        center = np.asarray([(cx + 0.5 + dx) / GRID_SIZE, (cy + 0.5 + dy) / GRID_SIZE], dtype=np.float32)
        rim = []
        for point_index in range(1, POINTS):
            y, x = np.unravel_index(np.argmax(value[..., point_index]), value[..., point_index].shape)
            rim.append([(x + 0.5) / GRID_SIZE, (y + 0.5) / GRID_SIZE])
        rim_array = np.asarray(rim, dtype=np.float32)
        radii = np.mean(np.abs(rim_array - center[None]), axis=0)
        outputs.append([center[0], center[1], max(float(radii[0]), 1e-3), max(float(radii[1]), 1e-3)])
    return np.asarray(outputs, dtype=np.float32)


def main() -> None:
    """Train, export, and evaluate the center/rim keypoint detector."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--qat-epochs", type=int, default=2)
    parser.add_argument("--tiny-repeats", type=int, default=50)
    parser.add_argument("--board-repeats", type=int, default=5)
    parser.add_argument("--generic-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    generic_images, generic_targets = load_zips(["train_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(["initial_temp_gauge/board_captures_1.zip"], labels=("temp_dial",))
    images = np.concatenate([generic_images[:args.generic_count], np.repeat(tiny_images, args.tiny_repeats, axis=0), np.repeat(board_images, args.board_repeats, axis=0)])
    targets = np.concatenate([generic_targets[:args.generic_count], np.repeat(tiny_targets, args.tiny_repeats, axis=0), np.repeat(board_targets, args.board_repeats, axis=0)])
    images, targets = make_scale_augmented_training_set(images, targets)
    contract = make_targets(targets)
    dataset = tf.data.Dataset.from_tensor_slices((images, contract)).shuffle(len(images), seed=SEED).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss)
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=loss)
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images, args.output / "model_int8.tflite")
    report: dict[str, object] = {"train_samples": int(len(images)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        predictions = np.concatenate([decode(predict_int8(args.output / "model_int8.tflite", test_images)), np.ones((len(test_targets), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
