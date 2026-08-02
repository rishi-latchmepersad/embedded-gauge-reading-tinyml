#!/usr/bin/env python3
"""Train a P2/FPN dense ellipse detector for small gauges.

The network keeps a stride-4 feature map alive from the encoder and fuses it
with coarser semantic features.  A separate heatmap head selects the face;
the box head only regresses center offsets and radii at that selected cell.
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
from train_ellipse_robust_384 import SEED, _block, load_zips, make_scale_augmented_training_set

IMAGE_SIZE = 384
GRID_SIZE = 96
GRID_VALUES = GRID_SIZE * GRID_SIZE


def configure_gpu() -> None:
    """Apply the project's 15 GB logical GPU limit before model creation."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def build_model() -> keras.Model:
    """Build a quantization-friendly P2/P3/P4 feature pyramid."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    # P2 is retained at 96x96 so a tiny face is not represented only by a
    # coarse 12x12 or 24x24 feature map.
    p1 = _block(inputs, 16, 2, "stem_down")
    p2 = _block(p1, 24, 2, "p2_down")
    p2 = _block(p2, 24, 1, "p2_refine")
    p3 = _block(p2, 40, 2, "p3_down")
    p3 = _block(p3, 40, 1, "p3_refine")
    p4 = _block(p3, 64, 2, "p4_down")
    p4 = _block(p4, 64, 1, "p4_refine")
    # why: lateral projections keep high-resolution location while the
    # upsampled branches inject enough context to distinguish gauge faces.
    f4 = layers.Conv2D(32, 1, padding="same", activation="relu", name="lat_p4")(p4)
    f3 = layers.Add(name="fpn_p3")([
        layers.Conv2D(32, 1, padding="same", activation="relu", name="lat_p3")(p3),
        layers.UpSampling2D(2, interpolation="nearest", name="up_p4")(f4),
    ])
    f2 = layers.Add(name="fpn_p2")([
        layers.Conv2D(32, 1, padding="same", activation="relu", name="lat_p2")(p2),
        layers.UpSampling2D(2, interpolation="nearest", name="up_p3")(f3),
    ])
    f2 = _block(f2, 32, 1, "fpn_p2_refine")
    heatmap = layers.Conv2D(1, 1, activation="sigmoid", name="face_heatmap")(f2)
    box = layers.Conv2D(4, 1, activation="sigmoid", name="ellipse_box")(f2)
    return keras.Model(
        inputs,
        layers.Flatten(name="p2_contract")(layers.Concatenate()([heatmap, box])),
        name="p2_fpn_ellipse_384",
    )


def make_targets(targets: np.ndarray) -> np.ndarray:
    """Create exact-positive center heatmaps and cell-local ellipse targets."""
    result = np.zeros((len(targets), GRID_SIZE, GRID_SIZE, 5), dtype=np.float32)
    yy, xx = np.mgrid[:GRID_SIZE, :GRID_SIZE]
    for index, (cx, cy, rx, ry) in enumerate(targets[:, :4]):
        gx, gy = cx * GRID_SIZE - 0.5, cy * GRID_SIZE - 0.5
        result[index, ..., 0] = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * 1.25**2))
        cell_x, cell_y = int(np.floor(gx)), int(np.floor(gy))
        if 0 <= cell_x < GRID_SIZE and 0 <= cell_y < GRID_SIZE:
            result[index, cell_y, cell_x, 0] = 1.0
            result[index, cell_y, cell_x, 1:] = [gx - cell_x, gy - cell_y, rx, ry]
    return result.reshape(len(targets), -1)


def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Optimize heatmap selection strongly and regress only positive cells."""
    true_values = tf.reshape(y_true, (-1, GRID_SIZE, GRID_SIZE, 5))
    pred_values = tf.reshape(y_pred, (-1, GRID_SIZE, GRID_SIZE, 5))
    true_heat, pred_heat = true_values[..., 0], tf.clip_by_value(pred_values[..., 0], 1e-4, 1.0 - 1e-4)
    positive = tf.cast(true_heat >= 0.95, tf.float32)
    negative_weight = tf.pow(1.0 - true_heat, 4.0)
    focal = -tf.reduce_sum(
        tf.pow(1.0 - pred_heat, 2.0) * tf.math.log(pred_heat) * positive
        + tf.pow(pred_heat, 2.0) * tf.math.log(1.0 - pred_heat) * negative_weight * (1.0 - positive),
        axis=(1, 2),
    ) / (tf.reduce_sum(positive, axis=(1, 2)) + 1.0)
    params = tf.reduce_sum(positive[..., None] * tf.abs(true_values[..., 1:] - pred_values[..., 1:]), axis=(1, 2, 3))
    params /= tf.reduce_sum(positive, axis=(1, 2)) + 1e-6
    return tf.reduce_mean(focal + 10.0 * params)


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export the QAT graph as an int8-only TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([image[None].astype(np.float32)] for image in images[:512])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model: Path, images: np.ndarray) -> np.ndarray:
    """Run the int8 model and restore its five dense channels."""
    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    predictions = []
    for image in images:
        quantized = np.clip(np.round(image / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        predictions.append((raw - out_zero) * out_scale)
    return np.asarray(predictions, dtype=np.float32).reshape(-1, GRID_SIZE, GRID_SIZE, 5)


def decode(values: np.ndarray) -> np.ndarray:
    """Select the heatmap peak and decode its local center and radii."""
    decoded = []
    for value in values:
        y, x = np.unravel_index(np.argmax(value[..., 0]), value[..., 0].shape)
        dx, dy, rx, ry = value[y, x, 1:]
        decoded.append([(x + 0.5 + dx) / GRID_SIZE, (y + 0.5 + dy) / GRID_SIZE, max(float(rx), 1e-3), max(float(ry), 1e-3)])
    return np.asarray(decoded, dtype=np.float32)


def main() -> None:
    """Train, quantize, export, and score the P2/FPN detector."""
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
