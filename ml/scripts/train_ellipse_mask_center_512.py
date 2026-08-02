#!/usr/bin/env python3
"""Train a higher-resolution 512x512 QAT ellipse-center model."""

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

IMAGE_SIZE = 512
GRID_SIZE = 128
GRID_VALUES = GRID_SIZE * GRID_SIZE
CHANNELS = 1 + 1 + 2  # mask, center heatmap, center offset.


def configure_gpu() -> None:
    """Apply the fixed 15 GB TensorFlow GPU limit."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])


def build_model() -> keras.Model:
    """Build a QAT-friendly 512-to-128 encoder-decoder."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    skips: list[tf.Tensor] = []
    x = inputs
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = _block(x, filters, 2, f"hr512_enc{stage}_down")
        x = _block(x, filters, 1, f"hr512_enc{stage}_refine")
        skips.append(x)
    for stage, filters, skip_index in ((0, 48, 3), (1, 32, 2), (2, 24, 1)):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"hr512_up{stage}")(x)
        x = layers.Concatenate(name=f"hr512_join{stage}")([x, skips[skip_index]])
        x = _block(x, filters, 1, f"hr512_dec{stage}")
    mask = layers.Conv2D(1, 1, activation="sigmoid", name="face_mask")(x)
    heat = layers.Conv2D(1, 1, activation="sigmoid", name="center_heatmap")(x)
    offset = layers.Conv2D(2, 1, activation="sigmoid", name="center_offset")(x)
    pooled = layers.GlobalAveragePooling2D(name="geometry_gap")(x)
    geometry = layers.Dense(4, activation="sigmoid", name="geometry")(layers.Dense(32, activation="relu")(pooled))
    return keras.Model(inputs, layers.Concatenate(name="ellipse_contract")([layers.Flatten()(mask), layers.Flatten()(heat), layers.Flatten()(offset), geometry]), name="ellipse_mask_center_512")


def make_targets(targets: np.ndarray) -> np.ndarray:
    """Rasterize mask, center heatmap, center offset, and ellipse geometry."""
    coords = (np.arange(GRID_SIZE, dtype=np.float32) + 0.5) / GRID_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    cx, cy = targets[:, 0, None, None], targets[:, 1, None, None]
    rx, ry = np.maximum(targets[:, 2, None, None], 1e-3), np.maximum(targets[:, 3, None, None], 1e-3)
    distance = ((xx[None] - cx) / rx) ** 2 + ((yy[None] - cy) / ry) ** 2
    masks = (distance <= 1.0).astype(np.float32).reshape(len(targets), GRID_VALUES)
    heat = np.zeros((len(targets), GRID_SIZE, GRID_SIZE), dtype=np.float32)
    offset = np.zeros((len(targets), GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)
    for index, (center_x, center_y, _, _) in enumerate(targets[:, :4]):
        gx, gy = center_x * GRID_SIZE - 0.5, center_y * GRID_SIZE - 0.5
        heat[index] = np.exp(-((np.arange(GRID_SIZE)[None] - gx) ** 2 + (np.arange(GRID_SIZE)[:, None] - gy) ** 2) / (2.0 * 2.0**2))
        cell_x, cell_y = int(np.floor(gx)), int(np.floor(gy))
        if 0 <= cell_x < GRID_SIZE and 0 <= cell_y < GRID_SIZE:
            heat[index, cell_y, cell_x] = 1.0
            offset[index, cell_y, cell_x] = [gx - cell_x, gy - cell_y]
    return np.concatenate([masks, heat.reshape(len(targets), GRID_VALUES), offset.reshape(len(targets), 2 * GRID_VALUES), targets[:, :4]], axis=1).astype(np.float32)


def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Prioritize center localization while retaining face-mask geometry."""
    true_mask = tf.reshape(y_true[:, :GRID_VALUES], (-1, GRID_SIZE, GRID_SIZE, 1))
    pred_mask = tf.reshape(y_pred[:, :GRID_VALUES], (-1, GRID_SIZE, GRID_SIZE, 1))
    heat_start, offset_start = GRID_VALUES, 2 * GRID_VALUES
    true_heat = tf.reshape(y_true[:, heat_start:offset_start], (-1, GRID_SIZE, GRID_SIZE, 1))
    pred_heat = tf.reshape(y_pred[:, heat_start:offset_start], (-1, GRID_SIZE, GRID_SIZE, 1))
    true_offset = tf.reshape(y_true[:, offset_start:offset_start + 2 * GRID_VALUES], (-1, GRID_SIZE, GRID_SIZE, 2))
    pred_offset = tf.reshape(y_pred[:, offset_start:offset_start + 2 * GRID_VALUES], (-1, GRID_SIZE, GRID_SIZE, 2))
    true_geometry = y_true[:, offset_start + 2 * GRID_VALUES:]
    pred_geometry = y_pred[:, offset_start + 2 * GRID_VALUES:]
    weights = 1.0 + 5.0 * true_mask[..., 0]
    bce = tf.reduce_mean(weights * keras.losses.binary_crossentropy(true_mask, pred_mask))
    intersection = tf.reduce_sum(true_mask * pred_mask, axis=(1, 2, 3))
    denominator = tf.reduce_sum(true_mask + pred_mask, axis=(1, 2, 3))
    dice = tf.reduce_mean(1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0))
    heat = tf.reduce_mean(keras.losses.binary_crossentropy(true_heat, pred_heat))
    positive = tf.cast(true_heat[..., 0] >= 0.95, tf.float32)[..., None]
    offset = tf.reduce_sum(positive * tf.abs(true_offset - pred_offset)) / (tf.reduce_sum(positive) * 2.0 + 1e-6)
    geometry = tf.reduce_mean(keras.losses.huber(true_geometry, pred_geometry))
    return bce + dice + 3.0 * heat + 8.0 * offset + 12.0 * geometry


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export the QAT model as int8-only TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([image[None].astype(np.float32)] for image in images[:512])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model: Path, images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the 512 model and return mask, heatmap, and geometry outputs."""
    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    masks, heats, geometries = [], [], []
    for image in images:
        q = np.clip(np.round(image / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], q[None])
        interpreter.invoke()
        values = (interpreter.get_tensor(out["index"])[0].astype(np.float32) - out_zero) * out_scale
        masks.append(values[:GRID_VALUES])
        heats.append(values[GRID_VALUES:2 * GRID_VALUES])
        geometries.append(values[2 * GRID_VALUES + 2 * GRID_VALUES:])
    return np.asarray(masks).reshape(-1, GRID_SIZE, GRID_SIZE, 1), np.asarray(heats).reshape(-1, GRID_SIZE, GRID_SIZE, 1), np.asarray(geometries)


def decode(masks: np.ndarray, heats: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    """Decode center heatmap with geometry radii."""
    outputs = []
    for mask, heat, geom in zip(masks, heats, geometry):
        y, x = np.unravel_index(np.argmax(heat[..., 0]), heat[..., 0].shape)
        outputs.append([(x + 0.5) / GRID_SIZE, (y + 0.5) / GRID_SIZE, max(float(geom[2]), 1e-3), max(float(geom[3]), 1e-3)])
    return np.asarray(outputs, dtype=np.float32)


def main() -> None:
    """Train, quantize, export, and evaluate the 512 model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--qat-epochs", type=int, default=2)
    parser.add_argument("--generic-count", type=int, default=1000)
    parser.add_argument("--tiny-repeats", type=int, default=50)
    parser.add_argument("--board-repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
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
    # why: the shared dataset loader keeps the canonical 384px storage format;
    # only this experiment changes the model input contract to 512px.
    resized_images = tf.image.resize(images, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
    contract = make_targets(targets)
    dataset = tf.data.Dataset.from_tensor_slices((resized_images, contract)).shuffle(len(images), seed=SEED).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss)
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=loss)
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, resized_images, args.output / "model_int8.tflite")
    report: dict[str, object] = {"train_samples": int(len(images)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        test_images = tf.image.resize(test_images, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
        masks, heats, geometry = predict_int8(args.output / "model_int8.tflite", test_images)
        predictions = np.concatenate([decode(masks, heats, geometry), np.ones((len(test_targets), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
