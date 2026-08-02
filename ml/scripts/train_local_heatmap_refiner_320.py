#!/usr/bin/env python3
"""Train a high-resolution local heatmap refiner for coarse gauge proposals."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_coarse_fine_ellipse_224 import stage1_decode
from train_ellipse_mask_640_center import IMAGE_SIZE, predict_int8 as predict_stage1
from train_ellipse_robust_384 import SEED, _block, load_zips

LOCAL_SIZE = 320
GRID_SIZE = 80
GRID_VALUES = GRID_SIZE * GRID_SIZE
OUTPUT_VALUES = GRID_VALUES * 3 + 4
STAGE1_MODEL = Path("artifacts/gauge_ellipse_mask_center_scaleconf_384_aux_v1/model_int8.tflite")


def configure_gpu() -> None:
    """Limit TensorFlow to the project's 15 GB GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])


def build_model() -> keras.Model:
    """Build a compact 320-to-80 local center heatmap network."""
    layers = keras.layers
    inputs = keras.Input((LOCAL_SIZE, LOCAL_SIZE, 1), name="local_crop")
    skips: list[tf.Tensor] = []
    x = inputs
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = _block(x, filters, 2, f"local_enc{stage}_down")
        x = _block(x, filters, 1, f"local_enc{stage}_refine")
        skips.append(x)
    for stage, filters, skip_index in ((0, 48, 3), (1, 32, 2), (2, 24, 1)):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"local_up{stage}")(x)
        x = layers.Concatenate(name=f"local_join{stage}")([x, skips[skip_index]])
        x = _block(x, filters, 1, f"local_dec{stage}")
    heatmap = layers.Flatten(name="local_heat_flat")(layers.Conv2D(1, 1, activation="sigmoid", name="local_heatmap")(x))
    offset = layers.Flatten(name="local_offset_flat")(layers.Conv2D(2, 1, activation="sigmoid", name="local_offset")(x))
    pooled = layers.GlobalAveragePooling2D(name="local_gap")(x)
    geometry = layers.Dense(4, activation="sigmoid", name="local_geometry")(layers.Dense(32, activation="relu")(pooled))
    return keras.Model(inputs, layers.Concatenate(name="local_contract")([heatmap, offset, geometry]), name="local_heatmap_refiner_320")


def crop_image(image: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop a normalized square proposal and return its source coordinates."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    ix1, iy1 = int(np.floor(x1 * width)), int(np.floor(y1 * height))
    ix2, iy2 = int(np.ceil(x2 * width)), int(np.ceil(y2 * height))
    side = max(ix2 - ix1, iy2 - iy1, 1)
    canvas = np.zeros((side, side), dtype=np.float32)
    sx1, sy1, sx2, sy2 = max(0, ix1), max(0, iy1), min(width, ix1 + side), min(height, iy1 + side)
    dx, dy = sx1 - ix1, sy1 - iy1
    canvas[dy:dy + sy2 - sy1, dx:dx + sx2 - sx1] = image[sy1:sy2, sx1:sx2, 0]
    crop = cv2.resize(canvas, (LOCAL_SIZE, LOCAL_SIZE), interpolation=cv2.INTER_AREA)[..., None]
    source = np.asarray([ix1 / width, iy1 / height, (ix1 + side) / width, (iy1 + side) / height], dtype=np.float32)
    return crop, source


def make_examples(images: np.ndarray, targets: np.ndarray, stage1_model: Path, repeats: int) -> tuple[np.ndarray, np.ndarray]:
    """Create mixed random and stage-one-error crops with local targets."""
    rng = np.random.default_rng(SEED + 17)
    proposals = stage1_decode(stage1_model, images)
    crops: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for image, target, proposal in zip(images, targets, proposals):
        for repeat in range(repeats):
            if repeat % 2 == 0:
                side = float(np.clip(2.2 * max(target[2], target[3]) * rng.uniform(0.8, 1.8), 0.16, 1.4))
                cx = float(target[0] + rng.normal(0.0, 0.18 * side))
                cy = float(target[1] + rng.normal(0.0, 0.18 * side))
            else:
                side = float(np.clip(2.2 * max(proposal[2], proposal[3]) * rng.uniform(0.9, 1.2), 0.18, 1.4))
                cx = float(proposal[0] + rng.normal(0.0, 0.04 * side))
                cy = float(proposal[1] + rng.normal(0.0, 0.04 * side))
            crop, source = crop_image(image, np.asarray([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], dtype=np.float32))
            sx, sy = source[2] - source[0], source[3] - source[1]
            local_cx, local_cy = (target[0] - source[0]) / sx, (target[1] - source[1]) / sy
            gx, gy = local_cx * GRID_SIZE - 0.5, local_cy * GRID_SIZE - 0.5
            target_map = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)
            yy, xx = np.mgrid[:GRID_SIZE, :GRID_SIZE]
            target_map[..., 0] = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * 1.5**2))
            cell_x, cell_y = int(np.floor(gx)), int(np.floor(gy))
            if 0 <= cell_x < GRID_SIZE and 0 <= cell_y < GRID_SIZE:
                target_map[cell_y, cell_x, 0] = 1.0
                target_map[cell_y, cell_x, 1:3] = [gx - cell_x, gy - cell_y]
            local_target = np.asarray([local_cx, local_cy, target[2] / sx, target[3] / sy], dtype=np.float32)
            crops.append(crop)
            # why: the model contract is channel-blocked (heatmap, offsets,
            # geometry), not pixel-interleaved; preserve that exact layout.
            labels.append(np.concatenate([target_map[..., 0].reshape(-1), target_map[..., 1:3].reshape(-1), local_target]))
    return np.asarray(crops, dtype=np.float32), np.clip(np.asarray(labels, dtype=np.float32), 0.0, 1.0)


def refiner_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Combine focal-like heatmap, masked offset, and geometry losses."""
    true_heat = tf.reshape(y_true[:, :GRID_VALUES], (-1, GRID_SIZE, GRID_SIZE))
    pred_heat = tf.reshape(y_pred[:, :GRID_VALUES], (-1, GRID_SIZE, GRID_SIZE))
    true_offset = tf.reshape(y_true[:, GRID_VALUES:GRID_VALUES * 3], (-1, GRID_SIZE, GRID_SIZE, 2))
    pred_offset = tf.reshape(y_pred[:, GRID_VALUES:GRID_VALUES * 3], (-1, GRID_SIZE, GRID_SIZE, 2))
    true_geometry, pred_geometry = y_true[:, GRID_VALUES * 3:], y_pred[:, GRID_VALUES * 3:]
    true_heat, pred_heat = tf.clip_by_value(true_heat, 1e-4, 1.0 - 1e-4), tf.clip_by_value(pred_heat, 1e-4, 1.0 - 1e-4)
    positive = tf.cast(true_heat >= 0.95, tf.float32)
    # why: focal weighting must use prediction error on positives; weighting
    # the positive term by (1-target) silently removed the center gradient.
    heat = -tf.reduce_mean((tf.pow(1.0 - pred_heat, 2.0) * tf.math.log(pred_heat) * positive) + (tf.pow(pred_heat, 2.0) * tf.math.log(1.0 - pred_heat) * (1.0 - true_heat) ** 4))
    offset = tf.reduce_sum(positive[..., None] * tf.abs(true_offset - pred_offset), axis=(1, 2, 3)) / (tf.reduce_sum(positive, axis=(1, 2)) + 1e-6)
    geometry = tf.reduce_mean(tf.keras.losses.huber(true_geometry, pred_geometry))
    return heat + 4.0 * tf.reduce_mean(offset) + 4.0 * geometry


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export the local refiner as a fully integer TFLite graph."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([image[None].astype(np.float32)] for image in images[: min(512, len(images))])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_local(model: Path, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the local int8 model and return heatmaps plus geometry."""
    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    ins, iz, os, oz = *inp["quantization"], *out["quantization"]
    maps: list[np.ndarray] = []
    for image in images:
        q = np.clip(np.round(image / ins + iz), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], q[None])
        interpreter.invoke()
        raw = (interpreter.get_tensor(out["index"])[0].astype(np.float32) - oz) * os
        maps.append(raw)
    values = np.asarray(maps, dtype=np.float32)
    # why: the flat TFLite contract stores complete heatmap and offset
    # channels consecutively; reshaping the first 19,200 values directly
    # would silently mix neighboring channels and destroy the argmax.
    heat = values[:, :GRID_VALUES].reshape(-1, GRID_SIZE, GRID_SIZE, 1)
    offset = values[:, GRID_VALUES:GRID_VALUES * 3].reshape(-1, GRID_SIZE, GRID_SIZE, 2)
    return np.concatenate([heat, offset], axis=-1), values[:, GRID_VALUES * 3:]


def refine_predictions(model: Path, stage1_model: Path, images: np.ndarray) -> np.ndarray:
    """Run coarse proposals, local heatmap refinement, and full-frame mapping."""
    proposals = stage1_decode(stage1_model, images)
    crops, boxes = [], []
    for image, proposal in zip(images, proposals):
        side = float(np.clip(2.2 * max(proposal[2], proposal[3]), 0.18, 1.4))
        crop, source = crop_image(image, np.asarray([proposal[0] - side / 2, proposal[1] - side / 2, proposal[0] + side / 2, proposal[1] + side / 2], dtype=np.float32))
        crops.append(crop)
        boxes.append(source)
    maps, geometry = predict_local(model, np.asarray(crops, dtype=np.float32))
    result = []
    for index, (local_map, local_geometry, source, proposal) in enumerate(zip(maps, geometry, boxes, proposals)):
        y, x = np.unravel_index(np.argmax(local_map[..., 0]), local_map[..., 0].shape)
        dx, dy = local_map[y, x, 1:3]
        local_center = np.asarray([(x + 0.5 + dx) / GRID_SIZE, (y + 0.5 + dy) / GRID_SIZE], dtype=np.float32)
        sx, sy = source[2] - source[0], source[3] - source[1]
        refined = np.asarray([source[0] + local_center[0] * sx, source[1] + local_center[1] * sy, local_geometry[2] * sx, local_geometry[3] * sy], dtype=np.float32)
        refined[:2] = 0.75 * proposal[:2] + 0.25 * refined[:2]
        result.append(refined)
    return np.asarray(result, dtype=np.float32)


def main() -> None:
    """Train, export, and evaluate the high-resolution local refiner."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-model", type=Path, default=STAGE1_MODEL)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--qat-epochs", type=int, default=2)
    parser.add_argument("--generic-limit", type=int, default=1000)
    parser.add_argument("--tiny-repeat", type=int, default=50)
    parser.add_argument("--board-repeat", type=int, default=5)
    parser.add_argument("--crop-repeats", type=int, default=4)
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    generic_images, generic_targets = load_zips(["train_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(["initial_temp_gauge/board_captures_1.zip"], labels=("temp_dial",))
    images = np.concatenate([generic_images[:args.generic_limit], np.repeat(tiny_images, args.tiny_repeat, axis=0), np.repeat(board_images, args.board_repeat, axis=0)])
    targets = np.concatenate([generic_targets[:args.generic_limit], np.repeat(tiny_targets, args.tiny_repeat, axis=0), np.repeat(board_targets, args.board_repeat, axis=0)])
    local_images, local_targets = make_examples(images, targets, args.stage1_model, args.crop_repeats)
    # why: the prior batch-2 run was dominated by Python/step overhead and
    # could not finish an epoch inside the job window; this compact model has
    # ample headroom for a larger batch without approaching the 50 GB RAM cap.
    dataset = tf.data.Dataset.from_tensor_slices((local_images, local_targets)).shuffle(len(local_images), seed=SEED).batch(16).prefetch(tf.data.AUTOTUNE)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=refiner_loss)
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=refiner_loss)
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, local_images, args.output / "model_int8.tflite")
    report: dict[str, object] = {"train_crops": int(len(local_images)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        resized = tf.image.resize(test_images, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
        predictions = np.concatenate([refine_predictions(args.output / "model_int8.tflite", args.stage1_model, resized), np.ones((len(test_targets), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
