#!/usr/bin/env python3
"""Train a 512px local scalar ellipse refiner after the 384px proposal stage."""

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
from train_ellipse_robust_384 import SEED, _block, load_zips

LOCAL_SIZE = 512
STAGE1_MODEL = Path("artifacts/gauge_ellipse_mask_center_scaleconf_384_aux_v1/model_int8.tflite")


def configure_gpu() -> None:
    """Apply the project's 15 GB GPU cap."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])


def build_model() -> keras.Model:
    """Build a small scalar refiner whose high-resolution input is local."""
    layers = keras.layers
    inputs = keras.Input((LOCAL_SIZE, LOCAL_SIZE, 1), name="local_crop")
    x = inputs
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = _block(x, filters, 2, f"scalar512_enc{stage}_down")
        x = _block(x, filters, 1, f"scalar512_enc{stage}_refine")
    x = layers.GlobalAveragePooling2D(name="scalar512_gap")(x)
    x = layers.Dense(32, activation="relu", name="scalar512_hidden")(x)
    return keras.Model(inputs, layers.Dense(4, activation="sigmoid", name="scalar512_ellipse")(x), name="coarse_fine_scalar_512")


def crop_image(image: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop a normalized square proposal and resize it to 512px."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    ix1, iy1 = int(np.floor(x1 * width)), int(np.floor(y1 * height))
    ix2, iy2 = int(np.ceil(x2 * width)), int(np.ceil(y2 * height))
    side = max(ix2 - ix1, iy2 - iy1, 1)
    canvas = np.zeros((side, side), dtype=np.float32)
    sx1, sy1, sx2, sy2 = max(0, ix1), max(0, iy1), min(width, ix1 + side), min(height, iy1 + side)
    canvas[sy1 - iy1:sy2 - iy1, sx1 - ix1:sx2 - ix1] = image[sy1:sy2, sx1:sx2, 0]
    return cv2.resize(canvas, (LOCAL_SIZE, LOCAL_SIZE), interpolation=cv2.INTER_AREA)[..., None], np.asarray([ix1 / width, iy1 / height, (ix1 + side) / width, (iy1 + side) / height], dtype=np.float32)


def make_examples(images: np.ndarray, targets: np.ndarray, stage1: Path, repeats: int) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy proposal crops and absolute local ellipse targets."""
    proposals = stage1_decode(stage1, images)
    rng = np.random.default_rng(SEED)
    crops, labels = [], []
    for image, target, proposal in zip(images, targets, proposals):
        for _ in range(repeats):
            side = float(np.clip(2.2 * max(proposal[2], proposal[3]) * rng.uniform(0.8, 1.8), 0.16, 1.4))
            cx = float(proposal[0] + rng.normal(0.0, 0.12 * side))
            cy = float(proposal[1] + rng.normal(0.0, 0.12 * side))
            crop, source = crop_image(image, np.asarray([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], dtype=np.float32))
            sx, sy = source[2] - source[0], source[3] - source[1]
            crops.append(crop)
            labels.append([(target[0] - source[0]) / sx, (target[1] - source[1]) / sy, target[2] / sx, target[3] / sy])
    return np.asarray(crops, dtype=np.float32), np.clip(np.asarray(labels, dtype=np.float32), 0.0, 1.0)


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export the QAT local refiner as int8-only TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([image[None].astype(np.float32)] for image in images[:512])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_local(model: Path, crops: np.ndarray) -> np.ndarray:
    """Run the integer local refiner."""
    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    values = []
    for crop in crops:
        q = np.clip(np.round(crop / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], q[None])
        interpreter.invoke()
        values.append((interpreter.get_tensor(out["index"])[0].astype(np.float32) - out_zero) * out_scale)
    return np.asarray(values, dtype=np.float32)


def refine(model: Path, stage1: Path, images: np.ndarray) -> np.ndarray:
    """Run the proposal crop and map local scalar geometry to full frame."""
    proposals = stage1_decode(stage1, images)
    crops, sources = [], []
    for image, proposal in zip(images, proposals):
        side = float(np.clip(2.2 * max(proposal[2], proposal[3]), 0.18, 1.4))
        crop, source = crop_image(image, np.asarray([proposal[0] - side / 2, proposal[1] - side / 2, proposal[0] + side / 2, proposal[1] + side / 2], dtype=np.float32))
        crops.append(crop)
        sources.append(source)
    local = predict_local(model, np.asarray(crops, dtype=np.float32))
    result = []
    for value, source in zip(local, sources):
        sx, sy = source[2] - source[0], source[3] - source[1]
        result.append([source[0] + value[0] * sx, source[1] + value[1] * sy, value[2] * sx, value[3] * sy])
    return np.asarray(result, dtype=np.float32)


def main() -> None:
    """Train, export, and evaluate the 512px local refiner."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--qat-epochs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--generic-count", type=int, default=1000)
    parser.add_argument("--tiny-limit", type=int, default=1000000)
    parser.add_argument("--board-limit", type=int, default=1000000)
    parser.add_argument("--stage1-model", type=Path, default=STAGE1_MODEL)
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    generic_images, generic_targets = load_zips(["train_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(["initial_temp_gauge/board_captures_1.zip"], labels=("temp_dial",))
    # why: proposal generation is CPU-bound TFLite work; repeat only inside
    # local crop synthesis instead of running the proposer on duplicate pixels.
    images = np.concatenate([generic_images[:args.generic_count], tiny_images[:args.tiny_limit], board_images[:args.board_limit]])
    targets = np.concatenate([generic_targets[:args.generic_count], tiny_targets[:args.tiny_limit], board_targets[:args.board_limit]])
    crops, labels = make_examples(images, targets, args.stage1_model, args.repeats)
    dataset = tf.data.Dataset.from_tensor_slices((crops, labels)).shuffle(len(crops), seed=SEED).batch(4).prefetch(tf.data.AUTOTUNE)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(delta=0.05))
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=keras.losses.Huber(delta=0.05))
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, crops, args.output / "model_int8.tflite")
    report: dict[str, object] = {"train_crops": int(len(crops)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        test_images = tf.image.resize(test_images, (384, 384)).numpy()
        predictions = np.concatenate([refine(args.output / "model_int8.tflite", args.stage1_model, test_images), np.ones((len(test_targets), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
