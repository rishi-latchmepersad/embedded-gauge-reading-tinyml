#!/usr/bin/env python3
"""Train a proposal-residual ellipse refiner with a QAT-compatible contract."""

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
from train_coarse_fine_ellipse_224 import crop_image, stage1_decode
from train_ellipse_robust_384 import SEED, _block, load_zips

LOCAL_SIZE = 224
STAGE1_MODEL = Path("artifacts/gauge_ellipse_mask_center_scaleconf_384_aux_v1/model_int8.tflite")


def configure_gpu() -> None:
    """Limit TensorFlow to the project's 15 GB GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])


def build_model() -> keras.Model:
    """Build the compact local residual refiner."""
    layers = keras.layers
    inputs = keras.Input((LOCAL_SIZE, LOCAL_SIZE, 1), name="local_crop")
    x = inputs
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = _block(x, filters, 2, f"residual_enc{stage}_down")
        x = _block(x, filters, 1, f"residual_enc{stage}_refine")
    x = layers.GlobalAveragePooling2D(name="residual_gap")(x)
    x = layers.Dense(32, activation="relu", name="residual_hidden")(x)
    # Output is 0.5 plus twice the normalized correction; sigmoid keeps the
    # graph fully QAT-friendly while preserving a signed residual around 0.5.
    outputs = layers.Dense(4, activation="sigmoid", name="ellipse_residual")(x)
    return keras.Model(inputs, outputs, name="residual_coarse_fine_ellipse_224")


def make_examples(images: np.ndarray, targets: np.ndarray, stage1: Path, repeats: int) -> tuple[np.ndarray, np.ndarray]:
    """Create crops and encode target-minus-proposal residuals."""
    proposals = stage1_decode(stage1, images)
    rng = np.random.default_rng(SEED + 9)
    crops: list[np.ndarray] = []
    labels: list[list[float]] = []
    for image, target, proposal in zip(images, targets, proposals):
        for _ in range(repeats):
            side = float(np.clip(2.2 * max(proposal[2], proposal[3]) * rng.uniform(0.9, 1.2), 0.18, 1.4))
            cx = float(proposal[0] + rng.normal(0.0, 0.04 * side))
            cy = float(proposal[1] + rng.normal(0.0, 0.04 * side))
            crop, source = crop_image(image, np.asarray([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], dtype=np.float32))
            sx, sy = source[2] - source[0], source[3] - source[1]
            proposal_local = np.asarray([(proposal[0] - source[0]) / sx, (proposal[1] - source[1]) / sy, proposal[2] / sx, proposal[3] / sy], dtype=np.float32)
            target_local = np.asarray([(target[0] - source[0]) / sx, (target[1] - source[1]) / sy, target[2] / sx, target[3] / sy], dtype=np.float32)
            crops.append(crop)
            # why: residual targets make the refiner translation-stable and
            # leave the coarse stage in control when local evidence is weak.
            labels.append(np.clip(0.5 + 2.0 * (target_local - proposal_local), 0.0, 1.0).tolist())
    return np.asarray(crops, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export an int8-only TFLite residual refiner."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([image[None].astype(np.float32)] for image in images[:512])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_local(model: Path, crops: np.ndarray) -> np.ndarray:
    """Run the int8 residual refiner on local crops."""
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
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        values.append((raw - out_zero) * out_scale)
    return np.asarray(values, dtype=np.float32)


def refine(model: Path, stage1: Path, images: np.ndarray) -> np.ndarray:
    """Decode proposal-relative residuals and map the ellipse to full frame."""
    proposals = stage1_decode(stage1, images)
    crops: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    for image, proposal in zip(images, proposals):
        side = float(np.clip(2.2 * max(proposal[2], proposal[3]), 0.18, 1.4))
        crop, source = crop_image(image, np.asarray([proposal[0] - side / 2, proposal[1] - side / 2, proposal[0] + side / 2, proposal[1] + side / 2], dtype=np.float32))
        crops.append(crop)
        sources.append(source)
    values = predict_local(model, np.asarray(crops, dtype=np.float32))
    outputs = []
    for proposal, source, value in zip(proposals, sources, values):
        sx, sy = source[2] - source[0], source[3] - source[1]
        proposal_local = np.asarray([(proposal[0] - source[0]) / sx, (proposal[1] - source[1]) / sy, proposal[2] / sx, proposal[3] / sy], dtype=np.float32)
        local = proposal_local + (value - 0.5) / 2.0
        outputs.append([source[0] + local[0] * sx, source[1] + local[1] * sy, max(local[2] * sx, 1e-3), max(local[3] * sy, 1e-3)])
    return np.asarray(outputs, dtype=np.float32)


def main() -> None:
    """Train, export, and evaluate the residual refiner."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-model", type=Path, default=STAGE1_MODEL)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--qat-epochs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    generic_images, generic_targets = load_zips(["train_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(["initial_temp_gauge/board_captures_1.zip"], labels=("temp_dial",))
    images = np.concatenate([generic_images[:1000], np.repeat(tiny_images, 50, axis=0), np.repeat(board_images, 5, axis=0)])
    targets = np.concatenate([generic_targets[:1000], np.repeat(tiny_targets, 50, axis=0), np.repeat(board_targets, 5, axis=0)])
    local_images, local_targets = make_examples(images, targets, args.stage1_model, args.repeats)
    dataset = tf.data.Dataset.from_tensor_slices((local_images, local_targets)).shuffle(len(local_images), seed=SEED).batch(16).prefetch(tf.data.AUTOTUNE)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(delta=0.05))
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=keras.losses.Huber(delta=0.05))
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, local_images, args.output / "model_int8.tflite")
    report: dict[str, object] = {"train_crops": int(len(local_images)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        resized = tf.image.resize(test_images, (384, 384)).numpy()
        predictions = np.concatenate([refine(args.output / "model_int8.tflite", args.stage1_model, resized), np.ones((len(test_targets), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
