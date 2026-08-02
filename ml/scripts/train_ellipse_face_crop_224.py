#!/usr/bin/env python3
"""Train a universal ellipse refiner on normalized gauge-face crops.

The crop is deliberately larger than the annotated face.  At deployment a
coarse detector supplies the crop, while this model spends all 224 pixels on
the face and predicts center plus radii in crop coordinates.
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
from train_ellipse_scalar_640 import WeightedLoss


IMAGE_SIZE = 224


def configure_gpu() -> None:
    """Limit TensorFlow to the project-approved 15 GB GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def build_model() -> keras.Model:
    """Build a compact crop refiner with an absolute-coordinate output head."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = inputs
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        for block, stride in enumerate((2, 1)):
            x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"s{stage}_{block}_conv")(x)
            x = layers.BatchNormalization(epsilon=1e-3, name=f"s{stage}_{block}_bn")(x)
            x = layers.ReLU(name=f"s{stage}_{block}_relu")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="shared")(x)
    return keras.Model(inputs, layers.Dense(5, activation="sigmoid", name="ellipse")(x), name="ellipse_face_crop_224")


def make_face_crops(
    images: np.ndarray,
    targets: np.ndarray,
    jitter: bool,
    seed: int,
    padding: float = 2.8,
    jitter_fraction: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a padded square around each face and transform its ellipse labels."""
    rng = np.random.default_rng(seed)
    boxes: list[list[float]] = []
    transformed: list[np.ndarray] = []
    for target in targets:
        cx, cy, rx, ry = target[:4]
        # why: 2.8 radii leaves context for a coarse crop error but makes tiny
        # faces occupy enough pixels for the refinement network.
        side = max(padding * float(rx), padding * float(ry), 0.16)
        crop_cx, crop_cy = float(cx), float(cy)
        if jitter:
            crop_cx += float(rng.normal(0.0, jitter_fraction * side))
            crop_cy += float(rng.normal(0.0, jitter_fraction * side))
            side *= float(rng.uniform(0.90, 1.10))
        left, top = crop_cx - side / 2.0, crop_cy - side / 2.0
        boxes.append([top, left, top + side, left + side])
        transformed.append(np.array([(cx - left) / side, (cy - top) / side, rx / side, ry / side, target[4]], dtype=np.float32))
    with tf.device("/CPU:0"):
        cropped = tf.image.crop_and_resize(images, np.asarray(boxes, dtype=np.float32), np.arange(len(images)), (IMAGE_SIZE, IMAGE_SIZE), method="bilinear", extrapolation_value=0.0).numpy()
    return cropped.astype(np.float32), np.asarray(transformed, dtype=np.float32)


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export a fully integer TFLite crop refiner."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield representative crop images for quantization calibration."""
        rng = np.random.default_rng(42)
        for index in rng.choice(len(images), min(512, len(images)), replace=False):
            yield [images[index : index + 1].astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model_path: Path, images: np.ndarray) -> np.ndarray:
    """Run an int8 crop refiner and return dequantized ellipse values."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    values: list[np.ndarray] = []
    for image in images:
        quantized = np.clip(np.round(image / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        values.append((raw - out_zero) * out_scale)
    return np.asarray(values, dtype=np.float32)


def restore_predictions(predictions: np.ndarray, targets: np.ndarray, padding: float = 2.8) -> np.ndarray:
    """Convert crop-relative predictions back to the original normalized frame."""
    restored: list[list[float]] = []
    for prediction, target in zip(predictions, targets):
        cx, cy, rx, ry = target[:4]
        side = max(padding * float(rx), padding * float(ry), 0.16)
        left, top = cx - side / 2.0, cy - side / 2.0
        restored.append([left + prediction[0] * side, top + prediction[1] * side, prediction[2] * side, prediction[3] * side, prediction[4]])
    return np.asarray(restored, dtype=np.float32)


def main() -> None:
    """Train, QAT-finetune, export, and score the face-crop refiner."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--qat-epochs", type=int, default=4)
    parser.add_argument("--generic-limit", type=int, default=2000)
    parser.add_argument("--tiny-repeats", type=int, default=30)
    parser.add_argument("--board-repeats", type=int, default=2)
    parser.add_argument("--crop-padding", type=float, default=3.5)
    parser.add_argument("--jitter-fraction", type=float, default=0.12)
    args = parser.parse_args()
    configure_gpu()
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(BOARD_TRAIN_ZIPS, labels=("temp_dial",))
    generic_images, generic_targets = generic_images[: args.generic_limit], generic_targets[: args.generic_limit]
    images = np.concatenate([generic_images, np.repeat(tiny_images, args.tiny_repeats, axis=0), np.repeat(board_images, args.board_repeats, axis=0)])
    targets = np.concatenate([generic_targets, np.repeat(tiny_targets, args.tiny_repeats, axis=0), np.repeat(board_targets, args.board_repeats, axis=0)])
    images, targets = make_scale_augmented_training_set(images, targets)
    crops, crop_targets = make_face_crops(images, targets, jitter=True, seed=42, padding=args.crop_padding, jitter_fraction=args.jitter_fraction)
    dataset = tf.data.Dataset.from_tensor_slices((crops, crop_targets)).shuffle(len(crops), seed=42).batch(16).prefetch(tf.data.AUTOTUNE)
    print("training", crops.shape, crop_targets.shape)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=WeightedLoss())
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=WeightedLoss())
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, crops, args.output / "model_int8.tflite")
    report: dict[str, object] = {"train_samples": int(len(crops)), "crop_padding": args.crop_padding, "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        test_crops, _ = make_face_crops(test_images, test_targets, jitter=False, seed=42, padding=args.crop_padding)
        # The evaluator uses target-centered crops here to measure the refiner
        # itself; the separate routed evaluator supplies coarse crops.
        predictions = restore_predictions(predict_int8(args.output / "model_int8.tflite", test_crops), test_targets, padding=args.crop_padding)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
