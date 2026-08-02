#!/usr/bin/env python3
"""Train a local 256px geometry refiner on top of the OCDet proposer.

The global detector is intentionally responsible for recall.  This second
stage receives a padded square crop around a flip-consistent proposal and
regresses the local ellipse geometry at higher spatial detail.  The model is
class-agnostic and therefore does not create a specialist per gauge family.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_ellipse_all_test_sets import _load_zip, _metrics  # noqa: E402
from train_ellipse_robust_384 import (  # noqa: E402
    BOARD_TRAIN_ZIPS,
    load_zips,
    make_scale_augmented_training_set,
)
from train_ocdet_ellipse_320 import decode, predict_int8  # noqa: E402

LOCAL_SIZE = 256
PROPOSER_SIZE = 320
CROP_FACTOR = 3.2
SEED = 42
PROPOSER = Path("artifacts/ocdet_ellipse_320_v2/model_int8.tflite")


def configure_gpu() -> None:
    """Limit TensorFlow to the approved 15 GB GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def block(x: tf.Tensor, filters: int, stride: int, name: str) -> tf.Tensor:
    """Apply one quantization-friendly convolutional block."""
    layers = keras.layers
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def build_model() -> keras.Model:
    """Build a compact local scalar geometry refiner."""
    layers = keras.layers
    inputs = keras.Input((LOCAL_SIZE, LOCAL_SIZE, 1), name="local_crop")
    x = inputs
    for stage, filters in enumerate((24, 32, 48, 64, 80)):
        x = block(x, filters, 2, f"enc{stage}_down")
        x = block(x, filters, 1, f"enc{stage}_refine")
    x = layers.GlobalAveragePooling2D(name="geometry_gap")(x)
    x = layers.Dense(64, activation="relu", name="geometry_hidden")(x)
    return keras.Model(
        inputs,
        layers.Dense(4, activation="sigmoid", name="local_ellipse")(x),
        name="ocdet_refiner_256",
    )


def crop_image(image: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop a normalized square with gray padding and return its source box."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    ix1, iy1 = int(np.floor(x1 * width)), int(np.floor(y1 * height))
    ix2, iy2 = int(np.ceil(x2 * width)), int(np.ceil(y2 * height))
    side = max(ix2 - ix1, iy2 - iy1, 1)
    canvas = np.full((side, side), 0.5, dtype=np.float32)
    source_x1, source_y1 = max(0, ix1), max(0, iy1)
    source_x2, source_y2 = min(width, ix1 + side), min(height, iy1 + side)
    dst_x1, dst_y1 = source_x1 - ix1, source_y1 - iy1
    canvas[dst_y1 : dst_y1 + source_y2 - source_y1, dst_x1 : dst_x1 + source_x2 - source_x1] = image[
        source_y1:source_y2, source_x1:source_x2, 0
    ]
    resized = cv2.resize(canvas, (LOCAL_SIZE, LOCAL_SIZE), interpolation=cv2.INTER_AREA)
    source_box = np.asarray([ix1 / width, iy1 / height, (ix1 + side) / width, (iy1 + side) / height], dtype=np.float32)
    return resized[..., None], source_box


def make_examples(
    images: np.ndarray, targets: np.ndarray, repeats: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate jittered local crops from ground-truth geometry."""
    rng = np.random.default_rng(SEED + 12)
    crops: list[np.ndarray] = []
    local_targets: list[list[float]] = []
    for image, target in zip(images, targets):
        for _ in range(repeats):
            # why: a 3x radius crop leaves context for the refiner while still
            # making the local center a meaningful high-resolution problem.
            side = float(np.clip(CROP_FACTOR * max(target[2], target[3]) * rng.uniform(0.85, 1.25), 0.30, 1.20))
            cx = float(target[0] + rng.normal(0.0, 0.045 * side))
            cy = float(target[1] + rng.normal(0.0, 0.045 * side))
            box = np.asarray([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], dtype=np.float32)
            crop, source_box = crop_image(image, box)
            sx, sy = source_box[2] - source_box[0], source_box[3] - source_box[1]
            crops.append(crop)
            local_targets.append([
                np.clip((target[0] - source_box[0]) / sx, 0.0, 1.0),
                np.clip((target[1] - source_box[1]) / sy, 0.0, 1.0),
                np.clip(target[2] / sx, 0.001, 1.0),
                np.clip(target[3] / sy, 0.001, 1.0),
            ])
    return np.asarray(crops, dtype=np.float32), np.asarray(local_targets, dtype=np.float32)


class GeometryLoss(keras.losses.Loss):
    """Weight center coordinates more heavily than the tolerant radii."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return a smooth center-prioritized regression loss."""
        error = tf.abs(y_true - y_pred)
        quadratic = tf.minimum(error, 0.05)
        linear = error - quadratic
        huber = 0.5 * tf.square(quadratic) + 0.05 * linear
        weights = tf.constant([3.0, 3.0, 1.0, 1.0], dtype=tf.float32)
        return tf.reduce_sum(huber * weights, axis=-1)


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export the QAT refiner as an integer-only TFLite graph."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield varied local crops for activation calibration."""
        rng = np.random.default_rng(SEED)
        for index in rng.choice(len(images), min(512, len(images)), replace=False):
            yield [images[index : index + 1].astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_local(model_path: Path, images: np.ndarray) -> np.ndarray:
    """Run the int8 local refiner on a crop batch."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    input_detail, output_detail = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    values: list[np.ndarray] = []
    for image in images:
        quantized = np.clip(np.round(image / input_scale + input_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])[0].astype(np.float32)
        values.append((raw - output_zero) * output_scale)
    return np.asarray(values, dtype=np.float32)


def proposer_tta(images: np.ndarray, model: Path) -> np.ndarray:
    """Return median geometry proposals across horizontal/vertical flips."""
    proposals: list[np.ndarray] = []
    for mode in range(4):
        transformed = images.copy()
        if mode in (1, 3):
            transformed = transformed[:, :, ::-1, :]
        if mode in (2, 3):
            transformed = transformed[:, ::-1, :, :]
        values = decode(predict_int8(model, transformed))
        if mode in (1, 3):
            values[:, 0] = 1.0 - values[:, 0]
        if mode in (2, 3):
            values[:, 1] = 1.0 - values[:, 1]
        proposals.append(values)
    return np.median(np.stack(proposals, axis=0), axis=0).astype(np.float32)


def refine(model: Path, proposer: Path, images: np.ndarray, blend: float) -> np.ndarray:
    """Run flip-consistent proposal, local crop, and source-frame remapping."""
    proposals = proposer_tta(images, proposer)
    crops: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    for image, proposal in zip(images, proposals):
        # why: the minimum crop prevents a single imperfect tiny-gauge proposal
        # from clipping the face before the high-resolution stage sees it.
        side = float(np.clip(CROP_FACTOR * max(proposal[2], proposal[3]), 0.30, 1.20))
        box = np.asarray([
            proposal[0] - side / 2,
            proposal[1] - side / 2,
            proposal[0] + side / 2,
            proposal[1] + side / 2,
        ], dtype=np.float32)
        crop, source_box = crop_image(image, box)
        crops.append(crop)
        boxes.append(source_box)
    local = predict_local(model, np.asarray(crops, dtype=np.float32))
    outputs: list[list[float]] = []
    for proposal, value, source_box in zip(proposals, local, boxes):
        sx, sy = source_box[2] - source_box[0], source_box[3] - source_box[1]
        local_center = np.asarray([
            source_box[0] + value[0] * sx,
            source_box[1] + value[1] * sy,
        ], dtype=np.float32)
        center = (1.0 - blend) * proposal[:2] + blend * local_center
        outputs.append([center[0], center[1], value[2] * sx, value[3] * sy, 1.0])
    return np.asarray(outputs, dtype=np.float32)


def main() -> None:
    """Train, QAT, export, parity-check, and evaluate the two-stage model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposer", type=Path, default=PROPOSER)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--qat-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generic-count", type=int, default=2500)
    parser.add_argument("--tiny-repeats", type=int, default=35)
    parser.add_argument("--board-repeats", type=int, default=3)
    parser.add_argument("--crop-repeats", type=int, default=2)
    parser.add_argument("--center-blend", type=float, default=0.75)
    parser.add_argument("--local-size", type=int, default=256)
    parser.add_argument("--crop-factor", type=float, default=3.2)
    args = parser.parse_args()

    global LOCAL_SIZE, CROP_FACTOR
    LOCAL_SIZE = args.local_size
    CROP_FACTOR = args.crop_factor

    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(BOARD_TRAIN_ZIPS, labels=("temp_dial",))
    images = np.concatenate([
        generic_images[: args.generic_count],
        np.repeat(tiny_images, args.tiny_repeats, axis=0),
        np.repeat(board_images, args.board_repeats, axis=0),
    ])
    targets = np.concatenate([
        generic_targets[: args.generic_count],
        np.repeat(tiny_targets, args.tiny_repeats, axis=0),
        np.repeat(board_targets, args.board_repeats, axis=0),
    ])
    images, targets = make_scale_augmented_training_set(images, targets)
    local_images, local_targets = make_examples(images, targets, args.crop_repeats)
    dataset = (
        tf.data.Dataset.from_tensor_slices((local_images, local_targets))
        .shuffle(len(local_images), seed=SEED)
        .batch(args.batch_size)
        .prefetch(1)
    )
    print("training", local_images.shape, flush=True)

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=GeometryLoss())
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=GeometryLoss())
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)

    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, local_images, args.output / "model_int8.tflite")
    float_values = np.asarray(qat.predict(local_images[:32], verbose=0), dtype=np.float32)
    int8_values = predict_local(args.output / "model_int8.tflite", local_images[:32])
    report: dict[str, object] = {
        "local_size": LOCAL_SIZE,
        "train_samples": int(len(local_images)),
        "keras_tflite_contract_mae": float(np.mean(np.abs(float_values - int8_values))),
        "tests": {},
    }
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name, image_size=PROPOSER_SIZE)
        predictions = refine(args.output / "model_int8.tflite", args.proposer, test_images, args.center_blend)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
