#!/usr/bin/env python3
"""Train a multiscale crop mixture-of-experts ellipse refiner.

The model sees one gauge crop, but internally evaluates three resized views
of that crop.  A learned gate blends the three expert predictions so the
network can emphasize fine detail for tiny gauges and broader context for
larger or cluttered gauges.
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
from train_ellipse_face_crop_224 import make_face_crops, restore_predictions
from train_ellipse_robust_384 import BOARD_TRAIN_ZIPS, SEED, load_zips, make_scale_augmented_training_set

IMAGE_SIZE = 224
PADDINGS = (2.8, 3.5, 4.6)
BRANCH_COUNT = len(PADDINGS)
ELLIPSE_VALUES = 5
OUTPUT_VALUES = BRANCH_COUNT * ELLIPSE_VALUES + BRANCH_COUNT


def configure_gpu() -> None:
    """Apply the project-wide 15 GB TensorFlow GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def _conv_block(x: tf.Tensor, filters: int, stride: int, name: str) -> tf.Tensor:
    """Build a small Conv-BN-ReLU block suitable for QAT and TFLite."""
    layers = keras.layers
    x = layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(epsilon=1e-3, name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def encode_crop(inputs: tf.Tensor, prefix: str) -> tf.Tensor:
    """Encode one resized crop into a compact embedding.

    This stays as a plain tensor transform instead of a nested Model because
    TFMOT cannot quantize nested keras.Model instances inside the outer graph.
    """
    x = inputs
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = _conv_block(x, filters, 2, f"{prefix}_s{stage}_down")
        x = _conv_block(x, filters, 1, f"{prefix}_s{stage}_refine")
    x = keras.layers.GlobalAveragePooling2D(name=f"{prefix}_gap")(x)
    return keras.layers.Dense(64, activation="relu", name=f"{prefix}_hidden")(x)


def build_model() -> keras.Model:
    """Build a three-crop expert model with a learned scale gate."""
    layers = keras.layers
    inputs = [
        keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name=f"crop_{index}")
        for index in range(BRANCH_COUNT)
    ]
    branch_features = [encode_crop(view, f"branch{index}") for index, view in enumerate(inputs)]

    branch_preds: list[tf.Tensor] = []
    for index, features in enumerate(branch_features):
        hidden = layers.Dense(48, activation="relu", name=f"branch{index}_pred_hidden")(features)
        branch_preds.append(layers.Dense(ELLIPSE_VALUES, activation="sigmoid", name=f"branch{index}_ellipse")(hidden))

    gate_input = layers.Concatenate(name="gate_input")(branch_features)
    gate_hidden = layers.Dense(48, activation="relu", name="gate_hidden")(gate_input)
    gate = layers.Dense(BRANCH_COUNT, activation="softmax", name="scale_gate")(gate_hidden)
    return keras.Model(inputs, layers.Concatenate(name="ellipse_contract")([*branch_preds, gate]), name="multiscale_crop_moe_224")


def _component_huber(error: tf.Tensor, delta: float = 0.05) -> tf.Tensor:
    """Compute a smooth L1 penalty for a batch of absolute errors."""
    return tf.where(error <= delta, 0.5 * tf.square(error) / delta, error - 0.5 * delta)


def multi_scale_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Train the matching scale expert harder than the off-scale experts."""
    true_ellipses = tf.reshape(y_true[:, : BRANCH_COUNT * ELLIPSE_VALUES], (-1, BRANCH_COUNT, ELLIPSE_VALUES))
    true_gate = y_true[:, BRANCH_COUNT * ELLIPSE_VALUES :]
    pred_ellipses = tf.reshape(y_pred[:, : BRANCH_COUNT * ELLIPSE_VALUES], (-1, BRANCH_COUNT, ELLIPSE_VALUES))
    pred_gate = y_pred[:, BRANCH_COUNT * ELLIPSE_VALUES :]

    error = tf.abs(true_ellipses - pred_ellipses)
    # why: center matters more than radii, but the radius channels still need
    # enough pressure to keep the crop geometry sane.
    per_component = tf.constant([2.5, 2.5, 1.0, 1.0, 0.25], dtype=tf.float32)
    branch_loss = tf.reduce_sum(_component_huber(error) * per_component[None, None, :], axis=-1)
    branch_weights = 0.20 + 0.80 * true_gate
    branch_loss = tf.reduce_sum(branch_loss * branch_weights, axis=1) / tf.reduce_sum(branch_weights, axis=1)

    gate_loss = tf.reduce_mean(keras.losses.categorical_crossentropy(true_gate, pred_gate))
    return tf.reduce_mean(branch_loss) + 0.35 * gate_loss


def export_int8(model: keras.Model, branch_images: list[np.ndarray], output: Path) -> None:
    """Export the trained graph as a fully integer TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield representative crops for activation calibration."""
        rng = np.random.default_rng(SEED)
        for index in rng.choice(len(branch_images[0]), min(512, len(branch_images[0])), replace=False):
            yield [branch[index : index + 1].astype(np.float32) for branch in branch_images]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model_path: Path, branch_images: list[np.ndarray]) -> np.ndarray:
    """Run the exported integer graph and restore dequantized predictions."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inputs, out = interpreter.get_input_details(), interpreter.get_output_details()[0]
    out_scale, out_zero = out["quantization"]
    values: list[np.ndarray] = []
    for sample_index in range(len(branch_images[0])):
        for detail, branch in zip(inputs, branch_images):
            scale, zero = detail["quantization"]
            quantized = np.clip(np.round(branch[sample_index : sample_index + 1] / scale + zero), -128, 127).astype(np.int8)
            interpreter.set_tensor(detail["index"], quantized)
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        values.append((raw - out_zero) * out_scale)
    return np.asarray(values, dtype=np.float32)


def decode_predictions(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Blend the expert predictions and restore them to full-frame geometry."""
    branches = values[:, : BRANCH_COUNT * ELLIPSE_VALUES].reshape(-1, BRANCH_COUNT, ELLIPSE_VALUES)
    gate = values[:, BRANCH_COUNT * ELLIPSE_VALUES :]
    gate = gate / np.maximum(gate.sum(axis=1, keepdims=True), 1e-6)
    restored = []
    for branch_index, padding in enumerate(PADDINGS):
        restored.append(restore_predictions(branches[:, branch_index], targets, padding=padding))
    stacked = np.stack(restored, axis=1)
    return np.sum(stacked * gate[..., None], axis=1)


def main() -> None:
    """Train, export, and evaluate the multiscale crop mixture-of-experts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--qat-epochs", type=int, default=5)
    parser.add_argument("--generic-limit", type=int, default=2500)
    parser.add_argument("--tiny-repeats", type=int, default=36)
    parser.add_argument("--board-repeats", type=int, default=3)
    parser.add_argument("--crop-padding", type=float, default=3.5)
    parser.add_argument("--jitter-fraction", type=float, default=0.12)
    args = parser.parse_args()

    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(BOARD_TRAIN_ZIPS, labels=("temp_dial",))
    generic_images, generic_targets = generic_images[: args.generic_limit], generic_targets[: args.generic_limit]

    images = np.concatenate(
        [
            generic_images,
            np.repeat(tiny_images, args.tiny_repeats, axis=0),
            np.repeat(board_images, args.board_repeats, axis=0),
        ]
    )
    targets = np.concatenate(
        [
            generic_targets,
            np.repeat(tiny_targets, args.tiny_repeats, axis=0),
            np.repeat(board_targets, args.board_repeats, axis=0),
        ]
    )
    images, targets = make_scale_augmented_training_set(images, targets)
    branch_crops: list[np.ndarray] = []
    branch_targets: list[np.ndarray] = []
    for padding in PADDINGS:
        crops, local_targets = make_face_crops(
            images,
            targets,
            jitter=True,
            seed=SEED,
            padding=padding,
            jitter_fraction=args.jitter_fraction,
        )
        branch_crops.append(crops)
        branch_targets.append(local_targets)
    max_radius = np.max(branch_targets[1][:, 2:4], axis=1)
    gate_index = np.digitize(max_radius, bins=np.asarray([0.18, 0.30], dtype=np.float32), right=False)
    gate = np.eye(BRANCH_COUNT, dtype=np.float32)[gate_index]
    dataset_targets = np.concatenate([branch_targets[0], branch_targets[1], branch_targets[2], gate], axis=1).astype(np.float32)
    dataset = (
        tf.data.Dataset.from_tensor_slices(((branch_crops[0], branch_crops[1], branch_crops[2]), dataset_targets))
        .shuffle(len(branch_crops[0]), seed=SEED)
        .batch(16)
        .prefetch(tf.data.AUTOTUNE)
    )
    print("training", branch_crops[0].shape, flush=True)

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=multi_scale_loss)
    model.fit(dataset, epochs=args.epochs, verbose=2)

    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=multi_scale_loss)
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)

    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, branch_crops, args.output / "model_int8.tflite")

    report: dict[str, object] = {"train_samples": int(len(crops)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        test_branch_crops: list[np.ndarray] = []
        for padding in PADDINGS:
            crops, _ = make_face_crops(
                test_images,
                test_targets,
                jitter=False,
                seed=SEED,
                padding=padding,
            )
            test_branch_crops.append(crops)
        predictions = decode_predictions(
            predict_int8(args.output / "model_int8.tflite", test_branch_crops),
            test_targets,
        )
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
