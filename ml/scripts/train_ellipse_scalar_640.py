#!/usr/bin/env python3
"""Train a high-resolution QAT scalar ellipse model for tiny gauge faces."""

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


IMAGE_SIZE = 640


def configure_gpu() -> None:
    """Limit TensorFlow to 15 GB on the host GPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def build_model() -> keras.Model:
    """Build a compact high-resolution encoder with an absolute-coordinate head."""
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
    outputs = layers.Dense(5, activation="sigmoid", name="ellipse")(x)
    return keras.Model(inputs, outputs, name="ellipse_scalar_640")


class WeightedLoss(keras.losses.Loss):
    """Give center and tiny-face radii sufficient direct supervision."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize a serializable robust coordinate loss."""
        super().__init__(**kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return weighted Huber-like normalized-coordinate error."""
        error = tf.abs(y_true - y_pred)
        quadratic = tf.minimum(error, 0.05)
        linear = error - quadratic
        weights = tf.constant([1.0, 1.0, 1.5, 1.5, 0.25], tf.float32)
        return tf.reduce_sum((0.5 * tf.square(quadratic) + 0.05 * linear) * weights, axis=-1)


def resize_cpu(images: np.ndarray) -> np.ndarray:
    """Resize the host dataset without staging the full tensor on GPU."""
    with tf.device("/CPU:0"):
        return tf.image.resize(images, [IMAGE_SIZE, IMAGE_SIZE]).numpy()


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export a full-integer TFLite graph with representative images."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield varied calibration frames."""
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
    """Run the exported int8 model and return normalized ellipses."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    input_scale, input_zero = inp["quantization"]
    output_scale, output_zero = out["quantization"]
    predictions = []
    for image in images:
        quantized = np.clip(np.round(image / input_scale + input_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        predictions.append((raw - output_zero) * output_scale)
    return np.asarray(predictions, dtype=np.float32)


def main() -> None:
    """Train, QAT-finetune, export, and score the 640-pixel candidate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--qat-epochs", type=int, default=8)
    parser.add_argument("--tiny-repeats", type=int, default=80)
    parser.add_argument("--board-repeats", type=int, default=5)
    args = parser.parse_args()
    configure_gpu()
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(BOARD_TRAIN_ZIPS, labels=("temp_dial",))
    # why: keep high-resolution staging bounded; tiny and board source frames
    # are all retained, while generic coverage is still ample for regularization.
    generic_images, generic_targets = generic_images[:2000], generic_targets[:2000]
    images = np.concatenate([generic_images, np.repeat(tiny_images, args.tiny_repeats, axis=0), np.repeat(board_images, args.board_repeats, axis=0)])
    targets = np.concatenate([generic_targets, np.repeat(tiny_targets, args.tiny_repeats, axis=0), np.repeat(board_targets, args.board_repeats, axis=0)])
    images, targets = make_scale_augmented_training_set(images, targets)
    images = resize_cpu(images)
    dataset = tf.data.Dataset.from_tensor_slices((images, targets)).shuffle(len(images), seed=42).batch(16).prefetch(tf.data.AUTOTUNE)
    print("training", images.shape, targets.shape)

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=WeightedLoss())
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=WeightedLoss())
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images, args.output / "model_int8.tflite")

    report: dict[str, object] = {"train_samples": int(len(images)), "image_size": IMAGE_SIZE, "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        predictions = predict_int8(args.output / "model_int8.tflite", resize_cpu(test_images))
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
