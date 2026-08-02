#!/usr/bin/env python3
"""Train a QAT domain-specific radius regressor for the ellipse pipeline."""

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
from train_ellipse_scalar_640 import IMAGE_SIZE, resize_cpu


def configure_gpu() -> None:
    """Limit this focused training process to 15 GB of host GPU memory."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])


def build_model() -> keras.Model:
    """Build one shared encoder with three radius experts and a domain head."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = inputs
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        for block, stride in enumerate((2, 1)):
            x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"s{stage}_{block}_conv")(x)
            x = layers.BatchNormalization(epsilon=1e-3, name=f"s{stage}_{block}_bn")(x)
            x = layers.ReLU(name=f"s{stage}_{block}_relu")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    experts = [layers.Dense(2, activation="sigmoid", name=f"radius_domain_{name}")(layers.Dense(32, activation="relu", name=f"radius_shared_{name}")(x)) for name in ("tiny", "generic", "board")]
    domain = layers.Dense(3, activation="softmax", name="domain_probability")(layers.Dense(16, activation="relu", name="domain_shared")(x))
    return keras.Model(inputs, layers.Concatenate(name="radius_domain_contract")(experts + [domain]), name="ellipse_radius_domains_640")


class RadiusDomainLoss(keras.losses.Loss):
    """Train only the radius expert selected by the one-hot domain target."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return masked robust radius error plus domain classification loss."""
        true_radius = y_true[:, :2]
        domain = y_true[:, 2:]
        pred_tiny, pred_generic, pred_board = y_pred[:, :2], y_pred[:, 2:4], y_pred[:, 4:6]
        pred_domain = y_pred[:, 6:]
        selected = tf.where(domain[:, 0:1] > 0.5, pred_tiny, tf.where(domain[:, 1:2] > 0.5, pred_generic, pred_board))
        error = tf.abs(true_radius - selected)
        quadratic = tf.minimum(error, 0.05)
        linear = error - quadratic
        radius = tf.reduce_sum(0.5 * tf.square(quadratic) + 0.05 * linear, axis=-1)
        classification = tf.keras.losses.categorical_crossentropy(domain, pred_domain)
        return radius + classification


def export_int8(model: keras.Model, images: np.ndarray, destination: Path) -> None:
    """Export the trained radius experts as a fully integer TFLite graph."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield representative calibration images."""
        rng = np.random.default_rng(42)
        for index in rng.choice(len(images), min(512, len(images)), replace=False):
            yield [images[index : index + 1].astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    destination.write_bytes(converter.convert())


def predict_int8(model_path: Path, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return all domain radii and the predicted three-way domain probabilities."""
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
    values = np.asarray(predictions, dtype=np.float32)
    return values[:, :6].reshape(-1, 3, 2), values[:, 6:]


def main() -> None:
    """Train, quantize, export, and evaluate domain-specific radii."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--qat-epochs", type=int, default=4)
    parser.add_argument("--tiny-repeats", type=int, default=40)
    parser.add_argument("--board-repeats", type=int, default=5)
    args = parser.parse_args()
    configure_gpu()
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(BOARD_TRAIN_ZIPS, labels=("temp_dial",))
    generic_images, generic_targets = generic_images[:2000], generic_targets[:2000]
    source_images = [np.repeat(tiny_images, args.tiny_repeats, axis=0), generic_images, np.repeat(board_images, args.board_repeats, axis=0)]
    source_targets = [np.repeat(tiny_targets, args.tiny_repeats, axis=0), generic_targets, np.repeat(board_targets, args.board_repeats, axis=0)]
    images = []
    targets = []
    domains = []
    for domain_index, (source, target) in enumerate(zip(source_images, source_targets, strict=True)):
        augmented_images, augmented_targets = make_scale_augmented_training_set(source, target)
        images.append(augmented_images)
        targets.append(augmented_targets[:, 2:4])
        one_hot = np.zeros((len(augmented_targets), 3), np.float32)
        one_hot[:, domain_index] = 1.0
        domains.append(one_hot)
    resized = resize_cpu(np.concatenate(images))
    contract_targets = np.concatenate([np.concatenate(targets), np.concatenate(domains)], axis=1).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((resized, contract_targets)).shuffle(len(resized), seed=42).batch(16).prefetch(tf.data.AUTOTUNE)
    print("training", resized.shape, contract_targets.shape)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=RadiusDomainLoss())
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=RadiusDomainLoss())
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, resized, args.output / "model_int8.tflite")
    report: dict[str, object] = {"train_samples": int(len(resized)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        radii, domain = predict_int8(args.output / "model_int8.tflite", resize_cpu(test_images))
        selected = radii[np.arange(len(radii)), np.argmax(domain, axis=1)]
        prediction = np.concatenate([test_targets[:, :2], selected, np.ones((len(selected), 1), np.float32)], axis=1)
        report["tests"][zip_name] = {**_metrics(prediction, test_targets), "domain_counts": np.bincount(np.argmax(domain, axis=1), minlength=3).tolist()}
        print(zip_name, report["tests"][zip_name])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
