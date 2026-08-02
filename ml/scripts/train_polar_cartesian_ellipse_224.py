#!/usr/bin/env python3
"""Train a QAT dual-domain polar-Cartesian local ellipse model.

The Cartesian crop retains absolute context while a polar unwrap converts the
face rim into a nearly horizontal signal, making the model less dependent on
the appearance of individual dial markings.
"""

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
from train_ellipse_face_crop_224 import make_face_crops
from train_ellipse_robust_384 import BOARD_TRAIN_ZIPS, load_zips, make_scale_augmented_training_set
from train_ellipse_scalar_640 import WeightedLoss

IMAGE_SIZE = 224
SEED = 42


def configure_gpu() -> None:
    """Limit TensorFlow to the approved 15 GB GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])


def make_polar(crops: np.ndarray) -> np.ndarray:
    """Unwrap each local face crop around its crop center into polar space."""
    polar: list[np.ndarray] = []
    center = (IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0)
    for crop in crops:
        gray = np.asarray(np.clip(crop[..., 0] * 255.0, 0, 255), dtype=np.uint8)
        # why: linear polar coordinates turn a circular/elliptical rim into a
        # strong radial transition while preserving all angles for occlusion.
        unwrapped = cv2.warpPolar(gray, (IMAGE_SIZE, IMAGE_SIZE), center, IMAGE_SIZE / 2.0, cv2.WARP_POLAR_LINEAR)
        polar.append(unwrapped.astype(np.float32)[..., None] / 255.0)
    return np.asarray(polar, dtype=np.float32)


@keras.utils.register_keras_serializable(package="gauge")
class ChannelSlice(keras.layers.Layer):
    """Select one input channel without an unsafe Lambda serialization."""

    def __init__(self, channel: int, **kwargs: object) -> None:
        """Store the channel index used by the view."""
        super().__init__(**kwargs)
        self.channel = int(channel)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return the selected channel while preserving a rank-4 tensor."""
        return inputs[:, :, :, self.channel : self.channel + 1]

    def get_config(self) -> dict[str, object]:
        """Return the serializable channel configuration."""
        return {**super().get_config(), "channel": self.channel}


def build_model() -> keras.Model:
    """Build two synchronized compact encoders with a shared geometry head."""
    layers = keras.layers
    cartesian = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="cartesian")
    polar = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="polar")
    branches: list[tf.Tensor] = []
    for branch_name, branch_input in (("cart", cartesian), ("polar", polar)):
        x = branch_input
        for stage, filters in enumerate((16, 24, 32, 48, 64)):
            x = layers.Conv2D(filters, 3, strides=2, padding="same", use_bias=False, name=f"{branch_name}_{stage}_conv")(x)
            x = layers.BatchNormalization(epsilon=1e-3, name=f"{branch_name}_{stage}_bn")(x)
            x = layers.ReLU(name=f"{branch_name}_{stage}_relu")(x)
        branches.append(layers.GlobalAveragePooling2D(name=f"{branch_name}_gap")(x))
    fused = layers.Concatenate(name="domain_fusion")(branches)
    fused = layers.Dense(96, activation="relu", name="fusion_hidden")(fused)
    return keras.Model([cartesian, polar], layers.Dense(5, activation="sigmoid", name="ellipse")(fused), name="polar_cartesian_ellipse_224")


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export a fully integer TFLite model using two-channel representatives."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield representative polar-Cartesian pairs."""
        rng = np.random.default_rng(SEED)
        for index in rng.choice(len(images), min(512, len(images)), replace=False):
            yield [images[index : index + 1, ..., :1].astype(np.float32), images[index : index + 1, ..., 1:2].astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model_path: Path, images: np.ndarray) -> np.ndarray:
    """Run int8 inference on two-channel local inputs."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    inputs, out = interpreter.get_input_details(), interpreter.get_output_details()[0]
    output_scale, output_zero = out["quantization"]
    values: list[np.ndarray] = []
    for image in images:
        # why: separate model inputs make both representations quantizable and
        # avoid unsupported channel-routing layers in TFMOT.
        for detail, channel in zip(inputs, (0, 1)):
            scale, zero = detail["quantization"]
            quantized = np.clip(np.round(image[..., channel : channel + 1] / scale + zero), -128, 127).astype(np.int8)
            interpreter.set_tensor(detail["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        values.append((raw - output_zero) * output_scale)
    return np.asarray(values, dtype=np.float32)


def restore(predictions: np.ndarray, targets: np.ndarray, padding: float) -> np.ndarray:
    """Map crop-relative ellipse predictions back to normalized frame units."""
    cx, cy, rx, ry = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]
    side = np.maximum(padding * np.maximum(rx, ry), 0.16)
    return np.stack([cx - side / 2.0 + predictions[:, 0] * side, cy - side / 2.0 + predictions[:, 1] * side, predictions[:, 2] * side, predictions[:, 3] * side, predictions[:, 4]], axis=1).astype(np.float32)


def main() -> None:
    """Train, QAT-finetune, export, and score target-centered local crops."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--qat-epochs", type=int, default=5)
    parser.add_argument("--generic-limit", type=int, default=3000)
    parser.add_argument("--tiny-repeats", type=int, default=40)
    parser.add_argument("--board-repeats", type=int, default=3)
    parser.add_argument("--padding", type=float, default=3.5)
    args = parser.parse_args()
    configure_gpu(); random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(BOARD_TRAIN_ZIPS, labels=("temp_dial",))
    generic_images, generic_targets = generic_images[:args.generic_limit], generic_targets[:args.generic_limit]
    images = np.concatenate([generic_images, np.repeat(tiny_images, args.tiny_repeats, axis=0), np.repeat(board_images, args.board_repeats, axis=0)])
    targets = np.concatenate([generic_targets, np.repeat(tiny_targets, args.tiny_repeats, axis=0), np.repeat(board_targets, args.board_repeats, axis=0)])
    images, targets = make_scale_augmented_training_set(images, targets)
    crops, local_targets = make_face_crops(images, targets, jitter=True, seed=SEED, padding=args.padding, jitter_fraction=0.12)
    polar = make_polar(crops)
    inputs = np.concatenate([crops, polar], axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(((crops, polar), local_targets)).shuffle(len(inputs), seed=SEED).batch(16).prefetch(tf.data.AUTOTUNE)
    print("training", inputs.shape, local_targets.shape, flush=True)
    model = build_model(); model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=WeightedLoss()); model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model); qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=WeightedLoss()); qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True); qat.save_weights(args.output / "model_qat.weights.h5"); export_int8(qat, inputs, args.output / "model_int8.tflite")
    report: dict[str, object] = {"train_samples": int(len(inputs)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        test_crops, _ = make_face_crops(test_images, test_targets, jitter=False, seed=SEED, padding=args.padding)
        test_inputs = np.concatenate([test_crops, make_polar(test_crops)], axis=-1)
        predictions = restore(predict_int8(args.output / "model_int8.tflite", test_inputs), test_targets, args.padding)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
