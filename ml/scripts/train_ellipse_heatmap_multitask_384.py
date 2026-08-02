#!/usr/bin/env python3
"""Train a QAT heatmap-plus-geometry ellipse model across all domains."""

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
from train_ellipse_mask_all_domains_384 import MASK_SIZE, MODEL_IMAGE_SIZE, make_masks, predict_int8
from train_ellipse_robust_384 import BOARD_TRAIN_ZIPS, load_zips, make_scale_augmented_training_set


MASK_VALUES = MASK_SIZE * MASK_SIZE


def configure_gpu() -> None:
    """Limit TensorFlow to 15 GB on the training GPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def build_model() -> keras.Model:
    """Build a shared encoder with spatial mask and scalar geometry heads."""
    layers = keras.layers
    inputs = keras.Input((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 1), name="image")
    skips: list[tf.Tensor] = []
    x = inputs
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = layers.Conv2D(filters, 3, strides=2, padding="same", use_bias=False, name=f"enc{stage}_down_conv")(x)
        x = layers.BatchNormalization(epsilon=1e-3, name=f"enc{stage}_down_bn")(x)
        x = layers.ReLU(name=f"enc{stage}_down_relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"enc{stage}_refine_conv")(x)
        x = layers.BatchNormalization(epsilon=1e-3, name=f"enc{stage}_refine_bn")(x)
        x = layers.ReLU(name=f"enc{stage}_refine_relu")(x)
        skips.append(x)
    # why: this branch sees the entire frame and is trained directly on all
    # four ellipse coordinates, avoiding radius inference from mask blur.
    geometry = layers.GlobalAveragePooling2D(name="geometry_gap")(x)
    geometry = layers.Dense(64, activation="relu", name="geometry_shared")(geometry)
    geometry = layers.Dense(4, activation="sigmoid", name="geometry")(geometry)
    for stage, filters, skip_index in ((0, 48, 3), (1, 32, 2), (2, 24, 1)):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"up{stage}")(x)
        x = layers.Concatenate(name=f"join{stage}")([x, skips[skip_index]])
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"dec{stage}_conv")(x)
        x = layers.BatchNormalization(epsilon=1e-3, name=f"dec{stage}_bn")(x)
        x = layers.ReLU(name=f"dec{stage}_relu")(x)
    mask = layers.Conv2D(1, 1, activation="sigmoid", name="face_mask")(x)
    mask = layers.Flatten(name="mask_flatten")(mask)
    outputs = layers.Concatenate(name="ellipse_contract")([mask, geometry])
    return keras.Model(inputs, outputs, name="ellipse_heatmap_multitask_384")


class MultiTaskLoss(keras.losses.Loss):
    """Balance spatial face-mask learning with scalar ellipse supervision."""

    def __init__(self, geometry_weight: float = 8.0, **kwargs: object) -> None:
        """Initialize the scalar-geometry loss weight."""
        super().__init__(**kwargs)
        self.geometry_weight = geometry_weight

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return a per-example mask BCE/Dice plus geometry Huber loss."""
        true_mask = tf.reshape(y_true[:, :MASK_VALUES], (-1, MASK_SIZE, MASK_SIZE, 1))
        pred_mask = tf.reshape(y_pred[:, :MASK_VALUES], (-1, MASK_SIZE, MASK_SIZE, 1))
        true_geometry = y_true[:, MASK_VALUES:]
        pred_geometry = y_pred[:, MASK_VALUES:]
        weights = 1.0 + 5.0 * true_mask[..., 0]
        bce = tf.reduce_mean(weights * keras.losses.binary_crossentropy(true_mask, pred_mask), axis=(1, 2))
        intersection = tf.reduce_sum(true_mask * pred_mask, axis=(1, 2, 3))
        denominator = tf.reduce_sum(true_mask + pred_mask, axis=(1, 2, 3))
        dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        error = tf.abs(true_geometry - pred_geometry)
        quadratic = tf.minimum(error, 0.05)
        linear = error - quadratic
        geometry = tf.reduce_sum(0.5 * tf.square(quadratic) + 0.05 * linear, axis=-1)
        return bce + dice + self.geometry_weight * geometry

    def get_config(self) -> dict[str, object]:
        """Return the serializable loss configuration."""
        return {**super().get_config(), "geometry_weight": self.geometry_weight}


def make_contract_targets(targets: np.ndarray) -> np.ndarray:
    """Concatenate rasterized masks and normalized ellipse geometry."""
    masks = make_masks(targets)[..., 0].reshape(len(targets), MASK_VALUES)
    return np.concatenate([masks, targets[:, :4]], axis=1).astype(np.float32)


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export a fully integer model using varied training images."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield representative images for activation calibration."""
        rng = np.random.default_rng(42)
        for index in rng.choice(len(images), min(512, len(images)), replace=False):
            yield [images[index : index + 1].astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_contract(model_path: Path, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run int8 inference and return dequantized masks and scalar geometry."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail, output_detail = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    values = []
    for image in images:
        quantized = np.clip(np.round(image / input_scale + input_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])[0].astype(np.float32)
        values.append((raw - output_zero) * output_scale)
    result = np.asarray(values, dtype=np.float32)
    return result[:, :MASK_VALUES].reshape(-1, MASK_SIZE, MASK_SIZE, 1), result[:, MASK_VALUES:]


def main() -> None:
    """Train, QAT-finetune, export, and score the multitask candidate."""
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
    generic_images, generic_targets = generic_images[:4000], generic_targets[:4000]
    images = np.concatenate([generic_images, np.repeat(tiny_images, args.tiny_repeats, axis=0), np.repeat(board_images, args.board_repeats, axis=0)])
    targets = np.concatenate([generic_targets, np.repeat(tiny_targets, args.tiny_repeats, axis=0), np.repeat(board_targets, args.board_repeats, axis=0)])
    images, targets = make_scale_augmented_training_set(images, targets)
    # why: resizing the entire host dataset on GPU creates a multi-gigabyte
    # temporary tensor and starves the training graph; keep staging on CPU.
    with tf.device("/CPU:0"):
        images = tf.image.resize(images, [MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE]).numpy()
    contract_targets = make_contract_targets(targets)
    dataset = tf.data.Dataset.from_tensor_slices((images, contract_targets)).shuffle(len(images), seed=42).batch(32).prefetch(tf.data.AUTOTUNE)
    print("training", images.shape, contract_targets.shape)

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=MultiTaskLoss())
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=MultiTaskLoss())
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images, args.output / "model_int8.tflite")

    report: dict[str, object] = {"train_samples": int(len(images)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        test_images = tf.image.resize(test_images, [MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE]).numpy()
        _, geometry = predict_contract(args.output / "model_int8.tflite", test_images)
        predictions = np.concatenate([geometry, np.ones((len(geometry), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
