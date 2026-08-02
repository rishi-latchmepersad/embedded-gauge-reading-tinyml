#!/usr/bin/env python3
"""Train one all-domain spatial ellipse mask with center-prioritized loss."""

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
from train_ellipse_mask_all_domains_384 import (
    MASK_SIZE,
    build_model,
    make_masks,
    predict_int8,
    soft_box_from_masks,
)
from train_ellipse_robust_384 import BOARD_TRAIN_ZIPS, SEED, load_zips, make_scale_augmented_training_set


def configure_gpu() -> None:
    """Cap TensorFlow at the project's 15 GB GPU limit."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def _moment_center(mask: tf.Tensor) -> tf.Tensor:
    """Compute a differentiable foreground-weighted center from a mask."""
    coords = tf.linspace(0.5 / MASK_SIZE, 1.0 - 0.5 / MASK_SIZE, MASK_SIZE)
    yy, xx = tf.meshgrid(coords, coords, indexing="ij")
    weights = tf.nn.relu(mask[..., 0] - 0.10)
    total = tf.reduce_sum(weights, axis=(1, 2), keepdims=True) + 1e-6
    denominator = tf.squeeze(total, axis=(1, 2))
    center_x = tf.reduce_sum(weights * xx[None, ...], axis=(1, 2)) / denominator
    center_y = tf.reduce_sum(weights * yy[None, ...], axis=(1, 2)) / denominator
    return tf.stack([center_x, center_y], axis=1)


def universal_mask_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Combine focal BCE, Dice, and center-prioritized geometry supervision."""
    true_mask = y_true
    pred_mask = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
    bce = -(true_mask * tf.math.log(pred_mask) + (1.0 - true_mask) * tf.math.log(1.0 - pred_mask))
    probability = true_mask * pred_mask + (1.0 - true_mask) * (1.0 - pred_mask)
    # why: hard-negative weighting discourages the diffuse midpoint mask that
    # caused the largest generic-domain center failures.
    focal = tf.pow(1.0 - probability, 2.0) * bce
    focal *= 1.0 + 12.0 * true_mask
    focal_loss = tf.reduce_mean(focal, axis=(1, 2, 3))
    intersection = tf.reduce_sum(true_mask * pred_mask, axis=(1, 2, 3))
    denominator = tf.reduce_sum(true_mask + pred_mask, axis=(1, 2, 3))
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    true_center = _moment_center(true_mask)
    predicted_center = _moment_center(pred_mask)
    center_loss = tf.reduce_mean(keras.losses.huber(true_center, predicted_center), axis=-1)
    # why: center error is more important than radius variation for the crop.
    return tf.reduce_mean(focal_loss + dice_loss + 10.0 * center_loss)


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export the universal model as a fully integer TFLite graph."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield varied host-held images for activation calibration."""
        rng = np.random.default_rng(SEED)
        for index in rng.choice(len(images), min(512, len(images)), replace=False):
            yield [images[index : index + 1].astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def main() -> None:
    """Train, QAT-finetune, export, and score the universal model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--qat-epochs", type=int, default=8)
    parser.add_argument("--generic-limit", type=int, default=3000)
    parser.add_argument("--tiny-repeats", type=int, default=80)
    parser.add_argument("--board-repeats", type=int, default=3)
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    generic_images, generic_targets = load_zips(["train_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(BOARD_TRAIN_ZIPS, labels=("temp_dial",))
    generic_images, generic_targets = generic_images[: args.generic_limit], generic_targets[: args.generic_limit]
    images = np.concatenate([
        generic_images,
        np.repeat(tiny_images, args.tiny_repeats, axis=0),
        np.repeat(board_images, args.board_repeats, axis=0),
    ])
    targets = np.concatenate([
        generic_targets,
        np.repeat(tiny_targets, args.tiny_repeats, axis=0),
        np.repeat(board_targets, args.board_repeats, axis=0),
    ])
    images, targets = make_scale_augmented_training_set(images, targets)
    masks = make_masks(targets)
    print("training", images.shape, masks.shape, flush=True)
    dataset = tf.data.Dataset.from_tensor_slices((images, masks)).shuffle(len(images), seed=SEED).batch(32).prefetch(tf.data.AUTOTUNE)

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=universal_mask_loss)
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=universal_mask_loss)
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images, args.output / "model_int8.tflite")

    report: dict[str, object] = {"train_samples": int(len(images)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        predicted_masks = predict_int8(args.output / "model_int8.tflite", test_images)
        predicted_boxes = soft_box_from_masks(predicted_masks)
        predictions = np.concatenate([predicted_boxes, np.ones((len(predicted_boxes), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
