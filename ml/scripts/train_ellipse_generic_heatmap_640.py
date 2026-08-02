#!/usr/bin/env python3
"""Train a generic-domain center heatmap with hard-negative focal weighting."""

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
from train_ellipse_center_heatmap_640 import (
    HEATMAP_SIZE,
    HEATMAP_VALUES,
    build_model,
    decode_centers,
    export_int8,
    make_heatmaps,
    predict_int8,
)
from train_ellipse_robust_384 import SEED, load_zips, make_scale_augmented_training_set
from train_ellipse_scalar_640 import resize_cpu


class GenericFocalHeatmapLoss(keras.losses.Loss):
    """Use focal heatmap BCE plus center/radius regression for generic faces."""

    def __init__(self, geometry_weight: float = 5.0, **kwargs: object) -> None:
        """Initialize the geometry supervision weight."""
        super().__init__(**kwargs)
        self.geometry_weight = geometry_weight

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return a focal spatial loss that emphasizes hard background errors."""
        true_heatmap = y_true[:, :HEATMAP_VALUES]
        pred_heatmap = y_pred[:, :HEATMAP_VALUES]
        true_geometry = y_true[:, HEATMAP_VALUES:]
        pred_geometry = y_pred[:, HEATMAP_VALUES:]
        clipped = tf.clip_by_value(pred_heatmap, 1e-6, 1.0 - 1e-6)
        bce = -(true_heatmap * tf.math.log(clipped) + (1.0 - true_heatmap) * tf.math.log(1.0 - clipped))
        probability = true_heatmap * clipped + (1.0 - true_heatmap) * (1.0 - clipped)
        # why: gamma=2 suppresses easy background and trains on midpoint-like
        # false positives, which dominate the generic-domain tail errors.
        focal = tf.pow(1.0 - probability, 2.0) * bce
        focal *= 1.0 + 30.0 * true_heatmap
        heatmap = tf.reduce_mean(focal, axis=-1)
        error = tf.abs(true_geometry - pred_geometry)
        quadratic = tf.minimum(error, 0.05)
        linear = error - quadratic
        geometry = tf.reduce_sum(0.5 * tf.square(quadratic) + 0.05 * linear, axis=-1)
        return heatmap + self.geometry_weight * geometry

    def get_config(self) -> dict[str, object]:
        """Return the serializable loss configuration."""
        return {**super().get_config(), "geometry_weight": self.geometry_weight}


def configure_gpu() -> None:
    """Limit TensorFlow to the project's 15 GB GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def main() -> None:
    """Train, QAT-finetune, export, and score the generic specialist."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--qat-epochs", type=int, default=6)
    parser.add_argument("--train-limit", type=int, default=3000)
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    train_images, train_targets = load_zips(["train_1.zip"], labels=("GaugeFace",))
    val_images, val_targets = load_zips(["val_1.zip"], labels=("GaugeFace",))
    train_images, train_targets = train_images[: args.train_limit], train_targets[: args.train_limit]
    images = np.concatenate([train_images, val_images], axis=0)
    targets = np.concatenate([train_targets, val_targets], axis=0)
    images, targets = make_scale_augmented_training_set(images, targets)
    images = resize_cpu(images)
    contract_targets = np.concatenate([make_heatmaps(targets), targets[:, :4]], axis=1).astype(np.float32)
    dataset = (
        tf.data.Dataset.from_tensor_slices((images, contract_targets))
        .shuffle(len(images), seed=SEED)
        .batch(8)
        .prefetch(tf.data.AUTOTUNE)
    )
    print("training", images.shape, contract_targets.shape, flush=True)

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=GenericFocalHeatmapLoss())
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=GenericFocalHeatmapLoss())
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images, args.output / "model_int8.tflite")

    report: dict[str, object] = {"train_samples": int(len(images)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        maps, geometry = predict_int8(args.output / "model_int8.tflite", resize_cpu(test_images))
        centers = decode_centers(maps)
        predictions = np.concatenate([centers, geometry[:, 2:4], np.ones((len(geometry), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
