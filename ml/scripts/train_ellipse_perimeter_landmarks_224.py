#!/usr/bin/env python3
"""Train a universal int8 gauge-face landmark model and fit ellipses afterward.

The network predicts semantic geometry landmarks rather than gauge-specific
heads: one face center and eight evenly spaced points on the annotated ellipse.
The ellipse parameters are recovered deterministically from those points, which
keeps the learned task focused on localization and makes radius errors benign.
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
from train_ellipse_scalar_640 import resize_cpu


IMAGE_SIZE = 224
HEATMAP_SIZE = 56
LANDMARK_COUNT = 9
HEATMAP_VALUES = HEATMAP_SIZE * HEATMAP_SIZE
OUTPUT_VALUES = LANDMARK_COUNT * HEATMAP_VALUES


def configure_gpu() -> None:
    """Limit the first visible GPU to the project-approved 15 GB budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def build_model() -> keras.Model:
    """Build a small shared encoder with nine semantic landmark heatmaps."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = inputs
    skips: list[tf.Tensor] = []
    # why: the five compact stages keep the peak activation footprint suitable
    # for sequential NPU execution while preserving high-resolution skips.
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        for block, stride in enumerate((2, 1)):
            x = layers.Conv2D(
                filters, 3, strides=stride, padding="same", use_bias=False,
                name=f"enc{stage}_{block}_conv",
            )(x)
            x = layers.BatchNormalization(epsilon=1e-3, name=f"enc{stage}_{block}_bn")(x)
            x = layers.ReLU(name=f"enc{stage}_{block}_relu")(x)
        skips.append(x)
    for stage, filters, skip_index in ((0, 48, 3), (1, 32, 2), (2, 24, 1)):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"up{stage}")(x)
        x = layers.Concatenate(name=f"join{stage}")([x, skips[skip_index]])
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"dec{stage}_conv")(x)
        x = layers.BatchNormalization(epsilon=1e-3, name=f"dec{stage}_bn")(x)
        x = layers.ReLU(name=f"dec{stage}_relu")(x)
    heatmaps = layers.Conv2D(LANDMARK_COUNT, 1, activation="sigmoid", name="landmarks")(x)
    # why: Conv2D is spatial-major, but the loss/export contract is channel-
    # major so each landmark remains a contiguous heatmap in TFLite output.
    heatmaps = layers.Permute((3, 1, 2), name="landmark_channels_first")(heatmaps)
    return keras.Model(inputs, layers.Flatten(name="landmark_contract")(heatmaps), name="ellipse_perimeter_landmarks_224")


def landmark_targets(targets: np.ndarray) -> np.ndarray:
    """Rasterize center and eight ellipse-perimeter Gaussian targets."""
    coords = (np.arange(HEATMAP_SIZE, dtype=np.float32) + 0.5) / HEATMAP_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    angles = np.arange(8, dtype=np.float32) * (np.pi / 4.0)
    # why: the labels contain axis-aligned ellipse radii, so these points are
    # deterministic geometry supervision and do not encode gauge identity.
    points = np.concatenate(
        [targets[:, None, :2], targets[:, None, :2] + np.stack(
            [targets[:, None, 2] * np.cos(angles)[None], targets[:, None, 3] * np.sin(angles)[None]], axis=-1
        )], axis=1
    )
    sigma = np.where(np.arange(LANDMARK_COUNT) == 0, 0.020, 0.028).astype(np.float32)
    distance = (xx[None, None] - points[:, :, 0, None, None]) ** 2 + (yy[None, None] - points[:, :, 1, None, None]) ** 2
    values = np.exp(-distance / (2.0 * sigma[None, :, None, None] ** 2))
    return values.astype(np.float32).reshape(len(targets), OUTPUT_VALUES)


class LandmarkLoss(keras.losses.Loss):
    """Use stronger center supervision and balanced focal heatmap loss."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return a per-image weighted binary focal loss."""
        true = tf.reshape(y_true, (-1, LANDMARK_COUNT, HEATMAP_SIZE, HEATMAP_SIZE))
        pred = tf.reshape(y_pred, (-1, LANDMARK_COUNT, HEATMAP_SIZE, HEATMAP_SIZE))
        clipped = tf.clip_by_value(pred, 1e-5, 1.0 - 1e-5)
        bce = -(true * tf.math.log(clipped) + (1.0 - true) * tf.math.log(1.0 - clipped))
        # why: center correctness matters most, while focal weighting prevents
        # the large background from overwhelming sparse perimeter points.
        focal = tf.where(true > 0.05, 4.0 * (1.0 - clipped) ** 2.0, 0.25 * clipped ** 2.0)
        channel_weight = tf.concat([tf.constant([4.0]), tf.ones((LANDMARK_COUNT - 1,))], axis=0)
        return tf.reduce_mean(bce * focal * channel_weight[None, :, None, None], axis=(1, 2, 3))

    def get_config(self) -> dict[str, object]:
        """Return the serializable loss configuration."""
        return super().get_config()


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Convert the QAT model to a fully integer TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield deterministic representative images for activation calibration."""
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
    """Run the exported model and return dequantized landmark heatmaps."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail, output_detail = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = input_detail["quantization"]
    out_scale, out_zero = output_detail["quantization"]
    values: list[np.ndarray] = []
    for image in images:
        quantized = np.clip(np.round(image / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])[0].astype(np.float32)
        values.append((raw - out_zero) * out_scale)
    return np.asarray(values, dtype=np.float32).reshape(-1, LANDMARK_COUNT, HEATMAP_SIZE, HEATMAP_SIZE)


def decode_landmarks(heatmaps: np.ndarray) -> np.ndarray:
    """Decode each heatmap with background subtraction and a soft centroid."""
    coords = (np.arange(HEATMAP_SIZE, dtype=np.float32) + 0.5) / HEATMAP_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    decoded: list[list[list[float]]] = []
    for sample in heatmaps:
        points: list[list[float]] = []
        for channel, heatmap in enumerate(sample):
            weights = np.maximum(heatmap - (0.20 if channel == 0 else 0.10), 0.0) ** 2.0
            total = max(float(weights.sum()), 1e-6)
            points.append([float((weights * xx).sum() / total), float((weights * yy).sum() / total)])
        decoded.append(points)
    return np.asarray(decoded, dtype=np.float32)


def fit_ellipse(points: np.ndarray) -> np.ndarray:
    """Recover center and robust radii from decoded perimeter landmarks."""
    center = points[:, 0]
    perimeter = points[:, 1:]
    # Pairwise symmetry makes radius estimates tolerant of one bad landmark.
    x_radius = np.median(np.abs(perimeter[:, :, 0] - center[:, None, 0]), axis=1)
    y_radius = np.median(np.abs(perimeter[:, :, 1] - center[:, None, 1]), axis=1)
    return np.concatenate([center, x_radius[:, None], y_radius[:, None]], axis=1)


def main() -> None:
    """Train, QAT-finetune, export, and score the universal landmark model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--qat-epochs", type=int, default=6)
    parser.add_argument("--generic-limit", type=int, default=3000)
    parser.add_argument("--tiny-repeats", type=int, default=60)
    parser.add_argument("--board-repeats", type=int, default=3)
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
    images = tf.image.resize(images, (IMAGE_SIZE, IMAGE_SIZE), antialias=True).numpy()
    dataset = tf.data.Dataset.from_tensor_slices((images, landmark_targets(targets))).shuffle(len(images), seed=42).batch(16).prefetch(tf.data.AUTOTUNE)
    print("training", images.shape, landmark_targets(targets).shape)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=LandmarkLoss())
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=LandmarkLoss())
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images, args.output / "model_int8.tflite")
    report: dict[str, object] = {"train_samples": int(len(images)), "input_size": IMAGE_SIZE, "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        predictions = fit_ellipse(decode_landmarks(predict_int8(args.output / "model_int8.tflite", tf.image.resize(test_images, (IMAGE_SIZE, IMAGE_SIZE)).numpy())))
        report["tests"][zip_name] = _metrics(np.concatenate([predictions, np.ones((len(predictions), 1), dtype=np.float32)], axis=1), test_targets)
        print(zip_name, report["tests"][zip_name])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
