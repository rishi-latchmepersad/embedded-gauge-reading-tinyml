#!/usr/bin/env python3
"""Train one QAT 640 model with tiny and normal spatial heatmap heads."""

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
from train_ellipse_center_heatmap_640 import HEATMAP_SIZE, HEATMAP_VALUES
from train_ellipse_robust_384 import BOARD_TRAIN_ZIPS, load_zips, make_scale_augmented_training_set
from train_ellipse_scalar_640 import IMAGE_SIZE, resize_cpu


NORMAL_SIZE = 80
NORMAL_VALUES = NORMAL_SIZE * NORMAL_SIZE


def configure_gpu() -> None:
    """Limit TensorFlow to 15 GB on the host GPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def build_model() -> keras.Model:
    """Build one shared encoder with tiny and normal decoder heads."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = inputs
    skips: list[tf.Tensor] = []
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        for block, stride in enumerate((2, 1)):
            x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"enc{stage}_{block}_conv")(x)
            x = layers.BatchNormalization(epsilon=1e-3, name=f"enc{stage}_{block}_bn")(x)
            x = layers.ReLU(name=f"enc{stage}_{block}_relu")(x)
        skips.append(x)
    geometry = layers.GlobalAveragePooling2D(name="geometry_gap")(x)
    geometry = layers.Dense(64, activation="relu", name="geometry_shared")(geometry)
    geometry = layers.Dense(4, activation="sigmoid", name="geometry")(geometry)
    decoder_outputs: list[tf.Tensor] = []
    for head in ("tiny", "normal"):
        y = x
        decoder_stages = ((0, 48, 3), (1, 32, 2), (2, 24, 1)) if head == "tiny" else ((0, 32, 3), (1, 24, 2))
        for stage, filters, skip_index in decoder_stages:
            y = layers.UpSampling2D(2, interpolation="nearest", name=f"{head}_up{stage}")(y)
            y = layers.Concatenate(name=f"{head}_join{stage}")([y, skips[skip_index]])
            y = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{head}_dec{stage}_conv")(y)
            y = layers.BatchNormalization(epsilon=1e-3, name=f"{head}_dec{stage}_bn")(y)
            y = layers.ReLU(name=f"{head}_dec{stage}_relu")(y)
        heatmap = layers.Conv2D(1, 1, activation="sigmoid", name=f"{head}_heatmap")(y)
        decoder_outputs.append(layers.Flatten(name=f"{head}_flatten")(heatmap))
    return keras.Model(inputs, layers.Concatenate(name="ellipse_contract")(decoder_outputs + [geometry]), name="ellipse_domain_heatmaps_640")


class DomainHeatmapLoss(keras.losses.Loss):
    """Train only the heatmap head belonging to each source domain."""

    def __init__(self, geometry_weight: float = 5.0, **kwargs: object) -> None:
        """Initialize the scalar branch weight."""
        super().__init__(**kwargs)
        self.geometry_weight = geometry_weight

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return masked heatmap BCE plus geometry error."""
        true_tiny = y_true[:, :HEATMAP_VALUES]
        true_normal = y_true[:, HEATMAP_VALUES : HEATMAP_VALUES + NORMAL_VALUES]
        true_geometry = y_true[:, HEATMAP_VALUES + NORMAL_VALUES : HEATMAP_VALUES + NORMAL_VALUES + 4]
        domain = y_true[:, -1:]
        pred_tiny = y_pred[:, :HEATMAP_VALUES]
        pred_normal = y_pred[:, HEATMAP_VALUES : HEATMAP_VALUES + NORMAL_VALUES]
        pred_geometry = y_pred[:, HEATMAP_VALUES + NORMAL_VALUES : HEATMAP_VALUES + NORMAL_VALUES + 4]
        def bce_loss(true_heatmap: tf.Tensor, pred_heatmap: tf.Tensor) -> tf.Tensor:
            """Compute foreground-weighted elementwise heatmap BCE."""
            weights = 1.0 + 15.0 * true_heatmap
            clipped = tf.clip_by_value(pred_heatmap, 1e-6, 1.0 - 1e-6)
            bce = -(true_heatmap * tf.math.log(clipped) + (1.0 - true_heatmap) * tf.math.log(1.0 - clipped))
            return tf.reduce_mean(weights * bce, axis=-1)

        tiny_loss = bce_loss(true_tiny, pred_tiny)
        normal_loss = bce_loss(true_normal, pred_normal)
        heatmap = tf.where(tf.squeeze(domain, axis=-1) < 0.5, tiny_loss, normal_loss)
        error = tf.abs(true_geometry - pred_geometry)
        quadratic = tf.minimum(error, 0.05)
        linear = error - quadratic
        geometry = tf.reduce_sum(0.5 * tf.square(quadratic) + 0.05 * linear, axis=-1)
        return heatmap + self.geometry_weight * geometry

    def get_config(self) -> dict[str, object]:
        """Return serializable loss configuration."""
        return {**super().get_config(), "geometry_weight": self.geometry_weight}


def heatmaps(targets: np.ndarray, size: int) -> np.ndarray:
    """Create normalized Gaussian centers on the 160x160 grid."""
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    sigma = 0.018
    values = np.exp(-((xx[None] - targets[:, 0, None, None]) ** 2 + (yy[None] - targets[:, 1, None, None]) ** 2) / (2.0 * sigma**2))
    return values.astype(np.float32).reshape(len(targets), size * size)


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export the domain-specialized model as fully integer TFLite."""
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model_path: Path, images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run int8 inference and return both heads, geometry, and output scale."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    values = []
    for image in images:
        quantized = np.clip(np.round(image / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        values.append((raw - out_zero) * out_scale)
    result = np.asarray(values, dtype=np.float32)
    return result[:, :HEATMAP_VALUES].reshape(-1, HEATMAP_SIZE, HEATMAP_SIZE, 1), result[:, HEATMAP_VALUES : HEATMAP_VALUES + NORMAL_VALUES].reshape(-1, NORMAL_SIZE, NORMAL_SIZE, 1), result[:, HEATMAP_VALUES + NORMAL_VALUES : HEATMAP_VALUES + NORMAL_VALUES + 4]


def decode_sharp(heatmap: np.ndarray) -> np.ndarray:
    """Decode one heatmap with the tuned tiny-face soft centroid."""
    size = heatmap.shape[1]
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    result = []
    for values in heatmap[..., 0]:
        weights = np.maximum(values - 0.50, 0.0) ** 4.0
        total = max(float(weights.sum()), 1e-6)
        result.append([(weights * xx).sum() / total, (weights * yy).sum() / total])
    return np.asarray(result, dtype=np.float32)


def decode_normal(heatmap: np.ndarray) -> np.ndarray:
    """Decode the normal/board heatmap with its softer centroid."""
    size = heatmap.shape[1]
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    result = []
    for values in heatmap[..., 0]:
        weights = np.maximum(values - 0.10, 0.0)
        total = max(float(weights.sum()), 1e-6)
        result.append([(weights * xx).sum() / total, (weights * yy).sum() / total])
    return np.asarray(result, dtype=np.float32)


def main() -> None:
    """Train, QAT-finetune, export, and evaluate the single-model candidate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--qat-epochs", type=int, default=4)
    parser.add_argument("--tiny-repeats", type=int, default=40)
    parser.add_argument("--board-repeats", type=int, default=3)
    args = parser.parse_args()
    configure_gpu()
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(BOARD_TRAIN_ZIPS, labels=("temp_dial",))
    generic_images, generic_targets = generic_images[:2000], generic_targets[:2000]
    normal_images, normal_targets = np.concatenate([generic_images, np.repeat(board_images, args.board_repeats, axis=0)]), np.concatenate([generic_targets, np.repeat(board_targets, args.board_repeats, axis=0)])
    tiny_images, tiny_targets = np.repeat(tiny_images, args.tiny_repeats, axis=0), np.repeat(tiny_targets, args.tiny_repeats, axis=0)
    normal_images, normal_targets = make_scale_augmented_training_set(normal_images, normal_targets)
    tiny_images, tiny_targets = make_scale_augmented_training_set(tiny_images, tiny_targets)
    images = resize_cpu(np.concatenate([tiny_images, normal_images]))
    targets = np.concatenate([tiny_targets, normal_targets])
    domains = np.concatenate([np.zeros(len(tiny_targets), dtype=np.float32), np.ones(len(normal_targets), dtype=np.float32)])[:, None]
    tiny_heatmaps = np.zeros((len(targets), HEATMAP_VALUES), dtype=np.float32)
    normal_heatmaps = np.zeros((len(targets), NORMAL_VALUES), dtype=np.float32)
    tiny_count = len(tiny_targets)
    tiny_heatmaps[:tiny_count] = heatmaps(tiny_targets, HEATMAP_SIZE)
    normal_heatmaps[tiny_count:] = heatmaps(normal_targets, NORMAL_SIZE)
    contract_targets = np.concatenate([tiny_heatmaps, normal_heatmaps, targets[:, :4], domains], axis=1).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((images, contract_targets)).shuffle(len(images), seed=42).batch(8).prefetch(tf.data.AUTOTUNE)
    print("training", images.shape, contract_targets.shape)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=DomainHeatmapLoss())
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=DomainHeatmapLoss())
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    # Save the complete quantized object graph as well as weights.  TFMOT can
    # change wrapper-variable structure between builds, so a bare weights file
    # is not a sufficient reload contract for parity or later deployment work.
    qat.save(args.output / "model_qat.keras", include_optimizer=False)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images, args.output / "model_int8.tflite")
    report: dict[str, object] = {"train_samples": int(len(images)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        tiny_head, normal_head, geometry = predict_int8(args.output / "model_int8.tflite", resize_cpu(test_images))
        use_tiny = np.max(geometry[:, 2:4], axis=1) < 0.15
        centers = np.where(use_tiny[:, None], decode_sharp(tiny_head), decode_normal(normal_head))
        predictions = np.concatenate([centers, geometry[:, 2:4], np.ones((len(geometry), 1), dtype=np.float32)], axis=1)
        metrics = _metrics(predictions, test_targets)
        metrics["tiny_head_count"] = int(use_tiny.sum())
        report["tests"][zip_name] = metrics
        print(zip_name, metrics)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
