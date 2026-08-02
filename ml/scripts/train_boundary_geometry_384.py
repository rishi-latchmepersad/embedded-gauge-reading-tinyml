#!/usr/bin/env python3
"""Train a QAT boundary-plus-geometry gauge-face localizer.

The network predicts a high-resolution elliptical rim likelihood and a coarse
ellipse proposal.  A small classical decoder then fits the rim likelihood,
which makes the center depend on the face boundary instead of only on pooled
features.
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
from train_ellipse_robust_384 import BOARD_TRAIN_ZIPS, load_zips, make_scale_augmented_training_set

IMAGE_SIZE = 384
MAP_SIZE = 96
MAP_VALUES = MAP_SIZE * MAP_SIZE
SEED = 42


def configure_gpu() -> None:
    """Limit TensorFlow's visible GPU allocation to 15 GB."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def block(x: tf.Tensor, filters: int, stride: int, name: str) -> tf.Tensor:
    """Apply a quantization-friendly convolutional block."""
    layers = keras.layers
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(epsilon=1e-3, name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def build_model() -> keras.Model:
    """Build a compact encoder-decoder with boundary and scalar outputs."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    skips: list[tf.Tensor] = []
    x = inputs
    # why: the shallow skips preserve small gauge rims while the bottleneck
    # provides enough context to reject dial markings and panel edges.
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = block(x, filters, 2, f"enc{stage}_down")
        x = block(x, filters, 1, f"enc{stage}_refine")
        skips.append(x)
    geometry = layers.GlobalAveragePooling2D(name="geometry_gap")(x)
    geometry = layers.Dense(64, activation="relu", name="geometry_shared")(geometry)
    geometry = layers.Dense(4, activation="sigmoid", name="geometry")(geometry)
    # why: this decoder finishes at 96x96, retaining four times more spatial
    # evidence than a 24x24 bottleneck while remaining small for the NPU.
    for stage, filters, skip_index in ((0, 48, 3), (1, 32, 2), (2, 24, 1)):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"up{stage}")(x)
        x = layers.Concatenate(name=f"join{stage}")([x, skips[skip_index]])
        x = block(x, filters, 1, f"dec{stage}")
    boundary = layers.Conv2D(1, 1, activation="sigmoid", name="boundary")(x)
    boundary = layers.Flatten(name="boundary_flatten")(boundary)
    outputs = layers.Concatenate(name="contract")([boundary, geometry])
    return keras.Model(inputs, outputs, name="boundary_geometry_ellipse_384")


def make_targets(targets: np.ndarray) -> np.ndarray:
    """Rasterize a thin soft ellipse rim and append normalized geometry."""
    coords = (np.arange(MAP_SIZE, dtype=np.float32) + 0.5) / MAP_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    cx = targets[:, 0, None, None]
    cy = targets[:, 1, None, None]
    rx = np.maximum(targets[:, 2, None, None], 1e-3)
    ry = np.maximum(targets[:, 3, None, None], 1e-3)
    radial = np.sqrt(((xx[None] - cx) / rx) ** 2 + ((yy[None] - cy) / ry) ** 2)
    # why: a soft 2-pixel normalized band gives gradients even when a tiny
    # rim falls between output pixels, unlike a binary one-pixel contour.
    rim = np.exp(-((radial - 1.0) ** 2) / (2.0 * 0.035**2)).astype(np.float32)
    return np.concatenate([rim.reshape(len(targets), MAP_VALUES), targets[:, :4]], axis=1)


class BoundaryGeometryLoss(keras.losses.Loss):
    """Weight sparse rim evidence while keeping the proposal geometrically stable."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return focal-like rim loss plus smooth geometry loss."""
        true_map, pred_map = y_true[:, :MAP_VALUES], y_pred[:, :MAP_VALUES]
        true_geometry, pred_geometry = y_true[:, MAP_VALUES:], y_pred[:, MAP_VALUES:]
        # why: most map pixels are background; positive weighting prevents the
        # network from winning by predicting an empty frame.
        weights = 1.0 + 24.0 * true_map
        clipped = tf.clip_by_value(pred_map, 1e-5, 1.0 - 1e-5)
        bce = -(true_map * tf.math.log(clipped) + (1.0 - true_map) * tf.math.log(1.0 - clipped))
        boundary = tf.reduce_mean(weights * bce, axis=-1)
        error = tf.abs(true_geometry - pred_geometry)
        geometry = tf.reduce_mean(tf.where(error < 0.03, 0.5 * tf.square(error) / 0.03, error - 0.015), axis=-1)
        return boundary + 8.0 * geometry


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export the QAT model as a fully integer TFLite graph."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield varied training frames for activation calibration."""
        rng = np.random.default_rng(SEED)
        for index in rng.choice(len(images), min(512, len(images)), replace=False):
            yield [images[index : index + 1].astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model_path: Path, images: np.ndarray) -> np.ndarray:
    """Run the exported integer model and dequantize its contract."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    input_scale, input_zero = inp["quantization"]
    output_scale, output_zero = out["quantization"]
    predictions: list[np.ndarray] = []
    for image in images:
        quantized = np.clip(np.round(image / input_scale + input_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        predictions.append((raw - output_zero) * output_scale)
    return np.asarray(predictions, dtype=np.float32)


def decode_rim(contract: np.ndarray) -> np.ndarray:
    """Fit an ellipse to high-confidence rim pixels, falling back to geometry."""
    coords = (np.arange(MAP_SIZE, dtype=np.float32) + 0.5) / MAP_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    decoded: list[np.ndarray] = []
    for row in contract:
        heat = np.clip(row[:MAP_VALUES].reshape(MAP_SIZE, MAP_SIZE), 0.0, 1.0)
        fallback = row[MAP_VALUES:]
        # why: retain enough points for tiny gauges while excluding weak
        # background responses that destabilize fitEllipse.
        threshold = max(0.18, float(np.quantile(heat, 0.985)))
        active = np.argwhere(heat >= threshold)
        points = np.column_stack([active[:, 1], active[:, 0]]).astype(np.float32) if len(active) else np.empty((0, 2), np.float32)
        if len(points) >= 12:
            try:
                (cx, cy), (major, minor), _ = cv2.fitEllipse(points.reshape(-1, 1, 2))
                axes = np.sort(np.asarray([major, minor], dtype=np.float32))[::-1] / MAP_SIZE / 2.0
                candidate = np.asarray([(cx + 0.5) / MAP_SIZE, (cy + 0.5) / MAP_SIZE, axes[0], axes[1]], dtype=np.float32)
                if np.all(np.isfinite(candidate)) and 0.03 < candidate[2] < 0.8 and 0.03 < candidate[3] < 0.8:
                    # why: coarse geometry stabilizes sparse/partial visible
                    # rims, while the learned rim supplies the center detail.
                    decoded.append(0.75 * candidate + 0.25 * fallback)
                    continue
            except cv2.error:
                pass
        decoded.append(fallback.astype(np.float32))
    return np.asarray(decoded, dtype=np.float32)


def main() -> None:
    """Train, QAT-finetune, export, and score all three test archives."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--qat-epochs", type=int, default=6)
    parser.add_argument("--tiny-repeats", type=int, default=60)
    parser.add_argument("--board-repeats", type=int, default=3)
    parser.add_argument("--highres", action="store_true", help="Use a 512px input and 128px rim map.")
    args = parser.parse_args()
    global IMAGE_SIZE, MAP_SIZE, MAP_VALUES
    if args.highres:
        # why: generic test_1 faces are large enough that the 96px rim map is
        # the spatial bottleneck; preserve the same decoder at four-times map
        # area rather than adding another pooled regression head.
        IMAGE_SIZE, MAP_SIZE = 512, 128
        MAP_VALUES = MAP_SIZE * MAP_SIZE
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    generic_images, generic_targets = load_zips(["train_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(BOARD_TRAIN_ZIPS, labels=("temp_dial",))
    generic_images, generic_targets = generic_images[:3000], generic_targets[:3000]
    images = np.concatenate([generic_images, np.repeat(tiny_images, args.tiny_repeats, axis=0), np.repeat(board_images, args.board_repeats, axis=0)])
    targets = np.concatenate([generic_targets, np.repeat(tiny_targets, args.tiny_repeats, axis=0), np.repeat(board_targets, args.board_repeats, axis=0)])
    images, targets = make_scale_augmented_training_set(images, targets)
    # why: large 512px batches can trigger a TensorFlow GPU resize-kernel
    # overflow before training; preprocessing belongs on CPU and avoids that
    # unrelated failure while keeping the network itself on the capped GPU.
    with tf.device("/CPU:0"):
        images = tf.image.resize(images, [IMAGE_SIZE, IMAGE_SIZE]).numpy()
    contract_targets = make_targets(targets)
    batch_size = 4 if args.highres else 16
    dataset = tf.data.Dataset.from_tensor_slices((images, contract_targets)).shuffle(len(images), seed=SEED).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print("training", images.shape, contract_targets.shape, flush=True)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=BoundaryGeometryLoss())
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=BoundaryGeometryLoss())
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images, args.output / "model_int8.tflite")
    report: dict[str, object] = {"train_samples": int(len(images)), "map_size": MAP_SIZE, "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        contract = predict_int8(args.output / "model_int8.tflite", tf.image.resize(test_images, [IMAGE_SIZE, IMAGE_SIZE]).numpy())
        decoded = decode_rim(contract)
        predictions = np.concatenate([decoded, np.ones((len(decoded), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
