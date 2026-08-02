#!/usr/bin/env python3
"""Train a universal center-heatmap and ellipse-rim model with QAT.

The center target is deliberately a narrow Gaussian rather than the centroid
of a filled mask.  A separate soft rim target preserves scale/shape evidence,
while the scalar head supplies stable radii when the rim is only a few pixels
wide on the low-resolution output grid.
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
from train_ellipse_robust_384 import (
    BOARD_TRAIN_ZIPS,
    IMAGE_SIZE,
    SEED,
    _block,
    load_zips,
    make_scale_augmented_training_set,
)


HEATMAP_SIZE = 96
HEATMAP_VALUES = HEATMAP_SIZE * HEATMAP_SIZE


def configure_gpu() -> None:
    """Cap TensorFlow's visible GPU allocation at 15 GB."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def build_model(input_channels: int = 1) -> keras.Model:
    """Build a compact shared encoder with center, rim, and geometry heads."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, input_channels), name="image")
    skips: list[tf.Tensor] = []
    x = inputs
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = _block(x, filters, 2, f"enc{stage}_down")
        x = _block(x, filters, 1, f"enc{stage}_refine")
        skips.append(x)

    # why: the bottleneck gives radii global context without making the spatial
    # center head depend on a global-average-pooling location prior.
    geometry = layers.GlobalAveragePooling2D(name="geometry_gap")(x)
    geometry = layers.Dense(48, activation="relu", name="geometry_shared")(geometry)
    geometry = layers.Dense(4, activation="sigmoid", name="geometry")(geometry)

    for stage, filters, skip_index in ((0, 48, 3), (1, 32, 2), (2, 24, 1)):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"up{stage}")(x)
        x = layers.Concatenate(name=f"join{stage}")([x, skips[skip_index]])
        x = _block(x, filters, 1, f"dec{stage}")

    # The 96x96 output keeps enough spatial precision for an 8 px full-frame
    # center tolerance while remaining small enough for NPU SRAM.
    center = layers.Conv2D(1, 1, activation="sigmoid", name="center_heatmap")(x)
    rim = layers.Conv2D(1, 1, activation="sigmoid", name="rim_heatmap")(x)
    flat_center = layers.Flatten(name="center_flatten")(center)
    flat_rim = layers.Flatten(name="rim_flatten")(rim)
    output = layers.Concatenate(name="center_rim_geometry")(
        [flat_center, flat_rim, geometry]
    )
    return keras.Model(inputs, output, name="ellipse_center_rim_384")


def make_edge_features_tensor(tensor: tf.Tensor) -> tf.Tensor:
    """Add local contrast and Sobel magnitude channels to a tensor batch."""
    # why: a large local average removes the wall illumination trend while
    # retaining the dial rim, which is the useful cue in overexposed captures.
    blur = tf.nn.avg_pool2d(tensor, ksize=15, strides=1, padding="SAME")
    local_contrast = tf.clip_by_value(0.5 + 2.0 * (tensor - blur), 0.0, 1.0)
    sobel = tf.image.sobel_edges(tensor)
    edge_magnitude = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))
    edge_magnitude = tf.clip_by_value(2.0 * edge_magnitude, 0.0, 1.0)
    return tf.concat([tensor, local_contrast, edge_magnitude], axis=-1)


def make_targets(geometry: np.ndarray) -> np.ndarray:
    """Rasterize Gaussian centers and elliptical rims beside scalar geometry."""
    coords = (np.arange(HEATMAP_SIZE, dtype=np.float32) + 0.5) / HEATMAP_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    center_maps: list[np.ndarray] = []
    rim_maps: list[np.ndarray] = []
    for cx, cy, rx, ry in geometry[:, :4]:
        # why: sigma is fixed in output-grid units so center supervision stays
        # equally sharp for large and tiny gauges.
        sigma = 1.6 / HEATMAP_SIZE
        center = np.exp(-0.5 * (((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2))
        ellipse_distance = ((xx - cx) / max(float(rx), 1e-3)) ** 2 + ((yy - cy) / max(float(ry), 1e-3)) ** 2
        # why: a band, rather than a filled disk, makes radius evidence explicit
        # and avoids the filled-mask centroid/radius tradeoff.
        rim = np.exp(-0.5 * ((ellipse_distance - 1.0) / 0.16) ** 2)
        center_maps.append(center.astype(np.float32))
        rim_maps.append(rim.astype(np.float32))
    return np.concatenate(
        [
            np.asarray(center_maps).reshape(-1, HEATMAP_VALUES),
            np.asarray(rim_maps).reshape(-1, HEATMAP_VALUES),
            geometry[:, :4].astype(np.float32),
        ],
        axis=1,
    )


class CenterRimLoss(keras.losses.Loss):
    """Weight sparse center peaks more strongly than diffuse rim pixels."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return center focal-MSE, rim BCE, and scalar geometry loss."""
        true_center = tf.reshape(y_true[:, :HEATMAP_VALUES], (-1, HEATMAP_SIZE, HEATMAP_SIZE))
        pred_center = tf.reshape(y_pred[:, :HEATMAP_VALUES], (-1, HEATMAP_SIZE, HEATMAP_SIZE))
        true_rim = tf.reshape(y_true[:, HEATMAP_VALUES : 2 * HEATMAP_VALUES], (-1, HEATMAP_SIZE, HEATMAP_SIZE))
        pred_rim = tf.reshape(y_pred[:, HEATMAP_VALUES : 2 * HEATMAP_VALUES], (-1, HEATMAP_SIZE, HEATMAP_SIZE))
        true_geometry = y_true[:, 2 * HEATMAP_VALUES :]
        pred_geometry = y_pred[:, 2 * HEATMAP_VALUES :]

        # why: the positive center peak is tiny relative to the background;
        # square-error weighting prevents an all-zero heatmap solution.
        center_weight = 1.0 + 12.0 * true_center
        center_loss = tf.reduce_mean(center_weight * tf.square(true_center - pred_center), axis=(1, 2))
        rim_weight = 1.0 + 3.0 * true_rim
        # why: the Keras helper treats the final spatial axis as a class axis
        # for rank-3 tensors; write the pixelwise BCE explicitly so the 96x96
        # weight map remains broadcast-compatible.
        clipped_rim = tf.clip_by_value(pred_rim, 1e-5, 1.0 - 1e-5)
        rim_bce = -(true_rim * tf.math.log(clipped_rim) + (1.0 - true_rim) * tf.math.log(1.0 - clipped_rim))
        rim_loss = tf.reduce_mean(rim_weight * rim_bce, axis=(1, 2))
        geometry_loss = tf.reduce_sum(tf.abs(true_geometry - pred_geometry), axis=-1)
        return 20.0 * center_loss + 2.0 * rim_loss + 3.0 * geometry_loss

    def get_config(self) -> dict[str, object]:
        """Return the serializable loss configuration."""
        return super().get_config()


def export_int8(model: keras.Model, images: np.ndarray, output: Path, edge_features: bool = False) -> None:
    """Export a fully integer TFLite model from varied training frames."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield representative images for activation calibration."""
        rng = np.random.default_rng(SEED)
        for index in rng.choice(len(images), min(512, len(images)), replace=False):
            sample = tf.convert_to_tensor(images[index : index + 1].astype(np.float32))
            if edge_features:
                sample = make_edge_features_tensor(sample)
            yield [sample.numpy()]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model_path: Path, images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the integer model and return center maps, rim maps, and geometry."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    values: list[np.ndarray] = []
    for image in images:
        quantized = np.clip(np.round(image / input_scale + input_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])[0].astype(np.float32)
        values.append((raw - output_zero) * output_scale)
    result = np.asarray(values, dtype=np.float32)
    return (
        result[:, :HEATMAP_VALUES].reshape(-1, HEATMAP_SIZE, HEATMAP_SIZE, 1),
        result[:, HEATMAP_VALUES : 2 * HEATMAP_VALUES].reshape(-1, HEATMAP_SIZE, HEATMAP_SIZE, 1),
        result[:, 2 * HEATMAP_VALUES :],
    )


def decode_center(center_maps: np.ndarray) -> np.ndarray:
    """Decode the center using a sharpened intensity-weighted expectation."""
    coords = (np.arange(HEATMAP_SIZE, dtype=np.float32) + 0.5) / HEATMAP_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    decoded: list[list[float]] = []
    for heatmap in center_maps[..., 0]:
        weights = np.maximum(heatmap - 0.15, 0.0) ** 3.0
        total = max(float(weights.sum()), 1e-6)
        decoded.append([float((weights * xx).sum() / total), float((weights * yy).sum() / total)])
    return np.asarray(decoded, dtype=np.float32)


def main() -> None:
    """Train, QAT-finetune, export, and score the universal candidate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--qat-epochs", type=int, default=12)
    parser.add_argument("--tiny-repeats", type=int, default=100)
    parser.add_argument("--board-repeats", type=int, default=4)
    parser.add_argument("--edge-features", action="store_true")
    parser.add_argument("--clean-board-split", action="store_true")
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
    board_zips = ["initial_temp_gauge/board_captures_1.zip"] if args.clean_board_split else BOARD_TRAIN_ZIPS
    board_images, board_targets = load_zips(board_zips, labels=("temp_dial",))
    generic_images, generic_targets = generic_images[:4000], generic_targets[:4000]
    images = np.concatenate([generic_images, np.repeat(tiny_images, args.tiny_repeats, axis=0), np.repeat(board_images, args.board_repeats, axis=0)])
    targets = np.concatenate([generic_targets, np.repeat(tiny_targets, args.tiny_repeats, axis=0), np.repeat(board_targets, args.board_repeats, axis=0)])
    images, targets = make_scale_augmented_training_set(images, targets)
    with tf.device("/CPU:0"):
        images = tf.image.resize(images, [IMAGE_SIZE, IMAGE_SIZE]).numpy()
    contract_targets = make_targets(targets)
    dataset = tf.data.Dataset.from_tensor_slices((images, contract_targets)).shuffle(len(images), seed=SEED).batch(32)
    if args.edge_features:
        # why: compute derived channels per batch so the host stores only the
        # original grayscale set instead of a second multi-channel copy.
        dataset = dataset.map(
            lambda batch, labels: (make_edge_features_tensor(batch), labels),
            num_parallel_calls=1,
        )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    print("training", images.shape, contract_targets.shape, "board_source=clean_board_1_only" if args.clean_board_split else "board_source=all")

    model = build_model(input_channels=3 if args.edge_features else 1)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=CenterRimLoss())
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=CenterRimLoss())
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images, args.output / "model_int8.tflite", edge_features=args.edge_features)

    report: dict[str, object] = {"train_samples": int(len(images)), "heatmap_size": HEATMAP_SIZE, "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        with tf.device("/CPU:0"):
            test_images = tf.image.resize(test_images, [IMAGE_SIZE, IMAGE_SIZE]).numpy()
            if args.edge_features:
                test_images = make_edge_features_tensor(tf.convert_to_tensor(test_images)).numpy()
        center, _, geometry = predict_int8(args.output / "model_int8.tflite", test_images)
        predictions = np.concatenate([decode_center(center), geometry[:, 2:4], np.ones((len(geometry), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
