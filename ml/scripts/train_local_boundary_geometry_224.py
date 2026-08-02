#!/usr/bin/env python3
"""Train a local QAT rim decoder after a routed coarse gauge proposal."""

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
from eval_routed_face_crop_refiner import coarse_route, crop_from_coarse
from train_ellipse_face_crop_224 import make_face_crops
from train_ellipse_robust_384 import BOARD_TRAIN_ZIPS, load_zips, make_scale_augmented_training_set

IMAGE_SIZE = 224
MAP_SIZE = 56
MAP_VALUES = MAP_SIZE * MAP_SIZE
SEED = 42


def configure_gpu() -> None:
    """Limit TensorFlow to the project-approved 15 GB GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])


def block(x: tf.Tensor, filters: int, stride: int, name: str) -> tf.Tensor:
    """Apply one integer-friendly convolutional block."""
    layers = keras.layers
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(epsilon=1e-3, name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def build_model() -> keras.Model:
    """Build a local spatial decoder with a scalar fallback proposal."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="local_crop")
    skips: list[tf.Tensor] = []
    x = inputs
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = block(x, filters, 2, f"enc{stage}_down")
        x = block(x, filters, 1, f"enc{stage}_refine")
        skips.append(x)
    geometry = layers.GlobalAveragePooling2D(name="geometry_gap")(x)
    geometry = layers.Dense(32, activation="relu", name="geometry_shared")(geometry)
    geometry = layers.Dense(4, activation="sigmoid", name="geometry")(geometry)
    for stage, filters, skip_index in ((0, 48, 3), (1, 32, 2), (2, 24, 1)):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"up{stage}")(x)
        x = layers.Concatenate(name=f"join{stage}")([x, skips[skip_index]])
        x = block(x, filters, 1, f"dec{stage}")
    boundary = layers.Flatten(name="boundary_flatten")(layers.Conv2D(1, 1, activation="sigmoid", name="boundary")(x))
    return keras.Model(inputs, layers.Concatenate(name="contract")([boundary, geometry]), name="local_boundary_geometry_224")


def make_targets(local_targets: np.ndarray) -> np.ndarray:
    """Rasterize the local normalized ellipse rim and append local geometry."""
    coords = (np.arange(MAP_SIZE, dtype=np.float32) + 0.5) / MAP_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    cx = local_targets[:, 0, None, None]
    cy = local_targets[:, 1, None, None]
    rx = np.maximum(local_targets[:, 2, None, None], 1e-3)
    ry = np.maximum(local_targets[:, 3, None, None], 1e-3)
    radial = np.sqrt(((xx[None] - cx) / rx) ** 2 + ((yy[None] - cy) / ry) ** 2)
    rim = np.exp(-((radial - 1.0) ** 2) / (2.0 * 0.045**2)).astype(np.float32)
    return np.concatenate([rim.reshape(len(local_targets), MAP_VALUES), local_targets[:, :4]], axis=1)


class Loss(keras.losses.Loss):
    """Balance sparse rim likelihood against local center and radius error."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return weighted boundary and robust geometry losses."""
        true_map, pred_map = y_true[:, :MAP_VALUES], y_pred[:, :MAP_VALUES]
        true_geom, pred_geom = y_true[:, MAP_VALUES:], y_pred[:, MAP_VALUES:]
        weights = 1.0 + 20.0 * true_map
        pred_map = tf.clip_by_value(pred_map, 1e-5, 1.0 - 1e-5)
        bce = -(true_map * tf.math.log(pred_map) + (1.0 - true_map) * tf.math.log(1.0 - pred_map))
        boundary = tf.reduce_mean(weights * bce, axis=-1)
        error = tf.abs(true_geom - pred_geom)
        geometry = tf.reduce_mean(tf.where(error < 0.04, 0.5 * tf.square(error) / 0.04, error - 0.02), axis=-1)
        return boundary + 10.0 * geometry


def export_int8(model: keras.Model, crops: np.ndarray, output: Path) -> None:
    """Export the QAT model with int8 inputs and outputs."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield representative local crops."""
        rng = np.random.default_rng(SEED)
        for index in rng.choice(len(crops), min(512, len(crops)), replace=False):
            yield [crops[index : index + 1].astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model_path: Path, crops: np.ndarray) -> np.ndarray:
    """Run local int8 inference and return the dequantized contract."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    values: list[np.ndarray] = []
    for crop in crops:
        q = np.clip(np.round(crop / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], q[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        values.append((raw - out_zero) * out_scale)
    return np.asarray(values, dtype=np.float32)


def decode_rim(contract: np.ndarray) -> np.ndarray:
    """Fit an ellipse to local rim pixels and use scalar geometry as fallback."""
    decoded: list[np.ndarray] = []
    for row in contract:
        heat = np.clip(row[:MAP_VALUES].reshape(MAP_SIZE, MAP_SIZE), 0.0, 1.0)
        fallback = row[MAP_VALUES:]
        threshold = max(0.20, float(np.quantile(heat, 0.985)))
        points = np.argwhere(heat >= threshold)
        if len(points) >= 12:
            try:
                (cx, cy), (major, minor), _ = cv2.fitEllipse(np.column_stack([points[:, 1], points[:, 0]]).astype(np.float32).reshape(-1, 1, 2))
                axes = np.sort(np.asarray([major, minor], dtype=np.float32))[::-1] / MAP_SIZE / 2.0
                candidate = np.asarray([(cx + 0.5) / MAP_SIZE, (cy + 0.5) / MAP_SIZE, axes[0], axes[1]], dtype=np.float32)
                if np.all(np.isfinite(candidate)) and candidate[2] < 0.8 and candidate[3] < 0.8:
                    decoded.append(0.8 * candidate + 0.2 * fallback)
                    continue
            except cv2.error:
                pass
        decoded.append(fallback.astype(np.float32))
    return np.asarray(decoded, dtype=np.float32)


def route_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the verified coarse-route model namespace."""
    return argparse.Namespace(
        low_model=args.low_model, high_model=args.high_model, radius_model=args.radius_model,
        gate_model=args.gate_model, radius_domain_model=args.radius_domain_model,
        board_model=args.board_model, tiny_model=args.tiny_model,
    )


def main() -> None:
    """Train, export, and evaluate the routed local boundary model."""
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--qat-epochs", type=int, default=5)
    parser.add_argument("--generic-limit", type=int, default=2000)
    parser.add_argument("--tiny-repeats", type=int, default=40)
    parser.add_argument("--board-repeats", type=int, default=3)
    parser.add_argument("--padding", type=float, default=3.5)
    parser.add_argument("--low-model", type=Path, default=root / "artifacts/gauge_ellipse_mask_center_scaleconf_384_aux_v1/model_int8.tflite")
    parser.add_argument("--high-model", type=Path, default=root / "artifacts/gauge_ellipse_center_heatmap_640_v1/model_int8.tflite")
    parser.add_argument("--radius-model", type=Path, default=root / "artifacts/gauge_ellipse_scalar_640_v2/model_int8.tflite")
    parser.add_argument("--gate-model", type=Path, default=root / "artifacts/gauge_ellipse_domain_classifier_640_v1/model_int8.tflite")
    parser.add_argument("--radius-domain-model", type=Path, default=root / "artifacts/gauge_ellipse_radius_domains_640_v1/model_int8.tflite")
    parser.add_argument("--board-model", type=Path, default=root / "artifacts/gauge_ellipse_board_heatmap_640_v1/model_int8.tflite")
    parser.add_argument("--tiny-model", type=Path, default=root / "artifacts/gauge_ellipse_domain_heatmaps_640_v1/model_int8.tflite")
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(BOARD_TRAIN_ZIPS, labels=("temp_dial",))
    generic_images, generic_targets = generic_images[:args.generic_limit], generic_targets[:args.generic_limit]
    images = np.concatenate([generic_images, np.repeat(tiny_images, args.tiny_repeats, axis=0), np.repeat(board_images, args.board_repeats, axis=0)])
    targets = np.concatenate([generic_targets, np.repeat(tiny_targets, args.tiny_repeats, axis=0), np.repeat(board_targets, args.board_repeats, axis=0)])
    images, targets = make_scale_augmented_training_set(images, targets)
    crops, local_targets = make_face_crops(images, targets, jitter=True, seed=SEED, padding=args.padding, jitter_fraction=0.12)
    contract_targets = make_targets(local_targets)
    dataset = tf.data.Dataset.from_tensor_slices((crops, contract_targets)).shuffle(len(crops), seed=SEED).batch(16).prefetch(tf.data.AUTOTUNE)
    print("training", crops.shape, contract_targets.shape, flush=True)
    model = build_model(); model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=Loss()); model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model); qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=Loss()); qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True); qat.save_weights(args.output / "model_qat.weights.h5"); export_int8(qat, crops, args.output / "model_int8.tflite")
    routed = route_args(args); report: dict[str, object] = {"train_samples": int(len(crops)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        coarse, tiny = coarse_route(test_images, routed)
        local_crops, metadata = crop_from_coarse(test_images, coarse, args.padding)
        local = decode_rim(predict_int8(args.output / "model_int8.tflite", local_crops))
        left, top, side = metadata.T
        refined = np.stack([left + local[:, 0] * side, top + local[:, 1] * side, local[:, 2] * side, local[:, 3] * side, np.ones(len(local), np.float32)], axis=1)
        selected = coarse.copy(); selected[tiny, :4] = 0.25 * coarse[tiny, :4] + 0.75 * refined[tiny, :4]; selected[tiny, 4] = 1.0
        report["tests"][zip_name] = {"coarse": _metrics(coarse, test_targets), "local_boundary": _metrics(refined, test_targets), "selected_tiny": _metrics(selected, test_targets)}
        print(zip_name, json.dumps(report["tests"][zip_name]), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
