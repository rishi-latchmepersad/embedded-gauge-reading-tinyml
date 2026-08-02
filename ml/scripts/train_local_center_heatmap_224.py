#!/usr/bin/env python3
"""Train a routed local center-heatmap refiner for gauge faces.

This experiment is intentionally different from the current scalar ellipse
refiners: the network only has to place the center peak correctly inside a
coarse crop, while the coarse router and existing radius specialists keep the
radius estimate stable enough for the tolerated variation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embedded_gauge_reading_tinyml.heatmap_losses import (  # noqa: E402
    focal_heatmap_loss,
    softargmax_coordinate_loss,
)
from eval_ellipse_all_test_sets import _load_zip, _metrics  # noqa: E402
from eval_routed_face_crop_refiner import coarse_route, crop_from_coarse  # noqa: E402
from train_ellipse_face_crop_224 import make_face_crops  # noqa: E402
from train_ellipse_robust_384 import BOARD_TRAIN_ZIPS, load_zips, make_scale_augmented_training_set  # noqa: E402

IMAGE_SIZE = 224
HEATMAP_SIZE = 56
HEATMAP_VALUES = HEATMAP_SIZE * HEATMAP_SIZE
SEED = 42


def configure_gpu() -> None:
    """Limit TensorFlow to the project-approved 15 GB GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def block(x: tf.Tensor, filters: int, stride: int, name: str) -> tf.Tensor:
    """Apply one quantization-friendly convolutional block."""
    layers = keras.layers
    x = layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(epsilon=1e-3, name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def build_model() -> keras.Model:
    """Build a small U-Net that predicts a single center heatmap."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="local_crop")
    skips: list[tf.Tensor] = []
    x = inputs
    # why: the local crop already contains a good coarse proposal, so a compact
    # encoder-decoder can spend its capacity on sub-pixel center placement.
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = block(x, filters, 2, f"enc{stage}_down")
        x = block(x, filters, 1, f"enc{stage}_refine")
        skips.append(x)
    for stage, filters, skip_index in ((0, 48, 3), (1, 32, 2), (2, 24, 1)):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"up{stage}")(x)
        x = layers.Concatenate(name=f"join{stage}")([x, skips[skip_index]])
        x = block(x, filters, 1, f"dec{stage}")
    x = block(x, 32, 1, "head_refine")
    heatmap = layers.Conv2D(1, 1, activation="sigmoid", name="center_heatmap")(x)
    return keras.Model(inputs, heatmap, name="local_center_heatmap_224")


def make_targets(local_targets: np.ndarray) -> np.ndarray:
    """Rasterize the local center into a single Gaussian heatmap channel."""
    coords = (np.arange(HEATMAP_SIZE, dtype=np.float32) + 0.5) / HEATMAP_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    cx = local_targets[:, 0, None, None]
    cy = local_targets[:, 1, None, None]
    # why: a slightly soft peak keeps the QAT model from collapsing to a sharp
    # single-pixel spike that is brittle after quantization.
    sigma = 0.020
    heatmap = np.exp(-((xx[None] - cx) ** 2 + (yy[None] - cy) ** 2) / (2.0 * sigma**2))
    return heatmap.astype(np.float32).reshape(len(local_targets), HEATMAP_SIZE, HEATMAP_SIZE, 1)


class HeatmapLoss(keras.losses.Loss):
    """Combine focal pixel supervision with coordinate accuracy."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return a weighted heatmap loss plus a soft-argmax coordinate term."""
        return focal_heatmap_loss(y_true, y_pred) + 0.35 * softargmax_coordinate_loss(y_true, y_pred)


def export_int8(model: keras.Model, crops: np.ndarray, output: Path) -> None:
    """Export the QAT model as a fully integer TFLite graph."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield representative local crops for calibration."""
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
    """Run the int8 model and return dequantized heatmaps."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    predictions: list[np.ndarray] = []
    for crop in crops:
        quantized = np.clip(np.round(crop / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        predictions.append((raw - out_zero) * out_scale)
    return np.asarray(predictions, dtype=np.float32)


def decode_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """Decode the predicted center heatmap using a local weighted centroid."""
    coords = (np.arange(HEATMAP_SIZE, dtype=np.float32) + 0.5) / HEATMAP_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    centers: list[list[float]] = []
    for heatmap in heatmaps[..., 0]:
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        y0, y1 = max(0, y - 4), min(HEATMAP_SIZE, y + 5)
        x0, x1 = max(0, x - 4), min(HEATMAP_SIZE, x + 5)
        local = np.maximum(heatmap[y0:y1, x0:x1] - 0.03, 0.0) ** 2
        total = float(local.sum())
        if total > 1e-6:
            centers.append([
                float((local * xx[y0:y1, x0:x1]).sum() / total),
                float((local * yy[y0:y1, x0:x1]).sum() / total),
            ])
        else:
            centers.append([float((x + 0.5) / HEATMAP_SIZE), float((y + 0.5) / HEATMAP_SIZE)])
    return np.asarray(centers, dtype=np.float32)


def restore_center_predictions(
    center_crop_predictions: np.ndarray,
    targets: np.ndarray,
    padding: float,
) -> np.ndarray:
    """Map crop-relative center predictions back to the original full frame."""
    restored: list[list[float]] = []
    for prediction, target in zip(center_crop_predictions, targets):
        cx, cy, rx, ry = target[:4]
        side = max(padding * float(rx), padding * float(ry), 0.16)
        left, top = cx - side / 2.0, cy - side / 2.0
        restored.append([left + prediction[0] * side, top + prediction[1] * side, rx, ry, target[4]])
    return np.asarray(restored, dtype=np.float32)


def route_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the verified coarse-route model namespace."""
    return argparse.Namespace(
        low_model=args.low_model,
        high_model=args.high_model,
        radius_model=args.radius_model,
        gate_model=args.gate_model,
        radius_domain_model=args.radius_domain_model,
        board_model=args.board_model,
        tiny_model=args.tiny_model,
    )


def main() -> None:
    """Train, export, and evaluate the routed local center heatmap refiner."""
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
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
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
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
    crops, local_targets = make_face_crops(
        images,
        targets,
        jitter=True,
        seed=SEED,
        padding=args.padding,
        jitter_fraction=0.12,
    )
    heatmaps = make_targets(local_targets)
    dataset = (
        tf.data.Dataset.from_tensor_slices((crops, heatmaps))
        .shuffle(len(crops), seed=SEED)
        .batch(16)
        .prefetch(tf.data.AUTOTUNE)
    )
    print("training", crops.shape, heatmaps.shape, flush=True)

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=HeatmapLoss())
    model.fit(dataset, epochs=args.epochs, verbose=2)

    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=HeatmapLoss())
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)

    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, crops, args.output / "model_int8.tflite")

    routed = route_args(args)
    report: dict[str, object] = {"train_samples": int(len(crops)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        coarse, tiny = coarse_route(test_images, routed)
        local_crops, metadata = crop_from_coarse(test_images, coarse, args.padding)
        heatmap_predictions = predict_int8(args.output / "model_int8.tflite", local_crops)
        local_centers = decode_heatmaps(heatmap_predictions)
        left, top, side = metadata.T
        refined = np.stack(
            [
                left + local_centers[:, 0] * side,
                top + local_centers[:, 1] * side,
                coarse[:, 2],
                coarse[:, 3],
                np.ones(len(local_centers), dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        selected = coarse.copy()
        selected[tiny, :2] = 0.25 * coarse[tiny, :2] + 0.75 * refined[tiny, :2]
        report["tests"][zip_name] = {
            "coarse": _metrics(coarse, test_targets),
            "local_heatmap": _metrics(refined, test_targets),
            "selected_tiny": _metrics(selected, test_targets),
        }
        print(zip_name, json.dumps(report["tests"][zip_name], indent=2), flush=True)

    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
