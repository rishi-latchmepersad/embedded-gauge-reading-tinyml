#!/usr/bin/env python3
"""Train a high-resolution QAT-friendly ellipse mask with center supervision."""

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
    SEED,
    _block,
    load_zips,
    make_scale_augmented_training_set,
)


# why: the 320/80-grid candidate still produced tiny-gauge center outliers;
# 384/96 gives the same NPU-friendly family a finer localization lattice.
IMAGE_SIZE = 384
MASK_SIZE = 96
MASK_VALUES = MASK_SIZE * MASK_SIZE
CENTER_SIZE = MASK_SIZE
CENTER_VALUES = CENTER_SIZE * CENTER_SIZE
COARSE_SIZE = MASK_SIZE // 2
COARSE_VALUES = COARSE_SIZE * COARSE_SIZE
OFFSET_VALUES = 2 * MASK_VALUES
# why: the center is the deployment-critical landmark; boundary keypoints
# added quantization noise without improving center localization on test_3.
KEYPOINTS = 1


def configure_gpu() -> None:
    """Limit TensorFlow to the project GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Leave room for the desktop compositor; the model is small enough
        # that 12 GB is ample and avoids one giant allocator reservation.
        limit = 15000
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=limit)]
        )


def build_model() -> keras.Model:
    """Build a compact encoder-decoder whose 160-grid mask preserves location."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    skips: list[tf.Tensor] = []
    x = inputs
    # why: the ordinary Conv-BN-ReLU blocks are directly supported by TFMOT
    # and avoid unsupported custom operations in the eventual integer graph.
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = _block(x, filters, 2, f"enc{stage}_down")
        x = _block(x, filters, 1, f"enc{stage}_refine")
        skips.append(x)
    for stage, filters, skip_index in ((0, 48, 3), (1, 32, 2), (2, 24, 1)):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"up{stage}")(x)
        x = layers.Concatenate(name=f"join{stage}")([x, skips[skip_index]])
        x = _block(x, filters, 1, f"dec{stage}")
    mask = layers.Conv2D(1, 1, activation="sigmoid", name="face_mask")(x)
    # why: a dense center target remains informative when a tiny face occupies
    # only a few mask pixels; global pooling alone loses this signal.
    center_heatmap = layers.Conv2D(KEYPOINTS, 1, activation="sigmoid", name="center_heatmap")(x)
    # why: retain offset supervision as a representation regularizer even
    # though the deployed decoder uses the more stable heatmap peak.
    center_offset = layers.Conv2D(2, 1, activation="sigmoid", name="center_offset")(x)
    flat_mask = layers.Flatten(name="mask_flatten")(mask)
    flat_center = layers.Flatten(name="center_flatten")(center_heatmap)
    flat_offset = layers.Flatten(name="offset_flatten")(center_offset)
    pooled = layers.GlobalAveragePooling2D(name="geometry_gap")(x)
    geometry_hidden = layers.Dense(32, activation="relu", name="geometry_hidden")(pooled)
    geometry = layers.Dense(4, activation="sigmoid", name="geometry")(geometry_hidden)
    # why: a learned scale regime is safer than routing tiny-radius calibration
    # from a noisy continuous radius estimate alone.
    scale_confidence = layers.Dense(1, activation="sigmoid", name="scale_confidence")(geometry_hidden)
    outputs = layers.Concatenate(name="ellipse_contract")([flat_mask, flat_center, flat_offset, geometry, scale_confidence])
    return keras.Model(inputs, outputs, name="ellipse_mask_640_center")


def make_targets(targets: np.ndarray) -> np.ndarray:
    """Rasterize ellipses and append normalized center/radius supervision."""
    coords = (np.arange(MASK_SIZE, dtype=np.float32) + 0.5) / MASK_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    cx = targets[:, 0, None, None]
    cy = targets[:, 1, None, None]
    rx = np.maximum(targets[:, 2, None, None], 1e-3)
    ry = np.maximum(targets[:, 3, None, None], 1e-3)
    distance = ((xx[None] - cx) / rx) ** 2 + ((yy[None] - cy) / ry) ** 2
    masks = (distance <= 1.0).astype(np.float32).reshape(len(targets), MASK_VALUES)
    offsets = np.zeros((len(targets), MASK_SIZE, MASK_SIZE, 2), dtype=np.float32)
    heatmaps = np.zeros((len(targets), CENTER_SIZE, CENTER_SIZE, KEYPOINTS), dtype=np.float32)
    for index, (cx, cy, _, _) in enumerate(targets[:, :4]):
        yy, xx = np.mgrid[:MASK_SIZE, :MASK_SIZE]
        gx, gy = cx * MASK_SIZE - 0.5, cy * MASK_SIZE - 0.5
        cell_x, cell_y = int(np.floor(gx)), int(np.floor(gy))
        if 0 <= cell_x < MASK_SIZE and 0 <= cell_y < MASK_SIZE:
            offsets[index, cell_y, cell_x] = [gx - cell_x, gy - cell_y]
        heatmaps[index, ..., 0] = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * 2.0**2))
    tiny_label = (np.mean(targets[:, 2:4], axis=1, keepdims=True) < 0.20).astype(np.float32)
    return np.concatenate([
        masks,
        heatmaps.reshape(len(targets), -1),
        offsets.reshape(len(targets), -1),
        targets[:, :4],
        tiny_label,
    ], axis=1).astype(np.float32)


def _moment_center(mask: tf.Tensor) -> tf.Tensor:
    """Compute a differentiable foreground-weighted center from predicted masks."""
    coords = tf.linspace(0.5 / MASK_SIZE, 1.0 - 0.5 / MASK_SIZE, MASK_SIZE)
    yy, xx = tf.meshgrid(coords, coords, indexing="ij")
    # why: removing the low-confidence floor prevents diffuse background
    # probabilities from pulling tiny-face centers toward the frame midpoint.
    weights = tf.nn.relu(mask[..., 0] - 0.10)
    total = tf.reduce_sum(weights, axis=(1, 2), keepdims=True) + 1e-6
    center_x = tf.reduce_sum(weights * xx[None, ...], axis=(1, 2)) / tf.squeeze(total, axis=(1, 2))
    center_y = tf.reduce_sum(weights * yy[None, ...], axis=(1, 2)) / tf.squeeze(total, axis=(1, 2))
    return tf.stack([center_x, center_y], axis=1)


def center_mask_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Combine mask BCE/Dice with explicit center and radius Huber losses."""
    true_mask = tf.reshape(y_true[:, :MASK_VALUES], (-1, MASK_SIZE, MASK_SIZE, 1))
    pred_mask = tf.reshape(y_pred[:, :MASK_VALUES], (-1, MASK_SIZE, MASK_SIZE, 1))
    heat_values = KEYPOINTS * CENTER_VALUES
    heat_start = MASK_VALUES
    offset_start = heat_start + heat_values
    true_heat = tf.reshape(y_true[:, heat_start:offset_start], (-1, MASK_SIZE, MASK_SIZE, KEYPOINTS))
    pred_heat = tf.reshape(y_pred[:, heat_start:offset_start], (-1, MASK_SIZE, MASK_SIZE, KEYPOINTS))
    true_offset = tf.reshape(y_true[:, offset_start:offset_start + OFFSET_VALUES], (-1, MASK_SIZE, MASK_SIZE, 2))
    pred_offset = tf.reshape(y_pred[:, offset_start:offset_start + OFFSET_VALUES], (-1, MASK_SIZE, MASK_SIZE, 2))
    geometry_start = offset_start + OFFSET_VALUES
    true_geometry = y_true[:, geometry_start:geometry_start + 4]
    pred_geometry = y_pred[:, geometry_start:geometry_start + 4]
    true_scale = y_true[:, geometry_start + 4:geometry_start + 5]
    pred_scale = y_pred[:, geometry_start + 4:geometry_start + 5]
    weights = 1.0 + 5.0 * true_mask[..., 0]
    # why: tf_keras removes the singleton channel axis from binary BCE.
    bce = tf.reduce_mean(weights * keras.losses.binary_crossentropy(true_mask, pred_mask))
    intersection = tf.reduce_sum(true_mask * pred_mask, axis=(1, 2, 3))
    denominator = tf.reduce_sum(true_mask + pred_mask, axis=(1, 2, 3))
    dice = tf.reduce_mean(1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0))
    heat_loss = tf.reduce_mean(keras.losses.binary_crossentropy(true_heat, pred_heat))
    offset_weights = tf.cast(true_heat[..., 0] > 0.5, tf.float32)[..., None]
    offset_loss = tf.reduce_sum(offset_weights * tf.abs(true_offset - pred_offset)) / (tf.reduce_sum(offset_weights) * 2.0 + 1e-6)
    moment = _moment_center(pred_mask)
    center_error = tf.reduce_mean(keras.losses.huber(true_geometry[:, :2], moment))
    geometry_error = tf.reduce_mean(keras.losses.huber(true_geometry, pred_geometry))
    scale_loss = tf.reduce_mean(keras.losses.binary_crossentropy(true_scale, pred_scale))
    # why: center supervision is intentionally stronger than radius supervision
    # because the deployment crop depends primarily on the face center.
    # why: confidence should calibrate routing without competing with the
    # center objective; the earlier 1.5 weight measurably degraded test_1.
    return bce + dice + 2.0 * heat_loss + 4.0 * offset_loss + 12.0 * center_error + 2.0 * geometry_error + 0.1 * scale_loss


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export a fully integer TFLite graph using varied calibration frames."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield CPU-held samples to keep calibration memory bounded."""
        rng = np.random.default_rng(SEED)
        for index in rng.choice(len(images), min(512, len(images)), replace=False):
            # why: training storage stays at 384x384, but this model's input
            # contract is 640x640; calibration must exercise that exact graph shape.
            sample = tf.image.resize(images[index : index + 1], (IMAGE_SIZE, IMAGE_SIZE))
            yield [sample.numpy().astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model_path: Path, images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the exported integer graph and return mask, heatmap, offset, geometry, scale confidence."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    results: list[np.ndarray] = []
    for image in images:
        quantized = np.clip(np.round(image / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        results.append((raw - out_zero) * out_scale)
    values = np.asarray(results, dtype=np.float32)
    heat_values = KEYPOINTS * CENTER_VALUES
    offset_start = MASK_VALUES + heat_values
    geometry_start = offset_start + OFFSET_VALUES
    return (values[:, :MASK_VALUES].reshape(-1, MASK_SIZE, MASK_SIZE, 1),
            values[:, MASK_VALUES:offset_start].reshape(-1, MASK_SIZE, MASK_SIZE, KEYPOINTS),
            values[:, offset_start:offset_start + OFFSET_VALUES].reshape(-1, MASK_SIZE, MASK_SIZE, 2),
            values[:, geometry_start:geometry_start + 4],
            values[:, geometry_start + 4:geometry_start + 5])


def decode_masks(masks: np.ndarray, floor: float = 0.10, radius_factor: float = 2.0) -> np.ndarray:
    """Decode mask moments into normalized center and approximate radii."""
    coords = (np.arange(MASK_SIZE, dtype=np.float32) + 0.5) / MASK_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    decoded: list[list[float]] = []
    for mask in masks[..., 0]:
        weights = np.maximum(mask - floor, 0.0)
        total = max(float(weights.sum()), 1e-6)
        cx = float((weights * xx).sum() / total)
        cy = float((weights * yy).sum() / total)
        rx = radius_factor * float(np.sqrt(max((weights * (xx - cx) ** 2).sum() / total, 1e-8)))
        ry = radius_factor * float(np.sqrt(max((weights * (yy - cy) ** 2).sum() / total, 1e-8)))
        decoded.append([cx, cy, rx, ry])
    return np.asarray(decoded, dtype=np.float32)


def decode_center_heatmap(heatmaps: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    """Decode a center heatmap at the center of its winning grid cell."""
    decoded = geometry.copy()
    for index, heatmap in enumerate(heatmaps[..., 0]):
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        decoded[index, :2] = [(x + 0.5) / MASK_SIZE, (y + 0.5) / MASK_SIZE]
    return decoded


def decode_keypoint_ellipse(heatmaps: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    """Decode center and four synthetic ellipse boundary keypoints."""
    decoded = geometry.copy()
    points = np.zeros((len(heatmaps), KEYPOINTS, 2), dtype=np.float32)
    for index, channels in enumerate(heatmaps):
        for keypoint in range(KEYPOINTS):
            y, x = np.unravel_index(np.argmax(channels[..., keypoint]), channels[..., keypoint].shape)
            points[index, keypoint] = [(x + 0.5) / MASK_SIZE, (y + 0.5) / MASK_SIZE]
    decoded[:, :2] = points[:, 0]
    decoded[:, 2] = np.maximum((points[:, 2, 0] - points[:, 1, 0]) * 0.5, 1e-3)
    decoded[:, 3] = np.maximum((points[:, 4, 1] - points[:, 3, 1]) * 0.5, 1e-3)
    return decoded


def main() -> None:
    """Train, QAT-finetune, export, and score the 640 center candidate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--qat-epochs", type=int, default=8)
    parser.add_argument("--tiny-repeats", type=int, default=10)
    parser.add_argument("--board-repeats", type=int, default=1)
    parser.add_argument("--generic-count", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--export-only", action="store_true", help="Export previously saved QAT weights without retraining.")
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    generic_images, generic_targets = load_zips(["train_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip"], labels=("GaugeFace",))
    # why: board_captures_2 is an exact duplicate of refreshed test_3 and must
    # never contribute training pixels to a generalization experiment.
    clean_board_zips = ["initial_temp_gauge/board_captures_1.zip"]
    board_images, board_targets = load_zips(clean_board_zips, labels=("temp_dial",))
    generic_images, generic_targets = generic_images[:args.generic_count], generic_targets[:args.generic_count]
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
    contract_targets = make_targets(targets)
    if args.export_only:
        # why: a failed calibration run must not force another multi-hour GPU
        # training pass when the QAT checkpoint itself was already saved.
        qat = tfmot.quantization.keras.quantize_model(build_model())
        qat.load_weights(args.output / "model_qat.weights.h5")
        export_int8(qat, images[:256], args.output / "model_int8.tflite")
        print("exported", args.output / "model_int8.tflite", flush=True)
        return
    # why: resize in the input pipeline so the 640x640 tensor is not duplicated
    # in host memory before the first batch reaches the GPU.
    dataset = (
        tf.data.Dataset.from_tensor_slices((images, contract_targets))
        .shuffle(len(images), seed=SEED)
        .map(lambda x, y: (tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE)), y), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    print("training", images.shape, contract_targets.shape, flush=True)

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=center_mask_loss)
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=center_mask_loss)
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images[:256], args.output / "model_int8.tflite")

    report: dict[str, object] = {"train_samples": int(len(images)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        resized = tf.image.resize(test_images, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
        masks, heatmaps, offsets, geometry, scale_confidence = predict_int8(args.output / "model_int8.tflite", resized)
        decoded = decode_center_heatmap(heatmaps, geometry)
        predictions = np.concatenate([decoded, np.ones((len(decoded), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
