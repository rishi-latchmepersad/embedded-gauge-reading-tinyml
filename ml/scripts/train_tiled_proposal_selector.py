#!/usr/bin/env python3
"""Train a tiny learned ranker for full-frame and tiled gauge proposals."""

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
from train_ellipse_mask_640_center import decode_masks, predict_int8
from train_ellipse_robust_384 import SEED, load_zips
from evaluate_tiled_proposal import make_tile

MODEL = Path("artifacts/gauge_ellipse_mask_center_scaleconf_384_aux_v1/model_int8.tflite")
IMAGE_SIZE = 384
VIEWS = ((0.0, 0.0, 1.0), (0.0, 0.0, 0.6), (0.4, 0.0, 0.6), (0.0, 0.4, 0.6), (0.4, 0.4, 0.6))
FEATURES = 12


def configure_gpu() -> None:
    """Limit TensorFlow to the project's 15 GB GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])


def build_selector() -> keras.Model:
    """Build a small per-candidate quality regressor."""
    inputs = keras.Input((FEATURES,), name="candidate_features")
    x = keras.layers.Dense(24, activation="relu", name="selector_hidden_1")(inputs)
    x = keras.layers.Dense(12, activation="relu", name="selector_hidden_2")(x)
    output = keras.layers.Dense(1, activation="sigmoid", name="candidate_quality")(x)
    return keras.Model(inputs, output, name="tiled_proposal_selector")


def make_candidates(images: np.ndarray, targets: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Generate mapped proposals, feature vectors, and optional quality labels."""
    crops: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    view_meta: list[tuple[int, float, float, float]] = []
    for image_index, image in enumerate(images):
        for x, y, side in VIEWS:
            if side == 1.0:
                crops.append(image)
                sources.append(np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32))
            else:
                crop, source = make_tile(image, x, y, side)
                crops.append(crop)
                sources.append(source)
            view_meta.append((image_index, x, y, side))
    masks, heatmaps, _, geometry, scale = predict_int8(MODEL, np.asarray(crops, dtype=np.float32))
    proposals = decode_masks(masks)
    mapped_rows: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    label_rows: list[float] = []
    for row, (owner, x0, y0, side) in enumerate(view_meta):
        source = sources[row]
        proposal = proposals[row].copy()
        heat = float(np.max(heatmaps[row, ..., 0]))
        scale_value = float(scale[row, 0])
        if scale_value >= 0.5 and heat >= 0.55:
            peak_y, peak_x = np.unravel_index(np.argmax(heatmaps[row, ..., 0]), heatmaps[row, ..., 0].shape)
            proposal[:2] = [(peak_x + 0.5) / heatmaps.shape[1], (peak_y + 0.5) / heatmaps.shape[2]]
            proposal[2:4] = geometry[row, 2:4] * np.asarray([0.487, 0.368], dtype=np.float32)
        sx, sy = source[2] - source[0], source[3] - source[1]
        mapped = np.asarray([source[0] + proposal[0] * sx, source[1] + proposal[1] * sy, proposal[2] * sx, proposal[3] * sy], dtype=np.float32)
        margin = min(float(mapped[0]), float(mapped[1]), float(1.0 - mapped[0]), float(1.0 - mapped[1]))
        features = np.asarray([heat, scale_value, *mapped, margin, x0, y0, side, float(proposal[2]), float(proposal[3])], dtype=np.float32)
        mapped_rows.append(mapped)
        feature_rows.append(features)
        if targets is not None:
            target = targets[owner]
            center_error = float(np.linalg.norm(mapped[:2] - target[:2]))
            radius_error = float(np.mean(np.abs(mapped[2:4] - target[2:4])))
            # why: rank by the deployment objective, with center deliberately
            # weighted more heavily than the tolerated radius variation.
            label_rows.append(float(np.exp(-center_error / 0.04 - radius_error / 0.12)))
    count = len(images)
    mapped_array = np.asarray(mapped_rows, dtype=np.float32).reshape(count, len(VIEWS), 4)
    feature_array = np.asarray(feature_rows, dtype=np.float32).reshape(count, len(VIEWS), FEATURES)
    labels = None if targets is None else np.asarray(label_rows, dtype=np.float32).reshape(count, len(VIEWS), 1)
    return mapped_array, feature_array, labels


def export_int8(model: keras.Model, features: np.ndarray, output: Path) -> None:
    """Export the selector as a fully integer TFLite graph."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    flat = features.reshape(-1, FEATURES)
    converter.representative_dataset = lambda: ([row[None].astype(np.float32)] for row in flat[: min(2048, len(flat))])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_selector(model: Path, features: np.ndarray) -> np.ndarray:
    """Run the selector and return one quality score per candidate."""
    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    values = []
    for row in features.reshape(-1, FEATURES):
        q = np.clip(np.round(row / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], q[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0, 0].astype(np.float32)
        values.append((raw - out_zero) * out_scale)
    return np.asarray(values, dtype=np.float32).reshape(len(features), len(VIEWS))


def main() -> None:
    """Train the selector and evaluate full-frame plus tiled proposals."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--qat-epochs", type=int, default=4)
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    generic_images, generic_targets = load_zips(["train_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(["initial_temp_gauge/board_captures_1.zip"], labels=("temp_dial",))
    train_images = np.concatenate([generic_images[:1000], tiny_images, board_images])
    train_targets = np.concatenate([generic_targets[:1000], tiny_targets, board_targets])
    _, train_features, train_labels = make_candidates(train_images, train_targets)
    assert train_labels is not None
    flat_features = train_features.reshape(-1, FEATURES)
    flat_labels = train_labels.reshape(-1, 1)
    dataset = tf.data.Dataset.from_tensor_slices((flat_features, flat_labels)).shuffle(len(flat_features), seed=SEED).batch(128).prefetch(tf.data.AUTOTUNE)
    model = build_selector()
    model.compile(optimizer=keras.optimizers.Adam(2e-3), loss=keras.losses.Huber(delta=0.05))
    model.fit(dataset, epochs=args.epochs, verbose=0)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(3e-4), loss=keras.losses.Huber(delta=0.05))
    qat.fit(dataset, epochs=args.qat_epochs, verbose=0)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "selector_qat.weights.h5")
    export_int8(qat, train_features, args.output / "selector_int8.tflite")
    report: dict[str, object] = {"tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name)
        candidates, features, _ = make_candidates(images)
        scores = predict_selector(args.output / "selector_int8.tflite", features)
        selected = candidates[np.arange(len(candidates)), np.argmax(scores, axis=1)]
        predictions = np.concatenate([selected, np.ones((len(selected), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
