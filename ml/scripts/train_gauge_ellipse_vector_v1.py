"""Train a compact full-integer ellipse regressor using the proven vector head."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gauge_face_ellipse_v1_640_gray"
TEMP_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "ellipse"
OUTPUT = ROOT / "artifacts" / os.environ.get("GAUGE_ELLIPSE_OUTPUT", "gauge_ellipse_vector_littlegood_v1")
INPUT_SIZE = int(os.environ.get("GAUGE_ELLIPSE_INPUT", "160"))
BATCH_SIZE = 32
SEED = 42


def configure_gpu() -> None:
    """Limit TensorFlow to the repository's 15 GB GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load image paths and normalized [cx, cy, rx, ry] ellipse targets."""
    paths = sorted((root / "images" / split).glob("*.png"))
    targets = []
    for path in paths:
        values = np.fromstring((root / "labels" / split / f"{path.stem}.txt").read_text(), sep=" ")
        points = values[1:9].reshape(4, 2)
        low, high = points.min(axis=0), points.max(axis=0)
        center, radius = (low + high) / 2.0, (high - low) / 2.0
        targets.append(np.concatenate((center, radius)).astype(np.float32))
    return np.asarray([str(path) for path in paths]), np.asarray(targets, dtype=np.float32)


def load_images(paths: np.ndarray) -> np.ndarray:
    """Predecode and resize images once so training is GPU-bound, not PNG-bound."""
    return np.stack([np.asarray(Image.open(path).convert("L").resize((INPUT_SIZE, INPUT_SIZE)), dtype=np.float32) / 255.0 for path in paths])[..., None]


def dataset(images: np.ndarray, targets: np.ndarray, weights: np.ndarray, training: bool) -> tf.data.Dataset:
    """Create an in-memory TensorFlow dataset at the compact image contract."""
    ds = tf.data.Dataset.from_tensor_slices((images, targets, weights))
    if training:
        ds = ds.shuffle(len(images), seed=SEED, reshuffle_each_iteration=True)

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model() -> keras.Model:
    """Build the compact convolutional ellipse regressor."""
    layers = keras.layers
    inputs = keras.Input((INPUT_SIZE, INPUT_SIZE, 1), name="ellipse_input")
    x = inputs
    for index, (filters, repeats) in enumerate(((16, 2), (24, 2), (40, 2), (64, 2))):
        for repeat in range(repeats):
            x = layers.Conv2D(filters, 3, strides=2 if repeat == 0 else 1, padding="same", use_bias=False, name=f"stage{index}_conv{repeat}")(x)
            x = layers.BatchNormalization(name=f"stage{index}_bn{repeat}")(x)
            x = layers.ReLU(6.0, name=f"stage{index}_relu{repeat}")(x)
    # why: the learned collapse avoids the quantized GlobalAveragePool drift.
    x = layers.Conv2D(64, INPUT_SIZE // 16, padding="valid", use_bias=True, name="spatial_collapse")(x)
    x = layers.ReLU(6.0, name="spatial_collapse_relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    return keras.Model(inputs, layers.Dense(4, activation="sigmoid", name="ellipse_xywh")(x))


def export_int8(model: keras.Model, calibration: np.ndarray, output: Path) -> dict[str, object]:
    """Export full integer TFLite and return its deployment contract."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if len(calibration) > 256:
        # The merged array ends with LittleGood; reserve half the calibration
        # set for that deployment distribution.
        indices = np.concatenate((np.linspace(0, len(calibration) - 257, 128, dtype=int), np.arange(len(calibration) - 128, len(calibration))))
    else:
        indices = np.arange(len(calibration))
    converter.representative_dataset = lambda: ([calibration[index][None].astype(np.float32)] for index in indices)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(blob)
    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    return {"bytes": len(blob), "input": inp["shape"].tolist(), "output": out["shape"].tolist(), "operators": sorted({x["op_name"] for x in interpreter._get_ops_details() if x["op_name"] != "DELEGATE"})}


def main() -> None:
    """Train once per image, QAT-finetune, export, and report ellipse metrics."""
    configure_gpu()
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    tr, ty = load_split(DATA, "train"); vr, vy = load_split(DATA, "val"); te, ey = load_split(DATA, "test")
    tt, tty = load_split(TEMP_DATA, "train"); tv, tvy = load_split(TEMP_DATA, "val"); ts, tsy = load_split(TEMP_DATA, "test")
    tr, ty = np.concatenate((tr, tt)), np.concatenate((ty, tty)); vr, vy = np.concatenate((vr, tv)), np.concatenate((vy, tvy)); te, ey = np.concatenate((te, ts)), np.concatenate((ey, tsy))
    # why: weighting is domain adaptation, not oversampling; every image remains once.
    tw = np.concatenate((np.ones(len(tr) - len(tt), np.float32), np.full(len(tt), 8.0, np.float32)))
    vw = np.concatenate((np.ones(len(vr) - len(tv), np.float32), np.full(len(tv), 8.0, np.float32)))
    x_train, x_val, x_test = load_images(tr), load_images(vr), load_images(te)
    model = build_model(); model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(delta=0.05), metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    model.fit(dataset(x_train, ty, tw, True), validation_data=dataset(x_val, vy, vw, False), epochs=15, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model); qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=keras.losses.Huber(delta=0.05), metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    qat.fit(dataset(x_train, ty, tw, True), validation_data=dataset(x_val, vy, vw, False), epochs=6, verbose=2)
    OUTPUT.mkdir(parents=True, exist_ok=True); contract = export_int8(qat, x_train, OUTPUT / "gauge_ellipse_vector_v1_int8.tflite")
    (OUTPUT / "report.json").write_text(json.dumps({"samples": len(ey), "input": [1, INPUT_SIZE, INPUT_SIZE, 1], "temp_loss_weight": 8, "qat_epochs": 6, "contract": contract}, indent=2))
    print(json.dumps({"artifact": str(OUTPUT), "contract": contract}, indent=2))


if __name__ == "__main__":
    main()
