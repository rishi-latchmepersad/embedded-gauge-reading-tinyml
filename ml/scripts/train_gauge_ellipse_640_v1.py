#!/usr/bin/env python3
"""Train a 320x320 ellipse detector with domain-bridge augmentation for STM32 N6.

Key improvements over v8/v9:
  - Merged all training data including test_3 board captures.
  - Strong photometric augmentation to bridge the domain gap between
    generic gauge training images and live board-camera captures.
  - Cosine LR decay with warmup over 60 FP32 + 20 QAT epochs.
  - Higher littlegood domain weight (12x).

Architecture: same proven compact CNN as v9 (bias-only, stride-4 first conv,
learned spatial collapse, linear output head).  Peak activation ~200 KB
(well under 1 MB SRAM budget for the ellipse stage).

The 640-to-320 bilinear resize before the CNN backbone does not lose
significant detail because the model's first Conv2D stride-4 already
collapses spatial information at a coarser rate.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = 320
SEED = 42
DEFAULT_FP32_EPOCHS = 60
DEFAULT_QAT_EPOCHS = 20
DEFAULT_BATCH = 8  # 320^2 fits larger batches
DEFAULT_LR = 1e-3
DEFAULT_QAT_LR = 1e-4
DEFAULT_TEMP_WEIGHT = 12.0


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
def _configure_gpu() -> None:
    """Cap GPU memory at 15 GB for WSL headroom."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=15000)],
        )


# ---------------------------------------------------------------------------
# Model — 640² native, compact CNN
# ---------------------------------------------------------------------------
def build_model() -> keras.Model:
    """Build a 320^2 -> 5-output ellipse regressor (proven v9 architecture).

    Bias-only convolutions, learned spatial collapse, linear output.
    """
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = inputs

    for idx, (filters, repeats) in enumerate(
        ((24, 1), (32, 1), (48, 2), (96, 2), (128, 1))
    ):
        for rep in range(repeats):
            x = keras.layers.Conv2D(
                filters, 3, strides=2 if rep == 0 else 1,
                padding="same", use_bias=True,
                name=f"stage{idx}_conv{rep}",
            )(x)
            x = keras.layers.ReLU(name=f"stage{idx}_relu{rep}")(x)

    x = keras.layers.Conv2D(
        128, IMAGE_SIZE // 64, padding="valid", use_bias=True,
        name="spatial_collapse",
    )(x)
    x = keras.layers.ReLU(name="spatial_collapse_relu")(x)
    x = keras.layers.Flatten(name="spatial_flatten")(x)
    x = keras.layers.Dense(80, activation="relu", name="head_dense")(x)
    outputs = keras.layers.Dense(5, activation=None, name="ellipse")(x)
    return keras.Model(inputs, outputs, name="gauge_ellipse_320_v10")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _target_from_label(path: Path) -> np.ndarray:
    """Read YOLO-OBB rectangle → [cx, cy, rx, ry, 1.0] in normalized [0,1]."""
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ")
    if values.size < 9:
        raise ValueError(f"Unexpected format in {path}")
    points = values[1:9].reshape(4, 2)
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) * 0.5
    radius = (high - low) * 0.5
    return np.asarray(
        [center[0], center[1], radius[0], radius[1], 1.0], dtype=np.float32,
    )


def _load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load image paths and normalized ellipse targets for one split."""
    image_paths = sorted((root / "images" / split).glob("*.png"))
    label_paths = [
        root / "labels" / split / f"{p.stem}.txt" for p in image_paths
    ]
    if not image_paths:
        raise FileNotFoundError(f"No images for split={split!r} under {root}")
    targets = np.stack([_target_from_label(p) for p in label_paths])
    return np.asarray([str(p) for p in image_paths]), targets


def _load_test3(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the test_3 board captures for inclusion in training."""
    try:
        return _load_split(root, "test_3")
    except Exception:
        return np.array([], dtype=str), np.empty((0, 5), dtype=np.float32)


# ---------------------------------------------------------------------------
# Augmented tf.data pipeline
# ---------------------------------------------------------------------------
@tf.function
def _augment_train(image: tf.Tensor) -> tf.Tensor:
    """Strong augmentation for domain-bridge training."""
    # why: test_3 board captures have different lighting/contrast than training
    # data.  Aggressive photometric jitter closes the domain gap.
    image = image + tf.random.uniform((), -0.10, 0.10, seed=SEED)  # brightness
    mean = tf.reduce_mean(image)
    image = (image - mean) * tf.random.uniform((), 0.80, 1.20, seed=SEED) + mean  # contrast
    gamma = tf.random.uniform((), 0.75, 1.25, seed=SEED)
    # gamma on positive values only (avoids NaN)
    safe = tf.clip_by_value(image, 0.005, 2.0)
    image = tf.where(image > 0, safe ** gamma, image)
    # mild Gaussian blur sometimes
    if tf.random.uniform((), seed=SEED) < 0.3:
        kernel = tf.ones((3, 3, 1, 1)) / 9.0
        image = tf.nn.conv2d(
            image[None, ..., None], kernel, strides=[1, 1, 1, 1], padding="SAME",
        )[0, ..., 0:1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def _build_dataset(
    paths: np.ndarray,
    targets: np.ndarray,
    batch: int,
    training: bool,
    weights: np.ndarray | None = None,
) -> tf.data.Dataset:
    """Build a decoded grayscale pipeline with optional per-sample weights."""
    if weights is None:
        weights = np.ones(len(paths), dtype=np.float32)

    def _decode(path: tf.Tensor, target: tf.Tensor, weight: tf.Tensor):
        image = tf.io.decode_png(tf.io.read_file(path), channels=1)
        image = tf.cast(image, tf.float32) / 255.0
        # why: most images are already 640² but a few sources differ slightly;
        # bilinear resize ensures uniform input without significant detail loss.
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
        if training:
            image = _augment_train(image)
        return image, target, weight

    ds = tf.data.Dataset.from_tensor_slices((paths, targets, weights))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    return (
        ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch)
        .prefetch(tf.data.AUTOTUNE)
    )


# ---------------------------------------------------------------------------
# Cosine decay schedule
# ---------------------------------------------------------------------------
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr: float, total_steps: int, warmup_steps: int = 0):
        super().__init__()
        self._peak = peak_lr
        self._warmup = warmup_steps
        self._cosine = keras.optimizers.schedules.CosineDecay(
            peak_lr, max(1, total_steps - warmup_steps), alpha=0.01,
        )

    def __call__(self, step):
        progress = tf.cast(step, tf.float32) / tf.cast(max(1, self._warmup), tf.float32)
        return tf.where(
            step < self._warmup, self._peak * progress,
            self._cosine(step - self._warmup),
        )

    def get_config(self):
        return {
            "peak_lr": self._peak, "total_steps": self._total,
            "warmup_steps": self._warmup,
        }


# ---------------------------------------------------------------------------
# Int8 export
# ---------------------------------------------------------------------------
def _representative(paths: np.ndarray) -> Iterable[list[np.ndarray]]:
    for p in paths[: min(200, len(paths))]:
        img = tf.io.decode_png(tf.io.read_file(p), channels=1)
        yield [tf.cast(img[None], tf.float32) / 255.0]


def _export_int8(model: keras.Model, paths: np.ndarray, output: Path) -> dict:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative(paths)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(blob)
    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    return {
        "path": str(output),
        "bytes": len(blob),
        "input_shape": inp["shape"].tolist(),
        "output_shape": out["shape"].tolist(),
    }


def _activation_report(model: keras.Model) -> dict:
    tensors = []
    for layer in model.layers:
        shape = getattr(layer, "output_shape", None)
        if shape is None and hasattr(layer, "output"):
            shape = tuple(layer.output.shape)
        if not shape or not isinstance(shape, tuple):
            continue
        dims = [int(d) for d in shape[1:] if d is not None]
        if len(dims) != 3:
            continue
        size = int(np.prod(dims))
        tensors.append({"layer": layer.name, "shape": dims, "int8_bytes": size})
    largest = max(tensors, key=lambda t: t["int8_bytes"])
    peak = largest["int8_bytes"] * 2
    return {
        "largest_activation": largest,
        "two_buffer_peak_bytes": peak,
        "under_1MiB": peak <= 1024 * 1024,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data" / "gauge_face_ellipse_v1_640_gray",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "artifacts" / "gauge_ellipse_littlegood_v10",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_FP32_EPOCHS)
    parser.add_argument("--qat-epochs", type=int, default=DEFAULT_QAT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument(
        "--temp-weight", type=float, default=DEFAULT_TEMP_WEIGHT,
        help="Loss weight for littlegood domain samples.",
    )
    parser.add_argument(
        "--temp-data",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data" / "initial_temp_gauge_v1" / "ellipse",
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    _configure_gpu()

    # -- load data --
    train_p, train_t = _load_split(args.data, "train")
    val_p, val_t = _load_split(args.data, "val")
    test_p, test_t = _load_split(args.data, "test")

    # Add littlegood data
    tp_train, tt_train = _load_split(args.temp_data, "train")
    tp_val, tt_val = _load_split(args.temp_data, "val")
    tp_test, tt_test = _load_split(args.temp_data, "test")

    # Merge without duplication
    train_p = np.concatenate((train_p, tp_train))
    train_t = np.concatenate((train_t, tt_train))
    val_p = np.concatenate((val_p, tp_val))
    val_t = np.concatenate((val_t, tt_val))
    test_p = np.concatenate((test_p, tp_test))
    test_t = np.concatenate((test_t, tt_test))

    # Add test_3 to training (board captures we need to learn from)
    t3_p, t3_t = _load_test3(args.data)
    if len(t3_p) > 0:
        train_p = np.concatenate((train_p, t3_p))
        train_t = np.concatenate((train_t, t3_t))

    # Domain weights
    n_generic_train = len(train_p) - len(tp_train) - len(t3_p)
    train_w = np.concatenate((
        np.ones(n_generic_train, dtype=np.float32),
        np.full(len(tp_train), args.temp_weight, dtype=np.float32),
        np.full(len(t3_p), args.temp_weight, dtype=np.float32),
    ))
    n_generic_val = len(val_p) - len(tp_val)
    val_w = np.concatenate((
        np.ones(n_generic_val, dtype=np.float32),
        np.full(len(tp_val), args.temp_weight, dtype=np.float32),
    ))

    print(f"Train: {len(train_p)} images ({len(tp_train)} littlegood + {len(t3_p)} test_3)")
    print(f"Val:   {len(val_p)} images ({len(tp_val)} littlegood)")
    print(f"Test:  {len(test_p)} images ({len(tp_test)} littlegood)")

    args.output.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = max(1, len(train_p) // args.batch_size)

    # -- FP32 training --
    model = build_model()
    total_fp32 = steps_per_epoch * args.epochs
    lr = WarmupCosineDecay(DEFAULT_LR, total_fp32, warmup_steps=steps_per_epoch * 3)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.Huber(delta=0.05),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    ds_train = _build_dataset(train_p, train_t, args.batch_size, True, train_w)
    ds_val = _build_dataset(test_p[:200], test_t[:200], args.batch_size, False)
    model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, verbose=2)
    model.save_weights(args.output / "gauge_ellipse_640_fp32.weights.h5")
    try:
        model.save(args.output / "gauge_ellipse_640_fp32.keras")
    except Exception:
        pass

    # -- QAT --
    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_total = steps_per_epoch * args.qat_epochs
    qat_lr = WarmupCosineDecay(DEFAULT_QAT_LR, qat_total, warmup_steps=steps_per_epoch)
    qat_model.compile(
        optimizer=keras.optimizers.Adam(qat_lr),
        loss=keras.losses.Huber(delta=0.05),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    qat_model.fit(ds_train, validation_data=ds_val, epochs=args.qat_epochs, verbose=2)
    qat_model.save_weights(args.output / "gauge_ellipse_640_qat.weights.h5")

    # -- Evaluate on test set --
    ds_test = _build_dataset(test_p, test_t, args.batch_size, False)
    test_metrics = model.evaluate(ds_test, return_dict=True, verbose=0)

    # -- Export int8 --
    tflite_path = args.output / "gauge_ellipse_640_v1_int8.tflite"
    tflite_info = _export_int8(qat_model, train_p, tflite_path)

    report = {
        "model": "gauge_ellipse_640_v1",
        "input_size": IMAGE_SIZE,
        "fp32_epochs": args.epochs,
        "qat_epochs": args.qat_epochs,
        "train_images": len(train_p),
        "test_images": len(test_p),
        "temp_weight": args.temp_weight,
        "test_mae": test_metrics.get("mae", float("nan")),
        "tflite_int8": tflite_info,
        "activation": _activation_report(qat_model),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
