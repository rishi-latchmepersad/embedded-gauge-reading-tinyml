#!/usr/bin/env python3
"""Train an improved gauge-ellipse detector with QAT for STM32N6 deployment.

Improvements over the v8 baseline:
  - Deeper backbone: 24→32→64→128→192 channels (vs 16→24→48→96→128).
  - Extra conv layer in stages 3-4 for more receptive field.
  - Photometric augmentation (brightness, contrast, gamma).
  - Cosine LR decay over 80 FP32 + 20 QAT epochs.
  - Higher LittleGood domain weight (12× vs 8×).
  - Larger spatial-collapse receptive field for better centering.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = 320
SEED = 42
DEFAULT_FP32_EPOCHS = 80
DEFAULT_QAT_EPOCHS = 20
DEFAULT_BATCH = 8
DEFAULT_TEMP_WEIGHT = 12.0
DEFAULT_LR = 1e-3
DEFAULT_QAT_LR = 1e-4


# ---------------------------------------------------------------------------
# GPU config
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
# Model — deeper backbone, learned spatial collapse
# ---------------------------------------------------------------------------
def build_model() -> keras.Model:
    """Build a deeper NPU-oriented ellipse regressor.

    Stages (filters, stride, repeats):
      s0: 24  stride=4 (80×80 feature map)            ~ 150 KiB int8
      s1: 32  stride=2 (40×40)                          ~ 100 KiB
      s2: 64  stride=2, 2 repeats (20×20)              ~ 50 KiB
      s3: 128 stride=2, 2 repeats (10×10)              ~ 25 KiB
      s4: 192 stride=2, 1 repeat  (5×5)                ~  9 KiB
      spatial_collapse: Conv2D(192, 5) → ReLU → Flatten
      head: Dense(96) → Dense(5, linear)

    All convs are bias-only (no BN) to keep the int8 export faithful.
    Output: [cx_norm, cy_norm, rx_norm, ry_norm, confidence]
    """
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = inputs
    for idx, (filters, stride, repeats) in enumerate(
        ((24, 4, 1), (32, 2, 1), (64, 2, 2), (128, 2, 2), (192, 2, 1))
    ):
        for rep in range(repeats):
            x = keras.layers.Conv2D(
                filters, 3,
                strides=stride if rep == 0 else 1,
                padding="same",
                use_bias=True,
                name=f"stage{idx}_conv{rep}",
            )(x)
            x = keras.layers.ReLU(name=f"stage{idx}_relu{rep}")(x)
    # why: a 5×5 collapse conv (IMAGE_SIZE/64 = 5) avoids the quantized MEAN
    # reduction drift while letting the network weigh spatial positions.
    x = keras.layers.Conv2D(
        192, IMAGE_SIZE // 64, padding="valid", use_bias=True,
        name="spatial_collapse",
    )(x)
    x = keras.layers.ReLU(name="spatial_collapse_relu")(x)
    x = keras.layers.Flatten(name="spatial_flatten")(x)
    x = keras.layers.Dense(96, activation="relu", name="head_relu")(x)
    outputs = keras.layers.Dense(5, activation=None, name="ellipse")(x)
    return keras.Model(inputs, outputs, name="gauge_ellipse_v9")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _target_from_label(path: Path) -> np.ndarray:
    """Read YOLO-OBB rectangle → [cx, cy, rx, ry, 1.0] in normalized coords."""
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ")
    if values.size < 9:
        raise ValueError(f"Unexpected label format in {path}")
    points = values[1:9].reshape(4, 2)
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) * 0.5
    radius = (high - low) * 0.5
    return np.asarray(
        [center[0], center[1], radius[0], radius[1], 1.0], dtype=np.float32,
    )


def _load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load image paths and normalized targets for one split."""
    image_paths = sorted((root / "images" / split).glob("*.png"))
    label_paths = [
        root / "labels" / split / f"{p.stem}.txt" for p in image_paths
    ]
    if not image_paths:
        raise FileNotFoundError(
            f"No image files found for split={split!r} under {root}"
        )
    targets = np.stack([_target_from_label(p) for p in label_paths])
    return np.asarray([str(p) for p in image_paths]), targets


def _dataset(
    paths: np.ndarray,
    targets: np.ndarray,
    batch: int,
    training: bool,
    weights: np.ndarray | None = None,
) -> tf.data.Dataset:
    """Build a decoded grayscale pipeline with optional per-sample weights."""
    if weights is None:
        weights = np.ones(len(paths), dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, targets, weights))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    @tf.function
    def _decode(path: tf.Tensor, target: tf.Tensor, weight: tf.Tensor):
        """Decode PNG → [0,1] grayscale with photometric augmentation."""
        image = tf.io.decode_png(tf.io.read_file(path), channels=1)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(
            image, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear",
        )
        if training:
            # why: small brightness/contrast jitter prevents overfitting to the
            # exact lab lighting in the training exposures.
            image = image + tf.random.uniform((), -0.08, 0.08, seed=SEED)
            mean = tf.reduce_mean(image)
            image = (
                (image - mean)
                * tf.random.uniform((), 0.9, 1.1, seed=SEED)
                + mean
            )
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image, target, weight

    return (
        ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch)
        .prefetch(tf.data.AUTOTUNE)
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _compile(model: keras.Model, learning_rate: float) -> None:
    """Compile with Huber loss + MAE metric."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.Huber(delta=0.05),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )


def _cosine_decay(init_lr: float, total_steps: int, warmup: int = 0):
    """Cosine decay schedule with optional linear warmup."""
    steps_lr = init_lr
    if warmup > 0:
        base = keras.optimizers.schedules.CosineDecay(
            init_lr, total_steps - warmup, alpha=0.01,
        )
        warm = keras.optimizers.schedules.PolynomialDecay(
            1e-6, warmup, init_lr, power=1,
        )

        class _Warmup(keras.optimizers.schedules.LearningRateSchedule):
            def __call__(self, step):
                return tf.where(step < warmup, warm(step), base(step - warmup))

            def get_config(self):
                return {}

        steps_lr = _Warmup()
    else:
        steps_lr = keras.optimizers.schedules.CosineDecay(
            init_lr, total_steps, alpha=0.01,
        )
    return steps_lr


# ---------------------------------------------------------------------------
# Int8 export
# ---------------------------------------------------------------------------
def _export_int8(
    model: keras.Model, paths: np.ndarray, output: Path,
) -> dict:
    """Export full-integer TFLite blob and return its contract."""
    def _representative():
        for p in paths[: min(200, len(paths))]:
            img = tf.io.decode_png(tf.io.read_file(p), channels=1)
            img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
            yield [tf.cast(img[None], tf.float32) / 255.0]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative
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


# ---------------------------------------------------------------------------
# Activation budget report
# ---------------------------------------------------------------------------
def _activation_report(model: keras.Model) -> dict:
    """Estimate int8 activation sizes for the SRAM budget check."""
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
        / "data"
        / "gauge_face_ellipse_v1_640_gray",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "artifacts"
        / "gauge_ellipse_littlegood_v9",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_FP32_EPOCHS)
    parser.add_argument("--qat-epochs", type=int, default=DEFAULT_QAT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument(
        "--temp-weight",
        type=float,
        default=DEFAULT_TEMP_WEIGHT,
        help="Loss weight for LittleGood samples (no duplication).",
    )
    parser.add_argument(
        "--temp-data",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "initial_temp_gauge_v1"
        / "ellipse",
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
    temp_train_p, temp_train_t = _load_split(args.temp_data, "train")
    temp_val_p, temp_val_t = _load_split(args.temp_data, "val")
    temp_test_p, temp_test_t = _load_split(args.temp_data, "test")

    # merge without duplication
    train_p = np.concatenate((train_p, temp_train_p))
    train_t = np.concatenate((train_t, temp_train_t))
    val_p = np.concatenate((val_p, temp_val_p))
    val_t = np.concatenate((val_t, temp_val_t))
    test_p = np.concatenate((test_p, temp_test_p))
    test_t = np.concatenate((test_t, temp_test_t))

    train_w = np.concatenate((
        np.ones(len(train_p) - len(temp_train_p), dtype=np.float32),
        np.full(len(temp_train_p), args.temp_weight, dtype=np.float32),
    ))
    val_w = np.concatenate((
        np.ones(len(val_p) - len(temp_val_p), dtype=np.float32),
        np.full(len(temp_val_p), args.temp_weight, dtype=np.float32),
    ))

    ds_train = _dataset(train_p, train_t, args.batch_size, True, train_w)
    ds_val = _dataset(val_p, val_t, args.batch_size, False, val_w)
    ds_test = _dataset(test_p, test_t, args.batch_size, False)

    args.output.mkdir(parents=True, exist_ok=True)

    # -- FP32 training --
    model = build_model()
    steps_per_epoch = max(1, len(train_p) // args.batch_size)
    total_fp32 = steps_per_epoch * args.epochs
    lr = _cosine_decay(DEFAULT_LR, total_fp32, warmup=steps_per_epoch * 2)
    _compile(model, lr)
    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epochs,
        verbose=2,
    )
    model.save(args.output / "gauge_ellipse_v9_fp32.keras")
    model.save_weights(args.output / "gauge_ellipse_v9_fp32.weights.h5")

    # -- QAT --
    qat_model = tfmot.quantization.keras.quantize_model(model)
    try:
        qat_lr = _cosine_decay(DEFAULT_QAT_LR, steps_per_epoch * args.qat_epochs)
        _compile(qat_model, qat_lr)
    except Exception:
        _compile(qat_model, DEFAULT_QAT_LR)
    qat_model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.qat_epochs,
        verbose=2,
    )
    qat_model.save_weights(args.output / "gauge_ellipse_v9_qat.weights.h5")

    # -- eval --
    test_metrics = model.evaluate(ds_test, return_dict=True, verbose=0)

    # -- export --
    tflite = args.output / "gauge_ellipse_v9_int8.tflite"
    tflite_info = _export_int8(qat_model, train_p, tflite)

    report = {
        "model": "gauge_ellipse_v9",
        "input_size": IMAGE_SIZE,
        "fp32_epochs": args.epochs,
        "qat_epochs": args.qat_epochs,
        "temp_loss_weight": args.temp_weight,
        "train_images": len(train_p),
        "littlegood_train_images": len(temp_train_p),
        "test_images": len(test_p),
        "littlegood_test_images": len(temp_test_p),
        "test_mae": test_metrics.get("mae", float("nan")),
        "tflite_int8": tflite_info,
        "activation": _activation_report(qat_model),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
