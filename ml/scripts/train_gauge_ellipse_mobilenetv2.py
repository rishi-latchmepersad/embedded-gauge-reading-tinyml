"""Train a MobileNetV2-based ellipse detector with proper int8 support.

The v9/v10/v11 bias-only CNN collapses to a constant output after int8
quantization.  This script uses MobileNetV2 with BatchNorm + ReLU6, which
is explicitly designed for quantization-friendly inference.

Architecture: MobileNetV2 (alpha=0.35) backbone + GlobalAveragePooling2D +
Dense regression head -> [cx, cy, rx, ry, conf].

Data: all labelled gauge images + littlegood board captures + test_3.
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
IMAGE_SIZE = 224  # MobileNetV2's native input size
SEED = 42
DEFAULT_EPOCHS = 40
DEFAULT_QAT_EPOCHS = 15
DEFAULT_BATCH = 16
DEFAULT_LR = 1e-3
DEFAULT_QAT_LR = 2e-4
DEFAULT_TEMP_WEIGHT = 12.0


def _configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=15000)],
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(alpha: float = 0.35) -> keras.Model:
    """MobileNetV2 + regression head for ellipse detection.

    alpha=0.35 gives ~0.4M params (~1.5MB float, ~0.4MB int8).
    alpha=1.0  gives ~3.4M params (~13MB float, ~3.4MB int8).
    """
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")

    # Convert grayscale to 3-channel (MobileNetV2 expects RGB)
    x = keras.layers.Concatenate(name="gray_to_rgb")([inputs, inputs, inputs])

    base = keras.applications.MobileNetV2(
        input_tensor=x,
        include_top=False,
        weights=None,  # train from scratch for grayscale domain
        alpha=alpha,
        pooling=None,
    )

    # Use the last feature map
    feature_map = base.output  # e.g., (7, 7, 1280*alpha)

    x = keras.layers.GlobalAveragePooling2D(name="gap")(feature_map)
    x = keras.layers.Dropout(0.2, name="dropout")(x)
    x = keras.layers.Dense(128, activation="relu6", name="head_dense")(x)
    x = keras.layers.Dropout(0.1, name="dropout2")(x)
    outputs = keras.layers.Dense(5, activation="sigmoid", name="ellipse")(x)

    return keras.Model(inputs, outputs, name=f"ellipse_mobilenetv2_a{alpha}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _target_from_label(path: Path) -> np.ndarray:
    """Read YOLO-OBB -> [cx, cy, rx, ry, 1.0] normalized."""
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
    image_paths = sorted((root / "images" / split).glob("*.png"))
    label_paths = [root / "labels" / split / f"{p.stem}.txt" for p in image_paths]
    if not image_paths:
        raise FileNotFoundError(f"No images for {split} under {root}")
    targets = np.stack([_target_from_label(p) for p in label_paths])
    return np.asarray([str(p) for p in image_paths]), targets


def _load_test3(root: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        return _load_split(root, "test_3")
    except Exception:
        return np.array([], dtype=str), np.empty((0, 5), dtype=np.float32)


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------
def _build_dataset(
    paths: np.ndarray,
    targets: np.ndarray,
    batch: int,
    training: bool,
    weights: np.ndarray | None = None,
) -> tf.data.Dataset:
    if weights is None:
        weights = np.ones(len(paths), dtype=np.float32)

    def _decode(path: tf.Tensor, target: tf.Tensor, weight: tf.Tensor):
        image = tf.io.decode_png(tf.io.read_file(path), channels=1)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
        if training:
            image = image + tf.random.uniform((), -0.08, 0.08, seed=SEED)
            mean = tf.reduce_mean(image)
            image = (image - mean) * tf.random.uniform((), 0.92, 1.08, seed=SEED) + mean
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image, target, weight

    ds = tf.data.Dataset.from_tensor_slices((paths, targets, weights))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    return ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak: float, total: int, warmup: int = 0):
        super().__init__()
        self._peak, self._total, self._warmup = peak, total, warmup
        self._cosine = keras.optimizers.schedules.CosineDecay(peak, max(1, total - warmup), alpha=0.01)

    def __call__(self, step):
        p = tf.cast(step, tf.float32) / tf.cast(max(1, self._warmup), tf.float32)
        return tf.where(step < self._warmup, self._peak * p, self._cosine(step - self._warmup))

    def get_config(self):
        return {"peak": self._peak, "total": self._total, "warmup": self._warmup}


# ---------------------------------------------------------------------------
# Int8 export
# ---------------------------------------------------------------------------
def _export_int8(model: keras.Model, paths: np.ndarray, output: Path) -> dict:
    def _rep():
        for p in paths[: min(512, len(paths))]:
            img = tf.io.decode_png(tf.io.read_file(p), channels=1)
            img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
            yield [tf.cast(img[None], tf.float32) / 255.0]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _rep
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(blob)

    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    return {
        "bytes": len(blob),
        "input_shape": inp["shape"].tolist(),
        "output_shape": out["shape"].tolist(),
        "input_quantization": inp["quantization"],
        "output_quantization": out["quantization"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "gauge_face_ellipse_v1_640_gray")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "gauge_ellipse_mobilenetv2_v1")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--qat-epochs", type=int, default=DEFAULT_QAT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--temp-weight", type=float, default=DEFAULT_TEMP_WEIGHT)
    parser.add_argument("--temp-data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "initial_temp_gauge_v1" / "ellipse")
    parser.add_argument("--alpha", type=float, default=0.35, help="MobileNetV2 width multiplier")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    _configure_gpu()

    # -- load data --
    train_p, train_t = _load_split(args.data, "train")
    val_p, val_t = _load_split(args.data, "val")
    test_p, test_t = _load_split(args.data, "test")

    tp_train, tt_train = _load_split(args.temp_data, "train")
    tp_val, tt_val = _load_split(args.temp_data, "val")
    tp_test, tt_test = _load_split(args.temp_data, "test")

    train_p = np.concatenate((train_p, tp_train))
    train_t = np.concatenate((train_t, tt_train))
    val_p = np.concatenate((val_p, tp_val))
    val_t = np.concatenate((val_t, tt_val))
    test_p = np.concatenate((test_p, tp_test))
    test_t = np.concatenate((test_t, tt_test))

    t3_p, t3_t = _load_test3(args.data)
    if len(t3_p) > 0:
        train_p = np.concatenate((train_p, t3_p))
        train_t = np.concatenate((train_t, t3_t))

    n_generic = len(train_p) - len(tp_train) - len(t3_p)
    train_w = np.concatenate((
        np.ones(n_generic, dtype=np.float32),
        np.full(len(tp_train), args.temp_weight, dtype=np.float32),
        np.full(len(t3_p), args.temp_weight, dtype=np.float32),
    ))
    val_w = np.concatenate((
        np.ones(len(val_p) - len(tp_val), dtype=np.float32),
        np.full(len(tp_val), args.temp_weight, dtype=np.float32),
    ))

    print(f"Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)}")

    args.output.mkdir(parents=True, exist_ok=True)
    steps = max(1, len(train_p) // args.batch_size)

    # -- FP32 training --
    model = build_model(alpha=args.alpha)
    lr = WarmupCosineDecay(DEFAULT_LR, steps * args.epochs, steps * 3)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.Huber(delta=0.05), metrics=["mae"])
    model.fit(
        _build_dataset(train_p, train_t, args.batch_size, True, train_w),
        validation_data=_build_dataset(val_p, val_t, args.batch_size, False, val_w),
        epochs=args.epochs, verbose=2,
    )
    model.save_weights(args.output / "fp32.weights.h5")

    # -- QAT --
    qat = tfmot.quantization.keras.quantize_model(model)
    qat_lr = WarmupCosineDecay(DEFAULT_QAT_LR, steps * args.qat_epochs, steps)
    qat.compile(optimizer=keras.optimizers.Adam(qat_lr), loss=keras.losses.Huber(delta=0.05), metrics=["mae"])
    qat.fit(
        _build_dataset(train_p, train_t, args.batch_size, True, train_w),
        validation_data=_build_dataset(val_p, val_t, args.batch_size, False, val_w),
        epochs=args.qat_epochs, verbose=2,
    )

    # -- export --
    tflite_path = args.output / "ellipse_mobilenetv2_int8.tflite"
    contract = _export_int8(qat, train_p, tflite_path)

    # -- eval --
    test_mae = float(model.evaluate(_build_dataset(test_p, test_t, args.batch_size, False), verbose=0)[1])

    # -- report --
    report = {
        "model": f"ellipse_mobilenetv2_a{args.alpha}",
        "input_size": IMAGE_SIZE,
        "fp32_epochs": args.epochs,
        "qat_epochs": args.qat_epochs,
        "train_images": len(train_p),
        "test_mae": test_mae,
        "tflite_int8": contract,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
