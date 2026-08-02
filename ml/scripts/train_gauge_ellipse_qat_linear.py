"""Train QAT-safe ellipse detector with linear radius head.

Key improvement: radius head uses linear output (no sigmoid) so the
quantization grid covers the actual radius range, preserving variation
across images.  Center head keeps sigmoid for bounded [0,1] output.

Architecture: QAT-safe encoder + multi-head:
  - center_xy: Dense(2, sigmoid) for [cx, cy]
  - radius_xy: Dense(2, None) for [rx, ry] — linear, quantization-friendly
  - confidence: Dense(1, sigmoid)
"""

from __future__ import annotations

import sys
import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

IMAGE_SIZE = 224
SEED = 42
DEFAULT_FP32_EPOCHS = 50
DEFAULT_QAT_EPOCHS = 20
DEFAULT_BATCH = 16
DEFAULT_LR = 1e-3
DEFAULT_QAT_LR = 2e-4
DEFAULT_TEMP_WEIGHT = 12.0
DEFAULT_WIDTH = 1.5


def _configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=15000)],
        )


def _scaled_width(channels: int, wm: float) -> int:
    return max(8, int(round(float(channels) * float(wm) / 8.0) * 8))


def _conv_bn_relu(x, filters, strides=1, name=""):
    x = keras.layers.Conv2D(
        filters, 3, strides=strides, padding="same", use_bias=False,
        kernel_initializer="he_normal", name=f"{name}_conv",
    )(x)
    x = keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    x = keras.layers.ReLU(name=f"{name}_relu")(x)
    return x


def build_model(width_multiplier: float = 1.5) -> keras.Model:
    """QAT-safe ellipse regressor with linear radius head.

    Center: sigmoid (bounded [0,1])
    Radius: linear (unbounded, quantization grid covers actual range)
    Confidence: sigmoid
    """
    wm = float(width_multiplier)
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = keras.layers.Concatenate(name="gray_to_rgb")([inputs, inputs, inputs])

    for stage, (filters, repeats) in enumerate(
        [(32, 2), (48, 2), (64, 2), (96, 2), (128, 2)], 1
    ):
        f = _scaled_width(filters, wm)
        for rep in range(repeats):
            s = 2 if rep == 0 else 1
            x = _conv_bn_relu(x, f, strides=s, name=f"enc_s{stage}_{rep}")

    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(0.2, name="dropout")(x)
    shared = keras.layers.Dense(128, activation="relu", name="shared")(x)

    # Center head: sigmoid for bounded [0,1]
    center = keras.layers.Dense(64, activation="relu", name="center_dense")(shared)
    center_xy = keras.layers.Dense(2, activation="sigmoid", name="center_xy")(center)

    # Radius head: LINEAR (no activation) for quantization-friendly output
    # The quantization grid will cover the actual radius range [~0.1, ~0.4]
    # instead of wasting precision on the full [0,1] sigmoid range.
    radius = keras.layers.Dense(64, activation="relu", name="radius_dense")(shared)
    radius_xy = keras.layers.Dense(2, activation=None, name="radius_xy")(radius)

    # Confidence head
    conf = keras.layers.Dense(32, activation="relu", name="conf_dense")(shared)
    confidence = keras.layers.Dense(1, activation="sigmoid", name="confidence")(conf)

    return keras.Model(
        inputs, [center_xy, radius_xy, confidence],
        name=f"ellipse_qat_linear_w{wm}",
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _target_from_label(path: Path) -> np.ndarray:
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ")
    if values.size < 9:
        raise ValueError(f"Unexpected format in {path}")
    points = values[1:9].reshape(4, 2)
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) * 0.5
    radius = (high - low) * 0.5
    return np.asarray([center[0], center[1], radius[0], radius[1], 1.0], dtype=np.float32)


def _load_split(root: Path, split: str):
    image_paths = sorted((root / "images" / split).glob("*.png"))
    label_paths = [root / "labels" / split / f"{p.stem}.txt" for p in image_paths]
    if not image_paths:
        raise FileNotFoundError(f"No images for {split} under {root}")
    targets = np.stack([_target_from_label(p) for p in label_paths])
    return np.asarray([str(p) for p in image_paths]), targets


def _load_test3(root: Path):
    try:
        return _load_split(root, "test_3")
    except Exception:
        return np.array([], dtype=str), np.empty((0, 5), dtype=np.float32)


# ---------------------------------------------------------------------------
# tf.data
# ---------------------------------------------------------------------------
def _build_dataset(paths, targets, batch, training, weights=None):
    if weights is None:
        weights = np.ones(len(paths), dtype=np.float32)

    def _decode(path, target, weight):
        image = tf.io.decode_png(tf.io.read_file(path), channels=1)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
        if training:
            image = image + tf.random.uniform((), -0.08, 0.08, seed=SEED)
            mean = tf.reduce_mean(image)
            image = (image - mean) * tf.random.uniform((), 0.92, 1.08, seed=SEED) + mean
            image = tf.clip_by_value(image, 0.0, 1.0)
        center = target[:2]
        radius = target[2:4]
        conf = target[4:5]
        return image, (center, radius, conf), (weight, weight, weight)

    ds = tf.data.Dataset.from_tensor_slices((paths, targets, weights))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    return ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak, total, warmup=0):
        super().__init__()
        self._peak, self._total, self._warmup = peak, total, warmup
        self._cosine = keras.optimizers.schedules.CosineDecay(peak, max(1, total - warmup), alpha=0.01)

    def __call__(self, step):
        p = tf.cast(step, tf.float32) / tf.cast(max(1, self._warmup), tf.float32)
        return tf.where(step < self._warmup, self._peak * p, self._cosine(step - self._warmup))

    def get_config(self):
        return {"peak": self._peak, "total": self._total, "warmup": self._warmup}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def _export_int8(model, paths, output, max_cal=512):
    def _rep():
        rng = np.random.default_rng(42)
        for idx in rng.permutation(len(paths))[:max_cal]:
            img = tf.io.decode_png(tf.io.read_file(paths[idx]), channels=1)
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
    details = {"bytes": len(blob), "outputs": []}
    for d in interp.get_output_details():
        details["outputs"].append({
            "name": d["name"], "shape": d["shape"].tolist(),
            "quantization": d["quantization"],
        })
    details["input"] = interp.get_input_details()[0]["shape"].tolist()
    return details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "gauge_face_ellipse_v1_640_gray")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "gauge_ellipse_qat_linear_v1")
    parser.add_argument("--epochs", type=int, default=DEFAULT_FP32_EPOCHS)
    parser.add_argument("--qat-epochs", type=int, default=DEFAULT_QAT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--temp-weight", type=float, default=DEFAULT_TEMP_WEIGHT)
    parser.add_argument("--temp-data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "initial_temp_gauge_v1" / "ellipse")
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH)
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
        np.ones(n_generic, np.float32),
        np.full(len(tp_train), args.temp_weight, np.float32),
        np.full(len(t3_p), args.temp_weight, np.float32),
    ))
    val_w = np.concatenate((
        np.ones(len(val_p) - len(tp_val), np.float32),
        np.full(len(tp_val), args.temp_weight, np.float32),
    ))

    print(f"Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)}")
    print(f"Radius head: LINEAR (no activation)")

    args.output.mkdir(parents=True, exist_ok=True)
    steps = max(1, len(train_p) // args.batch_size)

    # -- FP32 training --
    model = build_model(width_multiplier=args.width)
    lr = WarmupCosineDecay(DEFAULT_LR, steps * args.epochs, steps * 3)

    losses = [
        keras.losses.Huber(delta=0.05),  # center (sigmoid)
        keras.losses.Huber(delta=0.05),  # radius (linear)
        keras.losses.Huber(delta=0.05),  # confidence (sigmoid)
    ]
    loss_weights = [1.0, 3.0, 0.1]  # radius gets 3x weight

    model.compile(optimizer=keras.optimizers.Adam(lr), loss=losses, loss_weights=loss_weights)
    model.fit(
        _build_dataset(train_p, train_t, args.batch_size, True, train_w),
        validation_data=_build_dataset(val_p, val_t, args.batch_size, False, val_w),
        epochs=args.epochs, verbose=2,
    )
    model.save_weights(args.output / "fp32.weights.h5")

    # -- QAT --
    print("Starting QAT...")
    qat = tfmot.quantization.keras.quantize_model(model)
    qat_lr = WarmupCosineDecay(DEFAULT_QAT_LR, steps * args.qat_epochs, steps)
    qat.compile(optimizer=keras.optimizers.Adam(qat_lr), loss=losses, loss_weights=loss_weights)
    qat.fit(
        _build_dataset(train_p, train_t, args.batch_size, True, train_w),
        validation_data=_build_dataset(val_p, val_t, args.batch_size, False, val_w),
        epochs=args.qat_epochs, verbose=2,
    )

    # -- Export --
    tflite_path = args.output / "ellipse_qat_linear_int8.tflite"
    contract = _export_int8(qat, train_p, tflite_path)

    # -- Eval --
    test_metrics = model.evaluate(
        _build_dataset(test_p, test_t, args.batch_size, False), verbose=0,
    )

    report = {
        "model": f"ellipse_qat_linear_w{args.width}",
        "input_size": IMAGE_SIZE,
        "radius_head": "linear",
        "fp32_epochs": args.epochs,
        "qat_epochs": args.qat_epochs,
        "train_images": len(train_p),
        "test_mae_center": float(test_metrics[1]) if len(test_metrics) > 1 else None,
        "test_mae_radius": float(test_metrics[2]) if len(test_metrics) > 2 else None,
        "tflite_int8": contract,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
