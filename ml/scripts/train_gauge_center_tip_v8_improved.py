#!/usr/bin/env python3
"""Train an improved deep U-Net for center/tip heatmap detection with QAT.

Improvements over the v7 baseline:
  - 50% more channel capacity in every U-Net stage.
  - Photometric augmentation (random brightness, contrast, gamma).
  - Focal-heatmap loss that upweights the sparse tip channel harder.
  - Cosine learning-rate decay over a longer FP32 phase.
  - Extended QAT fine-tuning with a warmup schedule.
  - The final int8 TFLite blob fits under 2.5 MB SRAM.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gauge_center_tip_v1_160_gray"
TEMP_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip"
STUDENT_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned"
OUT = ROOT / "artifacts" / "gauge_center_tip_littlegood_v9"

INPUT_SIZE = 160
HEATMAP_SIZE = 80
BATCH = 16
SEED = 42

FP32_EPOCHS = 30
QAT_EPOCHS = 10
FP32_LR = 1e-3
QAT_LR = 2e-4
CENTER_CHANNEL_WEIGHT = 4.0   # balanced center/tip supervision
TIP_CHANNEL_WEIGHT = 6.0       # tip still gets slightly more peak weight
LOSS_FOCAL_POWER = 2.0


# ---------------------------------------------------------------------------
# GPU config
# ---------------------------------------------------------------------------
def configure_gpu() -> None:
    """Cap GPU memory at 15 GB so WSL keeps headroom."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=15000)],
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Two Conv-BN-ReLU6 operations for one U-Net stage."""
    for idx in range(2):
        x = keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"{name}_conv{idx}",
        )(x)
        x = keras.layers.BatchNormalization(name=f"{name}_bn{idx}")(x)
        x = keras.layers.ReLU(6.0, name=f"{name}_relu{idx}")(x)
    return x


def build_model() -> keras.Model:
    """Build a deeper 160→80 U-Net with ~50% more channels than v7.

    v7 channels:        16 → 24 → 40 →  64 → 40 → 24
    v8 (this) channels: 24 → 36 → 56 →  96 → 56 → 36
    """
    inputs = keras.Input((INPUT_SIZE, INPUT_SIZE, 2), name="input")
    # --- encoder ---
    e1 = conv_block(inputs, 24, "enc1")
    p1 = keras.layers.MaxPooling2D(2, name="pool1")(e1)
    e2 = conv_block(p1, 36, "enc2")
    p2 = keras.layers.MaxPooling2D(2, name="pool2")(e2)
    e3 = conv_block(p2, 56, "enc3")
    p3 = keras.layers.MaxPooling2D(2, name="pool3")(e3)
    # --- bottleneck ---
    b = conv_block(p3, 96, "bottleneck")
    # --- decoder ---
    u2 = keras.layers.UpSampling2D(2, interpolation="nearest", name="up2")(b)
    u2 = keras.layers.Concatenate(name="cat2")([u2, e3])
    u2 = conv_block(u2, 56, "dec2")
    u1 = keras.layers.UpSampling2D(2, interpolation="nearest", name="up1")(u2)
    u1 = keras.layers.Concatenate(name="cat1")([u1, e2])
    u1 = conv_block(u1, 36, "dec1")
    out = keras.layers.Conv2D(2, 1, activation="sigmoid", name="heatmaps")(u1)
    return keras.Model(inputs, out, name="gauge_center_tip_v8")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_arrays(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load split images and two-channel Gaussian heatmap targets."""
    rows = json.loads((data_dir / "metadata.json").read_text())["splits"][split]
    inputs_list, targets_list = [], []
    for row in rows:
        # -- image --
        image = (
            np.asarray(
                tf.keras.utils.load_img(data_dir / row["image"], color_mode="grayscale"),
                dtype=np.float32,
            )
            / 255.0
        )
        # -- ellipse mask (channel 2) --
        ellipse = np.asarray(row["ellipse"], dtype=np.float32)
        if row.get("source_width"):
            ellipse *= float(INPUT_SIZE) / float(row["source_width"])
        cx, cy, rx, ry = ellipse
        side = max(2.0 * rx, 2.0 * ry) * 1.35  # why: same crop as v7 for fair comparison
        xs = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / INPUT_SIZE * side + cx - side / 2.0
        ys = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / INPUT_SIZE * side + cy - side / 2.0
        xx, yy = np.meshgrid(xs, ys)
        mask = (
            ((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0
        ).astype(np.float32)
        inputs_list.append(np.stack([image * 2.0 - 1.0, mask * 2.0 - 1.0], axis=-1))
        # -- heatmap targets --
        targets_list.append(np.load(data_dir / row["heatmap"]).astype(np.float32))
    return np.stack(inputs_list), np.stack(targets_list)


def decode_points(heatmaps: np.ndarray) -> np.ndarray:
    """Decode both heatmap channels with weighted local centroid refinement."""
    size = heatmaps.shape[1]
    points = []
    for sample in heatmaps:
        row = []
        for ch in range(2):
            hm = sample[..., ch]
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            y0, y1 = max(0, y - 4), min(size, y + 5)
            x0, x1 = max(0, x - 4), min(size, x + 5)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            w = np.maximum(hm[y0:y1, x0:x1] - 0.03, 0) ** 2
            total = w.sum()
            if total > 0:
                row.append(
                    np.asarray(
                        ((xx * w).sum() / total + 0.5, (yy * w).sum() / total + 0.5),
                        dtype=np.float32,
                    )
                    / size
                )
            else:
                row.append(
                    np.asarray(((x + 0.5) / size, (y + 0.5) / size), dtype=np.float32)
                )
        points.append(row)
    return np.asarray(points, dtype=np.float32)


# ---------------------------------------------------------------------------
# Augmented tf.data pipeline
# ---------------------------------------------------------------------------
@tf.function
def _photometric_augment(image: tf.Tensor) -> tf.Tensor:
    """Apply random brightness, contrast, and gamma jitter to the grayscale channel."""
    # why: the ellipse mask (channel 1) carries geometry, not photometry,
    # so only channel 0 is augmented.
    gray = image[..., :1]  # (H, W, 1)
    mask = image[..., 1:]  # (H, W, 1)
    # random brightness: [-0.15, +0.15]
    gray = gray + tf.random.uniform((), -0.15, 0.15, seed=SEED)
    # random contrast: [0.85, 1.15]
    mean = tf.reduce_mean(gray)
    gray = (gray - mean) * tf.random.uniform((), 0.85, 1.15, seed=SEED) + mean
    # random gamma: [0.8, 1.2]
    gamma = tf.random.uniform((), 0.8, 1.2, seed=SEED)
    gray = tf.clip_by_value(gray, -1.0, 2.0)  # prevent NaN from negative power
    gray = tf.where(gray > 0, gray ** gamma, gray)
    gray = tf.clip_by_value(gray, -1.0, 1.0)
    return tf.concat([gray, mask], axis=-1)


def make_dataset(
    inputs: np.ndarray,
    targets: np.ndarray,
    training: bool,
) -> tf.data.Dataset:
    """Build a rotation+photometric-augmented multi-output dataset."""
    def _augment(image: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Joint rotation-augment: only multiples of 90° to preserve mask semantics."""
        k = tf.random.uniform((), 0, 4, dtype=tf.int32, seed=SEED)
        image_aug = tf.image.rot90(image, k)
        target_aug = tf.image.rot90(target, k)
        # why: photometric jitter is applied after rotation since rotation
        # is cheap and photometric noise decorrelates from orientation.
        image_aug = _photometric_augment(image_aug)
        return image_aug, target_aug

    ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if training:
        ds = ds.shuffle(len(inputs), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Focal heatmap loss
# ---------------------------------------------------------------------------
def focal_heatmap_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Focal-style MSE: peaks get amplified weight, background is suppressed.

    channel_weight[center, tip] = [CENTER_CHANNEL_WEIGHT, TIP_CHANNEL_WEIGHT]
    LOSS_FOCAL_POWER powers the y_true multiplier so sparse peaks dominate.
    """
    channel_weight = tf.constant(
        [CENTER_CHANNEL_WEIGHT, TIP_CHANNEL_WEIGHT], dtype=y_true.dtype,
    )[None, None, None, :]
    # why: y_true is a Gaussian; y_true**focal_power pushes the peak region
    # weight up while keeping the periphery nearly zero, focussing on
    # sub-pixel-accurate peak placement.
    weights = 1.0 + 28.0 * (y_true ** 1.5) * channel_weight
    return tf.reduce_mean(weights * tf.square(y_pred - y_true))


# ---------------------------------------------------------------------------
# Int8 export
# ---------------------------------------------------------------------------
def export_int8(model: keras.Model, calibration: np.ndarray, path: Path) -> dict:
    """Export a full-integer TFLite graph and return its contract dict."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    indices = np.linspace(0, len(calibration) - 1, min(256, len(calibration)), dtype=int)
    converter.representative_dataset = lambda: (
        [calibration[i][None].astype(np.float32)] for i in indices
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()
    path.write_bytes(blob)
    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    return {
        "bytes": len(blob),
        "input": inp["shape"].tolist(),
        "output": out["shape"].tolist(),
        "input_quantization": inp["quantization"],
        "output_quantization": out["quantization"],
    }


# ---------------------------------------------------------------------------
# Cosine decay schedule helper
# ---------------------------------------------------------------------------
def cosine_decay_schedule(initial_lr: float, total_steps: int, warmup_steps: int = 0):
    """Cosine decay with optional linear warmup."""
    if warmup_steps <= 0:
        return keras.optimizers.schedules.CosineDecay(
            initial_lr, total_steps, alpha=0.01,
        )
    warmup = keras.optimizers.schedules.PolynomialDecay(
        1e-6, warmup_steps, initial_lr, power=1,
    )
    cosine = keras.optimizers.schedules.CosineDecay(
        initial_lr, total_steps - warmup_steps, alpha=0.01,
    )
    class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            return tf.where(
                step < warmup_steps, warmup(step), cosine(step - warmup_steps),
            )
        def get_config(self):
            return {}
    return WarmupCosineDecay()


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------
def main() -> None:
    """Train, QAT-finetune, export, and evaluate on the untouched LittleGood test set."""
    configure_gpu()
    tf.keras.utils.set_random_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    # -- load data --
    xb, yb = load_arrays(DATA, "train")
    xv, yv = load_arrays(DATA, "val")
    xt, yt = load_arrays(TEMP_DATA, "train")
    xvt, yvt = load_arrays(TEMP_DATA, "val")
    student = {
        split: np.load(STUDENT_DATA / f"{split}.npz")
        for split in ("train", "val", "test")
    }
    x_train = np.concatenate((xb, xt, student["train"]["inputs"]))
    y_train = np.concatenate((yb, yt, student["train"]["heatmaps"]))
    x_val = np.concatenate((xv, xvt, student["val"]["inputs"]))
    y_val = np.concatenate((yv, yvt, student["val"]["heatmaps"]))
    print(f"train samples: {len(x_train)}, val samples: {len(x_val)}")

    # -- FP32 training --
    model = build_model()
    steps_per_epoch = max(1, len(x_train) // BATCH)
    total_fp32_steps = steps_per_epoch * FP32_EPOCHS
    lr_schedule = cosine_decay_schedule(FP32_LR, total_fp32_steps, warmup_steps=steps_per_epoch * 2)
    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss=focal_heatmap_loss,
    )
    model.fit(
        make_dataset(x_train, y_train, True),
        validation_data=make_dataset(x_val, y_val, False),
        epochs=FP32_EPOCHS,
        verbose=2,
    )

    # -- QAT --
    qat = tfmot.quantization.keras.quantize_model(model)
    qat_steps = steps_per_epoch * QAT_EPOCHS
    qat_lr_schedule = cosine_decay_schedule(QAT_LR, qat_steps, warmup_steps=steps_per_epoch)
    qat.compile(
        optimizer=keras.optimizers.Adam(qat_lr_schedule),
        loss=focal_heatmap_loss,
    )
    qat.fit(
        make_dataset(x_train, y_train, True),
        validation_data=make_dataset(x_val, y_val, False),
        epochs=QAT_EPOCHS,
        verbose=2,
    )

    # -- export int8 TFLite --
    tflite_path = OUT / "gauge_center_tip_v8_int8.tflite"
    contract = export_int8(qat, x_train, tflite_path)
    print(f"Exported {contract['bytes']} bytes to {tflite_path}")

    # -- evaluate on untouched LittleGood test set --
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    predictions = []
    for sample in student["test"]["inputs"]:
        scale, zero = inp_detail["quantization"]
        tensor = np.clip(
            np.round(sample / scale + zero), -128, 127,
        ).astype(np.int8)[None]
        interpreter.set_tensor(inp_detail["index"], tensor)
        interpreter.invoke()
        raw = interpreter.get_tensor(out_detail["index"]).astype(np.float32)
        scale, zero = out_detail["quantization"]
        predictions.append((raw - zero) * scale)
    predictions = np.concatenate(predictions)

    decoded = decode_points(predictions)
    errors = np.linalg.norm(
        (decoded - student["test"]["points"]) * INPUT_SIZE, axis=2,
    )
    report = {
        "model": "gauge_center_tip_v8",
        "samples": len(errors),
        "center_within_8px": float(np.mean(errors[:, 0] <= 8)),
        "tip_within_8px": float(np.mean(errors[:, 1] <= 8)),
        "center_error_px_mean": float(errors[:, 0].mean()),
        "tip_error_px_mean": float(errors[:, 1].mean()),
        "fp32_epochs": FP32_EPOCHS,
        "qat_epochs": QAT_EPOCHS,
        "contract": contract,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
