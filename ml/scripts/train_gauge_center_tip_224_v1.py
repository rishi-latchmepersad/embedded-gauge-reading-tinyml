#!/usr/bin/env python3
"""Train a 224×224 U-Net for center/tip heatmap detection with QAT.

Higher-resolution variant of the v8 architecture:
  - 224×224×2 input (matching tip_focus_v18 board-proven resolution).
  - 112×112×2 heatmap output (2× the spatial precision of 80² / 160²).
  - Deeper U-Net with 5 encoder/decoder stages.
  - Focal heatmap loss with boosted tip channel weight.
  - Cosine LR decay over 40 FP32 + 15 QAT epochs.
  - Photometric + rotation augmentation.
  - Fits comfortably in 2.5 MB SRAM (< 1 MB peak activation).

The 160 px pre-processed inputs are bicubic-upscaled to 224 px for this
quick-look experiment; a future variation will crop directly from 640 px.
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
OUT = ROOT / "artifacts" / "gauge_center_tip_224_littlegood_v2"

INPUT_SIZE = 224
HEATMAP_SIZE = 112
BATCH = 16
SEED = 42

FP32_EPOCHS = 40
QAT_EPOCHS = 15
FP32_LR = 1e-3
QAT_LR = 2e-4


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
def configure_gpu() -> None:
    """Cap GPU memory at 15 GB for WSL headroom."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=15000)],
        )


# ---------------------------------------------------------------------------
# Model — deeper 224→112 U-Net
# ---------------------------------------------------------------------------
def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Two Conv-BN-ReLU6 operations."""
    for idx in range(2):
        x = keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"{name}_conv{idx}",
        )(x)
        x = keras.layers.BatchNormalization(name=f"{name}_bn{idx}")(x)
        x = keras.layers.ReLU(6.0, name=f"{name}_relu{idx}")(x)
    return x


def build_model() -> keras.Model:
    """Build a 5-stage 224→112 U-Net.

    Encoder:   224² →  112² →   56² →   28² →   14² →    7²
    Channels:   16   →   24  →   32  →   48  →   64  →   96
    Decoder:   7²    →  14² →   28² →   56² →  112²
    Channels:   96    →  48  →   32  →   24  →   16

    Peak activation: ~112×112×40 = 501 KB int8 (under 2.5 MB).
    """
    inputs = keras.Input((INPUT_SIZE, INPUT_SIZE, 2), name="input")

    # --- encoder ---
    e0 = conv_block(inputs, 16, "enc0")
    p0 = keras.layers.MaxPooling2D(2, name="pool0")(e0)       # 112

    e1 = conv_block(p0, 24, "enc1")
    p1 = keras.layers.MaxPooling2D(2, name="pool1")(e1)       # 56

    e2 = conv_block(p1, 32, "enc2")
    p2 = keras.layers.MaxPooling2D(2, name="pool2")(e2)       # 28

    e3 = conv_block(p2, 48, "enc3")
    p3 = keras.layers.MaxPooling2D(2, name="pool3")(e3)       # 14

    # --- bottleneck ---
    e4 = conv_block(p3, 64, "enc4")
    p4 = keras.layers.MaxPooling2D(2, name="pool4")(e4)       # 7

    b = conv_block(p4, 96, "bottleneck")

    # --- decoder ---
    u3 = keras.layers.UpSampling2D(2, interpolation="nearest", name="up3")(b)   # 14
    u3 = keras.layers.Concatenate(name="cat3")([u3, e4])
    u3 = conv_block(u3, 48, "dec3")

    u2 = keras.layers.UpSampling2D(2, interpolation="nearest", name="up2")(u3)   # 28
    u2 = keras.layers.Concatenate(name="cat2")([u2, e3])
    u2 = conv_block(u2, 32, "dec2")

    u1 = keras.layers.UpSampling2D(2, interpolation="nearest", name="up1")(u2)   # 56
    u1 = keras.layers.Concatenate(name="cat1")([u1, e2])
    u1 = conv_block(u1, 24, "dec1")

    u0 = keras.layers.UpSampling2D(2, interpolation="nearest", name="up0")(u1)   # 112
    u0 = keras.layers.Concatenate(name="cat0")([u0, e1])
    u0 = conv_block(u0, 16, "dec0")

    out = keras.layers.Conv2D(2, 1, activation="sigmoid", name="heatmaps")(u0)
    return keras.Model(inputs, out, name="gauge_center_tip_224")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_arrays(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load split images and two-channel Gaussian heatmap targets.

    Images are bicubic-upscaled from 160 to 224 and heatmaps are bilinear-
    resampled from 80² to 112² so the supervision stays pixel-aligned.
    """
    rows = json.loads((data_dir / "metadata.json").read_text())["splits"][split]
    inputs_list, targets_list = [], []
    for row in rows:
        # -- image (grayscale) --
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
            ellipse *= 160.0 / float(row["source_width"])
        cx, cy, rx, ry = ellipse
        side = max(2.0 * rx, 2.0 * ry) * 1.35
        xs = (np.arange(160, dtype=np.float32) + 0.5) / 160.0 * side + cx - side / 2.0
        ys = (np.arange(160, dtype=np.float32) + 0.5) / 160.0 * side + cy - side / 2.0
        xx, yy = np.meshgrid(xs, ys)
        mask = (
            ((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0
        ).astype(np.float32)
        stacked = np.stack([image * 2.0 - 1.0, mask * 2.0 - 1.0], axis=-1)
        # why: bicubic upscale preserves edge sharpness for the ellipse mask.
        upscaled = tf.image.resize(stacked, [INPUT_SIZE, INPUT_SIZE], method="bicubic").numpy()
        inputs_list.append(upscaled)

        # -- heatmap target —
        hm = np.load(data_dir / row["heatmap"]).astype(np.float32)  # (80, 80, 2)
        hm_112 = tf.image.resize(hm, [HEATMAP_SIZE, HEATMAP_SIZE], method="bilinear").numpy()
        targets_list.append(hm_112)

    return np.stack(inputs_list), np.stack(targets_list)


def load_student_arrays(
    npz_dir: Path, split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load littlegood preprocessed NPZ, upscaling inputs and heatmaps."""
    data = np.load(npz_dir / f"{split}.npz")
    x = data["inputs"]  # (N, 160, 160, 2)
    # why: bicubic for the image + mask channels to prevent aliasing.
    x_224 = tf.image.resize(x, [INPUT_SIZE, INPUT_SIZE], method="bicubic").numpy()
    hm = data["heatmaps"]  # (N, 80, 80, 2)
    hm_112 = tf.image.resize(hm, [HEATMAP_SIZE, HEATMAP_SIZE], method="bilinear").numpy()
    points = data["points"]  # saved, used for evaluation only
    return x_224, hm_112, points


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
# tf.data pipeline
# ---------------------------------------------------------------------------
@tf.function
def _photometric_augment(image: tf.Tensor) -> tf.Tensor:
    """Apply brightness, contrast, and gamma jitter to grayscale channel only."""
    gray = image[..., :1]
    mask = image[..., 1:]
    gray = gray + tf.random.uniform((), -0.12, 0.12, seed=SEED)
    mean = tf.reduce_mean(gray)
    gray = (gray - mean) * tf.random.uniform((), 0.88, 1.12, seed=SEED) + mean
    gamma = tf.random.uniform((), 0.82, 1.18, seed=SEED)
    safe = tf.clip_by_value(gray, 0.01, 2.0)
    gray = tf.where(gray > 0, safe ** gamma, gray)
    gray = tf.clip_by_value(gray, -1.0, 1.0)
    return tf.concat([gray, mask], axis=-1)


def make_dataset(
    inputs: np.ndarray,
    targets: np.ndarray,
    training: bool,
) -> tf.data.Dataset:
    """Build a rotation + photometric augmented dataset."""
    def _augment(image, target):
        k = tf.random.uniform((), 0, 4, dtype=tf.int32, seed=SEED)
        image = tf.image.rot90(image, k)
        target = tf.image.rot90(target, k)
        image = _photometric_augment(image)
        return image, target

    ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if training:
        ds = ds.shuffle(len(inputs), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Focal heatmap loss
# ---------------------------------------------------------------------------
def focal_heatmap_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Focal MSE that amplifies sparse peak weight and attenuates background."""
    channel_weight = tf.constant([4.0, 6.0], dtype=y_true.dtype)[None, None, None, :]
    weights = 1.0 + 28.0 * (y_true ** 1.5) * channel_weight
    return tf.reduce_mean(weights * tf.square(y_pred - y_true))


# ---------------------------------------------------------------------------
# Int8 export
# ---------------------------------------------------------------------------
def export_int8(model: keras.Model, calibration: np.ndarray, path: Path) -> dict:
    """Export full-integer TFLite graph and return its contract."""
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
    }


# ---------------------------------------------------------------------------
# Cosine decay
# ---------------------------------------------------------------------------
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup then cosine decay."""
    def __init__(self, peak_lr: float, total_steps: int, warmup_steps: int = 0):
        super().__init__()
        self._peak = peak_lr
        self._total = total_steps
        self._warmup = warmup_steps
        self._cosine = keras.optimizers.schedules.CosineDecay(
            peak_lr, max(1, total_steps - warmup_steps), alpha=0.01,
        )

    def __call__(self, step):
        progress = tf.cast(step, tf.float32) / tf.cast(
            max(1, self._warmup), tf.float32,
        )
        warmup_lr = self._peak * progress
        return tf.where(
            step < self._warmup,
            warmup_lr,
            self._cosine(step - self._warmup),
        )

    def get_config(self):
        return {
            "peak_lr": self._peak,
            "total_steps": self._total,
            "warmup_steps": self._warmup,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Train, QAT-finetune, export, and evaluate on untouched LittleGood test set."""
    configure_gpu()
    tf.keras.utils.set_random_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    # -- load data --
    xb, yb = load_arrays(DATA, "train")
    xv, yv = load_arrays(DATA, "val")
    xt, yt = load_arrays(TEMP_DATA, "train")
    xvt, yvt = load_arrays(TEMP_DATA, "val")
    # littlegood pre-processed
    xs_train, ys_train, ps_train = load_student_arrays(STUDENT_DATA, "train")
    xs_val, ys_val, ps_val = load_student_arrays(STUDENT_DATA, "val")
    xs_test, ys_test, ps_test = load_student_arrays(STUDENT_DATA, "test")

    x_train = np.concatenate((xb, xt, xs_train))
    y_train = np.concatenate((yb, yt, ys_train))
    x_val = np.concatenate((xv, xvt, xs_val))
    y_val = np.concatenate((yv, yvt, ys_val))

    # -- FP32 training --
    model = build_model()
    steps_per_epoch = max(1, len(x_train) // BATCH)
    total_fp32 = steps_per_epoch * FP32_EPOCHS
    lr = WarmupCosineDecay(FP32_LR, total_fp32, warmup_steps=steps_per_epoch * 3)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=focal_heatmap_loss,
    )
    model.fit(
        make_dataset(x_train, y_train, True),
        validation_data=make_dataset(x_val, y_val, False),
        epochs=FP32_EPOCHS,
        verbose=2,
    )

    # -- QAT --
    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_steps = steps_per_epoch * QAT_EPOCHS
    qat_lr = WarmupCosineDecay(QAT_LR, qat_steps, warmup_steps=steps_per_epoch)
    qat_model.compile(
        optimizer=keras.optimizers.Adam(qat_lr),
        loss=focal_heatmap_loss,
    )
    qat_model.fit(
        make_dataset(x_train, y_train, True),
        validation_data=make_dataset(x_val, y_val, False),
        epochs=QAT_EPOCHS,
        verbose=2,
    )

    # -- export int8 --
    tflite_path = OUT / "gauge_center_tip_224_v1_int8.tflite"
    contract = export_int8(qat_model, x_train, tflite_path)

    # -- evaluate on untouched LittleGood test set --
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]
    predictions = []
    for sample in xs_test:
        scale, zero = inp_d["quantization"]
        tensor = np.clip(
            np.round(sample / scale + zero), -128, 127,
        ).astype(np.int8)[None]
        interpreter.set_tensor(inp_d["index"], tensor)
        interpreter.invoke()
        raw = interpreter.get_tensor(out_d["index"]).astype(np.float32)
        scale, zero = out_d["quantization"]
        predictions.append((raw - zero) * scale)
    predictions = np.concatenate(predictions)

    decoded = decode_points(predictions)
    # Map normalized coords back to 160-pixel crop space for comparison.
    scaled_targets = ps_test * INPUT_SIZE
    scaled_preds = decoded * INPUT_SIZE
    errors = np.linalg.norm(scaled_preds - scaled_targets, axis=2)

    report = {
        "model": "gauge_center_tip_224_v1",
        "input_size": INPUT_SIZE,
        "heatmap_size": HEATMAP_SIZE,
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
