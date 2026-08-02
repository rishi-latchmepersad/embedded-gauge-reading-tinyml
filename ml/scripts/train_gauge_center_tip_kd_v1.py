#!/usr/bin/env python3
"""KD (Knowledge Distillation) center/tip U-Net — Teacher FP32 → Student int8.

Uses the proven v9 architecture with balanced focal weights.  Three phases:
  1. Train a float32 teacher (no QAT), save weights.
  2. Build a fresh student, apply QAT, train with:
       - Ground-truth focal heatmap loss
       - Distillation MSE loss matching teacher predictions
  3. Export int8 TFLite and evaluate on the untouched LittleGood test set.

The teacher guides the quantized student through the int8 grid's limited
representational capacity, which is the suspected cause of the long tail
of large errors in the direct-QAT v9 model.
"""

from __future__ import annotations

import json
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
OUT = ROOT / "artifacts" / "gauge_center_tip_kd_littlegood_v1"

INPUT_SIZE = 160
HEATMAP_SIZE = 80
BATCH = 16
SEED = 42

TEACHER_EPOCHS = 40
KD_EPOCHS = 30
QAT_EPOCHS = 15
TEACHER_LR = 1e-3
KD_LR = 8e-4
QAT_LR = 2e-4
DISTILLATION_WEIGHT = 0.3


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)],
        )


# ---------------------------------------------------------------------------
# Model (same architecture as v9)
# ---------------------------------------------------------------------------
def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    for idx in range(2):
        x = keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"{name}_conv{idx}",
        )(x)
        x = keras.layers.BatchNormalization(name=f"{name}_bn{idx}")(x)
        x = keras.layers.ReLU(6.0, name=f"{name}_relu{idx}")(x)
    return x


def build_model() -> keras.Model:
    inputs = keras.Input((INPUT_SIZE, INPUT_SIZE, 2), name="input")
    e1 = conv_block(inputs, 24, "enc1")
    p1 = keras.layers.MaxPooling2D(2, name="pool1")(e1)
    e2 = conv_block(p1, 36, "enc2")
    p2 = keras.layers.MaxPooling2D(2, name="pool2")(e2)
    e3 = conv_block(p2, 56, "enc3")
    p3 = keras.layers.MaxPooling2D(2, name="pool3")(e3)
    b = conv_block(p3, 96, "bottleneck")
    u2 = keras.layers.UpSampling2D(2, interpolation="nearest", name="up2")(b)
    u2 = keras.layers.Concatenate(name="cat2")([u2, e3])
    u2 = conv_block(u2, 56, "dec2")
    u1 = keras.layers.UpSampling2D(2, interpolation="nearest", name="up1")(u2)
    u1 = keras.layers.Concatenate(name="cat1")([u1, e2])
    u1 = conv_block(u1, 36, "dec1")
    out = keras.layers.Conv2D(2, 1, activation="sigmoid", name="heatmaps")(u1)
    return keras.Model(inputs, out, name="gauge_center_tip_kd")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_arrays(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    rows = json.loads((data_dir / "metadata.json").read_text())["splits"][split]
    inputs_list, targets_list = [], []
    for row in rows:
        image = (
            np.asarray(
                tf.keras.utils.load_img(data_dir / row["image"], color_mode="grayscale"),
                dtype=np.float32,
            )
            / 255.0
        )
        ellipse = np.asarray(row["ellipse"], dtype=np.float32)
        if row.get("source_width"):
            ellipse *= float(INPUT_SIZE) / float(row["source_width"])
        cx, cy, rx, ry = ellipse
        side = max(2.0 * rx, 2.0 * ry) * 1.35
        xs = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / INPUT_SIZE * side + cx - side / 2.0
        ys = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / INPUT_SIZE * side + cy - side / 2.0
        xx, yy = np.meshgrid(xs, ys)
        mask = (
            ((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0
        ).astype(np.float32)
        inputs_list.append(np.stack([image * 2.0 - 1.0, mask * 2.0 - 1.0], axis=-1))
        targets_list.append(np.load(data_dir / row["heatmap"]).astype(np.float32))
    return np.stack(inputs_list), np.stack(targets_list)


def decode_points(heatmaps: np.ndarray) -> np.ndarray:
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
                        ((xx * w).sum() / total + 0.5, (yy * w).sum() / total + 0.5), np.float32,
                    ) / size
                )
            else:
                row.append(np.asarray(((x + 0.5) / size, (y + 0.5) / size), np.float32))
        points.append(row)
    return np.asarray(points, np.float32)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
@tf.function
def _photometric_augment(image: tf.Tensor) -> tf.Tensor:
    gray = image[..., :1]
    mask = image[..., 1:]
    gray = gray + tf.random.uniform((), -0.15, 0.15, seed=SEED)
    mean = tf.reduce_mean(gray)
    gray = (gray - mean) * tf.random.uniform((), 0.85, 1.15, seed=SEED) + mean
    gamma = tf.random.uniform((), 0.8, 1.2, seed=SEED)
    gray = tf.clip_by_value(gray, -1.0, 2.0)
    gray = tf.where(gray > 0, gray ** gamma, gray)
    gray = tf.clip_by_value(gray, -1.0, 1.0)
    return tf.concat([gray, mask], axis=-1)


def make_dataset(inputs: np.ndarray, targets: np.ndarray, training: bool) -> tf.data.Dataset:
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
# Focal heatmap loss (proven v9 weights)
# ---------------------------------------------------------------------------
def focal_heatmap_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    channel_weight = tf.constant([4.0, 6.0], dtype=y_true.dtype)[None, None, None, :]
    weights = 1.0 + 28.0 * (y_true ** 1.5) * channel_weight
    return tf.reduce_mean(weights * tf.square(y_pred - y_true))


# ---------------------------------------------------------------------------
# KD training model — wraps the student, computes combined loss
# ---------------------------------------------------------------------------
class KDTrainer(keras.Model):
    """Subclass that adds distillation loss on top of compiled supervised loss."""

    def __init__(
        self,
        student: keras.Model,
        teacher: keras.Model,
        distillation_weight: float = DISTILLATION_WEIGHT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        self.distillation_weight = distillation_weight

    def call(self, inputs, training=None):
        return self.student(inputs, training=training)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            pred = self.student(x, training=True)
            main_loss = self.compiled_loss(y, pred, regularization_losses=self.losses)
            teacher_pred = tf.stop_gradient(self.teacher(x, training=False))
            distillation_loss = tf.reduce_mean(tf.square(pred - teacher_pred))
            total_loss = main_loss + self.distillation_weight * distillation_loss
        grads = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        return {
            "loss": total_loss,
            "main_loss": main_loss,
            "distill_loss": distillation_loss,
        }

    def test_step(self, data):
        x, y = data
        pred = self.student(x, training=False)
        main_loss = self.compiled_loss(y, pred, regularization_losses=self.losses)
        return {"loss": main_loss, "main_loss": main_loss}


# ---------------------------------------------------------------------------
# Cosine decay
# ---------------------------------------------------------------------------
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr: float, total_steps: int, warmup_steps: int = 0):
        super().__init__()
        self._peak = peak_lr
        self._total = total_steps
        self._warmup = warmup_steps
        self._cosine = keras.optimizers.schedules.CosineDecay(
            peak_lr, max(1, total_steps - warmup_steps), alpha=0.01,
        )

    def __call__(self, step):
        progress = tf.cast(step, tf.float32) / tf.cast(max(1, self._warmup), tf.float32)
        return tf.where(
            step < self._warmup, self._peak * progress, self._cosine(step - self._warmup),
        )

    def get_config(self):
        return {"peak_lr": self._peak, "total_steps": self._total, "warmup_steps": self._warmup}


# ---------------------------------------------------------------------------
# Int8 export
# ---------------------------------------------------------------------------
def export_int8(model: keras.Model, calibration: np.ndarray, path: Path) -> dict:
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
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    configure_gpu()
    tf.keras.utils.set_random_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    # -- load data --
    xb, yb = load_arrays(DATA, "train")
    xv, yv = load_arrays(DATA, "val")
    xt, yt = load_arrays(TEMP_DATA, "train")
    xvt, yvt = load_arrays(TEMP_DATA, "val")
    student_data = {
        s: np.load(STUDENT_DATA / f"{s}.npz") for s in ("train", "val", "test")
    }
    x_train = np.concatenate((xb, xt, student_data["train"]["inputs"]))
    y_train = np.concatenate((yb, yt, student_data["train"]["heatmaps"]))
    x_val = np.concatenate((xv, xvt, student_data["val"]["inputs"]))
    y_val = np.concatenate((yv, yvt, student_data["val"]["heatmaps"]))
    steps_per_epoch = max(1, len(x_train) // BATCH)

    # ---- Phase 1: train teacher (float32, no QAT) ----
    teacher = build_model()
    teacher_lr = WarmupCosineDecay(TEACHER_LR, steps_per_epoch * TEACHER_EPOCHS, steps_per_epoch * 3)
    teacher.compile(optimizer=keras.optimizers.Adam(teacher_lr), loss=focal_heatmap_loss)
    teacher.fit(
        make_dataset(x_train, y_train, True),
        validation_data=make_dataset(x_val, y_val, False),
        epochs=TEACHER_EPOCHS,
        verbose=2,
    )
    teacher.save_weights(OUT / "teacher_fp32.weights.h5")
    try:
        teacher.save(OUT / "teacher_fp32.keras")
    except Exception:
        pass

    # ---- Phase 2: KD training (student with distillation) ----
    student = build_model()
    kd_trainer = KDTrainer(student, teacher, distillation_weight=DISTILLATION_WEIGHT)
    kd_lr = WarmupCosineDecay(KD_LR, steps_per_epoch * KD_EPOCHS, steps_per_epoch * 2)
    kd_trainer.compile(
        optimizer=keras.optimizers.Adam(kd_lr),
        loss=focal_heatmap_loss,
    )
    kd_trainer.fit(
        make_dataset(x_train, y_train, True),
        validation_data=make_dataset(x_val, y_val, False),
        epochs=KD_EPOCHS,
        verbose=2,
    )
    student.save_weights(OUT / "student_kd_fp32.weights.h5")

    # ---- Phase 3: QAT on the KD student ----
    qat = tfmot.quantization.keras.quantize_model(student)
    qat_lr = WarmupCosineDecay(QAT_LR, steps_per_epoch * QAT_EPOCHS, steps_per_epoch)
    qat.compile(optimizer=keras.optimizers.Adam(qat_lr), loss=focal_heatmap_loss)
    qat.fit(
        make_dataset(x_train, y_train, True),
        validation_data=make_dataset(x_val, y_val, False),
        epochs=QAT_EPOCHS,
        verbose=2,
    )

    # ---- Export ----
    tflite_path = OUT / "gauge_center_tip_kd_v1_int8.tflite"
    contract = export_int8(qat, x_train, tflite_path)

    # ---- Evaluate on untouched test ----
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp_d, out_d = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    predictions = []
    for sample in student_data["test"]["inputs"]:
        scale, zero = inp_d["quantization"]
        t = np.clip(np.round(sample / scale + zero), -128, 127).astype(np.int8)[None]
        interpreter.set_tensor(inp_d["index"], t)
        interpreter.invoke()
        raw = interpreter.get_tensor(out_d["index"]).astype(np.float32)
        scale, zero = out_d["quantization"]
        predictions.append((raw - zero) * scale)
    predictions = np.concatenate(predictions)

    decoded = decode_points(predictions)
    errors = np.linalg.norm((decoded - student_data["test"]["points"]) * INPUT_SIZE, axis=2)

    # Angle error
    c_t_pred = decoded[:, 1] - decoded[:, 0]
    c_t_gt = student_data["test"]["points"][:, 1] - student_data["test"]["points"][:, 0]
    angle_pred = np.arctan2(c_t_pred[:, 1], c_t_pred[:, 0])
    angle_gt = np.arctan2(c_t_gt[:, 1], c_t_gt[:, 0])
    angle_err = np.abs(np.rad2deg(np.arctan2(np.sin(angle_pred - angle_gt), np.cos(angle_pred - angle_gt))))

    report = {
        "model": "gauge_center_tip_kd_v1",
        "samples": len(errors),
        "center_within_8px": float(np.mean(errors[:, 0] <= 8)),
        "tip_within_8px": float(np.mean(errors[:, 1] <= 8)),
        "center_error_px_mean": float(errors[:, 0].mean()),
        "tip_error_px_mean": float(errors[:, 1].mean()),
        "center_error_px_median": float(np.median(errors[:, 0])),
        "tip_error_px_median": float(np.median(errors[:, 1])),
        "center_error_px_p90": float(np.percentile(errors[:, 0], 90)),
        "tip_error_px_p90": float(np.percentile(errors[:, 1], 90)),
        "angle_error_deg_mean": float(angle_err.mean()),
        "angle_error_deg_median": float(np.median(angle_err)),
        "angle_within_5deg": float(np.mean(angle_err <= 5)),
        "teacher_epochs": TEACHER_EPOCHS,
        "kd_epochs": KD_EPOCHS,
        "qat_epochs": QAT_EPOCHS,
        "distillation_weight": DISTILLATION_WEIGHT,
        "contract": contract,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
