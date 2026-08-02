"""Train center/tip U-Net with adaptive crop scales.

Uses:
  - Generic gauge data (7,309 images) with fixed crop scale
  - LittleGood data (451 images) with adaptive crop scales (1.0-1.5x)
  - Total: 7,760 images

The adaptive crop scale ensures the model handles gauges at different
distances from the camera.  Large gauges get tight crops, small gauges
get wide crops, so the gauge fills a consistent fraction of the input.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gauge_center_tip_v1_160_gray"
ADAPTIVE_DATA = ROOT / "data" / "center_tip_adaptive_v2"
OUT = ROOT / "artifacts" / "gauge_center_tip_adaptive_v2"

INPUT_SIZE = 160
HEATMAP_SIZE = 80
BATCH = 16
SEED = 42
FP32_EPOCHS = 35
QAT_EPOCHS = 15
FP32_LR = 1e-3
QAT_LR = 2e-4
CENTER_WEIGHT = 4.0
TIP_WEIGHT = 6.0


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)],
        )


def conv_block(x, filters, name):
    for idx in range(2):
        x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{idx}")(x)
        x = keras.layers.BatchNormalization(name=f"{name}_bn{idx}")(x)
        x = keras.layers.ReLU(6.0, name=f"{name}_relu{idx}")(x)
    return x


def build_model():
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
    return keras.Model(inputs, out, name="gauge_center_tip_adaptive")


def load_arrays(data_dir, split):
    rows = json.loads((data_dir / "metadata.json").read_text())["splits"][split]
    inputs_list, targets_list = [], []
    for row in rows:
        image = np.asarray(
            tf.keras.utils.load_img(data_dir / row["image"], color_mode="grayscale"),
            dtype=np.float32,
        ) / 255.0
        ellipse = np.asarray(row["ellipse"], dtype=np.float32)
        if row.get("source_width"):
            ellipse *= float(INPUT_SIZE) / float(row["source_width"])
        cx, cy, rx, ry = ellipse
        side = max(2.0 * rx, 2.0 * ry) * 1.35
        xs = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / INPUT_SIZE * side + cx - side / 2.0
        ys = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / INPUT_SIZE * side + cy - side / 2.0
        xx, yy = np.meshgrid(xs, ys)
        mask = (((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0).astype(np.float32)
        inputs_list.append(np.stack([image * 2.0 - 1.0, mask * 2.0 - 1.0], axis=-1))
        targets_list.append(np.load(data_dir / row["heatmap"]).astype(np.float32))
    return np.stack(inputs_list), np.stack(targets_list)


def decode_points(heatmaps):
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
                row.append(np.asarray(((xx * w).sum() / total + 0.5, (yy * w).sum() / total + 0.5), np.float32) / size)
            else:
                row.append(np.asarray(((x + 0.5) / size, (y + 0.5) / size), np.float32))
        points.append(row)
    return np.asarray(points, np.float32)


@tf.function
def _photometric_augment(image):
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


def make_dataset(inputs, targets, training):
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


def focal_heatmap_loss(y_true, y_pred):
    channel_weight = tf.constant([CENTER_WEIGHT, TIP_WEIGHT], dtype=y_true.dtype)[None, None, None, :]
    weights = 1.0 + 28.0 * (y_true ** 1.5) * channel_weight
    return tf.reduce_mean(weights * tf.square(y_pred - y_true))


def export_int8(model, calibration, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    indices = np.linspace(0, len(calibration) - 1, min(256, len(calibration)), dtype=int)
    converter.representative_dataset = lambda: ([calibration[i][None].astype(np.float32)] for i in indices)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()
    path.write_bytes(blob)
    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    return {"bytes": len(blob), "input": inp["shape"].tolist(), "output": out["shape"].tolist()}


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


def main():
    configure_gpu()
    tf.keras.utils.set_random_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    # Load adaptive-crop data (all images with adaptive scales)
    adaptive_train = np.load(ADAPTIVE_DATA / "train.npz")
    adaptive_val = np.load(ADAPTIVE_DATA / "val.npz")
    adaptive_test = np.load(ADAPTIVE_DATA / "test.npz")

    x_train = adaptive_train["inputs"]
    y_train = adaptive_train["heatmaps"]
    x_val = adaptive_val["inputs"]
    y_val = adaptive_val["heatmaps"]
    print(f"Train: {len(x_train)}, Val: {len(x_val)}")

    steps = max(1, len(x_train) // BATCH)

    # FP32
    model = build_model()
    lr = WarmupCosineDecay(FP32_LR, steps * FP32_EPOCHS, steps * 2)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=focal_heatmap_loss)
    model.fit(make_dataset(x_train, y_train, True), validation_data=make_dataset(x_val, y_val, False), epochs=FP32_EPOCHS, verbose=2)

    # QAT
    qat = tfmot.quantization.keras.quantize_model(model)
    qat_lr = WarmupCosineDecay(QAT_LR, steps * QAT_EPOCHS, steps)
    qat.compile(optimizer=keras.optimizers.Adam(qat_lr), loss=focal_heatmap_loss)
    qat.fit(make_dataset(x_train, y_train, True), validation_data=make_dataset(x_val, y_val, False), epochs=QAT_EPOCHS, verbose=2)

    # Export
    tflite_path = OUT / "gauge_center_tip_adaptive_v1_int8.tflite"
    contract = export_int8(qat, x_train, tflite_path)

    # Evaluate on adaptive test set
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    predictions = []
    for sample in adaptive_test["inputs"]:
        s, z = inp_d["quantization"]
        t = np.clip(np.round(sample / s + z), -128, 127).astype(np.int8)[None]
        interp.set_tensor(inp_d["index"], t)
        interp.invoke()
        raw = interp.get_tensor(out_d["index"]).astype(np.float32)
        s, z = out_d["quantization"]
        predictions.append((raw - z) * s)
    predictions = np.concatenate(predictions)

    decoded = decode_points(predictions)
    errors = np.linalg.norm((decoded - adaptive_test["points"]) * INPUT_SIZE, axis=2)

    c_t_pred = decoded[:, 1] - decoded[:, 0]
    c_t_gt = adaptive_test["points"][:, 1] - adaptive_test["points"][:, 0]
    angle_pred = np.arctan2(c_t_pred[:, 1], c_t_pred[:, 0])
    angle_gt = np.arctan2(c_t_gt[:, 1], c_t_gt[:, 0])
    angle_err = np.abs(np.rad2deg(np.arctan2(np.sin(angle_pred - angle_gt), np.cos(angle_pred - angle_gt))))

    report = {
        "model": "gauge_center_tip_adaptive_v1",
        "train_images": len(x_train),
        "samples": len(errors),
        "center_within_8px": float(np.mean(errors[:, 0] <= 8)),
        "tip_within_8px": float(np.mean(errors[:, 1] <= 8)),
        "center_error_px_mean": float(errors[:, 0].mean()),
        "tip_error_px_mean": float(errors[:, 1].mean()),
        "angle_error_deg_mean": float(angle_err.mean()),
        "angle_within_5deg": float(np.mean(angle_err <= 5)),
        "contract": contract,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
