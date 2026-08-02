"""Train a full-frame int8 center/tip heatmap model.

The model receives the resized full frame plus ellipse mask and predicts both
points directly in the original normalized frame, eliminating crop remapping.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from train_gauge_center_tip_v1 import build_model, configure_gpu, export_int8, load_arrays

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gauge_center_tip_v1_160_gray"
TEMP = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip"
OUT = ROOT / "artifacts" / "gauge_center_tip_fullframe_littlegood_v3"


def tip_weighted_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Give the sparse tip peak stronger supervision than the center peak."""
    channel_weight = tf.constant([1.0, 8.0], dtype=y_true.dtype)[None, None, None, :]
    weights = 1.0 + 48.0 * y_true * channel_weight
    return tf.reduce_mean(weights * tf.square(y_pred - y_true))


def decode(heatmaps: np.ndarray) -> np.ndarray:
    """Decode both heatmap channels with local weighted centroids."""
    size = heatmaps.shape[1]
    points = []
    for sample in heatmaps:
        row = []
        for channel in range(2):
            heatmap = sample[..., channel]
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            y0, y1 = max(0, y - 6), min(size, y + 7); x0, x1 = max(0, x - 6), min(size, x + 7)
            yy, xx = np.mgrid[y0:y1, x0:x1]; weights = np.maximum(heatmap[y0:y1, x0:x1] - .03, 0) ** 2; total = weights.sum()
            row.append(np.asarray(((xx * weights).sum() / total + .5, (yy * weights).sum() / total + .5), dtype=np.float32) / size if total else np.asarray((x + .5, y + .5), dtype=np.float32) / size)
        points.append(row)
    return np.asarray(points, dtype=np.float32)


def main() -> None:
    """Train, QAT-finetune, export, and evaluate on all 97 LittleGood frames."""
    configure_gpu(); tf.keras.utils.set_random_seed(42); OUT.mkdir(parents=True, exist_ok=True)
    xb, yb = load_arrays(DATA, "train"); xv, yv = load_arrays(DATA, "val")
    xt, yt = load_arrays(TEMP, "test"); tb, ytb = load_arrays(TEMP, "train"); tv, ytv = load_arrays(TEMP, "val")
    x_train, y_train = np.concatenate((xb, tb)), np.concatenate((yb, ytb))
    x_val, y_val = np.concatenate((xv, tv)), np.concatenate((yv, ytv))
    model = build_model()
    # why: 160x160 logits reduce tip quantization/binning error while the
    # decoder remains a single nearest-neighbor upsample plus 1x1 convolution.
    up = tf.keras.layers.UpSampling2D(2, interpolation="nearest", name="fullframe_up160")(model.output)
    output = tf.keras.layers.Conv2D(2, 1, activation="sigmoid", name="fullframe_heatmaps160")(up)
    model = tf.keras.Model(model.input, output)
    y_train = tf.image.resize(y_train, (160, 160), method="bilinear").numpy(); y_val = tf.image.resize(y_val, (160, 160), method="bilinear").numpy()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tip_weighted_loss)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=16, epochs=12, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model); qat.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss=tip_weighted_loss)
    qat.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=16, epochs=4, verbose=2)
    path = OUT / "gauge_center_tip_fullframe_v1_int8.tflite"; export_int8(qat, x_train, path)
    interpreter = tf.lite.Interpreter(model_path=str(path)); interpreter.allocate_tensors(); inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    predictions = []
    for sample in xt:
        scale, zero = inp["quantization"]; interpreter.set_tensor(inp["index"], np.clip(np.round(sample / scale + zero), -128, 127).astype(np.int8)[None]); interpreter.invoke(); raw = interpreter.get_tensor(out["index"]).astype(np.float32); scale, zero = out["quantization"]; predictions.append((raw - zero) * scale)
    predictions = np.concatenate(predictions)
    rows = json.loads((TEMP / "metadata.json").read_text())["splits"]["test"]
    targets = np.asarray([row["center_xy_norm"] + row["tip_xy_norm"] for row in rows], dtype=np.float32).reshape(-1, 2, 2)
    errors = np.linalg.norm((decode(predictions) - targets) * 160, axis=2)
    report = {"samples": len(errors), "center_within_8px": float(np.mean(errors[:,0] <= 8)), "tip_within_8px": float(np.mean(errors[:,1] <= 8)), "center_error_px_mean": float(errors[:,0].mean()), "tip_error_px_mean": float(errors[:,1].mean()), "output_std": predictions.std((0,1,2)).tolist(), "bytes": path.stat().st_size}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
