"""Train a full-frame int8 center heatmap stage for LittleGood."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from train_gauge_center_tip_v1 import build_model
from train_gauge_center_tip_fullframe_vector_v1 import full_inputs
from train_gauge_center_tip_v1 import configure_gpu, export_int8


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "gauge_center_fullframe_heatmap_littlegood_v1"
SIZE = 80


def targets(points: np.ndarray) -> np.ndarray:
    """Create two identical center Gaussian channels at 112px resolution."""
    points = points.reshape(-1, 2, 2)
    output = np.zeros((len(points), SIZE, SIZE, 2), dtype=np.float32)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    for index, point in enumerate(points[:, 0]):
        px, py = point * SIZE - 0.5
        heatmap = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * 2.2**2))
        output[index, ..., 0] = heatmap; output[index, ..., 1] = heatmap
    return output


def center_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Weight center peaks so the heatmap does not collapse to background."""
    return tf.reduce_mean((1.0 + 48.0 * y_true) * tf.square(y_pred - y_true))


def decode_center(heatmaps: np.ndarray) -> np.ndarray:
    """Decode a center from a local weighted centroid."""
    points = []
    for sample in heatmaps:
        heatmap = sample[..., 0]; y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        y0, y1 = max(0, y - 6), min(SIZE, y + 7); x0, x1 = max(0, x - 6), min(SIZE, x + 7)
        yy, xx = np.mgrid[y0:y1, x0:x1]; w = np.maximum(heatmap[y0:y1, x0:x1] - 0.03, 0.0) ** 2; total = w.sum()
        points.append(np.asarray(((xx * w).sum() / total + .5, (yy * w).sum() / total + .5), np.float32) / SIZE if total else np.asarray((x + .5, y + .5), np.float32) / SIZE)
    return np.asarray(points, np.float32)


def main() -> None:
    """Train, QAT-export, and score full-frame center localization."""
    configure_gpu(); tf.keras.utils.set_random_seed(42)
    train_x, train_points = full_inputs("train", 160); val_x, val_points = full_inputs("val", 160); test_x, test_points = full_inputs("test", 160)
    train_y, val_y = targets(train_points), targets(val_points)
    model = build_model(); model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=center_loss); model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=16, epochs=15, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model); qat.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss=center_loss); qat.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=16, epochs=6, verbose=2)
    OUT.mkdir(parents=True, exist_ok=True); path = OUT / "gauge_center_fullframe_heatmap_v1_int8.tflite"; export_int8(qat, train_x, path)
    interpreter = tf.lite.Interpreter(model_path=str(path)); interpreter.allocate_tensors(); inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]; values=[]
    for sample in test_x:
        scale, zero = inp["quantization"]; interpreter.set_tensor(inp["index"], np.clip(np.round(sample / scale + zero), -128, 127).astype(np.int8)[None]); interpreter.invoke(); raw=interpreter.get_tensor(out["index"]).astype(np.float32); scale, zero=out["quantization"]; values.append((raw-zero)*scale)
    predicted=decode_center(np.concatenate(values)); errors=np.linalg.norm((predicted-test_points.reshape(-1,2,2)[:,0])*640.0,axis=1); report={"samples":len(errors),"center_within_8px":float(np.mean(errors<=8)),"center_error_px_mean":float(errors.mean()),"bytes":path.stat().st_size};(OUT/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
