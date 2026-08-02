"""Train a compact QAT center/tip regressor in ellipse-relative coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from train_gauge_center_tip_vector_v1 import build_model, configure_gpu, coordinate_loss, export_int8, predict_int8

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "gauge_center_tip_v1_160_gray"
STUDENT = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned"
POINTS = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip"
OUT = ROOT / "artifacts" / "gauge_center_tip_residual_littlegood_v1"
SEED = 42


def metadata_targets(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Encode generic crop points relative to their annotated ellipse."""
    rows = json.loads((POINTS / "metadata.json").read_text())["splits"][split]
    inputs, targets = [], []
    for row in rows:
        ellipse = np.asarray(row["ellipse"], dtype=np.float32)
        points = np.asarray((row["center_xy_norm"], row["tip_xy_norm"]), dtype=np.float32) * 160.0
        targets.append(np.clip(0.5 + (points - ellipse[:2]) / (4.0 * ellipse[2:]), 0.02, 0.98).reshape(4))
    return np.asarray(targets, dtype=np.float32), np.asarray(inputs)


def dataset(x: np.ndarray, y: np.ndarray, training: bool) -> tf.data.Dataset:
    """Build a non-duplicating tensor dataset with light photometric jitter."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(32).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    """Train, QAT-finetune, export, and score the residual model."""
    configure_gpu(); tf.keras.utils.set_random_seed(SEED)
    from train_gauge_center_tip_v1 import load_arrays
    xb, _ = load_arrays(BASE, "train"); xv, _ = load_arrays(BASE, "val")
    base_rows = json.loads((BASE / "metadata.json").read_text())["splits"]
    def generic(split: str) -> np.ndarray:
        values = []
        for row in base_rows[split]:
            e = np.asarray(row["ellipse"], np.float32); p = np.asarray((row["center_xy_norm"], row["tip_xy_norm"]), np.float32) * 160
            values.append(np.clip(0.5 + (p - e[:2]) / (4 * e[2:]), .02, .98).reshape(4))
        return np.asarray(values, np.float32)
    sx = {s: np.load(STUDENT / f"{s}.npz") for s in ("train", "val", "test")}
    def student_targets(split: str) -> np.ndarray:
        """Encode local point offsets relative to each predicted ellipse."""
        arrays = sx[split]
        ellipse = arrays["ellipses"]
        radii = ellipse[:, 2:4, None]
        side = np.max(2.0 * radii, axis=1, keepdims=True) * 1.18
        encoded = 0.5 + (arrays["points"] - 0.5) * side / (4.0 * radii)
        return np.clip(encoded, 0.02, 0.98).reshape(-1, 4)
    xtr = np.concatenate((xb, sx["train"]["inputs"])); ytr = np.concatenate((generic("train"), student_targets("train")))
    xval = np.concatenate((xv, sx["val"]["inputs"])); yval = np.concatenate((generic("val"), student_targets("val")))
    model = build_model(); model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=coordinate_loss)
    model.fit(dataset(xtr, ytr, True), validation_data=dataset(xval, yval, False), epochs=12, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model); qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=coordinate_loss)
    qat.fit(dataset(xtr, ytr, True), validation_data=dataset(xval, yval, False), epochs=4, verbose=2)
    OUT.mkdir(parents=True, exist_ok=True); path = OUT / "gauge_center_tip_residual_v1_int8.tflite"; contract = export_int8(qat, xtr, path)
    pred = predict_int8(path, sx["test"]["inputs"]); e = sx["test"]["ellipses"]; local = .5 + (pred.reshape(-1,2,2) - .5) * (4 * e[:,2:4,None]) / (np.max(2 * e[:,2:4,None], axis=1, keepdims=True) * 1.18)
    errors = np.linalg.norm((local - sx["test"]["points"]) * 160, axis=2)
    report = {"samples": len(errors), "center_within_8px": float(np.mean(errors[:,0] <= 8)), "tip_within_8px": float(np.mean(errors[:,1] <= 8)), "center_error_px_mean": float(errors[:,0].mean()), "tip_error_px_mean": float(errors[:,1].mean()), "contract": contract}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
