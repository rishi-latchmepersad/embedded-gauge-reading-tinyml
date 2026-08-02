"""Train a CVAT-augmented direct-center plus tip-heatmap QAT model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from train_gauge_center_tip_cvat_crop_v1 import load_cvat
from train_gauge_center_tip_hybrid_v1 import (
    build_model,
    configure_gpu,
    coordinate_loss,
    decode_tip,
    export_int8,
    predict_int8,
    tip_loss,
)


ROOT = Path(__file__).resolve().parents[1]
STUDENT = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned"
OUT = ROOT / "artifacts" / "gauge_center_tip_cvat_hybrid_littlegood_v1"
HEATMAP = 80


def tip_targets(points: np.ndarray) -> np.ndarray:
    """Rasterize only the tip point for the hybrid model's heatmap head."""
    yy, xx = np.mgrid[0:HEATMAP, 0:HEATMAP]
    result = np.zeros((len(points), HEATMAP, HEATMAP, 1), np.float32)
    for index, point in enumerate(points[:, 1]):
        px, py = np.clip(point * HEATMAP - 0.5, 0.5, HEATMAP - 0.5)
        result[index, ..., 0] = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 2.2**2))
    return result


def main() -> None:
    """Train with each source image once and evaluate the untouched test set."""
    configure_gpu(); tf.keras.utils.set_random_seed(42); OUT.mkdir(parents=True, exist_ok=True)
    cvat_x, cvat_y = load_cvat(); cvat_y = cvat_y.reshape(-1, 2, 2)
    student = {split: np.load(STUDENT / f"{split}.npz") for split in ("train", "val", "test")}
    x_train = np.concatenate((cvat_x, student["train"]["inputs"]))
    center_train = np.concatenate((cvat_y[:, 0], student["train"]["points"][:, 0]))
    tip_train = tip_targets(np.concatenate((cvat_y, student["train"]["points"])))
    x_val = student["val"]["inputs"]; center_val = student["val"]["points"][:, 0]; tip_val = tip_targets(student["val"]["points"])
    model = build_model(); losses = {"center_xy": coordinate_loss, "tip_heatmap": tip_loss}
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=losses, loss_weights={"center_xy": 2.0, "tip_heatmap": 3.0})
    model.fit(x_train, {"center_xy": center_train, "tip_heatmap": tip_train}, validation_data=(x_val, {"center_xy": center_val, "tip_heatmap": tip_val}), batch_size=16, epochs=15, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model); qat.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss=losses, loss_weights={"center_xy": 2.0, "tip_heatmap": 3.0})
    qat.fit(x_train, {"center_xy": center_train, "tip_heatmap": tip_train}, validation_data=(x_val, {"center_xy": center_val, "tip_heatmap": tip_val}), batch_size=16, epochs=5, verbose=2)
    path = OUT / "gauge_center_tip_cvat_hybrid_v1_int8.tflite"; contract = export_int8(qat, x_train, path)
    center, heatmap = predict_int8(path, student["test"]["inputs"]); prediction = np.stack((center, decode_tip(heatmap)), axis=1); truth = student["test"]["points"]
    errors = np.linalg.norm((prediction - truth) * 160.0, axis=2)
    report = {"cvat_samples": len(cvat_x), "littlegood_test_samples": len(truth), "center_within_8px": float(np.mean(errors[:, 0] <= 8)), "tip_within_8px": float(np.mean(errors[:, 1] <= 8)), "center_error_px_mean": float(errors[:, 0].mean()), "tip_error_px_mean": float(errors[:, 1].mean()), "contract": contract}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
