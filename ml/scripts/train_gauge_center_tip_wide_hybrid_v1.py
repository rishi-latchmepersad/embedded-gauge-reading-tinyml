"""Train one int8 wide-crop model with endpoint heatmaps and radius head."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from train_gauge_center_tip_direction_radius_v1 import dataset, export_int8, model_with_radius, predict, radius_targets
from train_gauge_center_tip_fullframe_v1 import decode, tip_weighted_loss
from train_gauge_center_tip_wide_endpoint_v1 import endpoint_targets
from train_gauge_center_tip_v1 import configure_gpu


ROOT = Path(__file__).resolve().parents[1]
GENERIC = ROOT / "tmp" / "generic_conditioned_wide_v1"
STUDENT = ROOT / "tmp" / "student_conditioned_wide_v1"
OUT = ROOT / "artifacts" / "gauge_center_tip_wide_hybrid_littlegood_v1"


def load_generic(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one generic split from the wider conditioned cache."""
    arrays = np.load(GENERIC / f"{split}.npz")
    return arrays["inputs"], arrays["points"]


def main() -> None:
    """Train, quantize, and evaluate the hybrid graph on untouched data."""
    configure_gpu()
    tf.keras.utils.set_random_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    gx, gp = load_generic("train")
    gv, gvp = load_generic("val")
    student = {split: np.load(STUDENT / f"{split}.npz") for split in ("train", "val", "test")}
    train_x = np.concatenate((gx, student["train"]["inputs"]))
    train_points = np.concatenate((gp, student["train"]["points"]))
    val_x = np.concatenate((gv, student["val"]["inputs"]))
    val_points = np.concatenate((gvp, student["val"]["points"]))
    train_h, val_h = endpoint_targets(train_points), endpoint_targets(val_points)
    train_r, val_r = radius_targets(train_points), radius_targets(val_points)
    model = model_with_radius()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=[tip_weighted_loss, keras.losses.Huber(0.03)], loss_weights=[1.0, 4.0])
    model.fit(dataset(train_x, train_h, train_r, True), validation_data=dataset(val_x, val_h, val_r, False), epochs=14, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=[tip_weighted_loss, keras.losses.Huber(0.03)], loss_weights=[1.0, 4.0])
    qat.fit(dataset(train_x, train_h, train_r, True), validation_data=dataset(val_x, val_h, val_r, False), epochs=5, verbose=2)
    path = OUT / "gauge_center_tip_wide_hybrid_v1_int8.tflite"
    contract = export_int8(qat, train_x, path)
    heat, radius = predict(path, student["test"]["inputs"])
    decoded = decode(heat)
    direction = decoded[:, 1] - decoded[:, 0]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-6
    prediction = np.stack((decoded[:, 0], decoded[:, 0] + direction * radius * 0.5), axis=1)
    errors = np.linalg.norm((prediction - student["test"]["points"]) * 160.0, axis=2)
    report = {"samples": len(errors), "center_within_8px": float(np.mean(errors[:, 0] <= 8)), "tip_within_8px": float(np.mean(errors[:, 1] <= 8)), "center_error_px_mean": float(errors[:, 0].mean()), "tip_error_px_mean": float(errors[:, 1].mean()), "contract": contract}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
