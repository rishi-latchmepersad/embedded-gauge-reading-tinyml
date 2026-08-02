"""Train an int8 needle-line heatmap with an explicit radius head."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from train_gauge_center_tip_direction_radius_v1 import dataset, export_int8, model_with_radius, predict, radius_targets
from train_gauge_center_tip_fullframe_v1 import decode
from train_gauge_center_tip_line_v1 import line_loss, line_targets
from train_gauge_center_tip_v1 import configure_gpu, load_arrays


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gauge_center_tip_v1_160_gray"
STUDENT = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned"
OUT = ROOT / "artifacts" / "gauge_center_tip_line_radius_littlegood_v1"
RADIUS_SCALE = 0.5


def direction_from_line(line: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Estimate the positive needle direction from a one-sided line heatmap."""
    yy, xx = np.mgrid[0:80, 0:80]
    relative = np.stack((xx - center[0] * 80.0, yy - center[1] * 80.0), axis=-1)
    weights = np.maximum(line - 0.12, 0.0) ** 2
    vector = (relative * weights[..., None]).sum(axis=(0, 1))
    return vector / (np.linalg.norm(vector) + 1e-6)


def main() -> None:
    """Train, QAT-export, and score on the untouched corrected test split."""
    configure_gpu()
    tf.keras.utils.set_random_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    generic_train, generic_heat = load_arrays(DATA, "train")
    generic_val, generic_val_heat = load_arrays(DATA, "val")
    student = {split: np.load(STUDENT / f"{split}.npz") for split in ("train", "val", "test")}
    generic_points = decode(generic_heat)
    generic_val_points = decode(generic_val_heat)
    x_train = np.concatenate((generic_train, student["train"]["inputs"]))
    h_train = np.concatenate((line_targets(generic_points), line_targets(student["train"]["points"])))
    r_train = np.concatenate((radius_targets(generic_points), radius_targets(student["train"]["points"])))
    x_val = np.concatenate((generic_val, student["val"]["inputs"]))
    h_val = np.concatenate((line_targets(generic_val_points), line_targets(student["val"]["points"])))
    r_val = np.concatenate((radius_targets(generic_val_points), radius_targets(student["val"]["points"])))
    losses = [line_loss, keras.losses.Huber(0.03)]
    model = model_with_radius()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=losses, loss_weights=[1.0, 4.0])
    model.fit(dataset(x_train, h_train, r_train, True), validation_data=dataset(x_val, h_val, r_val, False), epochs=14, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=losses, loss_weights=[1.0, 4.0])
    qat.fit(dataset(x_train, h_train, r_train, True), validation_data=dataset(x_val, h_val, r_val, False), epochs=5, verbose=2)
    path = OUT / "gauge_center_tip_line_radius_v1_int8.tflite"
    contract = export_int8(qat, x_train, path)
    heatmaps, radii = predict(path, student["test"]["inputs"])
    centers = decode(heatmaps[..., :1].repeat(2, axis=-1))[:, 0]
    predictions = []
    for index, center in enumerate(centers):
        direction = direction_from_line(heatmaps[index, ..., 1], center)
        tip = center + direction * radii[index, 0] * RADIUS_SCALE
        predictions.append((center, tip))
    errors = np.linalg.norm((np.asarray(predictions) - student["test"]["points"]) * 160.0, axis=2)
    report = {"samples": len(errors), "center_within_8px": float(np.mean(errors[:, 0] <= 8)), "tip_within_8px": float(np.mean(errors[:, 1] <= 8)), "center_error_px_mean": float(errors[:, 0].mean()), "tip_error_px_mean": float(errors[:, 1].mean()), "contract": contract}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
