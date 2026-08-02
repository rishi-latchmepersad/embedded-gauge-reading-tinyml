"""Train the mixed-domain 224x224 wider-crop int8 keypoint candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from train_gauge_center_tip_direction_radius_v1 import radius_targets
from train_gauge_center_tip_v1 import configure_gpu
from train_gauge_center_tip_wide224_v1 import build_model, decode, dataset, export_int8, heat_targets, resize_inputs


ROOT = Path(__file__).resolve().parents[1]
GENERIC = ROOT / "tmp" / "generic_conditioned_wide_v1"
STUDENT = ROOT / "tmp" / "student_conditioned_wide_v1"
OUT = ROOT / "artifacts" / "gauge_center_tip_wide224_mixed_littlegood_v1"


def load_generic(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load generic wide inputs and local point labels."""
    arrays = np.load(GENERIC / f"{split}.npz")
    return arrays["inputs"], arrays["points"]


def main() -> None:
    """Train, QAT-export, and evaluate the untouched LittleGood test set."""
    configure_gpu()
    tf.keras.utils.set_random_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    generic_train_x, generic_train_points = load_generic("train")
    generic_val_x, generic_val_points = load_generic("val")
    student = {split: np.load(STUDENT / f"{split}.npz") for split in ("train", "val", "test")}
    train_x = resize_inputs(np.concatenate((generic_train_x, student["train"]["inputs"])))
    train_points = np.concatenate((generic_train_points, student["train"]["points"]))
    val_x = resize_inputs(np.concatenate((generic_val_x, student["val"]["inputs"])))
    val_points = np.concatenate((generic_val_points, student["val"]["points"]))
    test_x = resize_inputs(student["test"]["inputs"])
    train_h, val_h = heat_targets(train_points), heat_targets(val_points)
    train_r, val_r = radius_targets(train_points), radius_targets(val_points)
    losses = [keras.losses.MeanSquaredError(), keras.losses.Huber(0.03)]
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=losses, loss_weights=[1.0, 4.0])
    model.fit(dataset(train_x, train_h, train_r, True), validation_data=dataset(val_x, val_h, val_r, False), epochs=14, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=losses, loss_weights=[1.0, 4.0])
    qat.fit(dataset(train_x, train_h, train_r, True), validation_data=dataset(val_x, val_h, val_r, False), epochs=5, verbose=2)
    path = OUT / "gauge_center_tip_wide224_mixed_v1_int8.tflite"
    contract = export_int8(qat, train_x, path)
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    outputs = interpreter.get_output_details()
    heat_detail = next(item for item in outputs if len(item["shape"]) == 4)
    radius_detail = next(item for item in outputs if len(item["shape"]) == 2)
    heatmaps, radii = [], []
    for sample in test_x:
        scale, zero = input_detail["quantization"]
        interpreter.set_tensor(input_detail["index"], np.clip(np.round(sample / scale + zero), -128, 127).astype(np.int8)[None])
        interpreter.invoke()
        for detail, destination in ((heat_detail, heatmaps), (radius_detail, radii)):
            raw = interpreter.get_tensor(detail["index"]).astype(np.float32)
            scale, zero = detail["quantization"]
            destination.append((raw - zero) * scale)
    decoded = decode(np.concatenate(heatmaps))
    direction = decoded[:, 1] - decoded[:, 0]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-6
    prediction = np.stack((decoded[:, 0], decoded[:, 0] + direction * np.concatenate(radii) * 0.5), axis=1)
    errors = np.linalg.norm((prediction - student["test"]["points"]) * 160.0, axis=2)
    report = {"samples": len(errors), "center_within_8px": float(np.mean(errors[:, 0] <= 8)), "tip_within_8px": float(np.mean(errors[:, 1] <= 8)), "center_error_px_mean": float(errors[:, 0].mean()), "tip_error_px_mean": float(errors[:, 1].mean()), "contract": contract}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
