"""Train a full-frame int8 coordinate head with corrected LittleGood labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras
from PIL import Image

from train_gauge_center_tip_vector_v1 import configure_gpu, coordinate_loss, export_int8, predict_int8


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "initial_temp_gauge_v1"
STUDENT = DATA / "student_conditioned"
OUT = ROOT / "artifacts" / "gauge_center_tip_fullframe_vector_littlegood_v1"
INPUT = 224


def build_model() -> keras.Model:
    """Build a compact full-frame coordinate regressor."""
    layers = keras.layers
    inputs = keras.Input((INPUT, INPUT, 2), name="fullframe_input")
    x = inputs
    for stage, (filters, repeats) in enumerate(((16, 2), (24, 2), (40, 2), (64, 2))):
        for repeat in range(repeats):
            x = layers.Conv2D(filters, 3, strides=2 if repeat == 0 else 1, padding="same", use_bias=False, name=f"stage{stage}_conv{repeat}")(x)
            x = layers.BatchNormalization(name=f"stage{stage}_bn{repeat}")(x)
            x = layers.ReLU(6.0, name=f"stage{stage}_relu{repeat}")(x)
    x = layers.Conv2D(64, INPUT // 16, padding="valid", use_bias=True, name="spatial_collapse")(x)
    x = layers.ReLU(6.0, name="spatial_collapse_relu")(x)
    x = layers.Flatten(name="spatial_flatten")(x)
    x = layers.Dense(64, activation="relu", name="head_relu")(x)
    return keras.Model(inputs, layers.Dense(4, activation="sigmoid", name="center_tip_xy")(x))


def full_inputs(split: str, size: int = INPUT) -> tuple[np.ndarray, np.ndarray]:
    """Build full-frame grayscale-plus-mask tensors and normalized targets."""
    rows = json.loads((DATA / "center_tip/metadata.json").read_text())["splits"][split]
    ellipses = np.load(STUDENT / f"{split}.npz")["ellipses"]
    inputs, targets = [], []
    yy, xx = np.mgrid[0:size, 0:size]
    axis = (np.arange(size, dtype=np.float32) + 0.5) / size * 640.0
    full_xx, full_yy = np.meshgrid(axis, axis)
    for row, ellipse in zip(rows, ellipses):
        image = np.asarray(Image.open(DATA / f"ellipse/images/{split}/{row['stem']}.png").convert("L"), dtype=np.uint8)
        gray = np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        cx, cy, rx, ry = ellipse
        mask = (((full_xx - cx) / max(rx, 1.0)) ** 2 + ((full_yy - cy) / max(ry, 1.0)) ** 2 <= 1.0).astype(np.float32)
        inputs.append(np.stack((gray * 2.0 - 1.0, mask * 2.0 - 1.0), axis=-1))
        targets.append(row["center_xy_norm"] + row["tip_xy_norm"])
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def main() -> None:
    """Train, QAT-export, and evaluate a full-frame LittleGood model."""
    configure_gpu()
    tf.keras.utils.set_random_seed(42)
    train_x, train_y = full_inputs("train")
    val_x, val_y = full_inputs("val")
    test_x, test_y = full_inputs("test")
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=coordinate_loss)
    model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=16, epochs=15, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=coordinate_loss)
    qat.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=16, epochs=6, verbose=2)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gauge_center_tip_fullframe_vector_v1_int8.tflite"
    contract = export_int8(qat, train_x, path)
    prediction = predict_int8(path, test_x).reshape(-1, 2, 2)
    errors = np.linalg.norm((prediction - test_y.reshape(-1, 2, 2)) * 640.0, axis=2)
    report = {"samples": len(test_x), "center_within_8px": float(np.mean(errors[:, 0] <= 8.0)), "tip_within_8px": float(np.mean(errors[:, 1] <= 8.0)), "center_error_px_mean": float(errors[:, 0].mean()), "tip_error_px_mean": float(errors[:, 1].mean()), "contract": contract}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
