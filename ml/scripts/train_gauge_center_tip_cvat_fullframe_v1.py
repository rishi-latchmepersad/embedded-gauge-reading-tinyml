"""Train a full-frame int8 keypoint model from the recovered CVAT archive."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from PIL import Image

from train_gauge_center_tip_fullframe_vector_v1 import build_model, full_inputs
from train_gauge_center_tip_vector_v1 import configure_gpu, coordinate_loss, export_int8, predict_int8


ROOT = Path(__file__).resolve().parents[1]
CVAT = ROOT.parent / "tmp" / "cvat_first450_filtered"
OUT = ROOT / "artifacts" / "gauge_center_tip_cvat_fullframe_littlegood_v1"
SIZE = 224


def center_of_box(element: ET.Element) -> tuple[float, float]:
    """Return the center of a CVAT point box in 640-frame pixels."""
    return ((float(element.attrib["xtl"]) + float(element.attrib["xbr"])) / 2.0 / 640.0, (float(element.attrib["ytl"]) + float(element.attrib["ybr"])) / 2.0 / 640.0)


def load_cvat() -> tuple[np.ndarray, np.ndarray]:
    """Load complete CVAT Center+Tip samples as full-frame tensors."""
    root = ET.parse(CVAT / "annotations.xml").getroot()
    inputs, targets = [], []
    axis = (np.arange(SIZE, dtype=np.float32) + 0.5) / SIZE * 640.0
    xx, yy = np.meshgrid(axis, axis)
    for image_node in root.findall("image"):
        objects = {label: node for node in image_node for label in ("Center", "Tip", "GaugeFace") if node.attrib.get("label") == label}
        if not {"Center", "Tip", "GaugeFace"}.issubset(objects):
            continue
        image = np.asarray(Image.open(CVAT / "images" / image_node.attrib["name"]).convert("L"), dtype=np.uint8)
        gray = np.asarray(Image.fromarray(image).resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        face = objects["GaugeFace"].attrib
        ellipse = np.asarray((float(face["cx"]), float(face["cy"]), float(face["rx"]), float(face["ry"])), dtype=np.float32)
        mask = (((xx - ellipse[0]) / max(ellipse[2], 1.0)) ** 2 + ((yy - ellipse[1]) / max(ellipse[3], 1.0)) ** 2 <= 1.0).astype(np.float32)
        inputs.append(np.stack((gray * 2.0 - 1.0, mask * 2.0 - 1.0), axis=-1))
        targets.append(center_of_box(objects["Center"]) + center_of_box(objects["Tip"]))
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def main() -> None:
    """Train, QAT-export, and score on untouched LittleGood test frames."""
    configure_gpu(); tf.keras.utils.set_random_seed(42)
    cvat_x, cvat_y = load_cvat()
    lg_train_x, lg_train_y = full_inputs("train", SIZE)
    lg_val_x, lg_val_y = full_inputs("val", SIZE)
    lg_test_x, lg_test_y = full_inputs("test", SIZE)
    train_x = np.concatenate((cvat_x, lg_train_x)); train_y = np.concatenate((cvat_y, lg_train_y))
    model = build_model(); model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=coordinate_loss)
    model.fit(train_x, train_y, validation_data=(lg_val_x, lg_val_y), batch_size=16, epochs=15, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model); qat.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss=coordinate_loss)
    qat.fit(train_x, train_y, validation_data=(lg_val_x, lg_val_y), batch_size=16, epochs=6, verbose=2)
    OUT.mkdir(parents=True, exist_ok=True); path = OUT / "gauge_center_tip_cvat_fullframe_v1_int8.tflite"; contract = export_int8(qat, train_x, path)
    prediction = predict_int8(path, lg_test_x).reshape(-1, 2, 2); truth = lg_test_y.reshape(-1, 2, 2); errors = np.linalg.norm((prediction - truth) * 640.0, axis=2)
    report = {"cvat_samples": len(cvat_x), "littlegood_test_samples": len(lg_test_x), "center_within_8px": float(np.mean(errors[:, 0] <= 8)), "tip_within_8px": float(np.mean(errors[:, 1] <= 8)), "center_error_px_mean": float(errors[:, 0].mean()), "tip_error_px_mean": float(errors[:, 1].mean()), "contract": contract}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
