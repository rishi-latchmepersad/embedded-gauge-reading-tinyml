"""Export scalar_full_finetune_from_best to INT8 TFLite for STM32N6."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from keras.applications.mobilenet_v2 import preprocess_input

def load_manifest(path: Path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "image_path": row["image_path"],
                "value": float(row["value"]),
            })
    return rows

def resolve_image_path(rel_path: str, ml_root: Path) -> Path:
    p = Path(rel_path)
    if p.is_absolute():
        return p
    rel = rel_path
    if rel.startswith("ml/") or rel.startswith("ml\\"):
        rel = rel[3:]
    rel = rel.lstrip("/\\")
    candidates = [
        ml_root / rel,
        ml_root / "data" / rel,
        ml_root / "data" / "captured_images" / Path(rel).name,
        ml_root / "data" / "raw" / Path(rel).name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return ml_root / rel

def representative_dataset():
    manifest = load_manifest(PROJECT_ROOT / "data" / "hard_cases_plus_board30.csv")
    # Use a subset for representative data
    for row in manifest[:50]:
        img_path = resolve_image_path(row["image_path"], PROJECT_ROOT)
        if not img_path.exists():
            continue
        img = keras.utils.load_img(img_path, target_size=(224, 224))
        arr = keras.utils.img_to_array(img).astype(np.float32)
        # The model expects [0,1] normalized input
        yield [np.expand_dims(arr / 255.0, axis=0)]

def main():
    model_path = PROJECT_ROOT / "artifacts" / "training" / "scalar_full_finetune_from_best" / "model.keras"
    output_dir = PROJECT_ROOT / "artifacts" / "deployment" / "prod_model_v0.6_scalar_int8"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_int8.tflite"

    print("[EXPORT] Loading model...")
    model = keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
    )
    print(f"[EXPORT] Model: {model.name}")
    print(f"[EXPORT] Inputs: {[i.shape for i in model.inputs]}")
    print(f"[EXPORT] Outputs: {[o.shape for o in model.outputs]}")

    print("[EXPORT] Converting to INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / 1024 / 1024
    print(f"[EXPORT] Saved: {output_path}")
    print(f"[EXPORT] Size: {size_mb:.2f} MB")

    # Save metadata
    metadata = {
        "source_model_path": str(model_path),
        "tflite_path": str(output_path),
        "input_shape": [1, 224, 224, 3],
        "output_shape": [1, 1],
        "deployment_kind": "scalar_hybrid",
        "board_notes": "Prod v0.6: scalar_full_finetune_from_best (test MAE 1.26C)",
    }
    import json
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("[EXPORT] Done.")

if __name__ == "__main__":
    main()
