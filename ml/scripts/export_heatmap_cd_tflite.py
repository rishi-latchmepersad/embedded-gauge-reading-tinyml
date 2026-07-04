"""Export trained Keras heatmap CD model to TFLite (both fp32 and int8)."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "heatmap_cd"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "heatmap_cd"

model = tf.keras.models.load_model(str(ARTIFACT_DIR / "best.keras"))

print("Exporting TFLite models...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_fp32 = converter.convert()
(ARTIFACT_DIR / "heatmap_cd_fp32.tflite").write_bytes(tflite_fp32)
print(f"  float32: {len(tflite_fp32) / 1024:.1f} KB")

with open(DATA_DIR / "metadata.json") as f:
    meta = json.load(f)

val_samples = meta["samples"]["val"]
val_img_paths = [DATA_DIR / "images" / "val" / f"{s['stem']}.jpg" for s in val_samples]

val_images = []
for p in val_img_paths:
    img = tf.io.decode_jpeg(tf.io.read_file(str(p)), channels=3)
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    val_images.append(img)
X_val = tf.stack(val_images, axis=0).numpy()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

def representative_dataset():
    for i in range(min(100, len(X_val))):
        yield [X_val[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_int8 = converter.convert()
(ARTIFACT_DIR / "heatmap_cd_int8.tflite").write_bytes(tflite_int8)
print(f"  int8:   {len(tflite_int8) / 1024:.1f} KB")

print("Done.")
