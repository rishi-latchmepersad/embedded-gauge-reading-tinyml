"""Export best model to TFLite and evaluate on hard cases."""

import keras
import tensorflow as tf
from pathlib import Path
import csv
import numpy as np
import sys

# Paths - use absolute paths for WSL compatibility
ML_ROOT = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml/ml")
REPO_ROOT = ML_ROOT.parent
model_path = (
    ML_ROOT
    / "artifacts"
    / "training"
    / "scalar_full_finetune_from_best_board30_clean_plus_new5_calibrated_all"
    / "model.keras"
)
output_path = (
    REPO_ROOT
    / "artifacts"
    / "training"
    / "scalar_full_finetune_from_best_board30_clean_plus_new5_calibrated_all"
    / "model.tflite"
)

print(f"[EXPORT] Loading model from {model_path}")
model = keras.models.load_model(model_path, compile=False)

print("[EXPORT] Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(output_path, "wb") as f:
    f.write(tflite_model)

print(f"[EXPORT] Saved TFLite model to {output_path}")
print(f"[EXPORT] Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")

# Evaluate on hard cases
print("\n[EVAL] Loading hard cases...")
manifest_path = REPO_ROOT / "data" / "hard_cases_plus_board30_valid_with_new6.csv"

# Load manifest
samples = []
with open(manifest_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["image_path"].startswith("#"):
            continue
        img_path = row["image_path"]
        if not Path(img_path).is_absolute():
            img_path = str(REPO_ROOT.parent / img_path)
        samples.append((img_path, float(row["value"])))

print(f"[EVAL] Loaded {len(samples)} hard cases")

# Evaluate with TFLite interpreter
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("[EVAL] Running inference...")
errors = []
for img_path, true_val in samples:
    try:
        # Load and preprocess image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize_with_pad(img, 224, 224)
        img = tf.cast(img, tf.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        pred_norm = interpreter.get_tensor(output_details[0]["index"])[0][0]

        # Denormalize prediction (assuming -30 to 50 range)
        pred_val = (pred_norm + 1) / 2 * (50 - (-30)) + (-30)

        error = abs(pred_val - true_val)
        errors.append(error)

        if error > 10:
            print(
                f"  Large error: {img_path} - True: {true_val:.1f}C, Pred: {pred_val:.1f}C, Error: {error:.1f}C"
            )

    except Exception as e:
        print(f"  Error processing {img_path}: {e}")

mae = np.mean(errors)
print(f"\n[EVAL] MAE on hard cases: {mae:.2f}C")
print(f"[EVAL] Max error: {max(errors):.2f}C")
print(f"[EVAL] Errors > 10C: {sum(1 for e in errors if e > 10)}")
