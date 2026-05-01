import keras
import tensorflow as tf
from pathlib import Path

# Import preprocess_input for model loading
from keras.applications.mobilenet_v2 import preprocess_input

model_path = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml/ml/artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5_calibrated_all/model.keras")
output_path = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml/ml/artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5_calibrated_all/model.tflite")

print("Loading model...")
model = keras.models.load_model(
    model_path, 
    compile=False,
    custom_objects={'preprocess_input': preprocess_input}
)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"Saved TFLite model: {output_path}")
print(f"Size: {len(tflite_model) / 1024 / 1024:.2f} MB")
