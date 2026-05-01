#!/usr/bin/env python3
"""Export best Keras model to TFLite format."""

import keras
import tensorflow as tf
from pathlib import Path

model_path = Path('artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5_calibrated_all/model.keras')
model = keras.models.load_model(model_path, compile=False)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save
output_path = Path('artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5_calibrated_all/model.tflite')
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f'Exported TFLite model to {output_path}')
print(f'Model size: {len(tflite_model) / 1024 / 1024:.2f} MB')
