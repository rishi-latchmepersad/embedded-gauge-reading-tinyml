"""Rebuild the scaled OBB model without CoordinateAttention (for Cube.AI compat)
and transfer trained weights, then export TFLite."""
import sys, importlib
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

spec = importlib.util.spec_from_file_location("ts", "scripts/train_obb_scaled_320.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)

# Load trained model
run_dir = Path("artifacts/training/obb_scaled_320_20260610_101410")
trained = keras.saving.load_model(
    run_dir / "best_model.keras",
    custom_objects={"OBBEqualLoss": ts.OBBEqualLoss},
)
print("Loaded trained model with CoordinateAttention.")

# Get backbone weights: the backbone goes from 'image' input to 'obb_coord_attn' input
# Find the layer before CoordinateAttention
coord_attn = trained.get_layer("obb_coord_attn")
gap = trained.get_layer("obb_gap")
print(f"CoordinateAttention input shape: {coord_attn.input.shape}")
print(f"GAP input shape: {gap.input.shape}")

# The backbone output feeds into coord_attn, which feeds into GAP
# We need: backbone → GAP (bypass coord attn)
# Get the backbone up to the point before coord_attn

# Now build a new model WITHOUT CoordinateAttention
from embedded_gauge_reading_tinyml.models import _conv_norm_swish, _residual_separable_block

reg = keras.regularizers.l2(1e-4)
inputs = keras.Input(shape=(320, 320, 3), name="image")

x = _conv_norm_swish(inputs, 48, strides=2, kernel_regularizer=reg)
x = _residual_separable_block(x, 48, dropout_rate=0.05, kernel_regularizer=reg)
x = keras.layers.MaxPool2D(pool_size=2)(x)

x = _residual_separable_block(x, 72, dropout_rate=0.05, kernel_regularizer=reg)
x = keras.layers.MaxPool2D(pool_size=2)(x)

x = _residual_separable_block(x, 96, dropout_rate=0.08, kernel_regularizer=reg)
x = _residual_separable_block(x, 96, dropout_rate=0.08, kernel_regularizer=reg)
x = keras.layers.MaxPool2D(pool_size=2)(x)

x = _residual_separable_block(x, 144, dropout_rate=0.10, kernel_regularizer=reg)
x = _residual_separable_block(x, 144, dropout_rate=0.10, kernel_regularizer=reg)
x = keras.layers.MaxPool2D(pool_size=2)(x)

x = _residual_separable_block(x, 216, dropout_rate=0.12, kernel_regularizer=reg)
x = _residual_separable_block(x, 216, dropout_rate=0.12, kernel_regularizer=reg)
x = _residual_separable_block(x, 216, dropout_rate=0.12, kernel_regularizer=reg)

# NO CoordinateAttention here — directly to GAP
x = keras.layers.GlobalAveragePooling2D(name="obb_gap")(x)
x = keras.layers.LayerNormalization(name="obb_pooled_norm")(x)

x = keras.layers.Dense(128, activation="swish", name="obb_dense_1", kernel_regularizer=reg)(x)
x = keras.layers.Dropout(0.3, name="obb_dropout_1")(x)
x = keras.layers.Dense(64, activation="swish", name="obb_dense_2", kernel_regularizer=reg)(x)
x = keras.layers.Dropout(0.3, name="obb_dropout_2")(x)

center_xy = keras.layers.Dense(2, activation="sigmoid", name="obb_center_xy")(x)
size_wh = keras.layers.Dense(2, activation="sigmoid", name="obb_size_wh")(x)
angle_raw = keras.layers.Dense(2, name="obb_angle_raw")(x)
angle_sincos = keras.layers.UnitNormalization(axis=-1, name="obb_angle_sincos")(angle_raw)
obb_params = keras.layers.Concatenate(name="obb_params")([center_xy, size_wh, angle_sincos])

new_model = keras.Model(inputs=inputs, outputs={"obb_params": obb_params}, name="scaled_obb_no_coord")

# Transfer weights: all backbone layers have the SAME name in both models
# The head layers (dense, dropout, etc.) also have the same names
print("\nTransferring weights layer by layer...")
transferred = 0
skipped = 0
for layer in new_model.layers:
    try:
        trained_layer = trained.get_layer(layer.name)
        trained_weights = trained_layer.get_weights()
        if trained_weights:
            layer.set_weights(trained_weights)
            transferred += 1
        else:
            skipped += 1
    except ValueError:
        skipped += 1

print(f"Transferred: {transferred} layers, Skipped (no match): {skipped}")

# Export to TFLite int8
print("\n=== Exporting TFLite int8 ===")

# Generate representative dataset using test examples
np.random.seed(42)
all_examples = ts._build_all_examples()
from sklearn.model_selection import train_test_split
train_exs, temp_exs = train_test_split(all_examples, test_size=0.20*2, random_state=42)
_, test_exs = train_test_split(temp_exs, test_size=0.5, random_state=42)

def representative_dataset():
    for ex in test_exs[:50]:
        if ex.image_path.endswith(".yuv422"):
            raw = tf.io.read_file(ex.image_path)
            yuyv = tf.io.decode_raw(raw, tf.uint8)
            yuyv = tf.reshape(yuyv, [320, 640])
            y = tf.cast(yuyv[:, 0::2], tf.float32)
            u = tf.cast(yuyv[:, 1::4], tf.float32) - 128.0
            v = tf.cast(yuyv[:, 3::4], tf.float32) - 128.0
            u = tf.repeat(u, 2, axis=1)
            v = tf.repeat(v, 2, axis=1)
            rgb = tf.stack([y + 1.402*v, y - 0.344136*u - 0.714136*v, y + 1.772*u], axis=-1)
            rgb = tf.clip_by_value(rgb, 0, 255) / 255.0
            rgb = tf.ensure_shape(rgb, [320, 320, 3])
            yield [tf.expand_dims(rgb, 0)]
        else:
            img = tf.io.read_file(ex.image_path)
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = ts._preprocess_colour(img, 320, 320)
            yield [tf.expand_dims(img, 0)]

converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
tflite_path = run_dir / "obb_scaled_cubeai_int8.tflite"
tflite_path.write_bytes(tflite_model)
print(f"TFLite int8 saved to: {tflite_path}")
print(f"Size: {len(tflite_model) / 1024:.1f} KB")
