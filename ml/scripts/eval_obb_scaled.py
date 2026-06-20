import sys, json, importlib, numpy as np, tensorflow as tf, keras
from pathlib import Path
from sklearn.model_selection import train_test_split

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Load training script as a module
spec = importlib.util.spec_from_file_location("ts", "scripts/train_obb_scaled_320.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)

# OBBEqualLoss for model loading
class OBBEqualLoss(keras.losses.Loss):
    def __init__(self, delta=0.05, reduction="sum_over_batch_size", name="obb_equal_loss"):
        super().__init__(reduction=reduction, name=name)
        self.delta = delta
    def call(self, y_true, y_pred):
        diff = tf.abs(y_true - y_pred)
        quad = 0.5 * tf.square(diff)
        lin = self.delta * (diff - 0.5 * self.delta)
        return tf.reduce_mean(tf.where(diff <= self.delta, quad, lin), axis=-1)
    def get_config(self):
        return {"delta": self.delta}

run_dir = Path("artifacts/training/obb_scaled_320_20260610_101410")
model_path = run_dir / "best_model.keras"
print(f"Loading model from {model_path}")
model = keras.saving.load_model(model_path, custom_objects={"OBBEqualLoss": OBBEqualLoss})

np.random.seed(42)
examples = ts._build_all_examples()
train_exs, temp_exs = train_test_split(examples, test_size=0.20 * 2, random_state=42)
_, test_exs = train_test_split(temp_exs, test_size=0.5, random_state=42)
print(f"Test examples: {len(test_exs)}")

def make_ds(exs):
    paths = [e.image_path for e in exs]
    values = [e.value for e in exs]
    obb = [e.obb_params for e in exs]
    crops = [e.crop_box_xyxy for e in exs]
    wgts = np.ones(len(exs), dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(paths), tf.constant(values, dtype=tf.float32),
        tf.constant(np.array(obb, dtype=np.float32)),
        tf.constant(np.array(crops, dtype=np.float32)), tf.constant(wgts),
    ))
    ds = ds.map(lambda p,v,o,c,w: ts._load_fullframe_obb_data_colour(p,v,o,c,320,320,w),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(16).prefetch(tf.data.AUTOTUNE)
    return ds

test_ds = make_ds(test_exs)
test_results = model.evaluate(test_ds, verbose=0)
print(f"  Test MAE: {test_results[1]:.4f}")

test_preds_raw = model.predict(test_ds, verbose=0)
test_preds = test_preds_raw["obb_params"]
test_targets = np.array([e.obb_params for e in test_exs])
per_param_mae = np.mean(np.abs(test_preds - test_targets), axis=0)
param_names = ["cx", "cy", "w", "h", "cos2t", "sin2t"]
for n, m in zip(param_names, per_param_mae):
    print(f"    {n}: {m:.4f}")

cx_err = np.mean(np.abs(test_preds[:, 0] - test_targets[:, 0])) * 320
cy_err = np.mean(np.abs(test_preds[:, 1] - test_targets[:, 1])) * 320
euclidean_err = np.mean(np.sqrt(
    ((test_preds[:, 0] - test_targets[:, 0]) * 320) ** 2 +
    ((test_preds[:, 1] - test_targets[:, 1]) * 320) ** 2,
))
print(f"\n  Center error @320: cx={cx_err:.1f}px, cy={cy_err:.1f}px, euclidean={euclidean_err:.1f}px")

print("\n=== Exporting TFLite int8 ===")
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

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
tflite_path = run_dir / "obb_scaled_int8.tflite"
tflite_path.write_bytes(tflite_model)
print(f"  TFLite saved to: {tflite_path}")
print(f"  Size: {len(tflite_model) / 1024:.1f} KB")
