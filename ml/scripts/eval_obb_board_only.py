"""Evaluate the scaled OBB model ONLY on board captures (320×320 YUV + captured_images PNG).

This simulates the real deployment scenario. The model was trained on a random
70/15/15 split of all data (phone + board). We report:
  - Performance on ALL YUV board captures (the deployment data)
  - Performance on YUV captures that were held out during training (if identifiable)

NOTE: Since the original training split was random, some YUV files may have been
in the training set. Results on "all YUV" are slightly optimistic vs true
generalization. We note this in the summary.
"""
import sys, json, importlib, numpy as np, tensorflow as tf, keras
from pathlib import Path
from sklearn.model_selection import train_test_split

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

spec = importlib.util.spec_from_file_location("ts", "scripts/train_obb_scaled_320.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)

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

# Load model
run_dir = Path("artifacts/training/obb_scaled_320_20260610_101410")
model = keras.saving.load_model(run_dir / "best_model.keras",
                                custom_objects={"OBBEqualLoss": OBBEqualLoss})

# Load all training examples (same pool as training)
np.random.seed(42)
all_examples = ts._build_all_examples()

# Filter: board captures = files in captured_images/ directory
board_examples = [ex for ex in all_examples if "captured_images" in ex.image_path]
phone_examples = [ex for ex in all_examples if "captured_images" not in ex.image_path]

# Separate YUV (deployment format) from PNG board captures
board_yuv = [ex for ex in board_examples if ex.image_path.endswith(".yuv422")]
board_png = [ex for ex in board_examples if not ex.image_path.endswith(".yuv422")]

print(f"Phone photos: {len(phone_examples)}")
print(f"Board captures total: {len(board_examples)}")
print(f"  YUV (.yuv422): {len(board_yuv)}")
print(f"  PNG (other):    {len(board_png)}")

# Recreate the original train/val/test split to identify held-out YUV examples
all_paths = [ex.image_path for ex in all_examples]
train_paths, temp_paths = train_test_split(all_paths, test_size=0.20*2, random_state=42)
val_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=42)

# Find YUV files that were held out
yuv_in_train = [p for p in train_paths if p.endswith(".yuv422")]
yuv_in_val = [p for p in val_paths if p.endswith(".yuv422")]
yuv_in_test = [p for p in test_paths if p.endswith(".yuv422")]
yuv_held_out = yuv_in_val + yuv_in_test

print(f"\nYUV split in training: train={len(yuv_in_train)}, val={len(yuv_in_val)}, test={len(yuv_in_test)}")
print(f"  YUV held out (not in training): {len(yuv_held_out)}")

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

def evaluate(exs, label):
    if not exs:
        print(f"\n=== {label}: 0 examples, skipping ===")
        return None
    ds = make_ds(exs)
    targets = np.array([e.obb_params for e in exs])
    preds_raw = model.predict(ds, verbose=0)
    preds = preds_raw["obb_params"]
    mae = np.mean(np.abs(preds - targets), axis=0)
    param_names = ["cx", "cy", "w", "h", "cos2t", "sin2t"]
    print(f"\n=== {label}: {len(exs)} examples ===")
    print(f"  MAE: {np.mean(mae):.4f}")
    for n, m in zip(param_names, mae):
        print(f"    {n}: {m:.4f}")
    cx_err = np.mean(np.abs(preds[:, 0] - targets[:, 0])) * 320
    cy_err = np.mean(np.abs(preds[:, 1] - targets[:, 1])) * 320
    euclidean_err = np.mean(np.sqrt(
        ((preds[:, 0] - targets[:, 0]) * 320) ** 2 +
        ((preds[:, 1] - targets[:, 1]) * 320) ** 2,
    ))
    print(f"  Center error: cx={cx_err:.1f}px, cy={cy_err:.1f}px, euclidean={euclidean_err:.1f}px")
    return {
        "mae": float(np.mean(mae)),
        "cx_err_px": float(cx_err),
        "cy_err_px": float(cy_err),
        "euclidean_err_px": float(euclidean_err),
        "per_param_mae": {n: float(m) for n, m in zip(param_names, mae)},
    }

# Evaluate on YUV board captures (the deployment data)
yuv_all = evaluate(board_yuv, "All YUV Board Captures")
yuv_held = evaluate([ex for ex in board_yuv if ex.image_path in yuv_held_out],
                     "Held-out YUV (val+test, no train leak)")

# Also evaluate on board PNGs
png_results = evaluate(board_png, "Board PNG Captures")

# Save
summary = {
    "phone_count": len(phone_examples),
    "board_png_count": len(board_png),
    "board_yuv_count": len(board_yuv),
    "yuv_in_training": len(yuv_in_train),
    "yuv_held_out_count": len(yuv_held_out),
}
if yuv_all: summary["yuv_all"] = yuv_all
if yuv_held: summary["yuv_held_out"] = yuv_held
if png_results: summary["board_png"] = png_results

with open(run_dir / "board_test_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to {run_dir / 'board_test_summary.json'}")
