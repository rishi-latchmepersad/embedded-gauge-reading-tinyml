"""Quick fine-tune OBB head for needle pivot. Preloads all images into RAM for speed."""
from __future__ import annotations
import json, sys
from datetime import datetime
from pathlib import Path

import keras
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedded_gauge_reading_tinyml.training import (
    TrainConfig, _build_training_examples, _split_examples,
    _load_crop_and_preprocess_image,
)
from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs

ML_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ML_ROOT / "artifacts" / "training"
WARMSTART = ARTIFACTS / "obb_improved_20260530_194719" / "model.keras"
RUN_DIR = ARTIFACTS / f"obb_pivot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

H, W = 224, 224
BATCH = 8
EPOCHS = 5
LR = 1e-4

print("[FT] Loading data...")
samples = load_dataset(labelled_dir=ML_ROOT/"data/labelled", raw_dir=ML_ROOT/"data/raw")
spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
examples, _ = _build_training_examples(samples, spec, image_height=H, image_width=W,
    keypoint_heatmap_size=28, strict_labels=False, crop_pad_ratio=0.25)
config = TrainConfig(gauge_id="littlegood_home_temp_gauge_c", seed=21, test_fraction=0.15, val_fraction=0.15)
split = _split_examples(examples, config)

def preload(ex_list):
    imgs, targets = [], []
    for ex in ex_list:
        img, _ = _load_crop_and_preprocess_image(ex.image_path, 0.0, ex.crop_box_xyxy, H, W)
        imgs.append(img.numpy())
        t = ex.obb_params.copy()
        t[0] = ex.center_xy[0] / W  # cx → needle pivot
        t[1] = ex.center_xy[1] / H  # cy → needle pivot
        targets.append(t)
    return np.array(imgs), np.array(targets, dtype=np.float32)

print("[FT] Preloading train...")
x_train, y_train = preload(split.train_examples)
print(f"  {x_train.shape[0]} samples")
print("[FT] Preloading val...")
x_val, y_val = preload(split.val_examples)
print(f"  {x_val.shape[0]} samples")

print("[FT] Loading warm-start model...")
model = keras.models.load_model(str(WARMSTART), compile=False)

# Freeze backbone only, keep all head layers trainable
model.layers[2].trainable = False  # mobilenetv2 backbone
print(f"[FT] Trainable weights: {len(model.trainable_weights)}")

# Only supervise center params (indices 0,1), zero-weight others
class CenterOnlyLoss(keras.losses.Loss):
    def __init__(self, delta=0.03, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta
        self.pw = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    def call(self, y_true, y_pred):
        err = y_pred - y_true
        abs_err = tf.abs(err)
        quad = tf.minimum(abs_err, self.delta)
        lin = abs_err - quad
        huber = 0.5*quad*quad + self.delta*lin
        return tf.reduce_mean(huber * self.pw[tf.newaxis, :])
    def get_config(self):
        return {"delta": self.delta}

import tensorflow as tf
model.compile(
    optimizer=keras.optimizers.Adam(LR),
    loss=CenterOnlyLoss(),
    metrics=[keras.metrics.MeanAbsoluteError()],
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=BATCH,
    epochs=EPOCHS,
    verbose=1,
)

# Test
print("[FT] Preloading test...")
x_test_all, y_test_all = preload(split.test_examples)
preds_dict = model.predict(x_test_all, verbose=0)
preds = preds_dict['obb_params'] if isinstance(preds_dict, dict) else preds_dict
cx_err = np.abs(preds[:, 0] - y_test_all[:, 0]) * W
cy_err = np.abs(preds[:, 1] - y_test_all[:, 1]) * H
center_dist = np.sqrt(cx_err**2 + cy_err**2)
print(f"[FT] Test center accuracy: cx MAE={cx_err.mean():.2f}px cy MAE={cy_err.mean():.2f}px "
      f"Euclidean={center_dist.mean():.2f}px (±{center_dist.std():.2f})")
print(f"[FT]  ≤2px: {(center_dist<=2).mean()*100:.0f}%  ≤5px: {(center_dist<=5).mean()*100:.0f}%")

model.save(str(RUN_DIR / "model.keras"))
(RUN_DIR / "metrics.json").write_text(json.dumps({
    "test_center_mae_px": float(center_dist.mean()),
    "test_cx_mae_px": float(cx_err.mean()),
    "test_cy_mae_px": float(cy_err.mean()),
}, indent=2), encoding="utf-8")

print(f"[FT] Done → {RUN_DIR / 'model.keras'}")
