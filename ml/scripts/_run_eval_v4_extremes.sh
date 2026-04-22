#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/rishi_latchmepersad/.local/bin:$PATH"
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 poetry run python -u - <<'PYEOF'
import sys, csv, numpy as np
from pathlib import Path
sys.path.insert(0, "src")
import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf

SCALAR_NEW = Path("artifacts/training/scalar_extremes_v4/model.keras")
SCALAR_OLD = Path("artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5/model.keras")
MANIFEST   = Path("data/hard_cases_plus_board30_valid_with_new5.csv")
IMAGE_SIZE = 224

from embedded_gauge_reading_tinyml.board_crop_compare import load_rgb_image, resize_with_pad_rgb
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import keras
keras.saving.get_custom_objects()["preprocess_input"] = preprocess_input

print("Loading new scalar (v4)...", flush=True)
scalar_new = tf.keras.models.load_model(str(SCALAR_NEW), compile=False)
print("Loading old scalar (prod)...", flush=True)
scalar_old = tf.keras.models.load_model(str(SCALAR_OLD), compile=False)
print("Models loaded.", flush=True)

REPO_ROOT = Path("..").resolve()

def _scalar_run(scalar_model, crop):
    crop_in = np.expand_dims(crop.astype(np.float32) / 255.0, 0)
    out = scalar_model.predict(crop_in, verbose=0)
    if isinstance(out, dict): return float(list(out.values())[0].flat[0])
    return float(np.asarray(out).flat[0])

def _fixed_crop(src):
    oh, ow = src.shape[:2]
    x0,y0,x1,y1 = 0.1027*ow, 0.2573*oh, 0.7987*ow, 0.8071*oh
    return resize_with_pad_rgb(src, (x0,y0,x1,y1), image_size=IMAGE_SIZE)

rows = list(csv.DictReader(open(MANIFEST)))
print(f"\n{'Image':<45} {'True':>6} {'Old':>7} {'New':>7} {'OldErr':>8} {'NewErr':>8}")
print("-"*85)
old_errs, new_errs = [], []
for row in rows:
    p = Path(row["image_path"])
    if not p.is_absolute(): p = REPO_ROOT / p
    true_val = float(row["value"])
    try:
        src = load_rgb_image(p)
        crop = _fixed_crop(src)
        old_pred = _scalar_run(scalar_old, crop)
        new_pred = _scalar_run(scalar_new, crop)
        old_err = abs(old_pred - true_val)
        new_err = abs(new_pred - true_val)
        old_errs.append(old_err); new_errs.append(new_err)
        print(f"{Path(row['image_path']).name:<45} {true_val:>6.1f} {old_pred:>7.1f} {new_pred:>7.1f} {old_err:>8.2f} {new_err:>8.2f}")
    except Exception as e:
        print(f"{Path(row['image_path']).name:<45} ERROR: {e}")

print("-"*85)
print(f"{'MAE':<45} {'':>6} {'':>7} {'':>7} {np.mean(old_errs):>8.3f} {np.mean(new_errs):>8.3f}")
print(f"{'Max error':<45} {'':>6} {'':>7} {'':>7} {np.max(old_errs):>8.3f} {np.max(new_errs):>8.3f}")
print(f"{'Cases >5C':<45} {'':>6} {'':>7} {'':>7} {sum(e>5 for e in old_errs):>8} {sum(e>5 for e in new_errs):>8}")
print(f"{'Extremes (|val|>=25) MAE':<45} {'':>6} {'':>7} {'':>7} {np.mean([e for e,r in zip(old_errs,rows) if abs(float(r['value']))>=25]):>8.3f} {np.mean([e for e,r in zip(new_errs,rows) if abs(float(r['value']))>=25]):>8.3f}")
PYEOF
