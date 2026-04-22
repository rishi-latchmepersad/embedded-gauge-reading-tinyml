#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/rishi_latchmepersad/.local/bin:$PATH"
cd /mnt/d/Projects/embedded-gauge-reading-tinyml

RECTIFIER_LOCAL="${HOME}/ml_eval_cache/rectifier_zoom_aug_v4.model.keras"
RECTIFIER_SRC="ml/artifacts/training/mobilenetv2_rectifier_zoom_aug_v4/model.keras"
MANIFEST="ml/data/full_labelled_plus_board30_valid_with_new5.csv"
OUT_CSV="ml/data/rectified_crop_boxes_v4_all.csv"

# Copy rectifier to local WSL storage to avoid /mnt/d stalls
cp -f "${RECTIFIER_SRC}" "${RECTIFIER_LOCAL}"

cd ml
"${HOME}/.local/bin/poetry" run python -u - <<'PYEOF'
import os, sys, csv, numpy as np
from pathlib import Path

sys.path.insert(0, "src")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU only — keep GPU free for training
import tensorflow as tf

RECTIFIER_LOCAL = Path(os.environ["HOME"]) / "ml_eval_cache/rectifier_zoom_aug_v4.model.keras"
MANIFEST        = Path("data/full_labelled_plus_board30_valid_with_new5.csv")
OUT_CSV         = Path("data/rectified_crop_boxes_v4_all.csv")
IMAGE_SIZE      = 224
CROP_SCALE      = 1.25

from embedded_gauge_reading_tinyml.board_crop_compare import load_rgb_image, resize_with_pad_rgb

print(f"[PRECOMPUTE] Loading rectifier from {RECTIFIER_LOCAL}", flush=True)
rectifier = tf.keras.models.load_model(str(RECTIFIER_LOCAL), compile=False)
print("[PRECOMPUTE] Rectifier loaded.", flush=True)

# Read manifest
rows = []
with open(MANIFEST) as f:
    for row in csv.DictReader(f):
        rows.append(row)

print(f"[PRECOMPUTE] Processing {len(rows)} images...", flush=True)

out_rows = []
REPO_ROOT = Path("..").resolve()

for i, row in enumerate(rows, 1):
    image_path = Path(row["image_path"])
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path
    value = row["value"]

    src = load_rgb_image(image_path)
    oh, ow = src.shape[:2]
    full = resize_with_pad_rgb(src, (0.0, 0.0, float(ow), float(oh)), image_size=IMAGE_SIZE)
    inp = np.expand_dims(full.astype(np.float32) / 255.0, 0)

    pred = rectifier.predict(inp, verbose=0)
    if isinstance(pred, dict):
        box = np.asarray(list(pred.values())[0]).reshape(-1)
    else:
        box = np.asarray(pred).reshape(-1)

    cx, cy = float(np.clip(box[0], 0, 1)), float(np.clip(box[1], 0, 1))
    bw = min(1.0, float(np.clip(box[2], 0.05, 1.0)) * CROP_SCALE)
    bh = min(1.0, float(np.clip(box[3], 0.05, 1.0)) * CROP_SCALE)

    # Plausibility guard — same as training.py
    if bw < 0.25 or bh < 0.25 or bw > 0.95 or bh > 0.95:
        # fall back to fixed training crop in canvas coords
        x0, y0, x1, y1 = 0.1027*IMAGE_SIZE, 0.2573*IMAGE_SIZE, 0.7987*IMAGE_SIZE, 0.8071*IMAGE_SIZE
    else:
        cw, ch = float(IMAGE_SIZE), float(IMAGE_SIZE)
        x0c = max(0.0, (cx - 0.5*bw) * cw)
        y0c = max(0.0, (cy - 0.5*bh) * ch)
        x1c = min(cw,  (cx + 0.5*bw) * cw)
        y1c = min(ch,  (cy + 0.5*bh) * ch)
        # map back to original image coords
        scale = min(cw / ow, ch / oh)
        pad_x = (cw - ow * scale) * 0.5
        pad_y = (ch - oh * scale) * 0.5
        x0 = max(0.0, (x0c - pad_x) / scale)
        y0 = max(0.0, (y0c - pad_y) / scale)
        x1 = min(float(ow), (x1c - pad_x) / scale)
        y1 = min(float(oh), (y1c - pad_y) / scale)
        if x1 <= x0 + 1: x1 = min(float(ow), x0 + 1.0)
        if y1 <= y0 + 1: y1 = min(float(oh), y0 + 1.0)

    out_rows.append({"image_path": row["image_path"], "value": value,
                     "x0": f"{x0:.2f}", "y0": f"{y0:.2f}",
                     "x1": f"{x1:.2f}", "y1": f"{y1:.2f}"})

    if i % 10 == 0 or i == len(rows):
        print(f"[PRECOMPUTE] {i}/{len(rows)}", flush=True)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image_path", "value", "x0", "y0", "x1", "y1"])
    w.writeheader()
    w.writerows(out_rows)

print(f"[PRECOMPUTE] Written {len(out_rows)} rows to {OUT_CSV}", flush=True)
PYEOF
