#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/rishi_latchmepersad/.local/bin:$PATH"
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

CUDA_VISIBLE_DEVICES="" poetry run python -u - <<'PYEOF'
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, "src")
import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from PIL import Image, ImageDraw

RECTIFIER  = Path("/home/rishi_latchmepersad/ml_eval_cache/rectifier_zoom_aug_v4.model.keras")
CROP_SCALE = 1.25
IMAGE_SIZE = 224
REPO_ROOT  = Path("..").resolve()

from embedded_gauge_reading_tinyml.board_crop_compare import load_rgb_image, resize_with_pad_rgb

print("Loading rectifier...", flush=True)
rect = tf.keras.models.load_model(str(RECTIFIER), compile=False)

images = [
    "captured_images/today_converted/capture_2026-04-09_06-41-57.png",
    "captured_images/today_converted/capture_2026-04-09_06-50-28.png",
    "captured_images/today_converted/capture_2026-04-09_06-51-13.png",
]

for rel in images:
    p = REPO_ROOT / rel
    src = load_rgb_image(p)
    oh, ow = src.shape[:2]
    full = resize_with_pad_rgb(src, (0.0, 0.0, float(ow), float(oh)), image_size=IMAGE_SIZE)
    inp = np.expand_dims(full.astype(np.float32) / 255.0, 0)

    pred = rect.predict(inp, verbose=0)
    if isinstance(pred, dict): box = np.asarray(list(pred.values())[0]).reshape(-1)
    else: box = np.asarray(pred).reshape(-1)
    cx, cy = float(np.clip(box[0],0,1)), float(np.clip(box[1],0,1))
    bw_raw = float(np.clip(box[2],0.05,1.0))
    bh_raw = float(np.clip(box[3],0.05,1.0))
    bw = min(1.0, bw_raw * CROP_SCALE)
    bh = min(1.0, bh_raw * CROP_SCALE)
    plausible = not (bw < 0.25 or bh < 0.25 or bw > 0.95 or bh > 0.95)

    print(f"\n{Path(rel).name}: {ow}x{oh}")
    print(f"  raw box: cx={cx:.3f} cy={cy:.3f} bw={bw_raw:.3f} bh={bh_raw:.3f}")
    print(f"  scaled:  bw={bw:.3f} bh={bh:.3f}  plausible={plausible}")

    # Draw annotated image
    img = Image.fromarray(full)
    draw = ImageDraw.Draw(img)
    # Fixed crop (red)
    fx0,fy0,fx1,fy1 = 0.1027*IMAGE_SIZE, 0.2573*IMAGE_SIZE, 0.7987*IMAGE_SIZE, 0.8071*IMAGE_SIZE
    draw.rectangle([fx0,fy0,fx1,fy1], outline="red", width=2)
    # Rectifier crop (green) — in canvas coords
    if plausible:
        rx0 = max(0.0,(cx-0.5*bw)*IMAGE_SIZE)
        ry0 = max(0.0,(cy-0.5*bh)*IMAGE_SIZE)
        rx1 = min(float(IMAGE_SIZE),(cx+0.5*bw)*IMAGE_SIZE)
        ry1 = min(float(IMAGE_SIZE),(cy+0.5*bh)*IMAGE_SIZE)
        draw.rectangle([rx0,ry0,rx1,ry1], outline="lime", width=2)
    else:
        draw.rectangle([fx0,fy0,fx1,fy1], outline="orange", width=2)  # fallback
    out_path = f"/mnt/d/Projects/embedded-gauge-reading-tinyml/ml/artifacts/debug_{Path(rel).stem}.png"
    img.save(out_path)
    print(f"  saved: {out_path}")
PYEOF
