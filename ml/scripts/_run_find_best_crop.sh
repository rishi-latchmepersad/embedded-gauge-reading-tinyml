#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/rishi_latchmepersad/.local/bin:$PATH"
PYTHON=/home/rishi_latchmepersad/tmp-tf-gpu-test/bin/python3
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

$PYTHON -u - <<'PYEOF'
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

CAPTURES_DIR = Path("../captured_images")
IMAGE_SIZE = 224

def load_yuv422(path, w=224, h=224):
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(h, w * 2)
    Y = raw[:, 0::2].astype(np.float32)
    U = np.repeat(raw[:, 1::4], 2, axis=1).astype(np.float32)
    V = np.repeat(raw[:, 3::4], 2, axis=1).astype(np.float32)
    R = np.clip(Y + 1.402 * (V - 128), 0, 255).astype(np.uint8)
    G = np.clip(Y - 0.344136 * (U - 128) - 0.714136 * (V - 128), 0, 255).astype(np.uint8)
    B = np.clip(Y + 1.772 * (U - 128), 0, 255).astype(np.uint8)
    return np.stack([R, G, B], axis=2)

# Candidate crop ratios to compare: (x0, y0, x1, y1, label)
CANDIDATES = [
    (0.1027, 0.2573, 0.7987, 0.8071, "OLD"),
    (0.05,   0.05,   0.95,   0.95,   "FULL_DIAL"),
    (0.08,   0.08,   0.92,   0.92,   "INNER_DIAL"),
    (0.05,   0.03,   0.95,   0.90,   "TOP_BIAS"),
]

def make_crop_img(rgb_224, x0r, y0r, x1r, y1r, label):
    x0, y0 = int(x0r*224), int(y0r*224)
    x1, y1 = int(x1r*224), int(y1r*224)
    crop = rgb_224[y0:y1, x0:x1]
    # resize crop back to 224x224 (what the model sees)
    resized = np.array(Image.fromarray(crop).resize((224, 224), Image.BILINEAR))
    img = Image.fromarray(resized)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 223, 14], fill=(0,0,0))
    draw.text((2, 2), label, fill=(255,255,0))
    return img

# Load one training sample and one live capture
train_img = np.array(Image.open("../captured_images/capture_p30c.jpg").convert('RGB').resize((224,224)))
live_yuv = sorted(CAPTURES_DIR.glob("capture_2026-04-20_16-*.yuv422"))[0]
live_img = load_yuv422(live_yuv)

# Build grid: rows=sources, cols=crop candidates
sources = [("TRAIN_30C", train_img), ("LIVE", live_img)]
cols = len(CANDIDATES)
rows = len(sources)
grid = Image.new('RGB', (cols * 224, rows * 224), (30, 30, 30))

for r, (src_label, rgb) in enumerate(sources):
    for c, (x0r, y0r, x1r, y1r, clabel) in enumerate(CANDIDATES):
        cell = make_crop_img(rgb, x0r, y0r, x1r, y1r, f"{src_label} | {clabel}")
        grid.paste(cell, (c * 224, r * 224))

grid.save("/tmp/crop_candidates.png")
print(f"Saved /tmp/crop_candidates.png  {grid.size}")
print("Columns: " + " | ".join(f"{c[4]}" for c in CANDIDATES))
print("Row 0 = training 30C, Row 1 = live board capture")
PYEOF
