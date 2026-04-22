#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/rishi_latchmepersad/.local/bin:$PATH"
PYTHON=/home/rishi_latchmepersad/tmp-tf-gpu-test/bin/python3
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

$PYTHON -u - <<'PYEOF'
import sys, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, "src")
import tensorflow as tf

RECTIFIER_KERAS = Path("artifacts/training/mobilenetv2_rectifier_zoom_aug_v4/model.keras")
CAPTURES_DIR = Path("../captured_images")
IMAGE_SIZE = 224
CROP_SCALE = 1.25

TX0, TY0, TX1, TY1 = int(0.103*224), int(0.254*224), int(0.794*224), int(0.803*224)

# Inline YUV load
def load_yuv422(path, w=224, h=224):
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(h, w * 2)
    Y = raw[:, 0::2].astype(np.float32)
    U = np.repeat(raw[:, 1::4], 2, axis=1).astype(np.float32)
    V = np.repeat(raw[:, 3::4], 2, axis=1).astype(np.float32)
    R = np.clip(Y + 1.402 * (V - 128), 0, 255).astype(np.uint8)
    G = np.clip(Y - 0.344136 * (U - 128) - 0.714136 * (V - 128), 0, 255).astype(np.uint8)
    B = np.clip(Y + 1.772 * (U - 128), 0, 255).astype(np.uint8)
    return np.stack([R, G, B], axis=2)

print("Loading rectifier v4...")
rectifier = tf.keras.models.load_model(str(RECTIFIER_KERAS), compile=False)
print("Ready.")

yuv = sorted(CAPTURES_DIR.glob("capture_2026-04-20_16-*.yuv422"))[0]
rgb = load_yuv422(yuv)

# Run rectifier
raw_pred = rectifier.predict(rgb.astype(np.float32)[None], verbose=0)
if isinstance(raw_pred, dict):
    raw_pred = list(raw_pred.values())[0]
box = np.array(raw_pred).reshape(-1)
cx, cy = float(box[0]), float(box[1])
bw = min(1.0, float(np.clip(box[2], 0.05, 1.0)) * CROP_SCALE)
bh = min(1.0, float(np.clip(box[3], 0.05, 1.0)) * CROP_SCALE)
rx0 = int(max(0, (cx - bw / 2) * IMAGE_SIZE))
ry0 = int(max(0, (cy - bh / 2) * IMAGE_SIZE))
rx1 = int(min(IMAGE_SIZE, (cx + bw / 2) * IMAGE_SIZE))
ry1 = int(min(IMAGE_SIZE, (cy + bh / 2) * IMAGE_SIZE))
print(f"Rectifier box: cx={cx:.3f} cy={cy:.3f} bw={bw:.3f} bh={bh:.3f}")
print(f"Pixel box: x={rx0} y={ry0} w={rx1-rx0} h={ry1-ry0}")

# Draw annotated full frame
full_img = Image.fromarray(rgb)
draw = ImageDraw.Draw(full_img)
draw.rectangle([TX0, TY0, TX1, TY1], outline=(255, 0, 0), width=2)     # red = fixed crop
draw.rectangle([rx0, ry0, rx1, ry1], outline=(0, 255, 0), width=2)     # green = rectifier crop
full_img.save("/tmp/full_annotated.png")

# Save rectifier crop
rect_crop = rgb[ry0:ry1, rx0:rx1]
Image.fromarray(rect_crop).save("/tmp/rectifier_crop.png")

print("Saved /tmp/full_annotated.png (red=fixed, green=rectifier) and /tmp/rectifier_crop.png")
PYEOF
