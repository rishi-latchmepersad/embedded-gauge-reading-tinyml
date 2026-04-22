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

def annotate(img_arr, label=""):
    img = Image.fromarray(img_arr).resize((224, 224))
    draw = ImageDraw.Draw(img)
    # 10% grid
    for i in range(1, 10):
        v = int(i * 22.4)
        draw.line([(v, 0), (v, 223)], fill=(200, 50, 50), width=1)
        draw.line([(0, v), (223, v)], fill=(50, 50, 200), width=1)
    # old crop box
    TX0, TY0, TX1, TY1 = int(0.103*224), int(0.254*224), int(0.794*224), int(0.803*224)
    draw.rectangle([TX0, TY0, TX1, TY1], outline=(255, 0, 0), width=2)
    return np.array(img)

# Training samples near 30-35C
training = [
    ("../captured_images/capture_p30c.jpg", 30),
    ("../captured_images/capture_p35c.jpg", 35),
    ("../captured_images/capture_p31c.jpg", 31),
    ("../captured_images/capture_2026-04-03_13-48-34.png", 30),
]

# Live board captures
live_yuv = sorted(CAPTURES_DIR.glob("capture_2026-04-20_16-*.yuv422"))[:3]

frames = []
for p, val in training:
    path = Path(p)
    if path.exists():
        arr = np.array(Image.open(path).convert('RGB').resize((224, 224)))
        frames.append((annotate(arr), f"TRAIN {val}C"))

for yuv_path in live_yuv:
    arr = load_yuv422(yuv_path)
    frames.append((annotate(arr), f"LIVE {yuv_path.name[-12:-7]}"))

# Stitch into a comparison grid (2 rows)
cols = min(4, len(frames))
rows = (len(frames) + cols - 1) // cols
grid = Image.new('RGB', (cols * 224, rows * 224), (40, 40, 40))
for i, (arr, label) in enumerate(frames):
    r, c = divmod(i, cols)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 223, 12], fill=(0, 0, 0))
    draw.text((2, 1), label, fill=(255, 255, 0))
    grid.paste(img, (c * 224, r * 224))

grid.save("/tmp/crop_comparison.png")
print(f"Saved /tmp/crop_comparison.png  size={grid.size}  frames={len(frames)}")

# Also print pixel coords for the needle in live frame
# The needle at 34C on the training p30c image (after resize to 224x224):
# We can estimate from the gauge geometry: center~(112,120), radius~95px
# At 30C: sweep from -30C=135deg to +50C=405deg (270deg total)
# fraction = (30 - (-30)) / (50 - (-30)) = 60/80 = 0.75
# angle from 135deg CW = 135 + 0.75*270 = 135 + 202.5 = 337.5deg from top-right
# In image coords (0deg=right, CCW): needle angle = 90 - 337.5 = -247.5 => 112.5deg
import math
cx, cy, r = 112, 112, 90
for temp, label in [(30, "30C"), (34, "34C"), (35, "35C")]:
    frac = (temp - (-30)) / (50 - (-30))
    # sweep: min at 135deg CW from 12-o'clock (=225deg from positive-x), max at 405deg CW
    angle_cw_from_top = 135 + frac * 270  # degrees CW from 12-o'clock
    angle_rad = math.radians(angle_cw_from_top - 90)  # convert to math convention
    nx = cx + r * math.cos(angle_rad)
    ny = cy + r * math.sin(angle_rad)
    print(f"Needle tip at {label}: pixel ({nx:.0f}, {ny:.0f})  ratio ({nx/224:.3f}, {ny/224:.3f})")
PYEOF
