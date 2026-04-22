#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/rishi_latchmepersad/.local/bin:$PATH"
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

python3 -u - <<'PYEOF'
import sys, numpy as np
from pathlib import Path
from PIL import Image

CAPTURES_DIR = Path("../captured_images")
IMAGE_SIZE = 224
TX0, TY0, TX1, TY1 = int(0.103*224), int(0.254*224), int(0.794*224), int(0.803*224)

def load_yuv422(path, w=224, h=224):
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    # YUYV packed: each 4 bytes = 2 pixels
    raw = raw.reshape(h, w, 2)
    y = raw[:, :, 0].astype(np.float32)
    u = raw[:, :, 1].astype(np.float32)
    # simple YUV->RGB (BT.601)
    r = np.clip(y + 1.402 * (u - 128), 0, 255).astype(np.uint8)
    g = np.clip(y - 0.344136 * (u - 128) - 0.714136 * (u - 128), 0, 255).astype(np.uint8)
    b = np.clip(y + 1.772 * (u - 128), 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=2)

yuv = sorted(CAPTURES_DIR.glob("capture_2026-04-20_16-*.yuv422"))[0]
print(f"Loading: {yuv.name}")

# Use the actual YUV422 YUYV format
raw = np.frombuffer(yuv.read_bytes(), dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE * 2)
# YUYV: Y0 U0 Y1 V0 for each pair of pixels
Y = raw[:, 0::2].astype(np.float32)
U = np.repeat(raw[:, 1::4], 2, axis=1).astype(np.float32)
V = np.repeat(raw[:, 3::4], 2, axis=1).astype(np.float32)
R = np.clip(Y + 1.402 * (V - 128), 0, 255).astype(np.uint8)
G = np.clip(Y - 0.344136 * (U - 128) - 0.714136 * (V - 128), 0, 255).astype(np.uint8)
B = np.clip(Y + 1.772 * (U - 128), 0, 255).astype(np.uint8)
rgb = np.stack([R, G, B], axis=2)

Image.fromarray(rgb).save("/tmp/full_frame.png")
crop = rgb[TY0:TY1, TX0:TX1]
Image.fromarray(crop).save("/tmp/fixed_crop.png")

print(f"Full frame: shape={rgb.shape}  luma_mean={Y.mean():.1f}")
print(f"Fixed crop [y={TY0}:{TY1}, x={TX0}:{TX1}]: shape={crop.shape}  luma_mean={Y[TY0:TY1, TX0:TX1].mean():.1f}")
print(f"Saved /tmp/full_frame.png and /tmp/fixed_crop.png")

# Also save a few training PNGs for comparison
for name in ["capture_2026-04-03_15-46-04.png", "capture_2026-04-03_08-20-49.png"]:
    p = CAPTURES_DIR / name
    if p.exists():
        tr = np.array(Image.open(p).convert('RGB'))
        tr_crop = tr[TY0:TY1, TX0:TX1]
        Image.fromarray(tr_crop).save(f"/tmp/train_crop_{p.stem[-8:]}.png")
        print(f"Training {name}: shape={tr.shape}  luma_mean={tr[:,:,0].mean():.1f}  crop_luma={tr[TY0:TY1,TX0:TX1,0].mean():.1f}")
PYEOF
