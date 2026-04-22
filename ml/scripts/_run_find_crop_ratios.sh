#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/rishi_latchmepersad/.local/bin:$PATH"
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

python3 -u - <<'PYEOF'
import numpy as np
from pathlib import Path
from PIL import Image

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
    return np.stack([R, G, B], axis=2), Y

# Load several live captures and compute bright-region bounding box
yuv_files = sorted(CAPTURES_DIR.glob("capture_2026-04-20_16-*.yuv422"))
print(f"Found {len(yuv_files)} live captures")

all_x0, all_y0, all_x1, all_y1 = [], [], [], []

for yuv_path in yuv_files[:8]:
    rgb, Y = load_yuv422(yuv_path)
    luma = Y  # (224,224)

    # Find the dial face: it's the large bright circular region
    # Threshold above background noise
    bright = luma > 80
    rows = np.where(bright.any(axis=1))[0]
    cols = np.where(bright.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        continue
    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    all_x0.append(x0); all_y0.append(y0)
    all_x1.append(x1); all_y1.append(y1)
    print(f"  {yuv_path.name}: bright bbox x={x0}..{x1} y={y0}..{y1}  "
          f"ratios x={x0/224:.3f}..{x1/224:.3f} y={y0/224:.3f}..{y1/224:.3f}")

if all_x0:
    mx0 = int(np.median(all_x0)); mx1 = int(np.median(all_x1))
    my0 = int(np.median(all_y0)); my1 = int(np.median(all_y1))
    print(f"\nMedian dial bbox: x={mx0}..{mx1} y={my0}..{my1}")
    print(f"Median ratios:    x={mx0/224:.3f}..{mx1/224:.3f} y={my0/224:.3f}..{my1/224:.3f}")
    cx = (mx0 + mx1) / 2; cy = (my0 + my1) / 2
    r = min(mx1 - mx0, my1 - my0) / 2
    print(f"Dial center: ({cx:.0f}, {cy:.0f})  radius≈{r:.0f}px")
    print(f"Center ratios: cx={cx/224:.3f} cy={cy/224:.3f} r={r/224:.3f}")

# Also save annotated images showing where needle is for a couple captures
for yuv_path in yuv_files[:2]:
    rgb, Y = load_yuv422(yuv_path)
    # Draw a grid overlay every 22.4px (10% intervals)
    img = Image.fromarray(rgb)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    for i in range(1, 10):
        v = int(i * 22.4)
        draw.line([(v, 0), (v, 223)], fill=(255, 100, 100), width=1)  # vertical
        draw.line([(0, v), (223, v)], fill=(100, 100, 255), width=1)  # horizontal
    # Draw old crop box in red
    TX0, TY0, TX1, TY1 = int(0.103*224), int(0.254*224), int(0.794*224), int(0.803*224)
    draw.rectangle([TX0, TY0, TX1, TY1], outline=(255, 0, 0), width=2)
    img.save(f"/tmp/grid_{yuv_path.stem}.png")
    print(f"\nSaved /tmp/grid_{yuv_path.stem}.png with 10% grid + old crop box (red)")

PYEOF
