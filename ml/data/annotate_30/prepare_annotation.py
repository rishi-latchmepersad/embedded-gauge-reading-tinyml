#!/usr/bin/env python3
"""Convert 30 yuv422 captures to PNG and generate annotation HTML."""

import cv2
import numpy as np
from pathlib import Path

INPUT_DIR = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml/ml/data/captured_images")
OUT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = OUT_DIR / "images"

# Select 30 evenly-spaced captures from today
captures = sorted(INPUT_DIR.glob("capture_2026-06-01_*.yuv422"))
selected = captures[::3][:30]

print(f"Converting {len(selected)} captures...")

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

w, h = 224, 224
image_list = []

for f in selected:
    raw = f.read_bytes()
    yuyv = np.frombuffer(raw, dtype=np.uint8).reshape(h, w // 2, 4)

    y0 = yuyv[:, :, 0].astype(np.float32)
    u = yuyv[:, :, 1].astype(np.float32)
    y1 = yuyv[:, :, 2].astype(np.float32)
    v = yuyv[:, :, 3].astype(np.float32)

    y = np.empty((h, w), dtype=np.float32)
    y[:, 0::2] = y0
    y[:, 1::2] = y1

    u_exp = np.repeat(u, 2, axis=1)
    v_exp = np.repeat(v, 2, axis=1)

    r = np.clip(y + (v_exp - 128) * 1436.0 / 1024.0, 0, 255).astype(np.uint8)
    g = np.clip(y - (u_exp - 128) * 352.0 / 1024.0 - (v_exp - 128) * 731.0 / 1024.0, 0, 255).astype(np.uint8)
    b = np.clip(y + (u_exp - 128) * 1814.0 / 1024.0, 0, 255).astype(np.uint8)

    rgb = np.stack([r, g, b], axis=2)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    stem = f.stem
    out_path = IMAGES_DIR / f"{stem}.png"
    cv2.imwrite(str(out_path), bgr)
    image_list.append(f"{stem}.png")

# Write image manifest
with open(OUT_DIR / "images.txt", "w") as fh:
    for name in image_list:
        fh.write(name + "\n")

print(f"Written {len(image_list)} images to {IMAGES_DIR}")
print(f"Image manifest: {OUT_DIR / 'images.txt'}")
print("Now open annotate.html in a browser.")
