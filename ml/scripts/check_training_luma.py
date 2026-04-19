"""Compare luma distribution of training images vs board captures."""
import numpy as np
from pathlib import Path
import sys

REPO = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml")
ML = REPO / "ml"
CAPS = REPO / "captured_images"
IMAGE_SIZE = 224


def load_luma_yuv422(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    yuyv = np.frombuffer(raw, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE // 2, 4)
    luma = np.empty((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    luma[:, 0::2] = yuyv[:, :, 0]
    luma[:, 1::2] = yuyv[:, :, 2]
    return luma


def load_luma_png(path: Path) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(img, dtype=np.uint8)


# Training images from the manifest
import csv
manifest = ML / "data/hard_cases_plus_board30_valid_with_new5_closeup14c.csv"
train_means = []
with open(manifest) as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_col = next((k for k in row if "image" in k.lower() or "path" in k.lower() or "file" in k.lower()), None)
        if img_col is None:
            img_col = list(row.keys())[0]
        p = Path(row[img_col])
        if not p.is_absolute():
            p = ML / p
        if not p.exists():
            continue
        if p.suffix == ".yuv422":
            luma = load_luma_yuv422(p)
        else:
            luma = load_luma_png(p)
        train_means.append(luma.mean())

print(f"Training images: n={len(train_means)}")
if train_means:
    arr = np.array(train_means)
    print(f"  mean={arr.mean():.1f}  std={arr.std():.1f}  min={arr.min():.1f}  max={arr.max():.1f}")

# Board captures (17:xx)
board_means = []
for cap in sorted(CAPS.glob("capture_2026-04-18_17-*.yuv422")):
    if cap.stat().st_size == 0:
        continue
    luma = load_luma_yuv422(cap)
    board_means.append(luma.mean())

print(f"\nBoard captures (17:xx): n={len(board_means)}")
if board_means:
    arr = np.array(board_means)
    print(f"  mean={arr.mean():.1f}  std={arr.std():.1f}  min={arr.min():.1f}  max={arr.max():.1f}")

# Also check 13:xx (14C closeup) captures
c14_means = []
for cap in sorted(CAPS.glob("capture_2026-04-18_13-*.yuv422")):
    if cap.stat().st_size == 0:
        continue
    luma = load_luma_yuv422(cap)
    c14_means.append(luma.mean())

print(f"\n14C closeup captures (13:xx): n={len(c14_means)}")
if c14_means:
    arr = np.array(c14_means)
    print(f"  mean={arr.mean():.1f}  std={arr.std():.1f}  min={arr.min():.1f}  max={arr.max():.1f}")

print(f"\nBrightness gate threshold (DARK_MEAN): 130")
print(f"Board 17:xx captures below gate: {sum(1 for m in board_means if m < 130)}/{len(board_means)}")
