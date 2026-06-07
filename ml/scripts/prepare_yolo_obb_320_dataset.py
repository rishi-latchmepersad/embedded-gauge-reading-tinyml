"""
Prepare a YOLO OBB 320×320 dataset from CVAT-annotated PXL images.

Simulates the DCMIPP crop pipeline:
  1. Center-crop to a square (short side of the sensor / image)
  2. Downscale to 320×320 via bilinear resize
  3. Convert CVAT ellipse annotation → YOLO OBB 4-corner format

Output layout:
  ml/data/yolo_obb_320/
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt
    dataset.yaml
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedded_gauge_reading_tinyml.dataset import load_dataset  # noqa: E402

SEED = 42
VAL_FRAC = 0.20
IMG_SIZE = 320
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "yolo_obb_320"
CLASS_NAME = "gauge"


def ellipse_to_obb_corners(
    cx: float, cy: float, rx: float, ry: float, rotation: float
) -> list[tuple[float, float]]:
    """Convert a CVAT ellipse to 4 OBB corners (clockwise from top-right)."""
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)

    return [
        (cx + rx * cos_r + ry * sin_r, cy + rx * sin_r - ry * cos_r),
        (cx - rx * cos_r + ry * sin_r, cy - rx * sin_r - ry * cos_r),
        (cx - rx * cos_r - ry * sin_r, cy - rx * sin_r + ry * cos_r),
        (cx + rx * cos_r - ry * sin_r, cy + rx * sin_r + ry * cos_r),
    ]


def dcmipp_crop_resize(
    image: Image.Image, target_size: int
) -> tuple[Image.Image, int, int, int]:
    """Center-crop to square (short side), then resize to target_size."""
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = image.crop((left, top, left + side, top + side))
    resized = cropped.resize((target_size, target_size), Image.BILINEAR)
    return resized, left, top, side


def transform_coords_to_crop(
    corners: list[tuple[float, float]],
    crop_left: int,
    crop_top: int,
    crop_side: int,
    target_size: int,
) -> list[float]:
    """Shift corners into crop space, scale to target, normalise to [0, 1]."""
    scale = target_size / crop_side
    out: list[float] = []
    for x, y in corners:
        out.append((x - crop_left) * scale / target_size)
        out.append((y - crop_top) * scale / target_size)
    return out


def main() -> None:
    random.seed(SEED)

    all_samples = load_dataset()
    print(f"Loaded {len(all_samples)} annotated samples")

    random.shuffle(all_samples)
    val_count = int(len(all_samples) * VAL_FRAC)
    val_split = all_samples[:val_count]
    train_split = all_samples[val_count:]

    for split_name, samples in [("train", train_split), ("val", val_split)]:
        img_dir = OUT_DIR / "images" / split_name
        lbl_dir = OUT_DIR / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for sample in tqdm(samples, desc=f"{split_name}"):
            img = Image.open(sample.image_path).convert("RGB")

            # DCMIPP-style center-crop to square → 320×320
            cropped_img, crop_left, crop_top, crop_side = dcmipp_crop_resize(
                img, IMG_SIZE
            )

            stem = sample.image_path.stem
            cropped_img.save(str(img_dir / f"{stem}.jpg"), quality=95)

            # Convert ellipse to OBB and transform through crop+resize
            dial = sample.dial
            corners = ellipse_to_obb_corners(
                dial.cx, dial.cy, dial.rx, dial.ry, dial.rotation
            )
            obb_coords = transform_coords_to_crop(
                corners, crop_left, crop_top, crop_side, IMG_SIZE
            )

            # YOLO OBB: class_id x1 y1 x2 y2 x3 y3 x4 y4
            line = f"0 " + " ".join(f"{c:.6f}" for c in obb_coords)
            (lbl_dir / f"{stem}.txt").write_text(line + "\n")

    # Dataset YAML
    yaml_content = (
        f"path: {OUT_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )
    (OUT_DIR / "dataset.yaml").write_text(yaml_content)

    print(f"Done — {len(train_split)} train, {len(val_split)} val — in {OUT_DIR}")


if __name__ == "__main__":
    main()
