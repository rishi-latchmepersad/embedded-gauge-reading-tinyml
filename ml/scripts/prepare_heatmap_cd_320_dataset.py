"""
Prepare a 320×320 heatmap center-detector dataset from CVAT PXL images.

Pipeline matches the 320×320 YOLO OBB inference flow:
  1. DCMIPP center-crop to square → 320×320 (same as YOLO OBB)
  2. Compute OBB 4 corners from ellipse (temp_dial)
  3. Perspective warp OBB → axis-aligned 320×320 rectified crop
  4. Transform center point (temp_center) through the same warp
  5. Generate 80×80 Gaussian heatmap target

Also saves a visualization montage to verify the rectified crops look correct.

Output layout:
  ml/data/heatmap_cd_320/
    images/{train,val}/*.jpg      320×320 rectified gauge crops
    heatmaps/{train,val}/*.npy    80×80 float32 Gaussian heatmaps
    metadata.json
    viz_val_rectified.jpg         montage of 9 val crops with center overlay
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedded_gauge_reading_tinyml.dataset import load_dataset  # noqa: E402
from embedded_gauge_reading_tinyml.heatmap_utils import make_gaussian_heatmap  # noqa: E402

SEED = 42
VAL_FRAC = 0.20
IMG_SIZE = 320
RECTIFIED_SIZE = 320
HEATMAP_SIZE = 80
SIGMA_PIXELS = 3.0
CROP_MARGIN = 0.10
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "heatmap_cd_320"


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
) -> list[tuple[float, float]]:
    """Shift corners into DCMIPP crop space and scale to target_size."""
    scale = target_size / crop_side
    return [
        ((x - crop_left) * scale, (y - crop_top) * scale) for x, y in corners
    ]


def compute_rectified_warp_and_center(
    img_320: np.ndarray,
    obb_corners: list[tuple[float, float]],
    center_img_x: float,
    center_img_y: float,
    *,
    output_size: int = RECTIFIED_SIZE,
    margin: float = CROP_MARGIN,
) -> tuple[np.ndarray, float, float]:
    """
    Compute perspective warp from OBB to axis-aligned rectified crop
    and transform the center point through the same warp.
    """
    src_pts = np.array(obb_corners, dtype=np.float32)

    lo = margin * output_size
    hi = (1.0 - margin) * output_size
    dst_pts = np.array(
        [
            [hi, lo],  # TR
            [lo, lo],  # TL
            [lo, hi],  # BL
            [hi, hi],  # BR
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(
        img_320, M, (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    def warp_point(x: float, y: float, matrix: np.ndarray) -> tuple[float, float]:
        src = np.array([x, y, 1.0], dtype=np.float64)
        dst = matrix @ src
        return float(dst[0] / dst[2]), float(dst[1] / dst[2])

    center_out_x, center_out_y = warp_point(center_img_x, center_img_y, M)
    return warped, center_out_x, center_out_y


def draw_center_on_crop(
    crop: np.ndarray,
    cx: float,
    cy: float,
    radius: int = 4,
    color: tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    """Overlay a crosshair at (cx, cy) on the rectified crop."""
    pil = Image.fromarray(crop)
    draw = ImageDraw.Draw(pil)
    x, y = int(round(cx)), int(round(cy))
    draw.line((x - radius * 2, y, x + radius * 2, y), fill=color, width=2)
    draw.line((x, y - radius * 2, x, y + radius * 2), fill=color, width=2)
    return pil


def save_viz_montage(
    samples: list[dict],
    img_dir: Path,
    out_path: Path,
    n_cols: int = 3,
) -> None:
    """Save a grid of rectified crops with target center overlaid."""
    n = n_cols * n_cols
    selected = samples[:n]
    tile_w, tile_h = RECTIFIED_SIZE, RECTIFIED_SIZE + 24
    montage = Image.new("RGB", (tile_w * n_cols, tile_h * n_cols), color=(32, 32, 32))

    for idx, s in enumerate(selected):
        row, col = divmod(idx, n_cols)
        crop = Image.open(str(img_dir / f"{s['stem']}.jpg"))
        cx, cy = s["center_xy_rectified"]
        annotated = draw_center_on_crop(np.array(crop), cx, cy)
        x_off = col * tile_w
        y_off = row * tile_h
        montage.paste(annotated, (x_off, y_off + 24))
        draw = ImageDraw.Draw(montage)
        draw.text((x_off + 4, y_off + 2), s["stem"][:28], fill=(200, 200, 200))

    montage.save(str(out_path), quality=92)
    print(f"  Viz saved to {out_path}")


def main() -> None:
    random.seed(SEED)
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples = load_dataset()
    print(f"Loaded {len(all_samples)} annotated samples")

    random.shuffle(all_samples)
    val_count = int(len(all_samples) * VAL_FRAC)
    val_split = all_samples[:val_count]
    train_split = all_samples[val_count:]

    metadata: dict = {
        "input_size": RECTIFIED_SIZE,
        "heatmap_size": HEATMAP_SIZE,
        "sigma_pixels": SIGMA_PIXELS,
        "crop_margin": CROP_MARGIN,
        "num_samples": len(all_samples),
        "samples": {"train": [], "val": []},
    }

    for split_name, samples in [("train", train_split), ("val", val_split)]:
        img_dir = out_dir / "images" / split_name
        hm_dir = out_dir / "heatmaps" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        hm_dir.mkdir(parents=True, exist_ok=True)

        for sample in tqdm(samples, desc=f"prep {split_name}"):
            pil_img = Image.open(sample.image_path).convert("RGB")

            cropped_img, crop_left, crop_top, crop_side = dcmipp_crop_resize(
                pil_img, IMG_SIZE
            )

            obb_corners = ellipse_to_obb_corners(
                sample.dial.cx, sample.dial.cy,
                sample.dial.rx, sample.dial.ry,
                sample.dial.rotation,
            )

            obb_in_320 = transform_coords_to_crop(
                obb_corners, crop_left, crop_top, crop_side, IMG_SIZE
            )

            scale = IMG_SIZE / crop_side
            center_in_320_x = (sample.center.x - crop_left) * scale
            center_in_320_y = (sample.center.y - crop_top) * scale

            img_320 = np.array(cropped_img)
            warped, cx_out, cy_out = compute_rectified_warp_and_center(
                img_320, obb_in_320, center_in_320_x, center_in_320_y,
            )

            cx_out = max(0.0, min(float(RECTIFIED_SIZE - 1), cx_out))
            cy_out = max(0.0, min(float(RECTIFIED_SIZE - 1), cy_out))

            cx_norm = cx_out / float(RECTIFIED_SIZE - 1)
            cy_norm = cy_out / float(RECTIFIED_SIZE - 1)

            heatmap = make_gaussian_heatmap(
                HEATMAP_SIZE, HEATMAP_SIZE, cx_norm, cy_norm, SIGMA_PIXELS
            )

            stem = sample.image_path.stem
            Image.fromarray(warped).save(str(img_dir / f"{stem}.jpg"), quality=95)
            np.save(str(hm_dir / f"{stem}.npy"), heatmap.astype(np.float32))

            metadata["samples"][split_name].append(
                {
                    "stem": stem,
                    "center_xy_rectified": [float(f"{cx_out:.2f}"), float(f"{cy_out:.2f}")],
                    "center_xy_norm": [float(f"{cx_norm:.6f}"), float(f"{cy_norm:.6f}")],
                }
            )

    metadata["train_count"] = len(train_split)
    metadata["val_count"] = len(val_split)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Save validation visualization montage
    save_viz_montage(
        metadata["samples"]["val"],
        out_dir / "images" / "val",
        out_dir / "viz_val_rectified.jpg",
    )

    print(f"Done — {len(train_split)} train, {len(val_split)} val — in {out_dir}")


if __name__ == "__main__":
    main()
