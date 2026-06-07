"""
Prepare a heatmap center-detector dataset from CVAT-annotated PXL images.

Pipeline matches the inference flow:
  1. DCMIPP center-crop to square → 320×320 (same as YOLO OBB pipeline)
  2. Compute OBB 4 corners from the ellipse (temp_dial) annotation
  3. Compute perspective transform: OBB → axis-aligned 224×224 rectified crop
  4. Transform the center point (temp_center) through the same warp
  5. Generate 56×56 Gaussian heatmap target centered at transformed center

Output layout:
  ml/data/heatmap_cd/
    images/{train,val}/*.jpg      224×224 rectified gauge crops
    heatmaps/{train,val}/*.npy    56×56 float32 Gaussian heatmaps
    metadata.json                 split info + center coords for validation
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedded_gauge_reading_tinyml.dataset import load_dataset  # noqa: E402
from embedded_gauge_reading_tinyml.heatmap_utils import make_gaussian_heatmap  # noqa: E402

SEED = 42
VAL_FRAC = 0.20
IMG_SIZE = 320
RECTIFIED_SIZE = 224
HEATMAP_SIZE = 56
SIGMA_PIXELS = 2.5
CROP_MARGIN = 0.10
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "heatmap_cd"


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

    Args:
        img_320: 320×320 RGB image as uint8 numpy array.
        obb_corners: 4 OBB corner (x, y) tuples in 320×320 space,
                     ordered [TR, TL, BL, BR] matching ellipse_to_obb_corners.
        center_img_x, center_img_y: Center point in 320×320 image space.
        output_size: Size of output rectified image (square).
        margin: Fractional margin around the OBB in the output.

    Returns:
        warped: output_size × output_size rectified RGB crop (uint8).
        center_out_x, center_out_y: Center point in output image space.
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

    print(f"Done — {len(train_split)} train, {len(val_split)} val — in {out_dir}")


if __name__ == "__main__":
    main()
