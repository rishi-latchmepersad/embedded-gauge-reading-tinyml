"""
End-to-end pipeline validator.

Tests:
  1. Heatmap CD only on PXL val set (GT OBB → rectify → CD → compare center)
  2. Heatmap CD on board captures (qualitative visual check)
  3. Saves overlay visualizations

Usage:
  python scripts/validate_pipeline.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedded_gauge_reading_tinyml.dataset import load_dataset  # noqa: E402

SEED = 42
VAL_FRAC = 0.20
IMG_SIZE = 320
RECTIFIED_SIZE = 320
HEATMAP_SIZE = 80
CROP_MARGIN = 0.10

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
REPO_ROOT = ML_ROOT.parent
ARTIFACT_DIR = ML_ROOT / "artifacts" / "heatmap_cd_320"
DATA_DIR = ML_ROOT / "data"
CD_DATA_DIR = DATA_DIR / "heatmap_cd_320"
VIZ_DIR = ARTIFACT_DIR / "pipeline_viz"


def ellipse_to_obb_corners(
    cx: float, cy: float, rx: float, ry: float, rotation: float
) -> list[tuple[float, float]]:
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
    scale = target_size / crop_side
    return [((x - crop_left) * scale, (y - crop_top) * scale) for x, y in corners]


def compute_rectified_warp(
    img: np.ndarray,
    obb_corners: list[tuple[float, float]],
    *,
    output_size: int = RECTIFIED_SIZE,
    margin: float = CROP_MARGIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp OBB to axis-aligned rectified crop. Returns (warped, M)."""
    src_pts = np.array(obb_corners, dtype=np.float32)
    lo = margin * output_size
    hi = (1.0 - margin) * output_size
    dst_pts = np.array(
        [[hi, lo], [lo, lo], [lo, hi], [hi, hi]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        img, M, (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, M


def warp_point(x: float, y: float, M: np.ndarray) -> tuple[float, float]:
    src = np.array([x, y, 1.0], dtype=np.float64)
    dst = M @ src
    return float(dst[0] / dst[2]), float(dst[1] / dst[2])


def run_heatmap_cd_f32(interpreter, img_320_uint8: np.ndarray) -> np.ndarray:
    """Run f32 TFLite heatmap CD. Normalize to [-1,1] to match training."""
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    img_f32 = img_320_uint8.astype(np.float32) / 127.5 - 1.0  # [0,255] → [-1,1]
    interpreter.set_tensor(in_det[0]["index"], img_f32[None, ...])
    interpreter.invoke()
    hm = interpreter.get_tensor(out_det[0]["index"])[0, :, :, 0]
    return hm.astype(np.float64)


def run_heatmap_cd_int8(interpreter, img_320_uint8: np.ndarray) -> np.ndarray:
    """Run int8 TFLite heatmap CD. Raw uint8 [0,255] input. Dequantize output properly."""
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    interpreter.set_tensor(in_det[0]["index"], img_320_uint8[None, ...])
    interpreter.invoke()
    hm_q = interpreter.get_tensor(out_det[0]["index"])[0, :, :, 0]
    # Dequantize using output quantization params (scale, zero_point)
    q = out_det[0]["quantization"]
    if isinstance(q, tuple) and q[1] != 0:
        scale, zp = q
        return (hm_q.astype(np.float64) - zp) * scale
    else:
        return hm_q.astype(np.float64) / 255.0


def softargmax_2d(heatmap: np.ndarray) -> tuple[float, float]:
    hm = heatmap.astype(np.float64)
    hm_sum = hm.sum()
    if hm_sum <= 0:
        return (0.0, 0.0)
    hm_norm = hm / hm_sum
    ys, xs = np.meshgrid(np.arange(hm.shape[0]), np.arange(hm.shape[1]), indexing="ij")
    cx = float(np.sum(xs * hm_norm))
    cy = float(np.sum(ys * hm_norm))
    return (cx, cy)


def draw_crosshair(draw, x, y, color, size=5):
    """Draw crosshair on PIL ImageDraw."""
    draw.line((x - size, y, x + size, y), fill=color, width=2)
    draw.line((x, y - size, x, y + size), fill=color, width=2)


def main() -> None:
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf

    # Try f32 model first (higher accuracy), fall back to int8
    f32_path = ARTIFACT_DIR / "heatmap_cd_fp32.tflite"
    int8_path = ARTIFACT_DIR / "heatmap_cd_int8.tflite"

    def load_tflite(path: Path) -> tf.lite.Interpreter:
        interp = tf.lite.Interpreter(str(path))
        interp.allocate_tensors()
        return interp

    cd_f32 = load_tflite(f32_path) if f32_path.exists() else None
    cd_int8 = load_tflite(int8_path) if int8_path.exists() else None

    run_fn = run_heatmap_cd_f32 if cd_f32 is not None else run_heatmap_cd_int8
    cd_interp = cd_f32 if cd_f32 is not None else cd_int8
    model_tag = "f32" if cd_f32 is not None else "int8"
    print(f"  Using model: {model_tag}")
    print()

    all_samples = load_dataset()
    import random
    random.seed(SEED)
    random.shuffle(all_samples)
    val_split = all_samples[:int(len(all_samples) * VAL_FRAC)]

    scale_io = (RECTIFIED_SIZE - 1) / (HEATMAP_SIZE - 1)

    def validate_split(samples, name: str):
        errs_px80 = []
        for sample in samples:
            pil_img = Image.open(sample.image_path).convert("RGB")
            cropped, crop_left, crop_top, crop_side = dcmipp_crop_resize(pil_img, IMG_SIZE)

            obb = ellipse_to_obb_corners(
                sample.dial.cx, sample.dial.cy,
                sample.dial.rx, sample.dial.ry,
                sample.dial.rotation,
            )
            obb_320 = transform_coords_to_crop(obb, crop_left, crop_top, crop_side, IMG_SIZE)

            scale = IMG_SIZE / crop_side
            cx_in = (sample.center.x - crop_left) * scale
            cy_in = (sample.center.y - crop_top) * scale

            img_320 = np.array(cropped)
            rectified, M = compute_rectified_warp(img_320, obb_320)

            cx_gt, cy_gt = warp_point(cx_in, cy_in, M)
            cx_gt = max(0, min(RECTIFIED_SIZE - 1, cx_gt))
            cy_gt = max(0, min(RECTIFIED_SIZE - 1, cy_gt))

            hm = run_fn(cd_interp, rectified)
            cx_pred, cy_pred = softargmax_2d(hm)
            cx_pred_320 = cx_pred * scale_io
            cy_pred_320 = cy_pred * scale_io

            err = math.sqrt((cx_pred_320 - cx_gt)**2 + (cy_pred_320 - cy_gt)**2)
            errs_px80.append(err / scale_io)

        errs = np.array(errs_px80)
        print(f"  {name}: {len(errs)} samples")
        print(f"    Heatmap px (80×80): mean={errs.mean():.3f}  median={np.median(errs):.3f}")
        print(f"    Input px (320×320): mean={errs.mean() * scale_io:.2f}  median={np.median(errs) * scale_io:.2f}")
        return errs

    print("=" * 60)
    print(f"TEST 1: Heatmap CD on PXL val set ({model_tag}, GT OBB)")
    print("=" * 60)
    pxl_errs = validate_split(val_split, "PXL val")

    # Also test with int8 for comparison
    if cd_f32 is not None and cd_int8 is not None:
        print()
        print("  [Comparison with int8 model on same split]")
        int8_errs_px80 = []
        for sample in val_split:
            pil_img = Image.open(sample.image_path).convert("RGB")
            cropped, crop_left, crop_top, crop_side = dcmipp_crop_resize(pil_img, IMG_SIZE)
            obb = ellipse_to_obb_corners(
                sample.dial.cx, sample.dial.cy,
                sample.dial.rx, sample.dial.ry,
                sample.dial.rotation,
            )
            obb_320 = transform_coords_to_crop(obb, crop_left, crop_top, crop_side, IMG_SIZE)
            scale = IMG_SIZE / crop_side
            cx_in = (sample.center.x - crop_left) * scale
            cy_in = (sample.center.y - crop_top) * scale
            img_320 = np.array(cropped)
            rectified, M = compute_rectified_warp(img_320, obb_320)
            cx_gt, cy_gt = warp_point(cx_in, cy_in, M)
            hm = run_heatmap_cd_int8(cd_int8, rectified)
            cx_pred, cy_pred = softargmax_2d(hm)
            err = math.sqrt((cx_pred * scale_io - cx_gt)**2 + (cy_pred * scale_io - cy_gt)**2)
            int8_errs_px80.append(err / scale_io)

        i8 = np.array(int8_errs_px80)
        print(f"    int8: mean={i8.mean():.3f}  median={np.median(i8):.3f} heatmap px")

    print()
    print("=" * 60)
    print(f"TEST 2: Heatmap CD on board captures ({model_tag})")
    print("=" * 60)

    import csv
    repo_root = REPO_ROOT
    board_csv = DATA_DIR / "board_captures_labeled.csv"
    brd_errs_px80 = []
    with open(board_csv) as f:
        reader = csv.DictReader(f)
        first_iter = True
        for row in reader:
            bimg_path = repo_root / row["image_path"]
            if not bimg_path.exists():
                continue
            center_x = float(row["center_x"])
            center_y = float(row["center_y"])

            bimg = Image.open(bimg_path).convert("L")
            bimg_320 = bimg.resize((RECTIFIED_SIZE, RECTIFIED_SIZE), Image.BILINEAR)
            bimg_rgb = np.stack([np.array(bimg_320)] * 3, axis=-1).astype(np.uint8)

            s = RECTIFIED_SIZE / bimg.width
            cx_gt = center_x * s
            cy_gt = center_y * s

            hm = run_fn(cd_interp, bimg_rgb)
            cx_pred, cy_pred = softargmax_2d(hm)
            cx_pred_320 = cx_pred * scale_io
            cy_pred_320 = cy_pred * scale_io

            err = math.sqrt((cx_pred_320 - cx_gt)**2 + (cy_pred_320 - cy_gt)**2)
            brd_errs_px80.append(err / scale_io)
            if len(brd_errs_px80) >= 100:
                break

    if brd_errs_px80:
        brd = np.array(brd_errs_px80)
        print(f"  Board samples: {len(brd)}")
        print(f"    Heatmap px: mean={brd.mean():.3f}  median={np.median(brd):.3f}")
        print(f"    Input px:   mean={brd.mean() * scale_io:.2f}  median={np.median(brd) * scale_io:.2f}")
    else:
        print("  No board captures found — check path")

    print()
    print("=" * 60)
    print("TEST 3: Qualitative viz on PXL val samples")
    print("=" * 60)

    viz_pxl = VIZ_DIR / "pxl_val"
    viz_pxl.mkdir(parents=True, exist_ok=True)
    saved = 0
    for sample in val_split:
        if saved >= 12:
            break
        pil_img = Image.open(sample.image_path).convert("RGB")
        cropped, crop_left, crop_top, crop_side = dcmipp_crop_resize(pil_img, IMG_SIZE)

        obb = ellipse_to_obb_corners(
            sample.dial.cx, sample.dial.cy,
            sample.dial.rx, sample.dial.ry,
            sample.dial.rotation,
        )
        obb_320 = transform_coords_to_crop(obb, crop_left, crop_top, crop_side, IMG_SIZE)

        scale = IMG_SIZE / crop_side
        cx_in = (sample.center.x - crop_left) * scale
        cy_in = (sample.center.y - crop_top) * scale

        img_320 = np.array(cropped)
        rectified, M = compute_rectified_warp(img_320, obb_320)
        cx_gt, cy_gt = warp_point(cx_in, cy_in, M)

        hm = run_fn(cd_interp, rectified)
        cx_pred, cy_pred = softargmax_2d(hm)
        cx_pred_320 = cx_pred * scale_io
        cy_pred_320 = cy_pred * scale_io

        vis = Image.fromarray(rectified)
        draw = ImageDraw.Draw(vis)
        draw_crosshair(draw, cx_gt, cy_gt, (0, 255, 0), size=8)
        draw_crosshair(draw, cx_pred_320, cy_pred_320, (255, 0, 0), size=8)

        stem = sample.image_path.stem
        vis.save(str(viz_pxl / f"{stem}_pred.jpg"), quality=92)
        saved += 1

    print(f"  Saved {saved} to {viz_pxl}")

    print()
    print("=" * 60)
    print("TEST 4: Qualitative viz on board captures")
    print("=" * 60)

    viz_brd = VIZ_DIR / "board_captures"
    viz_brd.mkdir(parents=True, exist_ok=True)
    saved = 0
    with open(board_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if saved >= 12:
                break
            bimg_path = repo_root / row["image_path"]
            if not bimg_path.exists():
                continue
            center_x = float(row["center_x"])
            center_y = float(row["center_y"])

            bimg = Image.open(bimg_path).convert("L")
            bimg_320 = bimg.resize((RECTIFIED_SIZE, RECTIFIED_SIZE), Image.BILINEAR)
            bimg_rgb = np.stack([np.array(bimg_320)] * 3, axis=-1).astype(np.uint8)

            s = RECTIFIED_SIZE / bimg.width
            cx_gt = center_x * s
            cy_gt = center_y * s

            hm = run_fn(cd_interp, bimg_rgb)
            cx_pred, cy_pred = softargmax_2d(hm)
            cx_pred_320 = cx_pred * scale_io
            cy_pred_320 = cy_pred * scale_io

            vis = Image.fromarray(bimg_rgb)
            draw = ImageDraw.Draw(vis)
            draw_crosshair(draw, cx_gt, cy_gt, (0, 255, 0), size=8)
            draw_crosshair(draw, cx_pred_320, cy_pred_320, (255, 0, 0), size=8)

            stem = Path(row["image_path"]).stem
            vis.save(str(viz_brd / f"{stem}_pred.jpg"), quality=92)
            saved += 1

    print(f"  Saved {saved} to {viz_brd}")


if __name__ == "__main__":
    main()
