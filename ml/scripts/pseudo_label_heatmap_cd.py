"""
Pseudo-label unlabelled gauge images to augment heatmap CD training.

Sources:
  1. PNGs in captured_images/ NOT in ai_annotated_centers.csv (68 files, 224×224)
     → DCMIPP → OBB → warp → DS-CNN v2 → pseudo-label heatmap
  2. Rectified probe crops (39 files, 224×224 RGB, already rectified)
     → resize to 320×320 → DS-CNN v2 → pseudo-label heatmap
  3. YUV422 time-lapse captures (3404 files, 128×128)
     → convert to RGB → DS-CNN v2 → pseudo-label heatmap

Each pseudo-label is added to the train set. Low-confidence predictions are filtered.

Usage:
  python scripts/pseudo_label_heatmap_cd.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedded_gauge_reading_tinyml.heatmap_utils import softargmax_2d, make_gaussian_heatmap  # noqa: E402

SEED = 42
IMG_SIZE = 320
RECTIFIED_SIZE = 320
HEATMAP_SIZE = 80
SIGMA_PIXELS = 3.0
CROP_MARGIN = 0.10
OBB_CONF_THRESH = 0.3
CONFIDENCE_THRESH = 0.4  # minimum heatmap peak value to accept pseudo-label

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
DATA_DIR = ML_ROOT / "data"
CD_DATA_DIR = DATA_DIR / "heatmap_cd_320"
OBB_PT = ML_ROOT / "artifacts" / "yolo_obb_320" / "train" / "weights" / "best.pt"
CD_MODEL = ML_ROOT / "artifacts" / "heatmap_cd_ds_v2" / "final.keras"
CSV_PATH = DATA_DIR / "ai_annotated_centers.csv"


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


def obb_xywhr_to_corners_cw_tr(
    cx: float, cy: float, w: float, h: float, angle: float
) -> list[tuple[float, float]]:
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dw, dh = w / 2.0, h / 2.0
    tr = (cx + dw * cos_a - (-dh) * sin_a, cy + dw * sin_a + (-dh) * cos_a)
    tl = (cx + (-dw) * cos_a - (-dh) * sin_a, cy + (-dw) * sin_a + (-dh) * cos_a)
    bl = (cx + (-dw) * cos_a - dh * sin_a, cy + (-dw) * sin_a + dh * cos_a)
    br = (cx + dw * cos_a - dh * sin_a, cy + dw * sin_a + dh * cos_a)
    return [tr, tl, bl, br]


def compute_rectified_warp_and_center(
    img_320: np.ndarray,
    corners_cw_tr: list[tuple[float, float]],
) -> np.ndarray:
    src_pts = np.array(corners_cw_tr, dtype=np.float32)
    lo = CROP_MARGIN * RECTIFIED_SIZE
    hi = (1.0 - CROP_MARGIN) * RECTIFIED_SIZE
    dst_pts = np.array(
        [[hi, lo], [lo, lo], [lo, hi], [hi, hi]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        img_320, M, (RECTIFIED_SIZE, RECTIFIED_SIZE),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return warped


def yuv422_to_rgb(raw: bytes, w: int = 128, h: int = 128) -> np.ndarray | None:
    """Convert YUYV 4:2:2 raw bytes to RGB numpy array."""
    expected = w * h * 2
    if len(raw) < expected:
        return None
    n_pixels = w * h
    yuyv = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4)
    y = np.zeros(n_pixels, dtype=np.uint8)
    u = np.zeros(n_pixels, dtype=np.uint8)
    v = np.zeros(n_pixels, dtype=np.uint8)
    for i in range(0, n_pixels, 2):
        yi = i // 2
        y[i] = yuyv[yi, 0]
        u[i] = yuyv[yi, 1]
        y[i + 1] = yuyv[yi, 2]
        v[i + 1] = yuyv[yi, 3]
        u[i + 1] = u[i]
        v[i + 1] = v[i]
    y = y.reshape(h, w).astype(np.float32)
    u = u.reshape(h, w).astype(np.float32)
    v = v.reshape(h, w).astype(np.float32)
    r = np.clip(y + 1.402 * (v - 128), 0, 255).astype(np.uint8)
    g = np.clip(y - 0.344 * (u - 128) - 0.714 * (v - 128), 0, 255).astype(np.uint8)
    b = np.clip(y + 1.772 * (u - 128), 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def load_cd_model() -> tuple:
    """Load DS-CNN v2 model for pseudo-label prediction."""
    from tensorflow import keras
    model = keras.models.load_model(str(CD_MODEL))
    return model


def predict_center(
    model, img_320: np.ndarray
) -> tuple[float, float, float] | None:
    """Run model on 320×320 image, return (cx, cy, peak_value) or None."""
    inp = img_320.astype(np.float32) / 127.5 - 1.0
    inp = inp[None, ...]
    pred = model.predict(inp, verbose=0)[0, :, :, 0]
    cx, cy = softargmax_2d(pred)
    peak = float(pred.max())
    if peak < CONFIDENCE_THRESH:
        return None
    return cx, cy, peak


def process_png(
    model, obb_model, img_path: Path
) -> tuple[np.ndarray, float, float, float] | None:
    """Process PNG through full OBB → warp → center pipeline.

    Returns (warped_320, cx, cy, peak_confidence) or None.
    """
    pil_img = Image.open(img_path).convert("RGB")
    w, h = pil_img.size

    # DCMIPP crop+resize to 320×320
    cropped_img, _, _, _ = dcmipp_crop_resize(pil_img, IMG_SIZE)

    # Run YOLO OBB
    results = obb_model(cropped_img, imgsz=IMG_SIZE, verbose=False)
    if len(results) == 0 or results[0].obb is None or len(results[0].obb) == 0:
        return None

    obb = results[0].obb
    best_idx = int(obb.conf.argmax())
    if obb.conf[best_idx] < OBB_CONF_THRESH:
        return None

    xywhr = obb.xywhr[best_idx].cpu().numpy()
    corners = obb_xywhr_to_corners_cw_tr(*xywhr)

    # Warp to rectified
    img_320 = np.array(cropped_img)
    warped = compute_rectified_warp_and_center(img_320, corners)

    # Predict center
    result = predict_center(model, warped)
    if result is None:
        return None

    cx, cy, peak = result
    return warped, cx, cy, peak


def process_board_crop(
    model, img_path: Path
) -> tuple[np.ndarray, float, float, float] | None:
    """Process rectified board_crop → resize → center prediction.

    Returns (resized_320, cx, cy, peak_confidence) or None.
    """
    pil_img = Image.open(img_path).convert("RGB")
    # Already rectified, just resize to 320×320
    resized = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_320 = np.array(resized)

    result = predict_center(model, img_320)
    if result is None:
        return None

    cx, cy, peak = result
    return img_320, cx, cy, peak


def process_yuv422(
    model, file_path: Path
) -> tuple[np.ndarray, float, float, float] | None:
    """Convert YUV422 to RGB, resize to 320×320, predict center.

    YUV422 is 128×128 — resize to 320×320 first.
    """
    try:
        raw = file_path.read_bytes()
        rgb = yuv422_to_rgb(raw)
        if rgb is None:
            return None
        pil_img = Image.fromarray(rgb)
        resized = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_320 = np.array(resized)
    except Exception:
        return None

    result = predict_center(model, img_320)
    if result is None:
        return None

    cx, cy, peak = result
    return img_320, cx, cy, peak


def add_to_dataset(
    warped: np.ndarray,
    cx: float,
    cy: float,
    peak: float,
    stem: str,
    metadata: list[dict],
    existing_stems: set[str],
    aug_images_dir: Path,
    aug_heatmaps_dir: Path,
) -> bool:
    """Save a pseudo-labeled sample to the training set."""
    if stem in existing_stems:
        return False

    cx_norm = cx / float(HEATMAP_SIZE - 1)
    cy_norm = cy / float(HEATMAP_SIZE - 1)
    heatmap = make_gaussian_heatmap(HEATMAP_SIZE, HEATMAP_SIZE, cx_norm, cy_norm, SIGMA_PIXELS)

    Image.fromarray(warped).save(str(aug_images_dir / f"{stem}.jpg"), quality=95)
    np.save(str(aug_heatmaps_dir / f"{stem}.npy"), heatmap.astype(np.float32))

    existing_stems.add(stem)
    metadata.append({
        "stem": stem,
        "center_xy_rectified": [float(f"{cx:.2f}"), float(f"{cy:.2f}")],
        "center_xy_norm": [float(f"{cx_norm:.6f}"), float(f"{cy_norm:.6f}")],
        "pseudo_label": True,
        "confidence": float(f"{peak:.4f}"),
    })
    return True


def main() -> None:
    import tensorflow as tf

    # Load DS-CNN v2 model
    print(f"Loading DS-CNN v2 from {CD_MODEL}")
    model = load_cd_model()

    # Load existing stems
    existing_stems: set[str] = set()
    if (CD_DATA_DIR / "metadata.json").exists():
        meta = json.loads((CD_DATA_DIR / "metadata.json").read_text())
        for split in ("train", "val"):
            for s in meta["samples"][split]:
                existing_stems.add(s["stem"])
    print(f"Existing dataset has {len(existing_stems)} samples")

    # Prepare output dirs
    aug_images_dir = CD_DATA_DIR / "images" / "train"
    aug_heatmaps_dir = CD_DATA_DIR / "heatmaps" / "train"
    aug_images_dir.mkdir(parents=True, exist_ok=True)
    aug_heatmaps_dir.mkdir(parents=True, exist_ok=True)

    new_samples_meta: list[dict] = []

    # --- Source 1: PNGs not in CSV ---
    csv_entries: set[str] = set()
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                csv_entries.add(row["image_path"])

    # Load OBB model for PNG processing
    print("\n--- Source 1: PNGs not in CSV ---")
    from ultralytics import YOLO
    obb_model = YOLO(str(OBB_PT))
    print(f"Loaded OBB model from {OBB_PT}")

    png_files = sorted(DATA_DIR.glob("captured_images/*.png"))
    uncaptioned_pngs = [p for p in png_files if str(p.relative_to(DATA_DIR)) not in csv_entries]
    print(f"Found {len(uncaptioned_pngs)} PNG files not in CSV")

    for png_path in tqdm(uncaptioned_pngs, desc="PNGs"):
        # Skip files in subdirs (not direct children of captured_images/)
        if png_path.parent != DATA_DIR / "captured_images":
            continue
        result = process_png(model, obb_model, png_path)
        if result is None:
            continue
        warped, cx, cy, peak = result
        stem = f"pseudo_{png_path.stem}"
        add_to_dataset(warped, cx, cy, peak, stem, new_samples_meta,
                       existing_stems, aug_images_dir, aug_heatmaps_dir)

    # --- Source 2: Rectified probe board_crops ---
    print("\n--- Source 2: Rectified probe board_crops ---")
    probe_dir = DATA_DIR / "captured_images" / "_live_rectified_probe"
    board_crops = sorted(probe_dir.glob("*/board_crop.png"))
    print(f"Found {len(board_crops)} rectified board_crops")

    for crop_path in tqdm(board_crops, desc="board_crops"):
        result = process_board_crop(model, crop_path)
        if result is None:
            continue
        warped, cx, cy, peak = result
        stem = f"pseudo_board_{crop_path.parent.name}"
        add_to_dataset(warped, cx, cy, peak, stem, new_samples_meta,
                       existing_stems, aug_images_dir, aug_heatmaps_dir)

    # --- Source 3: YUV422 time-lapse captures (skipped — all <0.2 confidence) ---
    print("\n--- Source 3: YUV422 — skipped (max confidence 0.17, below threshold) ---")

    print(f"\nAdded {len(new_samples_meta)} new pseudo-labeled samples")

    # Update metadata.json
    if new_samples_meta:
        existing_meta = json.loads((CD_DATA_DIR / "metadata.json").read_text())
        existing_meta["samples"]["train"].extend(new_samples_meta)
        existing_meta["num_samples"] = len(existing_meta["samples"]["train"]) + len(existing_meta["samples"]["val"])
        existing_meta["train_count"] = len(existing_meta["samples"]["train"])
        (CD_DATA_DIR / "metadata.json").write_text(json.dumps(existing_meta, indent=2))
        print(f"Updated metadata: {existing_meta['train_count']} train, "
              f"{existing_meta['val_count']} val")
    else:
        print("No new samples added")


if __name__ == "__main__":
    main()
