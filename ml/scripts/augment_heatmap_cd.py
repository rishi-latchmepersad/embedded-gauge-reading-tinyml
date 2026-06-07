"""
Augment heatmap CD dataset with AI-annotated centers from board captures.

Pipeline per captured image:
  1. DCMIPP crop+resize to 320×320 (center-crop square, resize)
  2. Run YOLO OBB (PyTorch via ultralytics) to get gauge corners
  3. Perspective warp OBB → rectified 320×320 crop
  4. Transform AI-annotated center through same warp
  5. Generate 80×80 Gaussian heatmap target
  6. Append to existing training set

Usage:
  python scripts/augment_heatmap_cd.py  
  python scripts/train_heatmap_cd_320.py   (after augmentation)
"""

from __future__ import annotations

import csv
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
from embedded_gauge_reading_tinyml.heatmap_utils import make_gaussian_heatmap  # noqa: E402

SEED = 42
IMG_SIZE = 320
RECTIFIED_SIZE = 320
HEATMAP_SIZE = 80
SIGMA_PIXELS = 3.0
CROP_MARGIN = 0.10
OBB_CONF_THRESH = 0.3

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
DATA_DIR = ML_ROOT / "data"
CD_DATA_DIR = DATA_DIR / "heatmap_cd_320"
OBB_PT = ML_ROOT / "artifacts" / "yolo_obb_320" / "train" / "weights" / "best.pt"
CSV_PATH = DATA_DIR / "ai_annotated_centers.csv"

random.seed(SEED)
np.random.seed(SEED)


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
    """Convert OBB to 4 corners in clockwise order starting from top-right.

    Returns corners = [TR, TL, BL, BR] in clockwise order.
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dw, dh = w / 2.0, h / 2.0
    # Rotate relative offsets by angle
    tr = (cx + dw * cos_a - (-dh) * sin_a, cy + dw * sin_a + (-dh) * cos_a)
    tl = (cx + (-dw) * cos_a - (-dh) * sin_a, cy + (-dw) * sin_a + (-dh) * cos_a)
    bl = (cx + (-dw) * cos_a - dh * sin_a, cy + (-dw) * sin_a + dh * cos_a)
    br = (cx + dw * cos_a - dh * sin_a, cy + dw * sin_a + dh * cos_a)
    return [tr, tl, bl, br]


def compute_rectified_warp_and_center(
    img_320: np.ndarray,
    corners_cw_tr: list[tuple[float, float]],
    center_x: float,
    center_y: float,
) -> tuple[np.ndarray, float, float]:
    """Warp OBB corners (cw from TR) to rectified crop and transform center."""
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

    def warp_point(x, y, mat):
        s = np.array([x, y, 1.0], dtype=np.float64)
        d = mat @ s
        return float(d[0] / d[2]), float(d[1] / d[2])

    cx_out, cy_out = warp_point(center_x, center_y, M)
    cx_out = max(0.0, min(float(RECTIFIED_SIZE - 1), cx_out))
    cy_out = max(0.0, min(float(RECTIFIED_SIZE - 1), cy_out))
    return warped, cx_out, cy_out


def deduplicate_entries(
    entries: list[tuple[str, float, float]]
) -> list[tuple[str, float, float]]:
    """Deduplicate by image stem, keeping the .png version."""
    by_stem: dict[str, list[tuple[str, float, float]]] = {}
    for path, cx, cy in entries:
        p = Path(path)
        stem = p.stem
        for suffix in ["_preview", "_glare_p32c", "_uyvy", "_yuy2"]:
            stem = stem.replace(suffix, "")
        by_stem.setdefault(stem, []).append((path, cx, cy))

    deduped: list[tuple[str, float, float]] = []
    for stem, group in by_stem.items():
        best = sorted(group, key=lambda x: len(x[0]))[0]
        deduped.append(best)
    return deduped


def main() -> None:
    # Read CSV
    rows: list[tuple[str, float, float]] = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["image_path"], float(row["center_x"]), float(row["center_y"])))

    print(f"Read {len(rows)} annotated entries from CSV")

    # Deduplicate
    unique = deduplicate_entries(rows)
    print(f"After deduplication: {len(unique)} unique images")

    # Load YOLO OBB model (PyTorch via ultralytics)
    from ultralytics import YOLO
    print(f"Loading YOLO OBB from {OBB_PT}")
    obb_model = YOLO(str(OBB_PT))

    # Load existing metadata
    existing_stems: set[str] = set()
    if (CD_DATA_DIR / "metadata.json").exists():
        meta = json.loads((CD_DATA_DIR / "metadata.json").read_text())
        for split in ("train", "val"):
            for s in meta["samples"][split]:
                existing_stems.add(s["stem"])
        print(f"Existing dataset has {len(existing_stems)} samples")

    # Prepare output directories
    aug_images_dir = CD_DATA_DIR / "images" / "train"
    aug_heatmaps_dir = CD_DATA_DIR / "heatmaps" / "train"
    aug_images_dir.mkdir(parents=True, exist_ok=True)
    aug_heatmaps_dir.mkdir(parents=True, exist_ok=True)

    new_samples_meta: list[dict] = []
    failed, no_det = 0, 0

    # Process each unique image
    for img_rel_path, cx_orig, cy_orig in tqdm(unique, desc="augment"):
        img_abs = DATA_DIR / img_rel_path
        if not img_abs.exists():
            failed += 1
            continue

        try:
            pil_img = Image.open(img_abs).convert("RGB")
        except Exception:
            failed += 1
            continue

        # DCMIPP crop+resize to 320×320
        cropped_img, crop_left, crop_top, crop_side = dcmipp_crop_resize(pil_img, IMG_SIZE)

        # Transform center to 320×320 space
        scale = IMG_SIZE / crop_side
        cx_320 = (cx_orig - crop_left) * scale
        cy_320 = (cy_orig - crop_top) * scale
        cx_320 = max(0.0, min(float(IMG_SIZE - 1), cx_320))
        cy_320 = max(0.0, min(float(IMG_SIZE - 1), cy_320))

        # Run YOLO OBB inference (Runs on the 320×320 image directly)
        results = obb_model(cropped_img, imgsz=IMG_SIZE, verbose=False)
        if len(results) == 0 or results[0].obb is None or len(results[0].obb) == 0:
            no_det += 1
            continue

        # Get best detection
        obb = results[0].obb
        best_idx = int(obb.conf.argmax())
        if obb.conf[best_idx] < OBB_CONF_THRESH:
            no_det += 1
            continue

        # Get OBB via xywhr and generate corners in consistent CW-from-TR order
        xywhr = obb.xywhr[best_idx].cpu().numpy()  # (cx, cy, w, h, angle)
        corners = obb_xywhr_to_corners_cw_tr(*xywhr)

        # Warp to rectified
        img_320 = np.array(cropped_img)
        warped, cx_out, cy_out = compute_rectified_warp_and_center(
            img_320, corners, cx_320, cy_320
        )

        # Generate heatmap
        cx_norm = cx_out / float(RECTIFIED_SIZE - 1)
        cy_norm = cy_out / float(RECTIFIED_SIZE - 1)
        heatmap = make_gaussian_heatmap(HEATMAP_SIZE, HEATMAP_SIZE, cx_norm, cy_norm, SIGMA_PIXELS)

        # Unique stem
        stem = Path(img_rel_path).stem
        for suffix in ["_preview", "_glare_p32c", "_uyvy", "_yuy2", ".gray"]:
            stem = stem.replace(suffix, "")
        stem = f"aug_{stem}"

        if stem in existing_stems:
            continue
        existing_stems.add(stem)

        Image.fromarray(warped).save(str(aug_images_dir / f"{stem}.jpg"), quality=95)
        np.save(str(aug_heatmaps_dir / f"{stem}.npy"), heatmap.astype(np.float32))

        new_samples_meta.append({
            "stem": stem,
            "center_xy_rectified": [float(f"{cx_out:.2f}"), float(f"{cy_out:.2f}")],
            "center_xy_norm": [float(f"{cx_norm:.6f}"), float(f"{cy_norm:.6f}")],
        })

    print(f"\nProcessed: {len(new_samples_meta)} new, {failed} missing, {no_det} no det")

    # Update metadata.json
    if new_samples_meta:
        existing_meta = json.loads((CD_DATA_DIR / "metadata.json").read_text())
        existing_meta["samples"]["train"].extend(new_samples_meta)
        existing_meta["num_samples"] = len(existing_meta["samples"]["train"]) + len(existing_meta["samples"]["val"])
        existing_meta["train_count"] = len(existing_meta["samples"]["train"])
        (CD_DATA_DIR / "metadata.json").write_text(json.dumps(existing_meta, indent=2))
        print(f"Updated metadata: {existing_meta['train_count']} train, {existing_meta['val_count']} val")
    else:
        print("No new samples added")


if __name__ == "__main__":
    main()
