"""
Prepare axis-aligned center-detection dataset matching firmware behavior.

The firmware (AppCenterDetector_FillInputFromCrop) produces a 223×176 axis-aligned
crop → letterbox to 320×320. The old pipeline used perspective-rectified crops,
so the model sees a domain mismatch at inference. This script fixes that.

Data sources:
  - PXL images:  ml/data/raw/PXL_*.jpg + CVAT annotations in ml/data/labelled/*.zip
  - Board caps:  ml/data/captured_images/*.png + ml/data/ai_annotated_board_captures.csv

Output: ml/data/heatmap_cd_320_ax/ with JPEGs + metadata.json matching the
        format expected by train_heatmap_cd_ds_v4.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedded_gauge_reading_tinyml.heatmap_utils import make_gaussian_heatmap

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

IMG_SIZE = 320
HEATMAP_SIZE = 160
SIGMA_PIXELS = 6.0

TRAIN_CROP_W = 222
TRAIN_CROP_H = 175

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_ROOT / "raw"
LABELLED_DIR = DATA_ROOT / "labelled"
CAPTURED_DIR = DATA_ROOT / "captured_images"
ANNOT_CSV = DATA_ROOT / "ai_annotated_board_captures.csv"
OLD_META_PATH = DATA_ROOT / "heatmap_cd_320" / "metadata.json"
OUTPUT_DIR = DATA_ROOT / "heatmap_cd_320_ax"


def dcmipp_crop_resize(
    img: np.ndarray, output_size: int = IMG_SIZE
) -> np.ndarray:
    h, w = img.shape[:2]
    size = min(h, w)
    x = (w - size) // 2
    y = (h - size) // 2
    cropped = img[y : y + size, x : x + size]
    interp = cv2.INTER_AREA if size > output_size else cv2.INTER_LINEAR
    return cv2.resize(cropped, (output_size, output_size), interpolation=interp)


def transform_center_dcmipp(
    cx: float, cy: float, src_w: int, src_h: int, output_size: int = IMG_SIZE
) -> tuple[float, float]:
    size = min(src_h, src_w)
    crop_x = (src_w - size) // 2
    crop_y = (src_h - size) // 2
    cx_crop = cx - crop_x
    cy_crop = cy - crop_y
    scale = output_size / size
    return cx_crop * scale, cy_crop * scale


def axis_aligned_letterbox_crop(
    img_320: np.ndarray, center_x: float, center_y: float
) -> tuple[np.ndarray, float, float]:
    """Matches firmware AppCenterDetector_FillInputFromCrop exactly.

    - crop to TRAIN_CROP_W x TRAIN_CROP_H centered on (center_x, center_y)
    - letterbox-resize to 320x320 with aspect-preserving scale + symmetric pad
    - nearest-neighbour sampling matching firmware's floorf(src + 0.5f)
    - returns (letterbox, cx_norm, cy_norm) where cx_norm/cy_norm are the
      center position in the 320x320 output space, normalised to [0,1].
    """
    crop_x = round(center_x - TRAIN_CROP_W / 2)
    crop_y = round(center_y - TRAIN_CROP_H / 2)
    max_x = IMG_SIZE - TRAIN_CROP_W
    max_y = IMG_SIZE - TRAIN_CROP_H
    crop_x = max(0, min(max_x, crop_x))
    crop_y = max(0, min(max_y, crop_y))

    # --- Match AppCenterDetector_ComputeResizeWithPadGeometry ---
    output_w_f = float(IMG_SIZE)
    output_h_f = float(IMG_SIZE)
    crop_w_f = float(TRAIN_CROP_W)
    crop_h_f = float(TRAIN_CROP_H)
    scale = min(output_w_f / crop_w_f, output_h_f / crop_h_f)
    resized_w = crop_w_f * scale
    resized_h = crop_h_f * scale
    pad_x = 0.5 * (output_w_f - resized_w)
    pad_y = 0.5 * (output_h_f - resized_h)

    # --- Match AppCenterDetector_FillInputFromCrop sampling ---
    letterbox = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    row_lower = pad_y
    row_upper = pad_y + crop_h_f * scale
    col_lower = pad_x
    col_upper = pad_x + crop_w_f * scale

    for row in range(IMG_SIZE):
        out_row_f = float(row)
        if out_row_f < row_lower or out_row_f >= row_upper:
            continue
        src_row_f = (out_row_f - pad_y) / scale + float(crop_y)
        src_row = min(max(int(np.floor(src_row_f + 0.5)), 0), IMG_SIZE - 1)

        for col in range(IMG_SIZE):
            out_col_f = float(col)
            if out_col_f < col_lower or out_col_f >= col_upper:
                continue
            src_col_f = (out_col_f - pad_x) / scale + float(crop_x)
            src_col = min(max(int(np.floor(src_col_f + 0.5)), 0), IMG_SIZE - 1)

            letterbox[row, col] = img_320[src_row, src_col]

    # --- Map center to output coordinates ---
    crop_cx = center_x - crop_x
    crop_cy = center_y - crop_y
    out_cx = crop_cx * scale + pad_x
    out_cy = crop_cy * scale + pad_y

    cx_norm = out_cx / float(IMG_SIZE - 1)
    cy_norm = out_cy / float(IMG_SIZE - 1)
    cx_norm = max(0.0, min(1.0, cx_norm))
    cy_norm = max(0.0, min(1.0, cy_norm))

    return letterbox, cx_norm, cy_norm


def parse_cvat_zips(
    labelled_dir: Path,
) -> dict[str, dict]:
    """
    Return dict mapping stem -> {cx, cy, src_w, src_h}
    from all CVAT annotation zips.
    """
    records: dict[str, dict] = {}
    for zip_path in sorted(labelled_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            tree = ET.parse(zf.open("annotations.xml"))
            root = tree.getroot()
            for img_el in root.findall(".//image"):
                name = img_el.get("name")
                stem = Path(name).stem
                w = int(img_el.get("width", "0"))
                h = int(img_el.get("height", "0"))
                cx = None
                cy = None
                for child in img_el:
                    label = child.get("label", "")
                    if label == "temp_center" and child.tag == "points":
                        pts = child.get("points", "")
                        parts = pts.split(",")
                        if len(parts) == 2:
                            cx = float(parts[0])
                            cy = float(parts[1])
                if cx is None:
                    continue
                records[stem] = {"cx": cx, "cy": cy, "src_w": w, "src_h": h}
    return records


def load_csv_centers(csv_path: Path) -> dict[str, tuple[float, float]]:
    records: dict[str, tuple[float, float]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["image_path"]
            stem = Path(path).stem
            cx = float(row["center_x"])
            cy = float(row["center_y"])
            records[stem] = (cx, cy)
    return records


def process_pxl_samples(
    cvat_records: dict[str, dict],
    old_split: dict[str, str],
) -> list[dict]:
    samples: list[dict] = []
    for stem, ann in cvat_records.items():
        src_path = RAW_DIR / f"{stem}.jpg"
        if not src_path.exists():
            print(f"  [SKIP] PXL image not found: {src_path}")
            continue
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"  [SKIP] Cannot read: {src_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_320 = dcmipp_crop_resize(img_rgb)

        cx_src, cy_src = ann["cx"], ann["cy"]
        src_w, src_h = ann["src_w"], ann["src_h"]
        cx_320, cy_320 = transform_center_dcmipp(cx_src, cy_src, src_w, src_h)

        letterbox, cx_norm, cy_norm = axis_aligned_letterbox_crop(img_320, cx_320, cy_320)
        letterbox_bgr = cv2.cvtColor(letterbox, cv2.COLOR_RGB2BGR)

        split = old_split.get(stem, "train")
        samples.append({
            "stem": stem,
            "image": letterbox_bgr,
            "center_xy_norm": [round(cx_norm, 6), round(cy_norm, 6)],
            "center_xy_320": [round(cx_320, 2), round(cy_320, 2)],
            "split": split,
        })
    return samples


def process_board_samples(
    csv_centers: dict[str, tuple[float, float]],
    old_split: dict[str, str],
) -> list[dict]:
    samples: list[dict] = []
    csv_stems = set(csv_centers.keys())
    for stem, (cx_csv, cy_csv) in csv_centers.items():
        stem_with_aug = f"aug_{stem}"
        split = old_split.get(stem_with_aug)
        if split is None:
            continue

        for ext in [".png", ".jpg", ".jpeg"]:
            src_path = CAPTURED_DIR / f"{stem}{ext}"
            if src_path.exists():
                break
        else:
            print(f"  [SKIP] Board image not found for stem={stem}")
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            print(f"  [SKIP] Cannot read: {src_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        src_h, src_w = img.shape[:2]
        img_320 = dcmipp_crop_resize(img_rgb)
        cx_320, cy_320 = transform_center_dcmipp(cx_csv, cy_csv, src_w, src_h)

        letterbox, cx_norm, cy_norm = axis_aligned_letterbox_crop(img_320, cx_320, cy_320)
        letterbox_bgr = cv2.cvtColor(letterbox, cv2.COLOR_RGB2BGR)

        samples.append({
            "stem": stem_with_aug,
            "image": letterbox_bgr,
            "center_xy_norm": [round(cx_norm, 6), round(cy_norm, 6)],
            "center_xy_320": [round(cx_320, 2), round(cy_320, 2)],
            "split": split,
        })
    return samples


def load_old_split(meta_path: Path) -> dict[str, str]:
    with open(meta_path) as f:
        meta = json.load(f)
    split_map: dict[str, str] = {}
    for s in meta["samples"].get("train", []):
        split_map[s["stem"]] = "train"
    for s in meta["samples"].get("val", []):
        split_map[s["stem"]] = "val"
    return split_map


def main():
    parser = argparse.ArgumentParser(
        description="Prepare axis-aligned center-detection dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    os.makedirs(output_dir / "images" / "train", exist_ok=True)
    os.makedirs(output_dir / "images" / "val", exist_ok=True)

    print("Loading old split assignments...")
    old_split = load_old_split(OLD_META_PATH)
    print(f"  Found {len(old_split)} stems in old metadata")

    print("\nParsing CVAT annotations...")
    cvat_records = parse_cvat_zips(LABELLED_DIR)
    print(f"  Found {len(cvat_records)} PXL annotations")

    print("\nLoading CSV centers for board captures...")
    csv_centers = load_csv_centers(ANNOT_CSV)
    print(f"  Found {len(csv_centers)} CSV entries")

    print("\n--- Processing PXL samples ---")
    pxl_samples = process_pxl_samples(cvat_records, old_split)
    print(f"  Processed {len(pxl_samples)} PXL samples")

    print("\n--- Processing board capture samples ---")
    board_samples = process_board_samples(csv_centers, old_split)
    print(f"  Processed {len(board_samples)} board capture samples")

    all_samples = pxl_samples + board_samples
    print(f"\nTotal: {len(all_samples)} samples")

    print("\n--- Saving JPEGs and building metadata ---")
    train_entries: list[dict] = []
    val_entries: list[dict] = []
    for s in all_samples:
        split_dir = "train" if s["split"] == "train" else "val"
        jpg_path = output_dir / "images" / split_dir / f"{s['stem']}.jpg"
        cv2.imwrite(str(jpg_path), s["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])

        entry = {
            "stem": s["stem"],
            "center_xy_norm": s["center_xy_norm"],
        }
        if s["split"] == "train":
            train_entries.append(entry)
        else:
            val_entries.append(entry)

    metadata = {
        "input_size": IMG_SIZE,
        "heatmap_size": HEATMAP_SIZE,
        "sigma_pixels": SIGMA_PIXELS,
        "num_samples": len(all_samples),
        "samples": {
            "train": train_entries,
            "val": val_entries,
        },
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {meta_path}")
    print(f"  Train: {len(train_entries)}, Val: {len(val_entries)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
