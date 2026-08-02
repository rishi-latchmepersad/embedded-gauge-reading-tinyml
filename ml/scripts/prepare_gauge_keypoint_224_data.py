"""Prepare ellipse-conditioned gauge crops and heatmap targets.

Pipeline:
1. For each CVAT image, extract the gauge-face ellipse and center/tip keypoints.
2. Compute a square crop around the ellipse (configurable scale), resize to 224x224.
3. Rasterize Gaussian heatmaps for center and tip keypoints.

Supported annotation formats (all three CVAT label conventions in the repo):
  - GaugeFace/Center/Tip as <ellipse> (train_2, val_2, test_2)
  - GaugeFace as <ellipse>, Center/Tip as <box> (train_1, val_1, test_1)
  - temp_dial as <ellipse>, temp_center/temp_tip as <points> (test_3)

Output layout (ml/data/gauge_keypoint_224/):
    train/images/000000.jpg   (224x224 grayscale crop)
    train/center.npy          (N, HMAP, HMAP) float32 Gaussian heatmaps
    train/tip.npy             (N, HMAP, HMAP)
    val/images/...
    val/center.npy, val/tip.npy
    test/images/...
    test/center.npy, test/tip.npy
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
LABELLED = ROOT / "data" / "labelled"
OUTPUT = ROOT / "data" / "gauge_keypoint_224"
INPUT_SIZE = 224
HEATMAP_SIZE = 56
CROP_SCALE = 1.35  # expansion factor around the ellipse for the crop
GAUSSIAN_SIGMA = 2.0  # sigma in heatmap pixels


SPLIT_FILES: dict[str, list[str]] = {
    "train": [
        "train_1.zip",
        "train_2.zip",
        # why: use every trainable board archive (repaired 2026-07-31) but
        # never board_captures_2, which is an exact duplicate of test_3.
        "initial_temp_gauge/board_captures_1.zip",
        "initial_temp_gauge/board_captures_3.zip",
        "initial_temp_gauge/board_captures_4.zip",
        "initial_temp_gauge/gauge_1_batch_1.zip",
        "initial_temp_gauge/gauge_1_batch_2.zip",
        "initial_temp_gauge/gauge_1_batch_3.zip",
        "initial_temp_gauge/gauge_1_batch_4.zip",
        "initial_temp_gauge/gauge_1_batch_5.zip",
        "initial_temp_gauge/gauge_1_batch_6.zip",
        "initial_temp_gauge/gauge_1_batch_7.zip",
        "initial_temp_gauge/gauge_1_batch_8.zip",
    ],
    "val": ["val_1.zip", "val_2.zip"],
    "test": ["test_1.zip", "test_2.zip", "test_3.zip"],
}


def _box_midpoint(el: ET.Element) -> tuple[float, float]:
    """Return the (x, y) midpoint of a CVAT box element in image pixels."""
    xtl, ytl = float(el.get("xtl")), float(el.get("ytl"))
    xbr, ybr = float(el.get("xbr")), float(el.get("ybr"))
    return (xtl + xbr) / 2.0, (ytl + ybr) / 2.0


def _points_xy(el: ET.Element) -> tuple[float, float]:
    """Return the (x, y) coordinates of a CVAT points element."""
    pts = el.get("points", "").split(",")
    return float(pts[0]), float(pts[1])


# Label name variants for gauge-face ellipse, center keypoint, and tip keypoint.
# Each set covers all CVAT labelling conventions in the repo.
_ELLIPSE_LABELS = {"GaugeFace", "temp_dial"}
_CENTER_LABELS = {"Center", "temp_center"}
_TIP_LABELS = {"Tip", "temp_tip"}


def _gaussian_heatmap(cx: float, cy: float) -> np.ndarray:
    """Create a 2D Gaussian heatmap with peak at the given pixel coordinate."""
    yy, xx = np.mgrid[0:HEATMAP_SIZE, 0:HEATMAP_SIZE].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    return np.exp(-(dx**2 + dy**2) / (2.0 * GAUSSIAN_SIGMA**2))


def _iter_records(zip_paths: list[Path]) -> list[dict]:
    """Walk every zip and return records with ellipse and Center/Tip points.

    Handles three annotation conventions:
      1. GaugeFace <ellipse> + Center/Tip <box>      (train_1, val_1, test_1)
      2. GaugeFace <ellipse> + Center/Tip <ellipse>  (train_2, val_2, test_2)
      3. temp_dial <ellipse> + temp_center/temp_tip <points> (test_3)

    Returns records like:
        {zip, name, width, height, cx, cy, rx, ry, center_x, center_y, tip_x, tip_y}
    """
    records: list[dict] = []
    for zp in zip_paths:
        if not zp.exists():
            continue
        with zipfile.ZipFile(zp) as z:
            try:
                xml_bytes = z.read("annotations.xml")
            except KeyError:
                continue
            root = ET.fromstring(xml_bytes)
            for img_node in root.findall("image"):
                width = int(img_node.get("width", 640))
                height = int(img_node.get("height", 640))
                name = img_node.get("name")
                ellipse = None
                center_pt = None
                tip_pt = None
                for el in img_node:
                    label = el.get("label", "")
                    # Gauge-face ellipse (GaugeFace or temp_dial)
                    if label in _ELLIPSE_LABELS and el.tag == "ellipse":
                        ellipse = (
                            float(el.get("cx")),
                            float(el.get("cy")),
                            float(el.get("rx")),
                            float(el.get("ry")),
                        )
                    # Center keypoint — box, ellipse, or points
                    elif label in _CENTER_LABELS:
                        if el.tag == "box":
                            center_pt = _box_midpoint(el)
                        elif el.tag == "ellipse":
                            center_pt = (float(el.get("cx")), float(el.get("cy")))
                        elif el.tag == "points":
                            center_pt = _points_xy(el)
                    # Tip keypoint — box, ellipse, or points
                    elif label in _TIP_LABELS:
                        if el.tag == "box":
                            tip_pt = _box_midpoint(el)
                        elif el.tag == "ellipse":
                            tip_pt = (float(el.get("cx")), float(el.get("cy")))
                        elif el.tag == "points":
                            tip_pt = _points_xy(el)

                # Only include samples that have all three annotations.
                if ellipse is None or center_pt is None or tip_pt is None:
                    continue
                records.append({
                    "zip": str(zp),
                    "name": name,
                    "width": width,
                    "height": height,
                    "cx": ellipse[0],
                    "cy": ellipse[1],
                    "rx": ellipse[2],
                    "ry": ellipse[3],
                    "center_x": center_pt[0],
                    "center_y": center_pt[1],
                    "tip_x": tip_pt[0],
                    "tip_y": tip_pt[1],
                })
    return records


def _crop_and_resize(
    img: Image.Image,
    cx: float, cy: float, rx: float, ry: float,
    width: int, height: int,
) -> tuple[Image.Image, float, float, float]:
    """Crop a square region around the ellipse and resize to INPUT_SIZE.

    Returns (cropped_224x224_image, left, top, side) — the parameters that
    map crop-local pixel coords back to source image coords.
    """
    side = max(2.0 * rx, 2.0 * ry) * CROP_SCALE
    left = cx - side / 2.0
    top = cy - side / 2.0
    # Clamp to image bounds.
    left_clamped = max(0.0, left)
    top_clamped = max(0.0, top)
    right_clamped = min(float(width), left + side)
    bottom_clamped = min(float(height), top + side)
    # Use clamped left/top for the actual crop origin.
    crop_left = max(0, int(left_clamped))
    crop_top = max(0, int(top_clamped))
    crop_right = min(width, int(right_clamped))
    crop_bottom = min(height, int(bottom_clamped))
    cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    # Resize keeping aspect ratio by using the original crop side as reference.
    resized = cropped.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    # We need the effective crop parameters for coordinate mapping.
    # The keypoint coordinates are in the CROPPED image, then scaled to INPUT_SIZE.
    # crop_left/top and crop_side are the actual crop boundaries.
    actual_side = max(crop_right - crop_left, crop_bottom - crop_top)
    return resized, crop_left, crop_top, actual_side


def _stage_split(split: str, records: list[dict]) -> None:
    """Process all records for a split: crop images, generate heatmaps, save."""
    img_dir = OUTPUT / split / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    all_center = []
    all_tip = []

    # Group by zip for efficient access.
    by_zip: dict[str, list[dict]] = {}
    for rec in records:
        by_zip.setdefault(rec["zip"], []).append(rec)

    counter = 0
    for zip_path, recs in tqdm(by_zip.items(), desc=f"stage {split}"):
        with zipfile.ZipFile(zip_path) as z:
            for rec in recs:
                # Find the image entry.
                basename = Path(rec["name"]).name
                matches = [n for n in z.namelist() if Path(n).name == basename]
                if not matches:
                    continue
                raw = z.read(matches[0])
                img = Image.open(__import__("io").BytesIO(raw)).convert("L")

                # Crop and resize using the ellipse.
                cropped, crop_left, crop_top, crop_side = _crop_and_resize(
                    img, rec["cx"], rec["cy"], rec["rx"], rec["ry"],
                    rec["width"], rec["height"],
                )
                out_name = f"{counter:06d}.jpg"
                cropped.save(img_dir / out_name, "JPEG", quality=95)
                counter += 1

                # Map keypoint coordinates into the 224x224 crop space.
                # The keypoint is in source image pixels; after crop+resize,
                # the normalized position in the crop [0,1] maps directly to
                # the 224x224 image.
                norm_cx = (rec["center_x"] - crop_left) / crop_side
                norm_cy = (rec["center_y"] - crop_top) / crop_side
                norm_tx = (rec["tip_x"] - crop_left) / crop_side
                norm_ty = (rec["tip_y"] - crop_top) / crop_side

                # Build 56x56 heatmap from normalized coordinate [0,1].
                # Heatmap coordinates: hm_px = norm * (HEATMAP_SIZE - 1)
                hm_center_x = norm_cx * (HEATMAP_SIZE - 1)
                hm_center_y = norm_cy * (HEATMAP_SIZE - 1)
                hm_tip_x = norm_tx * (HEATMAP_SIZE - 1)
                hm_tip_y = norm_ty * (HEATMAP_SIZE - 1)

                all_center.append(_gaussian_heatmap(hm_center_x, hm_center_y))
                all_tip.append(_gaussian_heatmap(hm_tip_x, hm_tip_y))

    print(f"  {split}: wrote {counter} samples")
    np.save(OUTPUT / split / "center.npy", np.array(all_center, dtype=np.float32))
    np.save(OUTPUT / split / "tip.npy", np.array(all_tip, dtype=np.float32))


def main() -> None:
    import argparse

    global HEATMAP_SIZE, CROP_SCALE, GAUSSIAN_SIGMA, OUTPUT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heatmap-size", type=int, default=56,
                        help="Output heatmap resolution (default: 56)")
    parser.add_argument("--crop-scale", type=float, default=1.35,
                        help="Ellipse crop expansion factor (default: 1.35)")
    parser.add_argument("--gaussian-sigma", type=float, default=2.0,
                        help="Gaussian sigma in heatmap pixels (default: 2.0)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT,
                        help="Output directory for staged data")
    args = parser.parse_args()

    HEATMAP_SIZE = args.heatmap_size
    CROP_SCALE = args.crop_scale
    GAUSSIAN_SIGMA = args.gaussian_sigma
    OUTPUT = args.output_dir

    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_records = {}
    for split, zip_names in SPLIT_FILES.items():
        zip_paths = [LABELLED / n for n in zip_names]
        print(f"Reading {split} from {[p.name for p in zip_paths]}")
        records = _iter_records(zip_paths)
        print(f"  {len(records)} samples with all 3 annotations (ellipse + center + tip)")
        all_records[split] = records
        _stage_split(split, records)

    total = sum(len(v) for v in all_records.values())
    print(f"\nDone. Staged {total} keypoint samples at {OUTPUT}")
    print(f"  heatmap_size={HEATMAP_SIZE}, crop_scale={CROP_SCALE}, sigma={GAUSSIAN_SIGMA}")


if __name__ == "__main__":
    sys.exit(main())
