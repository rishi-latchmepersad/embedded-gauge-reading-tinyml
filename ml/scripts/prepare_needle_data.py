#!/usr/bin/env python3
"""Prepare training data for the two-model gauge reading pipeline.

Extracts images and labels from CVAT zip files into numpy arrays for
fast training. Creates three data products:

1. Ellipse data: 224x224 grayscale images + normalized (cx, cy, rx, ry) labels
2. Needle data: 224x224 grayscale crops of gauge face + 56x56 center/tip heatmaps

The zips contain three different annotation schemas:
- test_1/train_1/val_1: "GaugeFace" ellipse (cx, cy, rx, ry in pixels)
- test_2: "GaugeFace" + "Center"/"Tip" ellipses (needle keypoints as ellipse centers)
- test_3: "temp_dial" ellipse + "temp_center"/"temp_tip" points

This script normalizes all formats into a unified representation.
"""

from __future__ import annotations

import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LABELLED = ROOT / "data" / "labelled"
OUTPUT = ROOT / "data" / "needle_pipeline"

# Heatmap configuration matching the model output
HEATMAP_SIZE = 56
HEATMAP_SIGMA = 2.5
IMAGE_SIZE = 224
SEED = 42

# Split assignment: which zips go into which split
# Why: train_1 is the large web dataset, train_2 adds board captures,
# test_1/test_2/test_3 are the held-out evaluation sets.
# The initial_temp_gauge batches and board_captures zips contain
# needle-labeled data (temp_dial + temp_center + temp_tip).
SPLIT_FILES: dict[str, list[str]] = {
    "train": ["train_1.zip", "train_2.zip"],
    "val": ["val_1.zip", "val_2.zip"],
    "test": ["test_1.zip", "test_2.zip", "test_3.zip"],
}

# Additional needle-labeled data from initial_temp_gauge/ directory.
# These have temp_dial + temp_center + temp_tip labels.
# We split them into train/val to boost needle training data.
# Why: only 22 train + 11 val needle-labeled images from the main zips.
NEEDLE_TRAIN_FILES: list[str] = [
    "initial_temp_gauge/gauge_1_batch_1.zip",
    "initial_temp_gauge/gauge_1_batch_2.zip",
    "initial_temp_gauge/gauge_1_batch_3.zip",
    "initial_temp_gauge/gauge_1_batch_4.zip",
    "initial_temp_gauge/gauge_1_batch_5.zip",
    "initial_temp_gauge/gauge_1_batch_6.zip",
    "initial_temp_gauge/board_captures_1.zip",
]
NEEDLE_VAL_FILES: list[str] = [
    "initial_temp_gauge/gauge_1_batch_7.zip",
    "initial_temp_gauge/gauge_1_batch_8.zip",
    "initial_temp_gauge/board_captures_2.zip",
    "initial_temp_gauge/board_captures_3.zip",
    "initial_temp_gauge/board_captures_4.zip",
]


def _make_gaussian_heatmap(
    h: int, w: int, x_norm: float, y_norm: float, sigma: float
) -> np.ndarray:
    """Generate a 2D Gaussian heatmap at normalized (x, y) position."""
    cx = x_norm * (w - 1)
    cy = y_norm * (h - 1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))


def _parse_cvat_zip(zip_path: Path) -> list[dict]:
    """Parse a CVAT zip and return a list of image records with all labels.

    Each record contains:
      - name: image filename (relative to images/ in the zip)
      - width, height: original image dimensions
      - gauge_face: dict with cx, cy, rx, ry (normalized to [0,1])
      - needle_center: dict with x, y (normalized) or None
      - needle_tip: dict with x, y (normalized) or None
      - zip_path: path to the source zip
    """
    records: list[dict] = []

    with zipfile.ZipFile(zip_path) as z:
        try:
            xml_bytes = z.read("annotations.xml")
        except KeyError:
            print(f"  WARNING: {zip_path.name} has no annotations.xml")
            return []

        root = ET.fromstring(xml_bytes)

        for img_node in root.findall("image"):
            name = img_node.get("name")
            width = float(img_node.get("width", 640))
            height = float(img_node.get("height", 640))

            rec: dict = {
                "name": name,
                "width": width,
                "height": height,
                "gauge_face": None,
                "needle_center": None,
                "needle_tip": None,
                "zip_path": str(zip_path),
            }

            # Parse bounding boxes (train_1/test_1 use <box> with Center/Tip labels).
            # The box center is the keypoint location.
            for box in img_node.findall("box"):
                label = box.get("label")
                xtl = float(box.get("xtl", 0))
                ytl = float(box.get("ytl", 0))
                xbr = float(box.get("xbr", 0))
                ybr = float(box.get("ybr", 0))
                # Box center = keypoint location
                cx_box = (xtl + xbr) / 2.0
                cy_box = (ytl + ybr) / 2.0

                if label == "Center":
                    rec["needle_center"] = {
                        "x": cx_box / width,
                        "y": cy_box / height,
                    }
                elif label == "Tip":
                    rec["needle_tip"] = {
                        "x": cx_box / width,
                        "y": cy_box / height,
                    }

            # Parse ellipses (temp_dial / GaugeFace)
            for el in img_node.findall("ellipse"):
                label = el.get("label")
                cx = float(el.get("cx", 0))
                cy = float(el.get("cy", 0))
                rx = float(el.get("rx", 0))
                ry = float(el.get("ry", 0))

                if label in ("GaugeFace", "temp_dial"):
                    # Both label names refer to the gauge face ellipse.
                    # "GaugeFace" is used in train_1/val_1/test_1/test_2.
                    # "temp_dial" is used in test_3 (board captures).
                    rec["gauge_face"] = {
                        "cx": cx / width,
                        "cy": cy / height,
                        "rx": rx / width,
                        "ry": ry / height,
                    }
                elif label == "Center":
                    # test_2: needle center as small ellipse
                    rec["needle_center"] = {
                        "x": cx / width,
                        "y": cy / height,
                    }
                elif label == "Tip":
                    # test_2: needle tip as small ellipse
                    rec["needle_tip"] = {
                        "x": cx / width,
                        "y": cy / height,
                    }

            # Parse points (test_3 format)
            for pt in img_node.findall("points"):
                label = pt.get("label")
                coords = pt.get("points", "0,0").split(",")
                px = float(coords[0])
                py = float(coords[1])

                if label == "temp_center":
                    rec["needle_center"] = {
                        "x": px / width,
                        "y": py / height,
                    }
                elif label == "temp_tip":
                    rec["needle_tip"] = {
                        "x": px / width,
                        "y": py / height,
                    }

            # Only include if we have a gauge face ellipse
            if rec["gauge_face"] is not None:
                records.append(rec)

    return records


def _extract_image(z: zipfile.ZipFile, name: str) -> np.ndarray | None:
    """Extract an image from a zip file and return as grayscale numpy array.

    Falls back to searching disk directories when the image is not inside the
    zip (board_captures zips only contain annotations.xml, not images).
    """
    basename = Path(name).name

    # First try: find inside the zip
    matches = [n for n in z.namelist() if Path(n).name == basename]
    if matches:
        raw = z.read(matches[0])
        img = Image.open(BytesIO(raw)).convert("L")
        return np.array(img, dtype=np.float32)

    # Fallback: search known disk locations for the image.
    # Why: board_captures zips in initial_temp_gauge/ only have annotations.xml.
    # The actual images are extracted under captured_images/ and heatmap_cd_320_ax/.
    search_dirs = [
        ROOT / "data" / "raw",
        ROOT / "data" / "captured_images" / "clean_board_captures_extracted" / "board_captures_1" / "images",
        ROOT / "data" / "captured_images" / "clean_board_captures_extracted" / "board_captures_2" / "images",
        ROOT / "data" / "captured_images" / "clean_board_captures_extracted" / "board_captures_3" / "images",
        ROOT / "data" / "heatmap_cd_320_ax" / "images" / "train",
        ROOT / "data" / "heatmap_cd_320_ax" / "images" / "val",
    ]
    for d in search_dirs:
        candidate = d / basename
        if candidate.exists():
            img = Image.open(candidate).convert("L")
            return np.array(img, dtype=np.float32)

    return None


def _resize_to_224(img: np.ndarray) -> np.ndarray:
    """Resize image to 224x224 using bilinear interpolation."""
    pil_img = Image.fromarray(img.astype(np.uint8), mode="L")
    pil_img = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.array(pil_img, dtype=np.float32)


def _crop_gauge_face(
    img: np.ndarray, face: dict, padding: float = 1.35
) -> np.ndarray:
    """Crop and resize the gauge face region from the image.

    Args:
        img: (H, W) grayscale image in [0, 255]
        face: dict with cx, cy, rx, ry in [0, 1] normalized coords
        padding: crop scale factor (1.0 = tight crop, 1.35 = 35% padding)
    """
    h, w = img.shape
    cx = face["cx"] * w
    cy = face["cy"] * h
    rx = face["rx"] * w * padding
    ry = face["ry"] * h * padding

    # Compute crop box (clamped to image bounds)
    x1 = max(0, int(cx - rx))
    y1 = max(0, int(cy - ry))
    x2 = min(w, int(cx + rx))
    y2 = min(h, int(cy + ry))

    crop = img[y1:y2, x1:x2]

    # Resize to 224x224
    if crop.size == 0:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    pil_crop = Image.fromarray(crop.astype(np.uint8), mode="L")
    pil_crop = pil_crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.array(pil_crop, dtype=np.float32)


def prepare_split(split: str, records: list[dict]) -> None:
    """Prepare one split: extract images, create heatmaps, save as npy."""
    split_dir = OUTPUT / split
    split_dir.mkdir(parents=True, exist_ok=True)

    n = len(records)
    # Pre-allocate arrays
    images = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
    ellipse_labels = np.zeros((n, 4), dtype=np.float32)  # cx, cy, rx, ry
    center_hm = np.zeros((n, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    tip_hm = np.zeros((n, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    has_needle = np.zeros(n, dtype=bool)

    # Group by zip for efficient reading
    by_zip: dict[str, list[tuple[int, dict]]] = {}
    for i, rec in enumerate(records):
        by_zip.setdefault(rec["zip_path"], []).append((i, rec))

    loaded = 0
    for zip_path, recs in by_zip.items():
        with zipfile.ZipFile(zip_path) as z:
            for idx, rec in recs:
                img = _extract_image(z, rec["name"])
                if img is None:
                    print(f"  WARNING: {rec['name']} not found in {zip_path}")
                    continue

                # Resize full image to 224x224 for ellipse detection
                img_224 = _resize_to_224(img)
                images[idx, ..., 0] = img_224 / 255.0

                # Store ellipse labels (already normalized)
                face = rec["gauge_face"]
                ellipse_labels[idx] = [face["cx"], face["cy"], face["rx"], face["ry"]]

                # Create gauge face crop for needle detection
                crop = _crop_gauge_face(img, face, padding=1.35)
                crop_norm = crop / 255.0

                # Generate heatmaps for needle keypoints (if available)
                # Heatmaps are in CROP SPACE [0, 1] — matching model output.
                # During eval, GT coords are transformed from crop space to
                # original image space using the crop→original transform.
                if rec["needle_center"] is not None and rec["needle_tip"] is not None:
                    h_orig, w_orig = img.shape
                    face_cx = face["cx"] * w_orig
                    face_cy = face["cy"] * h_orig
                    face_rx = face["rx"] * w_orig * 1.35
                    face_ry = face["ry"] * h_orig * 1.35

                    # Needle coords in original image space (already [0,1] normalized)
                    nc_x = rec["needle_center"]["x"]
                    nc_y = rec["needle_center"]["y"]
                    nt_x = rec["needle_tip"]["x"]
                    nt_y = rec["needle_tip"]["y"]

                    # Transform to crop space [0, 1]
                    crop_x1 = max(0, face_cx - face_rx)
                    crop_y1 = max(0, face_cy - face_ry)
                    crop_w = face_rx * 2
                    crop_h = face_ry * 2

                    nc_crop_x = (nc_x * w_orig - crop_x1) / crop_w
                    nc_crop_y = (nc_y * h_orig - crop_y1) / crop_h
                    nt_crop_x = (nt_x * w_orig - crop_x1) / crop_w
                    nt_crop_y = (nt_y * h_orig - crop_y1) / crop_h

                    # Clamp to [0, 1]
                    nc_crop_x = max(0.0, min(1.0, nc_crop_x))
                    nc_crop_y = max(0.0, min(1.0, nc_crop_y))
                    nt_crop_x = max(0.0, min(1.0, nt_crop_x))
                    nt_crop_y = max(0.0, min(1.0, nt_crop_y))

                    center_hm[idx] = _make_gaussian_heatmap(
                        HEATMAP_SIZE, HEATMAP_SIZE,
                        nc_crop_x, nc_crop_y, HEATMAP_SIGMA
                    )
                    tip_hm[idx] = _make_gaussian_heatmap(
                        HEATMAP_SIZE, HEATMAP_SIZE,
                        nt_crop_x, nt_crop_y, HEATMAP_SIGMA
                    )
                    has_needle[idx] = True

                loaded += 1

    # Save arrays
    np.save(split_dir / "images.npy", images)
    np.save(split_dir / "ellipse_labels.npy", ellipse_labels)
    np.save(split_dir / "center_heatmaps.npy", center_hm)
    np.save(split_dir / "tip_heatmaps.npy", tip_hm)
    np.save(split_dir / "has_needle.npy", has_needle)

    n_with_needle = int(has_needle.sum())
    print(f"  {split}: {loaded} images, {n_with_needle} with needle labels")
    print(f"    Ellipse labels: cx range [{ellipse_labels[:, 0].min():.3f}, {ellipse_labels[:, 0].max():.3f}]")
    print(f"    Center heatmap peaks: {center_hm.max():.3f}, Tip heatmap peaks: {tip_hm.max():.3f}")


def main() -> None:
    """Extract and prepare all splits."""
    np.random.seed(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for split, zip_names in SPLIT_FILES.items():
        print(f"\nProcessing {split}...")
        all_records: list[dict] = []
        for zname in zip_names:
            zp = LABELLED / zname
            if not zp.exists():
                print(f"  WARNING: {zname} not found, skipping")
                continue
            records = _parse_cvat_zip(zp)
            print(f"  {zname}: {len(records)} images with gauge face labels")
            all_records.extend(records)

        # Append additional needle-labeled data to train/val splits.
        # Why: the main zips have only 22 train + 11 val needle-labeled images.
        # The initial_temp_gauge batches provide 402+268 = 670 more.
        if split == "train":
            for zname in NEEDLE_TRAIN_FILES:
                zp = LABELLED / zname
                if not zp.exists():
                    print(f"  WARNING: {zname} not found, skipping")
                    continue
                records = _parse_cvat_zip(zp)
                print(f"  {zname}: {len(records)} needle-labeled images")
                all_records.extend(records)
        elif split == "val":
            for zname in NEEDLE_VAL_FILES:
                zp = LABELLED / zname
                if not zp.exists():
                    print(f"  WARNING: {zname} not found, skipping")
                    continue
                records = _parse_cvat_zip(zp)
                print(f"  {zname}: {len(records)} needle-labeled images")
                all_records.extend(records)

        prepare_split(split, all_records)

    # Save metadata
    metadata = {
        "image_size": IMAGE_SIZE,
        "heatmap_size": HEATMAP_SIZE,
        "heatmap_sigma": HEATMAP_SIGMA,
        "splits": {
            split: len(list((OUTPUT / split / "images.npy").parent.glob("*.npy")))
            for split in SPLIT_FILES
        },
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nDone. Data saved to {OUTPUT}")


if __name__ == "__main__":
    sys.exit(main())
