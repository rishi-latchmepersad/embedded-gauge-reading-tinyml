#!/usr/bin/env python3
"""Pre-process geometry + board capture data into 224x224 crops.

One-time offline step that:
1. Loads each phone photo
2. Applies the loose crop box
3. Resizes to 224x224
4. Transforms center/tip coordinates to normalized [0,1] space
5. Saves cropped PNG + metadata JSON to disk

This makes training fast since no large images need loading at train time.

Usage:
    cd ml && poetry run python scripts/preprocess_crops.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_crop_dataset import (
    load_geometry_manifest,
    create_jittered_crop,
    JitterParams,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import (
    load_clean_geometry_examples,
)

CROPS_DIR = Path(__file__).resolve().parent.parent / "data" / "preprocessed_crops"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "merged_geometry_board_manifest.csv"
BASE_PATH = Path(__file__).resolve().parent.parent.parent  # repo root

IDENTITY_JITTER = JitterParams(shift_x=0, shift_y=0, scale=1.0, aspect=1.0)


def preprocess_one(example, output_dir: Path, base_path: Path) -> dict | None:
    """Crop, resize, and save one sample. Returns metadata dict."""
    try:
        crop = create_jittered_crop(example, IDENTITY_JITTER)
        if not crop.accepted:
            return None

        image_path = base_path / crop.source_image_path
        with Image.open(image_path) as img:
            crop_box = (crop.crop_x1, crop.crop_y1, crop.crop_x2, crop.crop_y2)
            cropped = img.convert("RGB").crop(crop_box).resize((224, 224), Image.Resampling.LANCZOS)

        # Save image
        img_name = Path(crop.source_image_path).stem + ".png"
        img_path = output_dir / "images" / img_name
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(str(img_path), "PNG")

        # Return metadata with normalized coords
        return {
            "image_path": f"images/{img_name}",
            "source_path": crop.source_image_path,
            "temperature_c": crop.temperature_c,
            "split": crop.split,
            "center_x_norm": float(crop.center_x_normalized),
            "center_y_norm": float(crop.center_y_normalized),
            "tip_x_norm": float(crop.tip_x_normalized),
            "tip_y_norm": float(crop.tip_y_normalized),
            "angle_degrees": float(crop.angle_degrees) if crop.angle_degrees is not None else 0.0,
            "quality_flag": example.quality_flag,
        }
    except Exception as e:
        return None


def main() -> None:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading merged manifest...")
    examples = load_geometry_manifest(MANIFEST_PATH)
    clean = [e for e in examples if e.quality_flag in ("clean", "review")]
    print(f"Loaded {len(examples)} examples, {len(clean)} clean/review")

    t0 = time.time()
    metadata = []
    for i, example in enumerate(clean):
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(clean) - i - 1) / rate
            print(f"  [{i+1}/{len(clean)}] {rate:.1f} samples/s, ETA {eta:.0f}s")

        result = preprocess_one(example, CROPS_DIR, BASE_PATH)
        if result is not None:
            metadata.append(result)

    # Save metadata
    meta_path = CROPS_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    n_train = sum(1 for m in metadata if m["split"] == "train")
    n_val = sum(1 for m in metadata if m["split"] == "val")
    print(f"\nDone: {len(metadata)} crops in {elapsed:.1f}s")
    print(f"  Train: {n_train}, Val: {n_val}")
    print(f"  Saved to {CROPS_DIR}")


if __name__ == "__main__":
    main()
