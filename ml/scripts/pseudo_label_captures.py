#!/usr/bin/env python3
"""Use rim-vote centre (from gauge geometry) to pseudo-label unlabelled YUV422 frames.

All CD crops in ``captured_images/`` were extracted at the same training-crop
window, so the gauge centre in every crop is at a nearly constant position.
We use the geometry-derived centre as the pseudo-label, matching the pattern
already used for the 76 hand-labelled capture samples in the training set.

This script:
  1. Finds all 224x224 YUV422 files NOT already in the training metadata.
  2. Converts each to int8 RGB (NPU input format).
  3. Assigns the geometry centre as the pseudo-label.
  4. Saves the RGB PNG to the training images directory.
  5. Appends metadata entries with ``source_kind="pseudo"``.
  6. Writes an updated ``metadata.json``.

Usage::

    python3 pseudo_label_captures.py [--output-dir PATH] [--metadata PATH]

The pseudo-labelled entries get sample weight 0.5× during training
(default ``CAPTURE_WEIGHT=0.5`` for ``source_kind="pseudo"``).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rim_vote_center import yuv422_to_rgb  # noqa: E402

# Gauge centre in a 224x224 CD crop (from app_gauge_geometry.h).
# Inner Celsius dial: centre_x = 0.5000 × 224 = 112.0, centre_y = 0.4460 × 224 = 99.9.
# Normalised: cx_norm = 112.0 / 224, cy_norm = 99.9 / 224.
# These match the existing 76 capture-sample labels.
GEOMETRY_CX_NORM = 112.0 / 224.0  # 0.5000
GEOMETRY_CY_NORM = 99.9 / 224.0   # 0.44598...

# Paths
DEFAULT_CAPTURED_DIR = PROJECT_ROOT / "data" / "captured_images"
DEFAULT_TRAINING_DIR = PROJECT_ROOT / "data" / "center_training_board_mimic"
DEFAULT_IMAGE_DIR = DEFAULT_TRAINING_DIR / "images"
DEFAULT_METADATA = DEFAULT_TRAINING_DIR / "metadata.json"

# Expected size for a 224×224 YUV422 crop (bytes)
CROP_SIZE_224 = 224 * 224 * 2  # 100352


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(description="Pseudo-label unlabelled YUV422 frames")
    parser.add_argument(
        "--captured-dir",
        default=str(DEFAULT_CAPTURED_DIR),
        help="Directory containing YUV422 capture files",
    )
    parser.add_argument(
        "--image-dir",
        default=str(DEFAULT_IMAGE_DIR),
        help="Output directory for RGB PNG images",
    )
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA),
        help="Path to metadata.json (will be read and overwritten)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list what would be done, don't write anything",
    )
    parser.add_argument(
        "--centre-x-norm",
        type=float,
        default=GEOMETRY_CX_NORM,
        help="Normalised centre X for pseudo-labels (default: from geometry)",
    )
    parser.add_argument(
        "--centre-y-norm",
        type=float,
        default=GEOMETRY_CY_NORM,
        help="Normalised centre Y for pseudo-labels (default: from geometry)",
    )
    parser.add_argument(
        "--jitter-px",
        type=int,
        default=2,
        help="Max random jitter in pixels per sample (default: 2, set 0 for none)",
    )
    return parser.parse_args(argv)


def find_unlabelled_yuv(
    captured_dir: Path,
    existing_sources: set[str],
) -> list[Path]:
    """Return YUV422 paths that are usable (224x224) and not in the existing metadata."""
    unlabelled: list[Path] = []
    for fpath in sorted(captured_dir.glob("capture_*.yuv422")):
        if fpath.stat().st_size != CROP_SIZE_224:
            continue
        base_key = fpath.stem  # e.g. "capture_0007"
        # The metadata source_path is typically ".../capture_XXXX.png"
        if base_key not in existing_sources and fpath.name not in existing_sources:
            unlabelled.append(fpath)
    return unlabelled


def build_existing_source_keys(metadata: list[dict]) -> set[str]:
    """Build a set of basenames (without extension) from existing metadata."""
    keys: set[str] = set()
    for entry in metadata:
        sp = entry.get("source_path", "")
        if sp:
            keys.add(Path(sp).stem)
        ip = entry.get("image_path", "")
        if ip:
            keys.add(Path(ip).stem)
    return keys


def main() -> None:
    args = parse_args()
    captured_dir = Path(args.captured_dir)
    image_dir = Path(args.image_dir)
    metadata_path = Path(args.metadata)

    cx_norm = args.centre_x_norm
    cy_norm = args.centre_y_norm
    jitter_px = args.jitter_px
    rng = np.random.default_rng(42)

    # ---- read existing metadata -------------------------------------------
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            metadata: list[dict] = json.load(f)
    else:
        metadata = []

    existing_keys = build_existing_source_keys(metadata)
    unlabelled = find_unlabelled_yuv(captured_dir, existing_keys)

    # Determine starting entry numbers
    next_entry_id = len(metadata) + 1
    next_capture_num = 1
    for entry in metadata:
        sp = entry.get("source_path", "")
        if "pseudo_" in sp:
            try:
                num = int(Path(sp).stem.replace("pseudo_capture_", ""))
                next_capture_num = max(next_capture_num, num + 1)
            except (ValueError, IndexError):
                pass
        else:
            try:
                num = int(Path(sp).stem.replace("capture_", ""))
                next_capture_num = max(next_capture_num, num + 1)
            except (ValueError, IndexError):
                pass

    print(f"Existing entries: {len(metadata)}")
    print(f"Unlabelled YUV422 files found: {len(unlabelled)}")
    print(f"Pseudo-label centre: ({cx_norm:.4f}, {cy_norm:.4f}) "
          f"= ({cx_norm*224:.1f}, {cy_norm*224:.1f}) px")
    print(f"Jitter: {jitter_px} px max")
    print()
    if not unlabelled:
        print("Nothing to do.")
        return

    if args.dry_run:
        print(f"Would process {len(unlabelled)} files (dry-run; no files written)")
        for fpath in unlabelled[:5]:
            print(f"  {fpath.name}")
        if len(unlabelled) > 5:
            print(f"  ... and {len(unlabelled) - 5} more")
        return

    # ---- process each unlabelled file -------------------------------------
    new_entries: list[dict] = []
    image_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    errors = 0

    for fpath in unlabelled:
        try:
            yuv_bytes = fpath.read_bytes()
            if len(yuv_bytes) != CROP_SIZE_224:
                continue

            # Convert YUV422 → RGB
            rgb = yuv422_to_rgb(yuv_bytes, 224, 224)
            img = Image.fromarray(rgb)

            # Optionally jitter the label slightly for diversity
            if jitter_px > 0:
                jx = rng.integers(-jitter_px, jitter_px + 1)
                jy = rng.integers(-jitter_px, jitter_px + 1)
            else:
                jx, jy = 0, 0

            label_cx = min(223, max(0, int(round(cx_norm * 224)) + jx))
            label_cy = min(223, max(0, int(round(cy_norm * 224)) + jy))
            label_cx_norm = label_cx / 224.0
            label_cy_norm = label_cy / 224.0

            # Save as PNG
            png_name = f"pseudo_capture_{next_capture_num:04d}.png"
            png_path = image_dir / png_name
            img.save(png_path, "PNG")

            # Build metadata entry
            entry = {
                "image_path": f"images/{png_name}",
                "source_path": str(fpath.relative_to(captured_dir.parent)
                                   if captured_dir.parent.exists()
                                   else fpath.name),
                "split": "train",
                "quality_flag": "pseudo",
                "source_kind": "pseudo",
                "temperature_c": None,
                "source_width": 224,
                "source_height": 224,
                "center_x_source": label_cx,
                "center_y_source": label_cy,
                "full_frame_center_x": label_cx,
                "full_frame_center_y": label_cy,
                "crop_center_x": float(label_cx),
                "crop_center_y": float(label_cy),
                "obb_accepted": True,
                "obb_fallback_reason": None,
                "crop_source": "geometry_pseudo",
                "crop_x_min": 0,
                "crop_y_min": 0,
                "crop_width": 224,
                "crop_height": 224,
                "center_x_norm": label_cx_norm,
                "center_y_norm": label_cy_norm,
                "full_frame_size": 224,
                "full_frame_resized_width": 224,
                "full_frame_resized_height": 224,
                "full_frame_pad_x": 0,
                "full_frame_pad_y": 0,
            }
            new_entries.append(entry)
            next_capture_num += 1
            processed += 1
            if processed % 50 == 0:
                print(f"  Processed {processed}/{len(unlabelled)}")

        except Exception as exc:
            print(f"  ERROR processing {fpath.name}: {exc}")
            errors += 1

    # ---- append and write metadata ----------------------------------------
    metadata.extend(new_entries)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {processed} pseudo-labelled, {errors} errors")
    print(f"Total metadata entries: {len(metadata)}")

    # Print split summary
    train = sum(1 for e in metadata if e.get("split") == "train")
    val = sum(1 for e in metadata if e.get("split") == "val")
    test = sum(1 for e in metadata if e.get("split") == "test")
    capture = sum(1 for e in metadata if e.get("source_kind") == "capture")
    pxl = sum(1 for e in metadata if e.get("source_kind") == "pxl")
    pseudo = sum(1 for e in metadata if e.get("source_kind") == "pseudo")
    print(f"  Train: {train}  Val: {val}  Test: {test}")
    print(f"  Capture: {capture}  PXL: {pxl}  Pseudo: {pseudo}")


if __name__ == "__main__":
    main()
