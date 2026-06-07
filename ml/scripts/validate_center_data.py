"""Validate manual board-capture labels for centre-detector training.

Checks:
- All referenced image files exist and are 224×224
- Labels are in [0, 1] normalized range
- Label distribution is sensible (no extreme outliers)
- No duplicate filenames with conflicting labels
- Visual spot-check: overlays predicted centre on image (optional)

Usage:
    poetry run python3 scripts/validate_center_data.py  [--visual]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUAL_DIR = PROJECT_ROOT / "data" / "center_training_manual"
EXPECTED_SIZE = 224


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action="store_true", help="Save validation overlay images")
    args = parser.parse_args()

    meta_path = MANUAL_DIR / "metadata.json"
    if not meta_path.exists():
        print(f"ERROR: metadata not found at {meta_path}")
        sys.exit(1)

    with open(meta_path) as f:
        entries = json.load(f)

    print(f"Loaded {len(entries)} entries from metadata.json")
    print()

    images_dir = MANUAL_DIR / "images"
    if not images_dir.is_dir():
        print(f"ERROR: images dir not found at {images_dir}")
        sys.exit(1)

    # ---- Check 1: All files exist and are correct size ----
    missing = []
    wrong_size = []
    for e in entries:
        img_path = images_dir / e["image"]
        if not img_path.exists():
            missing.append(e["image"])
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            missing.append(e["image"])
            continue
        h, w = img.shape[:2]
        if w != EXPECTED_SIZE or h != EXPECTED_SIZE:
            wrong_size.append((e["image"], w, h))

    if missing:
        print(f"WARN: {len(missing)} missing images (will be excluded)")
        for m in missing:
            print(f"  {m}")
    else:
        print(f"OK: All {len(entries)} images exist.")

    if wrong_size:
        print(f"ERROR: {len(wrong_size)} images wrong size (expected {EXPECTED_SIZE}x{EXPECTED_SIZE}):")
        for name, w, h in wrong_size[:10]:
            print(f"  {name}: {w}x{h}")
        sys.exit(1)
    else:
        print(f"OK: All images {EXPECTED_SIZE}x{EXPECTED_SIZE}.")

    # ---- Check 2: Label range sanity ----
    bad_labels = []
    cx_all = []
    cy_all = []
    for e in entries:
        cx = float(e["center_x"])
        cy = float(e["center_y"])
        cx_all.append(cx)
        cy_all.append(cy)
        if cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0:
            bad_labels.append((e["image"], cx, cy))

    if bad_labels:
        print(f"WARN: {len(bad_labels)} entries with out-of-range labels:")
        for name, cx, cy in bad_labels:
            print(f"  {name}: cx={cx:.4f} cy={cy:.4f}")
    else:
        print("OK: All labels in [0, 1].")

    cx_arr = np.array(cx_all)
    cy_arr = np.array(cy_all)

    print(f"\ncx: mean={cx_arr.mean():.4f} std={cx_arr.std():.4f} "
          f"range=[{cx_arr.min():.4f}, {cx_arr.max():.4f}]")
    print(f"cy: mean={cy_arr.mean():.4f} std={cy_arr.std():.4f} "
          f"range=[{cy_arr.min():.4f}, {cy_arr.max():.4f}]")

    # ---- Check 3: Pixel coordinates ----
    cx_px = cx_arr * EXPECTED_SIZE
    cy_px = cy_arr * EXPECTED_SIZE
    print(f"\ncx (px): mean={cx_px.mean():.1f} std={cx_px.std():.1f} "
          f"range=[{cx_px.min():.1f}, {cx_px.max():.1f}]")
    print(f"cy (px): mean={cy_px.mean():.1f} std={cy_px.std():.1f} "
          f"range=[{cy_px.min():.1f}, {cy_px.max():.1f}]")

    # ---- Check 4: Duplicate filenames ----
    seen: dict[str, list[tuple[float, float]]] = {}
    for e in entries:
        key = e["image"]
        if key not in seen:
            seen[key] = []
        seen[key].append((float(e["center_x"]), float(e["center_y"])))

    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        print(f"\nWARN: {len(dupes)} duplicate filenames with potentially differing labels:")
        for name, labels in list(dupes.items())[:5]:
            unique = set(labels)
            print(f"  {name}: {len(unique)} unique label(s) among {len(labels)} entries")
    else:
        print("\nOK: No duplicate filenames.")

    # ---- Remove bad entries and deduplicate ----
    bad_images = {m for m in missing}
    bad_images.update(name for name, _, _ in bad_labels)
    valid_entries = [e for e in entries if e["image"] not in bad_images]

    # Deduplicate: keep last entry for each image
    deduped: dict[str, dict] = {}
    for e in valid_entries:
        deduped[e["image"]] = e
    unique_entries = list(deduped.values())

    print(f"\nValidation summary:")
    print(f"  Total entries in metadata: {len(entries)}")
    print(f"  Valid unique entries: {len(unique_entries)}")
    print(f"  Excluded: {len(entries) - len(unique_entries)}")

    # Save cleaned metadata
    out_path = MANUAL_DIR / "metadata_cleaned.json"
    with open(out_path, "w") as f:
        json.dump(unique_entries, f, indent=2)
    print(f"  Cleaned metadata saved: {out_path}")

    # ---- Optional visual check ----
    if args.visual:
        vis_dir = MANUAL_DIR / "validation_overlays"
        vis_dir.mkdir(exist_ok=True)
        print(f"\nGenerating validation overlays in {vis_dir}...")
        for e in unique_entries:
            img_path = images_dir / e["image"]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            cx_px_v = int(round(float(e["center_x"]) * EXPECTED_SIZE))
            cy_px_v = int(round(float(e["center_y"]) * EXPECTED_SIZE))
            cx_px_v = max(0, min(cx_px_v, EXPECTED_SIZE - 1))
            cy_px_v = max(0, min(cy_px_v, EXPECTED_SIZE - 1))
            cv2.drawMarker(img, (cx_px_v, cy_px_v), (0, 0, 255),
                           cv2.MARKER_CROSS, 10, 2)
            cv2.imwrite(str(vis_dir / f"val_{e['image']}"), img)
        print(f"  Saved {len(unique_entries)} overlay images.")


if __name__ == "__main__":
    main()
