"""Create merged manifest: geometry labels + board captures.

Board captures are already 224x224 with coordinates in pixel space.
We set loose_crop = full image (0, 0, 224, 224) so the existing
pipeline treats them as pre-cropped gauge images.

Usage:
    python ml/scripts/create_merged_manifest.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path


BOARD_CAPTURES_CSV = Path(__file__).resolve().parent.parent / "data" / "board_captures_labeled_v2.csv"
GEOMETRY_MANIFEST_CSV = Path(__file__).resolve().parent.parent / "data" / "geometry_reader_manifest_v2_clean.csv"
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "data" / "merged_geometry_board_manifest.csv"


def load_board_captures() -> list[dict]:
    """Load board captures and convert to geometry manifest format."""
    samples = []
    with open(BOARD_CAPTURES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("image_path") or not row.get("center_x"):
                continue
            try:
                temp = float(row["temperature_c"])
                center_x = float(row["center_x"])
                center_y = float(row["center_y"])
                tip_x = float(row["tip_x"])
                tip_y = float(row["tip_y"])
                src_w = int(float(row.get("source_width", 224)))
                src_h = int(float(row.get("source_height", 224)))
            except (ValueError, KeyError):
                continue

            # Board captures are already cropped; use full image as crop box
            samples.append({
                "image_path": row["image_path"],
                "temperature_c": temp,
                "split": "train",  # Will be reassigned below
                "source_width": src_w,
                "source_height": src_h,
                "loose_crop_x1": 0,
                "loose_crop_y1": 0,
                "loose_crop_x2": src_w,
                "loose_crop_y2": src_h,
                "center_x_source": center_x,
                "center_y_source": center_y,
                "tip_x_source": tip_x,
                "tip_y_source": tip_y,
                "dial_radius_source": 80.0,  # approximate for 224x224
                "label_quality": "manual",
                "source_manifest": "board_captures_v2",
                "notes": "",
                "angle_degrees_from_labels": float(row.get("angle_degrees", 0)),
                "deterministic_temperature_c": temp,
                "absolute_temperature_difference_c": 0.0,
                "center_tip_distance_pixels": ((tip_x - center_x) ** 2 + (tip_y - center_y) ** 2) ** 0.5,
                "quality_flag": "clean",
            })

    return samples


def load_geometry_samples() -> list[dict]:
    """Load geometry manifest rows as dicts."""
    samples = []
    with open(GEOMETRY_MANIFEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("image_path"):
                continue
            quality = row.get("quality_flag", "").strip()
            if quality == "exclude":
                continue
            samples.append(dict(row))
    return samples


def main() -> None:
    """Merge geometry + board capture manifests with stratified splits."""
    geometry = load_geometry_samples()
    board = load_board_captures()

    print(f"Geometry samples: {len(geometry)}")
    print(f"Board captures: {len(board)}")

    # Assign splits to board captures (15% val)
    rng = random.Random(42)
    rng.shuffle(board)
    n_val = max(1, int(len(board) * 0.15))
    for i, s in enumerate(board):
        s["split"] = "val" if i < n_val else "train"

    # Merge
    merged = geometry + board
    rng.shuffle(merged)

    # Write
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path", "temperature_c", "split", "source_width", "source_height",
        "loose_crop_x1", "loose_crop_y1", "loose_crop_x2", "loose_crop_y2",
        "center_x_source", "center_y_source", "tip_x_source", "tip_y_source",
        "dial_radius_source", "label_quality", "source_manifest", "notes",
        "angle_degrees_from_labels", "deterministic_temperature_c",
        "absolute_temperature_difference_c", "center_tip_distance_pixels",
        "quality_flag",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in merged:
            writer.writerow({k: s.get(k, "") for k in fieldnames})

    n_train = sum(1 for s in merged if s.get("split") == "train")
    n_val = sum(1 for s in merged if s.get("split") == "val")
    print(f"Merged: {len(merged)} total ({n_train} train, {n_val} val)")
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
