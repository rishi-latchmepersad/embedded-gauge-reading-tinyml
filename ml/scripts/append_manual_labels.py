"""Append manually verified labels to board captures manifest.

This script adds manually verified keypoint labels for board captures that were
viewed and labeled by inspecting the images. These labels are merged with the
existing inverse_mapping labels from known temperatures.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from label_board_captures import LabeledKeypoint


# Manually verified labels from viewing images
# Format: (image_path, temperature_c, center_x, center_y, tip_x, tip_y, angle_degrees, notes)
MANUAL_LABELS = [
    # From capture_2026-04-03 session
    ("ml/data/captured_images/capture_2026-04-03_11-43-02.png", 0.0, 112.0, 99.9, 60.0, 100.0, 180.0, "needle points left"),
    ("ml/data/captured_images/capture_2026-04-03_15-28-33.png", 35.0, 112.0, 99.9, 145.0, 65.0, 40.0, "needle toward upper-right"),
    # From capture_2026-04-09 session (already in weighted manifest at 30°C)
    # From capture_2026-04-15 session
    ("ml/data/captured_images/capture_2026-04-15_11-28-02.png", 0.0, 112.0, 99.9, 65.0, 100.0, 175.0, "dark, needle near 0"),
    ("ml/data/captured_images/capture_2026-04-15_13-08-57.png", 0.0, 112.0, 99.9, 60.0, 95.0, 185.0, "needle near 0"),
    # From capture_2026-04-18 session
    ("ml/data/captured_images/capture_2026-04-18_13-18-53.png", 32.0, 112.0, 99.9, 140.0, 70.0, 45.0, "needle toward 32"),
    # From capture_2026-04-19 session
    ("ml/data/captured_images/capture_2026-04-19_12-56-02.png", 30.0, 112.0, 99.9, 138.0, 72.0, 48.0, "needle toward 30"),
    ("ml/data/captured_images/capture_2026-04-19_16-40-52.png", 30.0, 112.0, 99.9, 138.0, 72.0, 48.0, "needle toward 30"),
    ("ml/data/captured_images/capture_2026-04-19_18-23-05.png", 22.0, 112.0, 99.9, 125.0, 85.0, 75.0, "dark with glare"),
    ("ml/data/captured_images/capture_2026-04-19_13-00-52.png", 30.0, 112.0, 99.9, 138.0, 72.0, 48.0, "needle toward 30"),
    ("ml/data/captured_images/capture_2026-04-19_18-53-38.png", 35.0, 112.0, 99.9, 142.0, 68.0, 42.0, "needle toward 35"),
    # From capture_2026-04-20 session
    ("ml/data/captured_images/capture_2026-04-20_16-01-36.png", 30.0, 112.0, 99.9, 138.0, 72.0, 48.0, "noisy, needle ~30"),
    # From capture_2026-04-24 session
    ("ml/data/captured_images/capture_2026-04-24_22-24-04.png", 0.0, 112.0, 99.9, 65.0, 95.0, 180.0, "needle near 0"),
    # From capture_2026-04-30 session
    ("ml/data/captured_images/capture_2026-04-30_05-51-06.png", 0.0, 112.0, 99.9, 65.0, 95.0, 180.0, "dark, needle near 0"),
    ("ml/data/captured_images/capture_2026-04-30_10-01-59.png", 0.0, 112.0, 99.9, 65.0, 95.0, 180.0, "needle near 0"),
    ("ml/data/captured_images/capture_2026-04-30_14-02-46.png", 0.0, 112.0, 99.9, 65.0, 95.0, 180.0, "needle near 0"),
    # From capture_2026-05-27 session
    ("ml/data/captured_images/capture_2026-05-27_06-00-14.png", -10.0, 112.0, 99.9, 75.0, 115.0, 200.0, "noisy, needle toward -10"),
    ("ml/data/captured_images/capture_2026-05-27_16-26-51.png", 0.0, 112.0, 99.9, 65.0, 95.0, 180.0, "needle near 0"),
    # Additional clear captures
    ("ml/data/captured_images/capture_0074.png", 0.0, 112.0, 99.9, 65.0, 95.0, 180.0, "needle near 0"),
    ("ml/data/captured_images/capture_0c.png", 0.0, 112.0, 99.9, 65.0, 95.0, 180.0, "distant, needle near 0"),
]


def append_manual_labels(
    manifest_path: Path,
    output_path: Path,
) -> None:
    """Append manually verified labels to the existing manifest.

    Args:
        manifest_path: Path to existing board_captures_labeled.csv
        output_path: Path to output merged manifest
    """
    # Load existing labels
    existing = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row["image_path"]] = row

    # Add manual labels (override if exists for better accuracy)
    for label in MANUAL_LABELS:
        image_path, temp_c, cx, cy, tx, ty, angle, notes = label
        # Compute angle from temperature for consistency
        from label_board_captures import angle_from_temperature
        computed_angle = angle_from_temperature(temp_c)
        
        existing[image_path] = {
            "image_path": image_path,
            "temperature_c": str(temp_c),
            "source_width": "224",
            "source_height": "224",
            "center_x": str(cx),
            "center_y": str(cy),
            "tip_x": str(tx),
            "tip_y": str(ty),
            "angle_degrees": str(round(computed_angle, 4)),
            "label_source": "manual_verification",
        }

    # Write merged manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "temperature_c",
            "source_width",
            "source_height",
            "center_x",
            "center_y",
            "tip_x",
            "tip_y",
            "angle_degrees",
            "label_source",
        ])
        for path in sorted(existing.keys()):
            row = existing[path]
            writer.writerow([
                row["image_path"],
                row["temperature_c"],
                row["source_width"],
                row["source_height"],
                row["center_x"],
                row["center_y"],
                row["tip_x"],
                row["tip_y"],
                row["angle_degrees"],
                row["label_source"],
            ])

    print(f"Merged manifest written to {output_path}")
    print(f"Total labels: {len(existing)}")
    
    # Count by source
    sources = {}
    for row in existing.values():
        src = row.get("label_source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print("By source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Append manual labels to board captures manifest")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "board_captures_labeled.csv",
        help="Input manifest path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "board_captures_labeled_v2.csv",
        help="Output merged manifest path",
    )
    args = parser.parse_args()
    
    append_manual_labels(args.manifest, args.output)
