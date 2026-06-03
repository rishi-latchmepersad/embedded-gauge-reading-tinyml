import csv
from pathlib import Path

project_root = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml")
manifest = project_root / "ml" / "data" / "manual_annotated_centers.csv"

# Manual annotations for images that failed auto-detection
# Based on visual inspection above
manual = [
    # capture_0c.png - gauge visible, needle at ~0°C, center at needle pivot
    {"image_path": "captured_images/capture_0c.png", "center_x": "120.0", "center_y": "110.0"},
    # capture_0c_preview.png - color, clearer, center at needle pivot  
    {"image_path": "captured_images/capture_0c_preview.png", "center_x": "115.0", "center_y": "100.0"},
]

with open(manifest, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "center_x", "center_y"])
    writer.writerows(manual)

print(f"Added {len(manual)} manual annotations")

# Count total
with open(manifest) as f:
    total = sum(1 for _ in f) - 1  # minus header
print(f"Total annotated: {total}")
