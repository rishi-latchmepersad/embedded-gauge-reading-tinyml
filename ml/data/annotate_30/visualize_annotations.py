"""Visualize manual annotations by drawing center points on images."""

import csv
import sys
from pathlib import Path
from PIL import Image, ImageDraw

images_dir = Path(__file__).parent / "images"
annotations_csv = Path(__file__).parent / "manual_annotations.csv"
output_dir = Path(__file__).parent / "annotated_previews"
output_dir.mkdir(exist_ok=True)

# Load annotations
annotations = []
with open(annotations_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        annotations.append(row)

# Draw center point on each image
for ann in annotations:
    filename = ann["filename"]
    cx = int(ann["cx"])
    cy = int(ann["cy"])

    img_path = images_dir / filename
    if not img_path.exists():
        continue

    img = Image.open(img_path).convert("RGB")
    # Scale up for visibility
    img = img.resize((448, 448), Image.NEAREST)
    cx, cy = cx * 2, cy * 2

    draw = ImageDraw.Draw(img)

    # Draw large red crosshair
    size = 20
    draw.line([(cx - size, cy), (cx + size, cy)], fill="red", width=4)
    draw.line([(cx, cy - size), (cx, cy + size)], fill="red", width=4)

    # Draw red circle
    draw.ellipse([(cx - 10, cy - 10), (cx + 10, cy + 10)], outline="red", width=4)

    # Save
    output_path = output_dir / f"ann_{filename}"
    img.save(output_path)

print(f"Saved {len(annotations)} annotated previews to {output_dir}")
