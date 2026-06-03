#!/usr/bin/env python3
"""Auto-annotate gauge centers using Hough circle detection.

For each 224x224 capture, detects the outer circular rim and uses its center
as the needle pivot point. Writes a manifest CSV with image_path, center_x, center_y.
"""

from pathlib import Path
import csv

import cv2
import numpy as np
from PIL import Image


def find_gauge_center(image_path: Path) -> tuple[float, float] | None:
    """Detect gauge center via Hough circle transform.

    Returns (cx, cy) in pixel coords, or None if detection fails.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough circles: look for one prominent circle (the gauge rim)
    # For a 224x224 image, the gauge rim is typically 150-200 px diameter
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=60,
        maxRadius=110,
    )

    if circles is None or len(circles) == 0:
        return None

    # Pick the circle with the strongest vote
    circles = np.uint16(np.around(circles[0]))
    best = max(circles, key=lambda c: c[2])  # largest radius usually = outer rim
    cx, cy, radius = best
    return float(cx), float(cy)


def main():
    project_root = Path(__file__).resolve().parents[2]
    captured_dir = project_root / "ml" / "data" / "captured_images"
    manifest_path = project_root / "ml" / "data" / "ai_annotated_centers.csv"

    # Find all 224x224 PNG images
    image_paths = []
    for png_path in captured_dir.glob("*.png"):
        try:
            with Image.open(png_path) as im:
                if im.size == (224, 224):
                    image_paths.append(png_path)
        except Exception:
            pass

    print(f"Found {len(image_paths)} 224x224 images")

    results = []
    failed = []
    for img_path in sorted(image_paths):
        center = find_gauge_center(img_path)
        if center:
            cx, cy = center
            rel_path = img_path.relative_to(project_root / "ml" / "data")
            results.append({
                "image_path": str(rel_path),
                "center_x": f"{cx:.1f}",
                "center_y": f"{cy:.1f}",
            })
            print(f"  {img_path.name}: center=({cx:.1f}, {cy:.1f})")
        else:
            failed.append(img_path.name)
            print(f"  {img_path.name}: FAILED")

    # Write manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "center_x", "center_y"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAnnotated {len(results)} / {len(image_paths)} images")
    if failed:
        print(f"Failed on {len(failed)} images")
        for name in failed[:10]:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
