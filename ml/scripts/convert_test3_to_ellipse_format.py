"""Convert test_3.zip CVAT ellipse annotations to YOLO-OBB label format.

Outputs images/ and labels/ under data/gauge_face_ellipse_v1_640_gray/test_3/
for integration into the ellipse training pipeline.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TEST3_ZIP = ROOT / "data" / "labelled" / "test_3.zip"
OUT = ROOT / "data" / "gauge_face_ellipse_v1_640_gray" / "test_3"


def ellipse_to_obb_corners(cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    """Convert ellipse params to 4 corner points of the bounding box."""
    corners = np.array([
        [cx - rx, cy - ry],  # top-left
        [cx + rx, cy - ry],  # top-right
        [cx + rx, cy + ry],  # bottom-right
        [cx - rx, cy + ry],  # bottom-left
    ], dtype=np.float32)
    return corners


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    (OUT / "labels").mkdir(exist_ok=True)

    z = zipfile.ZipFile(TEST3_ZIP)
    tree = ET.parse(z.open("annotations.xml"))
    root = tree.getroot()

    for image_elem in root.findall("image"):
        name = image_elem.get("name")
        stem = Path(name).stem
        w = int(image_elem.get("width", "640"))
        h = int(image_elem.get("height", "640"))

        # Extract image
        data = z.read(f"images/{name}")
        img = Image.open(io.BytesIO(data)).convert("L")
        img.save(OUT / "images" / f"{stem}.png")

        # Find GaugeFace ellipse annotation
        for elem in image_elem:
            if elem.get("label") == "GaugeFace" and elem.tag == "ellipse":
                cx = float(elem.get("cx"))
                cy = float(elem.get("cy"))
                rx = float(elem.get("rx"))
                ry = float(elem.get("ry"))
                break
        else:
            print(f"  WARNING: {name} has no GaugeFace annotation, skipping")
            continue

        # Convert to YOLO-OBB format: class x1 y1 x2 y2 x3 y3 x4 y4
        corners = ellipse_to_obb_corners(cx, cy, rx, ry)
        # Normalize to [0,1]
        corners[:, 0] /= w
        corners[:, 1] /= h
        label_line = "0 " + " ".join(f"{v:.6f}" for v in corners.flatten())
        (OUT / "labels" / f"{stem}.txt").write_text(label_line + "\n")

        print(f"  {name} -> {stem}.png")

    print(f"\nConverted {len(list((OUT/'images').glob('*.png')))} images to {OUT}")


if __name__ == "__main__":
    main()
