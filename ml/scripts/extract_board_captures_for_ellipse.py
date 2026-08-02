"""Extract board-capture images from labelled zips into the ellipse training dataset.

Converts temp_dial ellipse annotations to YOLO-OBB format and merges them
into data/gauge_face_ellipse_v1_640_gray/images/train/ and labels/train/.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LABELLED = ROOT / "data" / "labelled"
TARGET = ROOT / "data" / "gauge_face_ellipse_v1_640_gray"


def ellipse_to_obb(cx: float, cy: float, rx: float, ry: float, w: float, h: float) -> str:
    """Convert ellipse to normalized YOLO-OBB label string."""
    corners = np.array([
        [cx - rx, cy - ry],
        [cx + rx, cy - ry],
        [cx + rx, cy + ry],
        [cx - rx, cy + ry],
    ], dtype=np.float32)
    corners[:, 0] = np.clip(corners[:, 0] / w, 0, 1)
    corners[:, 1] = np.clip(corners[:, 1] / h, 0, 1)
    return "0 " + " ".join(f"{v:.6f}" for v in corners.flatten())


def extract_zip(zip_path: Path, prefix: str) -> int:
    """Extract images and annotations from one CVAT zip into training data."""
    z = zipfile.ZipFile(zip_path)
    tree = ET.parse(z.open("annotations.xml"))
    root = tree.getroot()
    count = 0

    for image_elem in root.findall("image"):
        name = image_elem.get("name")
        w = int(image_elem.get("width"))
        h = int(image_elem.get("height"))

        # Find the gauge face ellipse
        ellipse = None
        for elem in image_elem:
            label = elem.get("label", "")
            if "dial" in label.lower() or label == "GaugeFace":
                if elem.tag == "ellipse":
                    ellipse = {
                        "cx": float(elem.get("cx")),
                        "cy": float(elem.get("cy")),
                        "rx": float(elem.get("rx")),
                        "ry": float(elem.get("ry")),
                    }
                    break

        if ellipse is None:
            continue

        # Read and save image as grayscale 640x640
        try:
            data = z.read(f"images/{name}")
            img = Image.open(io.BytesIO(data)).convert("L")
            img = img.resize((640, 640), Image.Resampling.BILINEAR)
        except KeyError:
            continue

        stem = f"{prefix}_{Path(name).stem}"
        img.save(TARGET / "images" / "train" / f"{stem}.png")

        # Scale ellipse from original coords to 640x640
        scale_x = 640.0 / w
        scale_y = 640.0 / h
        label_str = ellipse_to_obb(
            ellipse["cx"] * scale_x,
            ellipse["cy"] * scale_y,
            ellipse["rx"] * scale_x,
            ellipse["ry"] * scale_y,
            640.0, 640.0,
        )
        (TARGET / "labels" / "train" / f"{stem}.txt").write_text(label_str + "\n")
        count += 1

    return count


def main() -> None:
    (TARGET / "images" / "train").mkdir(parents=True, exist_ok=True)
    (TARGET / "labels" / "train").mkdir(parents=True, exist_ok=True)

    total = 0
    for zip_name, prefix in [
        ("initial_temp_gauge/board_captures_1.zip", "bc1"),
        ("initial_temp_gauge/board_captures_2.zip", "bc2"),
    ]:
        zip_path = LABELLED / zip_name
        if zip_path.exists():
            n = extract_zip(zip_path, prefix)
            total += n
            print(f"{zip_name}: {n} images extracted")

    print(f"\nTotal: {total} board-capture images added to training")


if __name__ == "__main__":
    main()
