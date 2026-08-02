"""Extract original 640x640 images from labelled zips into the ellipse dataset.

This makes the original images available for adaptive crop re-processing.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from PIL import Image
import io

ROOT = Path(__file__).resolve().parents[1]
LABELLED = ROOT / "data" / "labelled"
TARGET = ROOT / "data" / "gauge_face_ellipse_v1_640_gray" / "images" / "train"


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    existing = set(p.stem for p in TARGET.glob("*.png"))
    print(f"Existing images: {len(existing)}")

    # Extract from train_1.zip
    for zip_name in ["train_1.zip", "val_1.zip", "test_1.zip"]:
        zip_path = LABELLED / zip_name
        if not zip_path.exists():
            print(f"{zip_name}: not found, skipping")
            continue

        z = zipfile.ZipFile(zip_path)
        count = 0
        for name in z.namelist():
            if not name.endswith(".png"):
                continue
            stem = Path(name).stem
            if stem in existing:
                continue
            data = z.read(name)
            img = Image.open(io.BytesIO(data)).convert("L")
            img = img.resize((640, 640), Image.Resampling.LANCZOS)
            img.save(TARGET / f"{stem}.png")
            existing.add(stem)
            count += 1
        print(f"{zip_name}: {count} new images extracted")

    print(f"\nTotal images in train: {len(list(TARGET.glob('*.png')))}")


if __name__ == "__main__":
    main()
