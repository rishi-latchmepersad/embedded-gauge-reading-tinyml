"""Extract GaugeFace ellipse records from the CVAT 1.1 zips and stage them on disk.

Why extract to disk:
- 9,191 images at 640x640 grayscale = 14.7 GB of float32 if we hold them in
  memory. That doesn't fit in our 15 GB GPU budget.
- tf.data can stream from disk with no perf hit because we read whole
  JPEGs (each ~50-150 KB) and decode in parallel.
- Extracting once takes ~30 s and saves a lot of repeated zip seeks.

Output layout (relative to `ml/data/repvgg_ellipse/`):
    train/images/000000.jpg
    train/images/000001.jpg
    ...
    train/labels.json   # list of {image, cx, cy, rx, ry}
    val/images/...
    val/labels.json
    test/images/...
    test/labels.json
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
LABELLED = ROOT / "data" / "labelled"
OUTPUT = ROOT / "data" / "repvgg_ellipse"


# Why we pick these zips: train_1+train_2 are the canonical 9K-image
# training set with GaugeFace labels. val_1 is the held-out CVAT validation
# split. test_1+test_3 are the held-out test sets. The board_captures
# zips do NOT carry GaugeFace ellipse labels so we skip them here.
SPLIT_FILES: dict[str, list[str]] = {
    "train": [
        "train_1.zip",
        "train_2.zip",
    ],
    "val": [
        "val_1.zip",
        "val_2.zip",
    ],
    "test": [
        "test_1.zip",
        "test_2.zip",
        "test_3.zip",
    ],
}


def _iter_records(zip_paths: list[Path]) -> list[dict]:
    """Walk every zip, collect GaugeFace ellipses, and return flat records."""
    records: list[dict] = []
    for zp in zip_paths:
        if not zp.exists():
            print(f"  WARNING: {zp.name} not found, skipping")
            continue
        with zipfile.ZipFile(zp) as z:
            try:
                xml_bytes = z.read("annotations.xml")
            except KeyError:
                print(f"  WARNING: {zp.name} has no annotations.xml")
                continue
            root = ET.fromstring(xml_bytes)
            for img_node in root.findall("image"):
                width = float(img_node.get("width", 640))
                height = float(img_node.get("height", 640))
                name = img_node.get("name")
                for el in img_node.findall("ellipse"):
                    if el.get("label") != "GaugeFace":
                        continue
                    records.append({
                        "zip": str(zp),
                        "name": name,
                        "width": width,
                        "height": height,
                        # Normalised [0, 1] coordinates — the model head
                        # already produces values in this range.
                        "cx": float(el.get("cx")) / width,
                        "cy": float(el.get("cy")) / height,
                        "rx": float(el.get("rx")) / width,
                        "ry": float(el.get("ry")) / height,
                    })
    return records


def _stage_split(split: str, records: list[dict]) -> None:
    """Extract the JPEG for each record into <split>/images/ and write labels.json."""
    img_dir = OUTPUT / split / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels: list[dict] = []

    # Group records by zip so we open each archive only once.
    by_zip: dict[str, list[dict]] = {}
    for rec in records:
        by_zip.setdefault(rec["zip"], []).append(rec)

    counter = 0
    for zip_path, recs in tqdm(by_zip.items(), desc=f"stage {split}"):
        with zipfile.ZipFile(zip_path) as z:
            for rec in recs:
                # The CVAT export puts images under `images/...`. We need to
                # find the entry by basename because some zips use a nested
                # subfolder (e.g. `images/train/...`).
                basename = Path(rec["name"]).name
                matches = [n for n in z.namelist() if Path(n).name == basename]
                if not matches:
                    print(f"  WARNING: {basename} not found in {zip_path}")
                    continue
                raw = z.read(matches[0])
                # Use PIL to normalise format (some are PNG, most are JPEG).
                img = Image.open(__import__("io").BytesIO(raw)).convert("L")
                # Save as JPEG with a deterministic filename.
                out_name = f"{counter:06d}.jpg"
                img.save(img_dir / out_name, "JPEG", quality=95)
                labels.append({
                    "image": out_name,
                    "cx": rec["cx"], "cy": rec["cy"],
                    "rx": rec["rx"], "ry": rec["ry"],
                })
                counter += 1

    (OUTPUT / split / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"  {split}: wrote {len(labels)} samples to {img_dir}")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for split, zip_names in SPLIT_FILES.items():
        zip_paths = [LABELLED / n for n in zip_names]
        print(f"Reading {split} from {[p.name for p in zip_paths]}")
        records = _iter_records(zip_paths)
        print(f"  {len(records)} GaugeFace records")
        _stage_split(split, records)
    print(f"Done. Staged data at {OUTPUT}")


if __name__ == "__main__":
    sys.exit(main())
