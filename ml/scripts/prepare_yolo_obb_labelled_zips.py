#!/usr/bin/env python3
"""Prepare a leakage-safe YOLO OBB dataset from the labelled CVAT zips."""

from __future__ import annotations

import argparse
import io
import random
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

from eval_ellipse_all_test_sets import LABELLED

SEED = 42


def read_samples(zip_name: str) -> list[tuple[str, bytes, tuple[float, float, float, float]]]:
    """Extract image bytes and normalized ellipse geometry from one CVAT zip."""
    samples: list[tuple[str, bytes, tuple[float, float, float, float]]] = []
    with zipfile.ZipFile(LABELLED / zip_name) as archive:
        members = {Path(member).name: member for member in archive.namelist()}
        root = ET.fromstring(archive.read("annotations.xml"))
        for node in root.findall("image"):
            shape = next((item for item in node.findall("ellipse") if item.get("label") in {"GaugeFace", "temp_dial"}), None)
            member = members.get(Path(node.get("name", "")).name)
            if shape is None or member is None:
                continue
            width, height = float(node.get("width", 640)), float(node.get("height", 640))
            target = (float(shape.get("cx")) / width, float(shape.get("cy")) / height, float(shape.get("rx")) / width, float(shape.get("ry")) / height)
            samples.append((Path(member).name, archive.read(member), target))
    return samples


def write_split(samples: list[tuple[str, bytes, tuple[float, float, float, float]]], split: str, root: Path, image_size: int) -> None:
    """Write JPEG images and four-corner OBB labels for one split."""
    image_dir, label_dir = root / "images" / split, root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
    for index, (name, data, (cx, cy, rx, ry)) in enumerate(samples):
        image = Image.open(io.BytesIO(data)).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        stem = f"{split}_{index:06d}_{Path(name).stem}"
        image.save(image_dir / f"{stem}.jpg", quality=92)
        # why: CVAT ellipses may extend beyond the image; Ultralytics drops
        # invalid polygons, so clipping preserves those training examples.
        corners = tuple((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))) for x, y in ((cx-rx, cy-ry), (cx+rx, cy-ry), (cx+rx, cy+ry), (cx-rx, cy+ry)))
        values = " ".join(f"{value:.6f}" for point in corners for value in point)
        (label_dir / f"{stem}.txt").write_text(f"0 {values}\n")


def main() -> None:
    """Create train/validation OBB data without using any test archive."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()
    train_names = ["train_1.zip", "val_1.zip", "train_2.zip", "val_2.zip"]
    train: list[tuple[str, bytes, tuple[float, float, float, float]]] = []
    for zip_name in train_names:
        train.extend(read_samples(zip_name))
    random.Random(SEED).shuffle(train)
    split_at = max(1, int(len(train) * 0.90))
    write_split(train[:split_at], "train", args.output, args.image_size)
    write_split(train[split_at:], "val", args.output, args.image_size)
    yaml = f"path: {args.output.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: gauge\n"
    (args.output / "dataset.yaml").write_text(yaml)
    print("wrote", len(train[:split_at]), "train and", len(train[split_at:]), "val samples", flush=True)


if __name__ == "__main__":
    main()
