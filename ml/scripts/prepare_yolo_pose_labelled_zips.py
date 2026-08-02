#!/usr/bin/env python3
"""Prepare a YOLO pose dataset with the ellipse center as one keypoint."""

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
    """Read image bytes and normalized ellipse geometry from one CVAT ZIP."""
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
            samples.append((Path(member).name, archive.read(member), (float(shape.get("cx")) / width, float(shape.get("cy")) / height, float(shape.get("rx")) / width, float(shape.get("ry")) / height)))
    return samples


def write_split(samples: list[tuple[str, bytes, tuple[float, float, float, float]]], split: str, root: Path, image_size: int) -> None:
    """Write clipped AABBs plus ellipse-center keypoints for a split."""
    image_dir, label_dir = root / "images" / split, root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for index, (name, data, (cx, cy, rx, ry)) in enumerate(samples):
        image = Image.open(io.BytesIO(data)).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        stem = f"{split}_{index:06d}_{Path(name).stem}"
        image.save(image_dir / f"{stem}.jpg", quality=92)
        x0, y0, x1, y1 = max(0.0, cx - rx), max(0.0, cy - ry), min(1.0, cx + rx), min(1.0, cy + ry)
        box_cx, box_cy, box_w, box_h = (x0 + x1) / 2.0, (y0 + y1) / 2.0, max(1e-4, x1 - x0), max(1e-4, y1 - y0)
        # why: visibility remains 2 even when the ellipse itself is clipped;
        # the center is clamped only to satisfy YOLO's normalized label range.
        keypoint = min(1.0, max(0.0, cx)), min(1.0, max(0.0, cy)), 2
        (label_dir / f"{stem}.txt").write_text(f"0 {box_cx:.6f} {box_cy:.6f} {box_w:.6f} {box_h:.6f} {keypoint[0]:.6f} {keypoint[1]:.6f} {keypoint[2]}\n")


def main() -> None:
    """Create a leakage-safe pose dataset from the four training archives."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()
    samples: list[tuple[str, bytes, tuple[float, float, float, float]]] = []
    for zip_name in ("train_1.zip", "val_1.zip", "train_2.zip", "val_2.zip"):
        samples.extend(read_samples(zip_name))
    random.Random(SEED).shuffle(samples)
    split_at = max(1, int(len(samples) * 0.90))
    write_split(samples[:split_at], "train", args.output, args.image_size)
    write_split(samples[split_at:], "val", args.output, args.image_size)
    yaml = f"path: {args.output.resolve()}\ntrain: images/train\nval: images/val\nkpt_shape: [1, 3]\nnames:\n  0: gauge\n"
    (args.output / "dataset.yaml").write_text(yaml)
    print(f"wrote {split_at} train and {len(samples) - split_at} val samples", flush=True)


if __name__ == "__main__":
    main()
