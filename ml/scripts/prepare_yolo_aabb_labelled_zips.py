#!/usr/bin/env python3
"""Prepare a leakage-safe axis-aligned gauge-face detector dataset."""

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
    """Read image bytes and normalized ellipse center/radii from one CVAT ZIP."""
    samples: list[tuple[str, bytes, tuple[float, float, float, float]]] = []
    with zipfile.ZipFile(LABELLED / zip_name) as archive:
        members = {Path(member).name: member for member in archive.namelist()}
        root = ET.fromstring(archive.read("annotations.xml"))
        for node in root.findall("image"):
            shape = next(
                (item for item in node.findall("ellipse") if item.get("label") in {"GaugeFace", "temp_dial"}),
                None,
            )
            member = members.get(Path(node.get("name", "")).name)
            if shape is None or member is None:
                continue
            width = float(node.get("width", 640))
            height = float(node.get("height", 640))
            target = (
                float(shape.get("cx")) / width,
                float(shape.get("cy")) / height,
                float(shape.get("rx")) / width,
                float(shape.get("ry")) / height,
            )
            samples.append((Path(member).name, archive.read(member), target))
    return samples


def write_split(
    samples: list[tuple[str, bytes, tuple[float, float, float, float]]],
    split: str,
    root: Path,
    image_size: int,
) -> None:
    """Write images and clipped normalized AABB labels for a split."""
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for index, (name, data, (cx, cy, rx, ry)) in enumerate(samples):
        image = Image.open(io.BytesIO(data)).convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        stem = f"{split}_{index:06d}_{Path(name).stem}"
        image.save(image_dir / f"{stem}.jpg", quality=92)
        # why: ellipse edges can lie outside the frame; clipping creates a valid
        # detector target instead of silently discarding the training sample.
        x0, y0 = max(0.0, cx - rx), max(0.0, cy - ry)
        x1, y1 = min(1.0, cx + rx), min(1.0, cy + ry)
        box_cx, box_cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        box_w, box_h = max(1e-4, x1 - x0), max(1e-4, y1 - y0)
        (label_dir / f"{stem}.txt").write_text(
            f"0 {box_cx:.6f} {box_cy:.6f} {box_w:.6f} {box_h:.6f}\n"
        )


def main() -> None:
    """Create train/validation AABB data without reading any test archive."""
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
    yaml = f"path: {args.output.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: gauge\n"
    (args.output / "dataset.yaml").write_text(yaml)
    print(f"wrote {split_at} train and {len(samples) - split_at} val samples", flush=True)


if __name__ == "__main__":
    main()
