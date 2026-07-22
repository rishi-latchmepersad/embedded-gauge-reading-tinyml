"""Prepare the CVAT GaugeFace ellipses for the YOLO11n-OBB trainer.

The CVAT archives contain ellipse annotations, while the existing STM32N6
deployment path accepts YOLO OBB models.  This script converts each
``GaugeFace`` ellipse to the four corners of its tight axis-aligned ellipse
bounding box after applying the same square crop and resize used on-device.
"""

from __future__ import annotations

import argparse
import io
import math
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image
from tqdm import tqdm


SEED = 42
DEFAULT_SIZE = 128
CLASS_NAME = "gauge_face"


@dataclass(frozen=True)
class Ellipse:
    """Store one CVAT ellipse in source-image pixel coordinates."""

    cx: float
    cy: float
    rx: float
    ry: float


@dataclass(frozen=True)
class ImageRecord:
    """Store one image and its GaugeFace annotations from a CVAT archive."""

    archive: Path
    archive_index: int
    name: str
    width: int
    height: int
    ellipses: tuple[Ellipse, ...]


def _source_image_entry(archive: zipfile.ZipFile, name: str) -> zipfile.ZipInfo:
    """Find the image entry even when CVAT stores a task-relative name."""
    candidates = [name, f"images/{name}", Path(name).name, f"images/{Path(name).name}"]
    entries = {entry.filename.replace("\\", "/"): entry for entry in archive.infolist()}
    for candidate in candidates:
        if candidate in entries:
            return entries[candidate]
    basename = Path(name).name
    matches = [entry for entry in archive.infolist() if Path(entry.filename).name == basename]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not match CVAT image {name!r} in {archive.filename}")


def _read_records(archive_path: Path) -> list[ImageRecord]:
    """Read GaugeFace ellipses and image metadata from one CVAT archive."""
    with zipfile.ZipFile(archive_path) as archive:
        root = ElementTree.fromstring(archive.read("annotations.xml"))
        records: list[ImageRecord] = []
        for index, image in enumerate(root.findall("./image")):
            ellipses = tuple(
                Ellipse(
                    cx=float(node.attrib["cx"]),
                    cy=float(node.attrib["cy"]),
                    rx=float(node.attrib["rx"]),
                    ry=float(node.attrib["ry"]),
                )
                for node in image.findall("./ellipse")
                if node.attrib.get("label") == "GaugeFace"
            )
            if not ellipses:
                continue
            records.append(
                ImageRecord(
                    archive=archive_path,
                    archive_index=index,
                    name=image.attrib["name"],
                    width=int(image.attrib["width"]),
                    height=int(image.attrib["height"]),
                    ellipses=ellipses,
                )
            )
        return records


def _crop_resize(image: Image.Image, size: int) -> tuple[Image.Image, int, int, int]:
    """Apply a centered square crop and resize it to the model input size."""
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    cropped = image.crop((left, top, left + side, top + side))
    return cropped.resize((size, size), Image.Resampling.BILINEAR), left, top, side


def _ellipse_to_yolo_obb(
    ellipse: Ellipse, left: int, top: int, side: int, size: int
) -> list[float]:
    """Convert an ellipse to normalized YOLO OBB corners after crop/resize."""
    scale = size / side
    # CVAT GaugeFace ellipses have no rotation attribute, so their axes are
    # image-aligned; retaining all four corners preserves the radii exactly.
    points = (
        (ellipse.cx - ellipse.rx, ellipse.cy - ellipse.ry),
        (ellipse.cx + ellipse.rx, ellipse.cy - ellipse.ry),
        (ellipse.cx + ellipse.rx, ellipse.cy + ellipse.ry),
        (ellipse.cx - ellipse.rx, ellipse.cy + ellipse.ry),
    )
    values: list[float] = []
    for x, y in points:
        values.extend(((x - left) * scale / size, (y - top) * scale / size))
    return values


def _write_split(
    records: list[ImageRecord], split: str, output: Path, size: int
) -> int:
    """Extract, crop, and label one deterministic train/val/test split."""
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for record in tqdm(records, desc=split):
        with zipfile.ZipFile(record.archive) as archive:
            entry = _source_image_entry(archive, record.name)
            image = Image.open(io.BytesIO(archive.read(entry))).convert("RGB")
        cropped, left, top, side = _crop_resize(image, size)
        # Prefixing the archive and XML index prevents collisions across CVAT exports.
        stem = f"{record.archive.stem}_{record.archive_index:06d}_{Path(record.name).stem}"
        cropped.save(image_dir / f"{stem}.jpg", quality=95)
        lines = [
            "0 " + " ".join(f"{value:.6f}" for value in _ellipse_to_yolo_obb(e, left, top, side, size))
            for e in record.ellipses
        ]
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1
    return written


def main() -> None:
    """Build the versioned YOLO OBB dataset from the new CVAT archives."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "gauge_face_ellipse_v1")
    args = parser.parse_args()

    random.seed(SEED)
    labelled = Path(__file__).resolve().parents[1] / "data" / "labelled"
    split_archives = {
        "train": (labelled / "train_1.zip", labelled / "train_2.zip"),
        "val": (labelled / "val_1.zip", labelled / "val_2.zip"),
        "test": (labelled / "test_1.zip", labelled / "test_2.zip"),
    }
    for archive_group in split_archives.values():
        for archive in archive_group:
            if not archive.is_file():
                raise FileNotFoundError(archive)

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    counts: dict[str, int] = {}
    for split, archives in split_archives.items():
        records = [record for archive in archives for record in _read_records(archive)]
        counts[split] = _write_split(records, split, args.output, args.size)
    dataset_yaml = (
        f"path: {args.output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )
    (args.output / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    print(f"Prepared gauge_face_ellipse_v1: {counts}")


if __name__ == "__main__":
    main()
