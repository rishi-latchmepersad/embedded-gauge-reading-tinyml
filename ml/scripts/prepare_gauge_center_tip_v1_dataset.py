#!/usr/bin/env python3
"""Prepare ellipse-conditioned center/tip heatmap data for gauge_needle_v1.

The source archives contain a gauge-face ellipse, a center box, and one or
more Tip boxes.  The ellipse is used to make a square crop and an additional
binary mask channel.  When CVAT contains multiple Tip boxes, the endpoint
farthest from the center is the physical needle tip; the other endpoint is
normally the short tail of the needle.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class Shape:
    """Minimal geometric annotation extracted from a CVAT image entry."""

    label: str
    x: float
    y: float
    rx: float = 0.0
    ry: float = 0.0


def _shape_center(element: ET.Element) -> tuple[float, float]:
    """Return the center of a CVAT box or points shape."""
    if element.tag == "points":
        x, y = element.attrib["points"].split(",")[:2]
        return float(x), float(y)
    return (
        (float(element.attrib["xtl"]) + float(element.attrib["xbr"])) / 2.0,
        (float(element.attrib["ytl"]) + float(element.attrib["ybr"])) / 2.0,
    )


def _parse_image_entry(image: ET.Element) -> tuple[dict[str, str], list[Shape]]:
    """Parse one CVAT image and retain only labels needed by this model."""
    shapes: list[Shape] = []
    for element in image:
        label = element.attrib.get("label", "")
        if label == "GaugeFace" and element.tag == "ellipse":
            shapes.append(
                Shape(
                    label=label,
                    x=float(element.attrib["cx"]),
                    y=float(element.attrib["cy"]),
                    rx=float(element.attrib["rx"]),
                    ry=float(element.attrib["ry"]),
                )
            )
        elif label in {"Center", "Tip"} and element.tag in {"box", "points"}:
            x, y = _shape_center(element)
            shapes.append(Shape(label=label, x=x, y=y))
    return dict(image.attrib), shapes


def _choose_annotations(shapes: list[Shape]) -> tuple[Shape, Shape, Shape] | None:
    """Choose ellipse, center, and physical tip or return None if incomplete."""
    ellipses = [shape for shape in shapes if shape.label == "GaugeFace"]
    centers = [shape for shape in shapes if shape.label == "Center"]
    tips = [shape for shape in shapes if shape.label == "Tip"]
    if not ellipses or not centers or not tips:
        return None
    ellipse = ellipses[0]
    # why: duplicate center boxes are usually repeated exports of the same box;
    # nearest-to-ellipse is stable even when one annotation is an outlier.
    center = min(centers, key=lambda p: (p.x - ellipse.x) ** 2 + (p.y - ellipse.y) ** 2)
    # why: choose the outward endpoint so the decoded center->tip vector draws
    # the needle direction rather than its short counterweight tail.
    tip = max(tips, key=lambda p: (p.x - center.x) ** 2 + (p.y - center.y) ** 2)
    return ellipse, center, tip


def _gaussian(size: int, x: float, y: float, sigma: float = 3.0) -> np.ndarray:
    """Create one normalized Gaussian heatmap at pixel coordinate (x, y)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)


def _prepare_one(
    image_bytes: bytes,
    image_meta: dict[str, str],
    shapes: list[Shape],
    output_dir: Path,
    split: str,
    stem: str,
) -> dict[str, object] | None:
    """Crop one gauge, convert to grayscale, and write heatmap metadata."""
    chosen = _choose_annotations(shapes)
    if chosen is None:
        return None
    ellipse, center, tip = chosen
    with Image.open(__import__("io").BytesIO(image_bytes)) as source:
        gray = source.convert("L")
        width, height = gray.size
        side = max(2.0 * ellipse.rx, 2.0 * ellipse.ry) * 1.18
        left = ellipse.x - side / 2.0
        top = ellipse.y - side / 2.0
        # Pad outside-frame crops instead of clipping coordinates; this keeps
        # the ellipse-to-crop transform identical for every camera frame.
        pad = int(math.ceil(max(0.0, -left, -top, left + side - width, top + side - height)))
        if pad:
            canvas = Image.new("L", (width + 2 * pad, height + 2 * pad), 0)
            canvas.paste(gray, (pad, pad))
            gray = canvas
            left += pad
            top += pad
        crop = gray.crop((int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side))))
        crop = crop.resize((160, 160), Image.Resampling.BILINEAR)
        out_path = output_dir / "images" / split / f"{stem}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path)

    def norm(point: Shape) -> tuple[float, float]:
        """Transform a source point into normalized crop coordinates."""
        return ((point.x - left) / side, (point.y - top) / side)

    center_norm = norm(center)
    tip_norm = norm(tip)
    if not all(0.0 <= value <= 1.0 for value in (*center_norm, *tip_norm)):
        return None
    # why: 80 cells reduce the quantized tip localization step from 4 px to
    # 2 px in the 160 px crop while remaining within the SRAM budget.
    hm_size = 80
    center_xy = (center_norm[0] * (hm_size - 1), center_norm[1] * (hm_size - 1))
    tip_xy = (tip_norm[0] * (hm_size - 1), tip_norm[1] * (hm_size - 1))
    heatmaps = np.stack([_gaussian(hm_size, *center_xy), _gaussian(hm_size, *tip_xy)], axis=-1)
    hm_dir = output_dir / "heatmaps" / split
    hm_dir.mkdir(parents=True, exist_ok=True)
    np.save(hm_dir / f"{stem}.npy", heatmaps)
    return {
        "stem": stem,
        "image": str(Path("images") / split / f"{stem}.png"),
        "heatmap": str(Path("heatmaps") / split / f"{stem}.npy"),
        "center_xy_norm": list(center_norm),
        "tip_xy_norm": list(tip_norm),
        "ellipse": [ellipse.x, ellipse.y, ellipse.rx, ellipse.ry],
        "source_width": int(image_meta["width"]),
        "source_height": int(image_meta["height"]),
    }


def main() -> None:
    """Extract train/validation/test crops from the six labelled archives."""
    root = Path(__file__).resolve().parents[1]
    labelled = root / "data" / "labelled"
    output = root / "data" / "gauge_center_tip_v1_160_gray"
    archive_groups = {
        "train": [labelled / "train_1.zip", labelled / "train_2.zip"],
        "val": [labelled / "val_1.zip", labelled / "val_2.zip"],
        "test": [labelled / "test_1.zip", labelled / "test_2.zip"],
    }
    metadata: dict[str, object] = {"input_shape": [160, 160, 2], "heatmap_shape": [80, 80, 2], "splits": {}}
    for split, archives in archive_groups.items():
        rows: list[dict[str, object]] = []
        for archive_path in archives:
            with ZipFile(archive_path) as archive:
                root_xml = ET.fromstring(archive.read("annotations.xml"))
                members = set(archive.namelist())
                for image in tqdm(root_xml.findall("image"), desc=f"{split}:{archive_path.name}"):
                    meta, shapes = _parse_image_entry(image)
                    name = meta["name"]
                    if name not in members:
                        # Some archives store images beneath a directory.
                        name = next((item for item in members if item.endswith("/" + meta["name"])), name)
                    if name not in members:
                        continue
                    row = _prepare_one(archive.read(name), meta, shapes, output, split, Path(meta["name"]).stem)
                    if row is not None:
                        rows.append(row)
        metadata["splits"][split] = rows  # type: ignore[index]
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({split: len(rows) for split, rows in metadata["splits"].items()}, indent=2))  # type: ignore[union-attr]


if __name__ == "__main__":
    main()
