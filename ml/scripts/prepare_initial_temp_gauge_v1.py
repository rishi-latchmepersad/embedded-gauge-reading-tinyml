#!/usr/bin/env python3
"""Prepare the LittleGood temperature-gauge adaptation dataset.

The CVAT exports mix two storage styles: board-capture archives contain their
image files, while the high-resolution ``gauge_1`` archives contain only XML
and refer to matching files in ``ml/data/raw``.  This script resolves both
styles and emits the same 640-pixel ellipse and 160-pixel center/tip contracts
used by the deployment models.
"""

from __future__ import annotations

import io
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


def _point(element: ET.Element) -> tuple[float, float]:
    """Return the center of a CVAT points or box annotation."""
    if element.tag == "points":
        x, y = element.attrib["points"].split(",")[:2]
        return float(x), float(y)
    return ((float(element.attrib["xtl"]) + float(element.attrib["xbr"])) / 2.0,
            (float(element.attrib["ytl"]) + float(element.attrib["ybr"])) / 2.0)


def _source_bytes(archive: ZipFile, name: str, raw_dir: Path, captured_dirs: tuple[Path, ...]) -> bytes:
    """Resolve an image from its archive or the repository's raw-image stores."""
    basename = Path(name).name
    for member in archive.namelist():
        if Path(member).name == basename:
            return archive.read(member)
    raw_candidate = raw_dir / basename
    if raw_candidate.is_file():
        return raw_candidate.read_bytes()
    for captured_dir in captured_dirs:
        candidate = captured_dir / basename
        if candidate.is_file():
            return candidate.read_bytes()
    raise FileNotFoundError(f"Could not resolve labelled image {name!r}")


def _split_for_archive(name: str) -> str:
    """Assign source archives to train/validation/test without frame leakage."""
    if name.startswith("gauge_1_batch_"):
        number = int(name.split("_batch_")[1].split(".")[0])
        return "train" if number <= 5 else "val" if number == 6 else "test"
    if name.startswith("board_captures_"):
        number = int(name.split("_")[-1].split(".")[0])
        return "train" if number == 1 else "val" if number == 2 else "test"
    raise ValueError(f"Unexpected archive name: {name}")


def _gaussian(size: int, x: float, y: float, sigma: float = 3.0) -> np.ndarray:
    """Create one center/tip Gaussian target at heatmap pixel coordinates."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)


def main() -> None:
    """Extract LittleGood samples into ellipse and center/tip model layouts."""
    repo = Path(__file__).resolve().parents[1]
    source_dir = repo / "data" / "labelled" / "initial_temp_gauge"
    raw_dir = repo / "data" / "raw"
    captured_dirs = (
        repo / "data" / "captured_images",
        repo / "data" / "captured_images" / "clean_board_captures",
        repo / "data" / "captured_images" / "clean_board_captures_extracted",
    )
    output = repo / "data" / "initial_temp_gauge_v1"
    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "val", "test"):
        (output / "ellipse" / "images" / split).mkdir(parents=True)
        (output / "ellipse" / "labels" / split).mkdir(parents=True)
        (output / "center_tip" / "images" / split).mkdir(parents=True)
        (output / "center_tip" / "heatmaps" / split).mkdir(parents=True)
    metadata: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}

    for archive_path in sorted(source_dir.glob("*.zip")):
        split = _split_for_archive(archive_path.name)
        with ZipFile(archive_path) as archive:
            root = ET.fromstring(archive.read("annotations.xml"))
            for index, image_node in enumerate(root.findall("./image")):
                ellipse_node = next((e for e in image_node if e.attrib.get("label") == "temp_dial" and e.tag == "ellipse"), None)
                center_node = next((e for e in image_node if e.attrib.get("label") == "temp_center"), None)
                tip_node = next((e for e in image_node if e.attrib.get("label") == "temp_tip"), None)
                if ellipse_node is None or center_node is None or tip_node is None:
                    continue
                image_name = image_node.attrib["name"]
                image_bytes = _source_bytes(archive, image_name, raw_dir, captured_dirs)
                with Image.open(io.BytesIO(image_bytes)) as source:
                    gray = source.convert("L")
                    width, height = gray.size
                    side = min(width, height)
                    left = (width - side) / 2.0
                    top = (height - side) / 2.0
                    ellipse = np.array([float(ellipse_node.attrib["cx"]), float(ellipse_node.attrib["cy"]), float(ellipse_node.attrib["rx"]), float(ellipse_node.attrib["ry"])], dtype=np.float32)
                    center = np.array(_point(center_node), dtype=np.float32)
                    tip = np.array(_point(tip_node), dtype=np.float32)
                    scale = 640.0 / side
                    ellipse_norm = np.array([(ellipse[0] - left) / side, (ellipse[1] - top) / side, ellipse[2] / side, ellipse[3] / side], dtype=np.float32)
                    ellipse_norm = np.clip(ellipse_norm, 0.0, 1.0)
                    stem = f"{archive_path.stem}_{index:06d}_{Path(image_name).stem}"
                    ellipse_image = gray.crop((int(left), int(top), int(left + side), int(top + side))).resize((640, 640), Image.Resampling.BILINEAR)
                    ellipse_image.save(output / "ellipse" / "images" / split / f"{stem}.png", compress_level=1)
                    corners = [(ellipse_norm[0] - ellipse_norm[2], ellipse_norm[1] - ellipse_norm[3]), (ellipse_norm[0] + ellipse_norm[2], ellipse_norm[1] + ellipse_norm[3])]
                    values = [max(0.0, min(1.0, value)) for pair in corners for value in pair]
                    (output / "ellipse" / "labels" / split / f"{stem}.txt").write_text("0 " + " ".join(f"{value:.6f}" for value in [values[0], values[1], values[2], values[1], values[2], values[3], values[0], values[3]]) + "\n", encoding="utf-8")

                    crop_side = max(2.0 * float(ellipse[2]), 2.0 * float(ellipse[3])) * 1.18
                    crop_left = float(ellipse[0] - crop_side / 2.0)
                    crop_top = float(ellipse[1] - crop_side / 2.0)
                    pad = int(math.ceil(max(0.0, -crop_left, -crop_top, crop_left + crop_side - width, crop_top + crop_side - height)))
                    if pad:
                        padded = Image.new("L", (width + 2 * pad, height + 2 * pad), 0)
                        padded.paste(gray, (pad, pad))
                        gray = padded
                        crop_left += pad
                        crop_top += pad
                        center += pad
                        tip += pad
                        ellipse[0:2] += pad
                    center_norm = (center - np.array([crop_left, crop_top])) / crop_side
                    tip_norm = (tip - np.array([crop_left, crop_top])) / crop_side
                    if not np.all(np.concatenate((center_norm, tip_norm)) >= 0.0) or not np.all(np.concatenate((center_norm, tip_norm)) <= 1.0):
                        continue
                    crop = gray.crop((int(round(crop_left)), int(round(crop_top)), int(round(crop_left + crop_side)), int(round(crop_top + crop_side)))).resize((160, 160), Image.Resampling.BILINEAR)
                    crop.save(output / "center_tip" / "images" / split / f"{stem}.png", compress_level=1)
                    # why: 80x80 gives the tip decoder a 2 px crop quantization
                    # step instead of the previous 4 px step.
                    center_hm = center_norm * 79.0
                    tip_hm = tip_norm * 79.0
                    heatmaps = np.stack([_gaussian(80, *center_hm), _gaussian(80, *tip_hm)], axis=-1)
                    np.save(output / "center_tip" / "heatmaps" / split / f"{stem}.npy", heatmaps)
                    metadata[split].append({"stem": stem, "image": f"images/{split}/{stem}.png", "heatmap": f"heatmaps/{split}/{stem}.npy", "center_xy_norm": center_norm.tolist(), "tip_xy_norm": tip_norm.tolist(), "ellipse": ellipse.tolist()})
    (output / "center_tip" / "metadata.json").write_text(json.dumps({"input_shape": [160, 160, 2], "heatmap_shape": [40, 40, 2], "splits": metadata}, indent=2), encoding="utf-8")
    print(json.dumps({split: len(rows) for split, rows in metadata.items()}, indent=2))


if __name__ == "__main__":
    main()
