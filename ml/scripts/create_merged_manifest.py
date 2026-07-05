"""Create merged manifest: geometry labels + board captures.

Board captures are already 224x224 with coordinates in pixel space.
We set loose_crop = full image (0, 0, 224, 224) so the existing
pipeline treats them as pre-cropped gauge images.

Usage:
    python ml/scripts/create_merged_manifest.py
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET


BOARD_CAPTURES_CSV = Path(__file__).resolve().parent.parent / "data" / "board_captures_labeled_v2.csv"
BOARD_CAPTURES_4_ZIP = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "captured_images"
    / "clean_board_captures"
    / "board_captures_4.zip"
)
BOARD_CAPTURES_4_IMAGE_DIR = Path("ml/data/captured_images/clean_board_captures")
GEOMETRY_MANIFEST_CSV = Path(__file__).resolve().parent.parent / "data" / "geometry_reader_manifest_v2_clean.csv"
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "data" / "merged_geometry_board_manifest.csv"


def _angle_degrees_from_center_to_tip(center_x: float, center_y: float, tip_x: float, tip_y: float) -> float:
    """Return the gauge angle convention used by the board label CSVs."""
    dx = tip_x - center_x
    dy = tip_y - center_y
    angle = math.degrees(math.atan2(dy, dx))
    return (360.0 - angle) % 360.0


def _make_board_manifest_row(
    *,
    image_path: str,
    temperature_c: float,
    source_width: int,
    source_height: int,
    center_x: float,
    center_y: float,
    tip_x: float,
    tip_y: float,
    source_manifest: str,
    label_quality: str,
    quality_flag: str,
) -> dict[str, Any]:
    """Build one board-capture row in the merged-manifest schema."""
    angle_degrees = _angle_degrees_from_center_to_tip(center_x, center_y, tip_x, tip_y)
    center_tip_distance = ((tip_x - center_x) ** 2 + (tip_y - center_y) ** 2) ** 0.5
    return {
        "image_path": image_path,
        "temperature_c": temperature_c,
        "split": "train",
        "source_width": source_width,
        "source_height": source_height,
        "loose_crop_x1": 0,
        "loose_crop_y1": 0,
        "loose_crop_x2": source_width,
        "loose_crop_y2": source_height,
        "center_x_source": center_x,
        "center_y_source": center_y,
        "tip_x_source": tip_x,
        "tip_y_source": tip_y,
        "dial_radius_source": 80.0,
        "label_quality": label_quality,
        "source_manifest": source_manifest,
        "notes": "",
        "angle_degrees_from_labels": angle_degrees,
        "deterministic_temperature_c": temperature_c,
        "absolute_temperature_difference_c": 0.0,
        "center_tip_distance_pixels": center_tip_distance,
        "quality_flag": quality_flag,
    }


def _read_cvat_object_attribute(element: ET.Element, attribute_name: str) -> float | None:
    """Read a nested CVAT object attribute from an ellipse or point element."""
    for child in element.findall("attribute"):
        if child.attrib.get("name") == attribute_name and child.text is not None:
            return float(child.text.strip())
    return None


def load_board_capture_csv() -> list[dict[str, Any]]:
    """Load the legacy board capture CSV into merged-manifest rows."""
    samples: list[dict[str, Any]] = []
    with open(BOARD_CAPTURES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("image_path") or not row.get("center_x"):
                continue
            try:
                temp = float(row["temperature_c"])
                center_x = float(row["center_x"])
                center_y = float(row["center_y"])
                tip_x = float(row["tip_x"])
                tip_y = float(row["tip_y"])
                src_w = int(float(row.get("source_width", 224)))
                src_h = int(float(row.get("source_height", 224)))
            except (ValueError, KeyError):
                continue

            samples.append(
                _make_board_manifest_row(
                    image_path=row["image_path"],
                    temperature_c=temp,
                    source_width=src_w,
                    source_height=src_h,
                    center_x=center_x,
                    center_y=center_y,
                    tip_x=tip_x,
                    tip_y=tip_y,
                    source_manifest="board_captures_v2",
                    label_quality="manual",
                    quality_flag="clean",
                ),
            )

    return samples


def load_board_capture_zip() -> list[dict[str, Any]]:
    """Load the cold-end CVAT zip and convert it to merged-manifest rows."""
    if not BOARD_CAPTURES_4_ZIP.exists():
        return []

    samples: list[dict[str, Any]] = []
    with ZipFile(BOARD_CAPTURES_4_ZIP, "r") as zf:
        with zf.open("annotations.xml") as f:
            tree = ET.parse(f)

    root = tree.getroot()
    for image in root.findall("image"):
        file_name = image.attrib["name"]
        image_path = str(BOARD_CAPTURES_4_IMAGE_DIR / file_name)
        source_width = int(image.attrib["width"])
        source_height = int(image.attrib["height"])

        center_x = center_y = tip_x = tip_y = temperature_c = None
        for point in image.findall("points"):
            label = point.attrib.get("label")
            x_str, y_str = point.attrib["points"].split(",")
            if label == "temp_center":
                center_x = float(x_str)
                center_y = float(y_str)
            elif label == "temp_tip":
                tip_x = float(x_str)
                tip_y = float(y_str)

        for ellipse in image.findall("ellipse"):
            if ellipse.attrib.get("label") == "temp_dial":
                temperature_c = _read_cvat_object_attribute(ellipse, "temp_c")

        if None in (center_x, center_y, tip_x, tip_y, temperature_c):
            raise ValueError(f"Missing labels in {BOARD_CAPTURES_4_ZIP} for image {file_name}")

        samples.append(
            _make_board_manifest_row(
                image_path=image_path,
                temperature_c=temperature_c,
                source_width=source_width,
                source_height=source_height,
                center_x=center_x,
                center_y=center_y,
                tip_x=tip_x,
                tip_y=tip_y,
                source_manifest="board_captures_4.zip",
                label_quality="manual",
                quality_flag="clean",
            ),
        )

    return samples


def load_board_captures() -> list[dict[str, Any]]:
    """Load the board capture sources and prefer the newest cold-end labels."""
    samples_by_name: dict[str, dict[str, Any]] = {}
    for row in load_board_capture_csv():
        samples_by_name[Path(str(row["image_path"])).name] = row
    for row in load_board_capture_zip():
        samples_by_name[Path(str(row["image_path"])).name] = row
    return list(samples_by_name.values())


def load_geometry_samples() -> list[dict[str, Any]]:
    """Load geometry manifest rows as dicts."""
    samples: list[dict[str, Any]] = []
    with open(GEOMETRY_MANIFEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("image_path"):
                continue
            quality = row.get("quality_flag", "").strip()
            if quality == "exclude":
                continue
            samples.append(dict(row))
    return samples


def main() -> None:
    """Merge geometry + board capture manifests with stratified splits."""
    geometry = load_geometry_samples()
    board = load_board_captures()

    print(f"Geometry samples: {len(geometry)}")
    print(f"Board captures: {len(board)}")

    # Assign splits to board captures (15% val) after deduping and merging the new zip.
    rng = random.Random(42)
    rng.shuffle(board)
    n_val = max(1, int(len(board) * 0.15))
    for i, s in enumerate(board):
        s["split"] = "val" if i < n_val else "train"

    # Merge and dedupe by image path so the newer board_captures_4 rows win on overlap.
    merged_by_path: dict[str, dict[str, Any]] = {}
    for row in geometry:
        merged_by_path[str(row["image_path"])] = row
    for row in board:
        merged_by_path[str(row["image_path"])] = row
    merged = list(merged_by_path.values())
    rng.shuffle(merged)

    # Write
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path", "temperature_c", "split", "source_width", "source_height",
        "loose_crop_x1", "loose_crop_y1", "loose_crop_x2", "loose_crop_y2",
        "center_x_source", "center_y_source", "tip_x_source", "tip_y_source",
        "dial_radius_source", "label_quality", "source_manifest", "notes",
        "angle_degrees_from_labels", "deterministic_temperature_c",
        "absolute_temperature_difference_c", "center_tip_distance_pixels",
        "quality_flag",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in merged:
            writer.writerow({k: s.get(k, "") for k in fieldnames})

    n_train = sum(1 for s in merged if s.get("split") == "train")
    n_val = sum(1 for s in merged if s.get("split") == "val")
    print(f"Merged: {len(merged)} total ({n_train} train, {n_val} val)")
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
