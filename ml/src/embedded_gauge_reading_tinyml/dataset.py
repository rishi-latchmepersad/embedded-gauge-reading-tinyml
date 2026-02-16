"""
Dataset utilities for loading CVAT-labelled gauge images.
Reads CVAT "CVAT for images 1.1" exports (annotations.xml inside zips).
"""

from __future__ import annotations

# Standard library imports
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET


# Paths to datasets
ML_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ML_ROOT / "data"
LABELLED_DIR: Path = DATA_DIR / "labelled"
RAW_DIR: Path = DATA_DIR / "raw"


# Create a dataclass for each required label type.
@dataclass(frozen=True)  # frozen = true means this class is immutable
class PointLabel:
    """Single point label (x, y) with a semantic name."""

    x: float
    y: float
    label: str


@dataclass(frozen=True)
class EllipseLabel:
    """Ellipse label defined by center, radii, and rotation."""

    cx: float
    cy: float
    rx: float
    ry: float
    rotation: float
    label: str


# Create a dataclass for each required sample.
@dataclass(frozen=True)
class Sample:
    """One training sample consisting of image path + required labels."""

    image_path: Path
    dial: EllipseLabel
    center: PointLabel
    tip: PointLabel


def list_labelled_zips(labelled_dir: Path = LABELLED_DIR) -> list[Path]:
    """Return all zip files in the labelled directory (sorted)."""
    return sorted(labelled_dir.glob("*.zip"))


def _parse_point(points_attr: dict[str, str], label: str) -> PointLabel:
    """Convert a CVAT <points> attribute dict (from our labelling output) into a typed PointLabel object."""
    # CVAT point format is "x,y"
    x_str, y_str = points_attr["points"].split(",")
    return PointLabel(x=float(x_str), y=float(y_str), label=label)


def _parse_ellipse(ellipse_attr: dict[str, str], label: str) -> EllipseLabel:
    """Convert a CVAT <ellipse> attribute dict (from our labelling output) into a typed EllipseLabel object."""
    return EllipseLabel(
        cx=float(ellipse_attr["cx"]),
        cy=float(ellipse_attr["cy"]),
        rx=float(ellipse_attr["rx"]),
        ry=float(ellipse_attr["ry"]),
        rotation=float(ellipse_attr.get("rotation", "0")),
        label=label,
    )


def parse_cvat_zip(zip_path: Path, raw_dir: Path = RAW_DIR) -> list[Sample]:
    """Parse one CVAT zip batch export into a list of Samples."""
    samples: list[Sample] = []

    # Read annotations.xml directly from the zip (no extraction required).
    with ZipFile(zip_path, "r") as zf:
        with zf.open("annotations.xml") as f:
            tree = ET.parse(f)

    root = tree.getroot()
    for img in root.findall("image"):
        # Resolve image path.
        file_name = img.attrib["name"]
        image_path = raw_dir / file_name

        # Placeholders for required labels.
        dial: EllipseLabel | None = None
        center: PointLabel | None = None
        tip: PointLabel | None = None

        # Extract dial ellipse.
        for ellipse in img.findall("ellipse"):
            if ellipse.attrib.get("label") == "temp_dial":
                dial = _parse_ellipse(ellipse.attrib, "temp_dial")

        # Extract center and tip points.
        for points in img.findall("points"):
            label = points.attrib.get("label")
            if label == "temp_center":
                center = _parse_point(points.attrib, "temp_center")
            elif label == "temp_tip":
                tip = _parse_point(points.attrib, "temp_tip")

        # Enforce completeness of labelling
        if dial is None or center is None or tip is None:
            raise ValueError(f"Missing labels in {zip_path} for image {file_name}")

        samples.append(
            Sample(
                image_path=image_path,
                dial=dial,
                center=center,
                tip=tip,
            )
        )

    return samples


def load_dataset(
    labelled_dir: Path = LABELLED_DIR,
    raw_dir: Path = RAW_DIR,
) -> list[Sample]:
    """Load all CVAT zip exports and return a combined list of Samples."""
    all_samples: list[Sample] = []
    for zip_path in list_labelled_zips(labelled_dir):
        all_samples.extend(parse_cvat_zip(zip_path, raw_dir))
    return all_samples
