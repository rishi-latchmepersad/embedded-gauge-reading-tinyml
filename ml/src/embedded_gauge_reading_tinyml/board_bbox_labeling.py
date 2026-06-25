"""Helpers for quickly labeling gauge bounding boxes on board captures.

The quick annotator writes a tiny CSV that can be merged into the grouped
manifest builder as an extra source. That keeps the manual labeling loop fast
while preserving compatibility with the OBB training pipeline.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from embedded_gauge_reading_tinyml.capture_labeling import (
    resolve_absolute_image_path,
    to_repo_relative_path,
)

ML_ROOT: Path = Path(__file__).resolve().parents[2]
REPO_ROOT: Path = ML_ROOT.parent
DEFAULT_MANIFEST_PATH: Path = REPO_ROOT / "tmp" / "board_bbox_uncropped_manifest.csv"
DEFAULT_OUTPUT_CSV: Path = REPO_ROOT / "tmp" / "board_bbox_labels.csv"


@dataclass(frozen=True, slots=True)
class BoardBBoxRecord:
    """One labeled board image with an axis-aligned gauge bounding box."""

    image_path: Path
    source_width: int
    source_height: int
    crop_x_min: float | None = None
    crop_y_min: float | None = None
    crop_x_max: float | None = None
    crop_y_max: float | None = None
    quality_flag: str = "review"
    label_source: str = "manual_gui"
    notes: str = ""
    origin_manifest: str = ""

    def with_box(
        self,
        *,
        crop_x_min: float | None,
        crop_y_min: float | None,
        crop_x_max: float | None,
        crop_y_max: float | None,
    ) -> BoardBBoxRecord:
        """Return a copy with updated bounding-box coordinates."""

        return replace(
            self,
            crop_x_min=crop_x_min,
            crop_y_min=crop_y_min,
            crop_x_max=crop_x_max,
            crop_y_max=crop_y_max,
        )

    def with_quality_flag(self, quality_flag: str) -> BoardBBoxRecord:
        """Return a copy with an updated quality flag."""

        return replace(self, quality_flag=quality_flag.strip() or self.quality_flag)

    @property
    def has_box(self) -> bool:
        """Return ``True`` when all four box coordinates are present."""

        return (
            self.crop_x_min is not None
            and self.crop_y_min is not None
            and self.crop_x_max is not None
            and self.crop_y_max is not None
        )

    def to_csv_row(self) -> dict[str, str]:
        """Serialize the record into the CSV schema used by the labeler."""

        return {
            "image_path": self.image_path.as_posix(),
            "source_width": str(int(self.source_width)),
            "source_height": str(int(self.source_height)),
            "crop_x_min": _format_optional_float(self.crop_x_min),
            "crop_y_min": _format_optional_float(self.crop_y_min),
            "crop_x_max": _format_optional_float(self.crop_x_max),
            "crop_y_max": _format_optional_float(self.crop_y_max),
            "quality_flag": self.quality_flag.strip() or "review",
            "label_source": self.label_source.strip() or "manual_gui",
            "notes": self.notes.strip(),
            "origin_manifest": self.origin_manifest.strip(),
        }


@dataclass(frozen=True, slots=True)
class BoardBBoxCandidate:
    """One board image that should be reviewed in the GUI."""

    image_path: Path
    source_width: int
    source_height: int
    origin_manifest: str
    seed_record: BoardBBoxRecord | None = None


def _format_optional_float(value: float | None) -> str:
    """Format a possibly-missing float for CSV output."""

    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _parse_optional_float(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    """Return the first finite float found under the requested keys."""

    for key in keys:
        value = row.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return float(parsed)
    return None


def _parse_optional_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    """Return the first non-empty text value found under the requested keys."""

    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _row_to_record(row: Mapping[str, Any]) -> BoardBBoxRecord:
    """Convert one CSV row into a ``BoardBBoxRecord``."""

    image_path = resolve_absolute_image_path(row["image_path"])
    source_width = int(round(_parse_optional_float(row, ("source_width",)) or 0.0))
    source_height = int(round(_parse_optional_float(row, ("source_height",)) or 0.0))
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Invalid source size for {image_path}: {source_width}x{source_height}")

    return BoardBBoxRecord(
        image_path=to_repo_relative_path(image_path),
        source_width=source_width,
        source_height=source_height,
        crop_x_min=_parse_optional_float(row, ("crop_x_min",)),
        crop_y_min=_parse_optional_float(row, ("crop_y_min",)),
        crop_x_max=_parse_optional_float(row, ("crop_x_max",)),
        crop_y_max=_parse_optional_float(row, ("crop_y_max",)),
        quality_flag=_parse_optional_text(row, ("quality_flag",)) or "review",
        label_source=_parse_optional_text(row, ("label_source",)) or "manual_gui",
        notes=_parse_optional_text(row, ("notes",)),
        origin_manifest=_parse_optional_text(row, ("origin_manifest",)),
    )


def load_board_bbox_records(csv_path: Path) -> dict[str, BoardBBoxRecord]:
    """Load an existing bbox CSV into a lookup table keyed by image path."""

    if not csv_path.exists():
        return {}

    records: dict[str, BoardBBoxRecord] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "image_path" not in reader.fieldnames:
            raise ValueError(f"{csv_path} does not look like a bbox label CSV.")
        for row in reader:
            if not row.get("image_path"):
                continue
            record = _row_to_record(row)
            records[record.image_path.as_posix()] = record
    return records


def write_board_bbox_records(csv_path: Path, records: Sequence[BoardBBoxRecord]) -> None:
    """Write a bbox CSV in a stable column order."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "source_width",
        "source_height",
        "crop_x_min",
        "crop_y_min",
        "crop_x_max",
        "crop_y_max",
        "quality_flag",
        "label_source",
        "notes",
        "origin_manifest",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_row())


def _load_image_size(image_path: Path) -> tuple[int, int]:
    """Read image dimensions for one board capture."""

    from PIL import Image

    absolute_path = resolve_absolute_image_path(image_path)
    try:
        with Image.open(absolute_path) as image:
            width, height = image.size
        return int(width), int(height)
    except Exception:
        suffix = absolute_path.suffix.lower()
        file_size = absolute_path.stat().st_size
        if suffix == ".yuv422":
            inferred_pixels = file_size / 2.0
            inferred_dim = int(round(math.sqrt(inferred_pixels)))
            if inferred_dim > 0 and inferred_dim * inferred_dim * 2 == file_size:
                return inferred_dim, inferred_dim
        if suffix in {".raw", ".raw16"}:
            inferred_dim = int(round(math.sqrt(file_size)))
            if inferred_dim > 0 and inferred_dim * inferred_dim == file_size:
                return inferred_dim, inferred_dim
            if file_size % 2 == 0:
                inferred_pixels = file_size / 2.0
                inferred_dim = int(round(math.sqrt(inferred_pixels)))
                if inferred_dim > 0 and inferred_dim * inferred_dim * 2 == file_size:
                    return inferred_dim, inferred_dim
        raise


def load_board_bbox_candidates(
    manifest_path: Path,
    *,
    image_column: str = "image_path",
    limit: int | None = 50,
) -> list[BoardBBoxCandidate]:
    """Load the board images that should be labeled."""

    candidates: list[BoardBBoxCandidate] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or image_column not in reader.fieldnames:
            raise ValueError(f"{manifest_path} does not contain an {image_column!r} column.")
        for row in reader:
            raw_image_path = str(row.get(image_column, "")).strip()
            if not raw_image_path or raw_image_path.startswith("#"):
                continue
            image_path = to_repo_relative_path(raw_image_path)
            source_width, source_height = _load_image_size(image_path)
            candidates.append(
                BoardBBoxCandidate(
                    image_path=image_path,
                    source_width=source_width,
                    source_height=source_height,
                    origin_manifest=manifest_path.as_posix(),
                )
            )
            if limit is not None and len(candidates) >= limit:
                break
    return candidates
