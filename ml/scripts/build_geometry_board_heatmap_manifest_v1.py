"""Build a combined geometry + board heatmap manifest with a temporal board holdout.

The existing heatmap trainers already know how to read geometry manifests, but the
board center/tip labels live in separate CSVs. This helper merges them into one
geometry-style manifest so the trainer can consume the real board labels instead
of nearest-temperature stand-ins. The newest board captures are held out for test
so we can check generalization on the most recent captures.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from datetime import datetime
import re
from pathlib import Path
from typing import Any, Final

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
)


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_GEOMETRY_MANIFEST: Final[Path] = REPO_ROOT / "ml" / "data" / "geometry_reader_manifest_v2_clean.csv"
DEFAULT_BOARD_LABELS: Final[Path] = REPO_ROOT / "ml" / "data" / "board_center_tip_labels.csv"
DEFAULT_BOARD_REVIEWED_BOXES: Final[Path] = REPO_ROOT / "ml" / "data" / "board_center_tip_reviewed_boxes_labels.csv"
DEFAULT_OUTPUT_MANIFEST: Final[Path] = REPO_ROOT / "ml" / "data" / "geometry_board_heatmap_manifest_v1.csv"
DEFAULT_SUMMARY_PATH: Final[Path] = REPO_ROOT / "tmp" / "geometry_board_heatmap_manifest_v1_summary.json"


@dataclass(frozen=True, slots=True)
class BoardRow:
    """One board capture row with the metadata needed by the heatmap trainer."""

    image_path: str
    source_width: int
    source_height: int
    center_x_source: float
    center_y_source: float
    tip_x_source: float
    tip_y_source: float
    temperature_c: float
    label_quality: str
    quality_flag: str
    notes: str
    label_source: str
    origin_manifest: str
    angle_degrees_from_labels: float
    deterministic_temperature_c: float
    absolute_temperature_difference_c: float
    center_tip_distance_pixels: float
    dial_radius_source: float
    loose_crop_x1: float
    loose_crop_y1: float
    loose_crop_x2: float
    loose_crop_y2: float
    split: str = "train"

    def as_manifest_row(self) -> dict[str, str]:
        """Render the board row in the same CSV shape as the geometry manifest."""

        return {
            "image_path": self.image_path,
            "temperature_c": f"{self.temperature_c:.4f}",
            "split": self.split,
            "source_width": str(self.source_width),
            "source_height": str(self.source_height),
            "loose_crop_x1": f"{self.loose_crop_x1:.4f}",
            "loose_crop_y1": f"{self.loose_crop_y1:.4f}",
            "loose_crop_x2": f"{self.loose_crop_x2:.4f}",
            "loose_crop_y2": f"{self.loose_crop_y2:.4f}",
            "center_x_source": f"{self.center_x_source:.4f}",
            "center_y_source": f"{self.center_y_source:.4f}",
            "tip_x_source": f"{self.tip_x_source:.4f}",
            "tip_y_source": f"{self.tip_y_source:.4f}",
            "dial_radius_source": f"{self.dial_radius_source:.4f}",
            "label_quality": self.label_quality,
            "source_manifest": self.origin_manifest,
            "notes": self.notes,
            "angle_degrees_from_labels": f"{self.angle_degrees_from_labels:.4f}",
            "deterministic_temperature_c": f"{self.deterministic_temperature_c:.4f}",
            "absolute_temperature_difference_c": f"{self.absolute_temperature_difference_c:.4f}",
            "center_tip_distance_pixels": f"{self.center_tip_distance_pixels:.4f}",
            "quality_flag": self.quality_flag,
        }


def _parse_args() -> argparse.Namespace:
    """Parse manifest builder flags."""

    parser = argparse.ArgumentParser(description="Build a combined heatmap manifest.")
    parser.add_argument("--geometry-manifest", type=Path, default=DEFAULT_GEOMETRY_MANIFEST)
    parser.add_argument("--board-labels", type=Path, default=DEFAULT_BOARD_LABELS)
    parser.add_argument("--board-reviewed-boxes", type=Path, default=DEFAULT_BOARD_REVIEWED_BOXES)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--board-test-count", type=int, default=10)
    parser.add_argument("--board-val-fraction", type=float, default=0.15)
    return parser.parse_args()


def _resolve_image_path(raw_path: str) -> Path:
    """Resolve a manifest path relative to the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "ml":
        return REPO_ROOT / path
    return REPO_ROOT / "ml" / path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of string dictionaries."""

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_geometry_rows(manifest_path: Path) -> list[dict[str, str]]:
    """Load geometry rows and drop only the explicit exclude entries."""

    rows: list[dict[str, str]] = []
    for row in _read_csv_rows(manifest_path):
        quality = str(row.get("quality_flag", "")).strip().lower()
        if quality == "exclude":
            continue
        row = dict(row)
        row["quality_flag"] = "clean"
        rows.append(row)
    return rows


def _load_board_rows(label_path: Path) -> list[BoardRow]:
    """Load board center/tip labels and normalize them for the heatmap manifest."""

    rows: list[BoardRow] = []
    for row in _read_csv_rows(label_path):
        image_path = str(row.get("image_path", "")).strip()
        if not image_path:
            continue
        try:
            source_width = int(float(row["source_width"]))
            source_height = int(float(row["source_height"]))
            center_x = float(row["center_x_source"])
            center_y = float(row["center_y_source"])
            tip_x = float(row["tip_x_source"])
            tip_y = float(row["tip_y_source"])
            crop_x1 = float(row.get("crop_x_min", 0.0))
            crop_y1 = float(row.get("crop_y_min", 0.0))
            crop_x2 = float(row.get("crop_x_max", float(source_width)))
            crop_y2 = float(row.get("crop_y_max", float(source_height)))
        except (TypeError, ValueError, KeyError):
            continue

        # Use the label-derived angle when it exists, but recompute it from the
        # labeled center/tip so the manifest is self-consistent.
        raw_angle = str(row.get("true_angle_degrees", "")).strip()
        angle_from_labels = (
            float(raw_angle)
            if raw_angle
            else angle_degrees_from_center_to_tip(center_x, center_y, tip_x, tip_y)
        )
        temperature_c = celsius_from_inner_dial_angle_degrees(angle_from_labels)
        center_tip_distance = float(((tip_x - center_x) ** 2 + (tip_y - center_y) ** 2) ** 0.5)
        dial_radius_source = float(max(crop_x2 - crop_x1, crop_y2 - crop_y1) * 0.5)
        origin_manifest = str(row.get("origin_manifest", label_path.name))
        label_source = str(row.get("label_source", "manual_gui"))
        notes = str(row.get("notes", ""))
        rows.append(
            BoardRow(
                image_path=image_path,
                source_width=source_width,
                source_height=source_height,
                center_x_source=center_x,
                center_y_source=center_y,
                tip_x_source=tip_x,
                tip_y_source=tip_y,
                temperature_c=temperature_c,
                label_quality="manual",
                quality_flag="clean",
                notes=notes,
                label_source=label_source,
                origin_manifest=origin_manifest,
                angle_degrees_from_labels=angle_from_labels,
                deterministic_temperature_c=temperature_c,
                absolute_temperature_difference_c=0.0,
                center_tip_distance_pixels=center_tip_distance,
                dial_radius_source=dial_radius_source,
                loose_crop_x1=crop_x1,
                loose_crop_y1=crop_y1,
                loose_crop_x2=crop_x2,
                loose_crop_y2=crop_y2,
            )
        )
    return rows


def _board_mtime(row: BoardRow) -> float:
    """Return a recency score for a board capture.

    We prefer the timestamp embedded in the capture filename because the board
    images are often copied around after capture, which can scramble filesystem
    mtimes.
    """

    image_path = _resolve_image_path(row.image_path)
    stem = image_path.stem
    match = re.search(r"capture_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", stem)
    if match:
        try:
            return datetime.strptime(
                f"{match.group(1)} {match.group(2).replace('-', ':')}",
                "%Y-%m-%d %H:%M:%S",
            ).timestamp()
        except ValueError:
            pass
    if image_path.exists():
        return image_path.stat().st_mtime
    return 0.0


def _assign_board_splits(
    board_rows: list[BoardRow],
    *,
    test_count: int,
    val_fraction: float,
) -> list[BoardRow]:
    """Assign train/val/test splits to board rows using temporal ordering."""

    if not board_rows:
        return []

    ordered = sorted(board_rows, key=_board_mtime)
    total = len(ordered)
    test_count = max(1, min(test_count, total))
    remaining = total - test_count
    val_count = max(1, int(round(remaining * val_fraction))) if remaining > 0 else 0
    train_count = max(0, remaining - val_count)

    assigned: list[BoardRow] = []
    for index, row in enumerate(ordered):
        if index < train_count:
            split = "train"
        elif index < train_count + val_count:
            split = "val"
        else:
            split = "test"
        assigned.append(replace(row, split=split))
    return assigned


def _dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop duplicate image paths while keeping the first occurrence."""

    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for row in rows:
        image_path = str(row.get("image_path", "")).strip()
        if not image_path or image_path in seen:
            continue
        seen.add(image_path)
        deduped.append(row)
    return deduped


def main() -> None:
    """Build and write the combined manifest."""

    args = _parse_args()
    geometry_rows = _load_geometry_rows(args.geometry_manifest)
    board_rows = _load_board_rows(args.board_labels) + _load_board_rows(args.board_reviewed_boxes)
    board_rows = _assign_board_splits(
        board_rows,
        test_count=int(args.board_test_count),
        val_fraction=float(args.board_val_fraction),
    )

    combined_rows: list[dict[str, str]] = []
    combined_rows.extend(geometry_rows)
    combined_rows.extend(row.as_manifest_row() for row in board_rows)
    combined_rows = _dedupe_rows(combined_rows)

    fieldnames = [
        "image_path",
        "temperature_c",
        "split",
        "source_width",
        "source_height",
        "loose_crop_x1",
        "loose_crop_y1",
        "loose_crop_x2",
        "loose_crop_y2",
        "center_x_source",
        "center_y_source",
        "tip_x_source",
        "tip_y_source",
        "dial_radius_source",
        "label_quality",
        "source_manifest",
        "notes",
        "angle_degrees_from_labels",
        "deterministic_temperature_c",
        "absolute_temperature_difference_c",
        "center_tip_distance_pixels",
        "quality_flag",
    ]

    # Keep the original geometry ordering, then append board captures in split order.
    geometry_only = [row for row in combined_rows if "captured_images" not in row["image_path"]]
    board_only = [row for row in combined_rows if "captured_images" in row["image_path"]]
    board_only.sort(key=lambda row: (row["split"], row["source_manifest"], row["image_path"]))
    final_rows = geometry_only + board_only

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in final_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    counts: dict[str, int] = {}
    for row in final_rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1

    board_test_rows = [row for row in board_only if row["split"] == "test"]
    summary: dict[str, Any] = {
        "geometry_manifest": str(args.geometry_manifest),
        "board_labels": str(args.board_labels),
        "board_reviewed_boxes": str(args.board_reviewed_boxes),
        "output_manifest": str(args.output_manifest),
        "total_rows": len(final_rows),
        "split_counts": counts,
        "board_rows": len(board_only),
        "board_test_count": len(board_test_rows),
        "board_test_images": [row["image_path"] for row in board_test_rows],
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
