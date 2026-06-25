#!/usr/bin/env python3
"""Merge reviewed board bbox CSV labels into a grouped OBB training manifest.

The board labeler writes a compact CSV, while the OBB trainer expects the
grouped JSON manifest format used by the other geometry pipelines. This helper
bridges that gap by appending the reviewed board rows as `reviewed_geometry`
entries while preserving the existing grouped samples.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_GROUPED_MANIFEST: Path = REPO_ROOT / "tmp" / "labelled_captured_images_board_bbox.json"
DEFAULT_BOARD_LABELS_CSV: Path = REPO_ROOT / "tmp" / "board_bbox_labels.csv"
DEFAULT_OUTPUT_MANIFEST: Path = REPO_ROOT / "tmp" / "labelled_captured_images_board_bbox_plus_board_reviews.json"
SOURCE_KIND: str = "reviewed_geometry"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the merge helper."""

    parser = argparse.ArgumentParser(description="Merge board bbox CSV labels into the grouped JSON manifest.")
    parser.add_argument("--grouped-manifest", type=Path, default=DEFAULT_GROUPED_MANIFEST)
    parser.add_argument("--board-labels-csv", type=Path, default=DEFAULT_BOARD_LABELS_CSV)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    return parser.parse_args()


def _read_grouped_manifest(path: Path) -> dict[str, Any]:
    """Read an existing grouped manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if "images" not in payload:
        raise ValueError(f"{path} does not look like a grouped manifest.")
    return payload


def _read_board_rows(path: Path) -> list[dict[str, str]]:
    """Load reviewed board rows from the manual bbox CSV."""

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a header row.")
        required = {
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
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        for row in reader:
            if str(row.get("quality_flag", "")).strip() != "review":
                continue
            rows.append({key: str(value) if value is not None else "" for key, value in row.items()})
    return rows


def _existing_image_paths(images: list[dict[str, Any]]) -> set[str]:
    """Return the set of already-present image paths."""

    return {str(entry.get("image_path", "")).strip() for entry in images}


def _board_entry(row: dict[str, str], *, source_row_index: int) -> dict[str, Any]:
    """Convert one reviewed board CSV row into a grouped-manifest entry."""

    image_path = row["image_path"]
    source_row = OrderedDict(
        [
            ("image_path", row["image_path"]),
            ("source_width", int(float(row["source_width"]))),
            ("source_height", int(float(row["source_height"]))),
            ("crop_x_min", float(row["crop_x_min"])),
            ("crop_y_min", float(row["crop_y_min"])),
            ("crop_x_max", float(row["crop_x_max"])),
            ("crop_y_max", float(row["crop_y_max"])),
            ("quality_flag", row.get("quality_flag", "review")),
            ("label_source", row.get("label_source", "manual_gui")),
            ("notes", row.get("notes", "")),
            ("origin_manifest", row.get("origin_manifest", "")),
        ]
    )
    annotation = {
        "image_path": image_path,
        "source_manifest": str(DEFAULT_BOARD_LABELS_CSV.as_posix()),
        "source_kind": SOURCE_KIND,
        "source_row_index": source_row_index,
        "source_image_path": image_path,
        "source_row": source_row,
    }
    return {
        "image_path": image_path,
        "annotation_count": 1,
        "annotations": [annotation],
    }


def main() -> None:
    """Merge the board CSV into the grouped manifest and write a new JSON file."""

    args = _parse_args()
    grouped = _read_grouped_manifest(args.grouped_manifest)
    board_rows = _read_board_rows(args.board_labels_csv)
    existing_paths = _existing_image_paths(list(grouped["images"]))

    merged_images = list(grouped["images"])
    added = 0
    for index, row in enumerate(board_rows):
        image_path = str(row["image_path"]).strip()
        if not image_path or image_path in existing_paths:
            continue
        merged_images.append(_board_entry(row, source_row_index=index))
        existing_paths.add(image_path)
        added += 1

    output = {
        "schema_version": grouped.get("schema_version", 1),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifests": list(grouped.get("source_manifests", []))
        + [
            {
                "source_manifest": args.board_labels_csv.as_posix(),
                "source_kind": SOURCE_KIND,
                "row_count": len(board_rows),
                "added_count": added,
            }
        ],
        "images": merged_images,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "grouped_manifest": args.grouped_manifest.as_posix(),
                "board_labels_csv": args.board_labels_csv.as_posix(),
                "output_manifest": args.output_manifest.as_posix(),
                "reviewed_rows": len(board_rows),
                "added_rows": added,
                "total_images": len(merged_images),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
