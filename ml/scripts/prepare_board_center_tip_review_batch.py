#!/usr/bin/env python3
"""Prepare a board-only review batch for center and tip labeling.

The board bbox pass already gave us a curated set of captures to focus on.
This helper turns that bbox CSV into a clean point-label review batch that the
OpenCV labeler can open directly. We keep the workflow small and reusable:

- read the reviewed board bbox CSV,
- keep only images with usable boxes,
- drop explicitly excluded rows,
- write a center/tip review CSV with blank geometry fields.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.board_bbox_labeling import (  # noqa: E402
    DEFAULT_OUTPUT_CSV as DEFAULT_BOARD_BBOX_OUTPUT,
    BoardBBoxRecord,
    load_board_bbox_records,
)
from embedded_gauge_reading_tinyml.capture_labeling import (  # noqa: E402
    CaptureLabelRecord,
    write_label_records,
)

DEFAULT_INPUT: Path = DEFAULT_BOARD_BBOX_OUTPUT
DEFAULT_OUTPUT: Path = PROJECT_ROOT.parent / "tmp" / "board_center_tip_review_batch.csv"
DEFAULT_LIMIT: int = 50
_PREFERRED_SUFFIX_RANK: dict[str, int] = {
    ".png": 0,
    ".jpg": 0,
    ".jpeg": 0,
    ".bmp": 0,
    ".pgm": 0,
    ".yuv422": 1,
    ".raw": 1,
    ".raw16": 1,
}


def _record_to_review_label(record: BoardBBoxRecord) -> CaptureLabelRecord:
    """Convert a board bbox record into an empty center/tip review row."""

    return CaptureLabelRecord(
        image_path=record.image_path,
        source_width=record.source_width,
        source_height=record.source_height,
        center_x_source=None,
        center_y_source=None,
        tip_x_source=None,
        tip_y_source=None,
        temperature_c=None,
        label_quality="manual",
        quality_flag="review",
        notes=record.notes.strip(),
        label_source="board_center_tip_review_batch",
        origin_manifest=record.origin_manifest or record.image_path.as_posix(),
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Prepare a board-only center/tip review batch from bbox labels."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Board bbox label CSV to convert. Defaults to tmp/board_bbox_labels.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Review CSV for the OpenCV center/tip labeler.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of board captures to include in the review batch.",
    )
    parser.add_argument(
        "--include-excluded",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep rows marked exclude in the input bbox CSV.",
    )
    return parser.parse_args()


def _summarize(records: list[CaptureLabelRecord]) -> dict[str, int]:
    """Compute a tiny summary for the generated batch."""

    return {
        "sample_count": len(records),
        "with_notes": sum(1 for record in records if bool(record.notes.strip())),
    }


def main() -> None:
    """Build the board center/tip review batch."""

    args = _parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Board bbox labels not found: {args.input}")

    bbox_records = load_board_bbox_records(args.input)
    selected: list[BoardBBoxRecord] = []
    seen_families: set[str] = set()
    ordered_records = sorted(
        bbox_records.values(),
        key=lambda record: (
            record.image_path.stem,
            _PREFERRED_SUFFIX_RANK.get(record.image_path.suffix.lower(), 9),
            record.image_path.as_posix(),
        ),
    )
    for record in ordered_records:
        if not args.include_excluded and record.quality_flag.strip().lower() == "exclude":
            continue
        if not record.has_box:
            continue
        family_key = record.image_path.stem
        if family_key in seen_families:
            continue
        seen_families.add(family_key)
        selected.append(record)
        if len(selected) >= int(args.limit):
            break

    if not selected:
        raise ValueError(f"No reviewable board bbox rows were found in {args.input}.")

    review_records = [_record_to_review_label(record) for record in selected]
    write_label_records(args.output, review_records)

    summary = _summarize(review_records)
    print(f"Selected {summary['sample_count']} board captures for center/tip labeling.")
    print(f"Rows with notes: {summary['with_notes']}")
    print(f"Wrote review batch CSV: {args.output}")
    print(
        "Launch the annotator with: "
        f"poetry run python scripts/label_board_center_tip.py --input {args.output}"
    )


if __name__ == "__main__":
    main()
