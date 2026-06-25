#!/usr/bin/env python3
"""Prepare a center/tip review batch from already-reviewed board boxes.

This is the board-point counterpart to the bbox review workflow, but it starts
from a grouped manifest that already contains reviewed board bounding boxes.
That lets us collect more SimCC supervision without asking for additional
boxes. The output is a plain CSV that the OpenCV point labeler can open
directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.capture_labeling import (  # noqa: E402
    CaptureLabelRecord,
    load_label_records,
    resolve_source_size,
    to_repo_relative_path,
    write_label_records,
)

DEFAULT_MANIFEST: Path = PROJECT_ROOT.parent / "tmp" / "labelled_captured_images_board_bbox_hardcase_v3.json"
DEFAULT_OUTPUT: Path = PROJECT_ROOT.parent / "tmp" / "board_center_tip_review_batch_reviewed_boxes.csv"
DEFAULT_LIMIT: int = 50
DEFAULT_SOURCE_KIND: str = "reviewed_geometry"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Prepare a center/tip review batch from reviewed board boxes."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Grouped manifest with reviewed board boxes. Defaults to the hard-case board bbox manifest.",
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
        "--source-kind",
        type=str,
        default=DEFAULT_SOURCE_KIND,
        help="Grouped-manifest source kind to treat as the reviewed board box source.",
    )
    return parser.parse_args()


def _record_from_reviewed_image(
    image_path: Path,
    *,
    source_width: int,
    source_height: int,
    origin_manifest: str,
) -> CaptureLabelRecord:
    """Create an empty point-label record for one reviewed board image."""

    return CaptureLabelRecord(
        image_path=image_path,
        source_width=source_width,
        source_height=source_height,
        center_x_source=None,
        center_y_source=None,
        tip_x_source=None,
        tip_y_source=None,
        temperature_c=None,
        label_quality="manual",
        quality_flag="review",
        notes="",
        label_source="board_center_tip_reviewed_boxes",
        origin_manifest=origin_manifest,
    )


def main() -> None:
    """Build the board center/tip review batch from a reviewed manifest."""

    args = _parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Grouped manifest not found: {args.manifest}")

    existing_records = load_label_records(args.output)
    existing_paths = set(existing_records)

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, list):
        raise ValueError(f"{args.manifest} does not contain a grouped manifest.")

    review_records: list[CaptureLabelRecord] = []
    for entry in images:
        if len(review_records) >= int(args.limit):
            break
        if not isinstance(entry, dict):
            continue
        image_path_text = str(entry.get("image_path", "")).strip()
        if not image_path_text:
            continue
        if image_path_text in existing_paths:
            continue

        annotations = entry.get("annotations", [])
        if not isinstance(annotations, list):
            continue

        reviewed_row: dict[str, object] | None = None
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            if str(annotation.get("source_kind", "")).strip() != args.source_kind:
                continue
            source_row = annotation.get("source_row", {})
            if isinstance(source_row, dict):
                reviewed_row = source_row
                break
        if reviewed_row is None:
            continue

        image_path = to_repo_relative_path(image_path_text)
        first_annotation_row = reviewed_row
        source_width, source_height = resolve_source_size(
            image_path,
            row=first_annotation_row,
        )
        review_records.append(
            _record_from_reviewed_image(
                image_path,
                source_width=source_width,
                source_height=source_height,
                origin_manifest=args.manifest.as_posix(),
            )
        )

    if not review_records:
        raise ValueError(f"No reviewed board-box rows were found in {args.manifest}.")

    write_label_records(args.output, review_records)

    print(f"Selected {len(review_records)} reviewed board captures for center/tip labeling.")
    print(f"Wrote review batch CSV: {args.output}")
    print(
        "Launch the annotator with: "
        f"poetry run python scripts/label_board_center_tip.py --input {args.output}"
    )


if __name__ == "__main__":
    main()
