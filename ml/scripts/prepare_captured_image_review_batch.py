#!/usr/bin/env python3
"""Prepare a 50-image review batch from the captured-image board labels.

The output is a flat CSV that can be opened directly in the interactive
labeler. It keeps the full seed labels so the user can validate and correct
them instead of starting from scratch.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_DIR: Path = PROJECT_ROOT / "ml" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.capture_labeling import (  # noqa: E402
    load_capture_candidates,
    write_label_records,
)
from embedded_gauge_reading_tinyml.capture_review_batch import (  # noqa: E402
    DEFAULT_BATCH_LABEL,
    build_review_records,
    select_review_batch,
    temperature_bin_name,
)

DEFAULT_INPUT: Path = PROJECT_ROOT / "ml" / "data" / "board_captures_labeled_v2.csv"
DEFAULT_OUTPUT: Path = PROJECT_ROOT / "tmp" / "captured_image_review_batch_50.csv"


def _default_batch_label(output_path: Path) -> str:
    """Derive a readable batch label from the output filename."""

    stem = output_path.stem.strip()
    return stem or DEFAULT_BATCH_LABEL


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Prepare a balanced captured-image review batch for manual validation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input manifest or CSV. Defaults to ml/data/board_captures_labeled_v2.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Review CSV to write. Defaults to tmp/captured_image_review_batch_50.csv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of images to include in the starter batch.",
    )
    parser.add_argument(
        "--include-derivatives",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep preview or derived variants instead of filtering them out.",
    )
    return parser.parse_args()


def _summarize_selection(records: list[dict[str, str]]) -> None:
    """Print a short coverage summary for the generated batch."""

    temp_bins = Counter()
    source_labels = Counter()
    for row in records:
        try:
            temp_value = float(row.get("temperature_c", ""))
        except ValueError:
            temp_value = float("nan")
        temp_bins[temperature_bin_name(temp_value)] += 1
        source_labels[row.get("label_source", "")] += 1

    print(f"Temperature bins: {dict(temp_bins)}")
    print(f"Label sources: {dict(source_labels)}")


def main() -> None:
    """Generate the review CSV and print the selection summary."""

    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input path not found: {args.input}")

    candidates = load_capture_candidates(
        args.input,
        include_derivatives=bool(args.include_derivatives),
        recursive=False,
    )
    selected = select_review_batch(candidates, limit=int(args.limit))
    if len(selected) < args.limit:
        print(
            f"Warning: only {len(selected)} reviewable images were available "
            f"for a {args.limit}-image batch."
        )

    batch_label = _default_batch_label(args.output)
    records = build_review_records(selected, batch_label=batch_label)
    write_label_records(args.output, records)

    _summarize_selection([record.to_csv_row() for record in records])
    print(f"Wrote review batch CSV: {args.output}")
    print(
        "Launch the labeler with: "
        f"poetry run python scripts/label_captured_images_for_models.py --input {args.output}"
    )


if __name__ == "__main__":
    main()
