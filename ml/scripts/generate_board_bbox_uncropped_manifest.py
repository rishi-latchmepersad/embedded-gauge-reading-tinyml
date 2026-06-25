#!/usr/bin/env python3
"""Build a manifest of uncropped board captures for bbox labeling.

The source manifests contain a mix of full-frame captures, preview derivatives,
and already-cropped board images. This helper keeps only the uncropped source
images so the annotation pass sees the real field-of-view images.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
DEFAULT_OUTPUT: Path = REPO_ROOT / "tmp" / "board_bbox_uncropped_manifest.csv"

DEFAULT_SOURCES: tuple[Path, ...] = (
    PROJECT_ROOT / "data" / "hard_cases.csv",
    PROJECT_ROOT / "data" / "hard_cases_plus_board30_valid_with_new6.csv",
)


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """One image path and its numeric label from the source manifests."""

    image_path: str
    value: str


def _is_uncropped_image(image_path: str) -> bool:
    """Return ``True`` when the path points at a full-frame capture."""

    name = Path(image_path).name
    lowered = image_path.lower()
    if "board_crop" in lowered:
        return False
    if "_live_rectified_probe" in lowered:
        return False
    if "preview" in name:
        return False
    return True


def _load_rows(source_path: Path) -> list[ManifestRow]:
    """Load one source CSV and keep only the image/value columns we need."""

    rows: list[ManifestRow] = []
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "image_path" not in reader.fieldnames:
            raise ValueError(f"{source_path} is missing an image_path column.")
        value_column = "value" if "value" in reader.fieldnames else "temperature_c"
        for row in reader:
            image_path = str(row.get("image_path", "")).strip()
            if not image_path or image_path.startswith("#"):
                continue
            if not _is_uncropped_image(image_path):
                continue
            value = str(row.get(value_column, "")).strip()
            rows.append(ManifestRow(image_path=image_path, value=value))
    return rows


def build_manifest(source_paths: tuple[Path, ...]) -> list[ManifestRow]:
    """Merge and deduplicate the requested source manifests."""

    seen: set[str] = set()
    rows: list[ManifestRow] = []
    for source_path in source_paths:
        for row in _load_rows(source_path):
            if row.image_path in seen:
                continue
            seen.add(row.image_path)
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Generate an uncropped board bbox manifest.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path. Defaults to tmp/board_bbox_uncropped_manifest.csv.",
    )
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        default=[],
        help="Source manifest to merge. Defaults to the hard-case manifests.",
    )
    return parser.parse_args()


def main() -> None:
    """Write the uncropped manifest to disk."""

    args = parse_args()
    source_paths = tuple(args.source) if args.source else DEFAULT_SOURCES
    rows = build_manifest(source_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "value"])
        for row in rows:
            writer.writerow([row.image_path, row.value])
    print(f"Wrote {args.output} with {len(rows)} uncropped images from {len(source_paths)} source files.")


if __name__ == "__main__":
    main()
