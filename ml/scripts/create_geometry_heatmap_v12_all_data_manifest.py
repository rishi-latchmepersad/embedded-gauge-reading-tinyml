"""Build the v12 full-geometry manifest for geometry heatmap training.

The v12 run uses the union of the clean geometry manifest and the merged
board manifest with the manually completed hard-case capture.  The center-only
AI annotation CSV is intentionally excluded because it does not contain the tip
keypoint needed by the current geometry heatmap trainer.

Usage:
    poetry run python ml/scripts/create_geometry_heatmap_v12_all_data_manifest.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DEFAULT_PRIMARY_MANIFEST: Path = REPO_ROOT / "ml" / "data" / "merged_geometry_board_manifest_plus_hardcase_v1.csv"
DEFAULT_SECONDARY_MANIFEST: Path = REPO_ROOT / "ml" / "data" / "geometry_reader_manifest_v2_clean.csv"
DEFAULT_OUTPUT_PATH: Path = REPO_ROOT / "ml" / "data" / "geometry_heatmap_v12_all_data_manifest.csv"


def _resolve_path(repo_root: Path, path: Path) -> Path:
    """Resolve a path relative to the repository root when needed."""

    return path if path.is_absolute() else repo_root / path


def _load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    """Load rows from a manifest and drop rows that are explicitly excluded."""

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")

        for raw_row in reader:
            image_path = str(raw_row.get("image_path", "")).strip()
            if not image_path:
                continue

            quality_flag = str(raw_row.get("quality_flag", "")).strip().lower()
            if quality_flag == "exclude":
                continue

            row = {key: "" if value is None else str(value).strip() for key, value in raw_row.items()}
            row["image_path"] = image_path
            rows.append(row)

    return rows


def _merge_manifest_rows(manifest_paths: Sequence[Path]) -> list[dict[str, str]]:
    """Merge multiple manifests, keeping the first row seen for each image."""

    merged: list[dict[str, str]] = []
    seen_image_paths: set[str] = set()

    for manifest_path in manifest_paths:
        for row in _load_manifest_rows(manifest_path):
            image_path = row["image_path"]
            if image_path in seen_image_paths:
                continue
            seen_image_paths.add(image_path)
            merged.append(row)

    return merged


def _collect_fieldnames(rows: Sequence[dict[str, str]]) -> list[str]:
    """Collect CSV field names in first-seen order across all rows."""

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    return fieldnames


def _write_manifest(rows: Sequence[dict[str, str]], output_path: Path) -> None:
    """Write a merged manifest CSV with stable column order."""

    if not rows:
        raise ValueError("Refusing to write an empty manifest.")

    fieldnames = _collect_fieldnames(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _count_splits(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    """Summarize the split distribution in the merged manifest."""

    split_counts: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split", "")).strip() or "unknown"
        split_counts[split] = split_counts.get(split, 0) + 1
    return split_counts


def main() -> None:
    """Create the v12 training manifest from the full-geometry sources."""

    parser = argparse.ArgumentParser(description="Build the v12 full-geometry manifest")
    parser.add_argument("--primary-manifest", type=Path, default=DEFAULT_PRIMARY_MANIFEST)
    parser.add_argument("--secondary-manifest", type=Path, default=DEFAULT_SECONDARY_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    primary_manifest = _resolve_path(REPO_ROOT, args.primary_manifest)
    secondary_manifest = _resolve_path(REPO_ROOT, args.secondary_manifest)
    output_path = _resolve_path(REPO_ROOT, args.output)

    primary_rows = _load_manifest_rows(primary_manifest)
    secondary_rows = _load_manifest_rows(secondary_manifest)
    merged_rows = _merge_manifest_rows([primary_manifest, secondary_manifest])

    _write_manifest(merged_rows, output_path)

    split_counts = _count_splits(merged_rows)
    print(
        "[V12 MANIFEST] "
        f"primary={len(primary_rows)} secondary={len(secondary_rows)} "
        f"merged={len(merged_rows)} splits={split_counts}",
        flush=True,
    )
    print(f"[V12 MANIFEST] Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
