"""Build the v13 geometry manifest with train-only trusted extras.

The v13 recipe keeps the original clean geometry manifest as the validation
anchor, then adds the trusted board / PXL / hard-case examples as
train-only rows.  That lets us use the full labeled pool without moving the
validation set again.

Usage:
    poetry run python ml/scripts/create_geometry_heatmap_v13_trusted_train_manifest.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DEFAULT_PRIMARY_MANIFEST: Path = REPO_ROOT / "ml" / "data" / "geometry_reader_manifest_v2_clean.csv"
DEFAULT_SECONDARY_MANIFEST: Path = REPO_ROOT / "ml" / "data" / "merged_geometry_board_manifest_plus_hardcase_v1.csv"
DEFAULT_OUTPUT_PATH: Path = REPO_ROOT / "ml" / "data" / "geometry_heatmap_v13_trusted_train_manifest.csv"


def _resolve_path(repo_root: Path, path: Path) -> Path:
    """Resolve repository-relative paths against the repo root."""

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


def _merge_manifest_rows(
    primary_rows: Sequence[dict[str, str]],
    supplemental_rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    """Merge manifests while forcing unique supplemental rows into train."""

    merged: list[dict[str, str]] = [dict(row) for row in primary_rows]
    seen_image_paths: set[str] = {row["image_path"] for row in merged}
    forced_train_rows = 0

    for row in supplemental_rows:
        image_path = row["image_path"]
        if image_path in seen_image_paths:
            continue

        merged_row = dict(row)
        if str(merged_row.get("split", "")).strip() != "train":
            merged_row["split"] = "train"
            forced_train_rows += 1

        merged.append(merged_row)
        seen_image_paths.add(image_path)

    return merged, forced_train_rows


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
    """Create the v13 trusted manifest with supplemental train-only rows."""

    parser = argparse.ArgumentParser(description="Build the v13 trusted-train geometry manifest")
    parser.add_argument("--primary-manifest", type=Path, default=DEFAULT_PRIMARY_MANIFEST)
    parser.add_argument("--secondary-manifest", type=Path, default=DEFAULT_SECONDARY_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    primary_manifest = _resolve_path(REPO_ROOT, args.primary_manifest)
    secondary_manifest = _resolve_path(REPO_ROOT, args.secondary_manifest)
    output_path = _resolve_path(REPO_ROOT, args.output)

    primary_rows = _load_manifest_rows(primary_manifest)
    secondary_rows = _load_manifest_rows(secondary_manifest)
    merged_rows, forced_train_rows = _merge_manifest_rows(primary_rows, secondary_rows)
    _write_manifest(merged_rows, output_path)

    clean_rows = [row for row in merged_rows if str(row.get("quality_flag", "")).strip().lower() == "clean"]
    clean_split_counts = _count_splits(clean_rows)
    print(
        "[V13 MANIFEST] "
        f"primary={len(primary_rows)} secondary={len(secondary_rows)} "
        f"merged={len(merged_rows)} clean={len(clean_rows)} "
        f"train_only_extra_rows={forced_train_rows} clean_splits={clean_split_counts}",
        flush=True,
    )
    print(f"[V13 MANIFEST] Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
