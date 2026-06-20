"""Tests for the v12 full-geometry manifest builder."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import create_geometry_heatmap_v12_all_data_manifest as v12_manifest


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a tiny manifest CSV for merge tests."""

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_merge_manifest_rows_prefers_primary_duplicates_and_keeps_unique_rows(tmp_path: Path) -> None:
    """The v12 builder should keep the first row seen for each image path."""

    primary = tmp_path / "primary.csv"
    secondary = tmp_path / "secondary.csv"
    output = tmp_path / "merged.csv"

    _write_manifest(
        primary,
        [
            {
                "image_path": "ml/data/a.jpg",
                "temperature_c": "10.0",
                "split": "train",
                "quality_flag": "clean",
            },
            {
                "image_path": "ml/data/b.jpg",
                "temperature_c": "20.0",
                "split": "val",
                "quality_flag": "clean",
            },
        ],
    )
    _write_manifest(
        secondary,
        [
            {
                "image_path": "ml/data/b.jpg",
                "temperature_c": "21.0",
                "split": "test",
                "quality_flag": "clean",
            },
            {
                "image_path": "ml/data/c.jpg",
                "temperature_c": "30.0",
                "split": "train",
                "quality_flag": "clean",
            },
        ],
    )

    merged = v12_manifest._merge_manifest_rows([primary, secondary])
    assert [row["image_path"] for row in merged] == [
        "ml/data/a.jpg",
        "ml/data/b.jpg",
        "ml/data/c.jpg",
    ]
    assert merged[1]["temperature_c"] == "20.0"
    assert merged[1]["split"] == "val"

    v12_manifest._write_manifest(merged, output)
    with output.open("r", encoding="utf-8", newline="") as handle:
        written_rows = list(csv.DictReader(handle))

    assert len(written_rows) == 3
    assert written_rows[1]["temperature_c"] == "20.0"
    assert written_rows[2]["image_path"] == "ml/data/c.jpg"
