"""Tests for the v13 trusted-train manifest builder."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import create_geometry_heatmap_v13_trusted_train_manifest as v13_manifest


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a tiny manifest CSV for merge tests."""

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_merge_manifest_rows_keeps_base_splits_and_forces_new_rows_to_train(tmp_path: Path) -> None:
    """Supplemental rows should stay out of validation and duplicate rows should be ignored."""

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
                "split": "val",
                "quality_flag": "clean",
            },
            {
                "image_path": "ml/data/d.jpg",
                "temperature_c": "40.0",
                "split": "train",
                "quality_flag": "clean",
            },
        ],
    )

    primary_rows = v13_manifest._load_manifest_rows(primary)
    secondary_rows = v13_manifest._load_manifest_rows(secondary)
    merged, forced_train_rows = v13_manifest._merge_manifest_rows(primary_rows, secondary_rows)

    assert [row["image_path"] for row in merged] == [
        "ml/data/a.jpg",
        "ml/data/b.jpg",
        "ml/data/c.jpg",
        "ml/data/d.jpg",
    ]
    assert merged[1]["temperature_c"] == "20.0"
    assert merged[1]["split"] == "val"
    assert merged[2]["split"] == "train"
    assert merged[3]["split"] == "train"
    assert forced_train_rows == 1

    v13_manifest._write_manifest(merged, output)
    with output.open("r", encoding="utf-8", newline="") as handle:
        written_rows = list(csv.DictReader(handle))

    assert len(written_rows) == 4
    assert written_rows[1]["temperature_c"] == "20.0"
    assert written_rows[2]["split"] == "train"
