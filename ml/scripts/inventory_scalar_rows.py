"""Inventory all scalar-usable manifest rows in `ml/data`.

This script scans every CSV under `ml/data`, keeps rows with a numeric
temperature value and a resolvable image file, then reports:
- total valid scalar rows across all manifests
- unique image references
- unique resolved filesystem paths
- per-manifest counts
- duplicate/conflicting labels for the same resolved file

The goal is to answer a practical training question:
how many distinct scalar examples can we actually feed into the best model?
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Iterable


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "ml" / "data"
CAPTURED_DIR: Path = DATA_DIR / "captured_images"

_VALID_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".pgm",
    ".raw16",
    ".raw",
    ".yuv422",
)


@dataclass(frozen=True)
class ScalarRow:
    """A single scalar-labeled manifest row with its resolved file path."""

    source_file: str
    image_ref: str
    resolved_path: str
    value: float


@dataclass(frozen=True)
class InventoryReport:
    """Summary statistics for all scalar-usable manifest rows."""

    total_valid_rows: int
    unique_image_refs: int
    unique_resolved_paths: int
    unique_captured_stems: int
    conflicting_resolved_paths: int
    per_source_rows: dict[str, int]
    conflicting_paths: dict[str, list[float]]


def _iter_manifest_rows() -> Iterable[ScalarRow]:
    """Yield every valid scalar manifest row from `ml/data/*.csv`."""
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_path = (row.get("image_path") or row.get("path") or "").strip()
                if not raw_path or raw_path.startswith("#"):
                    continue
                if not raw_path.lower().endswith(_VALID_EXTENSIONS):
                    continue

                value_text = (row.get("value") or row.get("label") or "").strip()
                try:
                    value = float(value_text)
                except ValueError:
                    continue

                normalized = raw_path.replace("\\", "/")
                resolved = _resolve_path(normalized)
                if resolved is None:
                    continue

                yield ScalarRow(
                    source_file=csv_path.name,
                    image_ref=normalized,
                    resolved_path=str(resolved),
                    value=value,
                )


def _resolve_path(normalized_path: str) -> Path | None:
    """Resolve a manifest path against the repo and captured-images roots."""
    candidate_paths: list[Path] = [
        REPO_ROOT / normalized_path,
        CAPTURED_DIR / Path(normalized_path).name,
        REPO_ROOT / "ml" / "data" / "captured_images" / Path(normalized_path).name,
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.resolve()
    return None


def build_inventory_report() -> InventoryReport:
    """Compute the full scalar-row inventory."""
    rows: list[ScalarRow] = list(_iter_manifest_rows())

    per_source_rows = Counter(row.source_file for row in rows)
    unique_image_refs = len({row.image_ref for row in rows})
    unique_resolved_paths = len({row.resolved_path for row in rows})

    captured_stems = {
        Path(row.image_ref).stem for row in rows if "ml/data/captured_images/" in row.image_ref
    }

    resolved_to_values: dict[str, set[float]] = defaultdict(set)
    for row in rows:
        resolved_to_values[row.resolved_path].add(row.value)

    conflicting_paths = {
        path: sorted(values)
        for path, values in resolved_to_values.items()
        if len(values) > 1
    }

    return InventoryReport(
        total_valid_rows=len(rows),
        unique_image_refs=unique_image_refs,
        unique_resolved_paths=unique_resolved_paths,
        unique_captured_stems=len(captured_stems),
        conflicting_resolved_paths=len(conflicting_paths),
        per_source_rows=dict(per_source_rows),
        conflicting_paths=conflicting_paths,
    )


def main() -> int:
    """Print a human-readable inventory and write a JSON report in tmp."""
    report = build_inventory_report()
    report_path = REPO_ROOT / "tmp" / "scalar_inventory_report.json"
    report_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"total_valid_rows={report.total_valid_rows}")
    print(f"unique_image_refs={report.unique_image_refs}")
    print(f"unique_resolved_paths={report.unique_resolved_paths}")
    print(f"unique_captured_stems={report.unique_captured_stems}")
    print(f"conflicting_resolved_paths={report.conflicting_resolved_paths}")
    print(f"report_path={report_path}")
    print("top_sources:")
    for name, count in sorted(
        report.per_source_rows.items(), key=lambda item: item[1], reverse=True
    )[:12]:
        print(f"{name}={count}")

    if report.conflicting_paths:
        print("conflicting_paths:")
        for path, values in list(report.conflicting_paths.items())[:20]:
            print(f"{path} -> {values}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
