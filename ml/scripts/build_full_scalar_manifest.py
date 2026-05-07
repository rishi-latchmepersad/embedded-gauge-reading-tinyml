"""Build a deduped scalar training manifest from all project CSVs.

This script scans `ml/data/*.csv`, keeps rows with numeric temperature labels
and resolvable image files, deduplicates by resolved filesystem path, and
resolves label conflicts using a fixed source-precedence order.

The resulting manifest is intended to be the broadest clean scalar-training
set available in this repository.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "ml" / "data"
CAPTURED_DIR: Path = DATA_DIR / "captured_images"

VALID_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".pgm",
    ".raw16",
    ".raw",
    ".yuv422",
)

# Lower index means higher priority.
SOURCE_PRECEDENCE: list[str] = [
    "canonical_manifest_v1.csv",
    "combined_training_manifest.csv",
    "all_captured_images_manifest.csv",
    "unified_training_manifest_v1.csv",
    "full_labelled_plus_board30_valid_with_new5.csv",
    "hard_cases_extreme_weighted_v4.csv",
    "hard_cases_plus_board30_valid_with_new6.csv",
    "hard_cases_plus_board30_valid_with_new5_closeup14c.csv",
    "hard_cases_plus_board30_valid_with_new5.csv",
    "hard_cases_plus_board30_valid_with_new4.csv",
    "hard_cases_plus_board30_valid_with_new3.csv",
    "hard_cases_plus_board30_valid_with_new2.csv",
    "hard_cases_plus_board30_valid.csv",
    "hard_cases_plus_board30.csv",
    "hard_cases_remaining_focus.csv",
    "hard_cases.csv",
    "mid_band_focus_18_42.csv",
    "full_range_regression_focus.csv",
    "board_rectified_probe_20260422.csv",
    "rectified_crop_boxes_v5_all.csv",
    "rectified_crop_boxes_v4_all.csv",
    "rectified_crop_boxes_v5_20260422.csv",
    "rectified_crop_boxes_v4.csv",
    "new_labelled_captures4.csv",
    "new_labelled_captures3_keep.csv",
    "new_labelled_captures3.csv",
    "new_labelled_captures2.csv",
    "new_labelled_captures.csv",
    "recapture_targets.csv",
]


@dataclass(frozen=True)
class ManifestRow:
    """One scalar-label manifest row after path resolution."""

    source_file: str
    image_ref: str
    resolved_path: str
    value: float


def _source_rank(source_file: str) -> int:
    """Return a stable precedence rank for a manifest source file."""
    try:
        return SOURCE_PRECEDENCE.index(source_file)
    except ValueError:
        return len(SOURCE_PRECEDENCE)


def _resolve_path(normalized_path: str) -> Path | None:
    """Resolve a manifest path to an existing file on disk."""
    candidate_paths: list[Path] = [
        REPO_ROOT / normalized_path,
        CAPTURED_DIR / Path(normalized_path).name,
        REPO_ROOT / "ml" / "data" / "captured_images" / Path(normalized_path).name,
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.resolve()
    return None


def _iter_rows() -> Iterable[ManifestRow]:
    """Yield all valid scalar rows from every CSV in `ml/data`."""
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_path = (row.get("image_path") or row.get("path") or "").strip()
                if not raw_path or raw_path.startswith("#"):
                    continue
                if not raw_path.lower().endswith(VALID_EXTENSIONS):
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

                yield ManifestRow(
                    source_file=csv_path.name,
                    image_ref=normalized,
                    resolved_path=str(resolved),
                    value=value,
                )


def build_full_scalar_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the deduped scalar manifest and a conflicts table."""
    rows = list(_iter_rows())
    if not rows:
        raise ValueError("No valid scalar rows were found.")

    rows_df = pd.DataFrame([row.__dict__ for row in rows])
    rows_df["source_rank"] = rows_df["source_file"].map(_source_rank)
    rows_df = rows_df.sort_values(
        by=["resolved_path", "source_rank", "source_file", "image_ref"],
        ascending=[True, True, True, True],
    )

    kept_rows: list[dict[str, object]] = []
    conflict_rows: list[dict[str, object]] = []

    for resolved_path, group in rows_df.groupby("resolved_path", sort=False):
        values = group["value"].astype(float).tolist()
        if len(set(values)) > 1:
            # Keep the highest-priority source, but record the full conflict set.
            for _, row in group.iterrows():
                conflict_rows.append(row.to_dict())

        chosen = group.iloc[0].to_dict()
        chosen["image_path"] = chosen.pop("image_ref")
        kept_rows.append(chosen)

    manifest_df = pd.DataFrame(kept_rows)
    manifest_df = manifest_df[
        [
            "image_path",
            "value",
            "source_file",
            "resolved_path",
            "source_rank",
        ]
    ].sort_values(by=["value", "source_file", "image_path"]).reset_index(drop=True)

    conflicts_df = pd.DataFrame(conflict_rows)
    if not conflicts_df.empty:
        conflicts_df = conflicts_df[
            [
                "source_file",
                "image_ref",
                "resolved_path",
                "value",
                "source_rank",
            ]
        ].sort_values(
            by=["resolved_path", "source_rank", "source_file", "image_ref"]
        ).reset_index(drop=True)

    return manifest_df, conflicts_df


def main() -> int:
    """Write the manifest and conflicts table to `ml/data`."""
    parser = argparse.ArgumentParser(
        description="Build the full deduped scalar training manifest."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "full_scalar_manifest_v1.csv",
        help="Output manifest CSV path.",
    )
    parser.add_argument(
        "--conflicts-output",
        type=Path,
        default=DATA_DIR / "full_scalar_manifest_conflicts_v1.csv",
        help="Output CSV path for conflicting rows.",
    )
    args = parser.parse_args()

    manifest_df, conflicts_df = build_full_scalar_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(args.output, index=False)
    print(f"saved_manifest={args.output}")
    print(f"manifest_rows={len(manifest_df)}")

    if not conflicts_df.empty:
        conflicts_df.to_csv(args.conflicts_output, index=False)
        print(f"saved_conflicts={args.conflicts_output}")
        print(f"conflict_rows={len(conflicts_df)}")
    else:
        print("conflict_rows=0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
