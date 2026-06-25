#!/usr/bin/env python3
"""Generate a diverse 50-image board bbox labeling manifest.

This selector walks the full captured-image pool, excludes images that were
already boxed, filters out obvious derived/cropped artifacts, and then picks a
balanced round-robin sample across capture families and file types.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
DEFAULT_OUTPUT: Path = REPO_ROOT / "tmp" / "board_bbox_diverse_manifest.csv"
DEFAULT_LIMIT: int = 50

SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".yuv422"})
DEFAULT_SEARCH_ROOTS: tuple[Path, ...] = (
    PROJECT_ROOT / "data" / "captured_images",
    PROJECT_ROOT / "data" / "captured_images" / "today_converted",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One image path with the bucket used for balancing."""

    image_path: str
    bucket: str
    sort_key: str


def _load_labelled_paths(label_csv: Path) -> set[str]:
    """Read already-labeled paths so we can avoid duplicates."""

    if not label_csv.exists():
        return set()
    labelled: set[str] = set()
    with label_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            path = str(row.get("image_path", "")).strip()
            if path:
                labelled.add(path)
    return labelled


def _is_derived_path(image_path: str) -> bool:
    """Return ``True`` for previews, crops, overlays, or other derived assets."""

    lowered = image_path.lower()
    return any(
        token in lowered
        for token in (
            "board_crop",
            "_live_rectified_probe",
            "preview",
            "_view",
            "_glare_",
            ".gray.",
            "debug",
            "show_",
            "crop_",
            "_latest",
            "annotated",
            "ann_",
            "diag",
            "big_",
        )
    )


def _bucket_for_path(image_path: str) -> str:
    """Assign one stable diversity bucket to an image path."""

    path = Path(image_path)
    name = path.name
    suffix = path.suffix.lower()
    if "today_converted" in image_path:
        family = "today_converted"
    else:
        match = re.search(r"20\d{2}-\d{2}", name)
        if match:
            family = match.group(0)
        elif name.startswith(("capture_p", "capture_m", "capture_0", "capture_00", "capture_007", "ann_")):
            family = "named"
        else:
            family = "other"
    return f"{family}:{suffix}"


def _stable_sort_key(image_path: str) -> str:
    """Return a deterministic pseudo-random sort key for one path."""

    return hashlib.sha1(image_path.encode("utf-8")).hexdigest()


def collect_candidates(search_roots: tuple[Path, ...], labelled_paths: set[str]) -> list[Candidate]:
    """Walk the capture tree and keep only usable, unlabeled images."""

    candidates: list[Candidate] = []
    seen_paths: set[str] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for image_path in sorted(root.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            rel_path = image_path.relative_to(REPO_ROOT).as_posix()
            if rel_path in labelled_paths:
                continue
            if _is_derived_path(rel_path):
                continue
            if rel_path in seen_paths:
                continue
            seen_paths.add(rel_path)
            candidates.append(
                Candidate(
                    image_path=rel_path,
                    bucket=_bucket_for_path(rel_path),
                    sort_key=_stable_sort_key(rel_path),
                )
            )
    return candidates


def select_diverse_candidates(candidates: list[Candidate], *, limit: int) -> list[Candidate]:
    """Select a balanced round-robin sample across the candidate buckets."""

    buckets: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        buckets[candidate.bucket].append(candidate)

    for bucket_candidates in buckets.values():
        bucket_candidates.sort(key=lambda item: item.sort_key)

    ordered_buckets = sorted(
        buckets,
        key=lambda bucket: (-len(buckets[bucket]), bucket),
    )
    selected: list[Candidate] = []
    offsets = {bucket: 0 for bucket in ordered_buckets}

    while len(selected) < limit:
        progress = False
        for bucket in ordered_buckets:
            bucket_candidates = buckets[bucket]
            offset = offsets[bucket]
            if offset >= len(bucket_candidates):
                continue
            selected.append(bucket_candidates[offset])
            offsets[bucket] = offset + 1
            progress = True
            if len(selected) >= limit:
                break
        if not progress:
            break

    return selected


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Generate a diverse board bbox labeling manifest.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path. Defaults to tmp/board_bbox_diverse_manifest.csv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of images to select for labeling.",
    )
    parser.add_argument(
        "--labeled-csv",
        type=Path,
        default=REPO_ROOT / "tmp" / "board_bbox_labels.csv",
        help="Existing bbox CSV to exclude from the new batch.",
    )
    return parser.parse_args()


def main() -> None:
    """Write the diverse manifest to disk."""

    args = parse_args()
    labelled_paths = _load_labelled_paths(args.labeled_csv)
    candidates = collect_candidates(DEFAULT_SEARCH_ROOTS, labelled_paths)
    selected = select_diverse_candidates(candidates, limit=args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "value"])
        for candidate in selected:
            writer.writerow([candidate.image_path, ""])

    bucket_counts: dict[str, int] = defaultdict(int)
    for candidate in selected:
        bucket_counts[candidate.bucket] += 1
    print(f"Wrote {args.output} with {len(selected)} images.")
    for bucket in sorted(bucket_counts):
        print(f"  {bucket}: {bucket_counts[bucket]}")


if __name__ == "__main__":
    main()
