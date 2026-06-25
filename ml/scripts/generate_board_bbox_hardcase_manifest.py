#!/usr/bin/env python3
"""Generate a board-bbox labeling manifest biased toward hard captures.

The earlier diverse batch was useful for broad coverage, but the OBB replay
still misses on a few ugly board sessions. This selector intentionally
over-samples those awkward capture families first, then fills the remaining
slots with nearby full-frame captures from the same busy date ranges.

The output is a plain two-column CSV that the OpenCV bbox labeler can open
directly.
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
DEFAULT_OUTPUT: Path = REPO_ROOT / "tmp" / "board_bbox_hardcase_manifest.csv"
DEFAULT_LIMIT: int = 50

SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".yuv422"})
SEARCH_ROOTS: tuple[Path, ...] = (
    PROJECT_ROOT / "data" / "captured_images",
    PROJECT_ROOT / "data" / "captured_images" / "today_converted",
)

# These exact filenames and families were the worst board replay failures.
HARDCASE_TOKENS: tuple[str, ...] = (
    "capture_0074",
    "capture_0075",
    "capture_m30c",
    "capture_p35c",
    "capture_p50c",
    "capture_p10c",
)

# These date ranges produced lots of the same board-capture conditions.
HARDCASE_DATES: tuple[str, ...] = (
    "2026-04-19",
    "2026-05-30",
    "2026-05-31",
    "2026-06-01",
    "2026-06-02",
    "2026-06-03",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One unlabeled capture plus the bucket used for selection."""

    image_path: str
    bucket: str
    bucket_priority: int
    sort_key: str


def _load_labelled_paths(label_csv: Path) -> set[str]:
    """Read already labeled image paths so we can skip duplicates."""

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


def _bucket_for_path(image_path: str) -> tuple[str, int]:
    """Assign one hard-case bucket and a priority to an image path."""

    name = Path(image_path).name
    lowered = image_path.lower()

    for token in HARDCASE_TOKENS:
        if token in lowered:
            return (token, 0)

    for date in HARDCASE_DATES:
        if date in lowered:
            return (date, 1)

    match = re.search(r"20\d{2}-\d{2}", name)
    if match:
        return (match.group(0), 2)

    if name.startswith(("capture_p", "capture_m", "capture_0", "capture_00", "capture_007", "ann_")):
        return ("named", 2)

    if "today_converted" in lowered:
        return ("today_converted", 3)

    return ("other", 4)


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
            bucket, bucket_priority = _bucket_for_path(rel_path)
            candidates.append(
                Candidate(
                    image_path=rel_path,
                    bucket=bucket,
                    bucket_priority=bucket_priority,
                    sort_key=_stable_sort_key(rel_path),
                )
            )
    return candidates


def select_hardcase_candidates(candidates: list[Candidate], *, limit: int) -> list[Candidate]:
    """Select a balanced sample that starts with the hardest capture families."""

    buckets: dict[str, list[Candidate]] = defaultdict(list)
    priorities: dict[str, int] = {}
    for candidate in candidates:
        buckets[candidate.bucket].append(candidate)
        priorities[candidate.bucket] = min(priorities.get(candidate.bucket, candidate.bucket_priority), candidate.bucket_priority)

    for bucket_candidates in buckets.values():
        bucket_candidates.sort(key=lambda item: item.sort_key)

    ordered_buckets = sorted(
        buckets,
        key=lambda bucket: (priorities[bucket], -len(buckets[bucket]), bucket),
    )

    selected: list[Candidate] = []
    offsets = {bucket: 0 for bucket in ordered_buckets}
    while len(selected) < limit:
        progress = False
        for bucket in ordered_buckets:
            offset = offsets[bucket]
            bucket_candidates = buckets[bucket]
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

    parser = argparse.ArgumentParser(description="Generate a hard-case board bbox labeling manifest.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path. Defaults to tmp/board_bbox_hardcase_manifest.csv.",
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
    """Write the hard-case manifest to disk."""

    args = parse_args()
    labelled_paths = _load_labelled_paths(args.labeled_csv)
    candidates = collect_candidates(SEARCH_ROOTS, labelled_paths)
    selected = select_hardcase_candidates(candidates, limit=args.limit)

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
