#!/usr/bin/env python3
"""Generate a mixed-difficulty board bbox labeling manifest.

This selector is the next-step companion to the existing good/hard-case
manifests. It builds a new batch from the broader board capture pool with a
70/30 easy-to-moderate hard mix, while skipping anything already present in
the reviewed bbox CSV.

The output is a plain CSV that the OpenCV board bbox labeler can open
directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
DEFAULT_OUTPUT: Path = REPO_ROOT / "tmp" / "board_bbox_mixed_manifest.csv"
DEFAULT_LIMIT: int = 50
DEFAULT_EASY_FRACTION: float = 0.70

SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".yuv422"})
SEARCH_ROOTS: tuple[Path, ...] = (
    PROJECT_ROOT / "data" / "captured_images",
    PROJECT_ROOT / "data" / "captured_images" / "today_converted",
)

BAD_TOKENS: tuple[str, ...] = (
    "capture_0074",
    "capture_0075",
    "capture_m30c",
    "capture_p35c",
    "capture_p50c",
    "capture_p10c",
    "capture_p25c",
    "capture_p15c",
    "biglook_18-54-51",
)

HARDCASE_TOKENS: tuple[str, ...] = (
    "capture_0074",
    "capture_0075",
    "capture_m30c",
    "capture_p35c",
    "capture_p50c",
    "capture_p10c",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One unlabeled capture plus the bucket used for selection."""

    image_path: str
    bucket: str
    score: float
    sort_key: str


def _load_labelled_paths(label_csv: Path) -> set[str]:
    """Read already-labeled paths so we can skip duplicates."""

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
            "overlay",
        )
    )


def _is_bad_family(image_path: str) -> bool:
    """Return ``True`` for families we already know are not representative."""

    lowered = image_path.lower()
    return any(token in lowered for token in BAD_TOKENS)


def _is_hard_family(image_path: str) -> bool:
    """Return ``True`` for the known hard-case board capture families."""

    lowered = image_path.lower()
    return any(token in lowered for token in HARDCASE_TOKENS)


def _bucket_for_path(image_path: str) -> str:
    """Assign one stable diversity bucket to an image path."""

    path = Path(image_path)
    name = path.name
    if "today_converted" in image_path:
        return "today_converted"
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", name)
    if match:
        return match.group(1)
    if name.startswith(("capture_p", "capture_m", "capture_0", "capture_00", "capture_007", "ann_", "biglook_")):
        return "named"
    return "other"


def _stable_sort_key(image_path: str) -> str:
    """Return a deterministic pseudo-random sort key for one path."""

    return hashlib.sha1(image_path.encode("utf-8")).hexdigest()


def _score_luma(luma: np.ndarray) -> float:
    """Score one grayscale image as a likely good capture."""

    mean = float(np.mean(luma))
    std = float(np.std(luma))
    p10 = float(np.percentile(luma, 10))
    p90 = float(np.percentile(luma, 90))
    dynamic_range = p90 - p10

    brightness_score = 1.0 - min(abs(mean - 120.0) / 120.0, 1.0)
    noise_penalty = min(std / 90.0, 1.0)
    range_bonus = min(dynamic_range / 120.0, 1.0)
    return 2.2 * brightness_score + 0.8 * range_bonus - 1.2 * noise_penalty


def _load_luma_score(image_path: Path) -> float:
    """Load one image and compute its goodness score."""

    suffix = image_path.suffix.lower()
    try:
        if suffix == ".yuv422":
            raw = image_path.read_bytes()
            size = len(raw)
            inferred_dim = int(round(math.sqrt(size / 2.0)))
            if inferred_dim <= 0 or inferred_dim * inferred_dim * 2 != size:
                return float("-inf")
            data = np.frombuffer(raw, dtype=np.uint8).reshape(inferred_dim, inferred_dim * 2)
            luma = data[:, 0::2].astype(np.float32)
        else:
            with Image.open(image_path) as image:
                gray = image.convert("L")
                luma = np.asarray(gray, dtype=np.float32)
        return _score_luma(luma)
    except Exception:
        return float("-inf")


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
            if _is_derived_path(rel_path) or _is_bad_family(rel_path):
                continue
            if rel_path in seen_paths:
                continue
            seen_paths.add(rel_path)
            score = _load_luma_score(image_path)
            if not math.isfinite(score):
                continue
            bucket = _bucket_for_path(rel_path)
            if _is_hard_family(rel_path):
                bucket = f"hard:{bucket}"
            candidates.append(
                Candidate(
                    image_path=rel_path,
                    bucket=bucket,
                    score=score,
                    sort_key=_stable_sort_key(rel_path),
                )
            )
    return candidates


def _select_diverse(candidates: list[Candidate], *, limit: int) -> list[Candidate]:
    """Select one item per bucket in a round-robin pass."""

    buckets: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        buckets[candidate.bucket].append(candidate)
    for bucket_candidates in buckets.values():
        bucket_candidates.sort(key=lambda item: (-item.score, item.sort_key))

    ordered_buckets = sorted(
        buckets,
        key=lambda bucket: (-len(buckets[bucket]), bucket),
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


def _split_easy_hard(candidates: list[Candidate], *, easy_fraction: float, limit: int) -> list[Candidate]:
    """Pick a 70/30-style mix of easy and moderate-hard candidates."""

    easy_candidates = [candidate for candidate in candidates if not candidate.bucket.startswith("hard:")]
    hard_candidates = [candidate for candidate in candidates if candidate.bucket.startswith("hard:")]

    easy_target = max(1, int(round(limit * easy_fraction)))
    hard_target = max(0, limit - easy_target)

    easy_selected = _select_diverse(easy_candidates, limit=easy_target)
    hard_selected = _select_diverse(hard_candidates, limit=hard_target)

    selected = easy_selected + hard_selected
    if len(selected) < limit:
        used = {candidate.image_path for candidate in selected}
        remainder = [
            candidate
            for candidate in sorted(candidates, key=lambda item: (-item.score, item.sort_key))
            if candidate.image_path not in used
        ]
        selected.extend(remainder[: limit - len(selected)])

    return selected[:limit]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Generate a mixed board bbox labeling manifest.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path. Defaults to tmp/board_bbox_mixed_manifest.csv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of images to select for labeling.",
    )
    parser.add_argument(
        "--easy-fraction",
        type=float,
        default=DEFAULT_EASY_FRACTION,
        help="Fraction of the batch to draw from the easy/normal pool.",
    )
    parser.add_argument(
        "--labeled-csv",
        type=Path,
        default=REPO_ROOT / "tmp" / "board_bbox_labels.csv",
        help="Existing bbox CSV to exclude from the new batch.",
    )
    return parser.parse_args()


def main() -> None:
    """Write the mixed board-bbox manifest to disk."""

    args = parse_args()
    if not (0.0 < args.easy_fraction < 1.0):
        raise ValueError("--easy-fraction must be in (0, 1).")

    labelled_paths = _load_labelled_paths(args.labeled_csv)
    candidates = collect_candidates(SEARCH_ROOTS, labelled_paths)
    selected = _split_easy_hard(candidates, easy_fraction=float(args.easy_fraction), limit=int(args.limit))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "value"])
        for candidate in selected:
            writer.writerow([candidate.image_path, ""])

    bucket_counts: dict[str, int] = defaultdict(int)
    scores: list[float] = []
    for candidate in selected:
        bucket_counts[candidate.bucket] += 1
        scores.append(candidate.score)

    print(f"Wrote {args.output} with {len(selected)} images.")
    print(f"Mean score: {float(np.mean(scores)):.3f}")
    for bucket in sorted(bucket_counts):
        print(f"  {bucket}: {bucket_counts[bucket]}")


if __name__ == "__main__":
    main()
