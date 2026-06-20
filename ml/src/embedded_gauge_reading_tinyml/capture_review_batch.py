"""Build a small, balanced review batch from captured-image labels.

The starter batch is meant for the interactive labeler GUI. We keep the
selection deterministic, favor the cleaner label-source rows inside each
temperature band, and preserve the full center, tip, and temperature labels so
the user can correct them in one pass.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .capture_labeling import CaptureCandidate, CaptureLabelRecord

TEMP_BIN_ORDER: tuple[str, ...] = (
    "lt0",
    "0-10",
    "10-20",
    "20-30",
    "30-40",
    "40+",
)

LABEL_SOURCE_PRIORITY: dict[str, int] = {
    "manual_verification": 0,
    "inverse_mapping": 1,
}

DEFAULT_BATCH_LABEL: str = "captured_image_review_batch_50"


def temperature_bin_name(temperature_c: float) -> str:
    """Map a temperature to the review batch bin we use for balancing."""

    if not math.isfinite(float(temperature_c)):
        return "unknown"
    if temperature_c < 0.0:
        return "lt0"
    if temperature_c < 10.0:
        return "0-10"
    if temperature_c < 20.0:
        return "10-20"
    if temperature_c < 30.0:
        return "20-30"
    if temperature_c < 40.0:
        return "30-40"
    return "40+"


def _stable_hash(text: str) -> str:
    """Return a deterministic hash string for a path or label key."""

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _reviewable_candidate(candidate: CaptureCandidate) -> bool:
    """Keep only candidates that already have a full label seed."""

    record = candidate.seed_record
    return bool(
        record is not None
        and record.has_center
        and record.has_tip
        and record.has_temperature
    )


def _source_priority(record: CaptureLabelRecord) -> int:
    """Prefer verified rows before inverse-mapped rows when we sample."""

    return LABEL_SOURCE_PRIORITY.get(record.label_source.strip().lower(), 2)


def _candidate_sort_key(candidate: CaptureCandidate) -> tuple[int, str]:
    """Sort a candidate within its temperature bin."""

    record = candidate.seed_record
    assert record is not None
    return (
        _source_priority(record),
        _stable_hash(candidate.image_path.as_posix()),
    )


def select_review_batch(
    candidates: Sequence[CaptureCandidate],
    *,
    limit: int = 50,
) -> list[CaptureCandidate]:
    """Select a balanced review batch from the available candidates.

    We group by temperature band and then pull one item from each band in
    round-robin order. That keeps the batch assorted without losing the nice
    label coverage from the captured-image board set.
    """

    pools: dict[str, list[CaptureCandidate]] = {bin_name: [] for bin_name in TEMP_BIN_ORDER}
    for candidate in candidates:
        if not _reviewable_candidate(candidate):
            continue
        record = candidate.seed_record
        assert record is not None
        bin_name = temperature_bin_name(float(record.temperature_c))
        if bin_name not in pools:
            continue
        pools[bin_name].append(candidate)

    # Keep the order inside each temperature band deterministic and stable.
    for pool in pools.values():
        pool.sort(key=_candidate_sort_key)

    selected: list[CaptureCandidate] = []
    while len(selected) < limit:
        progressed = False
        for bin_name in TEMP_BIN_ORDER:
            pool = pools[bin_name]
            if not pool:
                continue
            selected.append(pool.pop(0))
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break

    return selected


def _merge_notes(
    existing_notes: str,
    *,
    batch_label: str,
    selection_rank: int,
    selected_total: int,
    temperature_bin: str,
    source_label: str,
) -> str:
    """Append a compact provenance note to the record."""

    parts = [existing_notes.strip()] if existing_notes.strip() else []
    parts.append(f"seed_batch={batch_label}")
    parts.append(f"selection_rank={selection_rank}/{selected_total}")
    parts.append(f"temp_bin={temperature_bin}")
    parts.append(f"source_label={source_label}")
    return "; ".join(parts)


def build_review_records(
    selected: Sequence[CaptureCandidate],
    *,
    batch_label: str = DEFAULT_BATCH_LABEL,
) -> list[CaptureLabelRecord]:
    """Convert the chosen candidates into review CSV rows."""

    selected_total = len(selected)
    records: list[CaptureLabelRecord] = []
    for index, candidate in enumerate(selected, start=1):
        record = candidate.seed_record
        if record is None:
            continue
        temperature_bin = temperature_bin_name(float(record.temperature_c or 0.0))
        notes = _merge_notes(
            record.notes,
            batch_label=batch_label,
            selection_rank=index,
            selected_total=selected_total,
            temperature_bin=temperature_bin,
            source_label=record.label_source,
        )
        records.append(
            replace(
                record,
                label_quality=batch_label,
                quality_flag="review",
                notes=notes,
                origin_manifest=record.origin_manifest or candidate.origin_manifest,
            )
        )
    return records
