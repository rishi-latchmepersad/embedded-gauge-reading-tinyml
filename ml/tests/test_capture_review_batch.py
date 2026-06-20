"""Regression tests for the captured-image review batch helper."""

from __future__ import annotations

from pathlib import Path

from embedded_gauge_reading_tinyml.capture_labeling import (
    CaptureCandidate,
    CaptureLabelRecord,
    load_capture_candidates,
)
from embedded_gauge_reading_tinyml.capture_review_batch import (
    build_review_records,
    select_review_batch,
    temperature_bin_name,
)


def _make_candidate(
    name: str,
    temperature_c: float,
    label_source: str,
) -> CaptureCandidate:
    """Construct a fully labeled synthetic candidate for unit tests."""

    record = CaptureLabelRecord(
        image_path=Path(f"ml/data/captured_images/{name}.png"),
        source_width=224,
        source_height=224,
        center_x_source=112.0,
        center_y_source=100.0,
        tip_x_source=140.0,
        tip_y_source=120.0,
        temperature_c=temperature_c,
        label_quality="manual",
        quality_flag="clean",
        notes="",
        label_source=label_source,
        origin_manifest="test",
    )
    return CaptureCandidate(
        image_path=record.image_path,
        source_width=record.source_width,
        source_height=record.source_height,
        origin_manifest="test",
        seed_record=record,
    )


def test_temperature_bin_name_covers_the_expected_ranges() -> None:
    """The binning helper should partition the review batch by temperature."""

    assert temperature_bin_name(-1.0) == "lt0"
    assert temperature_bin_name(0.0) == "0-10"
    assert temperature_bin_name(19.9) == "10-20"
    assert temperature_bin_name(29.9) == "20-30"
    assert temperature_bin_name(39.9) == "30-40"
    assert temperature_bin_name(40.0) == "40+"


def test_select_review_batch_round_robins_temperature_bins() -> None:
    """The sampler should spread the batch across the temperature range."""

    candidates = [
        _make_candidate(f"cold_{i}", -10.0, "inverse_mapping") for i in range(2)
    ] + [
        _make_candidate(f"mild_{i}", 5.0, "inverse_mapping") for i in range(2)
    ] + [
        _make_candidate(f"warm_{i}", 15.0, "inverse_mapping") for i in range(2)
    ] + [
        _make_candidate(f"hot_{i}", 25.0, "inverse_mapping") for i in range(2)
    ] + [
        _make_candidate(f"hotter_{i}", 35.0, "inverse_mapping") for i in range(2)
    ] + [
        _make_candidate(f"max_{i}", 45.0, "inverse_mapping") for i in range(2)
    ]

    selected = select_review_batch(candidates, limit=6)

    assert [
        temperature_bin_name(candidate.seed_record.temperature_c or 0.0)
        for candidate in selected
    ] == ["lt0", "0-10", "10-20", "20-30", "30-40", "40+"]


def test_select_review_batch_prefers_manual_verification_within_bin() -> None:
    """Manual-verified rows should be chosen before inverse-mapped rows."""

    candidates = [
        _make_candidate("inverse", 28.0, "inverse_mapping"),
        _make_candidate("manual", 28.0, "manual_verification"),
    ]

    selected = select_review_batch(candidates, limit=1)

    assert selected[0].seed_record is not None
    assert selected[0].seed_record.label_source == "manual_verification"


def test_build_review_records_adds_batch_provenance() -> None:
    """The exported review CSV should carry a compact provenance note."""

    candidate = _make_candidate("sample", 28.0, "manual_verification")
    record = build_review_records(
        [candidate],
        batch_label="captured_image_review_batch_50",
    )[0]

    assert record.label_quality == "captured_image_review_batch_50"
    assert record.quality_flag == "review"
    assert "seed_batch=captured_image_review_batch_50" in record.notes
    assert "temp_bin=20-30" in record.notes
    assert "source_label=manual_verification" in record.notes


def test_real_board_manifest_produces_fifty_images() -> None:
    """The board-capture manifest should provide a full 50-image starter batch."""

    manifest_path = Path("data/board_captures_labeled_v2.csv")
    candidates = load_capture_candidates(manifest_path)
    selected = select_review_batch(candidates, limit=50)

    assert len(selected) == 50
    assert all(candidate.seed_record is not None for candidate in selected)
    assert {candidate.image_path.suffix.lower() for candidate in selected} <= {
        ".png",
        ".jpg",
        ".jpeg",
    }
