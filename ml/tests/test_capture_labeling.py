"""Regression tests for the captured-image labeling helpers."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

from embedded_gauge_reading_tinyml.capture_labeling import (
    CaptureLabelRecord,
    load_capture_candidates,
    load_label_records,
    write_label_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILD_MANIFEST_PATH = PROJECT_ROOT / "ml" / "scripts" / "build_labelled_captured_images_manifest.py"


def _load_builder_module():
    """Import the manifest builder from its script path."""

    spec = importlib.util.spec_from_file_location("build_labelled_manifest", BUILD_MANIFEST_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_label_record_csv_round_trip(tmp_path: Path) -> None:
    """The review CSV should preserve the raw labels we edit by hand."""

    record = CaptureLabelRecord(
        image_path=Path("ml/data/captured_images/capture_0c.png"),
        source_width=224,
        source_height=224,
        center_x_source=112.0,
        center_y_source=99.9,
        tip_x_source=65.0,
        tip_y_source=95.0,
        temperature_c=0.0,
        label_quality="manual",
        quality_flag="clean",
        notes="needle near 0",
        label_source="manual_gui",
        origin_manifest="scan:ml/data/captured_images",
    )
    output_csv = tmp_path / "labels.csv"

    write_label_records(output_csv, [record])
    loaded = load_label_records(output_csv)

    assert record.image_path.as_posix() in loaded
    round_tripped = loaded[record.image_path.as_posix()]
    assert round_tripped.image_path == record.image_path
    assert round_tripped.source_width == record.source_width
    assert round_tripped.source_height == record.source_height
    assert round_tripped.center_x_source == pytest.approx(record.center_x_source or 0.0)
    assert round_tripped.center_y_source == pytest.approx(record.center_y_source or 0.0)
    assert round_tripped.tip_x_source == pytest.approx(record.tip_x_source or 0.0)
    assert round_tripped.tip_y_source == pytest.approx(record.tip_y_source or 0.0)
    assert round_tripped.temperature_c == pytest.approx(record.temperature_c or 0.0)
    assert round_tripped.quality_flag == "clean"


def test_load_capture_candidates_skips_preview_derivatives(tmp_path: Path) -> None:
    """Directory scans should ignore preview variants by default."""

    original = tmp_path / "capture_0c.png"
    preview = tmp_path / "capture_0c_preview.png"
    Image.new("RGB", (16, 16), (255, 0, 0)).save(original)
    Image.new("RGB", (16, 16), (0, 255, 0)).save(preview)

    candidates = load_capture_candidates(tmp_path)

    assert [candidate.image_path.name for candidate in candidates] == ["capture_0c.png"]
    assert candidates[0].source_width == 16
    assert candidates[0].source_height == 16


def test_reviewed_csv_merges_into_grouped_manifest(tmp_path: Path) -> None:
    """A reviewed-label CSV should merge as a new source family in the manifest."""

    builder = _load_builder_module()
    reviewed_csv = tmp_path / "reviewed_labels.csv"
    reviewed_record = CaptureLabelRecord(
        image_path=Path("ml/data/captured_images/capture_0c.png"),
        source_width=224,
        source_height=224,
        center_x_source=112.0,
        center_y_source=99.9,
        tip_x_source=65.0,
        tip_y_source=95.0,
        temperature_c=0.0,
        label_quality="manual",
        quality_flag="clean",
        notes="needle near 0",
        label_source="manual_gui",
        origin_manifest="scan:ml/data/captured_images",
    )
    with reviewed_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reviewed_record.to_csv_row().keys()))
        writer.writeheader()
        writer.writerow(reviewed_record.to_csv_row())

    extra_sources = builder._reviewed_sources_from_paths([reviewed_csv])
    payload = builder._build_manifest(extra_sources=extra_sources)

    assert any(
        source["source_kind"] == builder.REVIEWED_GEOMETRY_KIND
        for source in payload["source_manifests"]
    )
    capture_entries = [
        entry
        for entry in payload["images"]
        if entry["image_path"] == "ml/data/captured_images/capture_0c.png"
    ]
    assert capture_entries
    annotations = capture_entries[0]["annotations"]
    reviewed_annotations = [
        annotation
        for annotation in annotations
        if annotation["source_kind"] == builder.REVIEWED_GEOMETRY_KIND
    ]
    assert reviewed_annotations
    reviewed_row = reviewed_annotations[0]["source_row"]
    assert reviewed_row["center_x_source"] == pytest.approx(112.0)
    assert reviewed_row["true_angle_degrees"] == pytest.approx(
        float(reviewed_record.to_csv_row()["true_angle_degrees"])
    )
