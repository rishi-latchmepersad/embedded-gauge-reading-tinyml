"""Regression tests for the merged captured-image label manifest."""

from __future__ import annotations

import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "ml" / "data" / "labelled_captured_images.json"


def _load_manifest() -> dict[str, object]:
    """Load the merged manifest from disk."""

    assert MANIFEST_PATH.exists(), f"Missing manifest: {MANIFEST_PATH}"
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_has_expected_shape() -> None:
    """The merged manifest should expose the top-level keys we rely on."""

    manifest = _load_manifest()
    assert manifest["schema_version"] == 1
    assert isinstance(manifest["source_manifests"], list)
    assert isinstance(manifest["images"], list)
    assert len(manifest["source_manifests"]) == 4
    assert manifest["image_count"] == 401
    assert manifest["annotation_count"] == 420


def test_manifest_includes_expected_source_families() -> None:
    """The merged manifest should only include the trusted source pool."""

    manifest = _load_manifest()
    source_manifests = {
        str(entry["source_manifest"])
        for entry in manifest["source_manifests"]  # type: ignore[index]
    }
    expected = {
        "ml/data/geometry_reader_manifest_v2_clean.csv",
        "ml/data/hard_cases.csv",
        "ml/data/hard_cases_plus_board30_valid_with_new5.csv",
        "ml/data/new_labelled_captures4.csv",
    }
    assert source_manifests == expected


def test_manifest_includes_clean_pxl_geometry_fields() -> None:
    """The clean PXL geometry rows should keep their center, tip, and angle labels."""

    manifest = _load_manifest()
    geometry_rows: list[dict[str, object]] = []
    for entry in manifest["images"]:  # type: ignore[index]
        for annotation in entry["annotations"]:  # type: ignore[index]
            if str(annotation["source_manifest"]) == "ml/data/geometry_reader_manifest_v2_clean.csv":
                geometry_rows.append(annotation["source_row"])  # type: ignore[index]

    assert geometry_rows
    row = geometry_rows[0]
    for key in (
        "loose_crop_x1",
        "loose_crop_y1",
        "loose_crop_x2",
        "loose_crop_y2",
        "center_x_source",
        "center_y_source",
        "tip_x_source",
        "tip_y_source",
        "dial_radius_source",
        "angle_degrees_from_labels",
        "true_angle_degrees",
    ):
        assert key in row
        assert math.isfinite(float(row[key]))
    assert abs(float(row["true_angle_degrees"]) - float(row["angle_degrees_from_labels"])) < 1e-3


def test_manifest_derives_true_angles_for_temperature_labels() -> None:
    """Temperature-labeled rows should all carry a derived needle angle."""

    manifest = _load_manifest()
    temperature_labeled_rows = 0
    for entry in manifest["images"]:  # type: ignore[index]
        for annotation in entry["annotations"]:  # type: ignore[index]
            row = annotation["source_row"]  # type: ignore[index]
            temperature = row.get("temperature_c")
            value = row.get("value")
            if temperature is None and value is None:
                continue
            temperature_labeled_rows += 1
            assert "true_angle_degrees" in row
            assert math.isfinite(float(row["true_angle_degrees"]))

    assert temperature_labeled_rows == manifest["annotation_count"]
