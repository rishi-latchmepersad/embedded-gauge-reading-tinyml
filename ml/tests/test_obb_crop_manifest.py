"""Tests for the OBB crop manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path

from embedded_gauge_reading_tinyml.obb_crop_manifest import (
    ObbCropRecord,
    dump_obb_crop_manifest,
    load_obb_crop_manifest,
    load_obb_crop_overrides,
    resolve_crop_box_override,
)


def test_load_manifest_and_resolve_overrides(tmp_path: Path) -> None:
    """Accepted overrides should replace the default crop, rejected ones should not."""

    manifest_path = tmp_path / "obb_crop_manifest.json"
    payload = {
        "schema_version": "obb_crop_manifest.v1",
        "images": [
            {
                "image_path": "ml/data/captured_images/sample.png",
                "source_width": 800,
                "source_height": 600,
                "crop_x_min": 10.0,
                "crop_y_min": 20.0,
                "crop_x_max": 110.0,
                "crop_y_max": 220.0,
                "confidence": 0.91,
                "accepted": True,
                "fallback_reason": "",
            },
            {
                "image_path": str(Path("/tmp/outside_repo/example.png")),
                "source_width": 320,
                "source_height": 240,
                "crop_x_min": 1.0,
                "crop_y_min": 2.0,
                "crop_x_max": 3.0,
                "crop_y_max": 4.0,
                "confidence": 0.25,
                "accepted": False,
                "fallback_reason": "crop outside training window",
            },
        ],
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    records = load_obb_crop_manifest(manifest_path)
    assert len(records) == 2
    assert records[0].accepted is True
    assert records[1].accepted is False

    overrides = load_obb_crop_overrides(manifest_path)
    default_crop = (0.0, 0.0, 50.0, 50.0)

    accepted_crop, accepted_record = resolve_crop_box_override(
        Path("ml/data/captured_images/sample.png"),
        default_crop,
        overrides,
    )
    assert accepted_record is not None
    assert accepted_crop == (10.0, 20.0, 110.0, 220.0)

    rejected_crop, rejected_record = resolve_crop_box_override(
        Path("/tmp/outside_repo/example.png"),
        default_crop,
        overrides,
        require_accepted=True,
    )
    assert rejected_record is not None
    assert rejected_crop == default_crop

    permissive_crop, permissive_record = resolve_crop_box_override(
        Path("/tmp/outside_repo/example.png"),
        default_crop,
        overrides,
    )
    assert permissive_record is not None
    assert permissive_crop == (1.0, 2.0, 3.0, 4.0)


def test_dump_manifest_roundtrip(tmp_path: Path) -> None:
    """Writing and reloading the manifest should preserve the crop record."""

    output_path = tmp_path / "obb_crop_manifest.json"
    record = ObbCropRecord(
        image_path=Path("ml/data/captured_images/sample.png"),
        source_width=640,
        source_height=480,
        crop_box_xyxy=(5.0, 6.0, 7.0, 8.0),
        confidence=0.77,
        accepted=True,
        fallback_reason=None,
        source_kind="reviewed_geometry",
    )
    dump_obb_crop_manifest(
        [record],
        output_path,
        model_path=Path("ml/artifacts/training/obb_v2_box_20260622_203432/model_int8.tflite"),
        source_manifest=Path("ml/data/labelled_captured_images.json"),
    )

    roundtripped = load_obb_crop_manifest(output_path)
    assert len(roundtripped) == 1
    assert roundtripped[0].image_path == Path("ml/data/captured_images/sample.png")
    assert roundtripped[0].crop_box_xyxy == (5.0, 6.0, 7.0, 8.0)
    assert roundtripped[0].confidence == 0.77
