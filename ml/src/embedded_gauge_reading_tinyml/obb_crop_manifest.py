"""Helpers for storing and reusing OBB-generated crop boxes.

The manifest format stays intentionally small:
- one record per source image,
- the decoded crop box in source-image pixels,
- the model confidence and accept/reject decision, and
- enough metadata to keep the downstream trainer honest.

The training pipeline can use these records as an optional crop override layer
without changing the underlying sample selection or label logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ML_ROOT: Path = Path(__file__).resolve().parents[2]
REPO_ROOT: Path = ML_ROOT.parent


@dataclass(frozen=True, slots=True)
class ObbCropRecord:
    """One OBB crop decision for a single source image."""

    image_path: Path
    source_width: int
    source_height: int
    crop_box_xyxy: tuple[float, float, float, float]
    confidence: float
    accepted: bool
    fallback_reason: str | None = None
    model_path: Path | None = None
    source_kind: str = ""
    obb_box_xyxy: tuple[float, float, float, float] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the record into a JSON-friendly dictionary."""

        crop_x_min, crop_y_min, crop_x_max, crop_y_max = self.crop_box_xyxy
        payload: dict[str, Any] = {
            "image_path": self.image_path.as_posix(),
            "source_width": int(self.source_width),
            "source_height": int(self.source_height),
            "crop_x_min": float(crop_x_min),
            "crop_y_min": float(crop_y_min),
            "crop_x_max": float(crop_x_max),
            "crop_y_max": float(crop_y_max),
            "confidence": float(self.confidence),
            "accepted": bool(self.accepted),
            "fallback_reason": self.fallback_reason or "",
            "source_kind": self.source_kind,
        }
        if self.model_path is not None:
            payload["model_path"] = self.model_path.as_posix()
        if self.obb_box_xyxy is not None:
            box_x_min, box_y_min, box_x_max, box_y_max = self.obb_box_xyxy
            payload["obb_x_min"] = float(box_x_min)
            payload["obb_y_min"] = float(box_y_min)
            payload["obb_x_max"] = float(box_x_max)
            payload["obb_y_max"] = float(box_y_max)
        return payload


def _as_bool(value: Any) -> bool:
    """Parse a permissive boolean value from JSON."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _as_float(value: Any, *, field_name: str) -> float:
    """Parse a finite float value from JSON."""

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for {field_name}: {value!r}") from exc
    if not (parsed == parsed and parsed not in (float("inf"), float("-inf"))):
        raise ValueError(f"Invalid float for {field_name}: {value!r}")
    return float(parsed)


def _as_int(value: Any, *, field_name: str) -> int:
    """Parse a positive integer from JSON."""

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {field_name}: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"Expected a positive integer for {field_name}: {value!r}")
    return int(parsed)


def _normalize_image_path(image_path: Path | str) -> Path:
    """Normalize an image path for stable manifest lookups."""

    candidate = Path(image_path)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(REPO_ROOT)
        except ValueError:
            return candidate.resolve()
    return Path(candidate.as_posix())


def _record_from_json(row: Mapping[str, Any]) -> ObbCropRecord:
    """Convert one JSON object into an ``ObbCropRecord``."""

    image_path = _normalize_image_path(row["image_path"])
    source_width = _as_int(row["source_width"], field_name="source_width")
    source_height = _as_int(row["source_height"], field_name="source_height")
    crop_x_min = _as_float(row["crop_x_min"], field_name="crop_x_min")
    crop_y_min = _as_float(row["crop_y_min"], field_name="crop_y_min")
    crop_x_max = _as_float(row["crop_x_max"], field_name="crop_x_max")
    crop_y_max = _as_float(row["crop_y_max"], field_name="crop_y_max")
    confidence = _as_float(row.get("confidence", 0.0), field_name="confidence")
    accepted = _as_bool(row.get("accepted", False))
    fallback_reason = str(row.get("fallback_reason", "")).strip() or None
    model_path_value = str(row.get("model_path", "")).strip()
    model_path = _normalize_image_path(model_path_value) if model_path_value else None
    source_kind = str(row.get("source_kind", "")).strip()

    obb_box_xyxy: tuple[float, float, float, float] | None = None
    if all(key in row for key in ("obb_x_min", "obb_y_min", "obb_x_max", "obb_y_max")):
        obb_box_xyxy = (
            _as_float(row["obb_x_min"], field_name="obb_x_min"),
            _as_float(row["obb_y_min"], field_name="obb_y_min"),
            _as_float(row["obb_x_max"], field_name="obb_x_max"),
            _as_float(row["obb_y_max"], field_name="obb_y_max"),
        )

    return ObbCropRecord(
        image_path=image_path,
        source_width=source_width,
        source_height=source_height,
        crop_box_xyxy=(crop_x_min, crop_y_min, crop_x_max, crop_y_max),
        confidence=confidence,
        accepted=accepted,
        fallback_reason=fallback_reason,
        model_path=model_path,
        source_kind=source_kind,
        obb_box_xyxy=obb_box_xyxy,
    )


def _manifest_rows(payload: Any) -> list[Mapping[str, Any]]:
    """Extract the crop rows from a manifest payload."""

    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("images", "records", "rows"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, Mapping)]
    raise ValueError("Unsupported OBB crop manifest schema.")


def load_obb_crop_manifest(manifest_path: Path) -> list[ObbCropRecord]:
    """Load a crop manifest from JSON."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [_record_from_json(row) for row in _manifest_rows(payload)]


def load_obb_crop_overrides(manifest_path: Path) -> dict[Path, ObbCropRecord]:
    """Load the manifest into a lookup table keyed by image path."""

    records = load_obb_crop_manifest(manifest_path)
    return {_normalize_image_path(record.image_path): record for record in records}


def resolve_crop_box_override(
    image_path: Path,
    default_crop_box_xyxy: tuple[float, float, float, float],
    overrides: Mapping[Path, ObbCropRecord] | None,
    *,
    require_accepted: bool = False,
) -> tuple[tuple[float, float, float, float], ObbCropRecord | None]:
    """Resolve an optional OBB crop override for one image."""

    if not overrides:
        return default_crop_box_xyxy, None

    record = overrides.get(_normalize_image_path(image_path))
    if record is None:
        return default_crop_box_xyxy, None
    if record.accepted or not require_accepted:
        return record.crop_box_xyxy, record
    return default_crop_box_xyxy, record


def dump_obb_crop_manifest(
    records: list[ObbCropRecord],
    output_path: Path,
    *,
    model_path: Path,
    source_manifest: Path | None = None,
) -> Path:
    """Write a manifest payload to disk."""

    payload: dict[str, Any] = {
        "schema_version": "obb_crop_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path.as_posix(),
        "image_count": len(records),
        "images": [record.to_json_dict() for record in records],
    }
    if source_manifest is not None:
        payload["source_manifest"] = source_manifest.as_posix()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
