"""Build a grouped manifest for labeled captured gauge images.

The trusted source set is now the clean PXL geometry manifest plus the two
smaller temperature-only annotation sets and the hard cases.  This keeps the
manifest focused on the cleanest center labels instead of the broader board
capture mix.

The builder can also merge an optional reviewed-label CSV so the labeler tool
can refresh the grouped manifest without changing the default source mix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, DefaultDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "ml" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.gauge import (  # noqa: E402
    GaugeSpec,
    fraction_to_angle_rad,
    load_gauge_specs,
    value_to_fraction,
)


@dataclass(frozen=True)
class SourceSpec:
    """Describe one source manifest that should be merged into the JSON."""

    path: Path
    kind: str
    description: str


DATA_DIR = PROJECT_ROOT / "ml" / "data"
DEFAULT_OUTPUT_PATH = DATA_DIR / "labelled_captured_images.json"
DEFAULT_GAUGE_ID = "littlegood_home_temp_gauge_c"
REVIEWED_GEOMETRY_KIND = "reviewed_geometry"

# Keep the source list explicit so the merge stays reproducible and easy to audit.
SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        path=Path("ml/data/geometry_reader_manifest_v2_clean.csv"),
        kind="pxl_geometry",
        description="Clean PXL geometry labels with center, tip, and angle.",
    ),
    SourceSpec(
        path=Path("ml/data/hard_cases.csv"),
        kind="temperature_only",
        description="Trusted hard-case temperature labels.",
    ),
    SourceSpec(
        path=Path("ml/data/hard_cases_plus_board30_valid_with_new5.csv"),
        kind="temperature_only",
        description="Revised hard-case plus board30-valid temperature set.",
    ),
    SourceSpec(
        path=Path("ml/data/new_labelled_captures4.csv"),
        kind="temperature_only",
        description="Newest manual capture annotations.",
    ),
)


def _coerce_scalar(text: str) -> Any:
    """Convert CSV scalars to JSON-friendly values while keeping strings intact."""

    value = text.strip()
    if value == "":
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _float_or_none(row: dict[str, Any], *keys: str) -> float | None:
    """Return the first finite float value found under the requested row keys."""

    for key in keys:
        value = row.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return float(result)
    return None


@lru_cache(maxsize=1)
def _default_gauge_spec() -> GaugeSpec:
    """Return the single gauge calibration spec used by the captured images."""

    specs = load_gauge_specs()
    if DEFAULT_GAUGE_ID in specs:
        return specs[DEFAULT_GAUGE_ID]
    if len(specs) == 1:
        return next(iter(specs.values()))
    available = ", ".join(sorted(specs))
    raise ValueError(
        "Expected exactly one gauge calibration spec for the grouped manifest; "
        f"available gauge ids: {available}"
    )


def _temperature_from_row(row: dict[str, Any]) -> float | None:
    """Extract the best available temperature label from a normalized CSV row."""

    return _float_or_none(row, "temperature_c", "value")


def _temperature_to_true_angle_degrees(temperature_c: float, spec: GaugeSpec) -> float:
    """Map a temperature label to the corresponding needle angle in degrees."""

    fraction = value_to_fraction(temperature_c, spec)
    angle_rad = fraction_to_angle_rad(fraction, spec)
    return math.degrees(angle_rad) % 360.0


def _geometry_angle_degrees(row: dict[str, Any]) -> float | None:
    """Extract the best available angle from geometry labels when present."""

    angle = _float_or_none(row, "true_angle_degrees", "angle_degrees_from_labels", "angle_degrees")
    if angle is not None:
        return angle % 360.0

    center_x = _float_or_none(row, "center_x", "center_x_source")
    center_y = _float_or_none(row, "center_y", "center_y_source")
    tip_x = _float_or_none(row, "tip_x", "tip_x_source")
    tip_y = _float_or_none(row, "tip_y", "tip_y_source")
    if (
        center_x is not None
        and center_y is not None
        and tip_x is not None
        and tip_y is not None
    ):
        return math.degrees(math.atan2(tip_y - center_y, tip_x - center_x)) % 360.0

    return None


def _canonicalize_image_path(raw_path: str) -> str:
    """Normalize the image path into a repo-relative path when possible."""

    candidate = raw_path.strip().replace("\\", "/")
    if candidate == "":
        return candidate

    direct_candidates = [candidate]
    if candidate.startswith("captured_images/"):
        direct_candidates.append(f"ml/data/{candidate}")
    if candidate.startswith("ml/data/"):
        direct_candidates.append(candidate)
    if candidate.startswith("raw/"):
        direct_candidates.append(f"ml/data/{candidate}")

    basename = Path(candidate).name
    direct_candidates.append(f"ml/data/captured_images/{basename}")
    direct_candidates.append(f"ml/data/raw/{basename}")

    seen: set[str] = set()
    for normalized in direct_candidates:
        if normalized in seen:
            continue
        seen.add(normalized)
        if (PROJECT_ROOT / normalized).exists():
            return normalized

    # Fall back to the cleaned source path if we cannot resolve it locally.
    return candidate


def _read_source_rows(source: SourceSpec) -> list[dict[str, Any]]:
    """Load and normalize one CSV source manifest."""

    source_path = PROJECT_ROOT / source.path
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "image_path" not in reader.fieldnames:
            raise ValueError(f"{source.path} does not look like an image manifest")

        rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(reader, start=1):
            raw_image_path = str(row.get("image_path", "")).strip()
            if raw_image_path == "" or raw_image_path.startswith("#"):
                continue
            canonical_image_path = _canonicalize_image_path(raw_image_path)
            if not (
                "/captured_images/" in canonical_image_path
                or canonical_image_path.startswith("captured_images/")
                or "/raw/" in canonical_image_path
                or canonical_image_path.startswith("raw/")
            ):
                continue
            normalized_row = {
                key: _coerce_scalar(str(value)) if value is not None else None
                for key, value in row.items()
            }
            normalized_row["image_path"] = canonical_image_path
            if str(normalized_row.get("quality_flag", "")).strip().lower() == "exclude":
                continue
            rows.append(
                {
                    "image_path": canonical_image_path,
                    "source_manifest": source.path.as_posix(),
                    "source_kind": source.kind,
                    "source_row_index": row_index,
                    "source_image_path": raw_image_path,
                    "source_row": normalized_row,
                }
            )
    return rows


def _build_manifest(*, extra_sources: tuple[SourceSpec, ...] = ()) -> dict[str, Any]:
    """Merge all requested sources into a grouped image manifest."""

    gauge_spec = _default_gauge_spec()
    grouped: DefaultDict[str, list[dict[str, Any]]] = defaultdict(list)
    source_summaries: list[dict[str, Any]] = []

    for source in SOURCE_SPECS + extra_sources:
        rows = _read_source_rows(source)
        image_paths = {row["image_path"] for row in rows}
        source_summaries.append(
            {
                "source_manifest": source.path.as_posix(),
                "source_kind": source.kind,
                "description": source.description,
                "row_count": len(rows),
                "unique_image_count": len(image_paths),
            }
        )
        for row in rows:
            source_row = row["source_row"]
            geometry_angle = _geometry_angle_degrees(source_row)
            if geometry_angle is not None:
                source_row["true_angle_degrees"] = geometry_angle
            else:
                temperature_c = _temperature_from_row(source_row)
                if temperature_c is not None:
                    source_row["true_angle_degrees"] = _temperature_to_true_angle_degrees(
                        temperature_c,
                        gauge_spec,
                    )
            grouped[row["image_path"]].append(row)

    images: list[dict[str, Any]] = []
    for image_path in sorted(grouped):
        annotations = sorted(
            grouped[image_path],
            key=lambda item: (item["source_manifest"], item["source_row_index"]),
        )
        images.append(
            {
                "image_path": image_path,
                "annotation_count": len(annotations),
                "annotations": annotations,
            }
        )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifests": source_summaries,
        "image_count": len(images),
        "annotation_count": sum(item["annotation_count"] for item in images),
        "images": images,
    }


def _reviewed_sources_from_paths(reviewed_paths: list[Path]) -> tuple[SourceSpec, ...]:
    """Convert reviewed CSV paths into extra source specs."""

    reviewed_specs: list[SourceSpec] = []
    for reviewed_path in reviewed_paths:
        reviewed_specs.append(
            SourceSpec(
                path=reviewed_path,
                kind=REVIEWED_GEOMETRY_KIND,
                description=f"Reviewed labels from {reviewed_path.name}.",
            )
        )
    return tuple(reviewed_specs)


def main() -> None:
    """Write the grouped manifest to disk."""

    parser = argparse.ArgumentParser(
        description="Build labelled_captured_images.json from source CSV manifests."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path. Defaults to ml/data/labelled_captured_images.json.",
    )
    parser.add_argument(
        "--extra-source",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional reviewed-label CSV to merge as an additional geometry source. "
            "Can be passed multiple times."
        ),
    )
    args = parser.parse_args()

    extra_sources = _reviewed_sources_from_paths(list(args.extra_source))
    payload = _build_manifest(extra_sources=extra_sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Wrote {args.output} with {payload['image_count']} images and "
        f"{payload['annotation_count']} annotations."
    )


if __name__ == "__main__":
    main()
