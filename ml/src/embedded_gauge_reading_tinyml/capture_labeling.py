"""Helpers for reviewing and exporting captured-image geometry labels.

The annotation GUI uses these helpers so the saved CSV stays compatible with
the existing manifest builder and the combined center + SimCC trainer.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from embedded_gauge_reading_tinyml.firmware_preprocessing import (
    firmware_training_crop_box,
    load_capture_image,
)
from embedded_gauge_reading_tinyml.gauge.processing import (
    GaugeSpec,
    fraction_to_angle_rad,
    load_gauge_specs,
    value_to_fraction,
)

ML_ROOT: Path = Path(__file__).resolve().parents[2]
REPO_ROOT: Path = ML_ROOT.parent
DEFAULT_GAUGE_ID: str = "littlegood_home_temp_gauge_c"
DEFAULT_MANIFEST_PATH: Path = ML_ROOT / "data" / "labelled_captured_images.json"
DEFAULT_OUTPUT_CSV: Path = REPO_ROOT / "tmp" / "captured_image_review_labels.csv"

SUPPORTED_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {
        ".bmp",
        ".jpg",
        ".jpeg",
        ".pgm",
        ".png",
        ".raw",
        ".raw16",
        ".yuv422",
    }
)


@dataclass(frozen=True, slots=True)
class CaptureLabelRecord:
    """One editable label row for a captured image."""

    image_path: Path
    source_width: int
    source_height: int
    center_x_source: float | None = None
    center_y_source: float | None = None
    tip_x_source: float | None = None
    tip_y_source: float | None = None
    temperature_c: float | None = None
    label_quality: str = "manual"
    quality_flag: str = "review"
    notes: str = ""
    label_source: str = "manual_gui"
    origin_manifest: str = ""

    def with_geometry(
        self,
        *,
        center_x_source: float | None,
        center_y_source: float | None,
        tip_x_source: float | None,
        tip_y_source: float | None,
    ) -> CaptureLabelRecord:
        """Return a copy with updated geometry labels."""

        return replace(
            self,
            center_x_source=center_x_source,
            center_y_source=center_y_source,
            tip_x_source=tip_x_source,
            tip_y_source=tip_y_source,
        )

    def with_temperature(self, temperature_c: float | None) -> CaptureLabelRecord:
        """Return a copy with an updated temperature label."""

        return replace(self, temperature_c=temperature_c)

    def with_quality_flag(self, quality_flag: str) -> CaptureLabelRecord:
        """Return a copy with an updated quality flag."""

        return replace(self, quality_flag=quality_flag.strip() or self.quality_flag)

    @property
    def has_center(self) -> bool:
        """Return ``True`` when the record has a center label."""

        return self.center_x_source is not None and self.center_y_source is not None

    @property
    def has_tip(self) -> bool:
        """Return ``True`` when the record has a tip label."""

        return self.tip_x_source is not None and self.tip_y_source is not None

    @property
    def has_temperature(self) -> bool:
        """Return ``True`` when the record has a temperature label."""

        return self.temperature_c is not None

    @property
    def crop_box_xyxy(self) -> tuple[float, float, float, float]:
        """Return the firmware crop box for this source image."""

        return firmware_training_crop_box(self.source_width, self.source_height)

    def to_csv_row(self) -> dict[str, str]:
        """Serialize the record to the CSV schema used by the review tool."""

        true_angle_degrees, angle_source = derive_true_angle_degrees(self)
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = self.crop_box_xyxy
        row: dict[str, str] = {
            "image_path": self.image_path.as_posix(),
            "source_width": str(int(self.source_width)),
            "source_height": str(int(self.source_height)),
            "center_x_source": _format_optional_float(self.center_x_source),
            "center_y_source": _format_optional_float(self.center_y_source),
            "tip_x_source": _format_optional_float(self.tip_x_source),
            "tip_y_source": _format_optional_float(self.tip_y_source),
            "temperature_c": _format_optional_float(self.temperature_c),
            "label_quality": self.label_quality.strip() or "manual",
            "quality_flag": self.quality_flag.strip() or "review",
            "notes": self.notes.strip(),
            "label_source": self.label_source.strip() or "manual_gui",
            "origin_manifest": self.origin_manifest.strip(),
            "angle_source": angle_source,
            "true_angle_degrees": _format_optional_float(true_angle_degrees),
            "center_tip_distance_pixels": _format_optional_float(
                center_tip_distance_pixels(self)
            ),
            "crop_x_min": _format_optional_float(crop_x_min),
            "crop_y_min": _format_optional_float(crop_y_min),
            "crop_x_max": _format_optional_float(crop_x_max),
            "crop_y_max": _format_optional_float(crop_y_max),
        }
        return row


@dataclass(frozen=True, slots=True)
class CaptureCandidate:
    """One image to review in the labeling GUI."""

    image_path: Path
    source_width: int
    source_height: int
    origin_manifest: str
    seed_record: CaptureLabelRecord | None = None


def _format_optional_float(value: float | None) -> str:
    """Format an optional float for CSV output."""

    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _parse_optional_float(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    """Return the first finite float found under the requested keys."""

    for key in keys:
        value = row.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return float(parsed)
    return None


def _parse_optional_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    """Return the first non-empty string found under the requested keys."""

    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _parse_optional_int(row: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    """Return the first finite integer found under the requested keys."""

    value = _parse_optional_float(row, keys)
    if value is None:
        return None
    return int(round(value))


def is_original_capture_filename(filename: str) -> bool:
    """Match the same original-capture filter used by the trainer."""

    name = Path(filename).name
    return not any(tag in name for tag in (".gray.", "_preview", "_yuy2", "_glare_"))


def to_repo_relative_path(image_path: Path | str) -> Path:
    """Return a repo-relative path when the input points inside this checkout."""

    candidate = Path(image_path)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(REPO_ROOT)
        except ValueError:
            return candidate

    normalized = Path(str(candidate).replace("\\", "/"))
    if normalized.parts[:2] == ("ml", "data"):
        return normalized
    if normalized.parts[:1] == ("captured_images",):
        return Path("ml/data") / normalized
    if normalized.parts[:1] == ("raw",):
        return Path("ml/data") / normalized

    repo_candidate = (REPO_ROOT / normalized).resolve()
    try:
        return repo_candidate.relative_to(REPO_ROOT)
    except ValueError:
        return normalized


def resolve_absolute_image_path(image_path: Path | str) -> Path:
    """Resolve a repo-relative image path to an absolute filesystem path."""

    candidate = Path(image_path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _default_gauge_spec() -> GaugeSpec:
    """Return the single gauge spec used by the captured-image dataset."""

    specs = load_gauge_specs()
    if DEFAULT_GAUGE_ID in specs:
        return specs[DEFAULT_GAUGE_ID]
    if len(specs) == 1:
        return next(iter(specs.values()))
    available = ", ".join(sorted(specs))
    raise ValueError(
        "Expected exactly one gauge calibration spec for the captured-image dataset; "
        f"available gauge ids: {available}"
    )


def temperature_to_true_angle_degrees(temperature_c: float, spec: GaugeSpec | None = None) -> float:
    """Map a temperature label to the corresponding needle angle in degrees."""

    gauge_spec = spec if spec is not None else _default_gauge_spec()
    fraction = value_to_fraction(temperature_c, gauge_spec)
    angle_rad = fraction_to_angle_rad(fraction, gauge_spec)
    return math.degrees(angle_rad) % 360.0


def geometry_angle_degrees(record: CaptureLabelRecord) -> float | None:
    """Compute a needle angle from a labeled center and tip."""

    if not record.has_center or not record.has_tip:
        return None
    dx = float(record.tip_x_source) - float(record.center_x_source)
    dy = float(record.tip_y_source) - float(record.center_y_source)
    return math.degrees(math.atan2(dy, dx)) % 360.0


def derive_true_angle_degrees(
    record: CaptureLabelRecord,
    *,
    spec: GaugeSpec | None = None,
) -> tuple[float | None, str]:
    """Return the best angle estimate and the source it came from."""

    if record.temperature_c is not None:
        return temperature_to_true_angle_degrees(record.temperature_c, spec), "temperature"
    angle = geometry_angle_degrees(record)
    if angle is not None:
        return angle, "geometry"
    return None, "missing"


def angle_difference_degrees(a: float, b: float) -> float:
    """Return the smallest absolute difference between two angles."""

    diff = abs((float(a) - float(b)) % 360.0)
    return min(diff, 360.0 - diff)


def center_tip_distance_pixels(record: CaptureLabelRecord) -> float | None:
    """Return the Euclidean distance between the center and tip labels."""

    if not record.has_center or not record.has_tip:
        return None
    dx = float(record.tip_x_source) - float(record.center_x_source)
    dy = float(record.tip_y_source) - float(record.center_y_source)
    return float(math.hypot(dx, dy))


def map_source_point_to_canvas_norm(
    point_xy: tuple[float, float],
    *,
    crop_box_xyxy: tuple[float, float, float, float],
    source_width: int,
    source_height: int,
    image_size: int = 224,
) -> tuple[float, float]:
    """Map a source-space point into the 224x224 crop-plus-pad canvas."""

    x_min, y_min, x_max, y_max = crop_box_xyxy
    x_min_i = max(0.0, float(math.floor(x_min)))
    y_min_i = max(0.0, float(math.floor(y_min)))
    x_max_i = min(float(source_width), float(math.ceil(x_max)))
    y_max_i = min(float(source_height), float(math.ceil(y_max)))
    if x_max_i <= x_min_i:
        x_max_i = min(float(source_width), x_min_i + 1.0)
    if y_max_i <= y_min_i:
        y_max_i = min(float(source_height), y_min_i + 1.0)

    crop_width = max(x_max_i - x_min_i, 1.0)
    crop_height = max(y_max_i - y_min_i, 1.0)
    scale = min(float(image_size) / crop_width, float(image_size) / crop_height)
    resized_width = crop_width * scale
    resized_height = crop_height * scale
    pad_x = 0.5 * (float(image_size) - resized_width)
    pad_y = 0.5 * (float(image_size) - resized_height)

    point_x, point_y = point_xy
    canvas_x = (float(point_x) - x_min_i) * scale + pad_x
    canvas_y = (float(point_y) - y_min_i) * scale + pad_y
    return (
        float(np.clip(canvas_x / float(image_size), 0.0, 1.0)),
        float(np.clip(canvas_y / float(image_size), 0.0, 1.0)),
    )


def resolve_source_size(
    image_path: Path,
    row: Mapping[str, Any] | None = None,
    *,
    raw_width_hint: int | None = None,
    raw_height_hint: int | None = None,
) -> tuple[int, int]:
    """Resolve the source image dimensions from row metadata or the file itself."""

    if row is not None:
        width = _parse_optional_int(row, ("source_width", "width", "image_width"))
        height = _parse_optional_int(row, ("source_height", "height", "image_height"))
        if width is not None and height is not None and width > 0 and height > 0:
            return width, height

    absolute_path = resolve_absolute_image_path(image_path)
    suffix = absolute_path.suffix.lower()
    if suffix in {".bmp", ".jpg", ".jpeg", ".pgm", ".png"}:
        from PIL import Image

        with Image.open(absolute_path) as image:
            width, height = image.size
        return int(width), int(height)

    raw_bytes = absolute_path.read_bytes()
    raw_size = len(raw_bytes)
    if suffix == ".yuv422":
        if raw_width_hint is not None and raw_height_hint is not None:
            expected = int(raw_width_hint) * int(raw_height_hint) * 2
            if expected == raw_size:
                return int(raw_width_hint), int(raw_height_hint)
        pixels = raw_size // 2
        dim = int(round(math.sqrt(float(pixels))))
        if dim > 0 and dim * dim * 2 == raw_size:
            return dim, dim
        raise ValueError(
            f"Cannot infer yuv422 dimensions for {absolute_path}; "
            "pass --raw-width and --raw-height or use a manifest row with source_width/source_height."
        )

    if suffix in {".raw", ".raw16"}:
        if raw_width_hint is not None and raw_height_hint is not None:
            expected_8 = int(raw_width_hint) * int(raw_height_hint)
            expected_16 = expected_8 * 2
            if raw_size in {expected_8, expected_16}:
                return int(raw_width_hint), int(raw_height_hint)
        dim = int(round(math.sqrt(float(raw_size))))
        if dim > 0 and dim * dim == raw_size:
            return dim, dim
        dim16 = int(round(math.sqrt(float(raw_size // 2))))
        if dim16 > 0 and dim16 * dim16 * 2 == raw_size:
            return dim16, dim16
        raise ValueError(
            f"Cannot infer raw dimensions for {absolute_path}; "
            "pass --raw-width and --raw-height or use a manifest row with source_width/source_height."
        )

    raise ValueError(f"Unsupported image suffix for size inference: {absolute_path.suffix}")


def load_image_array(
    image_path: Path,
    *,
    source_width: int,
    source_height: int,
) -> np.ndarray:
    """Load one source image as an RGB uint8 array."""

    absolute_path = resolve_absolute_image_path(image_path)
    image, _kind = load_capture_image(
        absolute_path,
        image_width=source_width,
        image_height=source_height,
    )
    return np.asarray(image, dtype=np.uint8)


def _record_from_mapping(
    mapping: Mapping[str, Any],
    *,
    default_image_path: Path,
    default_source_width: int,
    default_source_height: int,
    origin_manifest: str = "",
) -> CaptureLabelRecord:
    """Build one label record from a CSV or JSON mapping."""

    image_path_text = _parse_optional_text(mapping, ("image_path", "image", "path"))
    image_path = to_repo_relative_path(image_path_text or default_image_path)
    source_width = _parse_optional_int(mapping, ("source_width", "width", "image_width"))
    source_height = _parse_optional_int(mapping, ("source_height", "height", "image_height"))
    if source_width is None:
        source_width = default_source_width
    if source_height is None:
        source_height = default_source_height

    return CaptureLabelRecord(
        image_path=image_path,
        source_width=int(source_width),
        source_height=int(source_height),
        center_x_source=_parse_optional_float(
            mapping,
            ("center_x_source", "center_x", "cx", "center_x_norm"),
        ),
        center_y_source=_parse_optional_float(
            mapping,
            ("center_y_source", "center_y", "cy", "center_y_norm"),
        ),
        tip_x_source=_parse_optional_float(mapping, ("tip_x_source", "tip_x", "tx")),
        tip_y_source=_parse_optional_float(mapping, ("tip_y_source", "tip_y", "ty")),
        temperature_c=_parse_optional_float(
            mapping,
            ("temperature_c", "value", "temp_c", "temp"),
        ),
        label_quality=_parse_optional_text(mapping, ("label_quality", "label_quality_flag"))
        or "manual",
        quality_flag=_parse_optional_text(mapping, ("quality_flag", "quality")) or "review",
        notes=_parse_optional_text(mapping, ("notes", "note", "comment")),
        label_source=_parse_optional_text(mapping, ("label_source", "source_kind"))
        or "manual_gui",
        origin_manifest=origin_manifest,
    )


def record_from_mapping(
    mapping: Mapping[str, Any],
    *,
    default_image_path: Path,
    default_source_width: int,
    default_source_height: int,
    origin_manifest: str = "",
) -> CaptureLabelRecord:
    """Public wrapper for converting a row mapping into a label record."""

    return _record_from_mapping(
        mapping,
        default_image_path=default_image_path,
        default_source_width=default_source_width,
        default_source_height=default_source_height,
        origin_manifest=origin_manifest,
    )


def load_label_records(csv_path: Path) -> dict[str, CaptureLabelRecord]:
    """Load an exported label CSV into a path-indexed record map."""

    if not csv_path.exists():
        return {}

    records: dict[str, CaptureLabelRecord] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}
        for row in reader:
            image_path_text = _parse_optional_text(row, ("image_path", "image", "path"))
            if not image_path_text:
                continue
            image_path = to_repo_relative_path(image_path_text)
            record = _record_from_mapping(
                row,
                default_image_path=image_path,
                default_source_width=_parse_optional_int(row, ("source_width",)) or 0,
                default_source_height=_parse_optional_int(row, ("source_height",)) or 0,
                origin_manifest=_parse_optional_text(row, ("origin_manifest",)),
            )
            records[image_path.as_posix()] = record
    return records


def write_label_records(csv_path: Path, records: Sequence[CaptureLabelRecord]) -> None:
    """Write the label records to disk using a stable CSV column order."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    fieldnames = [
        "image_path",
        "source_width",
        "source_height",
        "center_x_source",
        "center_y_source",
        "tip_x_source",
        "tip_y_source",
        "temperature_c",
        "label_quality",
        "quality_flag",
        "notes",
        "label_source",
        "origin_manifest",
        "angle_source",
        "true_angle_degrees",
        "center_tip_distance_pixels",
        "crop_x_min",
        "crop_y_min",
        "crop_x_max",
        "crop_y_max",
    ]
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.image_path.as_posix()):
            writer.writerow(record.to_csv_row())
    temp_path.replace(csv_path)


def load_capture_candidates(
    input_path: Path,
    *,
    include_derivatives: bool = False,
    recursive: bool = False,
    raw_width_hint: int | None = None,
    raw_height_hint: int | None = None,
) -> list[CaptureCandidate]:
    """Load the images that should appear in the review GUI."""

    if input_path.is_dir():
        return _load_candidates_from_directory(
            input_path,
            include_derivatives=include_derivatives,
            recursive=recursive,
            raw_width_hint=raw_width_hint,
            raw_height_hint=raw_height_hint,
        )
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        return _load_candidates_from_json(
            input_path,
            include_derivatives=include_derivatives,
            raw_width_hint=raw_width_hint,
            raw_height_hint=raw_height_hint,
        )
    if suffix == ".csv":
        return _load_candidates_from_csv(
            input_path,
            include_derivatives=include_derivatives,
            raw_width_hint=raw_width_hint,
            raw_height_hint=raw_height_hint,
        )
    raise ValueError(f"Unsupported manifest type: {input_path}")


def _load_candidates_from_directory(
    input_path: Path,
    *,
    include_derivatives: bool,
    recursive: bool,
    raw_width_hint: int | None,
    raw_height_hint: int | None,
) -> list[CaptureCandidate]:
    """Build candidates directly from an image directory."""

    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    candidates: list[CaptureCandidate] = []
    for entry in sorted(iterator, key=lambda path: path.as_posix()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        repo_relative = to_repo_relative_path(entry)
        if not include_derivatives and not is_original_capture_filename(repo_relative.name):
            continue
        width, height = resolve_source_size(
            repo_relative,
            raw_width_hint=raw_width_hint,
            raw_height_hint=raw_height_hint,
        )
        candidates.append(
            CaptureCandidate(
                image_path=repo_relative,
                source_width=width,
                source_height=height,
                origin_manifest=f"scan:{input_path.as_posix()}",
            )
        )
    return candidates


def _seed_record_from_annotations(
    image_path: Path,
    annotations: Sequence[Mapping[str, Any]],
    *,
    source_width: int,
    source_height: int,
    origin_manifest: str,
) -> CaptureLabelRecord | None:
    """Pick the best existing annotation for a grouped-manifest image."""

    if not annotations:
        return None

    def score(annotation: Mapping[str, Any]) -> tuple[int, int, int, int]:
        row = annotation.get("source_row", {})
        if not isinstance(row, Mapping):
            row = {}
        source_kind = str(annotation.get("source_kind", ""))
        kind_priority = {
            "reviewed_geometry": 4,
            "pxl_geometry": 3,
            "board_tip_geometry": 2,
            "center_radii": 1,
            "center_only": 0,
            "board_geometry": -1,
            "temperature_only": -2,
        }.get(source_kind, -3)
        has_center = int(
            _parse_optional_float(row, ("center_x_source", "center_x", "cx")) is not None
            and _parse_optional_float(row, ("center_y_source", "center_y", "cy")) is not None
        )
        has_tip = int(
            _parse_optional_float(row, ("tip_x_source", "tip_x", "tx")) is not None
            and _parse_optional_float(row, ("tip_y_source", "tip_y", "ty")) is not None
        )
        has_temp = int(
            _parse_optional_float(row, ("temperature_c", "value", "temp_c", "temp")) is not None
        )
        return kind_priority, has_center + has_tip + has_temp, int(annotation.get("source_row_index", 0)), has_temp

    best = max(annotations, key=score)
    row = best.get("source_row", {})
    if not isinstance(row, Mapping):
        row = {}
    return _record_from_mapping(
        row,
        default_image_path=image_path,
        default_source_width=source_width,
        default_source_height=source_height,
        origin_manifest=origin_manifest,
    )


def _load_candidates_from_json(
    input_path: Path,
    *,
    include_derivatives: bool,
    raw_width_hint: int | None,
    raw_height_hint: int | None,
) -> list[CaptureCandidate]:
    """Build candidates from the grouped JSON manifest."""

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, list):
        raise ValueError(f"{input_path} does not contain a grouped manifest")

    candidates: list[CaptureCandidate] = []
    for entry in images:
        if not isinstance(entry, Mapping):
            continue
        image_path_text = _parse_optional_text(entry, ("image_path",))
        if not image_path_text:
            continue
        repo_relative = to_repo_relative_path(image_path_text)
        if not include_derivatives and not is_original_capture_filename(repo_relative.name):
            continue
        annotations = entry.get("annotations", [])
        if not isinstance(annotations, list):
            annotations = []
        first_annotation_row: Mapping[str, Any] | None = None
        for annotation in annotations:
            if isinstance(annotation, Mapping):
                source_row = annotation.get("source_row", {})
                if isinstance(source_row, Mapping):
                    first_annotation_row = source_row
                    break
        width, height = resolve_source_size(
            repo_relative,
            row=first_annotation_row,
            raw_width_hint=raw_width_hint,
            raw_height_hint=raw_height_hint,
        )
        seed_record = _seed_record_from_annotations(
            repo_relative,
            annotations,
            source_width=width,
            source_height=height,
            origin_manifest=input_path.as_posix(),
        )
        candidates.append(
            CaptureCandidate(
                image_path=repo_relative,
                source_width=width,
                source_height=height,
                origin_manifest=input_path.as_posix(),
                seed_record=seed_record,
            )
        )
    return candidates


def _load_candidates_from_csv(
    input_path: Path,
    *,
    include_derivatives: bool,
    raw_width_hint: int | None,
    raw_height_hint: int | None,
) -> list[CaptureCandidate]:
    """Build candidates from a flat CSV manifest."""

    candidates: list[CaptureCandidate] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return candidates
        for row in reader:
            image_path_text = _parse_optional_text(row, ("image_path", "image", "path"))
            if not image_path_text:
                continue
            repo_relative = to_repo_relative_path(image_path_text)
            if not include_derivatives and not is_original_capture_filename(repo_relative.name):
                continue
            width, height = resolve_source_size(
                repo_relative,
                row=row,
                raw_width_hint=raw_width_hint,
                raw_height_hint=raw_height_hint,
            )
            seed_record = _record_from_mapping(
                row,
                default_image_path=repo_relative,
                default_source_width=width,
                default_source_height=height,
                origin_manifest=input_path.as_posix(),
            )
            candidates.append(
                CaptureCandidate(
                    image_path=repo_relative,
                    source_width=width,
                    source_height=height,
                    origin_manifest=input_path.as_posix(),
                    seed_record=seed_record,
                )
            )
    return candidates
