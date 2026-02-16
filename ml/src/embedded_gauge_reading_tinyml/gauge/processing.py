"""
Gauge processing and calibration utilities
i.e. the math to convert CVAT labels into gauge values, and loading gauge specs.
"""

from dataclasses import dataclass
import math  # used to compute needle angles
from pathlib import Path  # used to search for files
import tomllib  # used for the gauge parameters file

from embedded_gauge_reading_tinyml.dataset import Sample

CALIBRATION_TOML_PATH: Path = (  # Default location for gauge calibration parameters.
    Path(__file__).resolve().parent / "gauge_calibration_parameters.toml"
)  # Keep this near the package for portability.


@dataclass(frozen=True)
class GaugeSpec:
    """Per-gauge calibration: min angle, sweep, and value range for each
    gauge that we want to measure."""

    gauge_id: str  # Identify which gauge this spec applies to.
    min_angle_rad: float  # Angle in radians where the gauge reads min_value.
    sweep_rad: float  # Total clockwise sweep in radians.
    min_value: float  # Gauge value at min_angle_rad.
    max_value: float  # Gauge value at min_angle_rad + sweep_rad.


def needle_angle_clockwise_rad(sample: Sample) -> float:
    """Return raw needle angle in image coords (0 at 3 o'clock, clockwise positive)."""
    dx: float = sample.tip.x - sample.center.x  # Use dx for the needle vector.
    dy: float = sample.tip.y - sample.center.y  # Use dy for the needle vector.
    angle: float = math.atan2(dy, dx)  # atan2 matches clockwise in image coords.
    return angle  # Return raw angle for calibration.


def needle_fraction(sample: Sample, spec: GaugeSpec, *, strict: bool = True) -> float:
    """Return normalized [0, 1] needle position within the entire gauge sweep."""
    raw_angle: float = needle_angle_clockwise_rad(sample)  # Compute raw angle.
    shifted: float = (raw_angle - spec.min_angle_rad) % (
        2.0 * math.pi
    )  # Align the angle according to the min angle calibration, and take a mod with 2 pi to ensure it's positive
    if (
        strict and shifted > spec.sweep_rad
    ):  # Reject labels outside the sweep if strict.
        raise ValueError(
            "Needle angle outside sweep for gauge. Please gauge check calibration."
        )  # Fail fast on bad labels.
    fraction: float = min(
        shifted / spec.sweep_rad, 1.0
    )  # Clamp non-strict cases to the max value
    return fraction  # Return normalized position.


def needle_value(sample: Sample, spec: GaugeSpec, *, strict: bool = True) -> float:
    """Return gauge value scaled to [min_value, max_value] using the sweep fraction."""
    fraction: float = needle_fraction(
        sample, spec, strict=strict
    )  # Convert to normalized fraction.
    value: float = spec.min_value + fraction * (
        spec.max_value - spec.min_value
    )  # Linear scaling (across the entire gauge sweep)
    return value  # Return calibrated value.


def load_gauge_specs(path: Path = CALIBRATION_TOML_PATH) -> dict[str, GaugeSpec]:
    """Load per-gauge specs from a TOML file."""
    raw: dict[str, dict[str, float]] = tomllib.loads(  # Parse TOML text.
        path.read_text(encoding="utf-8")  # Read file contents as UTF-8.
    )
    specs: dict[str, GaugeSpec] = {}  # Prepare the output mapping.
    for gauge_id, spec_dict in raw.items():  # Iterate over each gauge section.
        min_rad: float = math.radians(spec_dict["min_deg"])  # Convert min to radians.
        sweep_rad: float = math.radians(
            spec_dict["sweep_deg"]
        )  # Convert sweep to radians.
        specs[gauge_id] = GaugeSpec(  # Build a typed GaugeSpec.
            gauge_id=gauge_id,  # Preserve the gauge identifier.
            min_angle_rad=min_rad,  # Store min angle in radians.
            sweep_rad=sweep_rad,  # Store sweep in radians.
            min_value=spec_dict["min_value"],  # Store min value.
            max_value=spec_dict["max_value"],  # Store max value.
        )
    return specs  # Return the completed mapping.
