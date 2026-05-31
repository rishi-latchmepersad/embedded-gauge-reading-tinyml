"""
Gauge processing and calibration utilities
i.e. the math to convert CVAT labels into gauge values, and loading gauge specs.
"""

from dataclasses import dataclass
import math  # used to compute needle angles
from pathlib import Path  # used to search for files
import tomllib  # used for the gauge parameters file
from typing import Any

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
    units: str = ""  # Engineering units (for example "C" or "psi").
    direction: str = "clockwise"  # Needle sweep direction in image coordinates.
    needle_colour: str = "dark"  # Needle colour: "dark" or "light".


def value_to_fraction(value: float, spec: GaugeSpec) -> float:
    """Map an engineering value into a normalized [0, 1] sweep fraction."""
    span: float = spec.max_value - spec.min_value
    if span <= 0.0:
        raise ValueError("Gauge spec must have max_value > min_value.")
    return min(max((value - spec.min_value) / span, 0.0), 1.0)


def fraction_to_value(fraction: float, spec: GaugeSpec) -> float:
    """Map a normalized sweep fraction back into engineering units."""
    fraction_clamped: float = min(max(fraction, 0.0), 1.0)
    return spec.min_value + fraction_clamped * (spec.max_value - spec.min_value)


def fraction_to_angle_rad(fraction: float, spec: GaugeSpec) -> float:
    """Map a normalized sweep fraction into the gauge angle frame."""
    fraction_clamped: float = min(max(fraction, 0.0), 1.0)
    return spec.min_angle_rad + fraction_clamped * spec.sweep_rad


def angle_rad_to_fraction(angle_rad: float, spec: GaugeSpec, *, strict: bool = True) -> float:
    """Map an angle in image coords to a normalized sweep fraction."""
    shifted: float = (angle_rad - spec.min_angle_rad) % (2.0 * math.pi)
    if strict and shifted > spec.sweep_rad:
        raise ValueError(
            "Needle angle outside sweep for gauge. Please gauge check calibration."
        )
    return min(shifted / spec.sweep_rad, 1.0)


def needle_angle_clockwise_rad(sample: Sample) -> float:
    """Return raw needle angle in image coords (0 at 3 o'clock, clockwise positive)."""
    dx: float = sample.tip.x - sample.center.x  # Use dx for the needle vector.
    dy: float = sample.tip.y - sample.center.y  # Use dy for the needle vector.
    angle: float = math.atan2(dy, dx)  # atan2 matches clockwise in image coords.
    return angle  # Return raw angle for calibration.


def needle_fraction(sample: Sample, spec: GaugeSpec, *, strict: bool = True) -> float:
    """Return normalized [0, 1] needle position within the entire gauge sweep."""
    raw_angle: float = needle_angle_clockwise_rad(sample)  # Compute raw angle.
    fraction: float = angle_rad_to_fraction(raw_angle, spec, strict=strict)
    return fraction  # Return normalized position.


def needle_value(sample: Sample, spec: GaugeSpec, *, strict: bool = True) -> float:
    """Return gauge value scaled to [min_value, max_value] using the sweep fraction."""
    fraction: float = needle_fraction(
        sample, spec, strict=strict
    )  # Convert to normalized fraction.
    value: float = fraction_to_value(fraction, spec)
    return value  # Return calibrated value.


def needle_unit_xy_from_value(value: float, spec: GaugeSpec) -> tuple[float, float]:
    """Convert a calibrated gauge value back into the corresponding unit needle vector."""
    fraction: float = value_to_fraction(value, spec)
    angle: float = fraction_to_angle_rad(fraction, spec)
    unit_dx: float = math.cos(angle)
    unit_dy: float = math.sin(angle)
    return (unit_dx, unit_dy)


def load_gauge_specs(path: Path = CALIBRATION_TOML_PATH) -> dict[str, GaugeSpec]:
    """Load per-gauge specs from a TOML file."""
    raw: dict[str, dict[str, Any]] = tomllib.loads(  # Parse TOML text.
        path.read_text(encoding="utf-8")  # Read file contents as UTF-8.
    )
    specs: dict[str, GaugeSpec] = {}  # Prepare the output mapping.
    for gauge_id, spec_dict in raw.items():  # Iterate over each gauge section.
        min_deg = float(spec_dict["min_deg"])
        sweep_deg = float(spec_dict["sweep_deg"])
        min_value = float(spec_dict["min_value"])
        max_value = float(spec_dict["max_value"])
        units = str(spec_dict.get("units", "")).strip()
        direction = str(spec_dict.get("direction", "clockwise")).strip().lower()
        needle_colour = str(spec_dict.get("needle_colour", "dark")).strip().lower()
        if needle_colour not in {"dark", "light"}:
            raise ValueError(
                f"{gauge_id}: needle_colour must be 'dark' or 'light', got '{needle_colour}'."
            )
        if sweep_deg <= 0.0:
            raise ValueError(f"{gauge_id}: sweep_deg must be > 0.")
        if max_value <= min_value:
            raise ValueError(f"{gauge_id}: max_value must be > min_value.")
        if direction not in {"clockwise", "counterclockwise"}:
            raise ValueError(
                f"{gauge_id}: direction must be 'clockwise' or 'counterclockwise'."
            )
        min_rad: float = math.radians(min_deg)  # Convert min to radians.
        sweep_rad: float = math.radians(sweep_deg)  # Convert sweep to radians.
        specs[gauge_id] = GaugeSpec(  # Build a typed GaugeSpec.
            gauge_id=gauge_id,  # Preserve the gauge identifier.
            min_angle_rad=min_rad,  # Store min angle in radians.
            sweep_rad=sweep_rad,  # Store sweep in radians.
            min_value=min_value,  # Store min value.
            max_value=max_value,  # Store max value.
            units=units,
            direction=direction,
            needle_colour=needle_colour,
        )
    return specs  # Return the completed mapping.
