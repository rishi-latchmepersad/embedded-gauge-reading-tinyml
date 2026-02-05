"""Unit tests for gauge processing helpers."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from embedded_gauge_reading_tinyml.dataset import EllipseLabel, PointLabel, Sample
from embedded_gauge_reading_tinyml.gauge import (
    GaugeSpec,
    load_gauge_specs,
    needle_angle_clockwise_rad,
    needle_fraction,
    needle_value,
)


def build_sample(
    tip_x: float,
    tip_y: float,
    *,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> Sample:
    """Create a minimal Sample with just enough labels for angle math."""
    # We only need center/tip for processing, so use tiny placeholders for dial and image.
    image_path: Path = Path("dummy.jpg")
    dial: EllipseLabel = EllipseLabel(
        cx=0.0,
        cy=0.0,
        rx=1.0,
        ry=1.0,
        rotation=0.0,
        label="temp_dial",
    )
    center: PointLabel = PointLabel(x=center_x, y=center_y, label="temp_center")
    tip: PointLabel = PointLabel(x=tip_x, y=tip_y, label="temp_tip")
    return Sample(image_path=image_path, dial=dial, center=center, tip=tip)


def make_spec(
    *,
    min_angle_rad: float = 0.0,
    sweep_rad: float = math.pi,
    min_value: float = 0.0,
    max_value: float = 100.0,
) -> GaugeSpec:
    """Construct a test GaugeSpec to keep tests focused on math behavior."""
    # Explicit parameters make the expected fractions and values easy to reason about.
    return GaugeSpec(
        gauge_id="test_gauge",
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        min_value=min_value,
        max_value=max_value,
    )


def test_needle_angle_clockwise_rad_cardinal_directions() -> None:
    """Angles should match image-coordinate clockwise math on cardinal directions."""
    # Right of center is 0 radians in image coordinates.
    right: Sample = build_sample(1.0, 0.0)
    assert needle_angle_clockwise_rad(right) == pytest.approx(0.0)

    # Down is +90 degrees (pi/2) in image coordinates.
    down: Sample = build_sample(0.0, 1.0)
    assert needle_angle_clockwise_rad(down) == pytest.approx(math.pi / 2.0)

    # Left is 180 degrees (pi).
    left: Sample = build_sample(-1.0, 0.0)
    assert needle_angle_clockwise_rad(left) == pytest.approx(math.pi)

    # Up is -90 degrees (-pi/2) with atan2 in image coordinates.
    up: Sample = build_sample(0.0, -1.0)
    assert needle_angle_clockwise_rad(up) == pytest.approx(-math.pi / 2.0)


def test_needle_fraction_within_sweep_strict() -> None:
    """Fraction should scale within the sweep when strict=True."""
    spec: GaugeSpec = make_spec(sweep_rad=math.pi)

    # Zero angle should map to the minimum fraction.
    at_zero: Sample = build_sample(1.0, 0.0)
    assert needle_fraction(at_zero, spec, strict=True) == pytest.approx(0.0)

    # Half sweep should map to 0.5.
    at_half: Sample = build_sample(0.0, 1.0)
    assert needle_fraction(at_half, spec, strict=True) == pytest.approx(0.5)

    # Full sweep should map to 1.0.
    at_full: Sample = build_sample(-1.0, 0.0)
    assert needle_fraction(at_full, spec, strict=True) == pytest.approx(1.0)


def test_needle_fraction_outside_sweep_strict_raises() -> None:
    """Strict mode should reject labels outside the defined sweep."""
    spec: GaugeSpec = make_spec(sweep_rad=math.pi)

    # Up gives -pi/2, which wraps to 3pi/2 and exceeds a pi sweep.
    outside: Sample = build_sample(0.0, -1.0)
    with pytest.raises(ValueError):
        needle_fraction(outside, spec, strict=True)


def test_needle_fraction_outside_sweep_non_strict_clamps() -> None:
    """Non-strict mode should clamp any out-of-range fraction to 1.0."""
    spec: GaugeSpec = make_spec(sweep_rad=math.pi)

    # Use the same outside point and ensure we clamp instead of raising.
    outside: Sample = build_sample(0.0, -1.0)
    assert needle_fraction(outside, spec, strict=False) == pytest.approx(1.0)


def test_needle_value_scales_into_range() -> None:
    """Needle value should scale linearly from min_value to max_value."""
    spec: GaugeSpec = make_spec(sweep_rad=math.pi, min_value=-30.0, max_value=50.0)

    # Half sweep should land at the midpoint of the value range.
    at_half: Sample = build_sample(0.0, 1.0)
    assert needle_value(at_half, spec, strict=True) == pytest.approx(10.0)


def test_load_gauge_specs_from_toml(tmp_path: Path) -> None:
    """TOML parsing should populate GaugeSpec fields with radians and values."""
    # Create a small, deterministic TOML file so the test is isolated.
    toml_text: str = "\n".join(
        [
            "[gauge_a]",
            "min_deg = 10.0",
            "sweep_deg = 180.0",
            "min_value = 0.0",
            "max_value = 100.0",
            "",
            "[gauge_b]",
            "min_deg = 45.0",
            "sweep_deg = 270.0",
            "min_value = -20.0",
            "max_value = 40.0",
        ]
    )
    toml_path: Path = tmp_path / "specs.toml"
    toml_path.write_text(toml_text, encoding="utf-8")

    # Load and verify the first gauge in radians and with proper IDs.
    specs: dict[str, GaugeSpec] = load_gauge_specs(toml_path)
    assert set(specs.keys()) == {"gauge_a", "gauge_b"}
    assert specs["gauge_a"].gauge_id == "gauge_a"
    assert specs["gauge_a"].min_angle_rad == pytest.approx(math.radians(10.0))
    assert specs["gauge_a"].sweep_rad == pytest.approx(math.radians(180.0))
    assert specs["gauge_a"].min_value == pytest.approx(0.0)
    assert specs["gauge_a"].max_value == pytest.approx(100.0)
