"""Unit tests for label utilities."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from embedded_gauge_reading_tinyml.dataset import EllipseLabel, PointLabel, Sample
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec
from embedded_gauge_reading_tinyml.labels import (
    LabelSummary,
    ValueSample,
    build_value_samples,
    summarize_label_sweep,
)


def build_sample(
    tip_x: float,
    tip_y: float,
    *,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> Sample:
    """Create a minimal Sample with center/tip labels for angle math."""
    # Use a placeholder image path to keep tests self-contained.
    image_path: Path = Path("dummy.jpg")
    # Build a dummy dial because Sample requires it even if labels.py doesn't use it.
    dial: EllipseLabel = EllipseLabel(
        cx=0.0,
        cy=0.0,
        rx=1.0,
        ry=1.0,
        rotation=0.0,
        label="temp_dial",
    )
    # Center and tip are what drive the needle angle.
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
    """Construct a GaugeSpec to keep tests focused on math behavior."""
    # Explicit parameters make the expected fractions and values easy to reason about.
    return GaugeSpec(
        gauge_id="test_gauge",
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        min_value=min_value,
        max_value=max_value,
    )


def test_build_value_samples_strict_in_sweep() -> None:
    """Strict mode should return calibrated values for in-sweep samples."""
    spec: GaugeSpec = make_spec(min_value=0.0, max_value=100.0)
    # Use two in-sweep samples to avoid any strict errors.
    samples: list[Sample] = [
        build_sample(1.0, 0.0),  # 0 radians -> fraction 0.0 -> value 0.0
        build_sample(0.0, 1.0),  # pi/2 -> fraction 0.5 -> value 50.0
    ]

    value_samples: list[ValueSample] = build_value_samples(samples, spec, strict=True)
    values: list[float] = [vs.value for vs in value_samples]

    # Validate the calibrated values using approximate float checks.
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(50.0)


def test_build_value_samples_strict_raises_on_out_of_sweep() -> None:
    """Strict mode should raise when a sample is outside the sweep."""
    spec: GaugeSpec = make_spec()
    # This sample wraps to 3pi/2, which is outside a pi sweep.
    samples: list[Sample] = [build_sample(0.0, -1.0)]

    with pytest.raises(ValueError):
        build_value_samples(samples, spec, strict=True)


def test_build_value_samples_non_strict_clamps() -> None:
    """Non-strict mode should clamp values at the max when out of sweep."""
    spec: GaugeSpec = make_spec(min_value=-30.0, max_value=50.0)
    # Use an out-of-sweep sample to force clamping.
    samples: list[Sample] = [build_sample(0.0, -1.0)]

    value_samples: list[ValueSample] = build_value_samples(samples, spec, strict=False)
    # Clamped fraction is 1.0, so the value should equal max_value.
    assert value_samples[0].value == pytest.approx(50.0)


def test_summarize_label_sweep_counts_and_range() -> None:
    """Summary should report correct counts and fraction range."""
    spec: GaugeSpec = make_spec()
    # Two in-sweep samples and one out-of-sweep sample.
    samples: list[Sample] = [
        build_sample(1.0, 0.0),  # fraction 0.0
        build_sample(0.0, 1.0),  # fraction 0.5
        build_sample(0.0, -1.0),  # out of sweep
    ]

    summary: LabelSummary = summarize_label_sweep(samples, spec)

    # Verify counts and range match the expected fractions.
    assert summary.total_samples == 3
    assert summary.in_sweep == 2
    assert summary.out_of_sweep == 1
    assert summary.min_fraction == pytest.approx(0.0)
    assert summary.max_fraction == pytest.approx(0.5)


def test_summarize_label_sweep_empty_samples() -> None:
    """Empty input should return zeroed summary values."""
    spec: GaugeSpec = make_spec()
    # Use an empty list to confirm default min/max handling.
    samples: list[Sample] = []

    summary: LabelSummary = summarize_label_sweep(samples, spec)

    # All outputs should be zero because there are no samples.
    assert summary.total_samples == 0
    assert summary.in_sweep == 0
    assert summary.out_of_sweep == 0
    assert summary.min_fraction == pytest.approx(0.0)
    assert summary.max_fraction == pytest.approx(0.0)
