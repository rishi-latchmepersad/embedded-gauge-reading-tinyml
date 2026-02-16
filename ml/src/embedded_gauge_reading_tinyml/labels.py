"""
Loads and validates labels/annotations for gauge reading samples.
"""

from dataclasses import dataclass
from pathlib import Path

from embedded_gauge_reading_tinyml.gauge.processing import (
    GaugeSpec,
    needle_fraction,
    needle_value,
)
from embedded_gauge_reading_tinyml.dataset import Sample


@dataclass(frozen=True)
class ValueSample:
    """Image path paired with a calibrated gauge value."""

    image_path: Path
    value: float


def build_value_samples(
    samples: list[Sample],
    spec: GaugeSpec,
    *,
    strict: bool = True,
) -> list[ValueSample]:
    """Convert labeled Samples into ValueSamples for training."""
    value_samples: list[ValueSample] = []
    for sample in samples:
        # Use the same value logic as training so labels stay consistent.
        value: float = needle_value(sample, spec, strict=strict)
        value_samples.append(ValueSample(image_path=sample.image_path, value=value))
    return value_samples


@dataclass(frozen=True)
class LabelSummary:
    """Summary stats for sweep coverage and label validity."""

    total_samples: int
    in_sweep: int
    out_of_sweep: int
    min_fraction: float
    max_fraction: float


def summarize_label_sweep(
    samples: list[Sample],
    spec: GaugeSpec,
) -> LabelSummary:
    """Compute sweep coverage stats to validate annotations."""
    fractions: list[float] = []
    out_of_sweep: int = 0
    for sample in samples:
        try:
            # Strict=True so we detect invalid labels rather than clamping them.
            fraction: float = needle_fraction(sample, spec, strict=True)
        except ValueError:
            out_of_sweep += 1
            continue
        fractions.append(fraction)

    # Guard against empty lists to avoid errors on min/max.
    if fractions:
        min_fraction: float = min(fractions)
        max_fraction: float = max(fractions)
    else:
        min_fraction = 0.0
        max_fraction = 0.0

    return LabelSummary(
        total_samples=len(samples),
        in_sweep=len(fractions),
        out_of_sweep=out_of_sweep,
        min_fraction=min_fraction,
        max_fraction=max_fraction,
    )
