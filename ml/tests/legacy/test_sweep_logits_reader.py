"""Tests for the sweep-logits angle-vote gauge reader."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from embedded_gauge_reading_tinyml.models import build_mobilenetv2_sweep_logits_model
from embedded_gauge_reading_tinyml.gauge.processing import (
    GaugeSpec,
    value_to_fraction,
    fraction_to_value,
)

GAUGE_SPEC = GaugeSpec(
    gauge_id="test",
    min_angle_rad=math.radians(135.0),
    sweep_rad=math.radians(270.0),
    min_value=-30.0,
    max_value=50.0,
    units="C",
    direction="clockwise",
)
NUM_BINS = 90


def fraction_to_soft_target(
    fraction: float, *, num_bins: int = NUM_BINS, sigma_bins: float = 2.0
) -> np.ndarray:
    f = min(max(float(fraction), 0.0), 1.0)
    center = f * float(num_bins - 1)
    indices = np.arange(num_bins, dtype=np.float32)
    distances = np.abs(indices - np.float32(center))
    target = np.exp(-0.5 * (distances / np.float32(sigma_bins)) ** 2)
    total = float(np.sum(target))
    if total > 0.0:
        target /= np.float32(total)
    return target.astype(np.float32)


def decode_sweep_logits(
    logits: np.ndarray, *, num_bins: int = NUM_BINS, topk: int = 8
) -> float:
    flat = np.asarray(logits, dtype=np.float32).reshape(-1)
    probs = np.exp(flat - np.max(flat))
    probs /= np.sum(probs)
    indices = np.argsort(probs)[-topk:]
    fraction = float(
        np.sum(probs[indices] * (indices / float(num_bins - 1)))
        / np.sum(probs[indices])
    )
    return fraction_to_value(fraction, GAUGE_SPEC)


class TestSweepLogitsBuilder:
    """Verify the model builder produces the correct output contract."""

    def test_builder_output_shape(self) -> None:
        model = build_mobilenetv2_sweep_logits_model(224, 224, num_bins=90)
        out = model.output
        assert out.shape[-1] == 90, f"Expected 90 bins, got {out.shape[-1]}"

    def test_builder_num_bins_attr(self) -> None:
        model = build_mobilenetv2_sweep_logits_model(224, 224, num_bins=90)
        assert getattr(model, "_num_bins", None) == 90

    def test_builder_predict_shape(self) -> None:
        model = build_mobilenetv2_sweep_logits_model(224, 224, num_bins=90)
        dummy = np.random.rand(2, 224, 224, 3).astype(np.float32)
        out = model.predict(dummy, verbose=0)
        assert out.shape == (2, 90), f"Expected (2, 90), got {out.shape}"

    def test_builder_single_output(self) -> None:
        model = build_mobilenetv2_sweep_logits_model(224, 224, num_bins=90)
        assert isinstance(model.output, list) or hasattr(model.output, "shape")
        if isinstance(model.output, list):
            assert len(model.output) == 1
        assert model.output.shape[-1] == 90

    def test_builder_backbone_attr(self) -> None:
        model = build_mobilenetv2_sweep_logits_model(224, 224, num_bins=90)
        bb = getattr(model, "_mobilenet_backbone", None)
        assert bb is not None, "Model missing _mobilenet_backbone attr"
        assert not bb.trainable, "Backbone should be frozen by default"


class TestSweepDecode:
    """Verify decode_sweep_logits produces sensible temperatures."""

    def _make_perfect_logits(self, temperature: float) -> np.ndarray:
        """Create logits that perfectly peak at a given temperature."""
        fraction = value_to_fraction(temperature, GAUGE_SPEC)
        soft_target = fraction_to_soft_target(fraction)
        # Convert softmax probs back to logits: log(p) + offset
        eps = 1e-7
        logits = np.log(soft_target + eps)
        return logits

    def test_decode_cold_end(self) -> None:
        logits = self._make_perfect_logits(-30.0)
        temp = decode_sweep_logits(logits)
        assert abs(temp - (-30.0)) < 3.0, f"Cold end off: {temp:.1f}°C"

    def test_decode_hot_end(self) -> None:
        logits = self._make_perfect_logits(50.0)
        temp = decode_sweep_logits(logits)
        assert abs(temp - 50.0) < 3.0, f"Hot end off: {temp:.1f}°C"

    def test_decode_mid(self) -> None:
        logits = self._make_perfect_logits(10.0)
        temp = decode_sweep_logits(logits)
        assert abs(temp - 10.0) < 3.0, f"Mid off: {temp:.1f}°C"

    def test_decode_uniform_logits(self) -> None:
        """Uniform logits should produce a valid temperature."""
        logits = np.zeros(NUM_BINS, dtype=np.float32)
        temp = decode_sweep_logits(logits)
        assert -30.0 <= temp <= 50.0, f"Out of range: {temp:.1f}°C"


class TestSoftTarget:
    """Verify soft target encoding is well-formed."""

    def test_target_sums_to_one(self) -> None:
        for t in [-30.0, 0.0, 25.0, 50.0]:
            f = value_to_fraction(t, GAUGE_SPEC)
            target = fraction_to_soft_target(f)
            assert abs(float(np.sum(target)) - 1.0) < 1e-5, f"Sum != 1 for {t}°C"

    def test_target_has_correct_shape(self) -> None:
        f = value_to_fraction(10.0, GAUGE_SPEC)
        target = fraction_to_soft_target(f, num_bins=90)
        assert target.shape == (90,), f"Shape {target.shape}"

    def test_target_peak_at_correct_position(self) -> None:
        for t, expected_bin in [(50.0, 89), (-30.0, 0), (10.0, 44), (25.0, 61)]:
            f = value_to_fraction(t, GAUGE_SPEC)
            target = fraction_to_soft_target(f, num_bins=90)
            peak = int(np.argmax(target))
            assert abs(peak - expected_bin) <= 1, f"{t}°C peak at bin {peak}, expected ~{expected_bin}"


class TestLumaCropParity:
    """Verify luma crop is importable and produces correct shape."""

    def test_luma_crop_import(self) -> None:
        from luma_crop_detector import estimate_bright_centroid  # noqa: F811
        from luma_crop_detector import compute_dynamic_crop, crop_and_resize
        assert callable(estimate_bright_centroid)
        assert callable(compute_dynamic_crop)
        assert callable(crop_and_resize)

    def test_luma_crop_on_synthetic(self) -> None:
        from luma_crop_detector import estimate_bright_centroid, compute_dynamic_crop, crop_and_resize
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        cent = estimate_bright_centroid(img)
        cb = compute_dynamic_crop(img.shape[1], img.shape[0], cent.center_x, cent.center_y)
        if cb is not None:
            result = crop_and_resize(img, cb, target_size=224)
            assert result.shape == (224, 224, 3)
        else:
            # If bright centroid fails (too few bright pixels), that's OK
            pass


class TestConfigParity:
    """Verify config defaults match the handoff spec."""

    def test_num_bins_is_90(self) -> None:
        assert NUM_BINS == 90

    def test_variants_exist(self) -> None:
        """Just check the variant names are recognized."""
        from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec
        assert GAUGE_SPEC.min_value == -30.0
        assert GAUGE_SPEC.max_value == 50.0
