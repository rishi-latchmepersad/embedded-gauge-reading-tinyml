"""Tests for the classical-inspired polar evidence CNN (polar_evidence_v1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import tensorflow as tf

from embedded_gauge_reading_tinyml.models import (
    build_mobilenetv2_polar_evidence_model,
    PolarEvidenceLayer,
)
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, angle_rad_to_fraction, fraction_to_value


INPUT_SIZE = 224
NUM_ANGLES = 180


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

class TestBuilder:
    def test_output_shapes(self):
        m = build_mobilenetv2_polar_evidence_model(
            INPUT_SIZE, INPUT_SIZE, num_angles=NUM_ANGLES, pretrained=False,
        )
        out = m.predict(np.random.rand(2, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32), verbose=0)
        assert out["center"].shape == (2, 2)
        assert out["polar_evidence"].shape == (2, NUM_ANGLES)
        assert out["confidence"].shape == (2, 1)

    def test_center_sigmoid_range(self):
        m = build_mobilenetv2_polar_evidence_model(
            INPUT_SIZE, INPUT_SIZE, num_angles=NUM_ANGLES, pretrained=False,
        )
        out = m.predict(np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32), verbose=0)
        assert out["center"].min() >= 0.0
        assert out["center"].max() <= 1.0

    def test_confidence_sigmoid_range(self):
        m = build_mobilenetv2_polar_evidence_model(
            INPUT_SIZE, INPUT_SIZE, num_angles=NUM_ANGLES, pretrained=False,
        )
        out = m.predict(np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32), verbose=0)
        assert out["confidence"].min() >= 0.0
        assert out["confidence"].max() <= 1.0

    def test_no_gauge_params_in_model(self):
        m = build_mobilenetv2_polar_evidence_model(
            INPUT_SIZE, INPUT_SIZE, num_angles=NUM_ANGLES,
        )
        gauge_layers = [l.name for l in m.layers if "gauge" in l.name.lower()]
        assert len(gauge_layers) == 0

    def test_param_count(self):
        m = build_mobilenetv2_polar_evidence_model(
            INPUT_SIZE, INPUT_SIZE, num_angles=NUM_ANGLES, pretrained=False,
        )
        assert 1_000_000 < m.count_params() < 5_000_000


# ---------------------------------------------------------------------------
# PolarEvidenceLayer
# ---------------------------------------------------------------------------

class TestPolarEvidenceLayer:
    def test_output_shape(self):
        layer = PolarEvidenceLayer(num_angles=180)
        features = tf.constant(np.random.rand(2, 28, 28, 32).astype(np.float32))
        center = tf.constant([[0.5, 0.4], [0.5, 0.4]], dtype=tf.float32)
        out = layer(features, center)
        assert out.shape == (2, 180)

    def test_center_differentiable(self):
        layer = PolarEvidenceLayer(num_angles=180)
        features = tf.constant(np.random.rand(1, 28, 28, 32).astype(np.float32))
        center = tf.Variable([[0.5, 0.4]], dtype=tf.float32)
        with tf.GradientTape() as tape:
            out = layer(features, center)
            loss = tf.reduce_sum(out)
        grad = tape.gradient(loss, center)
        assert grad is not None
        assert not tf.reduce_all(grad == 0.0)


# ---------------------------------------------------------------------------
# Decode chain: polar evidence → angle → temperature
# ---------------------------------------------------------------------------

class TestPolarDecode:
    @pytest.fixture
    def spec(self):
        return GaugeSpec(
            gauge_id="test", min_angle_rad=math.radians(135.0),
            sweep_rad=math.radians(270.0),
            min_value=-30.0, max_value=50.0, units="C",
        )

    def test_decode_known_angle(self, spec):
        bin_centers = np.linspace(-np.pi, np.pi, NUM_ANGLES + 1)[:NUM_ANGLES]
        true_angle_rad = math.radians(225.0)
        diff = true_angle_rad - bin_centers
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        dist = np.exp(-(diff ** 2) / (2.0 * 3.0 ** 2))
        probs = dist / np.sum(dist)
        sin_sum = np.sum(probs * np.sin(bin_centers))
        cos_sum = np.sum(probs * np.cos(bin_centers))
        angle_rad = math.atan2(sin_sum, cos_sum)
        fraction = angle_rad_to_fraction(angle_rad, spec, strict=True)
        temp = fraction_to_value(fraction, spec)
        assert abs(temp - (-3.33)) < 2.0

    def test_spec_swap_same_geometry(self):
        gauge_a = GaugeSpec(
            gauge_id="a", min_angle_rad=0.0, sweep_rad=math.pi,
            min_value=0.0, max_value=100.0, units="X",
        )
        gauge_b = GaugeSpec(
            gauge_id="b", min_angle_rad=0.0, sweep_rad=math.pi,
            min_value=0.0, max_value=200.0, units="Y",
        )
        angle_rad = math.radians(90.0)
        frac = angle_rad_to_fraction(angle_rad, gauge_a, strict=False)
        assert abs(fraction_to_value(frac, gauge_a) - 50.0) < 0.1
        assert abs(fraction_to_value(frac, gauge_b) - 100.0) < 0.1


# ---------------------------------------------------------------------------
# Angle distribution helper
# ---------------------------------------------------------------------------

def _make_angle_distribution(angle_rad: float, num_bins: int, sigma_bins: float) -> np.ndarray:
    """Create a Gaussian-smoothed target distribution over angle bins."""
    bin_centers = np.linspace(-np.pi, np.pi, num_bins + 1, dtype=np.float32)[:num_bins]
    diff = angle_rad - bin_centers
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    dist = np.exp(-(diff ** 2) / (2.0 * sigma_bins ** 2))
    dist = dist / np.sum(dist)
    return dist


class TestAngleDistribution:
    def test_peak_at_true_angle(self):
        dist = _make_angle_distribution(1.0, 180, 3.0)
        assert abs(dist.sum() - 1.0) < 0.01
        assert dist.shape == (180,)

    def test_distribution_is_smooth(self):
        dist = _make_angle_distribution(0.0, 180, 3.0)
        assert np.all(dist >= 0.0)


# ---------------------------------------------------------------------------
# Export smoke test
# ---------------------------------------------------------------------------

class TestExport:
    def test_keras_save_load_roundtrip(self, tmp_path):
        m = build_mobilenetv2_polar_evidence_model(
            INPUT_SIZE, INPUT_SIZE, num_angles=NUM_ANGLES, pretrained=False,
        )
        path = tmp_path / "model.keras"
        m.save(path)
        loaded = tf.keras.models.load_model(path)
        out = loaded.predict(
            np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32), verbose=0,
        )
        assert "center" in out
        assert "polar_evidence" in out
        assert "confidence" in out

    def test_tflite_conversion(self, tmp_path):
        m = build_mobilenetv2_polar_evidence_model(
            INPUT_SIZE, INPUT_SIZE, num_angles=NUM_ANGLES, pretrained=False,
        )
        def rep():
            for _ in range(5):
                yield [np.random.rand(1, 224, 224, 3).astype(np.float32) * 255.0]
        cvt = tf.lite.TFLiteConverter.from_keras_model(m)
        cvt.optimizations = [tf.lite.Optimize.DEFAULT]
        cvt.representative_dataset = rep
        cvt.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        cvt.inference_input_type = tf.uint8
        cvt.inference_output_type = tf.float32
        tflite_bytes = cvt.convert()
        assert len(tflite_bytes) > 0
        sz_kb = len(tflite_bytes) / 1024
        assert sz_kb < 5000
