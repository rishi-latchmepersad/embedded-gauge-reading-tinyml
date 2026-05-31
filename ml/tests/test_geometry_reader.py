"""Tests for the gauge-agnostic geometry reader (geometry_reader_v1)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from embedded_gauge_reading_tinyml.models import build_mobilenetv2_geometry_reader
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, angle_rad_to_fraction, fraction_to_value
from embedded_gauge_reading_tinyml.heatmap_utils import (
    make_gaussian_heatmap,
    softargmax_2d,
    generate_center_tip_heatmaps,
    HeatmapConfig,
)

INPUT_SIZE = 224
HEATMAP_SIZE = 28
SIGMA_PIXELS = 2.5


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

class TestBuilder:
    def test_output_shapes(self):
        model = build_mobilenetv2_geometry_reader(
            INPUT_SIZE, INPUT_SIZE, heatmap_size=HEATMAP_SIZE, pretrained=False,
        )
        dummy = np.random.rand(2, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32)
        out = model.predict(dummy, verbose=0)
        assert "center_heatmap" in out
        assert "tip_heatmap" in out
        assert out["center_heatmap"].shape == (2, HEATMAP_SIZE, HEATMAP_SIZE, 1)
        assert out["tip_heatmap"].shape == (2, HEATMAP_SIZE, HEATMAP_SIZE, 1)

    def test_output_is_sigmoid(self):
        model = build_mobilenetv2_geometry_reader(
            INPUT_SIZE, INPUT_SIZE, heatmap_size=HEATMAP_SIZE, pretrained=False,
        )
        dummy = np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32)
        out = model.predict(dummy, verbose=0)
        assert out["center_heatmap"].min() >= 0.0
        assert out["center_heatmap"].max() <= 1.0
        assert out["tip_heatmap"].min() >= 0.0
        assert out["tip_heatmap"].max() <= 1.0

    def test_no_gauge_params(self):
        model = build_mobilenetv2_geometry_reader(
            INPUT_SIZE, INPUT_SIZE, heatmap_size=HEATMAP_SIZE,
        )
        layers_with_gauge = [
            l.name for l in model.layers if "gauge" in l.name.lower()
        ]
        assert len(layers_with_gauge) == 0

    def test_backbone_stored(self):
        model = build_mobilenetv2_geometry_reader(
            INPUT_SIZE, INPUT_SIZE, heatmap_size=HEATMAP_SIZE, pretrained=False,
        )
        assert hasattr(model, "_mobilenet_backbone")
        assert model._mobilenet_backbone is not None

    def test_param_count_reasonable(self):
        model = build_mobilenetv2_geometry_reader(
            INPUT_SIZE, INPUT_SIZE, alpha=0.35, heatmap_size=HEATMAP_SIZE, pretrained=False,
        )
        # Heatmap decoder adds ~2.5M params to backbone; model should be <5M
        assert model.count_params() < 5_000_000


# ---------------------------------------------------------------------------
# Heatmap generation and soft-argmax decode
# ---------------------------------------------------------------------------

class TestHeatmapDecode:
    def test_gaussian_peak_at_center(self):
        hm = make_gaussian_heatmap(28, 28, 0.5, 0.5, sigma_pixels=2.5)
        row, col = softargmax_2d(hm)
        assert abs(col / 27 - 0.5) < 0.1
        assert abs(row / 27 - 0.5) < 0.1

    def test_gaussian_peak_at_corner(self):
        hm = make_gaussian_heatmap(28, 28, 0.9, 0.1, sigma_pixels=2.5)
        row, col = softargmax_2d(hm)
        assert abs(col / 27 - 0.9) < 0.1
        assert abs(row / 27 - 0.1) < 0.1

    def test_center_tip_pair_generation(self):
        config = HeatmapConfig(heatmap_height=28, heatmap_width=28, sigma_pixels=2.5)
        c_hm, t_hm = generate_center_tip_heatmaps(
            0.3, 0.4, 0.7, 0.6, config=config,
        )
        assert c_hm.shape == (28, 28)
        assert t_hm.shape == (28, 28)
        cx, cy = softargmax_2d(c_hm)
        tx, ty = softargmax_2d(t_hm)
        assert abs(cx / 27 - 0.3) < 0.2
        assert abs(cy / 27 - 0.4) < 0.2
        assert abs(tx / 27 - 0.7) < 0.2
        assert abs(ty / 27 - 0.6) < 0.2

    def test_decode_angle_from_heatmaps(self):
        config = HeatmapConfig(heatmap_height=28, heatmap_width=28, sigma_pixels=2.5)
        center = (0.5, 0.5)
        tip = (0.8, 0.3)
        c_hm, t_hm = generate_center_tip_heatmaps(
            center[0], center[1], tip[0], tip[1], config=config,
        )
        cy, cx = softargmax_2d(c_hm)
        ty, tx = softargmax_2d(t_hm)
        cx_px = cx / 27 * (INPUT_SIZE - 1)
        cy_px = cy / 27 * (INPUT_SIZE - 1)
        tx_px = tx / 27 * (INPUT_SIZE - 1)
        ty_px = ty / 27 * (INPUT_SIZE - 1)
        dx = tx_px - cx_px
        dy = ty_px - cy_px
        angle_rad = math.atan2(dy, dx)
        assert angle_rad != 0.0


# ---------------------------------------------------------------------------
# GaugeSpec decode chain: heatmaps → angle → fraction → temperature
# ---------------------------------------------------------------------------

class TestGaugeDecode:
    @pytest.fixture
    def spec(self):
        return GaugeSpec(
            gauge_id="test_gauge",
            min_angle_rad=math.radians(135.0),
            sweep_rad=math.radians(270.0),
            min_value=-30.0,
            max_value=50.0,
            units="C",
            direction="clockwise",
        )

    def test_angle_to_temperature(self, spec):
        angle_rad = math.radians(225.0)
        fraction = angle_rad_to_fraction(angle_rad, spec, strict=True)
        temp = fraction_to_value(fraction, spec)
        # At 225: 90 deg into a 270 deg sweep (=33.3%), -30 + 0.333*80 = -3.33C
        assert abs(temp - (-3.33)) < 1.0

    def test_hot_end(self, spec):
        angle_rad = math.radians(45.0)
        fraction = angle_rad_to_fraction(angle_rad % (2 * math.pi), spec, strict=True)
        temp = fraction_to_value(fraction, spec)
        assert abs(temp - 50.0) < 1.0

    def test_cold_end(self, spec):
        angle_rad = math.radians(135.0)
        fraction = angle_rad_to_fraction(angle_rad, spec, strict=True)
        temp = fraction_to_value(fraction, spec)
        assert abs(temp - (-30.0)) < 1.0

    def test_spec_swap_same_geometry_different_temp(self):
        gauge_a = GaugeSpec(
            gauge_id="gauge_a", min_angle_rad=0.0, sweep_rad=math.pi,
            min_value=0.0, max_value=100.0, units="X",
        )
        gauge_b = GaugeSpec(
            gauge_id="gauge_b", min_angle_rad=0.0, sweep_rad=math.pi,
            min_value=0.0, max_value=200.0, units="Y",
        )
        angle_rad = math.radians(90.0)
        frac_a = angle_rad_to_fraction(angle_rad, gauge_a, strict=True)
        frac_b = angle_rad_to_fraction(angle_rad, gauge_b, strict=True)
        temp_a = fraction_to_value(frac_a, gauge_a)
        temp_b = fraction_to_value(frac_b, gauge_b)
        assert abs(temp_a - 50.0) < 0.1
        assert abs(temp_b - 100.0) < 0.1


# ---------------------------------------------------------------------------
# Training helpers (augmentation, soft-argmax)
# ---------------------------------------------------------------------------

@tf.function
def _augment_image(image, center_hm, tip_hm):
    """Photometric augmentation matching the training script (no geometric transforms)."""
    rgb = image[..., :3]
    rgb = tf.image.random_brightness(rgb, 0.15)
    rgb = tf.image.random_contrast(rgb, 0.7, 1.3)
    rgb = tf.image.random_saturation(rgb, 0.7, 1.3)
    rgb = tf.image.random_hue(rgb, 0.05)
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)
    noise = tf.random.normal(tf.shape(image), 0.0, 0.02)
    rgb = tf.clip_by_value(rgb + noise, 0.0, 1.0)
    return rgb, center_hm, tip_hm


class TestTrainingHelpers:
    def test_augment_preserves_heatmap(self):
        img = np.random.rand(1, 224, 224, 3).astype(np.float32)
        hm = np.random.rand(1, 28, 28, 1).astype(np.float32)
        aug_img, aug_c, aug_t = _augment_image(tf.constant(img), tf.constant(hm), tf.constant(hm))
        assert aug_img.shape == (1, 224, 224, 3)
        assert aug_c.shape == (1, 28, 28, 1)
        assert aug_t.shape == (1, 28, 28, 1)


# ---------------------------------------------------------------------------
# INT8 export smoke test
# ---------------------------------------------------------------------------

class TestExport:
    def test_keras_save_load_roundtrip(self, tmp_path):
        model = build_mobilenetv2_geometry_reader(
            INPUT_SIZE, INPUT_SIZE, heatmap_size=HEATMAP_SIZE, pretrained=False,
        )
        path = tmp_path / "model.keras"
        model.save(path)
        loaded = tf.keras.models.load_model(path)
        dummy = np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32)
        out = loaded.predict(dummy, verbose=0)
        assert "center_heatmap" in out
        assert "tip_heatmap" in out

    def test_tflite_conversion(self, tmp_path):
        model = build_mobilenetv2_geometry_reader(
            INPUT_SIZE, INPUT_SIZE, heatmap_size=HEATMAP_SIZE, pretrained=False,
        )
        dummy = np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32)

        def rep():
            for _ in range(5):
                yield [dummy * 255.0]

        cvt = tf.lite.TFLiteConverter.from_keras_model(model)
        cvt.optimizations = [tf.lite.Optimize.DEFAULT]
        cvt.representative_dataset = rep
        cvt.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        cvt.inference_input_type = tf.uint8
        cvt.inference_output_type = tf.float32
        tflite_bytes = cvt.convert()
        assert len(tflite_bytes) > 0

        path = tmp_path / "model_int8.tflite"
        path.write_bytes(tflite_bytes)
        assert path.stat().st_size < 5_000_000
