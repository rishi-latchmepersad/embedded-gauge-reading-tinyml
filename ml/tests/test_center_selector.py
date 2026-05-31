"""Tests for the hybrid localizer center selector CNN."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from embedded_gauge_reading_tinyml.models import build_mobilenetv2_center_selector
from embedded_gauge_reading_tinyml.hybrid_localizer import (
    compute_all_hypotheses,
    compute_fast_hypotheses,
    compute_board_prior_center,
    needle_angle_from_polar_vote,
    estimate_dial_radius,
    rgb_to_luma,
    polar_spoke_vote,
    smooth_and_find_peak,
    is_angle_in_sweep,
    estimate_bright_centroid_on_crop,
    compute_crop_center,
    compute_image_center,
    estimate_rim_center,
    BOARD_PRIOR_CENTER_X_RATIO,
    BOARD_PRIOR_CENTER_Y_RATIO,
)
from embedded_gauge_reading_tinyml.gauge.processing import (
    GaugeSpec,
    angle_rad_to_fraction,
    fraction_to_value,
)
from train_center_selector_v2 import (
    make_soft_targets,
    compute_confidence,
    decode_center_with_confidence,
    CONFIDENCE_LOGIT_MARGIN,
    CONFIDENCE_MAX_ENTROPY,
    CONFIDENCE_MAX_OFFSET,
    OFFSET_SCALE,
)

INPUT_SIZE = 224
NUM_HYPOTHESES = 5


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

class TestCenterSelectorBuilder:
    def test_output_shapes(self):
        m = build_mobilenetv2_center_selector(
            INPUT_SIZE, INPUT_SIZE, num_hypotheses=NUM_HYPOTHESES, pretrained=False,
        )
        dummy = np.random.rand(2, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32)
        out = m.predict(dummy, verbose=0)
        assert out["center_logits"].shape == (2, NUM_HYPOTHESES)
        assert out["center_offset"].shape == (2, 2)

    def test_offset_tanh_range(self):
        m = build_mobilenetv2_center_selector(
            INPUT_SIZE, INPUT_SIZE, num_hypotheses=NUM_HYPOTHESES, pretrained=False,
        )
        for v in [0.0, 0.5, 1.0]:
            dummy = np.full((1, INPUT_SIZE, INPUT_SIZE, 3), v * 255, dtype=np.float32)
            out = m.predict(dummy, verbose=0)
            offset = out["center_offset"][0]
            assert offset.min() >= -1.0
            assert offset.max() <= 1.0

    def test_logits_are_unbounded(self):
        m = build_mobilenetv2_center_selector(
            INPUT_SIZE, INPUT_SIZE, num_hypotheses=NUM_HYPOTHESES, pretrained=False,
        )
        dummy = np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32)
        out = m.predict(dummy, verbose=0)
        logits = out["center_logits"][0]
        # Linear activation = any range
        assert logits.shape == (NUM_HYPOTHESES,)
        assert np.all(np.isfinite(logits))

    def test_param_count(self):
        m = build_mobilenetv2_center_selector(
            INPUT_SIZE, INPUT_SIZE, num_hypotheses=NUM_HYPOTHESES, pretrained=False, alpha=0.35,
        )
        # A tiny MobileNetV2 alpha=0.35 with a small head: < 1M params when random init
        # Pretrained would be bigger due to BN etc
        assert 100_000 < m.count_params() < 1_500_000

    def test_backbone_attr(self):
        m = build_mobilenetv2_center_selector(
            INPUT_SIZE, INPUT_SIZE, num_hypotheses=NUM_HYPOTHESES, pretrained=False,
        )
        bb = getattr(m, "_mobilenet_backbone", None)
        assert bb is not None
        assert not bb.trainable

    def test_trainable_backbone(self):
        m = build_mobilenetv2_center_selector(
            INPUT_SIZE, INPUT_SIZE, num_hypotheses=NUM_HYPOTHESES,
            pretrained=False, backbone_trainable=True,
        )
        bb = getattr(m, "_mobilenet_backbone", None)
        assert bb is not None
        assert bb.trainable


# ---------------------------------------------------------------------------
# Hypothesis computation
# ---------------------------------------------------------------------------

class TestHypotheses:
    def test_five_hypotheses_returned(self):
        img = np.random.randint(0, 256, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        hyps = compute_all_hypotheses(img)
        assert hyps.shape == (5, 2)
        assert np.all(np.isfinite(hyps))

    def test_four_fast_hypotheses_returned(self):
        img = np.random.randint(0, 256, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        hyps = compute_fast_hypotheses(img)
        assert hyps.shape == (4, 2)
        assert np.all(np.isfinite(hyps))

    def test_hypotheses_in_bounds(self):
        img = np.random.randint(0, 256, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        hyps = compute_all_hypotheses(img)
        h, w = img.shape[:2]
        assert np.all(hyps[:, 0] >= 0)
        assert np.all(hyps[:, 0] < w)
        assert np.all(hyps[:, 1] >= 0)
        assert np.all(hyps[:, 1] < h)

    def test_crop_center_is_fixed(self):
        w, h = INPUT_SIZE, INPUT_SIZE
        cx, cy = compute_crop_center(w, h)
        assert abs(cx - 112.0) < 0.1
        assert abs(cy - 100.0) < 0.1  # 0.446 * 224 ≈ 99.9

    def test_board_prior_center_matches_firmware(self):
        w, h = INPUT_SIZE, INPUT_SIZE
        cx, cy = compute_board_prior_center(w, h)
        assert abs(cx - BOARD_PRIOR_CENTER_X_RATIO * w) < 0.01
        assert abs(cy - BOARD_PRIOR_CENTER_Y_RATIO * h) < 0.01
        # Board prior is slightly left of crop center (0.49 vs 0.50)
        crop_cx, _ = compute_crop_center(w, h)
        assert cx < crop_cx

    def test_image_center_is_mid(self):
        w, h = INPUT_SIZE, INPUT_SIZE
        cx, cy = compute_image_center(w, h)
        assert cx == w / 2.0
        assert cy == h / 2.0

    def test_rim_center_default_fallback(self):
        """A blank image should fall back — rim detected=False."""
        luma = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32) + 128.0
        dial_r = estimate_dial_radius(INPUT_SIZE)
        cx, cy, detected = estimate_rim_center(luma, INPUT_SIZE, INPUT_SIZE, dial_r)
        assert 0.0 <= cx < INPUT_SIZE
        assert 0.0 <= cy < INPUT_SIZE

    def test_bright_centroid_fallback_on_dark(self):
        img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        cx, cy, detected = estimate_bright_centroid_on_crop(img)
        expected_cx, expected_cy = compute_crop_center(INPUT_SIZE, INPUT_SIZE)
        assert not detected
        assert abs(cx - expected_cx) < 2.0
        assert abs(cy - expected_cy) < 2.0

    def test_hypothesis_ordering(self):
        """Verify hypothesis order: bright, crop, board_prior, rim, image."""
        img = np.random.randint(0, 256, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        hyps = compute_all_hypotheses(img)
        # Hypothesis 1 (crop) should be at fixed position
        crop_cx, crop_cy = compute_crop_center(INPUT_SIZE, INPUT_SIZE)
        assert abs(hyps[1, 0] - crop_cx) < 0.01
        assert abs(hyps[1, 1] - crop_cy) < 0.01
        # Hypothesis 4 (image) should be at geometric center
        img_cx, img_cy = compute_image_center(INPUT_SIZE, INPUT_SIZE)
        assert abs(hyps[4, 0] - img_cx) < 0.01
        assert abs(hyps[4, 1] - img_cy) < 0.01


# ---------------------------------------------------------------------------
# Polar spoke vote
# ---------------------------------------------------------------------------

class TestPolarVote:
    def test_vote_shape(self):
        img = np.random.randint(0, 256, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        luma = rgb_to_luma(img)
        votes = polar_spoke_vote(luma, 112.0, 100.0, 70.0)
        assert votes.shape == (360,)
        assert np.all(votes >= 0.0)

    def test_peak_on_synthetic_needle(self):
        """Draw a dark needle on a light dial face and verify peak near needle angle."""
        img = np.full((INPUT_SIZE, INPUT_SIZE, 3), 180, dtype=np.uint8)  # light dial face
        cx, cy = 112.0, 100.0
        needle_angle_deg = 200.0
        rad = math.radians(needle_angle_deg)
        for r in range(15, 65):
            x = int(round(cx + r * math.cos(rad)))
            y = int(round(cy + r * math.sin(rad)))
            if 0 <= x < INPUT_SIZE and 0 <= y < INPUT_SIZE:
                img[y, x] = [0, 0, 0]  # dark needle
        # Small bright center hub
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if dx * dx + dy * dy <= 9:
                    yi, xi = int(cy) + dy, int(cx) + dx
                    if 0 <= yi < INPUT_SIZE and 0 <= xi < INPUT_SIZE:
                        img[yi, xi] = [220, 220, 220]

        luma = rgb_to_luma(img)
        votes = polar_spoke_vote(luma, cx, cy, 40.0)
        angle, _, _ = smooth_and_find_peak(votes)
        assert abs(angle - needle_angle_deg) < 15.0

    def test_peak_within_sweep(self):
        """A well-defined needle should produce a peak inside the gauge sweep."""
        img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        cx, cy = 112.0, 100.0
        for angle_deg in [150.0, 200.0, 300.0, 350.0]:
            img_fill = img.copy()
            rad = math.radians(angle_deg)
            for r in range(20, 60):
                x = int(cx + r * math.cos(rad))
                y = int(cy + r * math.sin(rad))
                if 0 <= x < INPUT_SIZE and 0 <= y < INPUT_SIZE:
                    img_fill[y, x] = [0, 0, 0]
            ci, cj = int(cy), int(cx)
            img_fill[ci - 5:ci + 5, cj - 5:cj + 5] = [255, 255, 255]
            luma = rgb_to_luma(img_fill)
            votes = polar_spoke_vote(luma, cx, cy, 40.0)
            angle, _, _ = smooth_and_find_peak(votes)
            assert is_angle_in_sweep(angle, margin=10.0), (
                f"Needle at {angle_deg}° gave peak at {angle:.1f}° outside sweep"
            )


# ---------------------------------------------------------------------------
# End-to-end polar decode
# ---------------------------------------------------------------------------

class TestPolarDecode:
    @pytest.fixture
    def spec(self):
        return GaugeSpec(
            gauge_id="test", min_angle_rad=math.radians(135.0),
            sweep_rad=math.radians(270.0),
            min_value=-30.0, max_value=50.0, units="C",
        )

    def test_needle_angle_from_polar_vote(self, spec):
        """Draw a dark needle on light background at 225° → ~-3.3°C."""
        img = np.full((INPUT_SIZE, INPUT_SIZE, 3), 180, dtype=np.uint8)
        cx, cy = 112.0, 100.0
        needle_angle_deg = 225.0
        rad = math.radians(needle_angle_deg)
        for r in range(15, 70):
            x = int(round(cx + r * math.cos(rad)))
            y = int(round(cy + r * math.sin(rad)))
            if 0 <= x < INPUT_SIZE and 0 <= y < INPUT_SIZE:
                img[y, x] = [0, 0, 0]
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if dx * dx + dy * dy <= 9:
                    yi, xi = int(cy) + dy, int(cx) + dx
                    if 0 <= yi < INPUT_SIZE and 0 <= xi < INPUT_SIZE:
                        img[yi, xi] = [220, 220, 220]

        angle = needle_angle_from_polar_vote(img, cx, cy, 55.0)
        angle_rad = math.radians(angle)
        fraction = angle_rad_to_fraction(angle_rad, spec, strict=False)
        temp = fraction_to_value(fraction, spec)
        # needle at 225° = 0.333 fraction → approx -3.33°C
        assert -10.0 < temp < 10.0


# ---------------------------------------------------------------------------
# Dial radius estimation
# ---------------------------------------------------------------------------

class TestDialRadius:
    def test_reasonable_radius(self):
        r = estimate_dial_radius(INPUT_SIZE)
        assert 50.0 < r < 120.0

    def test_minimum_clamp(self):
        r = estimate_dial_radius(16)
        assert r >= 16.0

    def test_small_crop(self):
        r = estimate_dial_radius(48)
        assert r >= 16.0


# ---------------------------------------------------------------------------
# Angle sweep check
# ---------------------------------------------------------------------------

class TestAngleSweep:
    def test_225_in_sweep(self):
        assert is_angle_in_sweep(225.0)

    def test_0_in_sweep(self):
        """0° wraps through 360 → inside the [135°, 405°) sweep."""
        assert is_angle_in_sweep(0.0)

    def test_135_in_sweep(self):
        """135° is the sweep start, included by margin."""
        assert is_angle_in_sweep(135.0)

    def test_405_equals_45_in_sweep(self):
        assert is_angle_in_sweep(45.0)
        assert is_angle_in_sweep(405.0)


# ---------------------------------------------------------------------------
# Export smoke test
# ---------------------------------------------------------------------------

class TestExport:
    def test_keras_save_load_roundtrip(self, tmp_path):
        m = build_mobilenetv2_center_selector(
            INPUT_SIZE, INPUT_SIZE, num_hypotheses=NUM_HYPOTHESES, pretrained=False,
        )
        path = tmp_path / "model.keras"
        m.save(path)
        loaded = tf.keras.models.load_model(path)
        dummy = np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32)
        out = loaded.predict(dummy, verbose=0)
        assert "center_logits" in out
        assert "center_offset" in out

    def test_tflite_conversion(self, tmp_path):
        m = build_mobilenetv2_center_selector(
            INPUT_SIZE, INPUT_SIZE, num_hypotheses=NUM_HYPOTHESES, pretrained=False,
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


# ---------------------------------------------------------------------------
# Soft target construction
# ---------------------------------------------------------------------------

class TestSoftTargets:
    def test_soft_targets_sum_to_one(self):
        """Soft targets should always sum to 1.0 (softmax-normalized)."""
        hyps = np.array([[0.5, 0.5], [0.3, 0.4], [0.7, 0.6], [0.5, 0.5], [0.4, 0.45]])
        targets = make_soft_targets(hyps, 0.5, 0.5, temperature=0.12)
        assert abs(np.sum(targets) - 1.0) < 1e-6

    def test_closest_hypothesis_gets_highest_target(self):
        """The hypothesis closest to true center should get the highest soft target."""
        hyps = np.array([
            [0.50, 0.50],  # closest to (0.5, 0.5)
            [0.10, 0.10],
            [0.49, 0.45],  # board_prior
            [0.90, 0.90],
            [0.20, 0.80],
        ])
        targets = make_soft_targets(hyps, 0.5, 0.5, temperature=0.12)
        assert np.argmax(targets) == 0

    def test_sharper_temperature_concentrates_mass(self):
        """Lower temperature should concentrate more mass on the closest hypothesis."""
        hyps = np.array([
            [0.50, 0.50],
            [0.45, 0.45],  # slightly further
            [0.49, 0.45],
            [0.90, 0.90],
            [0.10, 0.10],
        ])
        targets_sharp = make_soft_targets(hyps, 0.5, 0.5, temperature=0.02)
        targets_smooth = make_soft_targets(hyps, 0.5, 0.5, temperature=0.12)
        assert targets_sharp[0] > targets_smooth[0]

    def test_equal_distances_give_equal_targets(self):
        """Hypotheses at equal distance should get equal soft targets."""
        hyps = np.array([
            [0.45, 0.50],
            [0.55, 0.50],
            [0.49, 0.45],
            [0.50, 0.45],
            [0.50, 0.55],
        ])
        targets = make_soft_targets(hyps, 0.5, 0.5, temperature=0.12)
        assert np.allclose(targets[[0, 1, 3, 4]], targets[0], atol=1e-6)

    def test_very_far_hypothesis_gets_near_zero(self):
        """A hypothesis very far from true center should get near-zero target."""
        hyps = np.array([
            [0.50, 0.50],
            [0.01, 0.01],
            [0.49, 0.45],
            [0.99, 0.99],
            [0.01, 0.99],
        ])
        targets = make_soft_targets(hyps, 0.5, 0.5, temperature=0.12)
        assert targets[0] > 0.95
        assert targets[1] < 0.02
        assert targets[3] < 0.02
        assert targets[4] < 0.02

    def test_softer_temperature_allows_learning(self):
        """v2 temperature (0.12) should give smoother gradients than v1 (0.05)."""
        hyps = np.array([
            [0.50, 0.50],
            [0.48, 0.48],
            [0.49, 0.45],
            [0.60, 0.60],
            [0.30, 0.30],
        ])
        targets_v1 = make_soft_targets(hyps, 0.5, 0.5, temperature=0.05)
        targets_v2 = make_soft_targets(hyps, 0.5, 0.5, temperature=0.12)
        # v2 should distribute more mass to nearby hypotheses
        assert targets_v2[1] > targets_v1[1]  # close hypothesis gets more mass
        assert targets_v2[0] < targets_v1[0] + 0.1  # closest gets slightly less


# ---------------------------------------------------------------------------
# Center offset decode
# ---------------------------------------------------------------------------

class TestOffsetDecode:
    def test_zero_offset_returns_hypothesis_center(self):
        """Zero offset should return the chosen hypothesis center exactly."""
        logits = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        offset = np.array([0.0, 0.0])
        hyps = np.array([
            [100.0, 90.0], [112.0, 100.0], [109.76, 100.0], [105.0, 95.0], [112.0, 112.0]
        ])

        cx, cy, conf, is_high = decode_center_with_confidence(logits, offset, hyps)
        assert abs(cx - 100.0) < 0.01
        assert abs(cy - 90.0) < 0.01

    def test_positive_offset_shifts_right_down(self):
        """Positive offset should shift center right and down."""
        logits = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        offset = np.array([0.5, 0.3])
        hyps = np.array([
            [100.0, 90.0], [112.0, 100.0], [109.76, 100.0], [105.0, 95.0], [112.0, 112.0]
        ])

        cx, cy, _, _ = decode_center_with_confidence(logits, offset, hyps, offset_scale=10.0)
        assert cx > 100.0
        assert cy > 90.0
        assert abs(cx - (100.0 + 0.5 * 10.0)) < 0.01
        assert abs(cy - (90.0 + 0.3 * 10.0)) < 0.01

    def test_negative_offset_shifts_left_up(self):
        """Negative offset should shift center left and up."""
        logits = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        offset = np.array([-0.3, -0.5])
        hyps = np.array([
            [100.0, 90.0], [112.0, 100.0], [109.76, 100.0], [105.0, 95.0], [112.0, 112.0]
        ])

        cx, cy, _, _ = decode_center_with_confidence(logits, offset, hyps, offset_scale=10.0)
        assert cx < 112.0
        assert cy < 100.0
        assert abs(cx - (112.0 - 0.3 * 10.0)) < 0.01
        assert abs(cy - (100.0 - 0.5 * 10.0)) < 0.01

    def test_board_prior_hypothesis_selected(self):
        """Argmax should pick board_prior (index 2) when it has highest logit."""
        logits = np.array([0.0, 0.0, 3.0, 0.0, 0.0])
        offset = np.array([0.0, 0.0])
        hyps = np.array([
            [100.0, 90.0], [112.0, 100.0], [109.76, 100.0], [105.0, 95.0], [112.0, 112.0]
        ])

        cx, cy, _, _ = decode_center_with_confidence(logits, offset, hyps)
        assert abs(cx - 109.76) < 0.01
        assert abs(cy - 100.0) < 0.01


# ---------------------------------------------------------------------------
# Confidence and fallback thresholds
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_high_margin_low_entropy_small_offset_is_confident(self):
        """Clear winner, low uncertainty, small refinement = high confidence."""
        logits = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
        offset = np.array([0.1, 0.1])

        conf, is_high = compute_confidence(
            logits, offset,
            logit_margin_threshold=2.0,
            max_entropy=1.2,
            max_offset=8.0,
            offset_scale=10.0,
        )
        assert is_high
        assert conf > 0.7

    def test_equal_logits_is_not_confident(self):
        """Equal logits = maximum uncertainty = not confident."""
        logits = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        offset = np.array([0.0, 0.0])

        conf, is_high = compute_confidence(
            logits, offset,
            logit_margin_threshold=2.0,
            max_entropy=1.2,
            max_offset=8.0,
            offset_scale=10.0,
        )
        assert not is_high
        # Entropy of uniform 5-class distribution = ln(5) ≈ 1.609 > 1.2
        assert conf < 0.5

    def test_large_offset_reduces_confidence(self):
        """Large offset should reduce confidence even with clear logit winner."""
        logits = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
        offset = np.array([0.9, 0.9])

        conf, is_high = compute_confidence(
            logits, offset,
            logit_margin_threshold=2.0,
            max_entropy=1.2,
            max_offset=8.0,
            offset_scale=10.0,
        )
        assert not is_high

    def test_margin_threshold_gate(self):
        """Logit margin below threshold should fail confidence gate."""
        logits = np.array([1.5, 1.0, 0.0, 0.0, 0.0])
        offset = np.array([0.0, 0.0])

        _, is_high = compute_confidence(
            logits, offset,
            logit_margin_threshold=2.0,
            max_entropy=1.2,
            max_offset=8.0,
            offset_scale=10.0,
        )
        assert not is_high

    def test_entropy_threshold_gate(self):
        """High entropy should fail confidence gate."""
        logits = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        offset = np.array([0.0, 0.0])

        _, is_high = compute_confidence(
            logits, offset,
            logit_margin_threshold=0.5,
            max_entropy=0.5,
            max_offset=8.0,
            offset_scale=10.0,
        )
        assert not is_high

    def test_confidence_score_is_bounded(self):
        """Confidence score should always be in [0, 1]."""
        for _ in range(20):
            logits = np.random.randn(5) * 3
            offset = np.random.randn(2) * 0.5
            conf, _ = compute_confidence(
                logits, offset,
                logit_margin_threshold=2.0,
                max_entropy=1.2,
                max_offset=8.0,
                offset_scale=10.0,
            )
            assert 0.0 <= conf <= 1.0

    def test_perfect_prediction_gives_max_confidence(self):
        """Very strong logit winner + zero offset should give confidence near 1.0."""
        logits = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
        offset = np.array([0.0, 0.0])

        conf, is_high = compute_confidence(
            logits, offset,
            logit_margin_threshold=2.0,
            max_entropy=1.2,
            max_offset=8.0,
            offset_scale=10.0,
        )
        assert is_high
        assert conf > 0.95

    def test_abstain_gate_actually_abstains(self):
        """The gate should abstain when signals are ambiguous."""
        # Two hypotheses with similar logits
        logits = np.array([2.0, 1.8, 0.0, 0.0, 0.0])
        offset = np.array([0.6, 0.5])

        _, is_high = compute_confidence(
            logits, offset,
            logit_margin_threshold=2.0,
            max_entropy=1.2,
            max_offset=8.0,
            offset_scale=10.0,
        )
        assert not is_high  # margin=0.2 < 2.0
