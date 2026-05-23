"""
Unit tests for geometry crop dataset utilities.

These tests verify:
- Identity crop preserves normalized coordinates correctly
- Shifted crop transforms center/tip correctly
- Scaled crop transforms center/tip correctly
- Invalid crops are rejected
- Generated jittered crops are deterministic with fixed seed
- Transformed center/tip coordinates remain inside [0, 1] for accepted crops
"""

import pytest
from pathlib import Path
from embedded_gauge_reading_tinyml.geometry_crop_dataset import (
    SourceGeometryExample,
    JitterParams,
    JitteredCrop,
    apply_jitter_to_crop,
    transform_point_to_crop,
    transform_normalized_to_224,
    validate_crop,
    create_jittered_crop,
    generate_jitter_params,
    generate_jittered_crops_for_example,
    CropRejectionReason,
)


class TestTransformPointToCrop:
    """Tests for coordinate transformation utilities."""

    def test_identity_crop_preserves_coordinates(self):
        """Identity crop (full image) should preserve relative coordinates."""
        # Source image is 1000x1000, crop is also 1000x1000
        # Point at (500, 500) should normalize to (0.5, 0.5)
        x_norm, y_norm = transform_point_to_crop(
            point_x_source=500.0,
            point_y_source=500.0,
            crop_x1=0,
            crop_y1=0,
            crop_x2=1000,
            crop_y2=1000,
        )
        assert x_norm == pytest.approx(0.5, abs=1e-6)
        assert y_norm == pytest.approx(0.5, abs=1e-6)

    def test_corner_normalization(self):
        """Corners should map to (0,0), (1,0), (0,1), (1,1)."""
        # Top-left corner
        x_norm, y_norm = transform_point_to_crop(
            100.0, 200.0, 100, 200, 500, 600
        )
        assert x_norm == pytest.approx(0.0, abs=1e-6)
        assert y_norm == pytest.approx(0.0, abs=1e-6)

        # Bottom-right corner
        x_norm, y_norm = transform_point_to_crop(
            500.0, 600.0, 100, 200, 500, 600
        )
        assert x_norm == pytest.approx(1.0, abs=1e-6)
        assert y_norm == pytest.approx(1.0, abs=1e-6)

    def test_shifted_crop_transforms_correctly(self):
        """Shifted crop should transform center/tip correctly."""
        # Original crop: (0, 0) to (400, 400)
        # Point at (200, 200) is at center (0.5, 0.5)
        # Shifted crop: (50, 50) to (450, 450)
        # Same point (200, 200) is now at (0.375, 0.375)
        x_norm, y_norm = transform_point_to_crop(
            200.0, 200.0, 50, 50, 450, 450
        )
        # (200 - 50) / (450 - 50) = 150 / 400 = 0.375
        assert x_norm == pytest.approx(0.375, abs=1e-6)
        assert y_norm == pytest.approx(0.375, abs=1e-6)

    def test_scaled_crop_transforms_correctly(self):
        """Scaled crop should transform center/tip correctly."""
        # Original: (0, 0) to (400, 400), point at (200, 200) = (0.5, 0.5)
        # Scaled by 0.5: (100, 100) to (300, 300)
        # Point at (200, 200) is still at center (0.5, 0.5)
        x_norm, y_norm = transform_point_to_crop(
            200.0, 200.0, 100, 100, 300, 300
        )
        assert x_norm == pytest.approx(0.5, abs=1e-6)
        assert y_norm == pytest.approx(0.5, abs=1e-6)

    def test_normalized_to_224_conversion(self):
        """Normalized coordinates should convert correctly to 224x224."""
        x_224, y_224 = transform_normalized_to_224(0.5, 0.5)
        assert x_224 == pytest.approx(112.0, abs=1e-6)
        assert y_224 == pytest.approx(112.0, abs=1e-6)

        x_224, y_224 = transform_normalized_to_224(0.0, 0.0)
        assert x_224 == pytest.approx(0.0, abs=1e-6)
        assert y_224 == pytest.approx(0.0, abs=1e-6)

        x_224, y_224 = transform_normalized_to_224(1.0, 1.0)
        assert x_224 == pytest.approx(224.0, abs=1e-6)
        assert y_224 == pytest.approx(224.0, abs=1e-6)


class TestValidateCrop:
    """Tests for crop validation."""

    def _make_example(self, width=1000, height=1000):
        """Helper to create a test example."""
        return SourceGeometryExample(
            image_path="test.jpg",
            temperature_c=25.0,
            source_width=width,
            source_height=height,
            loose_crop_x1=100,
            loose_crop_y1=100,
            loose_crop_x2=500,
            loose_crop_y2=500,
            center_x_source=300.0,
            center_y_source=300.0,
            tip_x_source=200.0,
            tip_y_source=250.0,
            dial_radius_source=150.0,
            split="train",
        )

    def test_valid_crop_is_accepted(self):
        """Valid crop should be accepted."""
        example = self._make_example()
        accepted, reason = validate_crop(
            example,
            100, 100, 500, 500,
            0.5, 0.5,
            0.25, 0.375,
        )
        assert accepted is True
        assert reason is None

    def test_crop_outside_image_is_rejected(self):
        """Crop extending outside image bounds should be rejected."""
        example = self._make_example(width=1000, height=1000)
        
        # Crop extends past right edge
        accepted, reason = validate_crop(
            example,
            100, 100, 1100, 500,
            0.5, 0.5,
            0.25, 0.375,
        )
        assert accepted is False
        assert reason == CropRejectionReason.CROP_OUTSIDE_IMAGE

    def test_crop_negative_bounds_is_rejected(self):
        """Crop with negative bounds should be rejected."""
        example = self._make_example()
        
        accepted, reason = validate_crop(
            example,
            -50, 100, 500, 500,
            0.5, 0.5,
            0.25, 0.375,
        )
        assert accepted is False
        assert reason == CropRejectionReason.CROP_OUTSIDE_IMAGE

    def test_crop_too_small_is_rejected(self):
        """Crop smaller than 32px should be rejected."""
        example = self._make_example()
        
        accepted, reason = validate_crop(
            example,
            100, 100, 130, 130,
            0.5, 0.5,
            0.25, 0.375,
        )
        assert accepted is False
        assert reason == CropRejectionReason.CROP_TOO_SMALL

    def test_crop_aspect_unreasonable_is_rejected(self):
        """Crop with unreasonable aspect ratio should be rejected."""
        example = self._make_example()
        
        # Aspect ratio < 0.5 (width/height = 20/100 = 0.2)
        accepted, reason = validate_crop(
            example,
            100, 100, 200, 500,
            0.5, 0.5,
            0.25, 0.375,
        )
        assert accepted is False
        assert reason == CropRejectionReason.CROP_ASPECT_UNREASONABLE

    def test_center_outside_crop_is_rejected(self):
        """Center outside [0, 1] should be rejected."""
        example = self._make_example()
        
        accepted, reason = validate_crop(
            example,
            100, 100, 500, 500,
            -0.1, 0.5,  # Center x is outside
            0.25, 0.375,
        )
        assert accepted is False
        assert reason == CropRejectionReason.CENTER_OUTSIDE_CROP

    def test_tip_outside_crop_is_rejected(self):
        """Tip outside [0, 1] should be rejected."""
        example = self._make_example()
        
        accepted, reason = validate_crop(
            example,
            100, 100, 500, 500,
            0.5, 0.5,
            1.5, 0.375,  # Tip x is outside
        )
        assert accepted is False
        assert reason == CropRejectionReason.TIP_OUTSIDE_CROP


class TestCreateJitteredCrop:
    """Tests for creating jittered crops."""

    def _make_example(self):
        """Helper to create a test example."""
        return SourceGeometryExample(
            image_path="test.jpg",
            temperature_c=25.0,
            source_width=1000,
            source_height=1000,
            loose_crop_x1=100,
            loose_crop_y1=100,
            loose_crop_x2=900,
            loose_crop_y2=900,
            center_x_source=500.0,
            center_y_source=500.0,
            tip_x_source=400.0,
            tip_y_source=450.0,
            dial_radius_source=300.0,
            split="train",
        )

    def test_identity_jitter_produces_valid_crop(self):
        """Identity jitter (no shift, scale=1, aspect=1) should produce valid crop."""
        example = self._make_example()
        jitter = JitterParams(shift_x=0, shift_y=0, scale=1.0, aspect=1.0)
        
        crop = create_jittered_crop(example, jitter)
        
        assert crop.accepted is True
        assert crop.rejection_reason is None
        assert crop.center_x_normalized == pytest.approx(0.5, abs=0.01)
        assert crop.center_y_normalized == pytest.approx(0.5, abs=0.01)

    def test_small_shift_produces_valid_crop(self):
        """Small shift should produce valid crop with transformed coordinates."""
        example = self._make_example()
        jitter = JitterParams(shift_x=10, shift_y=10, scale=1.0, aspect=1.0)
        
        crop = create_jittered_crop(example, jitter)
        
        assert crop.accepted is True
        # Center should shift slightly in normalized coords
        assert 0.0 <= crop.center_x_normalized <= 1.0
        assert 0.0 <= crop.center_y_normalized <= 1.0

    def test_large_shift_is_rejected(self):
        """Large shift that pushes crop outside image should be rejected."""
        example = self._make_example()
        # Shift by 500px will push crop outside
        jitter = JitterParams(shift_x=500, shift_y=500, scale=1.0, aspect=1.0)
        
        crop = create_jittered_crop(example, jitter)
        
        assert crop.accepted is False
        assert crop.rejection_reason == CropRejectionReason.CROP_OUTSIDE_IMAGE.value

    def test_extreme_scale_is_rejected(self):
        """Extreme scale that makes crop too small should be rejected."""
        example = self._make_example()
        # Scale of 0.01 will make crop tiny
        jitter = JitterParams(shift_x=0, shift_y=0, scale=0.01, aspect=1.0)
        
        crop = create_jittered_crop(example, jitter)
        
        assert crop.accepted is False
        assert crop.rejection_reason == CropRejectionReason.CROP_TOO_SMALL.value

    def test_normalized_coords_in_valid_range(self):
        """Accepted crops should have normalized coords in [0, 1]."""
        example = self._make_example()
        jitter = JitterParams(shift_x=5, shift_y=5, scale=1.0, aspect=1.0)
        
        crop = create_jittered_crop(example, jitter)
        
        if crop.accepted:
            assert 0.0 <= crop.center_x_normalized <= 1.0
            assert 0.0 <= crop.center_y_normalized <= 1.0
            assert 0.0 <= crop.tip_x_normalized <= 1.0
            assert 0.0 <= crop.tip_y_normalized <= 1.0


class TestDeterministicJitter:
    """Tests for deterministic jitter generation."""

    def test_fixed_seed_produces_same_jitter(self):
        """Fixed seed should produce identical jitter params."""
        rng1 = __import__('numpy').random.default_rng(42)
        rng2 = __import__('numpy').random.default_rng(42)
        
        jitter1 = generate_jitter_params(rng1)
        jitter2 = generate_jitter_params(rng2)
        
        assert jitter1.shift_x == jitter2.shift_x
        assert jitter1.shift_y == jitter2.shift_y
        assert jitter1.scale == pytest.approx(jitter2.scale, abs=1e-6)
        assert jitter1.aspect == pytest.approx(jitter2.aspect, abs=1e-6)

    def test_different_seed_produces_different_jitter(self):
        """Different seeds should produce different jitter params."""
        rng1 = __import__('numpy').random.default_rng(42)
        rng2 = __import__('numpy').random.default_rng(43)
        
        jitter1 = generate_jitter_params(rng1)
        jitter2 = generate_jitter_params(rng2)
        
        # At least one parameter should differ
        differs = (
            jitter1.shift_x != jitter2.shift_x or
            jitter1.shift_y != jitter2.shift_y or
            abs(jitter1.scale - jitter2.scale) > 1e-6 or
            abs(jitter1.aspect - jitter2.aspect) > 1e-6
        )
        assert differs

    def test_generate_crops_deterministic_with_seed(self):
        """Generating crops with fixed seed should be deterministic."""
        example = SourceGeometryExample(
            image_path="test.jpg",
            temperature_c=25.0,
            source_width=1000,
            source_height=1000,
            loose_crop_x1=100,
            loose_crop_y1=100,
            loose_crop_x2=900,
            loose_crop_y2=900,
            center_x_source=500.0,
            center_y_source=500.0,
            tip_x_source=400.0,
            tip_y_source=450.0,
            dial_radius_source=300.0,
            split="train",
        )
        
        crops1 = generate_jittered_crops_for_example(example, num_crops=3, seed=42)
        crops2 = generate_jittered_crops_for_example(example, num_crops=3, seed=42)
        
        assert len(crops1) == len(crops2)
        for c1, c2 in zip(crops1, crops2):
            assert c1.crop_x1 == c2.crop_x1
            assert c1.crop_y1 == c2.crop_y1
            assert c1.crop_x2 == c2.crop_x2
            assert c1.crop_y2 == c2.crop_y2


class TestJitterRanges:
    """Tests for jitter parameter ranges."""

    def test_shift_range_is_correct(self):
        """Shift should be in range [-20, +20]."""
        import random
        rng = __import__('numpy').random.default_rng(42)
        
        for _ in range(100):
            jitter = generate_jitter_params(rng)
            assert -20 <= jitter.shift_x <= 20
            assert -20 <= jitter.shift_y <= 20

    def test_scale_range_is_correct(self):
        """Scale should be in range [0.85, 1.25]."""
        import random
        rng = __import__('numpy').random.default_rng(42)
        
        for _ in range(100):
            jitter = generate_jitter_params(rng)
            assert 0.85 <= jitter.scale <= 1.25

    def test_aspect_range_is_correct(self):
        """Aspect should be in range [0.90, 1.10]."""
        import random
        rng = __import__('numpy').random.default_rng(42)
        
        for _ in range(100):
            jitter = generate_jitter_params(rng)
            assert 0.90 <= jitter.aspect <= 1.10

