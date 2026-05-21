"""
Unit tests for gauge geometry utilities.

These tests verify the deterministic geometry calculations for the inner Celsius dial.
Tests cover:
- Angle calculation from center and tip coordinates
- Temperature conversion from angle
- Circular angle error calculation
- Wraparound handling at 0/360 boundary
"""

import pytest
from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
    circular_angle_error_degrees,
)


class TestAngleFromCenterToTip:
    """Tests for angle_degrees_from_center_to_tip function."""

    def test_tip_directly_right_of_center(self):
        """Tip at 0 degrees (directly right of center)."""
        # When tip is directly to the right, angle should be 0
        angle = angle_degrees_from_center_to_tip(
            center_x_pixels=100.0,
            center_y_pixels=100.0,
            tip_x_pixels=200.0,
            tip_y_pixels=100.0,
        )
        assert angle == pytest.approx(0.0, abs=1e-6)

    def test_tip_directly_below_center(self):
        """Tip at 90 degrees (directly below center in image space)."""
        # In image space, Y increases downward, so below is +90 degrees
        angle = angle_degrees_from_center_to_tip(
            center_x_pixels=100.0,
            center_y_pixels=100.0,
            tip_x_pixels=100.0,
            tip_y_pixels=200.0,
        )
        assert angle == pytest.approx(90.0, abs=1e-6)

    def test_tip_directly_left_of_center(self):
        """Tip at 180 degrees (directly left of center)."""
        angle = angle_degrees_from_center_to_tip(
            center_x_pixels=100.0,
            center_y_pixels=100.0,
            tip_x_pixels=0.0,
            tip_y_pixels=100.0,
        )
        assert angle == pytest.approx(180.0, abs=1e-6)

    def test_tip_directly_above_center(self):
        """Tip at 270 degrees (directly above center in image space)."""
        # In image space, Y decreases upward, so above is 270 degrees (or -90)
        angle = angle_degrees_from_center_to_tip(
            center_x_pixels=100.0,
            center_y_pixels=100.0,
            tip_x_pixels=100.0,
            tip_y_pixels=0.0,
        )
        assert angle == pytest.approx(270.0, abs=1e-6)

    def test_tip_at_45_degrees(self):
        """Tip at 45 degrees (down-right diagonal)."""
        # 45 degrees: equal dx and dy, both positive
        angle = angle_degrees_from_center_to_tip(
            center_x_pixels=100.0,
            center_y_pixels=100.0,
            tip_x_pixels=100.0 + 10.0,
            tip_y_pixels=100.0 + 10.0,
        )
        assert angle == pytest.approx(45.0, abs=1e-6)

    def test_tip_at_225_degrees(self):
        """Tip at 225 degrees (up-left diagonal)."""
        # 225 degrees: equal dx and dy, both negative
        angle = angle_degrees_from_center_to_tip(
            center_x_pixels=100.0,
            center_y_pixels=100.0,
            tip_x_pixels=100.0 - 10.0,
            tip_y_pixels=100.0 - 10.0,
        )
        assert angle == pytest.approx(225.0, abs=1e-6)

    def test_same_center_and_tip(self):
        """When center and tip are the same, angle is 0 (undefined, but returns 0)."""
        angle = angle_degrees_from_center_to_tip(
            center_x_pixels=100.0,
            center_y_pixels=100.0,
            tip_x_pixels=100.0,
            tip_y_pixels=100.0,
        )
        # atan2(0, 0) returns 0, which normalizes to 0
        assert angle == pytest.approx(0.0, abs=1e-6)


class TestCelsiusFromAngle:
    """Tests for celsius_from_inner_dial_angle_degrees function."""

    def test_cold_end_135_degrees(self):
        """135 degrees should map to -30C (cold end)."""
        temp = celsius_from_inner_dial_angle_degrees(135.0)
        assert temp == pytest.approx(-30.0, abs=1e-6)

    def test_hot_end_45_degrees(self):
        """45 degrees should map to 50C (hot end)."""
        temp = celsius_from_inner_dial_angle_degrees(45.0)
        assert temp == pytest.approx(50.0, abs=1e-6)

    def test_middle_of_sweep_270_degrees(self):
        """270 degrees (middle of 270 deg sweep from 135) should map to 10C.
        
        The sweep goes from 135 deg to 45 deg (wrapping through 0/360).
        Middle of sweep is at 135 + 270/2 = 270 degrees.
        At middle: -30 + 0.5 * 80 = 10C.
        """
        temp = celsius_from_inner_dial_angle_degrees(270.0)
        assert temp == pytest.approx(10.0, abs=1e-6)

    def test_zero_celsius_angle(self):
        """0C occurs at 236.25 degrees (30/80 = 0.375 of the sweep from cold end).
        
        0C is 30 degrees above -30C, so it's at 30/80 = 0.375 of the sweep.
        Angle = 135 + 0.375 * 270 = 236.25 degrees.
        """
        zero_c_angle = 135.0 + (30.0 / 80.0) * 270.0
        temp = celsius_from_inner_dial_angle_degrees(zero_c_angle)
        assert temp == pytest.approx(0.0, abs=1e-6)

    def test_wraparound_405_degrees(self):
        """405 degrees (45 + 360) should also map to 50C."""
        temp = celsius_from_inner_dial_angle_degrees(405.0)
        assert temp == pytest.approx(50.0, abs=1e-6)

    def test_negative_angle(self):
        """-315 degrees (equivalent to 45) should map to 50C."""
        temp = celsius_from_inner_dial_angle_degrees(-315.0)
        assert temp == pytest.approx(50.0, abs=1e-6)

    def test_quarter_sweep_180_degrees(self):
        """180 degrees should be partway from cold toward middle."""
        # (180 - 135) / 270 = 45/270 = 1/6 of sweep
        # -30 + (1/6) * 80 = -30 + 13.33 = -16.67C
        temp = celsius_from_inner_dial_angle_degrees(180.0)
        expected = -30.0 + (45.0 / 270.0) * 80.0
        assert temp == pytest.approx(expected, abs=1e-6)

    def test_custom_calibration(self):
        """Test with custom calibration parameters."""
        # Custom: cold at 0, sweep 180, range 0-100C
        temp = celsius_from_inner_dial_angle_degrees(
            angle_degrees=90.0,
            cold_angle_degrees=0.0,
            sweep_degrees=180.0,
            minimum_celsius=0.0,
            maximum_celsius=100.0,
        )
        # 90 is halfway through 180 sweep, so should be 50C
        assert temp == pytest.approx(50.0, abs=1e-6)


class TestCircularAngleError:
    """Tests for circular_angle_error_degrees function."""

    def test_identical_angles(self):
        """Error between identical angles is 0."""
        error = circular_angle_error_degrees(45.0, 45.0)
        assert error == pytest.approx(0.0, abs=1e-6)

    def test_simple_difference(self):
        """Error between 10 and 20 is 10."""
        error = circular_angle_error_degrees(10.0, 20.0)
        assert error == pytest.approx(10.0, abs=1e-6)

    def test_wraparound_359_and_1(self):
        """Error between 359 and 1 is 2."""
        error = circular_angle_error_degrees(359.0, 1.0)
        assert error == pytest.approx(2.0, abs=1e-6)

    def test_wraparound_1_and_359(self):
        """Error between 1 and 359 is 2 (order does not matter)."""
        error = circular_angle_error_degrees(1.0, 359.0)
        assert error == pytest.approx(2.0, abs=1e-6)

    def test_opposite_angles(self):
        """Error between 0 and 180 is 180 (maximum error)."""
        error = circular_angle_error_degrees(0.0, 180.0)
        assert error == pytest.approx(180.0, abs=1e-6)

    def test_wraparound_350_and_10(self):
        """Error between 350 and 10 is 20."""
        error = circular_angle_error_degrees(350.0, 10.0)
        assert error == pytest.approx(20.0, abs=1e-6)

    def test_negative_angles(self):
        """Error handles negative angles correctly."""
        # -1 is equivalent to 359
        error = circular_angle_error_degrees(-1.0, 1.0)
        assert error == pytest.approx(2.0, abs=1e-6)

    def test_large_angles(self):
        """Error handles angles > 360 correctly."""
        # 361 is equivalent to 1
        error = circular_angle_error_degrees(361.0, 359.0)
        assert error == pytest.approx(2.0, abs=1e-6)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_angle_to_temp_roundtrip(self):
        """Verify angle calculations are consistent with temperature mapping."""
        # At 135 (cold end), temperature should be -30C
        temp_cold = celsius_from_inner_dial_angle_degrees(135.0)
        assert temp_cold == pytest.approx(-30.0, abs=1e-6)

        # At 45 (hot end), temperature should be 50C
        temp_hot = celsius_from_inner_dial_angle_degrees(45.0)
        assert temp_hot == pytest.approx(50.0, abs=1e-6)

        # At 270 (middle of sweep), temperature should be 10C
        temp_middle = celsius_from_inner_dial_angle_degrees(270.0)
        assert temp_middle == pytest.approx(10.0, abs=1e-6)

    def test_synthetic_needle_positions(self):
        """Test with synthetic center/tip positions at known temperatures."""
        import math

        center_x, center_y = 1000.0, 1000.0
        radius = 100.0

        # Test at 225 degrees (down-left diagonal in image space)
        angle_rad = math.radians(225.0)
        tip_x = center_x + radius * math.cos(angle_rad)
        tip_y = center_y + radius * math.sin(angle_rad)

        # Calculate angle from positions
        computed_angle = angle_degrees_from_center_to_tip(
            center_x, center_y, tip_x, tip_y
        )

        # Should be very close to 225
        assert computed_angle == pytest.approx(225.0, abs=1e-6)

        # Temperature at 225 deg: (225-135)/270 * 80 - 30 = -3.33C
        temp = celsius_from_inner_dial_angle_degrees(computed_angle)
        expected_temp = -30.0 + ((225.0 - 135.0) / 270.0) * 80.0
        assert temp == pytest.approx(expected_temp, abs=1e-6)

    def test_full_range_coverage(self):
        """Test that full temperature range is covered."""
        # Cold end
        temp_cold = celsius_from_inner_dial_angle_degrees(135.0)
        assert temp_cold == pytest.approx(-30.0, abs=1e-6)

        # Hot end
        temp_hot = celsius_from_inner_dial_angle_degrees(45.0)
        assert temp_hot == pytest.approx(50.0, abs=1e-6)

        # Range should be 80C
        assert (temp_hot - temp_cold) == pytest.approx(80.0, abs=1e-6)
