"""Tests for the shared inner-Celsius and tip-focus mask helpers."""

from __future__ import annotations

import numpy as np

from embedded_gauge_reading_tinyml.inner_celsius_mask import (
    BACKGROUND_VALUE_FLOAT,
    BACKGROUND_VALUE_INT8,
    apply_tip_focus_lower_inset_mask,
    apply_tip_focus_lower_inset_mask_int8,
    create_tip_focus_lower_inset_mask,
)


def test_create_tip_focus_lower_inset_mask_blanks_expected_ellipse() -> None:
    """The helper should blank the lower inset ellipse and keep the rest visible."""
    mask = create_tip_focus_lower_inset_mask()

    assert mask.shape == (224, 224)
    assert mask.dtype == np.bool_
    assert not bool(mask[150, 112])
    assert not bool(mask[150, 166])
    assert bool(mask[0, 0])


def test_apply_tip_focus_lower_inset_mask_uses_float_background() -> None:
    """Float images should be masked with the shared zero-valued background."""
    image = np.ones((224, 224, 3), dtype=np.float32)
    masked = apply_tip_focus_lower_inset_mask(image)

    assert masked.dtype == np.float32
    assert np.allclose(masked[0, 0], 1.0)
    assert np.allclose(masked[150, 112], BACKGROUND_VALUE_FLOAT)
    assert np.allclose(masked[150, 166], BACKGROUND_VALUE_FLOAT)


def test_apply_tip_focus_lower_inset_mask_int8_uses_int8_background() -> None:
    """Int8 images should be masked with the model's quantization zero-point."""
    image = np.full((224, 224, 3), 17, dtype=np.int8)
    masked = apply_tip_focus_lower_inset_mask_int8(image)

    assert masked.dtype == np.int8
    assert int(masked[0, 0, 0]) == 17
    assert int(masked[150, 112, 0]) == BACKGROUND_VALUE_INT8
    assert int(masked[150, 166, 0]) == BACKGROUND_VALUE_INT8
