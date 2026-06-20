"""Tests for the polar QAT training wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from embedded_gauge_reading_tinyml.polar_model import build_polar_board_friendly_mask_model
from embedded_gauge_reading_tinyml.polar_qat import (
    PolarMaskQATTrainingModel,
    angle_vector_cosine_loss,
)


def test_polar_qat_wrapper_returns_quantized_mask_and_value() -> None:
    """The wrapper should emit the training-time mask and decoded value outputs."""

    base_model = build_polar_board_friendly_mask_model(polar_size=160)
    qat_model = PolarMaskQATTrainingModel(base_model=base_model, output_noise_stddev=0.0)
    batch = np.random.default_rng(7).random((2, 160, 160, 3)).astype(np.float32)

    outputs = qat_model(batch, training=False)
    needle_mask = np.asarray(outputs["needle_mask"], dtype=np.float32)
    gauge_value = np.asarray(outputs["gauge_value"], dtype=np.float32)
    angle_vector = np.asarray(outputs["needle_angle_vector"], dtype=np.float32)

    assert needle_mask.shape == (2, 160, 160, 1)
    assert gauge_value.shape == (2, 1)
    assert angle_vector.shape == (2, 2)
    assert float(needle_mask.min()) >= 0.0
    assert float(needle_mask.max()) <= 1.0
    assert np.allclose(np.linalg.norm(angle_vector, axis=-1), 1.0, atol=1e-5)


def test_polar_qat_wrapper_can_expose_aux_profile_value() -> None:
    """The wrapper should optionally expose the auxiliary profile readout."""

    base_model = build_polar_board_friendly_mask_model(polar_size=160)
    qat_model = PolarMaskQATTrainingModel(
        base_model=base_model,
        output_noise_stddev=0.0,
        profile_head_units=32,
    )
    batch = np.random.default_rng(8).random((2, 160, 160, 3)).astype(np.float32)

    outputs = qat_model(batch, training=False)

    assert "profile_value_aux" in outputs
    profile_value = np.asarray(outputs["profile_value_aux"], dtype=np.float32)
    assert profile_value.shape == (2, 1)


def test_angle_vector_cosine_loss_is_zero_for_identical_vectors() -> None:
    """The angle-vector auxiliary loss should vanish on matching unit vectors."""

    y_true = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    loss = angle_vector_cosine_loss(y_true, y_true).numpy()

    assert float(loss) == pytest.approx(0.0, abs=1e-7)
