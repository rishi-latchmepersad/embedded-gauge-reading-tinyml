"""Tests for the board-friendly polar mask model."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from embedded_gauge_reading_tinyml.polar_model import build_polar_board_friendly_mask_model


def test_board_friendly_polar_model_exports_and_predicts_mask(tmp_path) -> None:
    """The compact polar model should stay small and round-trip through Keras save/load."""

    model = build_polar_board_friendly_mask_model(polar_size=160)
    batch = np.random.default_rng(42).random((2, 160, 160, 3)).astype(np.float32)

    outputs = model.predict(batch, verbose=0)
    assert outputs.shape == (2, 160, 160, 1)
    assert model.count_params() < 300_000

    model_path = tmp_path / "polar_board_model.keras"
    model.save(model_path)
    loaded = tf.keras.models.load_model(model_path)
    loaded_outputs = loaded.predict(batch, verbose=0)

    assert loaded_outputs.shape == (2, 160, 160, 1)
