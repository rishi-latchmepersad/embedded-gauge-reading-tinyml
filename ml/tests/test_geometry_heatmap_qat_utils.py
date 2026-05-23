"""Tests for the geometry heatmap QAT helper utilities."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from embedded_gauge_reading_tinyml.geometry_heatmap_qat_utils import fake_quantize_01_tensor


def test_fake_quantize_01_tensor_preserves_shape_and_unit_interval() -> None:
    """Fake quantization should keep sigmoid-bounded tensors inside [0, 1]."""

    tensor = tf.constant([[0.0, 0.1, 0.5, 0.9, 1.0]], dtype=tf.float32)
    quantized = fake_quantize_01_tensor(tensor)

    quantized_array = np.asarray(quantized.numpy(), dtype=np.float32)
    assert quantized_array.shape == (1, 5)
    assert float(quantized_array.min()) >= 0.0
    assert float(quantized_array.max()) <= 1.0
    assert float(quantized_array[0, 0]) == 0.0
    assert float(quantized_array[0, 4]) == 1.0


def test_fake_quantize_01_tensor_rounds_to_a_nearby_8bit_grid() -> None:
    """A mid-range probability should land on a nearby 8-bit quantization level."""

    tensor = tf.constant([0.5], dtype=tf.float32)
    quantized = fake_quantize_01_tensor(tensor)

    assert float(quantized.numpy()[0]) == np.float32(128.0 / 255.0)
