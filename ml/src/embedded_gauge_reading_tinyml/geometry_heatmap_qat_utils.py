"""QAT helpers for geometry heatmap training."""

from __future__ import annotations

import tensorflow as tf


def fake_quantize_01_tensor(tensor: tf.Tensor, *, num_bits: int = 8) -> tf.Tensor:
    """Apply a small fake-quantization pass to a [0, 1] tensor.

    The geometry heatmap trainer uses this to keep the sigmoid-bounded heads
    close to the eventual int8 deployment contract during training.
    """

    values = tf.cast(tensor, tf.float32)
    quantized = tf.quantization.fake_quant_with_min_max_args(
        values,
        min=0.0,
        max=1.0,
        num_bits=num_bits,
        narrow_range=False,
    )
    return tf.cast(quantized, tf.float32)

