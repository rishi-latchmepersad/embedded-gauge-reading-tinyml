"""Small helpers for geometry heatmap QAT-style fine-tuning.

The Phase 8 fallback path uses quantization-noise fine-tuning instead of
tensorflow_model_optimization QAT when the optional dependency is not
available.  Keeping the quantization round-trip in one pure helper makes it
easy to test independently.
"""

from __future__ import annotations

import tensorflow as tf


QAT_OUTPUT_MIN: float = 0.0
QAT_OUTPUT_MAX: float = 1.0
QAT_OUTPUT_NUM_BITS: int = 8


def fake_quantize_01_tensor(tensor: tf.Tensor, *, num_bits: int = QAT_OUTPUT_NUM_BITS) -> tf.Tensor:
    """Round-trip a [0, 1] tensor through fake affine quantization.

    The geometry heatmap model emits sigmoid-bounded outputs, so this helper
    simulates the 8-bit output side of the TFLite export without changing the
    clean inference graph.
    """

    values = tf.cast(tf.convert_to_tensor(tensor), tf.float32)
    quantized = tf.quantization.fake_quant_with_min_max_args(
        values,
        min=QAT_OUTPUT_MIN,
        max=QAT_OUTPUT_MAX,
        num_bits=int(num_bits),
        narrow_range=False,
    )
    return tf.cast(quantized, tf.float32)

