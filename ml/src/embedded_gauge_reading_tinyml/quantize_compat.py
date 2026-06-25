"""Compatibility helpers for loading older QAT Keras artifacts.

Some saved quantization-aware checkpoints in this repo were written with a
slightly different `QuantizeLayer` variable layout than the one provided by the
current TensorFlow Model Optimization package. The helper below preserves the
current behavior when the variable counts match, and falls back to a
best-effort positional restore when the stored state is shorter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras.quantize_layer import (
    QuantizeLayer as _BaseQuantizeLayer,
)
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import (
    QuantizeWrapperV2 as _BaseQuantizeWrapperV2,
)


def _load_own_variables_compat(self: Any, store: Any) -> None:
    """Load variable state, tolerating shorter legacy stores when needed."""

    self._update_trackables()
    all_vars = self._trainable_weights + self._non_trainable_weights
    saved_values: list[Any] = []
    for key in sorted(store.keys(), key=lambda value: int(value)):
        saved_values.append(store[key])

    used_indices: set[int] = set()
    for variable in all_vars:
        variable_shape = tuple(int(dim) for dim in variable.shape)
        match_index = None
        for index, value in enumerate(saved_values):
            if index in used_indices:
                continue
            value_shape = tuple(int(dim) for dim in np.shape(value))
            if value_shape == variable_shape:
                match_index = index
                break
        if match_index is None:
            continue
        variable.assign(saved_values[match_index])
        used_indices.add(match_index)


# Monkey-patch the TF-MOT class in place so deserialization sees the fallback.
_BaseQuantizeLayer.load_own_variables = _load_own_variables_compat  # type: ignore[method-assign]
_BaseQuantizeWrapperV2.load_own_variables = _load_own_variables_compat  # type: ignore[method-assign]


def quantize_load_scope() -> tfmot.quantization.keras.quantize_scope:
    """Return a Keras quantization load scope with compatibility shims."""

    return tfmot.quantization.keras.quantize_scope({"QuantizeLayer": _BaseQuantizeLayer})
