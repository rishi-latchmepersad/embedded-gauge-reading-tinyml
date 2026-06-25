"""Shared TFLite helpers for the geometry heatmap export and replay flow.

The geometry heatmap model needs a couple of small compatibility shims:
- Keras v3 occasionally stores a stray ``quantization_config`` field on Dense
  layers that older loaders do not accept.
- TFLite inference needs explicit quantize/dequantize helpers so the export,
  contract inspection, and replay scripts all agree on tensor handling.

Keeping those details here makes the Phase 7 scripts thinner and easier to test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import keras
import numpy as np
import tensorflow as tf

from embedded_gauge_reading_tinyml.heatmap_utils import argmax_2d, softargmax_2d


@dataclass(frozen=True)
class LoadedTFLiteModel:
    """A loaded TFLite interpreter plus its cached tensor metadata."""

    interpreter: tf.lite.Interpreter
    input_details: dict[str, Any]
    output_details: list[dict[str, Any]]


HeatmapDecodeMethod = Literal[
    "softargmax",
    "argmax",
    "local_window_softargmax",
    "peak_weighted_centroid",
]


def patch_dense_quantization_config_deserialization() -> Callable[[], None]:
    """Patch Keras Dense deserialization to ignore legacy quantization fields."""

    original_keras_from_config = keras.layers.Dense.from_config
    original_tf_from_config = tf.keras.layers.Dense.from_config

    def _patched_from_config(cls: type[keras.layers.Dense], config: dict[str, Any]) -> keras.layers.Dense:
        """Drop unsupported config fields before recreating Dense layers."""

        cleaned_config = dict(config)
        cleaned_config.pop("quantization_config", None)
        return original_keras_from_config(cleaned_config)

    keras.layers.Dense.from_config = classmethod(_patched_from_config)
    tf.keras.layers.Dense.from_config = classmethod(_patched_from_config)

    def _restore() -> None:
        """Restore the original Dense deserializers."""

        keras.layers.Dense.from_config = original_keras_from_config
        tf.keras.layers.Dense.from_config = original_tf_from_config

    return _restore


def load_geometry_heatmap_keras_model(model_path: Path) -> keras.Model:
    """Load the saved geometry heatmap model in a Keras-v3-compatible way."""

    # Import models_geometry before loading so the @register_keras_serializable
    # decorator on Identity3x3Initializer runs and registers the class with
    # Keras's global serialization registry.  This MUST happen before the
    # load_model call or deserialization of the saved Conv2D layers will fail.
    import embedded_gauge_reading_tinyml.models_geometry  # noqa: F401
    from embedded_gauge_reading_tinyml.models_geometry import Identity3x3Initializer

    def _identity_init_pass_through(shape, dtype=None):
        """Legacy identity-initializer stub that simply forwards to the class."""
        return Identity3x3Initializer()(shape, dtype=dtype)

    custom_objects = {
        "_init": _identity_init_pass_through,
        "Identity3x3Initializer": Identity3x3Initializer,
        "embedded_gauge_reading_tinyml>Identity3x3Initializer": Identity3x3Initializer,
        "embedded_gauge_reading_tinyml.models_geometry.Identity3x3Initializer": Identity3x3Initializer,
    }
    restore = patch_dense_quantization_config_deserialization()
    try:
        with tf.keras.utils.custom_object_scope(custom_objects):
            return tf.keras.models.load_model(
                model_path,
                compile=False,
                safe_mode=False,
                custom_objects=custom_objects,
            )
    finally:
        restore()


def load_tflite_model(model_path: Path) -> LoadedTFLiteModel:
    """Load a TFLite model and cache its first input/output tensor metadata."""

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = list(interpreter.get_output_details())
    return LoadedTFLiteModel(
        interpreter=interpreter,
        input_details=input_details,
        output_details=output_details,
    )


def _quantization_scale_zero_point(details: Mapping[str, Any]) -> tuple[float, int]:
    """Read the scalar quantization tuple from a tensor description."""

    quantization = details.get("quantization", (0.0, 0))
    scale = float(quantization[0])
    zero_point = int(quantization[1])
    return scale, zero_point


def quantize_input_batch(batch: np.ndarray, input_details: Mapping[str, Any]) -> np.ndarray:
    """Quantize a float32 batch into the dtype expected by a TFLite input tensor."""

    input_array = np.asarray(batch, dtype=np.float32)
    input_dtype = np.dtype(input_details["dtype"])
    if input_dtype.kind == "f":
        return input_array.astype(input_dtype, copy=False)

    scale, zero_point = _quantization_scale_zero_point(input_details)
    if scale <= 0.0:
        raise ValueError("Quantized TFLite input tensor is missing a valid scale.")

    qinfo = np.iinfo(input_dtype)
    quantized = np.round(input_array / scale + zero_point)
    return np.clip(quantized, qinfo.min, qinfo.max).astype(input_dtype)


def dequantize_output_tensor(tensor: np.ndarray, output_details: Mapping[str, Any]) -> np.ndarray:
    """Convert one TFLite output tensor back into float32 values."""

    tensor_array = np.asarray(tensor)
    output_dtype = np.dtype(output_details["dtype"])
    if output_dtype.kind == "f":
        return tensor_array.astype(np.float32, copy=False)

    scale, zero_point = _quantization_scale_zero_point(output_details)
    if scale <= 0.0:
        raise ValueError("Quantized TFLite output tensor is missing a valid scale.")

    return (tensor_array.astype(np.float32) - float(zero_point)) * float(scale)


def reorder_tflite_outputs(
    outputs: Sequence[np.ndarray],
    semantic_output_order_indices: Sequence[int],
) -> list[np.ndarray]:
    """Reorder raw TFLite outputs into the semantic model order."""

    if len(outputs) != len(semantic_output_order_indices):
        raise ValueError(
            f"Output count {len(outputs)} does not match reorder mapping {len(semantic_output_order_indices)}."
        )
    reordered: list[np.ndarray] = []
    for index in semantic_output_order_indices:
        reordered.append(np.asarray(outputs[int(index)], dtype=np.float32))
    return reordered


def _as_2d_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Coerce a predicted heatmap to a dense 2D float array."""

    array = np.asarray(heatmap, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap after squeezing, got shape {array.shape!r}")
    return array


def _window_bounds(center_index: int, *, window_size: int, limit: int) -> tuple[int, int]:
    """Return an inclusive-exclusive slice around a peak index."""

    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")
    radius = window_size // 2
    start = max(0, center_index - radius)
    end = min(limit, center_index + radius + 1)
    if end - start < window_size:
        if start == 0:
            end = min(limit, window_size)
        elif end == limit:
            start = max(0, limit - window_size)
    return start, end


def _softargmax_on_window(heatmap: np.ndarray, *, window_size: int) -> tuple[float, float]:
    """Compute a local softargmax around the global peak."""

    peak_row, peak_col = argmax_2d(heatmap)
    row_start, row_end = _window_bounds(int(peak_row), window_size=window_size, limit=heatmap.shape[0])
    col_start, col_end = _window_bounds(int(peak_col), window_size=window_size, limit=heatmap.shape[1])
    local_heatmap = heatmap[row_start:row_end, col_start:col_end]
    local_row, local_col = softargmax_2d(local_heatmap)
    return float(local_row + row_start), float(local_col + col_start)


def _peak_weighted_centroid(heatmap: np.ndarray, *, window_size: int) -> tuple[float, float]:
    """Compute a centroid in a local peak window using the heatmap values as weights."""

    peak_row, peak_col = argmax_2d(heatmap)
    row_start, row_end = _window_bounds(int(peak_row), window_size=window_size, limit=heatmap.shape[0])
    col_start, col_end = _window_bounds(int(peak_col), window_size=window_size, limit=heatmap.shape[1])
    local_heatmap = np.asarray(heatmap[row_start:row_end, col_start:col_end], dtype=np.float32)
    total = float(np.sum(local_heatmap))
    if total <= 0.0:
        return float(peak_row), float(peak_col)
    y_coords, x_coords = np.meshgrid(
        np.arange(row_start, row_end, dtype=np.float32),
        np.arange(col_start, col_end, dtype=np.float32),
        indexing="ij",
    )
    expected_row = float(np.sum(local_heatmap * y_coords) / total)
    expected_col = float(np.sum(local_heatmap * x_coords) / total)
    return expected_row, expected_col


def decode_heatmap_point(
    heatmap: np.ndarray,
    *,
    method: HeatmapDecodeMethod = "softargmax",
    window_size: int = 3,
) -> tuple[float, float]:
    """Decode one heatmap into row/col coordinates in heatmap pixel space."""

    array = _as_2d_heatmap(heatmap)
    if method == "softargmax":
        return softargmax_2d(array)
    if method == "argmax":
        return argmax_2d(array)
    if method == "local_window_softargmax":
        return _softargmax_on_window(array, window_size=window_size)
    if method == "peak_weighted_centroid":
        return _peak_weighted_centroid(array, window_size=window_size)
    raise ValueError(f"Unknown heatmap decode method: {method}")


def decode_heatmap_point_xy(
    heatmap: np.ndarray,
    *,
    method: HeatmapDecodeMethod = "softargmax",
    window_size: int = 3,
    heatmap_size: int | None = None,
    input_size: int = 224,
) -> tuple[float, float]:
    """Decode one heatmap into x/y pixel coordinates in crop space."""

    row, col = decode_heatmap_point(heatmap, method=method, window_size=window_size)
    size = int(heatmap_size) if heatmap_size is not None else int(np.squeeze(np.asarray(heatmap)).shape[0])
    x = float(col) * float(input_size - 1) / float(size - 1)
    y = float(row) * float(input_size - 1) / float(size - 1)
    return x, y


def run_tflite_model(bundle: LoadedTFLiteModel, batch: np.ndarray) -> list[np.ndarray]:
    """Run one batch through a TFLite interpreter and return dequantized outputs."""

    interpreter = bundle.interpreter
    input_details = bundle.input_details

    quantized_batch = quantize_input_batch(batch, input_details)
    interpreter.set_tensor(int(input_details["index"]), quantized_batch)
    interpreter.invoke()

    outputs: list[np.ndarray] = []
    for output_details in bundle.output_details:
        raw_output = interpreter.get_tensor(int(output_details["index"]))
        outputs.append(dequantize_output_tensor(raw_output, output_details))
    return outputs


def summarize_tflite_contract(model_path: Path) -> dict[str, Any]:
    """Inspect the input/output tensor contract for one exported TFLite model."""

    bundle = load_tflite_model(model_path)
    input_scale, input_zero_point = _quantization_scale_zero_point(bundle.input_details)
    output_contracts: list[dict[str, Any]] = []
    for output_details in bundle.output_details:
        output_scale, output_zero_point = _quantization_scale_zero_point(output_details)
        output_contracts.append(
            {
                "name": str(output_details.get("name", "")),
                "shape": [int(value) for value in np.asarray(output_details["shape"], dtype=np.int64).tolist()],
                "dtype": str(np.dtype(output_details["dtype"])),
                "quantization": {
                    "scale": output_scale,
                    "zero_point": output_zero_point,
                },
                "quantized": bool(np.dtype(output_details["dtype"]).kind != "f"),
            }
        )

    input_dtype = np.dtype(bundle.input_details["dtype"])
    return {
        "model_path": str(model_path),
        "input": {
            "name": str(bundle.input_details.get("name", "")),
            "shape": [int(value) for value in np.asarray(bundle.input_details["shape"], dtype=np.int64).tolist()],
            "dtype": str(input_dtype),
            "quantization": {
                "scale": input_scale,
                "zero_point": input_zero_point,
            },
            "quantized": bool(input_dtype.kind != "f"),
        },
        "outputs": output_contracts,
        "requires_dequantization": any(output["quantized"] for output in output_contracts),
    }


def iterate_batched(items: Iterable[np.ndarray], *, batch_size: int) -> Iterable[np.ndarray]:
    """Yield stacked mini-batches from a stream of equally shaped arrays."""

    batch: list[np.ndarray] = []
    for item in items:
        batch.append(np.asarray(item, dtype=np.float32))
        if len(batch) >= batch_size:
            yield np.stack(batch, axis=0)
            batch.clear()
    if batch:
        yield np.stack(batch, axis=0)
