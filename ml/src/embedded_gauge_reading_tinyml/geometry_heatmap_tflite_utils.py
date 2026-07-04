"""TFLite and Keras helpers for geometry heatmap training and replay."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras


@dataclass(frozen=True, slots=True)
class LoadedTFLiteModel:
    """A loaded TFLite interpreter plus its tensor metadata."""

    interpreter: tf.lite.Interpreter
    input_details: dict[str, Any]
    output_details: list[dict[str, Any]]


def load_geometry_heatmap_keras_model(model_path: Path) -> keras.Model:
    """Load a serialized geometry heatmap Keras model."""

    return keras.models.load_model(model_path, compile=False, safe_mode=False)


def load_tflite_model(model_path: Path) -> LoadedTFLiteModel:
    """Load one TFLite model and capture its input/output tensor metadata."""

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    return LoadedTFLiteModel(
        interpreter=interpreter,
        input_details=input_details,
        output_details=output_details,
    )


def iterate_batched(items: Sequence[np.ndarray], *, batch_size: int) -> Iterable[np.ndarray]:
    """Yield contiguous slices from a sequence of arrays."""

    for index in range(0, len(items), batch_size):
        yield np.asarray(items[index : index + batch_size], dtype=np.float32)


def _quantize_input(batch: np.ndarray, input_details: dict[str, Any]) -> np.ndarray:
    """Quantize a float batch for an int8 TFLite interpreter."""

    scale, zero_point = input_details["quantization"]
    if float(scale) == 0.0:
        return np.asarray(batch)
    qmin = np.iinfo(np.int8).min
    qmax = np.iinfo(np.int8).max
    quantized = np.round(np.asarray(batch, dtype=np.float32) / float(scale) + float(zero_point))
    return np.clip(quantized, qmin, qmax).astype(np.int8)


def _dequantize_output(tensor: np.ndarray, details: dict[str, Any]) -> np.ndarray:
    """Dequantize one output tensor if the model is quantized."""

    scale, zero_point = details["quantization"]
    if float(scale) == 0.0:
        return np.asarray(tensor, dtype=np.float32)
    return (np.asarray(tensor, dtype=np.float32) - float(zero_point)) * float(scale)


def run_tflite_model(bundle: LoadedTFLiteModel, batch: np.ndarray) -> list[np.ndarray]:
    """Run one batch through a TFLite interpreter and return dequantized outputs."""

    interpreter = bundle.interpreter
    input_details = bundle.input_details
    output_details = bundle.output_details
    input_array = np.asarray(batch, dtype=np.float32)
    if input_details["dtype"] == np.int8:
        input_array = _quantize_input(input_array, input_details)
    interpreter.set_tensor(int(input_details["index"]), input_array)
    interpreter.invoke()
    outputs: list[np.ndarray] = []
    for details in output_details:
        raw = interpreter.get_tensor(int(details["index"]))
        outputs.append(_dequantize_output(raw, details))
    return outputs


def reorder_tflite_outputs(outputs: Sequence[np.ndarray], order_indices: Sequence[int]) -> list[np.ndarray]:
    """Reorder model outputs into semantic order."""

    if not order_indices:
        return [np.asarray(output) for output in outputs]
    reordered: list[np.ndarray] = []
    for index in order_indices:
        reordered.append(np.asarray(outputs[int(index)]))
    return reordered


def summarize_tflite_contract(model_path: Path) -> dict[str, Any]:
    """Summarize the TFLite input/output contract for export and replay logs."""

    bundle = load_tflite_model(model_path)
    inputs = [
        {
            "name": bundle.input_details.get("name", ""),
            "shape": [int(dim) for dim in bundle.input_details["shape"]],
            "dtype": str(bundle.input_details["dtype"]),
            "quantization": list(bundle.input_details.get("quantization", (0.0, 0))),
        }
    ]
    outputs = []
    for details in bundle.output_details:
        outputs.append(
            {
                "name": details.get("name", ""),
                "shape": [int(dim) for dim in details["shape"]],
                "dtype": str(details["dtype"]),
                "quantization": list(details.get("quantization", (0.0, 0))),
            }
        )
    contract: dict[str, Any] = {"input": inputs[0], "outputs": outputs}
    contract_path = model_path.with_suffix(".contract.json")
    if contract_path.exists():
        try:
            contract.update(json.loads(contract_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    return contract

