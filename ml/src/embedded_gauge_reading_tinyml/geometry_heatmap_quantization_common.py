"""Shared helpers for geometry heatmap quantization analysis and replay."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from tensorflow import keras

from embedded_gauge_reading_tinyml.geometry_board_replay import build_board_replay_sample
from embedded_gauge_reading_tinyml.geometry_heatmap_debug_utils import load_clean_geometry_examples
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import (
    LoadedTFLiteModel,
    iterate_batched,
    load_geometry_heatmap_keras_model,
    load_tflite_model,
    reorder_tflite_outputs,
    run_tflite_model,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import (
    HeatmapSample,
    load_selected_calibration_candidate,
    select_examples_from_split,
)
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import (
    GeometryDecodeMethod,
    GeometryDecodedPrediction,
    GeometryGuardrailResult,
    GeometryGuardrailThresholds,
    decode_heatmap_geometry_prediction,
    apply_geometry_guardrails,
)


SELECTED_PREPROCESSING_MODE = "python_training_rgb_bilinear"
DEFAULT_PREPROCESSING_MODE = SELECTED_PREPROCESSING_MODE
DEFAULT_INPUT_SIZE = 224
DEFAULT_HEATMAP_SIZE = 56
DEFAULT_SIGMA_PIXELS = 5.0
DEFAULT_SEMANTIC_OUTPUT_ORDER = [1, 0, 2]


@dataclass(frozen=True)
class SplitSamples:
    """A deterministic set of samples for one manifest split."""

    split: str
    samples: list[HeatmapSample]


def resolve_repo_path(repo_root: Path, path: Path) -> Path:
    """Resolve a repository-relative path."""

    return path if path.is_absolute() else repo_root / path


def load_semantic_output_order_indices(contract_path: Path) -> list[int]:
    """Load the semantic TFLite output reorder mapping if present."""

    with contract_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "semantic_output_order_indices" in payload:
        return [int(value) for value in payload["semantic_output_order_indices"]]
    if "int8" in payload and "semantic_output_order_indices" in payload["int8"]:
        return [int(value) for value in payload["int8"]["semantic_output_order_indices"]]
    return list(DEFAULT_SEMANTIC_OUTPUT_ORDER)


def load_split_samples(
    manifest_path: Path,
    base_path: Path,
    *,
    split: str,
    mode: str = SELECTED_PREPROCESSING_MODE,
    input_size: int = DEFAULT_INPUT_SIZE,
    heatmap_size: int = DEFAULT_HEATMAP_SIZE,
    sigma_pixels: float = DEFAULT_SIGMA_PIXELS,
) -> SplitSamples:
    """Load clean manifest rows for one split and build board replay samples."""

    examples = load_clean_geometry_examples(manifest_path)
    split_examples = select_examples_from_split(examples, split=split)
    samples = [
        build_board_replay_sample(
            example,
            base_path,
            mode=mode,  # type: ignore[arg-type]
            input_size=input_size,
            heatmap_size=heatmap_size,
            sigma_pixels=sigma_pixels,
        )
        for example in split_examples
    ]
    return SplitSamples(split=split, samples=samples)


def load_models(model_path: Path, tflite_path: Path) -> tuple[keras.Model, LoadedTFLiteModel]:
    """Load the saved Keras model and the matching TFLite model."""

    keras_model = load_geometry_heatmap_keras_model(model_path)
    tflite_model = load_tflite_model(tflite_path)
    return keras_model, tflite_model


def predict_keras_outputs(model: keras.Model, inputs: Sequence[np.ndarray], *, batch_size: int = 16) -> list[np.ndarray]:
    """Predict all Keras outputs for one list of preprocessed inputs."""

    batches: list[list[np.ndarray]] = []
    for batch in iterate_batched(inputs, batch_size=batch_size):
        outputs = model(batch, training=False)
        if isinstance(outputs, Mapping):
            ordered = [np.asarray(output, dtype=np.float32) for output in list(outputs.values())]
        else:
            ordered = [np.asarray(output, dtype=np.float32) for output in list(outputs)]
        batches.append(ordered)
    if not batches:
        raise ValueError("No Keras predictions were produced.")
    merged: list[np.ndarray] = []
    for output_index in range(len(batches[0])):
        merged.append(np.concatenate([batch_outputs[output_index] for batch_outputs in batches], axis=0))
    return merged


def predict_tflite_outputs(
    bundle: LoadedTFLiteModel,
    inputs: Sequence[np.ndarray],
    *,
    semantic_output_order_indices: Sequence[int] | None = None,
) -> list[np.ndarray]:
    """Predict TFLite outputs and reorder them into semantic model order."""

    outputs_accumulator: list[list[np.ndarray]] | None = None
    for input_array in inputs:
        raw_outputs = run_tflite_model(bundle, np.expand_dims(np.asarray(input_array, dtype=np.float32), axis=0))
        ordered_outputs = (
            reorder_tflite_outputs(raw_outputs, semantic_output_order_indices)
            if semantic_output_order_indices is not None
            else [np.asarray(output, dtype=np.float32) for output in raw_outputs]
        )
        if outputs_accumulator is None:
            outputs_accumulator = [[] for _ in ordered_outputs]
        for index, tensor in enumerate(ordered_outputs):
            outputs_accumulator[index].append(np.asarray(tensor, dtype=np.float32))
    if outputs_accumulator is None:
        raise ValueError("No TFLite predictions were produced.")
    return [np.concatenate(output_tensors, axis=0) for output_tensors in outputs_accumulator]


def decode_and_guard(
    sample: HeatmapSample,
    center_heatmap: np.ndarray,
    tip_heatmap: np.ndarray,
    confidence: float,
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    *,
    decode_method: GeometryDecodeMethod = "softargmax",
    window_size: int = 3,
) -> tuple[GeometryDecodedPrediction, GeometryGuardrailResult]:
    """Decode one sample and immediately apply guardrails."""

    decoded = decode_heatmap_geometry_prediction(
        sample,
        center_heatmap,
        tip_heatmap,
        confidence,
        calibration_candidate,
        decode_method=decode_method,
        window_size=window_size,
    )
    guarded = apply_geometry_guardrails(decoded, thresholds)
    return decoded, guarded
