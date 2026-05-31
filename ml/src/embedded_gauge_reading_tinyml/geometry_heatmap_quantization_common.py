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
from embedded_gauge_reading_tinyml.gauge_geometry import angle_degrees_from_center_to_tip
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import (
    GeometryDecodeMethod,
    GeometryDecodedPrediction,
    GeometryGuardrailResult,
    GeometryGuardrailThresholds,
    decode_heatmap_geometry_prediction,
    apply_geometry_guardrails,
)
from embedded_gauge_reading_tinyml.geometry_temperature_calibration import (
    CalibrationCandidate,
    predict_temperature_from_candidate,
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
    inner_celsius_mask: bool = False,
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
            inner_celsius_mask=inner_celsius_mask,
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
    aux_coords: np.ndarray | None = None,
    aux_offset_map: np.ndarray | None = None,
    axis_logits: np.ndarray | None = None,
    offset_scale_px: float = 8.0,
) -> tuple[GeometryDecodedPrediction, GeometryGuardrailResult]:
    """Decode one sample and immediately apply guardrails.

    When aux_coords (4-element array [cx_norm, cy_norm, tx_norm, ty_norm])
    is provided, the auxiliary coordinates override the heatmap-decoded
    coordinates for geometry/angle/temperature prediction.  Heatmaps are
    still decoded for guardrail quality features (peak, spread, entropy).

    When aux_offset_map (112x112x4 array with channels
    [center_dx, center_dy, tip_dx, tip_dy]) is provided, the decoded
    heatmap coordinates are refined by reading per-pixel offsets and
    adding them.  offset_scale_px maps tanh [-1,1] to heatmap pixels.
    """

    import dataclasses

    decoded = decode_heatmap_geometry_prediction(
        sample,
        center_heatmap,
        tip_heatmap,
        confidence,
        calibration_candidate,
        decode_method=decode_method,
        window_size=window_size,
    )

    if aux_offset_map is not None:
        decoded = _apply_offset_map_correction(
            decoded, aux_offset_map, offset_scale_px, calibration_candidate,
            true_temperature_c=sample.metadata["temperature_c"],
        )

    if axis_logits is not None:
        decoded = _apply_axis_logits_decode(
            decoded, axis_logits, calibration_candidate,
            true_temperature_c=sample.metadata["temperature_c"],
        )

    if aux_coords is not None:
        cx_norm = float(aux_coords[0])
        cy_norm = float(aux_coords[1])
        tx_norm = float(aux_coords[2])
        ty_norm = float(aux_coords[3])

        predicted_center_x_224 = cx_norm * 223.0
        predicted_center_y_224 = cy_norm * 223.0
        predicted_tip_x_224 = tx_norm * 223.0
        predicted_tip_y_224 = ty_norm * 223.0

        predicted_angle_degrees = angle_degrees_from_center_to_tip(
            predicted_center_x_224, predicted_center_y_224,
            predicted_tip_x_224, predicted_tip_y_224,
        )
        predicted_temperature_c_calibrated = predict_temperature_from_candidate(
            predicted_angle_degrees, calibration_candidate,
        )

        decoded = dataclasses.replace(
            decoded,
            predicted_center_x_224=predicted_center_x_224,
            predicted_center_y_224=predicted_center_y_224,
            predicted_tip_x_224=predicted_tip_x_224,
            predicted_tip_y_224=predicted_tip_y_224,
            predicted_angle_degrees=float(predicted_angle_degrees),
            predicted_temperature_c_calibrated=float(predicted_temperature_c_calibrated),
            absolute_error_c_calibrated=float(
                abs(predicted_temperature_c_calibrated - float(sample.metadata["temperature_c"]))
            ),
        )

    guarded = apply_geometry_guardrails(decoded, thresholds)
    return decoded, guarded


def _apply_offset_map_correction(
    decoded: Any,
    aux_offset_map: np.ndarray,
    offset_scale_px: float,
    calibration_candidate: Any,
    *,
    true_temperature_c: float = 0.0,
) -> Any:
    """Refine heatmap-decoded coords using the local offset map.

    Reads per-pixel dx/dy offsets at the rounded heatmap-decoded position
    and adds them to get corrected 224-space coordinates.
    """
    import dataclasses

    hw = aux_offset_map.shape[0]  # 112

    # Convert 224-space decoded coords to heatmap pixel space
    scale = 224.0 / float(hw)
    cx_h = decoded.predicted_center_x_224 / scale
    cy_h = decoded.predicted_center_y_224 / scale
    tx_h = decoded.predicted_tip_x_224 / scale
    ty_h = decoded.predicted_tip_y_224 / scale

    # Round to nearest pixel for offset lookup
    col_c = int(np.clip(np.round(cx_h), 0, hw - 1))
    row_c = int(np.clip(np.round(cy_h), 0, hw - 1))
    col_t = int(np.clip(np.round(tx_h), 0, hw - 1))
    row_t = int(np.clip(np.round(ty_h), 0, hw - 1))

    # Read offsets: channels are [center_dx, center_dy, tip_dx, tip_dy]
    dx_c = float(aux_offset_map[row_c, col_c, 0]) * offset_scale_px
    dy_c = float(aux_offset_map[row_c, col_c, 1]) * offset_scale_px
    dx_t = float(aux_offset_map[row_t, col_t, 2]) * offset_scale_px
    dy_t = float(aux_offset_map[row_t, col_t, 3]) * offset_scale_px

    # Apply correction in 112-space, then scale to 224
    corrected_center_x_224 = (cx_h + dx_c) * scale
    corrected_center_y_224 = (cy_h + dy_c) * scale
    corrected_tip_x_224 = (tx_h + dx_t) * scale
    corrected_tip_y_224 = (ty_h + dy_t) * scale

    predicted_angle = angle_degrees_from_center_to_tip(
        corrected_center_x_224, corrected_center_y_224,
        corrected_tip_x_224, corrected_tip_y_224,
    )
    predicted_temp = predict_temperature_from_candidate(predicted_angle, calibration_candidate)

    return dataclasses.replace(
        decoded,
        predicted_center_x_224=corrected_center_x_224,
        predicted_center_y_224=corrected_center_y_224,
        predicted_tip_x_224=corrected_tip_x_224,
        predicted_tip_y_224=corrected_tip_y_224,
        predicted_angle_degrees=float(predicted_angle),
        predicted_temperature_c_calibrated=float(predicted_temp),
        absolute_error_c_calibrated=float(abs(predicted_temp - true_temperature_c)),
    )


def _apply_axis_logits_decode(
    decoded: Any,
    axis_logits: np.ndarray,
    calibration_candidate: Any,
    *,
    true_temperature_c: float = 0.0,
) -> Any:
    """Decode center/tip from axis_simcc logits using 1D softargmax w3.

    axis_logits: (4, 112) with [center_x, center_y, tip_x, tip_y].
    Each axis is decoded via softargmax with window 3, then converted
    from 112-bin space to 224-pixel coordinates.
    """
    import dataclasses

    def _decode_axis(logits: np.ndarray) -> float:
        """1D softargmax with w3 around the argmax bin."""
        probs = _softmax_1d(logits)
        argmax = int(np.argmax(probs))
        lo = max(argmax - 1, 0)
        hi = min(argmax + 2, len(probs))
        window = probs[lo:hi]
        if window.sum() > 1e-12:
            window = window / window.sum()
        bins = np.arange(lo, hi, dtype=np.float64)
        return float(np.sum(bins * window))

    num_bins = axis_logits.shape[-1]  # 112
    cx_bin = _decode_axis(np.asarray(axis_logits[0], dtype=np.float64))
    cy_bin = _decode_axis(np.asarray(axis_logits[1], dtype=np.float64))
    tx_bin = _decode_axis(np.asarray(axis_logits[2], dtype=np.float64))
    ty_bin = _decode_axis(np.asarray(axis_logits[3], dtype=np.float64))

    # Convert bin index to 224-space: coord_px = bin * 223 / 111
    bin_to_px = 223.0 / (num_bins - 1)
    cx = cx_bin * bin_to_px
    cy = cy_bin * bin_to_px
    tx = tx_bin * bin_to_px
    ty = ty_bin * bin_to_px

    predicted_angle = angle_degrees_from_center_to_tip(cx, cy, tx, ty)
    predicted_temp = predict_temperature_from_candidate(predicted_angle, calibration_candidate)

    return dataclasses.replace(
        decoded,
        predicted_center_x_224=cx,
        predicted_center_y_224=cy,
        predicted_tip_x_224=tx,
        predicted_tip_y_224=ty,
        predicted_angle_degrees=float(predicted_angle),
        predicted_temperature_c_calibrated=float(predicted_temp),
        absolute_error_c_calibrated=float(abs(predicted_temp - true_temperature_c)),
    )


def _softmax_1d(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for 1D array."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()
