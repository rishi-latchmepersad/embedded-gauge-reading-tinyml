"""Guardrails for geometry heatmap predictions.

This module keeps the post-processing logic pure and testable.  The goal is not
to "fix" a bad prediction after the fact, but to reject or clamp obviously
unsafe geometry reads before they reach a board-style consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

import numpy as np

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    circular_angle_error_degrees,
)
from embedded_gauge_reading_tinyml.geometry_crop_dataset import SourceGeometryExample
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import HeatmapSample
from embedded_gauge_reading_tinyml.geometry_temperature_calibration import (
    CalibrationCandidate,
    DEFAULT_COLD_ANGLE_DEGREES,
    DEFAULT_MAXIMUM_CELSIUS,
    DEFAULT_MINIMUM_CELSIUS,
    DEFAULT_SWEEP_DEGREES,
    predict_temperature_from_candidate,
    unwrap_angle_from_cold,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import decode_heatmap_point


GeometryGuardrailStatus = Literal["accepted", "rejected", "clamped"]
GeometryDecodeMethod = Literal[
    "softargmax",
    "argmax",
    "local_window_softargmax",
    "peak_weighted_centroid",
]


@dataclass(frozen=True)
class GeometryGuardrailThresholds:
    """Thresholds used to accept, reject, or clamp one prediction."""

    center_peak_min: float = 0.40
    tip_peak_min: float = 0.40
    confidence_min: float = 0.40
    max_heatmap_entropy: float = 1.00
    max_heatmap_spread_px: float = 30.0
    center_tip_distance_ratio_min: float = 0.50
    center_tip_distance_ratio_max: float = 1.50
    edge_margin_px: float = 4.0
    temperature_physical_margin_c: float = 5.0
    clamp_temperature_to_physical_range: bool = True
    minimum_celsius: float = DEFAULT_MINIMUM_CELSIUS
    maximum_celsius: float = DEFAULT_MAXIMUM_CELSIUS
    cold_angle_degrees: float = DEFAULT_COLD_ANGLE_DEGREES
    sweep_degrees: float = DEFAULT_SWEEP_DEGREES


@dataclass(frozen=True)
class GeometryDecodedPrediction:
    """Decoded geometry prediction plus the source metadata needed for guards."""

    image_path: str
    split: str
    true_temperature_c: float
    true_angle_degrees: float
    true_center_x_224: float
    true_center_y_224: float
    true_tip_x_224: float
    true_tip_y_224: float
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    crop_width: int
    crop_height: int
    dial_radius_source: float
    jitter_shift_x: int
    jitter_shift_y: int
    jitter_scale: float
    jitter_aspect: float
    center_x_norm: float
    center_y_norm: float
    tip_x_norm: float
    tip_y_norm: float
    predicted_center_x_224: float
    predicted_center_y_224: float
    predicted_tip_x_224: float
    predicted_tip_y_224: float
    predicted_center_x_224_argmax: float
    predicted_center_y_224_argmax: float
    predicted_tip_x_224_argmax: float
    predicted_tip_y_224_argmax: float
    predicted_angle_degrees: float
    predicted_angle_degrees_argmax: float
    predicted_temperature_c_current_mapping: float
    predicted_temperature_c_current_mapping_argmax: float
    predicted_temperature_c_calibrated: float
    predicted_temperature_c_calibrated_argmax: float
    absolute_error_c_current_mapping: float
    absolute_error_c_current_mapping_argmax: float
    absolute_error_c_calibrated: float
    absolute_error_c_calibrated_argmax: float
    center_heatmap_peak_value: float
    tip_heatmap_peak_value: float
    center_heatmap_mean_value: float
    tip_heatmap_mean_value: float
    confidence: float
    center_heatmap: np.ndarray = field(repr=False, compare=False)
    tip_heatmap: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class GeometryQualityFeatures:
    """Derived quality signals used by the guardrails and reports."""

    predicted_center_x_norm: float
    predicted_center_y_norm: float
    predicted_tip_x_norm: float
    predicted_tip_y_norm: float
    center_normalized_in_bounds: bool
    tip_normalized_in_bounds: bool
    center_edge_margin_px: float
    tip_edge_margin_px: float
    min_edge_margin_px: float
    predicted_center_tip_distance_px: float
    true_center_tip_distance_px: float
    expected_center_tip_distance_px: float
    center_tip_distance_ratio: float
    center_heatmap_entropy: float
    tip_heatmap_entropy: float
    center_heatmap_spread_px: float
    tip_heatmap_spread_px: float
    angle_unwrapped_from_cold_degrees: float
    angle_within_valid_sweep: bool
    current_temperature_within_physical_range: bool
    calibrated_temperature_within_physical_range: bool
    calibrated_temperature_outside_physical_range: bool
    center_heatmap_peak_value: float
    tip_heatmap_peak_value: float
    confidence: float


@dataclass(frozen=True)
class GeometryGuardrailResult:
    """Final guarded reading returned to downstream consumers."""

    status: GeometryGuardrailStatus
    temperature_c: float
    raw_temperature_c: float
    rejection_reasons: tuple[str, ...]
    quality_features: GeometryQualityFeatures
    prediction: GeometryDecodedPrediction = field(repr=False, compare=False)


def _as_heatmap_array(heatmap: np.ndarray) -> np.ndarray:
    """Coerce one predicted heatmap to a dense 2D float array."""

    array = np.asarray(heatmap, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap after squeezing, got shape {array.shape!r}")
    return array


def _heatmap_entropy_and_spread(heatmap: np.ndarray) -> tuple[float, float]:
    """Return normalized entropy and spatial spread for one heatmap."""

    array = _as_heatmap_array(heatmap)
    total = float(np.sum(array))
    if total <= 0.0:
        return (math.nan, math.nan)

    probabilities = array / total
    positive_probabilities = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(positive_probabilities * np.log(positive_probabilities)))
    entropy_norm = entropy / float(np.log(probabilities.size))

    y_coords, x_coords = np.meshgrid(
        np.arange(array.shape[0], dtype=np.float32),
        np.arange(array.shape[1], dtype=np.float32),
        indexing="ij",
    )
    mean_x = float(np.sum(probabilities * x_coords))
    mean_y = float(np.sum(probabilities * y_coords))
    spread_px = float(np.sqrt(np.sum(probabilities * ((x_coords - mean_x) ** 2 + (y_coords - mean_y) ** 2))))
    return (entropy_norm, spread_px)


def _min_edge_margin_px(x: float, y: float, *, image_size: int = 224) -> float:
    """Compute the minimum pixel margin to any image edge for one point."""

    max_index = float(image_size - 1)
    return float(min(x, y, max_index - x, max_index - y))


def decode_heatmap_geometry_prediction(
    sample: HeatmapSample,
    predicted_center_heatmap: np.ndarray,
    predicted_tip_heatmap: np.ndarray,
    confidence: float,
    calibration_candidate: CalibrationCandidate,
    *,
    decode_method: GeometryDecodeMethod = "softargmax",
    window_size: int = 3,
) -> GeometryDecodedPrediction:
    """Decode one prediction bundle into geometry coordinates and temperatures."""

    center_heatmap = _as_heatmap_array(predicted_center_heatmap)
    tip_heatmap = _as_heatmap_array(predicted_tip_heatmap)

    center_row, center_col = decode_heatmap_point(center_heatmap, method=decode_method, window_size=window_size)
    tip_row, tip_col = decode_heatmap_point(tip_heatmap, method=decode_method, window_size=window_size)
    center_row_argmax, center_col_argmax = decode_heatmap_point(center_heatmap, method="argmax")
    tip_row_argmax, tip_col_argmax = decode_heatmap_point(tip_heatmap, method="argmax")
    predicted_center_x_224 = float((float(center_col) * 223.0) / float(center_heatmap.shape[1] - 1))
    predicted_center_y_224 = float((float(center_row) * 223.0) / float(center_heatmap.shape[0] - 1))
    predicted_tip_x_224 = float((float(tip_col) * 223.0) / float(tip_heatmap.shape[1] - 1))
    predicted_tip_y_224 = float((float(tip_row) * 223.0) / float(tip_heatmap.shape[0] - 1))
    predicted_center_x_224_argmax = float((float(center_col_argmax) * 223.0) / float(center_heatmap.shape[1] - 1))
    predicted_center_y_224_argmax = float((float(center_row_argmax) * 223.0) / float(center_heatmap.shape[0] - 1))
    predicted_tip_x_224_argmax = float((float(tip_col_argmax) * 223.0) / float(tip_heatmap.shape[1] - 1))
    predicted_tip_y_224_argmax = float((float(tip_row_argmax) * 223.0) / float(tip_heatmap.shape[0] - 1))

    true_center_x = float(sample.metadata["center_x_224"])
    true_center_y = float(sample.metadata["center_y_224"])
    true_tip_x = float(sample.metadata["tip_x_224"])
    true_tip_y = float(sample.metadata["tip_y_224"])

    predicted_angle_degrees = angle_degrees_from_center_to_tip(
        predicted_center_x_224,
        predicted_center_y_224,
        predicted_tip_x_224,
        predicted_tip_y_224,
    )
    predicted_angle_degrees_argmax = angle_degrees_from_center_to_tip(
        predicted_center_x_224_argmax,
        predicted_center_y_224_argmax,
        predicted_tip_x_224_argmax,
        predicted_tip_y_224_argmax,
    )
    true_angle_degrees = angle_degrees_from_center_to_tip(true_center_x, true_center_y, true_tip_x, true_tip_y)

    predicted_temperature_c_current_mapping = predict_temperature_from_candidate(
        predicted_angle_degrees,
        CalibrationCandidate(name="A_current_mapping", kind="current", params={}),
    )
    predicted_temperature_c_current_mapping_argmax = predict_temperature_from_candidate(
        predicted_angle_degrees_argmax,
        CalibrationCandidate(name="A_current_mapping", kind="current", params={}),
    )
    predicted_temperature_c_calibrated = predict_temperature_from_candidate(predicted_angle_degrees, calibration_candidate)
    predicted_temperature_c_calibrated_argmax = predict_temperature_from_candidate(
        predicted_angle_degrees_argmax,
        calibration_candidate,
    )

    true_temperature_c = float(sample.metadata["temperature_c"])

    return GeometryDecodedPrediction(
        image_path=str(sample.metadata["image_path"]),
        split=str(sample.metadata["split"]),
        true_temperature_c=true_temperature_c,
        true_angle_degrees=float(true_angle_degrees),
        true_center_x_224=true_center_x,
        true_center_y_224=true_center_y,
        true_tip_x_224=true_tip_x,
        true_tip_y_224=true_tip_y,
        crop_x1=int(sample.metadata["crop_x1"]),
        crop_y1=int(sample.metadata["crop_y1"]),
        crop_x2=int(sample.metadata["crop_x2"]),
        crop_y2=int(sample.metadata["crop_y2"]),
        crop_width=int(sample.metadata["crop_width"]),
        crop_height=int(sample.metadata["crop_height"]),
        dial_radius_source=float(sample.metadata["dial_radius_source"]),
        jitter_shift_x=int(sample.metadata["jitter_shift_x"]),
        jitter_shift_y=int(sample.metadata["jitter_shift_y"]),
        jitter_scale=float(sample.metadata["jitter_scale"]),
        jitter_aspect=float(sample.metadata["jitter_aspect"]),
        center_x_norm=float(sample.metadata["center_x_norm"]),
        center_y_norm=float(sample.metadata["center_y_norm"]),
        tip_x_norm=float(sample.metadata["tip_x_norm"]),
        tip_y_norm=float(sample.metadata["tip_y_norm"]),
        predicted_center_x_224=predicted_center_x_224,
        predicted_center_y_224=predicted_center_y_224,
        predicted_tip_x_224=predicted_tip_x_224,
        predicted_tip_y_224=predicted_tip_y_224,
        predicted_center_x_224_argmax=predicted_center_x_224_argmax,
        predicted_center_y_224_argmax=predicted_center_y_224_argmax,
        predicted_tip_x_224_argmax=predicted_tip_x_224_argmax,
        predicted_tip_y_224_argmax=predicted_tip_y_224_argmax,
        predicted_angle_degrees=float(predicted_angle_degrees),
        predicted_angle_degrees_argmax=float(predicted_angle_degrees_argmax),
        predicted_temperature_c_current_mapping=float(predicted_temperature_c_current_mapping),
        predicted_temperature_c_current_mapping_argmax=float(predicted_temperature_c_current_mapping_argmax),
        predicted_temperature_c_calibrated=float(predicted_temperature_c_calibrated),
        predicted_temperature_c_calibrated_argmax=float(predicted_temperature_c_calibrated_argmax),
        absolute_error_c_current_mapping=float(abs(predicted_temperature_c_current_mapping - true_temperature_c)),
        absolute_error_c_current_mapping_argmax=float(abs(predicted_temperature_c_current_mapping_argmax - true_temperature_c)),
        absolute_error_c_calibrated=float(abs(predicted_temperature_c_calibrated - true_temperature_c)),
        absolute_error_c_calibrated_argmax=float(abs(predicted_temperature_c_calibrated_argmax - true_temperature_c)),
        center_heatmap_peak_value=float(np.max(center_heatmap)),
        tip_heatmap_peak_value=float(np.max(tip_heatmap)),
        center_heatmap_mean_value=float(np.mean(center_heatmap)),
        tip_heatmap_mean_value=float(np.mean(tip_heatmap)),
        confidence=float(confidence),
        center_heatmap=center_heatmap,
        tip_heatmap=tip_heatmap,
    )


def compute_geometry_quality_features(
    prediction: GeometryDecodedPrediction,
    *,
    thresholds: GeometryGuardrailThresholds | None = None,
) -> GeometryQualityFeatures:
    """Compute the guardrail features from a decoded prediction."""

    del thresholds
    predicted_center_x_norm = float(prediction.predicted_center_x_224 / 223.0)
    predicted_center_y_norm = float(prediction.predicted_center_y_224 / 223.0)
    predicted_tip_x_norm = float(prediction.predicted_tip_x_224 / 223.0)
    predicted_tip_y_norm = float(prediction.predicted_tip_y_224 / 223.0)

    center_normalized_in_bounds = 0.0 <= predicted_center_x_norm <= 1.0 and 0.0 <= predicted_center_y_norm <= 1.0
    tip_normalized_in_bounds = 0.0 <= predicted_tip_x_norm <= 1.0 and 0.0 <= predicted_tip_y_norm <= 1.0

    center_edge_margin_px = _min_edge_margin_px(prediction.predicted_center_x_224, prediction.predicted_center_y_224)
    tip_edge_margin_px = _min_edge_margin_px(prediction.predicted_tip_x_224, prediction.predicted_tip_y_224)
    min_edge_margin_px = float(min(center_edge_margin_px, tip_edge_margin_px))

    predicted_center_tip_distance_px = float(
        math.hypot(
            prediction.predicted_tip_x_224 - prediction.predicted_center_x_224,
            prediction.predicted_tip_y_224 - prediction.predicted_center_y_224,
        )
    )
    true_center_tip_distance_px = float(
        math.hypot(
            prediction.true_tip_x_224 - prediction.true_center_x_224,
            prediction.true_tip_y_224 - prediction.true_center_y_224,
        )
    )

    # The crop is resized to 224x224, so the source dial radius scales by the crop axes.
    x_scale = 224.0 / float(prediction.crop_width)
    y_scale = 224.0 / float(prediction.crop_height)
    angle_radians = math.radians(prediction.predicted_angle_degrees)
    expected_center_tip_distance_px = float(
        prediction.dial_radius_source * math.hypot(math.cos(angle_radians) * x_scale, math.sin(angle_radians) * y_scale)
    )
    center_tip_distance_ratio = (
        predicted_center_tip_distance_px / expected_center_tip_distance_px if expected_center_tip_distance_px > 0.0 else math.nan
    )

    center_heatmap_entropy, center_heatmap_spread_px = _heatmap_entropy_and_spread(prediction.center_heatmap)
    tip_heatmap_entropy, tip_heatmap_spread_px = _heatmap_entropy_and_spread(prediction.tip_heatmap)

    angle_unwrapped_from_cold_degrees = float(
        unwrap_angle_from_cold(
            prediction.predicted_angle_degrees,
            cold_angle_degrees=DEFAULT_COLD_ANGLE_DEGREES,
        )
    )
    angle_within_valid_sweep = 0.0 <= angle_unwrapped_from_cold_degrees <= DEFAULT_SWEEP_DEGREES

    current_temperature_within_physical_range = (
        DEFAULT_MINIMUM_CELSIUS <= prediction.predicted_temperature_c_current_mapping <= DEFAULT_MAXIMUM_CELSIUS
    )
    calibrated_temperature_within_physical_range = (
        DEFAULT_MINIMUM_CELSIUS <= prediction.predicted_temperature_c_calibrated <= DEFAULT_MAXIMUM_CELSIUS
    )

    return GeometryQualityFeatures(
        predicted_center_x_norm=predicted_center_x_norm,
        predicted_center_y_norm=predicted_center_y_norm,
        predicted_tip_x_norm=predicted_tip_x_norm,
        predicted_tip_y_norm=predicted_tip_y_norm,
        center_normalized_in_bounds=center_normalized_in_bounds,
        tip_normalized_in_bounds=tip_normalized_in_bounds,
        center_edge_margin_px=center_edge_margin_px,
        tip_edge_margin_px=tip_edge_margin_px,
        min_edge_margin_px=min_edge_margin_px,
        predicted_center_tip_distance_px=predicted_center_tip_distance_px,
        true_center_tip_distance_px=true_center_tip_distance_px,
        expected_center_tip_distance_px=expected_center_tip_distance_px,
        center_tip_distance_ratio=center_tip_distance_ratio,
        center_heatmap_entropy=float(center_heatmap_entropy),
        tip_heatmap_entropy=float(tip_heatmap_entropy),
        center_heatmap_spread_px=float(center_heatmap_spread_px),
        tip_heatmap_spread_px=float(tip_heatmap_spread_px),
        angle_unwrapped_from_cold_degrees=angle_unwrapped_from_cold_degrees,
        angle_within_valid_sweep=angle_within_valid_sweep,
        current_temperature_within_physical_range=current_temperature_within_physical_range,
        calibrated_temperature_within_physical_range=calibrated_temperature_within_physical_range,
        calibrated_temperature_outside_physical_range=not calibrated_temperature_within_physical_range,
        center_heatmap_peak_value=prediction.center_heatmap_peak_value,
        tip_heatmap_peak_value=prediction.tip_heatmap_peak_value,
        confidence=prediction.confidence,
    )


def apply_geometry_guardrails(
    prediction: GeometryDecodedPrediction,
    thresholds: GeometryGuardrailThresholds,
) -> GeometryGuardrailResult:
    """Apply the configured guardrails to one decoded prediction."""

    features = compute_geometry_quality_features(prediction, thresholds=thresholds)
    rejection_reasons: list[str] = []

    if not features.center_normalized_in_bounds:
        rejection_reasons.append("center_normalized_out_of_bounds")
    if not features.tip_normalized_in_bounds:
        rejection_reasons.append("tip_normalized_out_of_bounds")
    if features.min_edge_margin_px < thresholds.edge_margin_px:
        rejection_reasons.append("predicted_point_near_edge")
    if features.center_heatmap_peak_value < thresholds.center_peak_min:
        rejection_reasons.append("center_peak_too_low")
    if features.tip_heatmap_peak_value < thresholds.tip_peak_min:
        rejection_reasons.append("tip_peak_too_low")
    if features.confidence < thresholds.confidence_min:
        rejection_reasons.append("confidence_too_low")
    if features.center_heatmap_entropy > thresholds.max_heatmap_entropy:
        rejection_reasons.append("center_heatmap_too_diffuse")
    if features.tip_heatmap_entropy > thresholds.max_heatmap_entropy:
        rejection_reasons.append("tip_heatmap_too_diffuse")
    if features.center_heatmap_spread_px > thresholds.max_heatmap_spread_px:
        rejection_reasons.append("center_heatmap_too_spread_out")
    if features.tip_heatmap_spread_px > thresholds.max_heatmap_spread_px:
        rejection_reasons.append("tip_heatmap_too_spread_out")
    if not features.angle_within_valid_sweep:
        rejection_reasons.append("predicted_angle_outside_valid_sweep")
    if not (
        thresholds.minimum_celsius - thresholds.temperature_physical_margin_c
        <= prediction.predicted_temperature_c_calibrated
        <= thresholds.maximum_celsius + thresholds.temperature_physical_margin_c
    ):
        rejection_reasons.append("temperature_outside_physical_margin")
    if not (
        thresholds.center_tip_distance_ratio_min
        <= features.center_tip_distance_ratio
        <= thresholds.center_tip_distance_ratio_max
    ):
        rejection_reasons.append("center_tip_distance_ratio_implausible")

    if rejection_reasons:
        return GeometryGuardrailResult(
            status="rejected",
            temperature_c=math.nan,
            raw_temperature_c=prediction.predicted_temperature_c_calibrated,
            rejection_reasons=tuple(rejection_reasons),
            quality_features=features,
            prediction=prediction,
        )

    raw_temperature_c = prediction.predicted_temperature_c_calibrated
    if raw_temperature_c < thresholds.minimum_celsius or raw_temperature_c > thresholds.maximum_celsius:
        if not thresholds.clamp_temperature_to_physical_range:
            return GeometryGuardrailResult(
                status="rejected",
                temperature_c=math.nan,
                raw_temperature_c=raw_temperature_c,
                rejection_reasons=("temperature_outside_physical_range",),
                quality_features=features,
                prediction=prediction,
            )
        clamped_temperature_c = float(np.clip(raw_temperature_c, thresholds.minimum_celsius, thresholds.maximum_celsius))
        return GeometryGuardrailResult(
            status="clamped",
            temperature_c=clamped_temperature_c,
            raw_temperature_c=raw_temperature_c,
            rejection_reasons=("temperature_clamped_to_physical_range",),
            quality_features=features,
            prediction=prediction,
        )

    return GeometryGuardrailResult(
        status="accepted",
        temperature_c=raw_temperature_c,
        raw_temperature_c=raw_temperature_c,
        rejection_reasons=(),
        quality_features=features,
        prediction=prediction,
    )


def guarded_temperature_from_prediction(
    sample: HeatmapSample,
    predicted_center_heatmap: np.ndarray,
    predicted_tip_heatmap: np.ndarray,
    confidence: float,
    calibration_candidate: CalibrationCandidate,
    *,
    thresholds: GeometryGuardrailThresholds | None = None,
    decode_method: GeometryDecodeMethod = "softargmax",
    window_size: int = 3,
) -> GeometryGuardrailResult:
    """Decode one prediction bundle and apply the configured guardrails."""

    if thresholds is None:
        thresholds = GeometryGuardrailThresholds()
    decoded = decode_heatmap_geometry_prediction(
        sample,
        predicted_center_heatmap,
        predicted_tip_heatmap,
        confidence,
        calibration_candidate,
        decode_method=decode_method,
        window_size=window_size,
    )
    return apply_geometry_guardrails(decoded, thresholds)
