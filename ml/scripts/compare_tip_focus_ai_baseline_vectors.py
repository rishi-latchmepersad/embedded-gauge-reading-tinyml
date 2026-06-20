#!/usr/bin/env python3
"""Compare tip-focus AI and classical baseline outputs on raw board captures.

The script mirrors the live board path closely enough to answer two questions
for a given capture:
1. What does the SimCC model predict after the 224-space crop + shared mask?
2. What does the firmware-matched classical baseline predict on the same frame?

It writes one overlay PNG and one JSON report per capture into ``tmp/`` so we
can inspect the crop path, the masked model input, and the final needle vectors
side by side.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
from numpy.typing import NDArray

# Keep matplotlib and TensorFlow in a headless, reproducible configuration.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPO_ROOT: Final[Path] = PROJECT_ROOT.parent
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "tmp" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import tensorflow as tf

from embedded_gauge_reading_tinyml.baseline_classical_cv import run_firmware_baseline
from embedded_gauge_reading_tinyml.board_crop_compare import estimate_board_crop_from_rgb, resize_with_pad_rgb
from embedded_gauge_reading_tinyml.firmware_preprocessing import firmware_training_crop_box, load_capture_image
from embedded_gauge_reading_tinyml.gauge.processing import (
    GaugeSpec,
    load_gauge_specs,
    needle_value_from_angle_deg,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import dequantize_output_tensor
from embedded_gauge_reading_tinyml.inner_celsius_mask import apply_inner_celsius_mask

RGBImage = NDArray[np.uint8]
FloatImage = NDArray[np.float32]

DEFAULT_GAUGE_ID: Final[str] = "littlegood_home_temp_gauge_c"
DEFAULT_MODEL_PATH: Final[Path] = (
    REPO_ROOT
    / "ml"
    / "artifacts"
    / "deployment"
    / "simcc_gauge_v2_spatial_qat_sc128_int8"
    / "model_int8.tflite"
)
DEFAULT_OUTPUT_DIR: Final[Path] = REPO_ROOT / "tmp" / "tip_focus_ai_baseline_compare"

TIP_FOCUS_COLD_ANGLE_DEG: Final[float] = 135.0
TIP_FOCUS_SLOPE: Final[float] = 0.2963033
TIP_FOCUS_INTERCEPT: Final[float] = -30.0009
TIP_FOCUS_SIMCC_BINS: Final[int] = 112
CAPTURE_FRAME_WIDTH: Final[int] = 320
CAPTURE_FRAME_HEIGHT: Final[int] = 320

TIP_FOCUS_CONFIDENCE_FLOOR: Final[float] = 0.40
TIP_FOCUS_AXIS_PEAK_FLOOR: Final[float] = 0.06
TIP_FOCUS_AXIS_SPREAD_MAX_PX: Final[float] = 32.0
TIP_FOCUS_EDGE_MARGIN_MIN_PX: Final[float] = 4.0
TIP_FOCUS_DISTANCE_RATIO_MIN: Final[float] = 0.35
TIP_FOCUS_DISTANCE_RATIO_MAX: Final[float] = 1.40
TIP_FOCUS_TEMP_MIN_C: Final[float] = -35.0
TIP_FOCUS_TEMP_MAX_C: Final[float] = 55.0


@dataclass(frozen=True, slots=True)
class TipFocusTfliteContract:
    """Cached TFLite interpreter metadata for the SimCC tip-focus model."""

    interpreter: tf.lite.Interpreter
    input_detail: dict[str, Any]
    output_details: list[dict[str, Any]]
    semantic_output_indices: dict[str, int]
    input_width: int
    input_height: int
    input_dtype: str


@dataclass(frozen=True, slots=True)
class TipFocusAiPrediction:
    """One AI prediction plus the quality gates we care about."""

    decoded_ok: bool
    guardrail_ok: bool
    guardrail_reasons: tuple[str, ...]
    input_width: int
    input_height: int
    input_dtype: str
    center_x_px_224: float
    center_y_px_224: float
    tip_x_px_224: float
    tip_y_px_224: float
    center_x_px_raw: float
    center_y_px_raw: float
    tip_x_px_raw: float
    tip_y_px_raw: float
    angle_image_deg: float
    angle_board_deg: float
    temperature_spec_c: float
    temperature_board_c: float
    confidence: float
    center_peak: float
    tip_peak: float
    center_spread_px: float
    tip_spread_px: float
    center_tip_distance_px: float
    expected_center_tip_distance_px: float
    min_edge_margin_px: float


@dataclass(frozen=True, slots=True)
class TipFocusBaselinePrediction:
    """Firmware-matched classical baseline prediction for one capture."""

    valid: bool
    center_x_raw: float
    center_y_raw: float
    tip_x_raw: float
    tip_y_raw: float
    angle_image_deg: float
    angle_board_deg: float
    temperature_spec_c: float
    temperature_board_c: float
    confidence: float
    peak_value: float
    runner_up_value: float
    dial_radius_px: float


@dataclass(frozen=True, slots=True)
class TipFocusComparisonReport:
    """Everything written for one capture comparison."""

    capture_path: Path
    source_kind: str
    source_size: tuple[int, int]
    crop_method: str
    crop_box_xyxy: tuple[float, float, float, float]
    truth_temperature_c: float | None
    ai: TipFocusAiPrediction
    baseline: TipFocusBaselinePrediction
    overlay_path: Path
    report_path: Path


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the comparison run."""

    parser = argparse.ArgumentParser(
        description="Compare the tip-focus AI and classical baseline on board captures."
    )
    parser.add_argument(
        "captures",
        nargs="+",
        type=Path,
        help="One or more board captures to replay.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the packaged SimCC TFLite model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where overlays and JSON reports will be written.",
    )
    parser.add_argument(
        "--gauge-id",
        type=str,
        default=DEFAULT_GAUGE_ID,
        help="Gauge spec to use for temperature decoding.",
    )
    parser.add_argument(
        "--truth-temperature-c",
        type=float,
        default=None,
        help="Optional known temperature for the capture set.",
    )
    return parser.parse_args()


def _match_output_name(names: Sequence[str], token: str) -> str | None:
    """Return the first output tensor name containing the requested token."""

    token_lower = token.lower()
    exact_matches = [name for name in names if name.lower() == token_lower]
    if exact_matches:
        return exact_matches[0]
    partial_matches = [name for name in names if token_lower in name.lower()]
    if not partial_matches:
        return None
    partial_matches.sort(key=lambda name: (len(name), name))
    return partial_matches[0]


def _detail_map(details: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map tensor names to their detail dictionaries."""

    return {str(detail["name"]): dict(detail) for detail in details}


def _resolve_semantic_output_indices(
    output_details: list[dict[str, Any]],
) -> dict[str, int]:
    """Resolve the SimCC tensor order, falling back to the contract's output order."""

    output_names = [str(detail.get("name", "")) for detail in output_details]
    semantic_to_name = {
        "cx": _match_output_name(output_names, "center_x_simcc"),
        "cy": _match_output_name(output_names, "center_y_simcc"),
        "tx": _match_output_name(output_names, "tip_x_simcc"),
        "ty": _match_output_name(output_names, "tip_y_simcc"),
        "conf": _match_output_name(output_names, "confidence"),
    }
    if all(name is not None for name in semantic_to_name.values()):
        return {
            semantic: output_names.index(name)
            for semantic, name in semantic_to_name.items()
            if name is not None
        }

    # The deployed pack exposes five outputs: confidence + four 112-bin heads.
    simcc_indices = [
        index
        for index, detail in enumerate(output_details)
        if int(np.asarray(detail["shape"], dtype=np.int64).reshape(-1)[-1]) == TIP_FOCUS_SIMCC_BINS
    ]
    conf_indices = [
        index
        for index, detail in enumerate(output_details)
        if int(np.asarray(detail["shape"], dtype=np.int64).reshape(-1)[-1]) == 1
    ]
    if len(simcc_indices) < 4 or not conf_indices:
        raise RuntimeError("Could not resolve the SimCC tip-focus output order.")

    return {
        "conf": conf_indices[0],
        "cx": simcc_indices[0],
        "cy": simcc_indices[1],
        "tx": simcc_indices[2],
        "ty": simcc_indices[3],
    }


def _load_tip_focus_tflite_contract(model_path: Path) -> TipFocusTfliteContract:
    """Load the packaged SimCC model and cache the tensor metadata."""

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = list(interpreter.get_output_details())
    semantic_output_indices = _resolve_semantic_output_indices(output_details)

    input_shape = np.asarray(input_detail["shape"], dtype=np.int64).reshape(-1)
    if input_shape.size < 4:
        raise ValueError(f"Unexpected SimCC input shape: {tuple(int(v) for v in input_shape)}")
    input_height = int(input_shape[1])
    input_width = int(input_shape[2])
    input_dtype = str(np.dtype(input_detail["dtype"]))

    return TipFocusTfliteContract(
        interpreter=interpreter,
        input_detail=dict(input_detail),
        output_details=output_details,
        semantic_output_indices=semantic_output_indices,
        input_width=input_width,
        input_height=input_height,
        input_dtype=input_dtype,
    )


def _soft_argmax_1d(axis: NDArray[np.float32]) -> float:
    """Decode one 1-D probability vector into a normalized coordinate."""

    weights = np.asarray(axis, dtype=np.float32).reshape(-1)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("SimCC axis had no positive weight.")

    bins = np.arange(weights.size, dtype=np.float32)
    weighted_sum = float(np.sum(weights * bins))
    return weighted_sum / total_weight / float(weights.size - 1)


def _decode_axis_stats(axis: NDArray[np.float32], *, input_size: int) -> tuple[float, float, float]:
    """Return normalized coordinate, peak value, and spread in 224-space pixels."""

    weights = np.asarray(axis, dtype=np.float32).reshape(-1)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("SimCC axis had no positive weight.")

    bins = np.arange(weights.size, dtype=np.float32)
    weighted_sum = float(np.sum(weights * bins))
    mean_bin = weighted_sum / total_weight
    variance = float(np.sum(weights * ((bins - mean_bin) ** 2)) / total_weight)
    peak_value = float(np.max(weights))
    coord_norm = weighted_sum / total_weight / float(weights.size - 1)
    bin_to_px_scale = float(input_size - 1) / float(weights.size - 1)
    spread_px = math.sqrt(max(variance, 0.0)) * bin_to_px_scale
    return coord_norm, peak_value, spread_px


def _board_temperature_from_angle_deg(angle_deg: float) -> float:
    """Apply the same linear angle-to-temperature fit used by the live board."""

    angle_delta = (angle_deg - TIP_FOCUS_COLD_ANGLE_DEG) % 360.0
    return TIP_FOCUS_SLOPE * angle_delta + TIP_FOCUS_INTERCEPT


def _angle_from_points(
    center_x: float,
    center_y: float,
    tip_x: float,
    tip_y: float,
) -> tuple[float, float]:
    """Return the image-space and board-space needle angles in degrees."""

    dx = tip_x - center_x
    dy = tip_y - center_y
    image_angle = math.degrees(math.atan2(dy, dx)) % 360.0
    board_angle = math.degrees(math.atan2(-dy, dx)) % 360.0
    return image_angle, board_angle


def _map_canvas_point_to_source(
    x_canvas: float,
    y_canvas: float,
    crop_box_xyxy: tuple[float, float, float, float],
    image_size: int,
) -> tuple[float, float]:
    """Invert the resize-with-pad geometry used by the firmware and replay code."""

    x_min, y_min, x_max, y_max = crop_box_xyxy
    crop_width = max(1.0, float(round(x_max - x_min)))
    crop_height = max(1.0, float(round(y_max - y_min)))
    scale = min(float(image_size) / crop_width, float(image_size) / crop_height)
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))
    offset_x = max(0, (image_size - resized_width) // 2)
    offset_y = max(0, (image_size - resized_height) // 2)
    source_x = x_min + (x_canvas - float(offset_x)) / scale
    source_y = y_min + (y_canvas - float(offset_y)) / scale
    return source_x, source_y


def _predict_tip_focus(
    contract: TipFocusTfliteContract,
    crop_image_224: FloatImage,
    *,
    gauge_spec: GaugeSpec,
    crop_box_xyxy: tuple[float, float, float, float],
) -> TipFocusAiPrediction:
    """Run the SimCC model on one crop and decode the needle geometry."""

    if crop_image_224.shape[0] != contract.input_height or crop_image_224.shape[1] != contract.input_width:
        raise ValueError(
            f"Unexpected crop shape {crop_image_224.shape}, expected "
            f"{contract.input_height}x{contract.input_width}."
        )

    # The caller passes the final masked crop, matching the shared training
    # replay path: resize -> normalize -> shared inner-Celsius mask.
    model_input = np.asarray(crop_image_224, dtype=np.float32)
    batch = np.expand_dims(model_input, axis=0)

    contract.interpreter.set_tensor(int(contract.input_detail["index"]), batch)
    contract.interpreter.invoke()
    cx_detail = contract.output_details[contract.semantic_output_indices["cx"]]
    cy_detail = contract.output_details[contract.semantic_output_indices["cy"]]
    tx_detail = contract.output_details[contract.semantic_output_indices["tx"]]
    ty_detail = contract.output_details[contract.semantic_output_indices["ty"]]
    conf_detail = contract.output_details[contract.semantic_output_indices["conf"]]

    center_x_axis = dequantize_output_tensor(
        contract.interpreter.get_tensor(int(cx_detail["index"])),
        cx_detail,
    )
    center_y_axis = dequantize_output_tensor(
        contract.interpreter.get_tensor(int(cy_detail["index"])),
        cy_detail,
    )
    tip_x_axis = dequantize_output_tensor(
        contract.interpreter.get_tensor(int(tx_detail["index"])),
        tx_detail,
    )
    tip_y_axis = dequantize_output_tensor(
        contract.interpreter.get_tensor(int(ty_detail["index"])),
        ty_detail,
    )
    confidence_tensor = dequantize_output_tensor(
        contract.interpreter.get_tensor(int(conf_detail["index"])),
        conf_detail,
    )

    decoded_ok = True
    guardrail_reasons: list[str] = []
    try:
        center_x_norm, center_x_peak, center_x_spread_px = _decode_axis_stats(
            center_x_axis, input_size=contract.input_width
        )
        center_y_norm, center_y_peak, center_y_spread_px = _decode_axis_stats(
            center_y_axis, input_size=contract.input_width
        )
        tip_x_norm, tip_x_peak, tip_x_spread_px = _decode_axis_stats(
            tip_x_axis, input_size=contract.input_width
        )
        tip_y_norm, tip_y_peak, tip_y_spread_px = _decode_axis_stats(
            tip_y_axis, input_size=contract.input_width
        )
    except ValueError:
        decoded_ok = False
        center_x_norm = center_y_norm = tip_x_norm = tip_y_norm = float("nan")
        center_x_peak = center_y_peak = tip_x_peak = tip_y_peak = float("nan")
        center_x_spread_px = center_y_spread_px = tip_x_spread_px = tip_y_spread_px = float("nan")

    if not decoded_ok:
        guardrail_reasons.append("decode_failed")

    confidence = float(np.asarray(confidence_tensor, dtype=np.float32).reshape(-1)[0])
    center_peak = min(center_x_peak, center_y_peak)
    tip_peak = min(tip_x_peak, tip_y_peak)
    center_x_px_224 = center_x_norm * float(contract.input_width - 1)
    center_y_px_224 = center_y_norm * float(contract.input_height - 1)
    tip_x_px_224 = tip_x_norm * float(contract.input_width - 1)
    tip_y_px_224 = tip_y_norm * float(contract.input_height - 1)
    center_x_px_raw, center_y_px_raw = _map_canvas_point_to_source(
        center_x_px_224, center_y_px_224, crop_box_xyxy, contract.input_width
    )
    tip_x_px_raw, tip_y_px_raw = _map_canvas_point_to_source(
        tip_x_px_224, tip_y_px_224, crop_box_xyxy, contract.input_width
    )
    angle_image_deg, angle_board_deg = _angle_from_points(
        center_x_px_224,
        center_y_px_224,
        tip_x_px_224,
        tip_y_px_224,
    )
    temperature_spec_c = needle_value_from_angle_deg(
        angle_image_deg,
        gauge_spec,
        strict=False,
    )
    temperature_board_c = _board_temperature_from_angle_deg(angle_board_deg)

    dx_px = tip_x_px_224 - center_x_px_224
    dy_px = tip_y_px_224 - center_y_px_224
    center_tip_distance_px = math.hypot(dx_px, dy_px)
    expected_center_tip_distance_px = (
        gauge_spec.inner_dial_radius_frame_ratio * float(contract.input_width - 1)
    )
    center_edge_margin_px = min(
        center_x_px_224,
        center_y_px_224,
        float(contract.input_width - 1) - center_x_px_224,
        float(contract.input_height - 1) - center_y_px_224,
    )
    tip_edge_margin_px = min(
        tip_x_px_224,
        tip_y_px_224,
        float(contract.input_width - 1) - tip_x_px_224,
        float(contract.input_height - 1) - tip_y_px_224,
    )
    min_edge_margin_px = min(center_edge_margin_px, tip_edge_margin_px)
    center_tip_distance_ratio = (
        center_tip_distance_px / expected_center_tip_distance_px
        if expected_center_tip_distance_px > 0.0
        else float("nan")
    )
    angle_delta_deg = (angle_board_deg - TIP_FOCUS_COLD_ANGLE_DEG) % 360.0
    temperature_finite_ok = math.isfinite(temperature_board_c)
    temperature_range_ok = (
        TIP_FOCUS_TEMP_MIN_C <= temperature_board_c <= TIP_FOCUS_TEMP_MAX_C
    )
    confidence_ok = confidence >= TIP_FOCUS_CONFIDENCE_FLOOR
    center_peak_ok = center_peak >= TIP_FOCUS_AXIS_PEAK_FLOOR
    tip_peak_ok = tip_peak >= TIP_FOCUS_AXIS_PEAK_FLOOR
    center_spread_ok = max(center_x_spread_px, center_y_spread_px) <= TIP_FOCUS_AXIS_SPREAD_MAX_PX
    tip_spread_ok = max(tip_x_spread_px, tip_y_spread_px) <= TIP_FOCUS_AXIS_SPREAD_MAX_PX
    edge_margin_ok = min_edge_margin_px >= TIP_FOCUS_EDGE_MARGIN_MIN_PX
    distance_ok = (
        TIP_FOCUS_DISTANCE_RATIO_MIN
        <= center_tip_distance_ratio
        <= TIP_FOCUS_DISTANCE_RATIO_MAX
    )
    angle_ok = angle_delta_deg <= 270.0

    if not center_peak_ok:
        guardrail_reasons.append("center_peak_low")
    if not tip_peak_ok:
        guardrail_reasons.append("tip_peak_low")
    if not confidence_ok:
        guardrail_reasons.append("confidence_low")
    if not center_spread_ok:
        guardrail_reasons.append("center_spread_high")
    if not tip_spread_ok:
        guardrail_reasons.append("tip_spread_high")
    if not edge_margin_ok:
        guardrail_reasons.append("edge_margin_low")
    if not distance_ok:
        guardrail_reasons.append("distance_ratio_bad")
    if not angle_ok:
        guardrail_reasons.append("angle_out_of_range")
    if not temperature_finite_ok:
        guardrail_reasons.append("temperature_non_finite")
    if not temperature_range_ok:
        guardrail_reasons.append("temperature_out_of_range")

    return TipFocusAiPrediction(
        decoded_ok=decoded_ok,
        guardrail_ok=decoded_ok and not guardrail_reasons,
        guardrail_reasons=tuple(guardrail_reasons),
        input_width=contract.input_width,
        input_height=contract.input_height,
        input_dtype=contract.input_dtype,
        center_x_px_224=float(center_x_px_224),
        center_y_px_224=float(center_y_px_224),
        tip_x_px_224=float(tip_x_px_224),
        tip_y_px_224=float(tip_y_px_224),
        center_x_px_raw=float(center_x_px_raw),
        center_y_px_raw=float(center_y_px_raw),
        tip_x_px_raw=float(tip_x_px_raw),
        tip_y_px_raw=float(tip_y_px_raw),
        angle_image_deg=float(angle_image_deg),
        angle_board_deg=float(angle_board_deg),
        temperature_spec_c=float(temperature_spec_c),
        temperature_board_c=float(temperature_board_c),
        confidence=float(confidence),
        center_peak=float(center_peak),
        tip_peak=float(tip_peak),
        center_spread_px=float(max(center_x_spread_px, center_y_spread_px)),
        tip_spread_px=float(max(tip_x_spread_px, tip_y_spread_px)),
        center_tip_distance_px=float(center_tip_distance_px),
        expected_center_tip_distance_px=float(expected_center_tip_distance_px),
        min_edge_margin_px=float(min_edge_margin_px),
    )


def _predict_baseline(
    image_bgr: RGBImage,
    gauge_spec: GaugeSpec,
) -> TipFocusBaselinePrediction:
    """Run the firmware-matched classical baseline on the full source image."""

    center_x_raw, center_y_raw, dial_radius_px, detection = run_firmware_baseline(
        image_bgr,
        gauge_spec,
    )
    if detection is None:
        return TipFocusBaselinePrediction(
            valid=False,
            center_x_raw=float(center_x_raw),
            center_y_raw=float(center_y_raw),
            tip_x_raw=float("nan"),
            tip_y_raw=float("nan"),
            angle_image_deg=float("nan"),
            angle_board_deg=float("nan"),
            temperature_spec_c=float("nan"),
            temperature_board_c=float("nan"),
            confidence=float("nan"),
            peak_value=float("nan"),
            runner_up_value=float("nan"),
            dial_radius_px=float(dial_radius_px),
        )

    tip_x_raw = float(center_x_raw) + float(detection.unit_dx) * float(dial_radius_px) * 0.78
    tip_y_raw = float(center_y_raw) + float(detection.unit_dy) * float(dial_radius_px) * 0.78
    angle_image_deg, angle_board_deg = _angle_from_points(
        float(center_x_raw),
        float(center_y_raw),
        tip_x_raw,
        tip_y_raw,
    )
    return TipFocusBaselinePrediction(
        valid=True,
        center_x_raw=float(center_x_raw),
        center_y_raw=float(center_y_raw),
        tip_x_raw=float(tip_x_raw),
        tip_y_raw=float(tip_y_raw),
        angle_image_deg=float(angle_image_deg),
        angle_board_deg=float(angle_board_deg),
        temperature_spec_c=float(
            needle_value_from_angle_deg(angle_image_deg, gauge_spec, strict=False)
        ),
        temperature_board_c=float(_board_temperature_from_angle_deg(angle_board_deg)),
        confidence=float(detection.confidence),
        peak_value=float(detection.peak_value),
        runner_up_value=float(detection.runner_up_value),
        dial_radius_px=float(dial_radius_px),
    )


def _render_overlay(
    *,
    source_image: RGBImage,
    crop_image_224: FloatImage,
    crop_box_xyxy: tuple[float, float, float, float],
    report: TipFocusComparisonReport,
    output_path: Path,
) -> None:
    """Render a compact side-by-side overlay for one capture."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 8), dpi=150)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.45, 1.45, 0.95))

    ax_source = fig.add_subplot(grid[0, 0])
    ax_crop = fig.add_subplot(grid[0, 1])
    ax_text = fig.add_subplot(grid[0, 2])

    source_panel = np.asarray(source_image, dtype=np.float32) / 255.0
    ax_source.imshow(source_panel)
    ax_source.set_title("Raw capture")
    ax_source.axis("off")

    x_min, y_min, x_max, y_max = crop_box_xyxy
    crop_rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2.0,
        edgecolor="lime" if "fallback" not in report.crop_method else "gold",
        facecolor="none",
    )
    ax_source.add_patch(crop_rect)

    if report.baseline.valid:
        ax_source.plot(
            [report.baseline.center_x_raw, report.baseline.tip_x_raw],
            [report.baseline.center_y_raw, report.baseline.tip_y_raw],
            color="orange",
            linewidth=2.5,
            label="baseline",
        )
        ax_source.scatter(
            [report.baseline.center_x_raw, report.baseline.tip_x_raw],
            [report.baseline.center_y_raw, report.baseline.tip_y_raw],
            c=["white", "orange"],
            s=[55, 55],
            marker="o",
            edgecolors="black",
            linewidths=0.8,
        )
    else:
        ax_source.scatter(
            [report.baseline.center_x_raw],
            [report.baseline.center_y_raw],
            c="white",
            s=55,
            marker="o",
            edgecolors="black",
            linewidths=0.8,
            label="baseline fallback",
        )

    if report.ai.decoded_ok:
        ai_color = "cyan" if report.ai.guardrail_ok else "tomato"
        ax_source.plot(
            [report.ai.center_x_px_raw, report.ai.tip_x_px_raw],
            [report.ai.center_y_px_raw, report.ai.tip_y_px_raw],
            color=ai_color,
            linewidth=2.5,
            label="AI",
        )
        ax_source.scatter(
            [report.ai.center_x_px_raw, report.ai.tip_x_px_raw],
            [report.ai.center_y_px_raw, report.ai.tip_y_px_raw],
            c=["cyan", ai_color],
            s=[55, 55],
            marker="x",
            linewidths=2.0,
        )

    ax_source.legend(loc="lower right", fontsize=8)

    ax_crop.imshow(np.clip(crop_image_224, 0.0, 1.0))
    ax_crop.set_title(f"Model input ({report.crop_method})")
    ax_crop.axis("off")

    keep_circle = patches.Circle(
        (112.0, 100.0),
        62.0,
        linewidth=1.5,
        edgecolor="white",
        facecolor="none",
        linestyle="--",
        alpha=0.9,
    )
    ax_crop.add_patch(keep_circle)
    ax_crop.axhline(150.0, color="white", linestyle=":", linewidth=1.2, alpha=0.8)

    if report.ai.decoded_ok:
        ai_color = "cyan" if report.ai.guardrail_ok else "tomato"
        ax_crop.plot(
            [report.ai.center_x_px_224, report.ai.tip_x_px_224],
            [report.ai.center_y_px_224, report.ai.tip_y_px_224],
            color=ai_color,
            linewidth=2.5,
        )
        ax_crop.scatter(
            [report.ai.center_x_px_224, report.ai.tip_x_px_224],
            [report.ai.center_y_px_224, report.ai.tip_y_px_224],
            c=["cyan", ai_color],
            s=[50, 50],
            marker="x",
            linewidths=2.0,
        )

    ax_text.axis("off")
    truth_text = (
        f"{report.truth_temperature_c:.2f} C" if report.truth_temperature_c is not None else "n/a"
    )
    ai_status = "ok" if report.ai.guardrail_ok else "reject"
    baseline_status = "ok" if report.baseline.valid else "reject"
    text_lines = [
        f"file: {report.capture_path.name}",
        f"source kind: {report.source_kind}",
        f"source size: {report.source_size[0]}x{report.source_size[1]}",
        f"crop method: {report.crop_method}",
        (
            "crop box: "
            f"x0={report.crop_box_xyxy[0]:.1f} y0={report.crop_box_xyxy[1]:.1f} "
            f"x1={report.crop_box_xyxy[2]:.1f} y1={report.crop_box_xyxy[3]:.1f}"
        ),
        "",
        f"truth temp: {truth_text}",
        f"AI decode: {ai_status}",
        f"AI confidence: {report.ai.confidence:.4f}",
        f"AI spec temp: {report.ai.temperature_spec_c:.2f} C",
        f"AI board temp: {report.ai.temperature_board_c:.2f} C",
        f"AI image angle: {report.ai.angle_image_deg:.2f} deg",
        f"AI board angle: {report.ai.angle_board_deg:.2f} deg",
        f"AI center: ({report.ai.center_x_px_224:.1f}, {report.ai.center_y_px_224:.1f})",
        f"AI tip: ({report.ai.tip_x_px_224:.1f}, {report.ai.tip_y_px_224:.1f})",
        f"AI spread px: {max(report.ai.center_spread_px, report.ai.tip_spread_px):.2f}",
        f"AI edge margin px: {report.ai.min_edge_margin_px:.2f}",
        f"AI distance ratio: {report.ai.center_tip_distance_px / report.ai.expected_center_tip_distance_px:.3f}",
        (
            "AI guardrails: "
            + ("ok" if report.ai.guardrail_ok else ", ".join(report.ai.guardrail_reasons) or "reject")
        ),
        "",
        f"baseline: {baseline_status}",
        f"baseline confidence: {report.baseline.confidence:.4f}",
        f"baseline spec temp: {report.baseline.temperature_spec_c:.2f} C",
        f"baseline board temp: {report.baseline.temperature_board_c:.2f} C",
        f"baseline image angle: {report.baseline.angle_image_deg:.2f} deg",
        f"baseline board angle: {report.baseline.angle_board_deg:.2f} deg",
        f"baseline center: ({report.baseline.center_x_raw:.1f}, {report.baseline.center_y_raw:.1f})",
        f"baseline tip: ({report.baseline.tip_x_raw:.1f}, {report.baseline.tip_y_raw:.1f})",
        "",
        f"AI - truth (spec): "
        f"{(report.ai.temperature_spec_c - report.truth_temperature_c):.2f} C"
        if report.truth_temperature_c is not None and math.isfinite(report.ai.temperature_spec_c)
        else "AI - truth (spec): n/a",
        f"baseline - truth (spec): "
        f"{(report.baseline.temperature_spec_c - report.truth_temperature_c):.2f} C"
        if report.truth_temperature_c is not None and math.isfinite(report.baseline.temperature_spec_c)
        else "baseline - truth (spec): n/a",
        f"AI - baseline (spec): "
        f"{(report.ai.temperature_spec_c - report.baseline.temperature_spec_c):.2f} C"
        if report.ai.decoded_ok and report.baseline.valid
        else "AI - baseline (spec): n/a",
    ]
    ax_text.text(
        0.0,
        1.0,
        "\n".join(text_lines),
        family="monospace",
        fontsize=9.0,
        va="top",
    )

    fig.suptitle(
        f"{report.capture_path.name} | AI {ai_status} | baseline {baseline_status}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _compare_capture(
    *,
    capture_path: Path,
    contract: TipFocusTfliteContract,
    gauge_spec: GaugeSpec,
    truth_temperature_c: float | None,
    output_dir: Path,
) -> TipFocusComparisonReport:
    """Run both pipelines on one capture and write the overlay artifacts."""

    source_image, source_kind = load_capture_image(
        capture_path,
        image_width=CAPTURE_FRAME_WIDTH,
        image_height=CAPTURE_FRAME_HEIGHT,
    )
    source_height, source_width = source_image.shape[:2]

    board_estimate = estimate_board_crop_from_rgb(source_image)
    if board_estimate is None:
        crop_box_xyxy = firmware_training_crop_box(source_width, source_height)
        crop_method = "fallback_training_crop"
    else:
        crop_box_xyxy = (
            float(board_estimate.crop_box.x_min),
            float(board_estimate.crop_box.y_min),
            float(board_estimate.crop_box.x_max),
            float(board_estimate.crop_box.y_max),
        )
        crop_method = f"board_bright_centroid({board_estimate.crop_box.bright_count})"

    crop_rgb = resize_with_pad_rgb(source_image, crop_box_xyxy, image_size=contract.input_width)
    crop_float = np.asarray(crop_rgb, dtype=np.float32) / 255.0
    masked_crop_float = apply_inner_celsius_mask(crop_float)

    ai_prediction = _predict_tip_focus(
        contract,
        masked_crop_float,
        gauge_spec=gauge_spec,
        crop_box_xyxy=crop_box_xyxy,
    )
    baseline_prediction = _predict_baseline(
        np.ascontiguousarray(source_image[:, :, ::-1]),
        gauge_spec,
    )

    capture_output_dir = output_dir / capture_path.stem
    overlay_path = capture_output_dir / "tip_focus_compare.png"
    report_path = capture_output_dir / "tip_focus_compare.json"
    report = TipFocusComparisonReport(
        capture_path=capture_path,
        source_kind=source_kind,
        source_size=(source_width, source_height),
        crop_method=crop_method,
        crop_box_xyxy=crop_box_xyxy,
        truth_temperature_c=truth_temperature_c,
        ai=ai_prediction,
        baseline=baseline_prediction,
        overlay_path=overlay_path,
        report_path=report_path,
    )

    _render_overlay(
        source_image=source_image,
        crop_image_224=masked_crop_float,
        crop_box_xyxy=crop_box_xyxy,
        report=report,
        output_path=overlay_path,
    )

    capture_output_dir.mkdir(parents=True, exist_ok=True)
    report_json = {
        "capture_path": str(report.capture_path),
        "source_kind": report.source_kind,
        "source_size": [int(report.source_size[0]), int(report.source_size[1])],
        "crop_method": report.crop_method,
        "crop_box_xyxy": [float(value) for value in report.crop_box_xyxy],
        "truth_temperature_c": report.truth_temperature_c,
        "overlay_path": str(report.overlay_path),
        "report_path": str(report.report_path),
        "ai": {
            "decoded_ok": report.ai.decoded_ok,
            "guardrail_ok": report.ai.guardrail_ok,
            "guardrail_reasons": list(report.ai.guardrail_reasons),
            "input_width": report.ai.input_width,
            "input_height": report.ai.input_height,
            "input_dtype": report.ai.input_dtype,
            "center_x_px_224": report.ai.center_x_px_224,
            "center_y_px_224": report.ai.center_y_px_224,
            "tip_x_px_224": report.ai.tip_x_px_224,
            "tip_y_px_224": report.ai.tip_y_px_224,
            "center_x_px_raw": report.ai.center_x_px_raw,
            "center_y_px_raw": report.ai.center_y_px_raw,
            "tip_x_px_raw": report.ai.tip_x_px_raw,
            "tip_y_px_raw": report.ai.tip_y_px_raw,
            "angle_image_deg": report.ai.angle_image_deg,
            "angle_board_deg": report.ai.angle_board_deg,
            "temperature_spec_c": report.ai.temperature_spec_c,
            "temperature_board_c": report.ai.temperature_board_c,
            "confidence": report.ai.confidence,
            "center_peak": report.ai.center_peak,
            "tip_peak": report.ai.tip_peak,
            "center_spread_px": report.ai.center_spread_px,
            "tip_spread_px": report.ai.tip_spread_px,
            "center_tip_distance_px": report.ai.center_tip_distance_px,
            "expected_center_tip_distance_px": report.ai.expected_center_tip_distance_px,
            "min_edge_margin_px": report.ai.min_edge_margin_px,
        },
        "baseline": {
            "valid": report.baseline.valid,
            "center_x_raw": report.baseline.center_x_raw,
            "center_y_raw": report.baseline.center_y_raw,
            "tip_x_raw": report.baseline.tip_x_raw,
            "tip_y_raw": report.baseline.tip_y_raw,
            "angle_image_deg": report.baseline.angle_image_deg,
            "angle_board_deg": report.baseline.angle_board_deg,
            "temperature_spec_c": report.baseline.temperature_spec_c,
            "temperature_board_c": report.baseline.temperature_board_c,
            "confidence": report.baseline.confidence,
            "peak_value": report.baseline.peak_value,
            "runner_up_value": report.baseline.runner_up_value,
            "dial_radius_px": report.baseline.dial_radius_px,
        },
    }
    report_path.write_text(json.dumps(report_json, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    """Compare the AI and baseline outputs for the requested captures."""

    args = _parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gauge_specs = load_gauge_specs()
    if args.gauge_id not in gauge_specs:
        raise KeyError(f"Unknown gauge id: {args.gauge_id}")
    gauge_spec = gauge_specs[args.gauge_id]

    contract = _load_tip_focus_tflite_contract(args.model_path)
    print(
        f"[TIP_FOCUS] model={args.model_path.name} input={contract.input_width}x{contract.input_height} "
        f"dtype={contract.input_dtype} outputs={len(contract.output_details)}",
        flush=True,
    )

    for capture_path in args.captures:
        if not capture_path.is_file():
            raise FileNotFoundError(f"Capture not found: {capture_path}")
        report = _compare_capture(
            capture_path=capture_path,
            contract=contract,
            gauge_spec=gauge_spec,
            truth_temperature_c=args.truth_temperature_c,
            output_dir=output_dir,
        )
        print(
            f"[TIP_FOCUS] {capture_path.name}: AI={report.ai.temperature_spec_c:.2f}C "
            f"({report.ai.guardrail_ok}) baseline="
            f"{report.baseline.temperature_spec_c:.2f}C ({report.baseline.valid})",
            flush=True,
        )


if __name__ == "__main__":
    main()
