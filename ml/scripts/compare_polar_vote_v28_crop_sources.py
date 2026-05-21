#!/usr/bin/env python3
"""Compare exact V28 accuracy across the candidate crop sources.

This script keeps the crop question honest before we flash the board again.
It evaluates the same exact ``polar_vote_circular_v28`` model on the same
labeled captures with four crop sources:

* the fixed training crop used by the firmware fallback,
* the saved rectified crop boxes from offline labeling,
* the board-style bright-centroid crop heuristic, and
* the rectifier-model crop decoded from the full-frame tensor.

The goal is to show which crop source reproduces the offline V28 behavior best.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import random
import sys
from typing import Any, Final, cast

import numpy as np
import tensorflow as tf

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPO_ROOT: Final[Path] = PROJECT_ROOT.parent
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "tmp" / "matplotlib"))

from embedded_gauge_reading_tinyml.board_crop_compare import (  # noqa: E402
    estimate_board_crop_from_rgb,
)
from embedded_gauge_reading_tinyml.board_pipeline import (  # noqa: E402
    OBB_CROP_SCALE,
    OBB_HEIGHT_SCALE,
    OBB_MIN_CROP_SIZE_PIXELS,
    OBB_SOURCE_HEIGHT_SCALE,
    OBB_SOURCE_WIDTH_SCALE,
    OBB_WIDTH_SCALE,
    RECTIFIER_CROP_SCALE,
    ModelSession,
    decode_obb_crop_box,
    decode_rectifier_crop_box,
    load_model_session,
    load_capture_image,
    prepare_full_frame,
)
from embedded_gauge_reading_tinyml.firmware_preprocessing import (  # noqa: E402
    build_training_style_polar_vote_float32,
    decode_circular_vote_logits,
    firmware_training_crop_box,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs  # noqa: E402
from embedded_gauge_reading_tinyml.polar_vote_v28 import (  # noqa: E402
    build_polar_vote_v28_model,
)

DEFAULT_MANIFEST: Final[Path] = PROJECT_ROOT / "data" / "hard_cases_plus_board30_valid_with_new6.csv"
DEFAULT_CAPTURE_ROOT: Final[Path] = PROJECT_ROOT / "data" / "captured_images"
DEFAULT_CROP_BOXES: Final[Path] = PROJECT_ROOT / "data" / "rectified_crop_boxes_v5_all.csv"
DEFAULT_WEIGHTS: Final[Path] = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "polar_vote_circular_v28"
    / "best_weights.weights.h5"
)
DEFAULT_OBB_MODEL: Final[Path] = (
    PROJECT_ROOT / "artifacts" / "deployment" / "prod_model_v0.3_obb_int8" / "model_int8.tflite"
)
DEFAULT_RECTIFIER_MODEL: Final[Path] = (
    PROJECT_ROOT
    / "artifacts"
    / "deployment"
    / "mobilenetv2_rectifier_hardcase_finetune_v3_int8"
    / "model_int8.tflite"
)
DEFAULT_OUTPUT_DIR: Final[Path] = REPO_ROOT / "tmp" / "polar_vote_v28_crop_sources"
DEFAULT_GAUGE_ID: Final[str] = "littlegood_home_temp_gauge_c"


@dataclass(frozen=True, slots=True)
class EvalItem:
    """One labeled capture and its target value."""

    image_path: Path
    value: float
    sample_weight: float


@dataclass(frozen=True, slots=True)
class StrategyRow:
    """Per-sample comparison row for all crop strategies."""

    image_path: str
    source_kind: str
    value: float
    sample_weight: float
    fixed_prediction: float
    fixed_abs_error: float
    rectified_prediction: float | None
    rectified_abs_error: float | None
    board_prediction: float | None
    board_abs_error: float | None
    obb_prediction: float
    obb_abs_error: float
    rectifier_prediction: float
    rectifier_abs_error: float
    rectifier_fallback_reason: str | None
    obb_fallback_reason: str | None
    obb_crop_x0: float
    obb_crop_y0: float
    obb_crop_x1: float
    obb_crop_y1: float
    rectifier_crop_x0: float
    rectifier_crop_y0: float
    rectifier_crop_x1: float
    rectifier_crop_y1: float


def _parse_args() -> argparse.Namespace:
    """Parse the comparison CLI."""
    parser = argparse.ArgumentParser(
        description="Compare exact V28 accuracy across crop strategies."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="CSV manifest with image_path,value rows.",
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=DEFAULT_CAPTURE_ROOT,
        help="Only keep manifest rows that live under this directory.",
    )
    parser.add_argument(
        "--crop-boxes",
        type=Path,
        default=DEFAULT_CROP_BOXES,
        help="CSV file with rectified crop boxes.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to the exact V28 best-weights checkpoint.",
    )
    parser.add_argument(
        "--rectifier-model",
        type=Path,
        default=DEFAULT_RECTIFIER_MODEL,
        help="Path to the rectifier model used for the board crop.",
    )
    parser.add_argument(
        "--obb-model",
        type=Path,
        default=DEFAULT_OBB_MODEL,
        help="Path to the OBB localizer model.",
    )
    parser.add_argument(
        "--obb-model-kind",
        type=str,
        choices=("auto", "keras", "tflite"),
        default="auto",
        help="Backend used to load the OBB model.",
    )
    parser.add_argument(
        "--rectifier-model-kind",
        type=str,
        choices=("auto", "keras", "tflite"),
        default="auto",
        help="Backend used to load the rectifier model.",
    )
    parser.add_argument(
        "--rectifier-crop-scale",
        type=float,
        default=RECTIFIER_CROP_SCALE,
        help="Width/height scale applied when decoding the rectifier crop box.",
    )
    parser.add_argument(
        "--obb-crop-scale",
        type=float,
        default=OBB_CROP_SCALE,
        help="Scale factor applied when decoding the OBB crop box.",
    )
    parser.add_argument(
        "--obb-width-scale",
        type=float,
        default=OBB_WIDTH_SCALE,
        help="Extra width multiplier applied when decoding the OBB crop box.",
    )
    parser.add_argument(
        "--obb-height-scale",
        type=float,
        default=OBB_HEIGHT_SCALE,
        help="Extra height multiplier applied when decoding the OBB crop box.",
    )
    parser.add_argument(
        "--obb-source-width-scale",
        type=float,
        default=OBB_SOURCE_WIDTH_SCALE,
        help="Width multiplier applied after projecting the OBB crop into source coordinates.",
    )
    parser.add_argument(
        "--obb-source-height-scale",
        type=float,
        default=OBB_SOURCE_HEIGHT_SCALE,
        help="Height multiplier applied after projecting the OBB crop into source coordinates.",
    )
    parser.add_argument(
        "--obb-min-crop-size",
        type=float,
        default=OBB_MIN_CROP_SIZE_PIXELS,
        help="Minimum edge length enforced for the OBB crop box.",
    )
    parser.add_argument(
        "--obb-center-x-bias-pixels",
        type=float,
        default=0.0,
        help="Horizontal pixel bias applied to the OBB crop center.",
    )
    parser.add_argument(
        "--obb-center-y-bias-pixels",
        type=float,
        default=0.0,
        help="Vertical pixel bias applied to the OBB crop center.",
    )
    parser.add_argument(
        "--obb-source-x-bias-pixels",
        type=float,
        default=0.0,
        help="Horizontal pixel bias applied after the OBB crop is projected into source coordinates.",
    )
    parser.add_argument(
        "--obb-source-y-bias-pixels",
        type=float,
        default=0.0,
        help="Vertical pixel bias applied after the OBB crop is projected into source coordinates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV and JSON outputs should be written.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of samples to score. Use 0 for all rows.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples before truncation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for shuffling.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Expected capture size used for yuv422/raw16 decoding.",
    )
    parser.add_argument(
        "--gauge-id",
        type=str,
        default=DEFAULT_GAUGE_ID,
        help="Gauge spec identifier from the calibration TOML.",
    )
    parser.add_argument(
        "--board-bright-threshold",
        type=int,
        default=80,
        help="Bright-pixel threshold for the board heuristic crop.",
    )
    parser.add_argument(
        "--board-border-pixels",
        type=int,
        default=16,
        help="Border width ignored by the board heuristic crop.",
    )
    parser.add_argument(
        "--board-width-scale",
        type=float,
        default=17.0 / 20.0,
        help="Board heuristic width scale relative to the training crop width.",
    )
    parser.add_argument(
        "--board-height-scale",
        type=float,
        default=17.0 / 20.0,
        help="Board heuristic height scale relative to the training crop height.",
    )
    parser.add_argument(
        "--board-center-x-bias-pixels",
        type=int,
        default=0,
        help="Board heuristic horizontal bias in pixels.",
    )
    parser.add_argument(
        "--board-center-y-bias-ratio",
        type=float,
        default=0.11,
        help="Board heuristic vertical bias ratio relative to crop height.",
    )
    parser.add_argument(
        "--board-center-y-bias-min-pixels",
        type=int,
        default=8,
        help="Lower bound for the board heuristic vertical bias in pixels.",
    )
    parser.add_argument(
        "--board-center-y-bias-max-pixels",
        type=int,
        default=18,
        help="Upper bound for the board heuristic vertical bias in pixels.",
    )
    parser.add_argument(
        "--center-search-px",
        type=int,
        default=5,
        help="Angular center search radius in pixels used by the offline V28 recipe.",
    )
    parser.add_argument(
        "--allow-missing-crop-boxes",
        action="store_true",
        help="Keep manifest rows even when no rectified crop box exists.",
    )
    return parser.parse_args()


def _resolve_repo_path(raw_path: str) -> Path:
    """Resolve a manifest path against the repository root when needed."""
    normalized = raw_path.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _is_under_root(path: Path, root: Path) -> bool:
    """Return True when a path resolves inside a capture root."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_manifest(manifest_path: Path, capture_root: Path) -> list[EvalItem]:
    """Load labeled rows and keep only the captures under the requested root."""
    items: list[EvalItem] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = _resolve_repo_path(str(row["image_path"]))
            if not _is_under_root(image_path, capture_root):
                continue
            value = float(row["value"])
            sample_weight = float(row.get("sample_weight", 1.0) or 1.0)
            items.append(
                EvalItem(
                    image_path=image_path,
                    value=value,
                    sample_weight=sample_weight,
                )
            )
    return items


def _select_items(
    items: list[EvalItem],
    *,
    max_samples: int,
    shuffle: bool,
    seed: int,
) -> list[EvalItem]:
    """Select a deterministic subset of the filtered samples."""
    if shuffle:
        rng = random.Random(seed)
        items = list(items)
        rng.shuffle(items)
    if max_samples > 0:
        return items[:max_samples]
    return items


def _load_crop_boxes(
    crop_boxes_path: Path,
) -> dict[str, tuple[float, float, float, float]]:
    """Load rectified crop boxes keyed by resolved absolute image path."""
    boxes: dict[str, tuple[float, float, float, float]] = {}
    with crop_boxes_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "x0", "y0", "x1", "y1"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{crop_boxes_path} does not contain the required columns: {sorted(required)}"
            )
        for row in reader:
            image_path = _resolve_repo_path(str(row["image_path"]))
            boxes[str(image_path.resolve())] = (
                float(row["x0"]),
                float(row["y0"]),
                float(row["x1"]),
                float(row["y1"]),
            )
    return boxes


def _format_box(crop_box_xyxy: tuple[float, float, float, float] | None) -> dict[str, float | None]:
    """Convert an xyxy box to JSON-friendly coordinates."""
    if crop_box_xyxy is None:
        return {"x0": None, "y0": None, "x1": None, "y1": None}
    return {
        "x0": float(crop_box_xyxy[0]),
        "y0": float(crop_box_xyxy[1]),
        "x1": float(crop_box_xyxy[2]),
        "y1": float(crop_box_xyxy[3]),
    }


def _quantize_input(batch: np.ndarray, input_details: dict[str, Any]) -> np.ndarray:
    """Convert a float batch into the dtype expected by a TFLite interpreter."""
    input_dtype = np.dtype(input_details["dtype"])
    batch_array = np.ascontiguousarray(batch)

    if batch_array.dtype == input_dtype:
        return batch_array

    if np.issubdtype(input_dtype, np.floating):
        return batch_array.astype(input_dtype, copy=False)

    if input_dtype == np.uint8 and batch_array.dtype == np.int8:
        return (batch_array.astype(np.int16) + 128).astype(np.uint8)
    if input_dtype == np.int8 and batch_array.dtype == np.uint8:
        return (batch_array.astype(np.int16) - 128).astype(np.int8)

    scale = float(input_details["quantization"][0])
    zero_point = int(input_details["quantization"][1])
    qmin, qmax = np.iinfo(input_dtype).min, np.iinfo(input_dtype).max
    quantized = np.round(batch_array.astype(np.float32) / scale + zero_point)
    return np.clip(quantized, qmin, qmax).astype(input_dtype)


def _dequantize_output(tensor: np.ndarray, details: dict[str, Any]) -> np.ndarray:
    """Convert a quantized output tensor back into float32 values."""
    scale = float(details["quantization"][0])
    zero_point = int(details["quantization"][1])
    return scale * (np.asarray(tensor, dtype=np.float32) - zero_point)


def _run_session(
    session: ModelSession,
    batch: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Run one model backend and return the first usable output tensor."""
    if session.kind == "keras":
        model = cast(tf.keras.Model, session.model)
        outputs = model.predict(batch, verbose=0)
        if isinstance(outputs, dict):
            if "rectifier_box" in outputs:
                return np.asarray(outputs["rectifier_box"], dtype=np.float32), "rectifier_box"
            if "obb_params" in outputs:
                return np.asarray(outputs["obb_params"], dtype=np.float32), "obb_params"
            first_key = next(iter(outputs))
            return np.asarray(outputs[first_key], dtype=np.float32), first_key
        if isinstance(outputs, (list, tuple)):
            return np.asarray(outputs[0], dtype=np.float32), "output_0"
        return np.asarray(outputs, dtype=np.float32), "output"

    interpreter = cast(tf.lite.Interpreter, session.model)
    if session.input_details is None or session.output_details is None:
        raise ValueError("TFLite session is missing tensor metadata.")
    quantized_batch = _quantize_input(batch, session.input_details)
    interpreter.set_tensor(int(session.input_details["index"]), quantized_batch)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(int(session.output_details["index"]))[0]
    return _dequantize_output(output_tensor, session.output_details), str(
        session.output_details.get("name", "output")
    )


def _predict_exact_v28(
    model: tf.keras.Model,
    source_image: np.ndarray,
    crop_box_xyxy: tuple[float, float, float, float],
    *,
    center_search_px: int,
    gauge_spec: Any,
) -> tuple[float, tuple[float, float, float, float]]:
    """Run the exact V28 recipe on one crop and return the prediction."""
    tensor = build_training_style_polar_vote_float32(
        source_image,
        crop_box_xyxy=crop_box_xyxy,
        center_search_px=center_search_px,
        gauge_spec=gauge_spec,
    )
    logits = model.predict(tensor[None, ...], verbose=0)[0]
    prediction = decode_circular_vote_logits(logits, gauge_spec)
    return float(prediction), crop_box_xyxy


def _weighted_mean_abs_error(errors: list[float], weights: list[float]) -> float:
    """Compute a sample-weighted MAE for the manifest comparison."""
    if not errors or not weights:
        return float("nan")
    if len(errors) != len(weights):
        raise ValueError("Errors and weights must have the same length.")
    error_array = np.asarray(errors, dtype=np.float32)
    weight_array = np.asarray(weights, dtype=np.float32)
    return float(np.sum(error_array * weight_array) / np.sum(weight_array))


def main() -> None:
    """Entry point for the crop-strategy comparison CLI."""
    args = _parse_args()
    items = _select_items(
        _load_manifest(args.manifest, args.capture_root),
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not items:
        raise FileNotFoundError("No captures were selected for replay.")

    crop_boxes = _load_crop_boxes(args.crop_boxes)
    exact_model = build_polar_vote_v28_model()
    exact_model.load_weights(args.weights)
    rectifier_session = load_model_session(args.rectifier_model, args.rectifier_model_kind)
    obb_session = load_model_session(args.obb_model, args.obb_model_kind)
    gauge_spec = load_gauge_specs()[args.gauge_id]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[StrategyRow] = []
    fixed_errors: list[float] = []
    fixed_weights: list[float] = []
    rectified_errors: list[float] = []
    rectified_weights: list[float] = []
    board_errors: list[float] = []
    board_weights: list[float] = []
    obb_errors: list[float] = []
    obb_weights: list[float] = []
    rectifier_errors: list[float] = []
    rectifier_weights: list[float] = []

    for item in items:
        source_image, source_kind = load_capture_image(
            item.image_path,
            image_width=args.image_size,
            image_height=args.image_size,
        )
        fixed_crop = firmware_training_crop_box(source_image.shape[1], source_image.shape[0])
        fixed_prediction, _ = _predict_exact_v28(
            exact_model,
            source_image,
            fixed_crop,
            center_search_px=args.center_search_px,
            gauge_spec=gauge_spec,
        )
        fixed_error = abs(fixed_prediction - item.value)
        fixed_errors.append(fixed_error)
        fixed_weights.append(item.sample_weight)

        rectified_key = str(item.image_path.resolve())
        rectified_crop = crop_boxes.get(rectified_key)
        rectified_prediction: float | None = None
        rectified_error: float | None = None
        if rectified_crop is not None:
            rectified_prediction, _ = _predict_exact_v28(
                exact_model,
                source_image,
                rectified_crop,
                center_search_px=args.center_search_px,
                gauge_spec=gauge_spec,
            )
            rectified_error = abs(rectified_prediction - item.value)
            rectified_errors.append(rectified_error)
            rectified_weights.append(item.sample_weight)
        elif not args.allow_missing_crop_boxes:
            raise KeyError(
                f"No rectified crop box found for {item.image_path}. "
                "Pass --allow-missing-crop-boxes to keep the row anyway."
            )

        board_estimate = estimate_board_crop_from_rgb(
            source_image,
            bright_threshold=args.board_bright_threshold,
            border_pixels=args.board_border_pixels,
            crop_width_scale=args.board_width_scale,
            crop_height_scale=args.board_height_scale,
            center_x_bias_pixels=args.board_center_x_bias_pixels,
            center_y_bias_ratio=args.board_center_y_bias_ratio,
            center_y_bias_min_pixels=args.board_center_y_bias_min_pixels,
            center_y_bias_max_pixels=args.board_center_y_bias_max_pixels,
        )
        board_prediction: float | None = None
        board_error: float | None = None
        if board_estimate is not None:
            board_box = (
                float(board_estimate.crop_box.x_min),
                float(board_estimate.crop_box.y_min),
                float(board_estimate.crop_box.x_max),
                float(board_estimate.crop_box.y_max),
            )
            board_prediction, _ = _predict_exact_v28(
                exact_model,
                source_image,
                board_box,
                center_search_px=args.center_search_px,
                gauge_spec=gauge_spec,
            )
            board_error = abs(board_prediction - item.value)
            board_errors.append(board_error)
            board_weights.append(item.sample_weight)

        full_frame = prepare_full_frame(source_image, image_size=args.image_size)
        full_frame_batch = np.expand_dims(full_frame.astype(np.float32) / 255.0, axis=0)
        obb_output, _obb_output_name = _run_session(
            obb_session,
            full_frame_batch,
        )
        obb_decision = decode_obb_crop_box(
            np.asarray(obb_output, dtype=np.float32).reshape(-1),
            source_width=source_image.shape[1],
            source_height=source_image.shape[0],
            input_size=args.image_size,
            obb_crop_scale=args.obb_crop_scale,
            obb_width_scale=args.obb_width_scale,
            obb_height_scale=args.obb_height_scale,
            obb_source_width_scale=args.obb_source_width_scale,
            obb_source_height_scale=args.obb_source_height_scale,
            min_crop_size=args.obb_min_crop_size,
            obb_center_x_bias_pixels=args.obb_center_x_bias_pixels,
            obb_center_y_bias_pixels=args.obb_center_y_bias_pixels,
            obb_source_x_bias_pixels=args.obb_source_x_bias_pixels,
            obb_source_y_bias_pixels=args.obb_source_y_bias_pixels,
        )
        obb_box = obb_decision.crop_box_xyxy
        obb_prediction, _ = _predict_exact_v28(
            exact_model,
            source_image,
            obb_box,
            center_search_px=args.center_search_px,
            gauge_spec=gauge_spec,
        )
        obb_error = abs(obb_prediction - item.value)
        obb_errors.append(obb_error)
        obb_weights.append(item.sample_weight)

        rectifier_output, _rectifier_output_name = _run_session(
            rectifier_session,
            full_frame_batch,
        )
        rectifier_decision = decode_rectifier_crop_box(
            np.asarray(rectifier_output, dtype=np.float32).reshape(-1),
            source_width=source_image.shape[1],
            source_height=source_image.shape[0],
            input_size=args.image_size,
            rectifier_crop_scale=args.rectifier_crop_scale,
        )
        rectifier_box = rectifier_decision.crop_box_xyxy
        rectifier_prediction, _ = _predict_exact_v28(
            exact_model,
            source_image,
            rectifier_box,
            center_search_px=args.center_search_px,
            gauge_spec=gauge_spec,
        )
        rectifier_error = abs(rectifier_prediction - item.value)
        rectifier_errors.append(rectifier_error)
        rectifier_weights.append(item.sample_weight)

        rows.append(
            StrategyRow(
                image_path=str(item.image_path.resolve()),
                source_kind=source_kind,
                value=item.value,
                sample_weight=item.sample_weight,
                fixed_prediction=fixed_prediction,
                fixed_abs_error=fixed_error,
                rectified_prediction=rectified_prediction,
                rectified_abs_error=rectified_error,
                board_prediction=board_prediction,
                board_abs_error=board_error,
                obb_prediction=obb_prediction,
                obb_abs_error=obb_error,
                rectifier_prediction=rectifier_prediction,
                rectifier_abs_error=rectifier_error,
                rectifier_fallback_reason=rectifier_decision.fallback_reason,
                obb_fallback_reason=obb_decision.fallback_reason,
                obb_crop_x0=float(obb_box[0]),
                obb_crop_y0=float(obb_box[1]),
                obb_crop_x1=float(obb_box[2]),
                obb_crop_y1=float(obb_box[3]),
                rectifier_crop_x0=float(rectifier_box[0]),
                rectifier_crop_y0=float(rectifier_box[1]),
                rectifier_crop_x1=float(rectifier_box[2]),
                rectifier_crop_y1=float(rectifier_box[3]),
            )
        )

    csv_path = args.output_dir / "crop_sources.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(asdict(rows[0]).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    summary = {
        "items": len(rows),
        "output_dir": str(args.output_dir),
        "weights": str(args.weights),
        "rectifier_model": str(args.rectifier_model),
        "fixed_mae": _weighted_mean_abs_error(fixed_errors, fixed_weights),
        "rectified_mae": _weighted_mean_abs_error(rectified_errors, rectified_weights),
        "board_mae": _weighted_mean_abs_error(board_errors, board_weights),
        "obb_mae": _weighted_mean_abs_error(obb_errors, obb_weights),
        "rectifier_mae": _weighted_mean_abs_error(rectifier_errors, rectifier_weights),
        "fixed_samples": len(fixed_errors),
        "rectified_samples": len(rectified_errors),
        "board_samples": len(board_errors),
        "obb_samples": len(obb_errors),
        "rectifier_samples": len(rectifier_errors),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[CROP] Wrote rows to {csv_path}", flush=True)
    print(f"[CROP] Wrote summary to {summary_path}", flush=True)
    print(
        "[CROP] "
        f"fixed_mae={summary['fixed_mae']:.6f} "
        f"rectified_mae={summary['rectified_mae']:.6f} "
        f"board_mae={summary['board_mae']:.6f} "
        f"obb_mae={summary['obb_mae']:.6f} "
        f"rectifier_mae={summary['rectifier_mae']:.6f} "
        f"rectifier_crop_scale={args.rectifier_crop_scale:.2f} "
        f"obb_width_scale={args.obb_width_scale:.3f} "
        f"obb_height_scale={args.obb_height_scale:.3f} "
        f"obb_source_width_scale={args.obb_source_width_scale:.3f} "
        f"obb_source_height_scale={args.obb_source_height_scale:.3f} "
        f"obb_source_x_bias_pixels={args.obb_source_x_bias_pixels:.1f} "
        f"obb_source_y_bias_pixels={args.obb_source_y_bias_pixels:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
