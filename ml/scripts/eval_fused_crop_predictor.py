#!/usr/bin/env python3
"""Fit and evaluate a fused crop predictor for exact polar-vote V28 replay.

The goal of this script is to answer one narrow question:
can we learn a crop selector that makes the exact offline V28 model stay close
to the rectified oracle on a held-out set?

It trains an ExtraTrees regressor from the available crop sources:
the rectifier crop, the OBB crop, and the classical board heuristic crop.
The target is the rectified crop boxes from offline labeling. The script then
scores the exact V28 model on a held-out manifest using the fused prediction.
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
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
import tensorflow as tf
from sklearn.ensemble import ExtraTreesRegressor

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
    load_capture_image,
    load_model_session,
    prepare_full_frame,
)
from embedded_gauge_reading_tinyml.firmware_preprocessing import (  # noqa: E402
    build_training_style_polar_vote_float32,
    decode_circular_vote_logits,
    firmware_training_crop_box,
)
from embedded_gauge_reading_tinyml.fused_crop_predictor import (  # noqa: E402
    resolve_dataset_image_path,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs  # noqa: E402
from embedded_gauge_reading_tinyml.polar_vote_v28 import (  # noqa: E402
    build_polar_vote_v28_model,
)

DEFAULT_CROP_BOXES: Final[Path] = PROJECT_ROOT / "data" / "rectified_crop_boxes_v5_all.csv"
DEFAULT_HOLDOUT_MANIFEST: Final[Path] = PROJECT_ROOT / "data" / "hard_cases_plus_board30_valid_with_new6.csv"
DEFAULT_EXACT_WEIGHTS: Final[Path] = (
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
DEFAULT_OUTPUT_DIR: Final[Path] = REPO_ROOT / "tmp" / "fused_crop_predictor_eval"
DEFAULT_GAUGE_ID: Final[str] = "littlegood_home_temp_gauge_c"
DEFAULT_ENSEMBLE_SIZE: Final[int] = 400


@dataclass(frozen=True, slots=True)
class CropBoxRow:
    """One rectified crop label used for training the crop predictor."""

    image_path: Path
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True, slots=True)
class HoldoutRow:
    """One holdout capture with the exact temperature target."""

    image_path: Path
    value: float
    sample_weight: float


@dataclass(frozen=True, slots=True)
class PredictionRow:
    """One scored holdout row from the fused crop predictor."""

    image_path: str
    value: float
    sample_weight: float
    fixed_prediction: float
    fixed_abs_error: float
    rectified_prediction: float
    rectified_abs_error: float
    board_prediction: float
    board_abs_error: float
    obb_prediction: float
    obb_abs_error: float
    rectifier_prediction: float
    rectifier_abs_error: float
    fused_prediction: float
    fused_abs_error: float
    fused_box_x0: float
    fused_box_y0: float
    fused_box_x1: float
    fused_box_y1: float
    fixed_box_x0: float
    fixed_box_y0: float
    fixed_box_x1: float
    fixed_box_y1: float
    rectified_box_x0: float
    rectified_box_y0: float
    rectified_box_x1: float
    rectified_box_y1: float
    board_box_x0: float
    board_box_y0: float
    board_box_x1: float
    board_box_y1: float
    obb_box_x0: float
    obb_box_y0: float
    obb_box_x1: float
    obb_box_y1: float
    rectifier_box_x0: float
    rectifier_box_y0: float
    rectifier_box_x1: float
    rectifier_box_y1: float


def _parse_args() -> argparse.Namespace:
    """Parse the fusion-evaluation command line."""
    parser = argparse.ArgumentParser(
        description="Fit a fused crop predictor and score exact V28 on a holdout set."
    )
    parser.add_argument(
        "--crop-boxes",
        type=Path,
        default=DEFAULT_CROP_BOXES,
        help="CSV file with rectified crop boxes.",
    )
    parser.add_argument(
        "--holdout-manifest",
        type=Path,
        default=DEFAULT_HOLDOUT_MANIFEST,
        help="CSV manifest with image_path,value rows for the held-out evaluation set.",
    )
    parser.add_argument(
        "--exact-weights",
        type=Path,
        default=DEFAULT_EXACT_WEIGHTS,
        help="Path to the exact V28 best-weights checkpoint.",
    )
    parser.add_argument(
        "--obb-model",
        type=Path,
        default=DEFAULT_OBB_MODEL,
        help="Path to the OBB localizer model.",
    )
    parser.add_argument(
        "--rectifier-model",
        type=Path,
        default=DEFAULT_RECTIFIER_MODEL,
        help="Path to the rectifier model.",
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV and JSON outputs should be written.",
    )
    parser.add_argument(
        "--gauge-id",
        type=str,
        default=DEFAULT_GAUGE_ID,
        help="Gauge spec identifier from the calibration TOML.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square canvas size used by the board-localizer models.",
    )
    parser.add_argument(
        "--center-search-px",
        type=int,
        default=5,
        help="Angular center search radius used by the exact V28 evaluator.",
    )
    parser.add_argument(
        "--center-mode",
        type=str,
        choices=("image_center", "classical_baseline"),
        default="image_center",
        help="How to choose the polar projection center.",
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
        "--rectifier-crop-scale",
        type=float,
        default=RECTIFIER_CROP_SCALE,
        help="Width/height scale applied when decoding the rectifier crop box.",
    )
    parser.add_argument(
        "--train-rows",
        type=int,
        default=0,
        help="Optional cap on the number of training rows. Use 0 for all rows.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the training rows before truncation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for shuffling and the tree ensemble.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_ENSEMBLE_SIZE,
        help="Number of trees in the fused ExtraTrees regressor.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples required at a tree leaf.",
    )
    return parser.parse_args()


def _load_crop_boxes(crop_boxes_path: Path) -> list[CropBoxRow]:
    """Load rectified crop boxes from CSV."""
    rows: list[CropBoxRow] = []
    with crop_boxes_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "x0", "y0", "x1", "y1"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Crop-box CSV {crop_boxes_path} is missing one of {sorted(required)}."
            )
        for row in reader:
            image_path = resolve_dataset_image_path(row["image_path"])
            rows.append(
                CropBoxRow(
                    image_path=image_path,
                    x0=float(row["x0"]),
                    y0=float(row["y0"]),
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                )
            )
    return rows


def _load_holdout_manifest(holdout_manifest_path: Path) -> list[HoldoutRow]:
    """Load the held-out exact-V28 evaluation set."""
    rows: list[HoldoutRow] = []
    with holdout_manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "value"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Holdout manifest {holdout_manifest_path} is missing one of {sorted(required)}."
            )
        for row in reader:
            rows.append(
                HoldoutRow(
                    image_path=resolve_dataset_image_path(row["image_path"]),
                    value=float(row["value"]),
                    sample_weight=float(row.get("sample_weight", 1.0) or 1.0),
                )
            )
    return rows


def _select_training_rows(
    rows: list[CropBoxRow],
    *,
    holdout_paths: set[str],
    train_rows: int,
    shuffle: bool,
    seed: int,
) -> list[CropBoxRow]:
    """Pick the training subset after excluding the holdout paths."""
    filtered = [row for row in rows if row.image_path.resolve().as_posix() not in holdout_paths]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(filtered)
    if train_rows > 0:
        return filtered[:train_rows]
    return filtered


def _clip_box(box: NDArray[np.float32], *, width: int, height: int) -> tuple[float, float, float, float]:
    """Clamp a crop prediction to a valid axis-aligned box."""
    x0 = float(np.clip(min(box[0], box[2]), 0.0, float(width - 1)))
    y0 = float(np.clip(min(box[1], box[3]), 0.0, float(height - 1)))
    x1 = float(np.clip(max(box[0], box[2]), x0 + 1.0, float(width)))
    y1 = float(np.clip(max(box[1], box[3]), y0 + 1.0, float(height)))
    return (x0, y0, x1, y1)


def _sample_weighted_mae(errors: list[float], weights: list[float]) -> float:
    """Compute the weighted mean absolute error."""
    if not errors or not weights:
        return float("nan")
    error_array = np.asarray(errors, dtype=np.float32)
    weight_array = np.asarray(weights, dtype=np.float32)
    return float(np.sum(error_array * weight_array) / np.sum(weight_array))


def _predict_exact_v28(
    model: tf.keras.Model,
    source_image: NDArray[np.uint8],
    crop_box_xyxy: tuple[float, float, float, float],
    *,
    gauge_spec: Any,
    center_search_px: int,
    center_mode: str,
) -> float:
    """Run the exact offline V28 model on one crop."""
    tensor = build_training_style_polar_vote_float32(
        source_image,
        crop_box_xyxy=crop_box_xyxy,
        output_dim=224,
        input_mode="rgb_edge6_vote7",
        center_search_px=center_search_px,
        center_mode=center_mode,
        gauge_spec=gauge_spec,
    )
    logits = model.predict(tensor[None, ...], verbose=0)[0]
    return float(decode_circular_vote_logits(logits, gauge_spec))


def _build_feature_vector(
    source_image: NDArray[np.uint8],
    *,
    image_size: int,
    obb_session: ModelSession,
    rectifier_session: ModelSession,
    obb_crop_scale: float,
    obb_width_scale: float,
    obb_height_scale: float,
    obb_source_width_scale: float,
    obb_source_height_scale: float,
    obb_min_crop_size: float,
    obb_center_x_bias_pixels: float,
    obb_center_y_bias_pixels: float,
    obb_source_x_bias_pixels: float,
    obb_source_y_bias_pixels: float,
    rectifier_crop_scale: float,
) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float], bool, bool]:
    """Build the fused feature vector and the source crops used by the model."""
    full_frame = prepare_full_frame(source_image, image_size=image_size)
    full_frame_batch = np.expand_dims(full_frame.astype(np.float32) / 255.0, axis=0)

    obb_output, _ = _run_session(obb_session, full_frame_batch)
    obb_decision = decode_obb_crop_box(
        np.asarray(obb_output, dtype=np.float32).reshape(-1),
        source_width=source_image.shape[1],
        source_height=source_image.shape[0],
        input_size=image_size,
        obb_crop_scale=obb_crop_scale,
        obb_width_scale=obb_width_scale,
        obb_height_scale=obb_height_scale,
        obb_source_width_scale=obb_source_width_scale,
        obb_source_height_scale=obb_source_height_scale,
        min_crop_size=obb_min_crop_size,
        obb_center_x_bias_pixels=obb_center_x_bias_pixels,
        obb_center_y_bias_pixels=obb_center_y_bias_pixels,
        obb_source_x_bias_pixels=obb_source_x_bias_pixels,
        obb_source_y_bias_pixels=obb_source_y_bias_pixels,
    )
    obb_box = obb_decision.crop_box_xyxy

    rectifier_output, _ = _run_session(rectifier_session, full_frame_batch)
    rectifier_decision = decode_rectifier_crop_box(
        np.asarray(rectifier_output, dtype=np.float32).reshape(-1),
        source_width=source_image.shape[1],
        source_height=source_image.shape[0],
        input_size=image_size,
        rectifier_crop_scale=rectifier_crop_scale,
    )
    rectifier_box = rectifier_decision.crop_box_xyxy

    board_estimate = estimate_board_crop_from_rgb(source_image)
    if board_estimate is None:
        board_box = firmware_training_crop_box(source_image.shape[1], source_image.shape[0])
        board_available = False
    else:
        board_box = (
            float(board_estimate.crop_box.x_min),
            float(board_estimate.crop_box.y_min),
            float(board_estimate.crop_box.x_max),
            float(board_estimate.crop_box.y_max),
        )
        board_available = True

    fixed_box = firmware_training_crop_box(source_image.shape[1], source_image.shape[0])
    # The rectified crop label is loaded separately for training/evaluation.
    features = np.asarray(
        [
            rectifier_box[0],
            rectifier_box[1],
            rectifier_box[2],
            rectifier_box[3],
            obb_box[0],
            obb_box[1],
            obb_box[2],
            obb_box[3],
            board_box[0],
            board_box[1],
            board_box[2],
            board_box[3],
            1.0 if rectifier_decision.accepted else 0.0,
            1.0 if obb_decision.accepted else 0.0,
            1.0 if board_available else 0.0,
        ],
        dtype=np.float32,
    )
    return (
        features,
        fixed_box,
        obb_box,
        rectifier_box,
        board_box,
        obb_decision.accepted,
        rectifier_decision.accepted,
    )


def _run_session(session: ModelSession, batch: np.ndarray) -> tuple[np.ndarray, str]:
    """Run one model session and return the first output tensor."""
    if session.kind == "keras":
        model = session.model  # type: ignore[assignment]
        outputs = model.predict(batch, verbose=0)
        if isinstance(outputs, dict):
            key = next(iter(outputs))
            return np.asarray(outputs[key], dtype=np.float32)[0], key
        if isinstance(outputs, (list, tuple)):
            return np.asarray(outputs[0], dtype=np.float32)[0], "output_0"
        return np.asarray(outputs, dtype=np.float32)[0], "output"

    if session.input_details is None or session.output_details is None:
        raise ValueError("TFLite session is missing tensor metadata.")
    input_details = session.input_details
    output_details = session.output_details
    scale = float(input_details["quantization"][0])
    zero_point = int(input_details["quantization"][1])
    qmin, qmax = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    quantized = np.round(batch / scale + zero_point)
    quantized = np.clip(quantized, qmin, qmax).astype(np.int8)
    interpreter = session.model  # type: ignore[assignment]
    interpreter.set_tensor(int(input_details["index"]), quantized)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(int(output_details["index"]))[0]
    out_scale = float(output_details["quantization"][0])
    out_zero_point = int(output_details["quantization"][1])
    return (
        out_scale * (np.asarray(output_tensor, dtype=np.float32) - out_zero_point),
        str(output_details.get("name", "output")),
    )


def main() -> None:
    """Fit the crop-fusion model and evaluate it on the holdout set."""
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    crop_rows = _load_crop_boxes(args.crop_boxes)
    holdout_rows = _load_holdout_manifest(args.holdout_manifest)
    holdout_paths = {row.image_path.resolve().as_posix() for row in holdout_rows}
    train_rows = _select_training_rows(
        crop_rows,
        holdout_paths=holdout_paths,
        train_rows=args.train_rows,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not train_rows:
        raise FileNotFoundError("No training rows were selected for the fused crop predictor.")
    if not holdout_rows:
        raise FileNotFoundError("No holdout rows were selected for the fused crop predictor.")

    crop_box_map = {
        row.image_path.resolve().as_posix(): (row.x0, row.y0, row.x1, row.y1)
        for row in crop_rows
    }

    obb_session = load_model_session(args.obb_model, args.obb_model_kind)
    rectifier_session = load_model_session(args.rectifier_model, args.rectifier_model_kind)
    exact_model = build_polar_vote_v28_model()
    exact_model.load_weights(args.exact_weights)
    gauge_spec = load_gauge_specs()[args.gauge_id]

    train_features: list[np.ndarray] = []
    train_targets: list[tuple[float, float, float, float]] = []
    skipped_train = 0
    for row in train_rows:
        if not row.image_path.exists():
            skipped_train += 1
            continue
        source_image, _ = load_capture_image(row.image_path, image_width=args.image_size, image_height=args.image_size)
        features, fixed_box, obb_box, rectifier_box, board_box, obb_accepted, rectifier_accepted = _build_feature_vector(
            source_image,
            image_size=args.image_size,
            obb_session=obb_session,
            rectifier_session=rectifier_session,
            obb_crop_scale=args.obb_crop_scale,
            obb_width_scale=args.obb_width_scale,
            obb_height_scale=args.obb_height_scale,
            obb_source_width_scale=args.obb_source_width_scale,
            obb_source_height_scale=args.obb_source_height_scale,
            obb_min_crop_size=args.obb_min_crop_size,
            obb_center_x_bias_pixels=args.obb_center_x_bias_pixels,
            obb_center_y_bias_pixels=args.obb_center_y_bias_pixels,
            obb_source_x_bias_pixels=args.obb_source_x_bias_pixels,
            obb_source_y_bias_pixels=args.obb_source_y_bias_pixels,
            rectifier_crop_scale=args.rectifier_crop_scale,
        )
        train_features.append(features)
        train_targets.append((row.x0, row.y0, row.x1, row.y1))

    if not train_features:
        raise FileNotFoundError("Training rows were selected, but no images could be loaded.")

    X_train = np.stack(train_features, axis=0)
    y_train = np.asarray(train_targets, dtype=np.float32)

    regressor = ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
        min_samples_leaf=args.min_samples_leaf,
    )
    regressor.fit(X_train, y_train)

    holdout_predictions: list[PredictionRow] = []
    fixed_errors: list[float] = []
    rectified_errors: list[float] = []
    board_errors: list[float] = []
    obb_errors: list[float] = []
    rectifier_errors: list[float] = []
    fused_errors: list[float] = []
    fused_weights: list[float] = []
    fixed_weights: list[float] = []
    rectified_weights: list[float] = []
    board_weights: list[float] = []
    obb_weights: list[float] = []
    rectifier_weights: list[float] = []

    for row in holdout_rows:
        if not row.image_path.exists():
            raise FileNotFoundError(f"Holdout image does not exist: {row.image_path}")

        source_image, _ = load_capture_image(row.image_path, image_width=args.image_size, image_height=args.image_size)
        features, fixed_box, obb_box, rectifier_box, board_box, obb_accepted, rectifier_accepted = _build_feature_vector(
            source_image,
            image_size=args.image_size,
            obb_session=obb_session,
            rectifier_session=rectifier_session,
            obb_crop_scale=args.obb_crop_scale,
            obb_width_scale=args.obb_width_scale,
            obb_height_scale=args.obb_height_scale,
            obb_source_width_scale=args.obb_source_width_scale,
            obb_source_height_scale=args.obb_source_height_scale,
            obb_min_crop_size=args.obb_min_crop_size,
            obb_center_x_bias_pixels=args.obb_center_x_bias_pixels,
            obb_center_y_bias_pixels=args.obb_center_y_bias_pixels,
            obb_source_x_bias_pixels=args.obb_source_x_bias_pixels,
            obb_source_y_bias_pixels=args.obb_source_y_bias_pixels,
            rectifier_crop_scale=args.rectifier_crop_scale,
        )
        fused_box = _clip_box(regressor.predict(features[None, :]).reshape(-1), width=source_image.shape[1], height=source_image.shape[0])

        fixed_prediction = _predict_exact_v28(
            exact_model,
            source_image,
            fixed_box,
            gauge_spec=gauge_spec,
            center_search_px=args.center_search_px,
            center_mode=args.center_mode,
        )
        rectified_box = crop_box_map.get(row.image_path.resolve().as_posix())
        if rectified_box is None:
            raise KeyError(f"No rectified crop box exists for {row.image_path}")

        rectified_prediction = _predict_exact_v28(
            exact_model,
            source_image,
            rectified_box,
            gauge_spec=gauge_spec,
            center_search_px=args.center_search_px,
            center_mode=args.center_mode,
        )
        board_prediction = _predict_exact_v28(
            exact_model,
            source_image,
            board_box,
            gauge_spec=gauge_spec,
            center_search_px=args.center_search_px,
            center_mode=args.center_mode,
        )
        obb_prediction = _predict_exact_v28(
            exact_model,
            source_image,
            obb_box,
            gauge_spec=gauge_spec,
            center_search_px=args.center_search_px,
            center_mode=args.center_mode,
        )
        rectifier_prediction = _predict_exact_v28(
            exact_model,
            source_image,
            rectifier_box,
            gauge_spec=gauge_spec,
            center_search_px=args.center_search_px,
            center_mode=args.center_mode,
        )
        fused_prediction = _predict_exact_v28(
            exact_model,
            source_image,
            fused_box,
            gauge_spec=gauge_spec,
            center_search_px=args.center_search_px,
            center_mode=args.center_mode,
        )

        fixed_error = abs(fixed_prediction - row.value)
        rectified_error = abs(rectified_prediction - row.value)
        board_error = abs(board_prediction - row.value)
        obb_error = abs(obb_prediction - row.value)
        rectifier_error = abs(rectifier_prediction - row.value)
        fused_error = abs(fused_prediction - row.value)

        fixed_errors.append(fixed_error)
        rectified_errors.append(rectified_error)
        board_errors.append(board_error)
        obb_errors.append(obb_error)
        rectifier_errors.append(rectifier_error)
        fused_errors.append(fused_error)
        fixed_weights.append(row.sample_weight)
        rectified_weights.append(row.sample_weight)
        board_weights.append(row.sample_weight)
        obb_weights.append(row.sample_weight)
        rectifier_weights.append(row.sample_weight)
        fused_weights.append(row.sample_weight)

        holdout_predictions.append(
            PredictionRow(
                image_path=row.image_path.resolve().as_posix(),
                value=row.value,
                sample_weight=row.sample_weight,
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
                fused_prediction=fused_prediction,
                fused_abs_error=fused_error,
                fused_box_x0=fused_box[0],
                fused_box_y0=fused_box[1],
                fused_box_x1=fused_box[2],
                fused_box_y1=fused_box[3],
                fixed_box_x0=fixed_box[0],
                fixed_box_y0=fixed_box[1],
                fixed_box_x1=fixed_box[2],
                fixed_box_y1=fixed_box[3],
                rectified_box_x0=rectified_box[0],
                rectified_box_y0=rectified_box[1],
                rectified_box_x1=rectified_box[2],
                rectified_box_y1=rectified_box[3],
                board_box_x0=board_box[0],
                board_box_y0=board_box[1],
                board_box_x1=board_box[2],
                board_box_y1=board_box[3],
                obb_box_x0=obb_box[0],
                obb_box_y0=obb_box[1],
                obb_box_x1=obb_box[2],
                obb_box_y1=obb_box[3],
                rectifier_box_x0=rectifier_box[0],
                rectifier_box_y0=rectifier_box[1],
                rectifier_box_x1=rectifier_box[2],
                rectifier_box_y1=rectifier_box[3],
            )
        )

    prediction_csv = args.output_dir / "fused_predictions.csv"
    with prediction_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(asdict(holdout_predictions[0]).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in holdout_predictions:
            writer.writerow(asdict(row))

    summary = {
        "train_rows": len(train_rows),
        "train_loaded": len(train_features),
        "train_skipped_missing": skipped_train,
        "holdout_rows": len(holdout_rows),
        "n_estimators": args.n_estimators,
        "min_samples_leaf": args.min_samples_leaf,
        "fused_box_mae": _sample_weighted_mae([row.fused_abs_error for row in holdout_predictions], fused_weights),
        "fixed_mae": _sample_weighted_mae(fixed_errors, fixed_weights),
        "rectified_mae": _sample_weighted_mae(rectified_errors, rectified_weights),
        "board_mae": _sample_weighted_mae(board_errors, board_weights),
        "obb_mae": _sample_weighted_mae(obb_errors, obb_weights),
        "rectifier_mae": _sample_weighted_mae(rectifier_errors, rectifier_weights),
        "fused_exact_v28_mae": _sample_weighted_mae(fused_errors, fused_weights),
        "feature_count": int(X_train.shape[1]),
        "feature_importances": regressor.feature_importances_.tolist(),
        "fixed_under3_pct": float(np.mean(np.asarray(fixed_errors, dtype=np.float32) < 3.0)),
        "rectified_under3_pct": float(np.mean(np.asarray(rectified_errors, dtype=np.float32) < 3.0)),
        "board_under3_pct": float(np.mean(np.asarray(board_errors, dtype=np.float32) < 3.0)),
        "obb_under3_pct": float(np.mean(np.asarray(obb_errors, dtype=np.float32) < 3.0)),
        "rectifier_under3_pct": float(np.mean(np.asarray(rectifier_errors, dtype=np.float32) < 3.0)),
        "fused_under3_pct": float(np.mean(np.asarray(fused_errors, dtype=np.float32) < 3.0)),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Prediction rows written to: {prediction_csv}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
