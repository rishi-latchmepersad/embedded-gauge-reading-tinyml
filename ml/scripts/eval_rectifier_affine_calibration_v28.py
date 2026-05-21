#!/usr/bin/env python3
"""Fit and evaluate an affine calibration for the rectifier crop decoder.

The existing rectifier predicts a normalized crop box, but the exact V28 model
is sensitive to tiny crop shifts. This script asks a narrow question:

* can we learn a tiny affine map from the rectifier outputs to the known-good
  rectified crop boxes, and does that recover the offline V28 oracle better than
  the raw decoder?

The result stays board-friendly because the learned post-processing is just a
matrix multiply and bias.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import os
import random
import sys
from typing import Final

import numpy as np
import tensorflow as tf

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPO_ROOT: Final[Path] = PROJECT_ROOT.parent
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "tmp" / "matplotlib"))

from embedded_gauge_reading_tinyml.board_pipeline import decode_rectifier_crop_box  # noqa: E402
from embedded_gauge_reading_tinyml.board_crop_compare import load_rgb_image  # noqa: E402
from embedded_gauge_reading_tinyml.firmware_preprocessing import (  # noqa: E402
    build_training_style_polar_vote_float32,
    decode_circular_vote_logits,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs  # noqa: E402
from embedded_gauge_reading_tinyml.polar_vote_v28 import (  # noqa: E402
    build_polar_vote_v28_model,
)

DEFAULT_FIT_MANIFEST: Final[Path] = PROJECT_ROOT / "data" / "full_labelled_plus_board30_valid_with_new5.csv"
DEFAULT_EVAL_MANIFEST: Final[Path] = PROJECT_ROOT / "data" / "hard_cases_plus_board30_valid_with_new6.csv"
DEFAULT_CAPTURE_ROOT: Final[Path] = PROJECT_ROOT / "data" / "captured_images"
DEFAULT_CROP_BOXES: Final[Path] = PROJECT_ROOT / "data" / "rectified_crop_boxes_v5_all.csv"
DEFAULT_RECTIFIER_MODEL: Final[Path] = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "mobilenetv2_rectifier_rectified_boxes_v1"
    / "model.keras"
)
DEFAULT_WEIGHTS: Final[Path] = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "polar_vote_circular_v28"
    / "best_weights.weights.h5"
)
DEFAULT_OUTPUT_DIR: Final[Path] = REPO_ROOT / "tmp" / "rectifier_affine_v28_eval"
DEFAULT_GAUGE_ID: Final[str] = "littlegood_home_temp_gauge_c"


@dataclass(frozen=True, slots=True)
class EvalItem:
    """One labeled capture and its target value."""

    image_path: Path
    value: float
    sample_weight: float


def _parse_args() -> argparse.Namespace:
    """Parse the calibration CLI."""
    parser = argparse.ArgumentParser(
        description="Fit an affine correction for rectifier crops and score exact V28."
    )
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--crop-boxes", type=Path, default=DEFAULT_CROP_BOXES)
    parser.add_argument("--rectifier-model", type=Path, default=DEFAULT_RECTIFIER_MODEL)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-fit-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--shuffle-fit", action="store_true")
    parser.add_argument("--shuffle-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--gauge-id", type=str, default=DEFAULT_GAUGE_ID)
    return parser.parse_args()


def _resolve_repo_path(raw_path: str) -> Path:
    """Resolve a manifest path against the repository root when needed."""
    normalized = raw_path.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _is_under_root(path: Path, root: Path) -> bool:
    """Return True when a path resolves inside a root directory."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_manifest(manifest_path: Path, capture_root: Path) -> list[EvalItem]:
    """Load labeled rows and keep only captures under the requested root."""
    items: list[EvalItem] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = _resolve_repo_path(str(row["image_path"]))
            if not _is_under_root(image_path, capture_root):
                continue
            items.append(
                EvalItem(
                    image_path=image_path,
                    value=float(row["value"]),
                    sample_weight=float(row.get("sample_weight", 1.0) or 1.0),
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
    """Deterministically select a subset of examples."""
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
    """Load rectified crop boxes keyed by resolved image path."""
    boxes: dict[str, tuple[float, float, float, float]] = {}
    with crop_boxes_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "x0", "y0", "x1", "y1"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Crop-box CSV must contain {sorted(required)}: {crop_boxes_path}"
            )
        for row in reader:
            boxes[_resolve_repo_path(row["image_path"]).as_posix()] = (
                float(row["x0"]),
                float(row["y0"]),
                float(row["x1"]),
                float(row["y1"]),
            )
    return boxes


def _quantize_unit_float_to_int8(tensor: np.ndarray) -> np.ndarray:
    """Match the firmware quantization rule q = round(x * 255) - 128."""
    quantized = np.floor(np.asarray(tensor, dtype=np.float32) * 255.0 + 0.5) - 128.0
    return np.clip(quantized, -128.0, 127.0).astype(np.int8)


def _rectifier_features(model: tf.keras.Model, image: np.ndarray) -> np.ndarray:
    """Run the rectifier and return its raw 4D crop prediction."""
    batch = np.expand_dims(image.astype(np.float32) / 255.0, axis=0)
    pred = model.predict(batch, verbose=0)
    if isinstance(pred, dict):
        pred = pred["rectifier_box"]
    return np.asarray(pred, dtype=np.float32).reshape(-1)


def _fit_affine(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a least-squares affine map from 4D features to 4D targets."""
    design = np.concatenate(
        [features.astype(np.float32), np.ones((features.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    coef, *_ = np.linalg.lstsq(design, targets.astype(np.float32), rcond=None)
    matrix = coef[:4, :].T
    bias = coef[4, :]
    return matrix, bias


def _apply_affine(
    features: np.ndarray,
    matrix: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Apply the fitted affine transform and clamp to [0, 1]."""
    pred = features @ matrix.T + bias
    return np.clip(pred, 0.0, 1.0)


def _predict_exact(
    model: tf.keras.Model,
    tensor: np.ndarray,
    gauge_spec: object,
) -> float:
    """Run the exact V28 model and decode the logits into Celsius."""
    logits = model.predict(tensor[None, ...], verbose=0)[0]
    return float(decode_circular_vote_logits(logits, gauge_spec))


def _source_crop_from_normalized_box(
    normalized_box: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Convert normalized x0/y0/x1/y1 values back into source-image pixels."""
    x0 = float(normalized_box[0]) * float(width)
    y0 = float(normalized_box[1]) * float(height)
    x1 = float(normalized_box[2]) * float(width)
    y1 = float(normalized_box[3]) * float(height)
    if x1 <= x0 + 1.0:
        x1 = min(float(width), x0 + 1.0)
    if y1 <= y0 + 1.0:
        y1 = min(float(height), y0 + 1.0)
    return (x0, y0, x1, y1)


def _evaluate_affine(
    items: list[EvalItem],
    *,
    rectifier: tf.keras.Model,
    polar_model: tf.keras.Model,
    matrix: np.ndarray | None,
    bias: np.ndarray | None,
    crop_boxes: dict[str, tuple[float, float, float, float]],
    gauge_spec: object,
    image_size: int,
    label: str,
) -> dict[str, float]:
    """Evaluate one decoder variant on a manifest."""
    errors: list[float] = []
    for item in items:
        source_image = load_rgb_image(item.image_path)
        source_height, source_width = source_image.shape[:2]
        full_frame_batch = np.expand_dims(source_image.astype(np.float32) / 255.0, axis=0)
        rectifier_pred_raw = rectifier.predict(full_frame_batch, verbose=0)
        if isinstance(rectifier_pred_raw, dict):
            rectifier_box = np.asarray(rectifier_pred_raw["rectifier_box"], dtype=np.float32).reshape(-1)
        else:
            rectifier_box = np.asarray(rectifier_pred_raw, dtype=np.float32).reshape(-1)

        if matrix is not None and bias is not None:
            normalized_box = _apply_affine(rectifier_box[None, :], matrix, bias)[0]
            selected_crop = _source_crop_from_normalized_box(
                normalized_box,
                width=source_width,
                height=source_height,
            )
        else:
            rectified_decision = decode_rectifier_crop_box(
                rectifier_box,
                source_width=source_width,
                source_height=source_height,
            )
            selected_crop = rectified_decision.crop_box_xyxy

        crop = build_training_style_polar_vote_float32(
            source_image,
            crop_box_xyxy=selected_crop,
            output_dim=image_size,
            input_mode="rgb_edge6_vote7",
            center_search_px=5,
            center_mode="image_center",
            gauge_spec=gauge_spec,
        )
        prediction = _predict_exact(polar_model, crop, gauge_spec)
        errors.append(abs(prediction - item.value))
    arr = np.asarray(errors, dtype=np.float32)
    return {
        f"{label}_mae": float(np.mean(arr)),
        f"{label}_samples": float(len(arr)),
    }


def main() -> None:
    """Fit the affine correction and report exact V28 performance."""
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fit_items = _select_items(
        _load_manifest(args.fit_manifest, args.capture_root),
        max_samples=args.max_fit_samples,
        shuffle=args.shuffle_fit,
        seed=args.seed,
    )
    eval_items = _select_items(
        _load_manifest(args.eval_manifest, args.capture_root),
        max_samples=args.max_eval_samples,
        shuffle=args.shuffle_eval,
        seed=args.seed,
    )
    crop_boxes = _load_crop_boxes(args.crop_boxes)
    rectifier = tf.keras.models.load_model(args.rectifier_model, compile=False, safe_mode=False)
    gauge_spec = load_gauge_specs()[args.gauge_id]
    polar_model = build_polar_vote_v28_model()
    polar_model.load_weights(str(args.weights))
    polar_model.trainable = False

    fit_features: list[np.ndarray] = []
    fit_targets: list[np.ndarray] = []
    for item in fit_items:
        source_image = load_rgb_image(item.image_path)
        source_height, source_width = source_image.shape[:2]
        crop_box = crop_boxes.get(
            item.image_path.as_posix(),
            (0.0, 0.0, float(source_width), float(source_height)),
        )
        feature = _rectifier_features(rectifier, source_image)
        fit_features.append(feature)
        fit_targets.append(
            np.asarray(
                (
                    crop_box[0] / float(source_width),
                    crop_box[1] / float(source_height),
                    crop_box[2] / float(source_width),
                    crop_box[3] / float(source_height),
                ),
                dtype=np.float32,
            )
        )

    features = np.stack(fit_features, axis=0)
    targets = np.stack(fit_targets, axis=0)
    matrix, bias = _fit_affine(features, targets)

    fit_pred = _apply_affine(features, matrix, bias)
    fit_mae = float(np.mean(np.abs(fit_pred - targets)))

    raw_metrics = _evaluate_affine(
        eval_items,
        rectifier=rectifier,
        polar_model=polar_model,
        matrix=None,
        bias=None,
        crop_boxes=crop_boxes,
        gauge_spec=gauge_spec,
        image_size=args.image_size,
        label="raw",
    )
    affine_metrics = _evaluate_affine(
        eval_items,
        rectifier=rectifier,
        polar_model=polar_model,
        matrix=matrix,
        bias=bias,
        crop_boxes=crop_boxes,
        gauge_spec=gauge_spec,
        image_size=args.image_size,
        label="affine",
    )

    summary = {
        "fit_manifest": str(args.fit_manifest),
        "eval_manifest": str(args.eval_manifest),
        "crop_boxes": str(args.crop_boxes),
        "rectifier_model": str(args.rectifier_model),
        "weights": str(args.weights),
        "fit_samples": len(fit_items),
        "eval_samples": len(eval_items),
        "fit_mae_rectified_boxes": fit_mae,
        "raw_eval_mae": raw_metrics["raw_mae"],
        "affine_eval_mae": affine_metrics["affine_mae"],
        "matrix": matrix.tolist(),
        "bias": bias.tolist(),
    }
    summary_path = args.output_dir / "rectifier_affine_v28_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[AFFINE] Fit samples: {len(fit_items)}", flush=True)
    print(f"[AFFINE] Eval samples: {len(eval_items)}", flush=True)
    print(f"[AFFINE] Fit box MAE: {fit_mae:.6f}", flush=True)
    print(f"[AFFINE] Raw exact-V28 MAE: {raw_metrics['raw_mae']:.6f}", flush=True)
    print(f"[AFFINE] Affine exact-V28 MAE: {affine_metrics['affine_mae']:.6f}", flush=True)
    print(f"[AFFINE] Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
