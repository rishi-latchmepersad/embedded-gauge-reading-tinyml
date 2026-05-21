#!/usr/bin/env python3
"""Sweep rectifier crop bias settings against the exact V28 polar oracle.

This script keeps the offline crop question focused on the rectifier stage.
It asks a simple question: if we nudge the rectifier crop center in source
pixels and vary the crop expansion factor, can we recover the exact V28
behavior that the rectified crop boxes already show offline?
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

DEFAULT_MANIFEST: Final[Path] = PROJECT_ROOT / "data" / "hard_cases_plus_board30_valid_with_new6.csv"
DEFAULT_CAPTURE_ROOT: Final[Path] = PROJECT_ROOT / "data" / "captured_images"
DEFAULT_CROP_BOXES: Final[Path] = PROJECT_ROOT / "data" / "rectified_crop_boxes_v5_all.csv"
DEFAULT_RECTIFIER_MODEL: Final[Path] = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "mobilenetv2_rectifier_zoom_aug_v4"
    / "model.keras"
)
DEFAULT_WEIGHTS: Final[Path] = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "polar_vote_circular_v28"
    / "best_weights.weights.h5"
)
DEFAULT_OUTPUT_DIR: Final[Path] = REPO_ROOT / "tmp" / "polar_vote_v28_rectifier_bias_sweep"
DEFAULT_GAUGE_ID: Final[str] = "littlegood_home_temp_gauge_c"
DEFAULT_RECTIFIER_CROP_SCALES: Final[tuple[float, ...]] = (1.20, 1.25, 1.30, 1.35, 1.40)
DEFAULT_RECTIFIER_X_BIASES: Final[tuple[float, ...]] = (-16.0, -12.0, -8.0, -4.0, 0.0, 4.0)
DEFAULT_RECTIFIER_Y_BIASES: Final[tuple[float, ...]] = (-8.0, -4.0, 0.0, 4.0, 8.0)


@dataclass(frozen=True, slots=True)
class EvalItem:
    """One labeled capture and its target value."""

    image_path: Path
    value: float
    sample_weight: float


def _parse_float_list(raw_values: list[str]) -> tuple[float, ...]:
    """Parse a CLI float list while keeping the command line compact."""
    return tuple(float(value) for value in raw_values)


def _parse_args() -> argparse.Namespace:
    """Parse the sweep CLI."""
    parser = argparse.ArgumentParser(
        description="Sweep rectifier crop bias settings against exact V28."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--crop-boxes", type=Path, default=DEFAULT_CROP_BOXES)
    parser.add_argument("--rectifier-model", type=Path, default=DEFAULT_RECTIFIER_MODEL)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--gauge-id", type=str, default=DEFAULT_GAUGE_ID)
    parser.add_argument(
        "--rectifier-crop-scales",
        nargs="+",
        type=float,
        default=list(DEFAULT_RECTIFIER_CROP_SCALES),
        help="Crop scale candidates to test.",
    )
    parser.add_argument(
        "--rectifier-source-x-bias-pixels",
        nargs="+",
        type=float,
        default=list(DEFAULT_RECTIFIER_X_BIASES),
        help="Horizontal source-pixel crop shifts to test.",
    )
    parser.add_argument(
        "--rectifier-source-y-bias-pixels",
        nargs="+",
        type=float,
        default=list(DEFAULT_RECTIFIER_Y_BIASES),
        help="Vertical source-pixel crop shifts to test.",
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


def _predict_exact(
    model: tf.keras.Model,
    tensor: np.ndarray,
    gauge_spec: object,
) -> float:
    """Run the exact V28 model and decode the logits into Celsius."""
    logits = model.predict(tensor[None, ...], verbose=0)[0]
    return float(decode_circular_vote_logits(logits, gauge_spec))


def _apply_source_bias(
    crop_box_xyxy: tuple[float, float, float, float],
    *,
    source_width: int,
    source_height: int,
    bias_x_pixels: float,
    bias_y_pixels: float,
) -> tuple[float, float, float, float]:
    """Shift a crop box in source coordinates while preserving its size."""
    x0, y0, x1, y1 = crop_box_xyxy
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    center_x = 0.5 * (x0 + x1) + bias_x_pixels
    center_y = 0.5 * (y0 + y1) + bias_y_pixels
    new_x0 = max(0.0, center_x - 0.5 * width)
    new_y0 = max(0.0, center_y - 0.5 * height)
    new_x1 = min(float(source_width), new_x0 + width)
    new_y1 = min(float(source_height), new_y0 + height)
    if new_x1 <= new_x0 + 1.0:
        new_x1 = min(float(source_width), new_x0 + 1.0)
    if new_y1 <= new_y0 + 1.0:
        new_y1 = min(float(source_height), new_y0 + 1.0)
    return (new_x0, new_y0, new_x1, new_y1)


def main() -> None:
    """Sweep rectifier center bias and crop scale against the exact V28 oracle."""
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_items = _select_items(
        _load_manifest(args.manifest, args.capture_root),
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    crop_boxes = _load_crop_boxes(args.crop_boxes)
    rectifier = tf.keras.models.load_model(args.rectifier_model, compile=False, safe_mode=False)
    gauge_spec = load_gauge_specs()[args.gauge_id]
    polar_model = build_polar_vote_v28_model()
    polar_model.load_weights(str(args.weights))
    polar_model.trainable = False

    summary_rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None

    print(f"[SWEEP] Samples: {len(manifest_items)}", flush=True)
    print(f"[SWEEP] Rectifier: {args.rectifier_model}", flush=True)
    print(f"[SWEEP] Crop boxes: {args.crop_boxes}", flush=True)
    print(f"[SWEEP] Crop scales: {args.rectifier_crop_scales}", flush=True)
    print(f"[SWEEP] Source x biases: {args.rectifier_source_x_bias_pixels}", flush=True)
    print(f"[SWEEP] Source y biases: {args.rectifier_source_y_bias_pixels}", flush=True)

    for crop_scale in args.rectifier_crop_scales:
        for bias_x in args.rectifier_source_x_bias_pixels:
            for bias_y in args.rectifier_source_y_bias_pixels:
                abs_errors: list[float] = []
                print(
                    f"[SWEEP] start scale={crop_scale:.3f} bias_x={bias_x:.1f} bias_y={bias_y:.1f}",
                    flush=True,
                )
                for item in manifest_items:
                    source_image = load_rgb_image(item.image_path)
                    source_height, source_width = source_image.shape[:2]
                    full_frame_batch = np.expand_dims(
                        source_image.astype(np.float32) / 255.0,
                        axis=0,
                    )
                    rectifier_pred_raw = rectifier.predict(full_frame_batch, verbose=0)
                    if isinstance(rectifier_pred_raw, dict):
                        rectifier_box = np.asarray(rectifier_pred_raw["rectifier_box"]).reshape(-1)
                    else:
                        rectifier_box = np.asarray(rectifier_pred_raw).reshape(-1)

                    rectified_decision = decode_rectifier_crop_box(
                        rectifier_box,
                        source_width=source_width,
                        source_height=source_height,
                        rectifier_crop_scale=crop_scale,
                    )
                    crop_box = _apply_source_bias(
                        rectified_decision.crop_box_xyxy,
                        source_width=source_width,
                        source_height=source_height,
                        bias_x_pixels=bias_x,
                        bias_y_pixels=bias_y,
                    )
                    crop = build_training_style_polar_vote_float32(
                        source_image,
                        crop_box_xyxy=crop_box,
                        output_dim=args.image_size,
                        input_mode="rgb_edge6_vote7",
                        center_search_px=5,
                        center_mode="image_center",
                        gauge_spec=gauge_spec,
                    )
                    prediction = _predict_exact(polar_model, crop, gauge_spec)
                    abs_errors.append(abs(prediction - item.value))

                mae = float(np.mean(np.asarray(abs_errors, dtype=np.float32)))
                summary_rows.append(
                    {
                        "rectifier_crop_scale": crop_scale,
                        "rectifier_source_x_bias_pixels": bias_x,
                        "rectifier_source_y_bias_pixels": bias_y,
                        "mae": mae,
                        "samples": len(abs_errors),
                    }
                )
                if best_row is None or mae < float(best_row["mae"]):
                    best_row = summary_rows[-1]
                print(
                    f"[SWEEP] done scale={crop_scale:.3f} bias_x={bias_x:.1f} "
                    f"bias_y={bias_y:.1f} mae={mae:.6f}",
                    flush=True,
                )

    summary_path = args.output_dir / "rectifier_bias_sweep_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "crop_boxes": str(args.crop_boxes),
                "rectifier_model": str(args.rectifier_model),
                "weights": str(args.weights),
                "samples": len(manifest_items),
                "best": best_row,
                "results": summary_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[SWEEP] Summary written to {summary_path}", flush=True)
    if best_row is not None:
        print(
            "[SWEEP] Best: "
            f"scale={float(best_row['rectifier_crop_scale']):.3f} "
            f"bias_x={float(best_row['rectifier_source_x_bias_pixels']):.1f} "
            f"bias_y={float(best_row['rectifier_source_y_bias_pixels']):.1f} "
            f"mae={float(best_row['mae']):.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
