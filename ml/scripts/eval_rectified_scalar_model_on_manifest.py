"""Evaluate a rectified scalar Keras model on a labeled manifest.

This script applies rectifier-aligned crop boxes when available and falls back
to the canonical fixed crop when a row is missing from the crop-box CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.board_crop_compare import (  # noqa: E402
    load_rgb_image,
    resize_with_pad_rgb,
)
from embedded_gauge_reading_tinyml.models import (  # noqa: E402
    GaugeValueFromSweepDistribution,
)

TRAINING_CROP_X_MIN: float = 0.1027
TRAINING_CROP_Y_MIN: float = 0.2573
TRAINING_CROP_X_MAX: float = 0.7987
TRAINING_CROP_Y_MAX: float = 0.8071


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the evaluation job."""
    parser = argparse.ArgumentParser(
        description="Evaluate a rectified scalar Keras model on a manifest."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crop-boxes", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--polar-dualview-model",
        action="store_true",
        help="Evaluate a model that takes both full-frame and polar-unwrapped inputs.",
    )
    parser.add_argument(
        "--polar-only-model",
        action="store_true",
        help="Evaluate a model that takes a single polar-unwrapped input.",
    )
    parser.add_argument(
        "--polar-sweep-distribution-model",
        action="store_true",
        help="Evaluate a single-input polar sweep-distribution model.",
    )
    return parser.parse_args()


def _resolve_path(raw_path: str) -> Path:
    """Resolve one manifest path relative to the repo root when needed."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_crop_boxes(crop_boxes_path: Path) -> dict[str, tuple[float, float, float, float]]:
    """Load crop boxes keyed by resolved absolute image path."""
    boxes: dict[str, tuple[float, float, float, float]] = {}
    with crop_boxes_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "x0", "y0", "x1", "y1"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Crop-box CSV must contain {sorted(required)}: {crop_boxes_path}"
            )
        for row in reader:
            boxes[_resolve_path(row["image_path"]).as_posix()] = (
                float(row["x0"]),
                float(row["y0"]),
                float(row["x1"]),
                float(row["y1"]),
            )
    return boxes


def _load_manifest(manifest_path: Path) -> list[tuple[Path, float]]:
    """Load labeled rows from a manifest."""
    rows: list[tuple[Path, float]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        if "image_path" not in reader.fieldnames or "value" not in reader.fieldnames:
            raise ValueError("Manifest must contain image_path and value columns.")
        for row in reader:
            rows.append((_resolve_path(row["image_path"]), float(row["value"])))
    return rows


def _load_image(image_path: Path) -> np.ndarray:
    """Load one labeled image as RGB."""
    if image_path.suffix.lower() == ".yuv422":
        from embedded_gauge_reading_tinyml.board_crop_compare import (  # noqa: WPS433
            load_yuv422_capture_as_rgb,
        )

        return load_yuv422_capture_as_rgb(image_path)
    return load_rgb_image(image_path)


def _load_model(model_path: Path) -> tf.keras.Model:
    """Load the saved Keras model with the legacy MobileNetV2 preprocess helper."""
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
            "GaugeValueFromSweepDistribution": GaugeValueFromSweepDistribution,
        },
        compile=False,
        safe_mode=False,
    )


def _fixed_crop_box(width: int, height: int) -> tuple[float, float, float, float]:
    """Return the canonical fixed crop box in source-image coordinates."""
    return (
        TRAINING_CROP_X_MIN * float(width),
        TRAINING_CROP_Y_MIN * float(height),
        TRAINING_CROP_X_MAX * float(width),
        TRAINING_CROP_Y_MAX * float(height),
    )


def _prepare_input(
    image: np.ndarray,
    crop_box_xyxy: tuple[float, float, float, float],
    *,
    image_size: int,
) -> np.ndarray:
    """Crop and resize an image to the scalar model input shape."""
    cropped = resize_with_pad_rgb(image, crop_box_xyxy, image_size=image_size)
    return cropped.astype(np.float32) / 255.0


def _prepare_polar_input(
    image: np.ndarray,
    crop_box_xyxy: tuple[float, float, float, float],
    *,
    image_size: int,
) -> np.ndarray:
    """Crop and warp an image into polar coordinates for the dual-view model."""
    cropped = resize_with_pad_rgb(image, crop_box_xyxy, image_size=image_size)
    height, width = cropped.shape[:2]
    center = (float(width) * 0.5, float(height) * 0.5)
    max_radius = max(1.0, float(min(height, width)) * 0.5)
    polar = cv2.warpPolar(
        cropped,
        (image_size, image_size),
        center,
        max_radius,
        cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS,
    )
    if polar.shape[0] != image_size or polar.shape[1] != image_size:
        polar = cv2.resize(polar, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return polar.astype(np.float32) / 255.0


def main() -> None:
    """Evaluate the rectified scalar model and print aggregate hard-case metrics."""
    args = _parse_args()
    crop_boxes = _load_crop_boxes(args.crop_boxes)
    rows = _load_manifest(args.manifest)
    model = _load_model(args.model)

    abs_errors: list[float] = []
    hard_abs_errors: list[float] = []
    values: list[float] = []
    predictions: list[float] = []
    misses: int = 0

    print(f"[RECT-EVAL] Model:    {args.model}")
    print(f"[RECT-EVAL] Manifest: {args.manifest} ({len(rows)} rows)")
    print(f"[RECT-EVAL] Boxes:    {args.crop_boxes}")

    for image_path, true_value in rows:
        if not image_path.exists():
            misses += 1
            continue

        image = _load_image(image_path)
        height, width = image.shape[:2]
        crop_box = crop_boxes.get(
            image_path.as_posix(),
            _fixed_crop_box(width, height),
        )
        model_input = _prepare_input(image, crop_box, image_size=args.image_size)
        if args.polar_dualview_model:
            polar_input = _prepare_polar_input(
                image,
                crop_box,
                image_size=args.image_size,
            )
            pred_raw = model.predict(
                {
                    "full_image": model_input[None, ...],
                    "polar_image": polar_input[None, ...],
                },
                verbose=0,
            )
        elif args.polar_only_model or args.polar_sweep_distribution_model:
            polar_input = _prepare_polar_input(
                image,
                crop_box,
                image_size=args.image_size,
            )
            pred_raw = model.predict(
                {"polar_image": polar_input[None, ...]},
                verbose=0,
            )
        else:
            pred_raw = model.predict(model_input[None, ...], verbose=0)
        if isinstance(pred_raw, dict):
            pred_raw = pred_raw.get("gauge_value", next(iter(pred_raw.values())))
        pred_value = float(np.asarray(pred_raw).reshape(-1)[0])
        abs_error = abs(pred_value - true_value)

        abs_errors.append(abs_error)
        values.append(true_value)
        predictions.append(pred_value)
        if true_value <= -20.0 or true_value >= 40.0:
            hard_abs_errors.append(abs_error)

        print(
            f"{image_path.name}: true={true_value:6.2f} pred={pred_value:7.2f} "
            f"abs_err={abs_error:6.2f}",
            flush=True,
        )

    errors = np.asarray(abs_errors, dtype=np.float32)
    print(f"n={len(errors)}")
    print(f"misses={misses}")
    print(f"mae={float(np.mean(errors)):.4f}")
    print(f"rmse={float(np.sqrt(np.mean(np.square(errors)))):.4f}")
    print(f"median_abs_error={float(np.median(errors)):.4f}")
    print(f"pct_under_5c={float(np.mean(errors < 5.0)):.4f}")
    print(f"pct_under_3c={float(np.mean(errors < 3.0)):.4f}")
    print(f"pct_under_1c={float(np.mean(errors < 1.0)):.4f}")
    print(f"hard_mae={float(np.mean(np.asarray(hard_abs_errors, dtype=np.float32))) if hard_abs_errors else 0.0:.4f}")
    if len(values) > 1:
        corr = float(np.corrcoef(np.asarray(values, dtype=np.float32), np.asarray(predictions, dtype=np.float32))[0, 1])
        print(f"correlation={corr:.4f}")


if __name__ == "__main__":
    main()
