"""Generate crop-box CSV rows from a trained MobileNetV2 + CoordConv localizer.

The goal is to refresh the downstream crop manifest with a stronger geometry
model so later scalar training sees cleaner, more board-consistent crops.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import keras
import numpy as np

# Add `ml/src` to sys.path so the script works before `poetry install`.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.board_crop_compare import resize_with_pad_rgb
from embedded_gauge_reading_tinyml.board_pipeline import decode_obb_crop_box
from embedded_gauge_reading_tinyml.models import build_mobilenetv2_obb_center_model
from embedded_gauge_reading_tinyml.firmware_preprocessing import load_capture_image


DEFAULT_MANIFEST: Path = PROJECT_ROOT / "data" / "merged_geometry_board_manifest.csv"
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_IMAGE_SIZE: int = 320
DEFAULT_MOBILENET_ALPHA: float = 0.50
DEFAULT_HEAD_UNITS_1: int = 384
DEFAULT_HEAD_UNITS_2: int = 160
DEFAULT_HEAD_DROPOUT_1: float = 0.30
DEFAULT_HEAD_DROPOUT_2: float = 0.20


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for crop-box generation."""
    parser = argparse.ArgumentParser(
        description="Generate OBB crop boxes from a trained localizer checkpoint."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a Keras .keras checkpoint produced by train_obb_center_mobilenetv2.py.",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="CSV manifest with image_path, source_width, source_height, and temperature_c.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Where to write the refreshed crop-box CSV.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Model input size used during preprocessing and OBB decode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of images to run through the localizer per batch.",
    )
    parser.add_argument(
        "--quality-flags",
        nargs="*",
        default=["clean", "manual"],
        help="Keep only rows whose quality_flag matches one of these values.",
    )
    parser.add_argument(
        "--mobilenet-alpha",
        type=float,
        default=DEFAULT_MOBILENET_ALPHA,
        help="MobileNetV2 width multiplier used by the localizer checkpoint.",
    )
    parser.add_argument(
        "--head-units-1",
        type=int,
        default=DEFAULT_HEAD_UNITS_1,
        help="Width of the first dense head layer.",
    )
    parser.add_argument(
        "--head-units-2",
        type=int,
        default=DEFAULT_HEAD_UNITS_2,
        help="Width of the second dense head layer.",
    )
    parser.add_argument(
        "--head-dropout-1",
        type=float,
        default=DEFAULT_HEAD_DROPOUT_1,
        help="Dropout after the first dense head layer.",
    )
    parser.add_argument(
        "--head-dropout-2",
        type=float,
        default=DEFAULT_HEAD_DROPOUT_2,
        help="Dropout after the second dense head layer.",
    )
    return parser.parse_args()


def _resolve_image_path(image_path: str) -> Path:
    """Resolve a manifest image path against the repo root."""
    candidate = Path(image_path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _load_manifest_rows(
    manifest_path: Path,
    *,
    quality_flags: set[str],
) -> list[dict[str, str]]:
    """Read and filter the manifest rows we want to refresh."""
    rows: list[dict[str, str]] = []
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            quality_flag = row.get("quality_flag", "").strip()
            if quality_flag and quality_flags and quality_flag not in quality_flags:
                continue
            rows.append(row)
    return rows


def _load_image_batch(
    rows: list[dict[str, str]],
    *,
    image_size: int,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Load a small batch of source images and resize them for inference."""
    batches: list[np.ndarray] = []
    kept_rows: list[dict[str, str]] = []
    for row in rows:
        image_path = _resolve_image_path(row["image_path"])
        source_width = int(float(row["source_width"]))
        source_height = int(float(row["source_height"]))
        source_image, _kind = load_capture_image(
            image_path,
            image_width=source_width,
            image_height=source_height,
        )
        full_frame = resize_with_pad_rgb(
            source_image,
            (0.0, 0.0, float(source_width), float(source_height)),
            image_size=image_size,
        )
        batches.append(full_frame.astype(np.float32) / 255.0)
        kept_rows.append(row)
    if not batches:
        return np.zeros((0, image_size, image_size, 3), dtype=np.float32), kept_rows
    return np.stack(batches, axis=0), kept_rows


def _predict_batch(
    model: keras.Model,
    batch: np.ndarray,
) -> np.ndarray:
    """Run one inference batch and return the obb_params tensor."""
    outputs: Any = model.predict(batch, verbose=0)
    if isinstance(outputs, dict):
        obb = outputs["obb_params"]
    else:
        obb = outputs[0]
    return np.asarray(obb, dtype=np.float32)


def main() -> None:
    """Generate refreshed crop boxes from the localizer checkpoint."""
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if not args.input_manifest.is_file():
        raise FileNotFoundError(f"Input manifest not found: {args.input_manifest}")

    quality_flags = {flag.strip() for flag in args.quality_flags if flag.strip()}
    rows = _load_manifest_rows(args.input_manifest, quality_flags=quality_flags)
    if not rows:
        raise RuntimeError("No manifest rows matched the requested filters.")

    # Rebuild the checkpoint architecture instead of deserializing the Lambda
    # layer config directly. The saved model's coordinate Lambda loses its `tf`
    # binding during deserialization, but its weights are still perfectly valid.
    model = build_mobilenetv2_obb_center_model(
        image_height=args.image_size,
        image_width=args.image_size,
        pretrained=False,
        backbone_trainable=False,
        alpha=args.mobilenet_alpha,
        head_units_1=args.head_units_1,
        head_units_2=args.head_units_2,
        head_dropout_1=args.head_dropout_1,
        head_dropout_2=args.head_dropout_2,
    )
    model.load_weights(args.model_path)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "value",
                "x0",
                "y0",
                "x1",
                "y1",
                "accepted",
                "fallback_reason",
                "center_x_norm",
                "center_y_norm",
                "box_w_norm",
                "box_h_norm",
                "theta_deg",
                "source_width",
                "source_height",
                "quality_flag",
            ],
        )
        writer.writeheader()

        total = len(rows)
        batch_size = max(1, int(args.batch_size))
        for start in range(0, total, batch_size):
            batch_rows = rows[start : start + batch_size]
            batch, kept_rows = _load_image_batch(batch_rows, image_size=args.image_size)
            predictions = _predict_batch(model, batch)

            for row, obb in zip(kept_rows, predictions, strict=True):
                source_width = int(float(row["source_width"]))
                source_height = int(float(row["source_height"]))
                decision = decode_obb_crop_box(
                    obb,
                    source_width=source_width,
                    source_height=source_height,
                    input_size=args.image_size,
                )
                x0, y0, x1, y1 = decision.crop_box_xyxy
                writer.writerow(
                    {
                        "image_path": row["image_path"],
                        "value": row.get("temperature_c", ""),
                        "x0": f"{x0:.2f}",
                        "y0": f"{y0:.2f}",
                        "x1": f"{x1:.2f}",
                        "y1": f"{y1:.2f}",
                        "accepted": str(decision.accepted),
                        "fallback_reason": decision.fallback_reason or "",
                        "center_x_norm": f"{decision.details.get('center_x', 0.0):.6f}",
                        "center_y_norm": f"{decision.details.get('center_y', 0.0):.6f}",
                        "box_w_norm": f"{decision.details.get('box_w', 0.0):.6f}",
                        "box_h_norm": f"{decision.details.get('box_h', 0.0):.6f}",
                        "theta_deg": f"{decision.details.get('theta_deg', 0.0):.3f}",
                        "source_width": source_width,
                        "source_height": source_height,
                        "quality_flag": row.get("quality_flag", ""),
                    }
                )

            if (start // batch_size) % 10 == 0:
                print(f"[OBB-CROP] processed {min(start + batch_size, total)}/{total}", flush=True)

    print(f"[OBB-CROP] wrote crop boxes to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
