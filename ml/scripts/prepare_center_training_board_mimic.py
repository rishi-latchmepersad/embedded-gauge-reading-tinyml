#!/usr/bin/env python3
"""Build a board-mimic center-detector dataset from the merged label manifest.

This generator mirrors the live board path more closely than the older
center-training crops:
1. Load a labelled full-frame source image from the merged geometry manifest.
2. Resize/pad that source to the 224x224 board canvas.
3. Run the current OBB model on the full-frame canvas.
4. Decode the OBB crop with the same geometry thresholds as firmware.
5. Crop the 155x123 center-detector window around the OBB center, or fall
   back to the fixed training crop when the OBB looks bad.
6. Convert the source-space center label into the crop-normalized label that
   the MobileNetV2 center model expects.

The saved PNGs are intentionally kept as preview artifacts.  The training
loaders now reconstruct the crop from ``source_path`` and apply the
board-style preprocess on the fly so the model sees the same crop geometry
the firmware uses.

The result is a small, traceable dataset that mimics the board AI pipeline
while still letting us train offline on the original full-frame labels.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedded_gauge_reading_tinyml.board_pipeline import (  # noqa: E402
    _quantize_input,
    decode_obb_crop_box,
    load_model_session,
)

MANIFEST_PATH = PROJECT_ROOT / "data" / "merged_geometry_board_manifest.csv"
OBB_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "deployment"
    / "prod_model_v0.3_obb_int8"
    / "model_int8.tflite"
)
OUT_DIR = PROJECT_ROOT / "data" / "center_training_board_mimic"
OUT_IMAGES = OUT_DIR / "images"

FULL_FRAME_SIZE = 224
CD_WIDTH = 155
CD_HEIGHT = 123

TRAINING_CROP_X_MIN_RATIO = 0.1027
TRAINING_CROP_Y_MIN_RATIO = 0.2573
TRAINING_CROP_X_MAX_RATIO = 0.7987
TRAINING_CROP_Y_MAX_RATIO = 0.8071

KEEP_QUALITY_FLAGS = {"clean", "review"}
SOURCE_KIND = Literal["pxl", "capture"]
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Drop only the obvious failure cases.  We want to keep dim but structured
# low-light examples, but exclude the blank / near-blank captures that can
# sneak past OBB because the network still finds a plausible box.
BLANK_FRAME_MEAN_THRESHOLD = 18.5
BLANK_FRAME_STD_THRESHOLD = 2.0
BLANK_FRAME_DARK_FRACTION_THRESHOLD = 0.99
BLANK_LUMA_LEVEL = 16


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """One manifest row with the fields needed for board-mimic training."""

    image_path: Path
    source_width: int
    source_height: int
    center_x_source: float
    center_y_source: float
    temperature_c: float
    split: str
    quality_flag: str
    source_kind: SOURCE_KIND


def _load_manifest_rows() -> list[ManifestRow]:
    """Load labelled source images from the merged geometry manifest."""
    rows: list[ManifestRow] = []

    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            quality_flag = str(raw.get("quality_flag", "clean")).strip().lower()
            if quality_flag not in KEEP_QUALITY_FLAGS:
                continue

            image_path_raw = str(raw.get("image_path", "")).strip()
            if not image_path_raw:
                continue

            image_path = REPO_ROOT / image_path_raw
            if not image_path.exists():
                continue

            source_kind: SOURCE_KIND = "capture" if "captured_images" in image_path_raw else "pxl"
            rows.append(
                ManifestRow(
                    image_path=image_path,
                    source_width=int(float(raw["source_width"])),
                    source_height=int(float(raw["source_height"])),
                    center_x_source=float(raw["center_x_source"]),
                    center_y_source=float(raw["center_y_source"]),
                    temperature_c=float(raw["temperature_c"]),
                    split=str(raw.get("split", "train")).strip().lower(),
                    quality_flag=quality_flag,
                    source_kind=source_kind,
                )
            )

    rows.sort(key=lambda row: row.image_path.as_posix())
    return rows


def _assign_stratified_splits(rows: list[ManifestRow]) -> list[ManifestRow]:
    """Assign train/val/test splits per source kind so holdouts cover both domains."""
    grouped: dict[SOURCE_KIND, list[ManifestRow]] = {"pxl": [], "capture": []}
    for row in rows:
        grouped[row.source_kind].append(row)

    assigned: list[ManifestRow] = []
    for source_kind, group_rows in grouped.items():
        if not group_rows:
            continue

        ordered_rows = sorted(group_rows, key=lambda row: row.image_path.as_posix())
        rng = np.random.default_rng(42 if source_kind == "pxl" else 43)
        order = np.arange(len(ordered_rows))
        rng.shuffle(order)
        shuffled_rows = [ordered_rows[index] for index in order]

        num_val = max(1, int(round(len(shuffled_rows) * VAL_FRACTION)))
        num_test = max(1, int(round(len(shuffled_rows) * TEST_FRACTION)))
        if num_val + num_test >= len(shuffled_rows):
            num_val = max(1, len(shuffled_rows) // 3)
            num_test = max(1, len(shuffled_rows) // 3)
        num_train = max(1, len(shuffled_rows) - num_val - num_test)
        if num_train + num_val + num_test > len(shuffled_rows):
            num_train = len(shuffled_rows) - num_val - num_test

        split_rows = (
            [replace(row, split="train") for row in shuffled_rows[:num_train]]
            + [replace(row, split="val") for row in shuffled_rows[num_train : num_train + num_val]]
            + [replace(row, split="test") for row in shuffled_rows[num_train + num_val :]]
        )
        assigned.extend(split_rows)

    assigned.sort(key=lambda row: row.image_path.as_posix())
    return assigned


def _resize_with_pad_geometry(
    source_width: int,
    source_height: int,
    output_size: int = FULL_FRAME_SIZE,
) -> tuple[int, int, int, int]:
    """Return the resized size plus integer pad offsets for a letterboxed image."""
    scale = min(float(output_size) / float(source_width), float(output_size) / float(source_height))
    resized_width = max(1, int(round(float(source_width) * scale)))
    resized_height = max(1, int(round(float(source_height) * scale)))
    offset_x = max(0, (output_size - resized_width) // 2)
    offset_y = max(0, (output_size - resized_height) // 2)
    return resized_width, resized_height, offset_x, offset_y


def _load_rgb_image(image_path: Path) -> np.ndarray:
    """Load any source image as an RGB uint8 array."""
    with Image.open(image_path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


def _prepare_full_frame(image_rgb: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    """Resize/pad the source image into the 224x224 board full-frame canvas."""
    source_height, source_width = image_rgb.shape[:2]
    resized_width, resized_height, offset_x, offset_y = _resize_with_pad_geometry(
        source_width,
        source_height,
        FULL_FRAME_SIZE,
    )

    if source_width == FULL_FRAME_SIZE and source_height == FULL_FRAME_SIZE:
        return np.ascontiguousarray(image_rgb), resized_width, resized_height, offset_x, offset_y

    resized = Image.fromarray(image_rgb, mode="RGB").resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    canvas = np.zeros((FULL_FRAME_SIZE, FULL_FRAME_SIZE, 3), dtype=np.uint8)
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = np.asarray(resized, dtype=np.uint8)
    return canvas, resized_width, resized_height, offset_x, offset_y


def _project_source_point_to_full_frame(
    x_source: float,
    y_source: float,
    source_width: int,
    source_height: int,
) -> tuple[float, float]:
    """Map a source-space label into the 224x224 full-frame canvas."""
    resized_width, resized_height, offset_x, offset_y = _resize_with_pad_geometry(
        source_width,
        source_height,
        FULL_FRAME_SIZE,
    )
    full_x = (x_source * (float(resized_width) / float(source_width))) + float(offset_x)
    full_y = (y_source * (float(resized_height) / float(source_height))) + float(offset_y)
    return full_x, full_y


def _build_cd_crop(
    full_frame: np.ndarray,
    crop_center_x: float,
    crop_center_y: float,
) -> tuple[np.ndarray, int, int]:
    """Build the firmware's 155x123 training crop and resize it to 224x224."""
    crop_x_min = int(round(crop_center_x - (CD_WIDTH * 0.5)))
    crop_y_min = int(round(crop_center_y - (CD_HEIGHT * 0.5)))
    crop_x_min = max(0, min(crop_x_min, FULL_FRAME_SIZE - CD_WIDTH))
    crop_y_min = max(0, min(crop_y_min, FULL_FRAME_SIZE - CD_HEIGHT))

    crop = full_frame[crop_y_min : crop_y_min + CD_HEIGHT, crop_x_min : crop_x_min + CD_WIDTH]
    resized_width, resized_height, offset_x, offset_y = _resize_with_pad_geometry(
        CD_WIDTH,
        CD_HEIGHT,
        FULL_FRAME_SIZE,
    )
    resized = Image.fromarray(crop, mode="RGB").resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    canvas = np.zeros((FULL_FRAME_SIZE, FULL_FRAME_SIZE, 3), dtype=np.uint8)
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = np.asarray(resized, dtype=np.uint8)
    return canvas, crop_x_min, crop_y_min


def _project_full_frame_point_to_cd_crop(
    full_x: float,
    full_y: float,
    crop_x_min: int,
    crop_y_min: int,
) -> tuple[float, float]:
    """Convert a full-frame point into normalized CD-crop coordinates."""
    resized_width, resized_height, offset_x, offset_y = _resize_with_pad_geometry(
        CD_WIDTH,
        CD_HEIGHT,
        FULL_FRAME_SIZE,
    )
    crop_scale_x = float(resized_width) / float(CD_WIDTH)
    crop_scale_y = float(resized_height) / float(CD_HEIGHT)
    padded_x = ((full_x - float(crop_x_min)) * crop_scale_x) + float(offset_x)
    padded_y = ((full_y - float(crop_y_min)) * crop_scale_y) + float(offset_y)
    return padded_x / float(FULL_FRAME_SIZE), padded_y / float(FULL_FRAME_SIZE)


def _training_crop_box() -> tuple[int, int, int, int]:
    """Return the fixed training crop used when the OBB crop looks bad."""
    x_min = int(float(FULL_FRAME_SIZE) * TRAINING_CROP_X_MIN_RATIO)
    y_min = int(float(FULL_FRAME_SIZE) * TRAINING_CROP_Y_MIN_RATIO)
    width = max(
        1,
        int(float(FULL_FRAME_SIZE) * (TRAINING_CROP_X_MAX_RATIO - TRAINING_CROP_X_MIN_RATIO)),
    )
    height = max(
        1,
        int(float(FULL_FRAME_SIZE) * (TRAINING_CROP_Y_MAX_RATIO - TRAINING_CROP_Y_MIN_RATIO)),
    )
    return x_min, y_min, width, height


def _label_in_unit_interval(x_norm: float, y_norm: float, *, epsilon: float = 1e-4) -> bool:
    """Return True when the label fits inside the sigmoid output range."""
    return (
        (-epsilon <= x_norm <= 1.0 + epsilon)
        and (-epsilon <= y_norm <= 1.0 + epsilon)
        and np.isfinite(x_norm)
        and np.isfinite(y_norm)
    )


def _summarize_source_luma(image_rgb: np.ndarray) -> dict[str, float]:
    """Compute the basic luminance stats we use to reject blank captures."""
    luma = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    dark_fraction = float(np.mean(luma <= BLANK_LUMA_LEVEL))
    return {
        "mean": float(luma.mean()),
        "std": float(luma.std()),
        "min": float(luma.min()),
        "max": float(luma.max()),
        "dark_fraction": dark_fraction,
    }


def _should_skip_source(image_rgb: np.ndarray) -> tuple[bool, dict[str, float]]:
    """Reject obvious blank / failed captures before we spend OBB cycles on them."""
    stats = _summarize_source_luma(image_rgb)
    blankish = (
        stats["mean"] <= BLANK_FRAME_MEAN_THRESHOLD
        and stats["std"] <= BLANK_FRAME_STD_THRESHOLD
    )
    near_uniform_dark = stats["dark_fraction"] >= BLANK_FRAME_DARK_FRACTION_THRESHOLD
    return (blankish or near_uniform_dark, stats)


def main() -> None:
    """Generate the board-mimic center training crops."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    rows = _assign_stratified_splits(_load_manifest_rows())
    print(f"Loaded {len(rows)} labeled source images from {MANIFEST_PATH.name}")

    obb_session = load_model_session(OBB_MODEL_PATH, "tflite")
    print(f"Loaded OBB model: {OBB_MODEL_PATH}")

    inp = obb_session.input_details
    out = obb_session.output_details
    if inp is None or out is None:
        raise RuntimeError("OBB session is missing tensor metadata.")

    entries: list[dict[str, object]] = []
    rejected_entries: list[dict[str, object]] = []
    accepted_obb = 0
    used_fixed_crop = 0
    rejected = 0
    skipped_blank = 0

    for index, row in enumerate(rows, start=1):
        source_rgb = _load_rgb_image(row.image_path)

        # The OBB model will happily return a plausible box on a blank frame, so
        # we screen out obvious failures before building a training example.
        skip_source, source_luma = _should_skip_source(source_rgb)
        if skip_source:
            skipped_blank += 1
            rejected_entries.append(
                {
                    "image_path": str(row.image_path.relative_to(REPO_ROOT)),
                    "source_kind": row.source_kind,
                    "quality_flag": row.quality_flag,
                    "reason": "blank_or_failed_capture",
                    "source_luma_mean": source_luma["mean"],
                    "source_luma_std": source_luma["std"],
                    "source_luma_min": source_luma["min"],
                    "source_luma_max": source_luma["max"],
                    "source_dark_fraction": source_luma["dark_fraction"],
                }
            )
            if skipped_blank <= 5:
                print(
                    f"[{index}/{len(rows)}] SKIP {row.image_path.name} "
                    f"reason=blank_or_failed_capture mean={source_luma['mean']:.1f} "
                    f"std={source_luma['std']:.1f}"
                )
            continue

        full_frame, full_resized_w, full_resized_h, full_pad_x, full_pad_y = _prepare_full_frame(source_rgb)

        full_batch = (full_frame.astype(np.float32) / 255.0)[None, ...]
        q_batch = _quantize_input(full_batch, inp)
        obb_session.model.set_tensor(int(inp["index"]), q_batch)
        obb_session.model.invoke()
        q_out = obb_session.model.get_tensor(int(out["index"]))[0]
        scale = float(out["quantization"][0])
        zero_point = int(out["quantization"][1])
        obb_params = scale * (np.asarray(q_out, dtype=np.float32) - float(zero_point))

        decision = decode_obb_crop_box(
            obb_params.reshape(-1),
            source_width=FULL_FRAME_SIZE,
            source_height=FULL_FRAME_SIZE,
            input_size=FULL_FRAME_SIZE,
        )

        full_center_x, full_center_y = _project_source_point_to_full_frame(
            row.center_x_source,
            row.center_y_source,
            row.source_width,
            row.source_height,
        )

        crop_source = "obb"
        crop_reason = decision.fallback_reason
        crop_center_x = float(decision.details["center_x"]) * float(FULL_FRAME_SIZE)
        crop_center_y = float(decision.details["center_y"]) * float(FULL_FRAME_SIZE)
        if not decision.accepted:
            crop_source = "fixed_training"
            used_fixed_crop += 1
            crop_x_min, crop_y_min, crop_width, crop_height = _training_crop_box()
            crop_center_x = float(crop_x_min) + (float(crop_width) * 0.5)
            crop_center_y = float(crop_y_min) + (float(crop_height) * 0.5)
        else:
            accepted_obb += 1

        cd_canvas, crop_x_min, crop_y_min = _build_cd_crop(
            full_frame,
            crop_center_x,
            crop_center_y,
        )

        center_x_norm, center_y_norm = _project_full_frame_point_to_cd_crop(
            full_center_x,
            full_center_y,
            crop_x_min,
            crop_y_min,
        )

        if not _label_in_unit_interval(center_x_norm, center_y_norm):
            if crop_source == "obb":
                crop_source = "fixed_training"
                crop_reason = "center outside OBB crop"
                used_fixed_crop += 1
                crop_x_min, crop_y_min, crop_width, crop_height = _training_crop_box()
                crop_center_x = float(crop_x_min) + (float(crop_width) * 0.5)
                crop_center_y = float(crop_y_min) + (float(crop_height) * 0.5)
                cd_canvas, crop_x_min, crop_y_min = _build_cd_crop(
                    full_frame,
                    crop_center_x,
                    crop_center_y,
                )
                center_x_norm, center_y_norm = _project_full_frame_point_to_cd_crop(
                    full_center_x,
                    full_center_y,
                    crop_x_min,
                    crop_y_min,
                )

        if not _label_in_unit_interval(center_x_norm, center_y_norm):
            rejected += 1
            print(
                f"[{index}/{len(rows)}] REJECT {row.image_path.name} "
                f"reason={crop_reason!s} label=({center_x_norm:.3f},{center_y_norm:.3f})"
            )
            continue

        source_stem = row.image_path.stem
        out_name = f"cd_{row.source_kind}_{source_stem}.png"
        out_path = OUT_IMAGES / out_name
        cv2.imwrite(str(out_path), cv2.cvtColor(cd_canvas, cv2.COLOR_RGB2BGR))

        entries.append(
            {
                "image_path": f"images/{out_name}",
                "source_path": str(row.image_path.relative_to(REPO_ROOT)),
                "split": row.split,
                "quality_flag": row.quality_flag,
                "source_kind": row.source_kind,
                "source_luma_mean": float(source_luma["mean"]),
                "source_luma_std": float(source_luma["std"]),
                "source_luma_min": float(source_luma["min"]),
                "source_luma_max": float(source_luma["max"]),
                "source_dark_fraction": float(source_luma["dark_fraction"]),
                "temperature_c": float(row.temperature_c),
                "source_width": int(row.source_width),
                "source_height": int(row.source_height),
                "center_x_source": float(row.center_x_source),
                "center_y_source": float(row.center_y_source),
                "full_frame_center_x": float(full_center_x),
                "full_frame_center_y": float(full_center_y),
                "crop_center_x": float(crop_center_x),
                "crop_center_y": float(crop_center_y),
                "obb_accepted": bool(decision.accepted),
                "obb_fallback_reason": crop_reason,
                "crop_source": crop_source,
                "crop_x_min": int(crop_x_min),
                "crop_y_min": int(crop_y_min),
                "crop_width": CD_WIDTH,
                "crop_height": CD_HEIGHT,
                "center_x_norm": float(center_x_norm),
                "center_y_norm": float(center_y_norm),
                "full_frame_size": FULL_FRAME_SIZE,
                "full_frame_resized_width": int(full_resized_w),
                "full_frame_resized_height": int(full_resized_h),
                "full_frame_pad_x": int(full_pad_x),
                "full_frame_pad_y": int(full_pad_y),
            }
        )

        if index % 25 == 0 or index == 1:
            print(
                f"[{index}/{len(rows)}] {row.image_path.name} "
                f"stage={crop_source} label=({center_x_norm:.3f},{center_y_norm:.3f})"
            )

    meta_path = OUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    qa_report_path = OUT_DIR / "qa_report.json"
    qa_report = {
        "input_rows": len(rows),
        "saved_entries": len(entries),
        "skipped_blank_frames": skipped_blank,
        "accepted_obb": accepted_obb,
        "fixed_crop_fallbacks": used_fixed_crop,
        "rejected_for_label_bounds": rejected,
        "quality_filter": {
            "blank_frame_mean_threshold": BLANK_FRAME_MEAN_THRESHOLD,
            "blank_frame_std_threshold": BLANK_FRAME_STD_THRESHOLD,
            "blank_frame_dark_fraction_threshold": BLANK_FRAME_DARK_FRACTION_THRESHOLD,
            "blank_luma_level": BLANK_LUMA_LEVEL,
        },
        "rejected_entries": rejected_entries,
    }
    qa_report_path.write_text(json.dumps(qa_report, indent=2), encoding="utf-8")

    split_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for entry in entries:
        split = str(entry["split"])
        source_kind = str(entry["source_kind"])
        split_counts[split] = split_counts.get(split, 0) + 1
        source_counts[source_kind] = source_counts.get(source_kind, 0) + 1

    print("\nBoard-mimic dataset written")
    print(f"  entries: {len(entries)}")
    print(f"  accepted OBB crops: {accepted_obb}")
    print(f"  fixed-crop fallbacks: {used_fixed_crop}")
    print(f"  skipped blank frames: {skipped_blank}")
    print(f"  rejected: {rejected}")
    print(f"  split counts: {split_counts}")
    print(f"  source counts: {source_counts}")
    print(f"  metadata: {meta_path}")
    print(f"  qa report: {qa_report_path}")
    print(f"  images: {OUT_IMAGES}")


if __name__ == "__main__":
    main()
