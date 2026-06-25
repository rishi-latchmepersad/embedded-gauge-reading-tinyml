#!/usr/bin/env python3
"""Replay an OBB checkpoint on labeled board captures and render overlays.

This script is intentionally narrow:
- it reads the manual board-bbox CSV from ``tmp/``,
- keeps only the reviewed boxes,
- runs the current OBB checkpoint on the same full-frame preprocessing used
  during training, and
- writes one overlay per image plus contact sheets for the full set and the
  worst cases.

The goal is to make the board-capture failure modes obvious before we trust
the localizer for downstream SimCC training or deployment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import Rectangle
from PIL import Image

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.board_crop_compare import resize_with_pad_rgb  # noqa: E402
from embedded_gauge_reading_tinyml.geometry_board_replay import load_board_replay_image  # noqa: E402

DEFAULT_MODEL_PATH: Path = REPO_ROOT / "tmp" / "obb_box_board_bbox_v2_long" / "obb_box_qat.tflite"
DEFAULT_LABELS_CSV: Path = REPO_ROOT / "tmp" / "board_bbox_labels.csv"
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "tmp" / "obb_board_bbox_replay"
IMAGE_SIZE: int = 224


@dataclass(frozen=True, slots=True)
class BoardReplayCase:
    """One labeled board capture and its reviewed gauge box."""

    image_path: Path
    source_width: int
    source_height: int
    crop_x_min: float
    crop_y_min: float
    crop_x_max: float
    crop_y_max: float
    quality_flag: str
    label_source: str
    notes: str
    origin_manifest: str


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """One prediction against one labeled board capture."""

    image_path: Path
    quality_flag: str
    label_source: str
    notes: str
    origin_manifest: str
    source_kind: str
    confidence: float
    gt_cx: float
    gt_cy: float
    gt_w: float
    gt_h: float
    pred_cx: float
    pred_cy: float
    pred_w: float
    pred_h: float
    center_error_px: float
    size_error_px: float
    iou: float
    source_overlay_path: Path
    canvas_overlay_path: Path


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the replay job."""

    parser = argparse.ArgumentParser(description="Replay an OBB checkpoint on labeled board captures.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick smoke checks.")
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Also render rows whose quality_flag is not 'review'.",
    )
    return parser.parse_args()


def _load_cases(labels_csv: Path, *, include_excluded: bool) -> list[BoardReplayCase]:
    """Load reviewed board-box labels in CSV order."""

    cases: list[BoardReplayCase] = []
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{labels_csv} is missing a header row.")
        required = {
            "image_path",
            "source_width",
            "source_height",
            "crop_x_min",
            "crop_y_min",
            "crop_x_max",
            "crop_y_max",
            "quality_flag",
            "label_source",
            "notes",
            "origin_manifest",
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"{labels_csv} is missing required columns: {missing}")

        for row in reader:
            quality_flag = str(row.get("quality_flag", "")).strip() or "review"
            if quality_flag != "review" and not include_excluded:
                continue

            image_path = Path(str(row["image_path"]).strip())
            crop_values = (
                float(row["crop_x_min"]),
                float(row["crop_y_min"]),
                float(row["crop_x_max"]),
                float(row["crop_y_max"]),
            )
            if any(not math.isfinite(value) for value in crop_values):
                continue
            cases.append(
                BoardReplayCase(
                    image_path=image_path,
                    source_width=int(float(row["source_width"])),
                    source_height=int(float(row["source_height"])),
                    crop_x_min=crop_values[0],
                    crop_y_min=crop_values[1],
                    crop_x_max=crop_values[2],
                    crop_y_max=crop_values[3],
                    quality_flag=quality_flag,
                    label_source=str(row.get("label_source", "")).strip(),
                    notes=str(row.get("notes", "")).strip(),
                    origin_manifest=str(row.get("origin_manifest", "")).strip(),
                )
            )
    return cases


def _load_interpreter(model_path: Path) -> tf.lite.Interpreter:
    """Load the exported int8 TFLite checkpoint for replay."""

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def _load_canvas(image_path: Path, source_width: int, source_height: int) -> tuple[np.ndarray, str]:
    """Load a capture and run the same full-frame resize-with-pad path as training."""

    absolute_path = image_path if image_path.is_absolute() else (REPO_ROOT / image_path).resolve()
    source_image, source_kind = load_board_replay_image(
        absolute_path,
        image_width=source_width,
        image_height=source_height,
    )
    canvas = np.asarray(
        resize_with_pad_rgb(
            source_image,
            (0.0, 0.0, float(source_width), float(source_height)),
            image_size=IMAGE_SIZE,
        ),
        dtype=np.uint8,
    )
    return canvas, source_kind


def _box_xyxy_to_canvas_norm(
    box_xyxy: tuple[float, float, float, float],
    *,
    source_width: int,
    source_height: int,
) -> np.ndarray:
    """Map a source-space box into normalized 224x224 canvas coordinates."""

    x1, y1, x2, y2 = box_xyxy
    scale = min(float(IMAGE_SIZE) / float(source_width), float(IMAGE_SIZE) / float(source_height))
    resized_width = float(source_width) * scale
    resized_height = float(source_height) * scale
    pad_x = 0.5 * (float(IMAGE_SIZE) - resized_width)
    pad_y = 0.5 * (float(IMAGE_SIZE) - resized_height)
    cx = (((x1 + x2) * 0.5) * scale + pad_x) / float(IMAGE_SIZE)
    cy = (((y1 + y2) * 0.5) * scale + pad_y) / float(IMAGE_SIZE)
    w = ((x2 - x1) * scale) / float(IMAGE_SIZE)
    h = ((y2 - y1) * scale) / float(IMAGE_SIZE)
    return np.array([cx, cy, w, h], dtype=np.float32)


def _canvas_norm_to_xyxy(
    box_norm: np.ndarray,
    *,
    source_width: int,
    source_height: int,
) -> tuple[float, float, float, float]:
    """Map normalized canvas box coordinates back into source pixels."""

    cx, cy, w, h = [float(value) for value in np.asarray(box_norm, dtype=np.float32).reshape(-1)[:4]]
    scale = min(float(IMAGE_SIZE) / float(source_width), float(IMAGE_SIZE) / float(source_height))
    resized_width = float(source_width) * scale
    resized_height = float(source_height) * scale
    pad_x = 0.5 * (float(IMAGE_SIZE) - resized_width)
    pad_y = 0.5 * (float(IMAGE_SIZE) - resized_height)

    cx_px = cx * float(IMAGE_SIZE)
    cy_px = cy * float(IMAGE_SIZE)
    w_px = w * float(IMAGE_SIZE)
    h_px = h * float(IMAGE_SIZE)

    x1 = (cx_px - 0.5 * w_px - pad_x) / scale
    y1 = (cy_px - 0.5 * h_px - pad_y) / scale
    x2 = (cx_px + 0.5 * w_px - pad_x) / scale
    y2 = (cy_px + 0.5 * h_px - pad_y) / scale
    return (
        float(np.clip(x1, 0.0, float(source_width))),
        float(np.clip(y1, 0.0, float(source_height))),
        float(np.clip(x2, 0.0, float(source_width))),
        float(np.clip(y2, 0.0, float(source_height))),
    )


def _xyxy_to_cxcywh(box_xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert an axis-aligned box into center/size form."""

    x1, y1, x2, y2 = box_xyxy
    return (
        0.5 * (x1 + x2),
        0.5 * (y1 + y2),
        max(0.0, x2 - x1),
        max(0.0, y2 - y1),
    )


def _axis_aligned_iou(
    pred_xyxy: tuple[float, float, float, float],
    gt_xyxy: tuple[float, float, float, float],
) -> float:
    """Compute a plain axis-aligned IoU for the replay summary."""

    px1, py1, px2, py2 = pred_xyxy
    gx1, gy1, gx2, gy2 = gt_xyxy
    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pred_area + gt_area - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def _render_overlay(
    *,
    image: np.ndarray,
    title: str,
    gt_xyxy: tuple[float, float, float, float],
    pred_xyxy: tuple[float, float, float, float],
    confidence: float,
    source_kind: str,
    quality_flag: str,
    notes: str,
    iou: float,
    center_error_px: float,
    size_error_px: float,
) -> np.ndarray:
    """Render one annotated overlay as a PNG-ready RGB array."""

    fig, ax = plt.subplots(figsize=(8.0, 7.0), dpi=150)
    ax.imshow(np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0))
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_xyxy
    pr_x1, pr_y1, pr_x2, pr_y2 = pred_xyxy

    ax.add_patch(
        Rectangle(
            (gt_x1, gt_y1),
            max(1.0, gt_x2 - gt_x1),
            max(1.0, gt_y2 - gt_y1),
            fill=False,
            edgecolor="lime",
            linewidth=2.5,
            label="gt",
        )
    )
    ax.add_patch(
        Rectangle(
            (pr_x1, pr_y1),
            max(1.0, pr_x2 - pr_x1),
            max(1.0, pr_y2 - pr_y1),
            fill=False,
            edgecolor="cyan",
            linewidth=2.5,
            label="pred",
        )
    )

    gt_cx, gt_cy, _gt_w, _gt_h = _xyxy_to_cxcywh(gt_xyxy)
    pr_cx, pr_cy, _pr_w, _pr_h = _xyxy_to_cxcywh(pred_xyxy)
    ax.scatter([gt_cx, pr_cx], [gt_cy, pr_cy], c=["lime", "cyan"], s=40, marker="o", edgecolors="black", linewidths=0.5)

    ax.set_axis_off()
    ax.set_title(title, fontsize=10)
    panel = "\n".join(
        [
            f"conf: {confidence:.3f}",
            f"iou: {iou:.3f}",
            f"center err: {center_error_px:.2f}px",
            f"size err: {size_error_px:.2f}px",
            f"source: {source_kind}",
            f"quality: {quality_flag}",
            f"notes: {notes or 'none'}",
        ]
    )
    ax.text(
        0.01,
        0.01,
        panel,
        transform=ax.transAxes,
        fontsize=8.5,
        family="monospace",
        va="bottom",
        ha="left",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 4.0, "edgecolor": "none"},
    )
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()
    plt.close(fig)
    return rgb


def _save_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    *,
    columns: int = 6,
    tile_width: int = 320,
    tile_height: int = 280,
    background: tuple[int, int, int] = (18, 18, 18),
) -> None:
    """Build a simple tiled contact sheet from the per-image overlay PNGs."""

    if not image_paths:
        return

    rows = int(math.ceil(len(image_paths) / float(columns)))
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), color=background)

    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as overlay:
            thumb = overlay.convert("RGB").copy()
            thumb.thumbnail((tile_width - 10, tile_height - 10), Image.Resampling.LANCZOS)
            x = (index % columns) * tile_width + (tile_width - thumb.width) // 2
            y = (index // columns) * tile_height + (tile_height - thumb.height) // 2
            sheet.paste(thumb, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    """Run the replay and write all overlays plus summary artifacts."""

    args = _parse_args()
    output_dir = args.output_dir
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_cases(args.labels_csv, include_excluded=args.include_excluded)
    if args.limit is not None:
        cases = cases[: max(0, int(args.limit))]
    if not cases:
        raise SystemExit("No reviewed board boxes were found in the labels CSV.")

    if args.model_path.suffix.lower() != ".tflite":
        raise SystemExit("This replay runner currently expects an int8 .tflite model.")

    interpreter = _load_interpreter(args.model_path)
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    output_by_shape = {tuple(detail["shape"]): detail for detail in output_details}
    if (1, 4) not in output_by_shape or (1, 1) not in output_by_shape:
        raise SystemExit("The TFLite model does not expose the expected [1,4] box and [1,1] conf outputs.")

    box_detail = output_by_shape[(1, 4)]
    conf_detail = output_by_shape[(1, 1)]

    print(f"[REPLAY] Loaded model: {args.model_path}")
    print(f"[REPLAY] Loaded {len(cases)} labeled board captures from {args.labels_csv}")

    results: list[ReplayResult] = []
    for index, case in enumerate(cases, start=1):
        source_image, source_kind = load_board_replay_image(
            case.image_path if case.image_path.is_absolute() else (REPO_ROOT / case.image_path).resolve(),
            image_width=case.source_width,
            image_height=case.source_height,
        )
        canvas = np.asarray(
            resize_with_pad_rgb(
                source_image,
                (0.0, 0.0, float(case.source_width), float(case.source_height)),
                image_size=IMAGE_SIZE,
            ),
            dtype=np.uint8,
        )
        input_float = canvas.astype(np.float32) / 127.5 - 1.0
        input_scale, input_zero = input_detail["quantization"]
        input_q = np.clip(np.round(input_float / float(input_scale) + float(input_zero)), -128, 127).astype(np.int8)
        interpreter.set_tensor(int(input_detail["index"]), np.expand_dims(input_q, axis=0))
        interpreter.invoke()

        box_scale, box_zero = box_detail["quantization"]
        conf_scale, conf_zero = conf_detail["quantization"]
        pred_box_q = interpreter.get_tensor(int(box_detail["index"]))
        pred_conf_q = interpreter.get_tensor(int(conf_detail["index"]))
        pred_box_norm = (pred_box_q.astype(np.float32) - float(box_zero)) * float(box_scale)
        pred_conf = float(((pred_conf_q.astype(np.float32) - float(conf_zero)) * float(conf_scale)).reshape(-1)[0])
        pred_box_norm = np.asarray(pred_box_norm, dtype=np.float32).reshape(-1)[:4]
        gt_box_norm = _box_xyxy_to_canvas_norm(
            (case.crop_x_min, case.crop_y_min, case.crop_x_max, case.crop_y_max),
            source_width=case.source_width,
            source_height=case.source_height,
        )

        pred_xyxy = _canvas_norm_to_xyxy(
            pred_box_norm,
            source_width=case.source_width,
            source_height=case.source_height,
        )
        gt_xyxy = (
            case.crop_x_min,
            case.crop_y_min,
            case.crop_x_max,
            case.crop_y_max,
        )

        gt_cx, gt_cy, gt_w, gt_h = _xyxy_to_cxcywh(gt_xyxy)
        pred_cx, pred_cy, pred_w, pred_h = _xyxy_to_cxcywh(pred_xyxy)
        center_error_px = math.hypot(pred_cx - gt_cx, pred_cy - gt_cy)
        size_error_px = 0.5 * (abs(pred_w - gt_w) + abs(pred_h - gt_h))
        iou = _axis_aligned_iou(pred_xyxy, gt_xyxy)

        source_overlay = _render_overlay(
            image=source_image,
            title=f"{case.image_path.name} | source",
            gt_xyxy=gt_xyxy,
            pred_xyxy=pred_xyxy,
            confidence=pred_conf,
            source_kind=source_kind,
            quality_flag=case.quality_flag,
            notes=case.notes,
            iou=iou,
            center_error_px=center_error_px,
            size_error_px=size_error_px,
        )
        canvas_overlay = _render_overlay(
            image=canvas,
            title=f"{case.image_path.name} | model canvas",
            gt_xyxy=(
                float(gt_box_norm[0] * IMAGE_SIZE - 0.5 * gt_box_norm[2] * IMAGE_SIZE),
                float(gt_box_norm[1] * IMAGE_SIZE - 0.5 * gt_box_norm[3] * IMAGE_SIZE),
                float(gt_box_norm[0] * IMAGE_SIZE + 0.5 * gt_box_norm[2] * IMAGE_SIZE),
                float(gt_box_norm[1] * IMAGE_SIZE + 0.5 * gt_box_norm[3] * IMAGE_SIZE),
            ),
            pred_xyxy=(
                float(pred_box_norm[0] * IMAGE_SIZE - 0.5 * pred_box_norm[2] * IMAGE_SIZE),
                float(pred_box_norm[1] * IMAGE_SIZE - 0.5 * pred_box_norm[3] * IMAGE_SIZE),
                float(pred_box_norm[0] * IMAGE_SIZE + 0.5 * pred_box_norm[2] * IMAGE_SIZE),
                float(pred_box_norm[1] * IMAGE_SIZE + 0.5 * pred_box_norm[3] * IMAGE_SIZE),
            ),
            confidence=pred_conf,
            source_kind=source_kind,
            quality_flag=case.quality_flag,
            notes=case.notes,
            iou=iou,
            center_error_px=center_error_px,
            size_error_px=size_error_px,
        )

        stem = case.image_path.stem
        source_overlay_path = overlays_dir / f"{index:03d}_{stem}_source.png"
        canvas_overlay_path = overlays_dir / f"{index:03d}_{stem}_canvas.png"
        Image.fromarray(source_overlay).save(source_overlay_path)
        Image.fromarray(canvas_overlay).save(canvas_overlay_path)

        results.append(
            ReplayResult(
                image_path=case.image_path,
                quality_flag=case.quality_flag,
                label_source=case.label_source,
                notes=case.notes,
                origin_manifest=case.origin_manifest,
                source_kind=source_kind,
                confidence=pred_conf,
                gt_cx=gt_cx,
                gt_cy=gt_cy,
                gt_w=gt_w,
                gt_h=gt_h,
                pred_cx=pred_cx,
                pred_cy=pred_cy,
                pred_w=pred_w,
                pred_h=pred_h,
                center_error_px=center_error_px,
                size_error_px=size_error_px,
                iou=iou,
                source_overlay_path=source_overlay_path,
                canvas_overlay_path=canvas_overlay_path,
            )
        )
        print(
            f"[REPLAY] {index:02d}/{len(cases)} {case.image_path.name}: "
            f"iou={iou:.3f} center={center_error_px:.2f}px size={size_error_px:.2f}px conf={pred_conf:.3f}"
        )

    source_overlays = [result.source_overlay_path for result in results]
    _save_contact_sheet(source_overlays, output_dir / "contact_sheet_all.png")

    worst_by_center = sorted(results, key=lambda row: row.center_error_px, reverse=True)[: min(12, len(results))]
    _save_contact_sheet([row.source_overlay_path for row in worst_by_center], output_dir / "contact_sheet_worst12.png")

    summary = {
        "model_path": str(args.model_path),
        "labels_csv": str(args.labels_csv),
        "output_dir": str(output_dir),
        "count": len(results),
        "mean_confidence": float(np.mean([row.confidence for row in results])),
        "mean_center_error_px": float(np.mean([row.center_error_px for row in results])),
        "mean_size_error_px": float(np.mean([row.size_error_px for row in results])),
        "mean_iou": float(np.mean([row.iou for row in results])),
        "median_center_error_px": float(np.median([row.center_error_px for row in results])),
        "worst_center_error_px": float(max(row.center_error_px for row in results)),
        "worst_iou": float(min(row.iou for row in results)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = [
        {
            "image_path": row.image_path.as_posix(),
            "quality_flag": row.quality_flag,
            "source_kind": row.source_kind,
            "confidence": row.confidence,
            "center_error_px": row.center_error_px,
            "size_error_px": row.size_error_px,
            "iou": row.iou,
            "source_overlay_path": row.source_overlay_path.as_posix(),
            "canvas_overlay_path": row.canvas_overlay_path.as_posix(),
        }
        for row in results
    ]
    (output_dir / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[REPLAY] Wrote overlays and contact sheets to {output_dir}")
    print(f"[REPLAY] Summary: {summary}")


if __name__ == "__main__":
    main()
