#!/usr/bin/env python3
"""Generate overlay visualizations for the tip_focus final test replay.

Overlay sets:
1. Worst 30 accepted INT8 test predictions
2. All accepted INT8 errors >10 C
3. All rejected test cases
4. Largest Keras-vs-INT8 drift cases
5. 30 random accepted INT8 cases
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_heatmap_quantization_common import (
    DEFAULT_HEATMAP_SIZE,
    DEFAULT_INPUT_SIZE,
    DEFAULT_PREPROCESSING_MODE,
    load_split_samples,
    resolve_repo_path,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "ml" / "data" / "geometry_reader_manifest_v2_clean.csv"


def _error(row: dict[str, str]) -> float:
    return abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"]))


def _drift(row_int8: dict[str, str], row_keras: dict[str, str]) -> float:
    return abs(float(row_int8["guarded_temperature_c"]) - float(row_keras["guarded_temperature_c"]))


def _is_accepted(status: str) -> bool:
    return status in ("accepted", "clamped")


def draw_overlay(crop: np.ndarray, row: dict[str, str], output_path: Path, title: str = "") -> None:
    """Draw geometry overlay on the 224x224 crop image."""
    h, w = crop.shape[:2]
    img = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    scale_x = w / 224.0
    scale_y = h / 224.0

    def sxy(x: float, y: float) -> tuple[float, float]:
        return (x * scale_x, y * scale_y)

    # True center (green circle)
    tc = sxy(float(row["true_center_x_224"]), float(row["true_center_y_224"]))
    draw.ellipse([tc[0] - 5, tc[1] - 5, tc[0] + 5, tc[1] + 5], fill="lime", outline="white", width=2)

    # True tip (red circle)
    tt = sxy(float(row["true_tip_x_224"]), float(row["true_tip_y_224"]))
    draw.ellipse([tt[0] - 5, tt[1] - 5, tt[0] + 5, tt[1] + 5], fill="red", outline="white", width=2)

    # True needle line (white)
    draw.line([tc, tt], fill="white", width=3)

    # Predicted center (cyan square)
    pc = sxy(float(row["predicted_center_x_224"]), float(row["predicted_center_y_224"]))
    draw.rectangle([pc[0] - 4, pc[1] - 4, pc[0] + 4, pc[1] + 4], fill="cyan", outline="black", width=2)

    # Predicted tip (yellow square)
    pt = sxy(float(row["predicted_tip_x_224"]), float(row["predicted_tip_y_224"]))
    draw.rectangle([pt[0] - 4, pt[1] - 4, pt[0] + 4, pt[1] + 4], fill="yellow", outline="black", width=2)

    # Predicted needle line (cyan)
    draw.line([pc, pt], fill="cyan", width=2)

    # Text overlay
    err = _error(row)
    text = [
        f"True: {float(row['true_temperature_c']):.1f}C  Pred: {float(row['guarded_temperature_c']):.1f}C",
        f"Error: {err:.2f}C  Status: {row['guardrail_status']}",
    ]
    if title:
        text.insert(0, title)
    text_y = 4
    for line in text:
        draw.text((4, text_y), line, fill="yellow" if err > 5 else "white", stroke_width=1, stroke_color="black")
        text_y += 14

    img.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tip_focus final test overlays")
    parser.add_argument("--predictions-path", type=str,
                        default="ml/artifacts/deployment/geometry_heatmap_v4_112_tflite/recovery_08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3/v4_112_tip_focus_final_test_predictions.csv")
    parser.add_argument("--output-dir", type=str,
                        default="ml/debug/geometry_heatmap_v4_112_tip_focus_int8_final_test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pred_path = resolve_repo_path(REPO_ROOT, Path(args.predictions_path))
    output_dir = resolve_repo_path(REPO_ROOT, Path(args.output_dir))

    # Load predictions
    with pred_path.open("r", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    keras_rows = [r for r in all_rows if r["model_type"] == "keras_v3"]
    int8_rows = [r for r in all_rows if r["model_type"] == "tflite_int8"]
    float_rows = [r for r in all_rows if r["model_type"] == "tflite_float32"]

    j = {r["image_path"]: r for r in keras_rows}
    int8_with_drift = []
    for r in int8_rows:
        d = _drift(r, j.get(r["image_path"], r))
        int8_with_drift.append((d, r))
    int8_with_drift.sort(key=lambda x: -x[0])

    accepted_int8 = [r for r in int8_rows if _is_accepted(r["guardrail_status"])]
    accepted_int8.sort(key=_error, reverse=True)

    over_10c = [r for r in accepted_int8 if _error(r) > 10.0]
    rejected = [r for r in int8_rows if not _is_accepted(r["guardrail_status"])]
    drift_cases = [r for _, r in int8_with_drift if r in int8_rows][:30]

    rng = random.Random(args.seed)
    random_accepted = rng.sample(accepted_int8, min(30, len(accepted_int8)))

    # Load test split samples for crop images
    examples = load_split_samples(
        MANIFEST_PATH, REPO_ROOT, split="test",
        mode=DEFAULT_PREPROCESSING_MODE,
        input_size=DEFAULT_INPUT_SIZE,
        heatmap_size=DEFAULT_HEATMAP_SIZE,
        sigma_pixels=2.5,
    ).samples
    sample_by_path: dict[str, object] = {}
    for s in examples:
        sample_by_path[str(s.metadata["image_path"])] = s

    overlay_sets = [
        ("worst_30_accepted", accepted_int8[:30], "Worst 30 Accepted INT8"),
        ("errors_over_10c", over_10c, "Errors >10C"),
        ("rejected", rejected, "Rejected"),
        ("largest_drift", drift_cases, "Largest Keras-vs-INT8 Drift"),
        ("random_30_accepted", random_accepted, "Random 30 Accepted INT8"),
    ]

    for subdir, rows, label in overlay_sets:
        out_path = output_dir / subdir
        out_path.mkdir(parents=True, exist_ok=True)
        count = 0
        for i, row in enumerate(rows):
            sample = sample_by_path.get(row["image_path"])
            if sample is None:
                continue
            crop = sample.crop_image
            stem = Path(row["image_path"]).stem
            fname = f"{subdir}_{i:03d}_err{_error(row):.2f}C_{stem}.jpg"
            draw_overlay(crop, row, out_path / fname, title=f"{label} #{i}")
            count += 1
        print(f"  {subdir}: {count} overlays")

    print(f"\nOverlay directory: {output_dir}")


if __name__ == "__main__":
    main()
