#!/usr/bin/env python3
"""Evaluate QARepVGG-Mini heatmap OBB model — pixel-level metrics.

Computes:
  centre MAE (px in 256×256), angle MAE (deg), box_size MAE (px on 256×256).

Usage:
  poetry run python scripts/eval_qarepvgg_heatmap.py \
      --model /path/to/best_warmup.keras
      --examples-path /path/to/saved_examples.npz  (optional — recompute if missing)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import tf_keras as keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.training import (
    TrainingExample,
    _compute_fullframe_obb_params,
)
from embedded_gauge_reading_tinyml.tf_models import build_qarepvgg_mini
from embedded_gauge_reading_tinyml.heatmap_utils import softargmax_2d

IMAGE_HEIGHT = 320
IMAGE_WIDTH = 320
HEATMAP_SIZE = 40


def _preprocess_colour(image: np.ndarray, height: int, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(height / h, width / w)
    scaled_h = max(int(h * scale), 1)
    scaled_w = max(int(w * scale), 1)
    resized = tf.image.resize(
        image, [scaled_h, scaled_w], method="nearest",
    ).numpy()
    pad_y = (height - scaled_h) // 2
    pad_x = (width - scaled_w) // 2
    padded = np.pad(
        resized,
        [[pad_y, height - scaled_h - pad_y],
         [pad_x, width - scaled_w - pad_x],
         [0, 0]],
    )
    return padded.astype(np.float32) / 255.0


def _load_image(path: str) -> np.ndarray:
    raw = tf.io.read_file(path)
    if path.endswith(".yuv422"):
        yuyv = tf.io.decode_raw(raw, tf.uint8)
        yuyv = tf.reshape(yuyv, [320, 640])
        y = tf.cast(yuyv[:, 0::2], tf.float32)
        u = tf.cast(yuyv[:, 1::4], tf.float32) - 128.0
        v = tf.cast(yuyv[:, 3::4], tf.float32) - 128.0
        u = tf.repeat(u, 2, axis=1)
        v = tf.repeat(v, 2, axis=1)
        rgb = tf.stack([
            y + 1.402 * v,
            y - 0.344136 * u - 0.714136 * v,
            y + 1.772 * u,
        ], axis=-1)
        image = tf.clip_by_value(rgb, 0, 255).numpy().astype(np.uint8)
    else:
        image = tf.io.decode_image(raw, channels=3, expand_animations=False).numpy()
    return _preprocess_colour(image, IMAGE_HEIGHT, IMAGE_WIDTH)


def _load_examples(manifest_path: Path, max_examples: int = 0,
                   use_board_captures: bool = False) -> list[TrainingExample]:
    REPO_ROOT = manifest_path.parent.parent.parent
    examples: list[TrainingExample] = []
    seen: set[str] = set()
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fpath_base = str(REPO_ROOT / row["image_path"])
            qf = row.get("quality_flag", "")
            if qf not in ("clean", "manual", ""):
                continue
            # Board captures mode: only captured_images/, same as training
            if use_board_captures:
                if "captured_images" not in row["image_path"]:
                    continue
                fpath_base = fpath_base.replace("/captured_images/", "/captured_images_320/")
                if fpath_base.endswith(".jpg"):
                    fpath_base = fpath_base[:-4] + ".png"
            if not Path(fpath_base).exists():
                continue
            source_w = int(float(row["source_width"]))
            source_h = int(float(row["source_height"]))
            cx = float(row["center_x_source"])
            cy = float(row["center_y_source"])
            radius = float(row.get("dial_radius_source", 0))
            if radius <= 0:
                continue
            obb = _compute_fullframe_obb_params(
                source_w, source_h, cx, cy, radius, radius, 0.0,
                IMAGE_HEIGHT, IMAGE_WIDTH,
            )
            examples.append(TrainingExample(image_path=fpath_base, value=0.0,
                crop_box_xyxy=(0,0,source_w,source_h), needle_unit_xy=(0,0),
                obb_params=obb))
            if max_examples and len(examples) >= max_examples:
                break
    return examples


def _decode_predictions(
    preds: dict[str, np.ndarray],
) -> list[dict[str, float]]:
    """Decode batched model outputs to per-image predictions.

    preds keys: heatmap (B,32,32,1), box_size (B,32,32,2), angle (B,32,32,2)
    """
    B = preds["heatmap"].shape[0]
    results = []
    for i in range(B):
        hm = 1.0 / (1.0 + np.exp(-preds["heatmap"][i, :, :, 0]))  # sigmoid
        box = preds["box_size"][i]       # 32, 32, 2
        ang = preds["angle"][i]          # 32, 32, 2

        # softargmax
        h_sum = np.sum(hm)
        if h_sum <= 0:
            py, px = 0.0, 0.0
        else:
            hm_n = hm / h_sum
            ys, xs = np.meshgrid(np.arange(HEATMAP_SIZE, dtype=np.float32), np.arange(HEATMAP_SIZE, dtype=np.float32), indexing="ij")
            py = float(np.sum(hm_n * ys))
            px = float(np.sum(hm_n * xs))

        # Normalized centre (0-1) in heatmap space
        cx_hm = px / (HEATMAP_SIZE - 1)
        cy_hm = py / (HEATMAP_SIZE - 1)

        # Read box/angle at argmax cell
        argmax_idx = np.argmax(hm)
        cell_y = argmax_idx // HEATMAP_SIZE
        cell_x = argmax_idx % HEATMAP_SIZE
        w_hm, h_hm = float(box[cell_y, cell_x, 0]), float(box[cell_y, cell_x, 1])
        sin2t, cos2t = float(ang[cell_y, cell_x, 0]), float(ang[cell_y, cell_x, 1])

        theta = 0.5 * np.arctan2(sin2t, cos2t)

        peak_val = float(hm[cell_y, cell_x])

        results.append({
            "cx_norm": cx_hm,
            "cy_norm": cy_hm,
            "w_norm": w_hm,
            "h_norm": h_hm,
            "sin2t": sin2t,
            "cos2t": cos2t,
            "angle_deg": float(np.degrees(theta)),
            "heatmap_peak": peak_val,
        })
    return results


def _compute_errors(
    preds: list[dict[str, float]],
    targets: list[np.ndarray],
) -> dict[str, float]:
    errors = []
    for p, t in zip(preds, targets):
        cx_t, cy_t, w_t, h_t, c2t_t, s2t_t = t
        cx_p = p["cx_norm"]
        cy_p = p["cy_norm"]
        w_p = p["w_norm"]
        h_p = p["h_norm"]
        gt_angle = 0.5 * np.arctan2(s2t_t, c2t_t)
        pred_angle = p["angle_deg"]

        center_px = np.sqrt((cx_p - cx_t)**2 + (cy_p - cy_t)**2) * (IMAGE_WIDTH - 1)
        box_px = np.sqrt((w_p - w_t)**2 + (h_p - h_t)**2) * (IMAGE_WIDTH - 1)
        angle_deg = float(np.degrees(abs(np.arctan2(
            np.sin(gt_angle - np.radians(pred_angle)),
            np.cos(gt_angle - np.radians(pred_angle)),
        ))))

        errors.append({
            "center_px": float(center_px),
            "box_px": float(box_px),
            "angle_deg": angle_deg,
        })

    arr_c = np.array([e["center_px"] for e in errors])
    arr_b = np.array([e["box_px"] for e in errors])
    arr_a = np.array([e["angle_deg"] for e in errors])
    return {
        "center_mae_px": float(np.mean(arr_c)),
        "center_median_px": float(np.median(arr_c)),
        "center_p10_px": float(np.percentile(arr_c, 10)),
        "center_p90_px": float(np.percentile(arr_c, 90)),
        "box_mae_px": float(np.mean(arr_b)),
        "angle_mae_deg": float(np.mean(arr_a)),
        "angle_median_deg": float(np.median(arr_a)),
        "n": len(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .keras checkpoint")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to manifest CSV (default: ml/data/merged_geometry_board_manifest.csv)")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--max-examples", type=int, default=200,
                        help="Max examples to evaluate (0 = all)")
    parser.add_argument("--board-captures", action="store_true",
                        help="Use only board captures (training data)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}"); sys.exit(1)

    manifest_path = Path(args.manifest) if args.manifest else (
        PROJECT_ROOT / "data" / "merged_geometry_board_manifest.csv"
    )

    print(f"Loading model from {model_path}...")
    model = build_qarepvgg_mini((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    model.load_weights(model_path)

    print(f"Loading examples from {manifest_path}...")
    examples = _load_examples(manifest_path, max_examples=args.max_examples,
                              use_board_captures=args.board_captures)
    print(f"  Found {len(examples)} examples")

    # Run inference
    print("Running inference...")
    all_preds: list[dict[str, float]] = []
    all_targets: list[np.ndarray] = []
    BATCH = 16
    for i in range(0, len(examples), BATCH):
        batch = examples[i:i + BATCH]
        images = np.stack([_load_image(ex.image_path) for ex in batch], axis=0)
        preds = model.predict(images, verbose=0)
        decoded = _decode_predictions(preds)
        all_preds.extend(decoded)
        all_targets.extend([np.array(ex.obb_params, dtype=np.float32) for ex in batch])
        if (i // BATCH) % 10 == 0:
            print(f"  [{i}/{len(examples)}]")

    # Compute metrics
    metrics = _compute_errors(all_preds, all_targets)
    print("\n" + "=" * 50)
    print(f"  Eval on {metrics['n']} examples")
    print(f"  Centre MAE:     {metrics['center_mae_px']:.2f} px")
    print(f"  Centre median:  {metrics['center_median_px']:.2f} px")
    print(f"  Centre P10/P90: {metrics['center_p10_px']:.2f} / {metrics['center_p90_px']:.2f} px")
    print(f"  Box MAE:        {metrics['box_mae_px']:.2f} px")
    print(f"  Angle MAE:      {metrics['angle_mae_deg']:.2f} deg")
    print(f"  Angle median:   {metrics['angle_median_deg']:.2f} deg")
    print("=" * 50)

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
