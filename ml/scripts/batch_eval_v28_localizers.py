"""Batch-evaluate localizers end-to-end against V28, loading V28 once."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import numpy as np
import tensorflow as tf

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPO_ROOT: Final[Path] = PROJECT_ROOT.parent
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "tmp" / "matplotlib"))

from embedded_gauge_reading_tinyml.board_pipeline import (
    decode_obb_crop_box,
    decode_rectifier_crop_box,
    load_capture_image,
    load_model_session,
)
from embedded_gauge_reading_tinyml.firmware_preprocessing import (
    build_training_style_polar_vote_float32,
    decode_circular_vote_logits,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.polar_vote_v28 import build_polar_vote_v28_model
from embedded_gauge_reading_tinyml.v28_localizer_pipeline import (
    LocalizerCropDecision,
    decode_heatmap_crop_box,
    decode_keypoint_crop_box,
    decode_source_crop_box,
    resize_full_frame,
)

# Import eval script functions by temporarily adding scripts to path
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import eval_v28_localizer_pipeline as eval_mod

DEFAULT_MANIFEST: Final[Path] = PROJECT_ROOT / "data" / "hard_cases_plus_board30_valid_with_new6.csv"
DEFAULT_CAPTURE_ROOT: Final[Path] = PROJECT_ROOT / "data" / "captured_images"
DEFAULT_WEIGHTS: Final[Path] = (
    PROJECT_ROOT / "artifacts" / "training" / "polar_vote_circular_v28" / "best_weights.weights.h5"
)
DEFAULT_OUTPUT_DIR: Final[Path] = REPO_ROOT / "tmp" / "v28_localizer_batch_eval"
DEFAULT_GAUGE_ID: Final[str] = "littlegood_home_temp_gauge_c"

LOCALIZERS = [
    ("compact_geometry_cascade_localizer", "keypoint_coords"),
    ("compact_geometry_cascade_localizer_longterm", "keypoint_coords"),
    ("compact_geometry_full_range", "keypoint_coords"),
    ("compact_geometry_hardcases_full_range_v1", "keypoint_coords"),
    ("compact_geometry_longterm", "keypoint_coords"),
    ("mobilenetv2_bluraware_obb_geometry_v33", "obb_params"),
    ("mobilenetv2_bluraware_obb_geometry_v34", "obb_params"),
    ("mobilenetv2_bluraware_obb_relation_geometry_v42", "obb_params"),
    ("mobilenetv2_bluraware_obb_sequence_geometry_hardtail_v44", "obb_params"),
    ("mobilenetv2_bluraware_obb_sequence_geometry_v43", "obb_params"),
    ("mobilenetv2_detector_geometry", "keypoint_heatmaps"),
    ("mobilenetv2_geometry_board_augment", "keypoint_coords"),
    ("mobilenetv2_geometry_cascade_localizer_longterm", "keypoint_coords"),
    ("mobilenetv2_geometry_detector", "keypoint_coords"),
    ("mobilenetv2_geometry_full_range_v1", "keypoint_coords"),
    ("mobilenetv2_geometry_keypoints_only", "keypoint_coords"),
    ("mobilenetv2_geometry_literature_v29", "keypoint_coords"),
    ("mobilenetv2_geometry_localizer_only_v30", "keypoint_coords"),
    ("mobilenetv2_geometry_longterm", "keypoint_coords"),
    ("mobilenetv2_geometry_raw_cvat_v18", "keypoint_coords"),
    ("mobilenetv2_geometry_uncertainty_full_range", "keypoint_coords"),
    ("mobilenetv2_keypoint_from_rectified_v5", "keypoint_heatmaps"),
    ("mobilenetv2_keypoint_geometry_clean", "keypoint_heatmaps"),
    ("mobilenetv2_keypoint_geometry_first", "keypoint_heatmaps"),
    ("mobilenetv2_obb_geometry_v32", "obb_params"),
    ("mobilenetv2_obb_localizer_v31", "obb_params"),
    ("mobilenetv2_obb_longterm", "obb_params"),
    ("mobilenetv2_rectifier_finetune", "rectifier_box"),
    ("mobilenetv2_rectifier_gpu_nopretrained", "rectifier_box"),
    ("mobilenetv2_rectifier_hardcase_finetune", "rectifier_box"),
    ("mobilenetv2_rectifier_hardcase_finetune_v2", "rectifier_box"),
    ("mobilenetv2_rectifier_hardcase_finetune_v3", "rectifier_box"),
    ("mobilenetv2_rectifier_rectified_boxes_all_v1", "rectifier_box"),
    ("mobilenetv2_rectifier_rectified_boxes_v1", "rectifier_box"),
    ("mobilenetv2_rectifier_zoom_aug_rectified_boxes_all_v1", "rectifier_box"),
    ("mobilenetv2_rectifier_zoom_aug_rectified_boxes_all_v2", "rectifier_box"),
    ("mobilenetv2_rectifier_zoom_aug_v4", "rectifier_box"),
    ("mobilenetv2_source_crop_box_v1", "source_crop_box"),
    ("sanity_geometry_transfer_v1", "keypoint_coords"),
]


def main():
    manifest = DEFAULT_MANIFEST
    capture_root = DEFAULT_CAPTURE_ROOT
    weights = DEFAULT_WEIGHTS
    output_dir = DEFAULT_OUTPUT_DIR
    gauge_id = DEFAULT_GAUGE_ID
    image_size = 224
    center_search_px = 5
    center_mode = "image_center"

    items = eval_mod._load_manifest(manifest, capture_root)
    items = eval_mod._select_items(items, max_samples=0, shuffle=False, seed=13)
    print(f"[BATCH] Loaded {len(items)} samples from {manifest}", flush=True)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    gauge_spec = load_gauge_specs()[gauge_id]
    exact_model = build_polar_vote_v28_model(
        polar_size=image_size,
        input_channels=7,
        base_filters=32,
        head_units=128,
        dropout=0.2,
    )
    exact_model.load_weights(str(weights))
    print("[BATCH] V28 model loaded.", flush=True)

    results = []
    for checkpoint_name, head in LOCALIZERS:
        model_path = PROJECT_ROOT / "artifacts" / "training" / checkpoint_name / "model.keras"
        if not model_path.exists():
            print(f"[SKIP] {checkpoint_name}: model.keras not found", flush=True)
            continue

        cp_output_dir = output_dir / checkpoint_name
        cp_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[EVAL] {checkpoint_name} head={head}", flush=True)
        try:
            localizer_session = load_model_session(model_path, "auto")
        except Exception as e:
            print(f"[FAIL] {checkpoint_name}: load error {e}", flush=True)
            results.append({
                "checkpoint": checkpoint_name, "head": head,
                "status": "load_error", "mae": None, "rmse": None, "max_error": None,
            })
            continue

        rows = []
        for index, item in enumerate(items, start=1):
            row = eval_mod._evaluate_one(
                item,
                localizer_session=localizer_session,
                exact_model=exact_model,
                gauge_spec=gauge_spec,
                image_size=image_size,
                localizer_head=head,
                center_search_px=center_search_px,
                center_mode=center_mode,
                obb_crop_scale=0.83,
                obb_width_scale=1.0,
                obb_height_scale=1.0,
                obb_source_width_scale=1.0,
                obb_source_height_scale=1.0,
                obb_min_crop_size=48.0,
                obb_center_x_bias_pixels=0.0,
                obb_center_y_bias_pixels=0.0,
                obb_source_x_bias_pixels=0.0,
                obb_source_y_bias_pixels=0.0,
                keypoint_crop_scale=0.83,
                keypoint_center_x_bias_pixels=0.0,
                keypoint_center_y_bias_pixels=0.0,
                keypoint_min_crop_size=48.0,
            )
            rows.append(row)

        metrics = eval_mod._summarize(rows, manifest, weights, gauge_id)
        eval_mod._write_outputs(rows, metrics, output_dir=cp_output_dir)

        print(
            f"[DONE] {checkpoint_name}: MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} "
            f"MAX={metrics['max_abs_error']:.3f} accepted={metrics['accepted_count']}",
            flush=True,
        )
        results.append({
            "checkpoint": checkpoint_name, "head": head, "status": "ok",
            "mae": metrics["mae"], "rmse": metrics["rmse"],
            "max_error": metrics["max_abs_error"],
            "accepted": metrics["accepted_count"], "rejected": metrics["rejected_count"],
        })

        # Free localizer memory without touching V28
        del localizer_session
        # tf.keras.backend.clear_session()  # disabled to keep V28 loaded

    # Summary CSV
    summary_csv = output_dir / "batch_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "head", "status", "mae", "rmse", "max_error", "accepted", "rejected"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n[BATCH] Summary written to {summary_csv}", flush=True)
    ok_results = [r for r in results if r["status"] == "ok"]
    ok_results.sort(key=lambda x: x["mae"])
    print("\n=== TOP 5 BY MAE ===", flush=True)
    for r in ok_results[:5]:
        print(f"  {r['checkpoint']} (head={r['head']}): MAE={r['mae']:.3f} RMSE={r['rmse']:.3f} MAX={r['max_error']:.3f}", flush=True)


if __name__ == "__main__":
    main()
