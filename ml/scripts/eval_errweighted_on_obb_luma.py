#!/usr/bin/env python3
"""Evaluate the polar_vote_hardcases_errweighted_v1_int8 model on OBB+luma crops,
using the same crop pipeline as the board pipeline (with rectifier fallback).

Computes both raw and calibrated metrics for the errweighted model and compares
with the scalar model from the same crop.

Usage:
  poetry run python scripts/eval_errweighted_on_obb_luma.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "tmp" / "matplotlib"))

from embedded_gauge_reading_tinyml.board_pipeline import (
    load_capture_image,
    load_model_session,
    predict_board_pipeline_on_capture,
    BoardPipelineResult,
)
from embedded_gauge_reading_tinyml.firmware_preprocessing import (
    build_training_style_polar_vote_float32,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs

MANIFEST: Path = PROJECT_ROOT / "data" / "hard_cases_plus_board30_valid_with_new6.csv"
CAPTURE_ROOT: Path = PROJECT_ROOT / "data" / "captured_images"
OBB_MODEL: Path = PROJECT_ROOT / "artifacts" / "deployment" / "prod_model_v0.3_obb_int8" / "model_int8.tflite"
RECTIFIER_MODEL: Path = PROJECT_ROOT / "artifacts" / "deployment" / "mobilenetv2_rectifier_hardcase_finetune_int8" / "model_int8.tflite"
SCALAR_MODEL: Path = PROJECT_ROOT / "artifacts" / "deployment" / "scalar_full_finetune_from_best_piecewise_calibrated_int8" / "model_int8.tflite"
ERRWT_TFLITE: Path = PROJECT_ROOT / "artifacts" / "deployment" / "polar_vote_hardcases_errweighted_v1_int8" / "model_int8.tflite"
CALIBRATION: Path = PROJECT_ROOT / "artifacts" / "calibration" / "prodv0_3_obb_scalar_calibration.json"
OUTPUT_DIR: Path = REPO_ROOT / "tmp" / "errweighted_pipeline_eval"

BIN_MIN: float = -30.0
BIN_MAX: float = 50.0
NUM_BINS: int = 224
DECODE_TOPK: int = 8
GAUGE_ID: str = "littlegood_home_temp_gauge_c"


class TopKExpectationDecoder:
    """Decode 224-bin logits via top-k expectation over the value range."""

    def __init__(self, min_val: float, max_val: float, num_bins: int, topk: int):
        self.bin_centers = np.linspace(min_val, max_val, num_bins, dtype=np.float32)
        self.topk = topk

    def __call__(self, logits: np.ndarray) -> float:
        if logits.ndim > 1:
            logits = logits.ravel()
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        topk_indices = np.argsort(probs)[-self.topk:]
        topk_probs = probs[topk_indices]
        topk_probs /= topk_probs.sum()
        return float(np.sum(self.bin_centers[topk_indices] * topk_probs))


def _resolve_path(raw: str) -> Path:
    p = Path(raw.replace("\\", "/"))
    return p if p.is_absolute() else REPO_ROOT / p


def _run_polar_tflite(errwt_session, polar_tensor: np.ndarray,
                      decoder: TopKExpectationDecoder) -> float:
    """Run errweighted TFLite on a 1x224x224x7 float [0,1] tensor."""
    intp = errwt_session.model
    in_det = errwt_session.input_details
    out_det = errwt_session.output_details
    in_scale, in_zp = float(in_det["quantization"][0]), int(in_det["quantization"][1])
    q = np.round(polar_tensor / in_scale + in_zp).clip(-128, 127).astype(np.int8)
    intp.set_tensor(int(in_det["index"]), q[None, ...])
    intp.invoke()
    raw_out = intp.get_tensor(int(out_det["index"]))[0]
    out_scale, out_zp = float(out_det["quantization"][0]), int(out_det["quantization"][1])
    logits = out_scale * (raw_out.astype(np.float32) - out_zp)
    return decoder(logits)


def _apply_piecewise_calibration(raw: float, cal_table: list[dict]) -> float:
    for seg in cal_table:
        lo, hi = seg["raw_range"]
        if lo <= raw <= hi:
            a, b = seg["coeffs"]
            return float(a * raw + b)
    return raw


def main() -> None:
    # Load manifest
    items: list[dict[str, Any]] = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ip = _resolve_path(row["image_path"])
            try:
                ip.resolve().relative_to(CAPTURE_ROOT.resolve())
            except ValueError:
                continue
            items.append({"path": ip, "value": float(row["value"]),
                          "weight": float(row.get("sample_weight", 1.0) or 1.0)})

    print(f"Loaded {len(items)} samples from {MANIFEST}")

    # Load all model sessions
    obb_s = load_model_session(str(OBB_MODEL), "tflite")
    rect_s = load_model_session(str(RECTIFIER_MODEL), "tflite")
    scalar_s = load_model_session(str(SCALAR_MODEL), "tflite")
    errwt_s = load_model_session(str(ERRWT_TFLITE), "tflite")
    gauge_spec = load_gauge_specs()[GAUGE_ID]
    decoder = TopKExpectationDecoder(BIN_MIN, BIN_MAX, NUM_BINS, DECODE_TOPK)
    cal_table: list[dict] = json.loads(CALIBRATION.read_text()).get("piecewise_calibration", [])
    print(f"Models loaded. Calibration: {len(cal_table)} segments.")

    # Evaluate each sample
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items, 1):
        source_image, _ = load_capture_image(item["path"])
        source_h, source_w = source_image.shape[:2]

        # 1. Run the board pipeline to get the crop, scalar prediction, etc.
        result: BoardPipelineResult = predict_board_pipeline_on_capture(
            capture_path=item["path"],
            obb_session=obb_s,
            rectifier_session=rect_s,
            scalar_session=scalar_s,
            image_size=224,
            obb_crop_scale=1.2,
            enable_luma_refinement=True,
            use_calibration=True,
            calibration_path=CALIBRATION,
        )

        crop_xyxy = result.selected_crop_box_xyxy
        scalar_raw = result.raw_prediction
        scalar_cal = result.calibrated_prediction

        # 2. Build polar tensor from the same crop
        polar_tensor = build_training_style_polar_vote_float32(
            source_image,
            crop_box_xyxy=crop_xyxy,
            output_dim=224,
            center_search_px=0,
            center_mode="image_center",
            gauge_spec=gauge_spec,
        )

        # 3. Run errweighted model
        errwt_pred = _run_polar_tflite(errwt_s, polar_tensor, decoder)
        errwt_cal = _apply_piecewise_calibration(errwt_pred, cal_table)

        scalar_err = abs(scalar_cal - item["value"])
        errwt_err = abs(errwt_pred - item["value"])
        errwt_cal_err = abs(errwt_cal - item["value"])

        rows.append({
            "image": item["path"].name,
            "gt": item["value"],
            "scalar_raw": scalar_raw,
            "scalar_cal": scalar_cal,
            "scalar_abs_err": scalar_err,
            "errwt_raw": errwt_pred,
            "errwt_cal": errwt_cal,
            "errwt_abs_err": errwt_err,
            "errwt_cal_abs_err": errwt_cal_err,
            "stage": result.selected_stage,
            "crop_xyxy": list(crop_xyxy),
        })

        print(f"  [{idx:03d}] {item['path'].name}  stage={result.selected_stage:16s}  "
              f"scalar={scalar_cal:6.2f}(err={scalar_err:5.2f})  "
              f"errwt={errwt_pred:6.2f}(err={errwt_err:5.2f})  "
              f"errwt_cal={errwt_cal:6.2f}(err={errwt_cal_err:5.2f})  "
              f"gt={item['value']:6.2f}")

    # Summarize
    def _print_metrics(name: str, err_key: str):
        vals = [r for r in rows if r.get(err_key) is not None]
        if not vals:
            print(f"  {name}: NO DATA")
            return
        errs = np.array([r[err_key] for r in vals])
        print(f"  {name} ({len(vals)}/{len(rows)}):  "
              f"MAE={errs.mean():6.2f}°C  "
              f"RMSE={np.sqrt((errs**2).mean()):6.2f}°C  "
              f"Med={np.median(errs):6.2f}°C  "
              f"p90={np.percentile(errs, 90):6.2f}°C  "
              f"Max={errs.max():6.2f}°C  "
              f"p5={np.percentile(errs, 5):6.2f}°C  "
              f"p95={np.percentile(errs, 95):6.2f}°C")

    print(f"\n{'='*70}")
    print(f"Reader comparison on full pipeline ({len(rows)} samples)")
    print(f"{'='*70}")
    _print_metrics("Scalar (calibrated)", "scalar_abs_err")
    _print_metrics("Errweighted (raw)", "errwt_abs_err")
    _print_metrics("Errweighted (calibrated)", "errwt_cal_abs_err")

    # Stage breakdown
    for stage in ("obb", "obb_plus_luma", "rectifier"):
        subset = [r for r in rows if r["stage"] == stage]
        if subset:
            errs = np.array([r["errwt_abs_err"] for r in subset])
            sel = np.array([r["scalar_abs_err"] for r in subset])
            print(f"\n  Stage={stage} ({len(subset)} samples):")
            print(f"    Scalar:    MAE={sel.mean():6.2f}°C  Med={np.median(sel):6.2f}°C")
            print(f"    Errweight: MAE={errs.mean():6.2f}°C  Med={np.median(errs):6.2f}°C")

    # Show worst/best errweighted
    rows_scalar = sorted(rows, key=lambda r: -r["scalar_abs_err"])
    rows_errwt = sorted(rows, key=lambda r: -r["errwt_abs_err"])
    print(f"\n  Worst 5 (Scalar):")
    for r in rows_scalar[:5]:
        print(f"    {r['image']}: scalar_err={r['scalar_abs_err']:6.2f} errwt_err={r['errwt_abs_err']:6.2f}")
    print(f"\n  Worst 5 (Errweighted):")
    for r in rows_errwt[:5]:
        print(f"    {r['image']}: errwt_err={r['errwt_abs_err']:6.2f} scalar_err={r['scalar_abs_err']:6.2f}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    err_arr = np.array([r["errwt_abs_err"] for r in rows])
    cal_err_arr = np.array([r["errwt_cal_abs_err"] for r in rows])
    scalar_err_arr = np.array([r["scalar_abs_err"] for r in rows])
    summary = {
        "num_samples": len(rows),
        "scalar_mae": float(scalar_err_arr.mean()),
        "scalar_rmse": float(np.sqrt((scalar_err_arr**2).mean())),
        "scalar_median": float(np.median(scalar_err_arr)),
        "errwt_mae": float(err_arr.mean()),
        "errwt_rmse": float(np.sqrt((err_arr**2).mean())),
        "errwt_median": float(np.median(err_arr)),
        "errwt_cal_mae": float(cal_err_arr.mean()),
        "errwt_cal_rmse": float(np.sqrt((cal_err_arr**2).mean())),
        "errwt_cal_median": float(np.median(cal_err_arr)),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUTPUT_DIR / "rows.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
