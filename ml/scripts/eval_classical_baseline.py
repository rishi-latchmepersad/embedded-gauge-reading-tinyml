"""Evaluate classical CV baseline on the same test set as the OBB+center pipeline.

Loads the same 52 test examples using the same loading + splitting logic.
For each test image, runs the classical baseline pipeline
(center hypotheses + polar voting) and reports needle angle error and
center distance error vs. ground truth.
"""
from __future__ import annotations
import sys, math
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedded_gauge_reading_tinyml.training import (
    TrainConfig, _build_training_examples, _split_examples,
    _load_crop_and_preprocess_image,
)
from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    run_classical_baseline,
    _center_hypotheses,
    _select_best_detection,
    _detect_needle_unit_vector_polar,
    detect_needle_unit_vector,
    needle_detection_quality,
)

ML_ROOT = Path(__file__).resolve().parents[1]
H, W = 224, 224

# ── 1. Load the same 52 test examples ──────────────────────────────────
print("[Eval] Loading data...")
samples = load_dataset(labelled_dir=ML_ROOT / "data/labelled", raw_dir=ML_ROOT / "data/raw")
spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
examples, _ = _build_training_examples(
    samples, spec,
    image_height=H, image_width=W,
    keypoint_heatmap_size=28, strict_labels=False, crop_pad_ratio=0.25,
)
config = TrainConfig(
    gauge_id="littlegood_home_temp_gauge_c",
    seed=21, test_fraction=0.15, val_fraction=0.15,
)
split = _split_examples(examples, config)
test_examples: list = split.test_examples
print(f"  {len(test_examples)} test examples")



# ── 2. Evaluate each test image ────────────────────────────────────────
# We preload images as in the OBB eval, then run the classical pipeline
# on the same cropped + resized 224×224 image.
print("[Eval] Evaluating classical baseline on cropped images...")

results: list[tuple] = []

# Determine the dial radius in the cropped image (same formula as eval script)
dial_radius_px: float = 0.56 * 123.0  # approx radius in 224px crop

for i, ex in enumerate(test_examples):
    # Load the cropped + resized 224×224 image (same pipeline as eval script)
    img_tensor, _ = _load_crop_and_preprocess_image(
        ex.image_path, 0.0, ex.crop_box_xyxy, H, W,
    )
    img_np: np.ndarray = img_tensor.numpy()

    # Convert TF-normalized [0,1] float RGB → uint8 BGR for OpenCV
    img_uint8: np.ndarray = (img_np * 255).astype(np.uint8)
    img_bgr: np.ndarray = img_uint8[..., ::-1].copy()

    # ── Classical pipeline ──────────────────────────────────────────
    # 1. Generate center hypotheses on the cropped image
    hypotheses = _center_hypotheses(img_bgr)

    # 2. Run polar voting for each hypothesis, pick best by confidence
    best_result = _select_best_detection(img_bgr, hypotheses, spec)

    # ── Ground truth (cropped-image space) ──────────────────────────
    gt_center_xy: tuple[float, float] = ex.center_xy
    gt_angle_deg: float = (
        math.degrees(math.atan2(ex.needle_unit_xy[1], ex.needle_unit_xy[0])) % 360
    )

    # ── Parse classical result ──────────────────────────────────────
    if best_result is not None:
        best_hyp, best_det = best_result
        pred_center_xy = best_hyp.center_xy

        pred_angle_deg = (
            math.degrees(math.atan2(best_det.unit_dy, best_det.unit_dx)) % 360
        )

        angle_err = abs((pred_angle_deg - gt_angle_deg + 180) % 360 - 180)
        center_dist = float(np.linalg.norm(
            np.array(pred_center_xy) - np.array(gt_center_xy)
        ))
        succeeded = True
    else:
        pred_center_xy = (math.nan, math.nan)
        pred_angle_deg = math.nan
        angle_err = math.nan
        center_dist = math.nan
        succeeded = False

    results.append((
        Path(ex.image_path).name,
        gt_angle_deg,
        pred_angle_deg,
        angle_err,
        center_dist,
        gt_center_xy,
        pred_center_xy,
        succeeded,
    ))

    if (i + 1) % 10 == 0 or i == len(test_examples) - 1:
        print(f"  [{i+1}/{len(test_examples)}] done")

# ── 3. Also run the baseline using the GT center directly (oracle) ────
print("[Eval] Running with GT center (oracle upper bound)...")
oracle_results: list[tuple] = []
for i, ex in enumerate(test_examples):
    img_tensor, _ = _load_crop_and_preprocess_image(
        ex.image_path, 0.0, ex.crop_box_xyxy, H, W,
    )
    img_np = img_tensor.numpy()
    img_uint8 = (img_np * 255).astype(np.uint8)
    img_bgr = img_uint8[..., ::-1].copy()

    gt_center_xy = ex.center_xy
    gt_angle_deg = (
        math.degrees(math.atan2(ex.needle_unit_xy[1], ex.needle_unit_xy[0])) % 360
    )

    # Run polar voting with GT center
    det = _detect_needle_unit_vector_polar(
        img_bgr, center_xy=tuple(gt_center_xy),
        dial_radius_px=dial_radius_px, gauge_spec=spec,
    )

    if det is not None:
        pred_angle_deg = (
            math.degrees(math.atan2(det.unit_dy, det.unit_dx)) % 360
        )
        angle_err = abs((pred_angle_deg - gt_angle_deg + 180) % 360 - 180)
        succeeded = True
    else:
        pred_angle_deg = math.nan
        angle_err = math.nan
        succeeded = False

    oracle_results.append((Path(ex.image_path).name, gt_angle_deg, pred_angle_deg, angle_err, succeeded))

# ── 4. Print per-sample table ─────────────────────────────────────────
print(f"\n{'='*120}")
header = f"{'File':40s} {'GT°':7s} {'Pred°':7s} {'Err°':7s} {'CtrDist':8s} {'OK':4s}"
print(header)
print("-" * 120)
for r in results:
    fname, gt, pred, err, cdist, _, _, ok = r
    err_str = f"{err:7.2f}" if ok else "   FAIL"
    cdist_str = f"{cdist:8.2f}" if ok else "    N/A"
    ok_str = "YES" if ok else "NO"
    pred_str = f"{pred:7.1f}" if ok else "   N/A"
    print(f"{fname:40s} {gt:7.1f} {pred_str} {err_str} {cdist_str} {ok_str:4s}")

# ── 5. Aggregate metrics ──────────────────────────────────────────────
def summary(name: str, angle_errs: np.ndarray, center_dists: np.ndarray | None = None):
    n = len(angle_errs)
    if n == 0:
        return f"{name:40s} ALL FAILED"
    mae = float(np.mean(angle_errs))
    rmse = float(np.sqrt(np.mean(angle_errs ** 2)))
    med = float(np.median(angle_errs))
    in2 = float((angle_errs <= 2.0).mean() * 100)
    in5 = float((angle_errs <= 5.0).mean() * 100)
    in10 = float((angle_errs <= 10.0).mean() * 100)
    line = (f"{name:40s} MAE={mae:6.2f}° median={med:6.2f}° rmse={rmse:6.2f}°  "
            f"≤2°={in2:.0f}%  ≤5°={in5:.0f}%  ≤10°={in10:.0f}% (n={n})")
    if center_dists is not None:
        cd_ok = center_dists[~np.isnan(center_dists)]
        if len(cd_ok) > 0:
            line += f"  center_dist={np.mean(cd_ok):.2f}±{np.std(cd_ok):.2f}px"
    return line

cls_errs = np.array([r[3] for r in results if r[7]], dtype=float)
cls_dists = np.array([r[4] for r in results if r[7]], dtype=float)
all_dists = np.array([r[4] for r in results], dtype=float)

oracle_errs = np.array([r[3] for r in oracle_results if r[4]], dtype=float)

total = len(results)
succeeded = len(cls_errs)
failed = total - succeeded

print(f"\n{'='*120}")
print(summary("Classical baseline (best hypothesis)", cls_errs, cls_dists))
print(summary("Oracle (GT center + polar voting)", oracle_errs))
print(f"\n  Total: {total}  Succeeded: {succeeded}  Failed: {failed}  "
      f"Success rate: {succeeded/total*100:.1f}%")

# Center dist stats over ALL samples (including failures → NaN)
print(f"\n  Center distance (all): mean={np.nanmean(all_dists):.2f}±{np.nanstd(all_dists):.2f}px")
