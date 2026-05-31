"""Evaluate OBB + center detector + polar voting pipeline end-to-end."""
from __future__ import annotations
import sys
from pathlib import Path

import math, keras
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedded_gauge_reading_tinyml.training import (
    TrainConfig, _build_training_examples, _split_examples,
    _load_crop_and_preprocess_image,
)
from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    _detect_needle_unit_vector_polar,
    _detect_needle_unit_vector_spoked_arc,
)
from embedded_gauge_reading_tinyml.models import (
    SpatialSoftArgmax2D, GaugeValueFromKeypoints, GaugeValueFromNeedleDirection,
    OrderedCornerBox, CornerKeypointsToBox,
)

CUSTOM_OBJECTS = {
    "preprocess_input": keras.applications.mobilenet_v2.preprocess_input,
    "SpatialSoftArgmax2D": SpatialSoftArgmax2D,
    "GaugeValueFromKeypoints": GaugeValueFromKeypoints,
    "GaugeValueFromNeedleDirection": GaugeValueFromNeedleDirection,
    "OrderedCornerBox": OrderedCornerBox,
    "CornerKeypointsToBox": CornerKeypointsToBox,
}

ML_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ML_ROOT / "artifacts" / "training"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--obb", type=str, default=str(ARTIFACTS / "obb_improved_20260530_194719" / "model.keras"))
parser.add_argument("--ctr", type=str, default=str(ARTIFACTS / "center_detector_v1_20260530_201137" / "best_model.keras"))
args = parser.parse_args()

OBB_PATH = Path(args.obb)
CTR_PATH = Path(args.ctr)

H, W = 224, 224

print("[Eval] Loading data...")
samples = load_dataset(labelled_dir=ML_ROOT/"data/labelled", raw_dir=ML_ROOT/"data/raw")
spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
examples, _ = _build_training_examples(samples, spec, image_height=H, image_width=W,
    keypoint_heatmap_size=28, strict_labels=False, crop_pad_ratio=0.25)
config = TrainConfig(gauge_id="littlegood_home_temp_gauge_c", seed=21, test_fraction=0.15, val_fraction=0.15)
split = _split_examples(examples, config)
test_examples = split.test_examples
print(f"  {len(test_examples)} test examples")

# Preload all test images
print("[Eval] Preloading test images...")
imgs, gts, img_names = [], [], []
for ex in test_examples:
    img, _ = _load_crop_and_preprocess_image(ex.image_path, 0.0, ex.crop_box_xyxy, H, W)
    imgs.append(img.numpy())
    gts.append(math.degrees(math.atan2(ex.needle_unit_xy[1], ex.needle_unit_xy[0])) % 360)
    img_names.append(Path(ex.image_path).name)
x_test = np.array(imgs)

print("[Eval] Loading models...")
obb = keras.models.load_model(str(OBB_PATH), compile=False, custom_objects=CUSTOM_OBJECTS)
ctr = keras.models.load_model(str(CTR_PATH), compile=False, custom_objects=CUSTOM_OBJECTS)

# Run OBB
print("[Eval] Running OBB...")
obb_out = obb.predict(x_test, verbose=0)
obb_preds = obb_out['obb_params'] if isinstance(obb_out, dict) else obb_out
obb_cx = obb_preds[:, 0]
obb_cy = obb_preds[:, 1]

# Run center detector
print("[Eval] Running center detector...")
ctr_out = ctr.predict(x_test, verbose=0)
ctr_cx = ctr_out['center_xy'][:, 0]
ctr_cy = ctr_out['center_xy'][:, 1]

results = []
for i, ex in enumerate(test_examples):
    gt_deg = gts[i]
    obb_center = np.array([obb_cx[i] * W, obb_cy[i] * H])
    ctr_center = np.array([ctr_cx[i] * W, ctr_cy[i] * H])
    gt_center = ex.center_xy  # needle pivot ground truth

    # Polar voting using each center source
    img_uint8 = (x_test[i] * 255).astype(np.uint8) if x_test[i].max() <= 1.0 else x_test[i].astype(np.uint8)
    img_bgr = img_uint8[..., ::-1].copy()  # RGB→BGR for opencv

    dial_radius = 0.56 * 123
    obb_res = _detect_needle_unit_vector_polar(img_bgr, center_xy=tuple(obb_center), dial_radius_px=dial_radius, gauge_spec=spec)
    ctr_res = _detect_needle_unit_vector_polar(img_bgr, center_xy=tuple(ctr_center), dial_radius_px=dial_radius, gauge_spec=spec)
    gt_res = _detect_needle_unit_vector_polar(img_bgr, center_xy=tuple(gt_center), dial_radius_px=dial_radius, gauge_spec=spec)
    obb_sa = _detect_needle_unit_vector_spoked_arc(img_bgr, center_xy=tuple(obb_center), dial_radius_px=dial_radius, gauge_spec=spec)
    ctr_sa = _detect_needle_unit_vector_spoked_arc(img_bgr, center_xy=tuple(ctr_center), dial_radius_px=dial_radius, gauge_spec=spec)
    gt_sa = _detect_needle_unit_vector_spoked_arc(img_bgr, center_xy=tuple(gt_center), dial_radius_px=dial_radius, gauge_spec=spec)

    def get_angle(res):
        if res is None: return None
        return math.degrees(math.atan2(res.unit_dy, res.unit_dx)) % 360

    obb_angle = get_angle(obb_res)
    ctr_angle = get_angle(ctr_res)
    gt_angle = get_angle(gt_res)
    obb_sa_angle = get_angle(obb_sa)
    ctr_sa_angle = get_angle(ctr_sa)
    gt_sa_angle = get_angle(gt_sa)

    def err_or_none(pred_a, gt_a):
        if pred_a is None: return None
        return abs((pred_a - gt_a + 180) % 360 - 180)

    obb_err = err_or_none(obb_angle, gt_deg)
    ctr_err = err_or_none(ctr_angle, gt_deg)
    gt_err = err_or_none(gt_angle, gt_deg)
    obb_sa_err = err_or_none(obb_sa_angle, gt_deg)
    ctr_sa_err = err_or_none(ctr_sa_angle, gt_deg)
    gt_sa_err = err_or_none(gt_sa_angle, gt_deg)
    obb_center_dist = np.linalg.norm(obb_center - gt_center)
    ctr_center_dist = np.linalg.norm(ctr_center - gt_center)

    results.append((img_names[i], gt_deg, obb_angle, ctr_angle, gt_angle,
                    obb_err, ctr_err, gt_err,
                    obb_sa_angle, ctr_sa_angle, gt_sa_angle,
                    obb_sa_err, ctr_sa_err, gt_sa_err,
                    obb_center_dist, ctr_center_dist))

# Print results
print(f"\n{'='*140}")
print(f"{'File':40s} {'GT°':6s} {'OBB°':6s} {'CTR°':6s} {'GT°':7s} {'SAOBB°':6s} {'SACTR°':7s} {'SAGT°':7s} {'dOBB':6s} {'dCTR':6s}")
print(f"{'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*6}")
for r in results:
    oa = f"{r[2]:6.1f}" if r[2] is not None else " FAIL "
    ca = f"{r[3]:6.1f}" if r[3] is not None else " FAIL "
    ga = f"{r[4]:7.1f}" if r[4] is not None else "  FAIL "
    soa = f"{r[8]:6.1f}" if r[8] is not None else " FAIL "
    sca = f"{r[9]:7.1f}" if r[9] is not None else "  FAIL "
    sga = f"{r[10]:7.1f}" if r[10] is not None else "  FAIL "
    oe = f"{r[5]:6.2f}" if r[5] is not None else "  N/A "
    ce = f"{r[6]:6.2f}" if r[6] is not None else "  N/A "
    print(f"{r[0]:40s} {r[1]:6.1f} {oa} {ca} {ga} {soa} {sca} {sga} {r[14]:6.2f} {r[15]:6.2f}")

# Summaries
def clean(arr):
    return np.array([v for v in arr if v is not None], dtype=float)

obb_errs = clean([r[5] for r in results])
ctr_errs = clean([r[6] for r in results])
gt_errs = clean([r[7] for r in results])
obb_sa_errs = clean([r[11] for r in results])
ctr_sa_errs = clean([r[12] for r in results])
gt_sa_errs = clean([r[13] for r in results])
obb_dists = np.array([r[14] for r in results])
ctr_dists = np.array([r[15] for r in results])

def summary(name, errs):
    if len(errs) == 0:
        return f"{name:30s} ALL FAILED"
    in2 = (errs <= 2).mean() * 100
    in5 = (errs <= 5).mean() * 100
    in10 = (errs <= 10).mean() * 100
    return f"{name:30s} MAE={errs.mean():6.2f}° median={np.median(errs):6.2f}° rmse={np.sqrt((errs**2).mean()):6.2f}°  ≤2°={in2:.0f}%  ≤5°={in5:.0f}%  ≤10°={in10:.0f}% (n={len(errs)})"

print(f"\n{'='*140}")
print(summary("OBB center (polar)", obb_errs))
print(summary("Center detector (polar)", ctr_errs))
print(summary("GT (polar)", gt_errs))
print(summary("OBB center (spoked-arc)", obb_sa_errs))
print(summary("Center detector (spoked-arc)", ctr_sa_errs))
print(summary("GT (spoked-arc)", gt_sa_errs))
print(f"\nCenter dist:  OBB={obb_dists.mean():.2f}±{obb_dists.std():.2f}px  CTR={ctr_dists.mean():.2f}±{ctr_dists.std():.2f}px")
