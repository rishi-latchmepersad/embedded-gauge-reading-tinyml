"""Compare spoked-arc refinement vs baseline polar voter with GT center."""
from __future__ import annotations
import sys, math
from pathlib import Path
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

H, W = 224, 224
ML_ROOT = Path(__file__).resolve().parents[1]

print("[Eval] Loading data...")
samples = load_dataset(labelled_dir=ML_ROOT/"data/labelled", raw_dir=ML_ROOT/"data/raw")
spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
examples, _ = _build_training_examples(samples, spec, image_height=H, image_width=W,
    keypoint_heatmap_size=28, strict_labels=False, crop_pad_ratio=0.25)
config = TrainConfig(gauge_id="littlegood_home_temp_gauge_c", seed=21, test_fraction=0.15, val_fraction=0.15)
split = _split_examples(examples, config)
test_examples = split.test_examples
print(f"  {len(test_examples)} test examples")

print("[Eval] Preloading test images...")
img_data = []
for ex in test_examples:
    img, _ = _load_crop_and_preprocess_image(ex.image_path, 0.0, ex.crop_box_xyxy, H, W)
    img_data.append((img.numpy(), math.degrees(math.atan2(ex.needle_unit_xy[1], ex.needle_unit_xy[0])) % 360, ex.center_xy, Path(ex.image_path).name))

results = []
for x_test, gt_deg, gt_center, fname in img_data:
    img_uint8 = (x_test * 255).astype(np.uint8) if x_test.max() <= 1.0 else x_test.astype(np.uint8)
    img_bgr = img_uint8[..., ::-1].copy()
    dial_radius = 0.56 * 123

    polar_res = _detect_needle_unit_vector_polar(img_bgr, center_xy=tuple(gt_center), dial_radius_px=dial_radius, gauge_spec=spec)
    spoke_res = _detect_needle_unit_vector_spoked_arc(img_bgr, center_xy=tuple(gt_center), dial_radius_px=dial_radius, gauge_spec=spec)

    def get_angle(r):
        if r is None: return None
        return math.degrees(math.atan2(r.unit_dy, r.unit_dx)) % 360

    def err(pa, ga):
        if pa is None: return None
        return abs((pa - ga + 180) % 360 - 180)

    polar_a = get_angle(polar_res)
    spoke_a = get_angle(spoke_res)
    polar_err = err(polar_a, gt_deg)
    spoke_err = err(spoke_a, gt_deg)

    results.append((fname, gt_deg, polar_a, spoke_a, polar_err, spoke_err))

# Print detailed table
print(f"\n{'File':40s} {'GT°':6s} {'Polar°':7s} {'Spoke°':7s} {'ΔPol':6s} {'ΔSpk':6s}")
print(f"{'-'*40} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*6}")
for r in results:
    pa = f"{r[2]:6.1f}" if r[2] is not None else " FAIL "
    sa = f"{r[3]:6.1f}" if r[3] is not None else " FAIL "
    pe = f"{r[4]:6.2f}" if r[4] is not None else "  N/A "
    se = f"{r[5]:6.2f}" if r[5] is not None else "  N/A "
    # Mark improvements
    marker = ""
    if r[4] is not None and r[5] is not None:
        if r[5] < r[4]:
            marker = " <-- better"
        elif r[5] > r[4] + 2:
            marker = " <-- worse"
    print(f"{r[0]:40s} {r[1]:6.1f} {pa} {sa} {pe} {se}{marker}")

# Summaries
def clean(arr):
    return np.array([v for v in arr if v is not None], dtype=float)

polar_errs = clean([r[4] for r in results])
spoke_errs = clean([r[5] for r in results])

def summary(name, errs):
    if len(errs) == 0:
        return f"{name:30s} ALL FAILED"
    in2 = (errs <= 2).mean() * 100
    in5 = (errs <= 5).mean() * 100
    in10 = (errs <= 10).mean() * 100
    return f"{name:30s} MAE={errs.mean():6.2f}° median={np.median(errs):6.2f}° rmse={np.sqrt((errs**2).mean()):6.2f}°  ≤2°={in2:.0f}%  ≤5°={in5:.0f}%  ≤10°={in10:.0f}% (n={len(errs)})"

print(f"\n{'='*80}")
print(summary("Polar (baseline)", polar_errs))
print(summary("Spoked-arc (new)", spoke_errs))

# Count improvements
improved = sum(1 for r in results if r[4] is not None and r[5] is not None and r[5] < r[4])
worsened = sum(1 for r in results if r[4] is not None and r[5] is not None and r[5] > r[4])
same = sum(1 for r in results if r[4] is not None and r[5] is not None and r[5] == r[4])
total = improved + worsened + same
print(f"\nImproved: {improved}/{total}  Worsened: {worsened}/{total}  Same: {same}/{total}")
