"""Diagnose why OBB+polar gives worse results: check OBB center errors vs GT center on test set."""
from __future__ import annotations
import math, sys
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.training import (
    _build_training_examples, _split_examples, TrainConfig,
    _load_crop_and_preprocess_image,
)
from embedded_gauge_reading_tinyml.models import (
    SpatialSoftArgmax2D, GaugeValueFromKeypoints, GaugeValueFromNeedleDirection,
    OrderedCornerBox, CornerKeypointsToBox,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs

ML_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_OBJECTS = {
    "preprocess_input": keras.applications.mobilenet_v2.preprocess_input,
    "SpatialSoftArgmax2D": SpatialSoftArgmax2D,
    "GaugeValueFromKeypoints": GaugeValueFromKeypoints,
    "GaugeValueFromNeedleDirection": GaugeValueFromNeedleDirection,
    "OrderedCornerBox": OrderedCornerBox,
    "CornerKeypointsToBox": CornerKeypointsToBox,
}

samples = load_dataset(labelled_dir=ML_ROOT/"data/labelled", raw_dir=ML_ROOT/"data/raw")
spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
examples, _ = _build_training_examples(samples, spec, image_height=224, image_width=224,
    keypoint_heatmap_size=28, strict_labels=False, crop_pad_ratio=0.25)
config = TrainConfig(gauge_id="littlegood_home_temp_gauge_c", seed=21, test_fraction=0.15, val_fraction=0.15)
split = _split_examples(examples, config)
test = split.test_examples[:20]

obb = keras.models.load_model(
    ML_ROOT/"artifacts/training/obb_improved_20260530_194719/model.keras",
    compile=False, custom_objects=CUSTOM_OBJECTS)

errors_center = []
errors_angle = []
for i, ex in enumerate(test):
    path = Path(ex.image_path)
    gt_cx, gt_cy = ex.center_xy
    gt_angle = math.degrees(math.atan2(ex.needle_unit_xy[1], ex.needle_unit_xy[0])) % 360

    img, _ = _load_crop_and_preprocess_image(str(path), 0.0, ex.crop_box_xyxy, 224, 224)
    img_batch = tf.expand_dims(img, 0)
    pred = obb.predict(img_batch, verbose=0)
    if isinstance(pred, dict): pred = pred["obb_params"]
    pred = pred[0]

    pred_cx, pred_cy = float(pred[0]) * 224, float(pred[1]) * 224
    center_dist = math.hypot(pred_cx - gt_cx, pred_cy - gt_cy)
    errors_center.append(center_dist)

    # Angle error from OBB center + polar
    dial_radius = 0.56 * 123

    from embedded_gauge_reading_tinyml.baseline_classical_cv import _detect_needle_unit_vector_polar
    # baseline_classical_cv uses BGR input
    img_bgr = (img.numpy()[..., ::-1] * 255).astype(np.uint8)
    result = _detect_needle_unit_vector_polar(
        img_bgr, center_xy=(pred_cx, pred_cy), dial_radius_px=dial_radius, gauge_spec=spec)
    if result is not None:
        pred_angle = math.degrees(math.atan2(result.unit_dy, result.unit_dx)) % 360
        angle_err = min(abs(pred_angle - gt_angle), 360 - abs(pred_angle - gt_angle))
        errors_angle.append(angle_err)

    # Also try with GT center
    result_gt = _detect_needle_unit_vector_polar(
        img_bgr, center_xy=(gt_cx, gt_cy), dial_radius_px=dial_radius, gauge_spec=spec)
    gt_angle_err = None
    if result_gt is not None:
        pred_gt = math.degrees(math.atan2(result_gt.unit_dy, result_gt.unit_dx)) % 360
        gt_angle_err = min(abs(pred_gt - gt_angle), 360 - abs(pred_gt - gt_angle))

    print(f"  [{i:2d}] {path.name}: gt_center=({gt_cx:5.1f},{gt_cy:5.1f}) "
          f"obb_center=({pred_cx:5.1f},{pred_cy:5.1f}) "
          f"Δc={center_dist:4.1f}px  "
          f"angle_err={angle_err:5.1f}° (OBBC)  "
          f"{'gt_angle_err='+f'{gt_angle_err:5.1f}°' if gt_angle_err is not None else 'FAILED'} (GTC)")

arr_c = np.array(errors_center)
arr_a = np.array(errors_angle)
print(f"\n=== Center Error ===")
print(f"  MAE: {arr_c.mean():.2f} px  RMSE: {np.sqrt((arr_c**2).mean()):.2f} px")
print(f"  Max: {arr_c.max():.2f} px  Median: {np.median(arr_c):.2f} px")
print(f"\n=== Angle Error (OBB center) ===")
print(f"  MAE: {arr_a.mean():.2f}°  RMSE: {np.sqrt((arr_a**2).mean()):.2f}°")
print(f"  Median: {np.median(arr_a):.2f}°")
print(f"  ≤2°: {(arr_a<=2).mean()*100:.0f}%  ≤5°: {(arr_a<=5).mean()*100:.0f}%  ≤10°: {(arr_a<=10).mean()*100:.0f}%")
print(f"\n=== Correlation (center_err vs angle_err) ===")
corr = np.corrcoef(arr_c, arr_a)[0, 1] if len(arr_c) == len(arr_a) else float("nan")
print(f"  Pearson r: {corr:.3f}")
