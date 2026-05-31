"""Run validate_polar_voting.py with OBB-predicted centers."""
from __future__ import annotations
import math, sys
from pathlib import Path
import keras, numpy as np, tensorflow as tf

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
from embedded_gauge_reading_tinyml.hybrid_localizer import rgb_to_luma, polar_spoke_vote, smooth_and_find_peak

import validate_polar_voting as vpv  # reuse its _detect_needle_polar

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
split = vpv._split_examples(examples, config)
test = split.test_examples[:20]

obb = keras.models.load_model(
    ML_ROOT/"artifacts/training/obb_improved_20260530_194719/model.keras",
    compile=False, custom_objects=CUSTOM_OBJECTS)

errors = []
for i, ex in enumerate(test):
    path = Path(ex.image_path)
    img, _ = vpv._load_crop_and_preprocess_image(str(path), 0.0, ex.crop_box_xyxy, 224, 224)
    img_batch = tf.expand_dims(img, 0)
    pred = obb.predict(img_batch, verbose=0)
    if isinstance(pred, dict): pred = pred["obb_params"]
    pred = pred[0]
    cx, cy = float(pred[0]) * 224, float(pred[1]) * 224

    img_u8 = (img.numpy() * 255).astype(np.uint8)
    dial_r = 0.56 * 123
    pred_angle = vpv._detect_needle_polar(img_u8, center_x=cx, center_y=cy, dial_radius=dial_r, gauge_spec=spec)

    gt_angle = math.degrees(math.atan2(ex.needle_unit_xy[1], ex.needle_unit_xy[0])) % 360
    if pred_angle is not None:
        diff = vpv._angular_error_deg(gt_angle, pred_angle)
        errors.append(diff)
        print(f"  [{i:2d}] {path.name}: gt={gt_angle:6.1f}° pred={pred_angle:6.1f}° err={diff:5.2f}°")
    else:
        print(f"  [{i:2d}] {path.name}: gt={gt_angle:6.1f}°  FAILED (no peak)")

    # Also show OBB center offset from GT center for diagnosis
    gt_cx = ex.needle_unit_xy[0] * 224  # approximate
    gt_cy = ex.needle_unit_xy[1] * 224
    # actual GT center comes from temp_center mapped through crop

arr = np.array(errors)
print(f"\n--- Summary ({len(arr)}/{len(test)} valid) ---")
print(f"  MAE: {arr.mean():.2f}°  RMSE: {np.sqrt((arr**2).mean()):.2f}°")
print(f"  Median: {np.median(arr):.2f}°")
print(f"  ≤2°: {(arr<=2).mean()*100:.0f}%  ≤5°: {(arr<=5).mean()*100:.0f}%  ≤10°: {(arr<=10).mean()*100:.0f}%")
