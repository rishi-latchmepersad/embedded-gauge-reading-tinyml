"""Evaluate OBB center accuracy on ALL samples to check if split changed."""
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

obb = keras.models.load_model(
    ML_ROOT/"artifacts/training/obb_improved_20260530_194719/model.keras",
    compile=False, custom_objects=CUSTOM_OBJECTS)

for split_name, split_examples in [("train", split.train_examples), ("val", split.val_examples), ("test", split.test_examples)]:
    all_dists = []
    for ex in split_examples:
        path = Path(ex.image_path)
        gt_cx, gt_cy = ex.center_xy
        img, _ = _load_crop_and_preprocess_image(str(path), 0.0, ex.crop_box_xyxy, 224, 224)
        img_batch = tf.expand_dims(img, 0)
        pred = obb.predict(img_batch, verbose=0)
        if isinstance(pred, dict): pred = pred["obb_params"]
        pred = pred[0]
        pred_cx, pred_cy = float(pred[0]) * 224, float(pred[1]) * 224
        all_dists.append(math.hypot(pred_cx - gt_cx, pred_cy - gt_cy))
    
    arr = np.array(all_dists)
    print(f"{split_name} (n={len(arr)}): center MAE={arr.mean():.2f}px RMSE={np.sqrt((arr**2).mean()):.2f}px max={arr.max():.2f}px median={np.median(arr):.2f}px")
    print(f"   ≤2px: {(arr<=2).mean()*100:.0f}%  ≤5px: {(arr<=5).mean()*100:.0f}%  ≤10px: {(arr<=10).mean()*100:.0f}%")
