"""Measure offset between dial center and needle pivot in training space."""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.training import _build_training_examples, TrainConfig
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs

ML_ROOT = Path(__file__).resolve().parents[1]
samples = load_dataset(labelled_dir=ML_ROOT/"data/labelled", raw_dir=ML_ROOT/"data/raw")
spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
examples, _ = _build_training_examples(samples, spec, image_height=224, image_width=224,
    keypoint_heatmap_size=28, strict_labels=False, crop_pad_ratio=0.25)

diffs_x, diffs_y, dists = [], [], []
for ex in examples:
    # obb_params[0,1] = dial center in normalized crop space
    dial_cx = ex.obb_params[0] * 224
    dial_cy = ex.obb_params[1] * 224
    pivot_cx, pivot_cy = ex.center_xy
    dx = dial_cx - pivot_cx
    dy = dial_cy - pivot_cy
    dist = math.hypot(dx, dy)
    diffs_x.append(dx)
    diffs_y.append(dy)
    dists.append(dist)

arr_x = np.array(diffs_x)
arr_y = np.array(diffs_y)
arr_d = np.array(dists)
print(f"Dial center → Needle pivot offset (in 224×224 crop space, n={len(arr_d)}):")
print(f"  dx: mean={arr_x.mean():.2f}  std={arr_x.std():.2f}  median={np.median(arr_x):.2f}")
print(f"  dy: mean={arr_y.mean():.2f}  std={arr_y.std():.2f}  median={np.median(arr_y):.2f}")
print(f"  Euclidean: mean={arr_d.mean():.2f}  std={arr_d.std():.2f}  median={np.median(arr_d):.2f}")
print(f"  Max: {arr_d.max():.2f}")
print(f"  ≤1px: {(arr_d<=1).mean()*100:.0f}%  ≤2px: {(arr_d<=2).mean()*100:.0f}%  ≤5px: {(arr_d<=5).mean()*100:.0f}%")
