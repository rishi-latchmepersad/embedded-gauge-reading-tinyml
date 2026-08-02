"""Evaluate full pipeline on the existing 97-image test set."""
import sys
sys.path.insert(0, "scripts")
import numpy as np, json
import tensorflow as tf
from pathlib import Path

# Load ellipse model
ell_path = Path("artifacts/gauge_ellipse_qat_encoder_v1/ellipse_qat_encoder_int8.tflite")
ell_interp = tf.lite.Interpreter(model_path=str(ell_path))
ell_interp.allocate_tensors()
ell_in, ell_out = ell_interp.get_input_details()[0], ell_interp.get_output_details()[0]

# Load center/tip model
ct_path = Path("artifacts/gauge_center_tip_littlegood_v9/gauge_center_tip_v8_int8.tflite")
ct_interp = tf.lite.Interpreter(model_path=str(ct_path))
ct_interp.allocate_tensors()
ct_in, ct_out = ct_interp.get_input_details()[0], ct_interp.get_output_details()[0]

# Load test data
test = np.load("data/initial_temp_gauge_v1/student_conditioned/test.npz")
test_inputs = test["inputs"]  # (97, 160, 160, 2)
test_points = test["points"]  # (97, 2, 2) normalized [0,1]

# For the existing test set, the ellipse is already conditioned (channel 1 is the mask).
# We need to evaluate the center/tip model directly since the ellipse is already baked in.
print("Evaluating v9 center/tip on 97-image test set (existing conditioned crops):")
predictions = []
for sample in test_inputs:
    s, zp = ct_in["quantization"]
    t = np.clip(np.round(sample[None] / float(s) + float(zp)), -128, 127).astype(np.int8)
    ct_interp.set_tensor(ct_in["index"], t)
    ct_interp.invoke()
    raw = ct_interp.get_tensor(ct_out["index"]).astype(np.float32)
    s, zp = ct_out["quantization"]
    predictions.append((raw - float(zp)) * float(s))
predictions = np.concatenate(predictions)

# Decode heatmaps
size = 80
decoded = []
for sample in predictions:
    row = []
    for ch in range(2):
        hm = sample[..., ch]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        y0, y1 = max(0, y - 4), min(size, y + 5)
        x0, x1 = max(0, x - 4), min(size, x + 5)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        w = np.maximum(hm[y0:y1, x0:x1] - 0.03, 0) ** 2
        total = float(w.sum())
        if total > 0:
            pt = np.asarray([(xx * w).sum() / total + 0.5, (yy * w).sum() / total + 0.5], np.float32) / size
        else:
            pt = np.asarray([(x + 0.5) / size, (y + 0.5) / size], np.float32)
        row.append(pt)
    decoded.append(row)
decoded = np.asarray(decoded, np.float32)

errors = np.linalg.norm((decoded - test_points) * 160, axis=2)
ec, et = errors[:, 0], errors[:, 1]

# Angle error
c_t_pred = decoded[:, 1] - decoded[:, 0]
c_t_gt = test_points[:, 1] - test_points[:, 0]
angle_pred = np.arctan2(c_t_pred[:, 1], c_t_pred[:, 0])
angle_gt = np.arctan2(c_t_gt[:, 1], c_t_gt[:, 0])
angle_err = np.abs(np.rad2deg(np.arctan2(np.sin(angle_pred - angle_gt), np.cos(angle_pred - angle_gt))))

print(f"Center <=8px: {np.mean(ec <= 8):.1%}")
print(f"Tip <=8px:    {np.mean(et <= 8):.1%}")
print(f"Center mean: {ec.mean():.1f}px  median: {np.median(ec):.1f}px  p90: {np.percentile(ec, 90):.1f}px")
print(f"Tip mean:    {et.mean():.1f}px  median: {np.median(et):.1f}px  p90: {np.percentile(et, 90):.1f}px")
print(f"Angle <=5deg: {np.mean(angle_err <= 5):.1%}")
print(f"Angle <=10deg: {np.mean(angle_err <= 10):.1%}")
print(f"Angle mean: {angle_err.mean():.1f}deg  median: {np.median(angle_err):.1f}deg")
