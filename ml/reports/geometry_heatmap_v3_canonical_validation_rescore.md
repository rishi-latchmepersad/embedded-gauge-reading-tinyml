# Geometry Heatmap v3 Canonical Validation Rescore

- Decoder: softargmax w3
- Calibration candidate: D_robust_linear
- Validation rows: 47

- Keras accepted MAE: 3.6002 C
- Acceptance: 0.7021
- Worst accepted error: 13.5297 C
- Accepted >20 C failures: 0
- Rejection reasons: tip_peak_too_low:12;predicted_angle_outside_valid_sweep:5;temperature_outside_physical_margin:5;center_tip_distance_ratio_implausible:3
- Center MAE px: 5.5034
- Tip MAE px: 20.4946
- Angle MAE deg: 36.6646
- Heatmap center peak/spread mean: 0.6779 / 23.0759
- Heatmap tip peak/spread mean: 0.5885 / 19.2109