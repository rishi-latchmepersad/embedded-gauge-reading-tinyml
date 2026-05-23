# Geometry Heatmap v3 Canonical Trainer Validation Contract

- Decoder: softargmax w3
- Validation split: val
- Validation count: 47
- Validation preprocessing mode: python_training_rgb_bilinear
- Validation resize method: rgb_bilinear
- Validation channel strategy: rgb
- Validation normalization: uint8_to_float32_0_1
- Validation source kind: rgb
- Canonical loader: load_split_samples
- Canonical preprocessing: python_training_rgb_bilinear
- Validation scoring path: model(x, training=False) via ReplayMetricCallback
- Legacy validation scoring used: no
- Uses build_board_replay_sample for validation: yes
- Uses load_heatmap_sample for validation: no
- Guardrails path: /mnt/d/Projects/embedded-gauge-reading-tinyml/ml/artifacts/training/geometry_heatmap_v2_board_replay/selected_board_guardrail_thresholds.json
- Calibration path: /mnt/d/Projects/embedded-gauge-reading-tinyml/ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json
- Fake quant/noise active at inference: no