# Geometry Heatmap v2 Board Input Contract

## Firmware Must Reproduce This Exactly

- Preprocessing mode: `python_training_rgb_bilinear`
- Source crop coordinate system: the manifest loose crop box in source-image pixels (`crop_x1`, `crop_y1`, `crop_x2`, `crop_y2`).
- Crop selection: use the same loose crop geometry as the clean manifest row. Do not add random jitter in firmware.
- Input size: 224x224.
- Output heatmap size: 56x56 per keypoint.
- Channel order: RGB.
- Input normalization: uint8 `[0, 255]` to float32 `[0, 1]` by division by `255`.
- Resize rule: preserve aspect ratio, scale by `min(224 / crop_w, 224 / crop_h)`, then zero-pad to 224x224.
- Resize method: bilinear.
- Padding rule: center the resized crop with symmetric zero padding, with any odd remainder landing on the bottom and right.
- Model outputs:
  - Semantic output 0: center heatmap sigmoid, shape `56x56`
  - Semantic output 1: tip heatmap sigmoid, shape `56x56`
  - Semantic output 2: confidence sigmoid scalar
  - Raw TFLite tensor order is tip heatmap, center heatmap, confidence, so firmware must reorder the first two tensors before decode.
- Heatmap decode:
  - Use softargmax for the default replay path.
  - Convert heatmap indices to 224-space pixels with `pixel = index * 224 / 55`.
  - Derive needle angle from center and tip with the standard gauge geometry helper.
- Calibration:
  - Candidate name: `D_robust_linear`
  - Candidate kind: `robust_linear`
  - Slope: `0.311885976726`
  - Intercept: `-33.141012138577`
  - Cold angle degrees: `135.000000000000`
- Guardrails:
  - `center_peak_min = 0.40`
  - `tip_peak_min = 0.35`
  - `confidence_min = 0.40`
  - `max_heatmap_entropy = 1.00`
  - `max_heatmap_spread_px = 30.00`
  - `center_tip_distance_ratio_min = 0.35`
  - `center_tip_distance_ratio_max = 1.40`
  - `edge_margin_px = 4.00`
  - `temperature_physical_margin_c = 2.00`
  - `minimum_celsius = -30.00`
  - `maximum_celsius = 50.00`
- Rejection and clamp behavior:
  - Reject if either point leaves the crop, approaches the edge too closely, or produces diffuse/low-confidence heatmaps.
  - Reject if the calibrated temperature exceeds the physical range by more than the allowed margin.
  - Clamp only when the decoded temperature is slightly outside the physical range, and always mark the reading as `clamped`.

## Known Bad Alternatives

- `board_like_rgb_nearest` loses acceptance on test and should not be used as a firmware substitute.
- `board_like_luma_nearest_if_supported` also loses acceptance and should not be used unless the model is retrained for luma input.
- Do not switch firmware to nearest-neighbor or luma preprocessing unless the model is retrained and replay-validated for that exact path.
