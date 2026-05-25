# Geometry Heatmap v4 112 INT8 Decoder Drift Audit

- Generated: 2026-05-24 12:27:40
- Candidates: 08_tip_focus, 04_short_warmup, 06_aggressive
- Decoders: softargmax_w3, argmax, local_window_softargmax_w3, local_window_softargmax_w5, peak_weighted_centroid_w3, peak_weighted_centroid_w5
- Validation samples: 47
- Calibration: D_robust_linear
- Guardrails: max_heatmap_spread_px=55.0 (v4 thresholds)

## Ranked Results

Pairs sorted by: gates pass first, then acceptance desc, MAE asc, temp drift asc.

| Rank | Candidate | Decoder | All Gates | Acc MAE(C) | Acc Rate | Worst Err(C) | >20C Fail | Temp Drift Mean(C) | Tip Drift Mean(px) | Center Drift(px) | Under 2C | Under 5C | Under 10C | Rejections |
|------|-----------|---------|-----------|------------|----------|--------------|-----------|--------------------|--------------------|------------------|----------|----------|-----------|------------|
| 1 | 08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3 | softargmax_w3 | FAIL | 3.78 | 0.7234 | 14.65 | 0 | 1.8405 | 12.6247 | 0.6617 | 23.4% | 59.6% | 68.1% | tip_heatmap_too_spread_out:12;center_... |
| 2 | 04_short_warmup__pt0.30_c0.10_t0.20_cf0.05_wu3 | softargmax_w3 | FAIL | 3.81 | 0.7234 | 14.59 | 0 | 1.8561 | 12.5244 | 0.6603 | 23.4% | 59.6% | 68.1% | tip_heatmap_too_spread_out:12;center_... |
| 3 | 06_aggressive__pt0.25_c0.05_t0.15_cf0.03_wu3 | softargmax_w3 | FAIL | 3.82 | 0.7234 | 14.67 | 0 | 1.8782 | 12.5584 | 0.6591 | 23.4% | 57.4% | 68.1% | tip_heatmap_too_spread_out:12;center_... |
| 4 | 06_aggressive__pt0.25_c0.05_t0.15_cf0.03_wu3 | argmax | FAIL | 20.96 | 0.2766 | 33.47 | 7 | 15.5011 | 54.7090 | 87.7776 | 0.0% | 0.0% | 2.1% | predicted_point_near_edge:28;center_t... |
| 5 | 06_aggressive__pt0.25_c0.05_t0.15_cf0.03_wu3 | local_window_softargmax_w3 | FAIL | 20.96 | 0.2766 | 33.47 | 7 | 15.5034 | 54.1593 | 87.7748 | 0.0% | 0.0% | 2.1% | predicted_point_near_edge:28;center_t... |
| 6 | 06_aggressive__pt0.25_c0.05_t0.15_cf0.03_wu3 | peak_weighted_centroid_w3 | FAIL | 20.96 | 0.2766 | 33.47 | 7 | 15.5034 | 54.1593 | 87.7748 | 0.0% | 0.0% | 2.1% | predicted_point_near_edge:28;center_t... |
| 7 | 06_aggressive__pt0.25_c0.05_t0.15_cf0.03_wu3 | local_window_softargmax_w5 | FAIL | 20.96 | 0.2766 | 33.47 | 7 | 15.5073 | 53.7753 | 87.7615 | 0.0% | 0.0% | 2.1% | predicted_point_near_edge:28;center_t... |
| 8 | 06_aggressive__pt0.25_c0.05_t0.15_cf0.03_wu3 | peak_weighted_centroid_w5 | FAIL | 20.96 | 0.2766 | 33.47 | 7 | 15.5073 | 53.7753 | 87.7615 | 0.0% | 0.0% | 2.1% | predicted_point_near_edge:28;center_t... |
| 9 | 08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3 | argmax | FAIL | 20.65 | 0.2553 | 30.52 | 7 | 16.6298 | 46.3271 | 88.2658 | 0.0% | 2.1% | 2.1% | predicted_point_near_edge:28;center_t... |
| 10 | 08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3 | peak_weighted_centroid_w3 | FAIL | 20.65 | 0.2553 | 30.51 | 7 | 16.6298 | 45.8773 | 88.2628 | 0.0% | 2.1% | 2.1% | predicted_point_near_edge:28;center_t... |
| 11 | 08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3 | local_window_softargmax_w3 | FAIL | 20.65 | 0.2553 | 30.51 | 7 | 16.6298 | 45.8773 | 88.2628 | 0.0% | 2.1% | 2.1% | predicted_point_near_edge:28;center_t... |
| 12 | 08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3 | local_window_softargmax_w5 | FAIL | 20.65 | 0.2553 | 30.49 | 7 | 16.6327 | 45.5693 | 88.2499 | 0.0% | 2.1% | 2.1% | predicted_point_near_edge:28;center_t... |
| 13 | 08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3 | peak_weighted_centroid_w5 | FAIL | 20.65 | 0.2553 | 30.49 | 7 | 16.6327 | 45.5693 | 88.2499 | 0.0% | 2.1% | 2.1% | predicted_point_near_edge:28;center_t... |
| 14 | 04_short_warmup__pt0.30_c0.10_t0.20_cf0.05_wu3 | argmax | FAIL | 21.38 | 0.2553 | 34.11 | 5 | 16.3172 | 52.8960 | 88.2839 | 0.0% | 2.1% | 2.1% | predicted_point_near_edge:29;center_t... |
| 15 | 04_short_warmup__pt0.30_c0.10_t0.20_cf0.05_wu3 | peak_weighted_centroid_w3 | FAIL | 21.38 | 0.2553 | 34.10 | 5 | 16.3204 | 52.3587 | 88.2809 | 0.0% | 2.1% | 2.1% | predicted_point_near_edge:29;center_t... |
| 16 | 04_short_warmup__pt0.30_c0.10_t0.20_cf0.05_wu3 | local_window_softargmax_w3 | FAIL | 21.38 | 0.2553 | 34.10 | 5 | 16.3204 | 52.3587 | 88.2809 | 0.0% | 2.1% | 2.1% | predicted_point_near_edge:29;center_t... |
| 17 | 04_short_warmup__pt0.30_c0.10_t0.20_cf0.05_wu3 | local_window_softargmax_w5 | FAIL | 21.39 | 0.2553 | 34.10 | 5 | 16.3238 | 51.9849 | 88.2674 | 0.0% | 2.1% | 2.1% | predicted_point_near_edge:29;center_t... |
| 18 | 04_short_warmup__pt0.30_c0.10_t0.20_cf0.05_wu3 | peak_weighted_centroid_w5 | FAIL | 21.39 | 0.2553 | 34.10 | 5 | 16.3238 | 51.9849 | 88.2674 | 0.0% | 2.1% | 2.1% | predicted_point_near_edge:29;center_t... |
