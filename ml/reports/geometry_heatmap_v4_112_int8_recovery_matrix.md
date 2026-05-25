# Geometry Heatmap v4 112 INT8 Recovery Matrix

- Generated: 2026-05-24 12:08:30
- Candidates: 8
- Quick mode: False

## Candidate Grid

| # | Name | peak_target | center_weight | tip_weight | conf_floor_weight | warmup_epochs |
|---|------|-------------|---------------|------------|-------------------|---------------|
| 1 | 01_conservative | 0.3 | 0.1 | 0.2 | 0.05 | 5 |
| 2 | 02_lower_peak_target | 0.25 | 0.1 | 0.2 | 0.05 | 5 |
| 3 | 03_lighter_shaping | 0.3 | 0.05 | 0.15 | 0.05 | 5 |
| 4 | 04_short_warmup | 0.3 | 0.1 | 0.2 | 0.05 | 3 |
| 5 | 05_light_all | 0.25 | 0.05 | 0.15 | 0.03 | 5 |
| 6 | 06_aggressive | 0.25 | 0.05 | 0.15 | 0.03 | 3 |
| 7 | 07_high_peak_low_floor | 0.3 | 0.05 | 0.2 | 0.03 | 5 |
| 8 | 08_tip_focus | 0.25 | 0.05 | 0.2 | 0.05 | 3 |

## Gates (must all pass)

- **accepted_mae_c**: <= 4.5
- **acceptance_rate**: >= 0.65
- **worst_accepted_error_c**: < 20.0
- **accepted_gt20_failures**: <= 0
- **temperature_delta_mean**: <= 1.0

## Results by Candidate

| Rank | Name | Status | Accepted MAE | Acceptance | Worst Error | >20C Fail | Temp Drift Mean | Center Drift | Tip Drift | Gates |
|------|------|--------|--------------|------------|-------------|-----------|--------------|--------------|-----------|-------|
| 1 | 08_tip_focus | fail | 3.7753 | 0.7234 | 14.65 | 0 | 1.8405 | 0.6617 | 12.6247 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 2 | 04_short_warmup | fail | 3.8077 | 0.7234 | 14.59 | 0 | 1.8561 | 0.6603 | 12.5244 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 3 | 06_aggressive | fail | 3.8198 | 0.7234 | 14.67 | 0 | 1.8782 | 0.6591 | 12.5584 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 4 | 07_high_peak_low_floor | fail | 3.9164 | 0.7021 | 15.33 | 0 | 1.9138 | 0.6711 | 11.9331 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 5 | 01_conservative | fail | 3.9191 | 0.7021 | 15.39 | 0 | 1.9050 | 0.6690 | 11.9094 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 6 | 05_light_all | fail | 3.9207 | 0.7021 | 15.43 | 0 | 1.8986 | 0.6698 | 11.9036 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 7 | 03_lighter_shaping | fail | 3.9266 | 0.7021 | 15.31 | 0 | 1.9011 | 0.6635 | 11.8742 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 8 | 02_lower_peak_target | fail | 3.9418 | 0.7021 | 15.48 | 0 | 1.9070 | 0.6647 | 11.9198 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |

## Detailed Metrics

### 08_tip_focus (08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3)
- Status: fail
- acceptance_rate: 0.723404
- accepted_count: 34.000000
- accepted_gt20_failures: 0.000000
- accepted_mae_c: 3.775255
- angle_mae_degrees: 40.948314
- center_delta_mean: 0.661733
- center_delta_median: 0.663186
- center_heatmap_peak_mean: 0.996094
- center_heatmap_spread_mean: 44.512238
- center_mae_px_224: 8.400742
- confidence_mean: 0.959525
- count: 47.000000
- guardrail_disagreement_count: 13.000000
- guardrail_disagreements: 10.000000
- percentage_under_10c: 68.085106
- percentage_under_2c: 23.404255
- percentage_under_5c: 59.574468
- temperature_delta_mean: 1.840473
- temperature_delta_median: 1.374894
- temperature_delta_p90: 3.610118
- tip_delta_mean: 12.624747
- tip_delta_median: 10.259005
- tip_heatmap_peak_mean: 0.860289
- tip_heatmap_spread_mean: 47.722407
- tip_mae_px_224: 28.697832
- worst_accepted_error_c: 14.648838
- center_delta_mean: 0.661733
- center_delta_median: 0.663186
- guardrail_disagreement_count: 13.000000
- guardrail_disagreements: 10.000000
- temperature_delta_mean: 1.840473
- temperature_delta_median: 1.374894
- temperature_delta_p90: 3.610118
- tip_delta_mean: 12.624747
- tip_delta_median: 10.259005
- Gate results: accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL

### 04_short_warmup (04_short_warmup__pt0.30_c0.10_t0.20_cf0.05_wu3)
- Status: fail
- acceptance_rate: 0.723404
- accepted_count: 34.000000
- accepted_gt20_failures: 0.000000
- accepted_mae_c: 3.807712
- angle_mae_degrees: 41.053159
- center_delta_mean: 0.660279
- center_delta_median: 0.656477
- center_heatmap_peak_mean: 0.996094
- center_heatmap_spread_mean: 44.504431
- center_mae_px_224: 8.407471
- confidence_mean: 0.959109
- count: 47.000000
- guardrail_disagreement_count: 13.000000
- guardrail_disagreements: 10.000000
- percentage_under_10c: 68.085106
- percentage_under_2c: 23.404255
- percentage_under_5c: 59.574468
- temperature_delta_mean: 1.856068
- temperature_delta_median: 1.423188
- temperature_delta_p90: 3.596059
- tip_delta_mean: 12.524418
- tip_delta_median: 9.913643
- tip_heatmap_peak_mean: 0.860622
- tip_heatmap_spread_mean: 47.936336
- tip_mae_px_224: 28.892814
- worst_accepted_error_c: 14.591628
- center_delta_mean: 0.660279
- center_delta_median: 0.656477
- guardrail_disagreement_count: 13.000000
- guardrail_disagreements: 10.000000
- temperature_delta_mean: 1.856068
- temperature_delta_median: 1.423188
- temperature_delta_p90: 3.596059
- tip_delta_mean: 12.524418
- tip_delta_median: 9.913643
- Gate results: accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL

### 06_aggressive (06_aggressive__pt0.25_c0.05_t0.15_cf0.03_wu3)
- Status: fail
- acceptance_rate: 0.723404
- accepted_count: 34.000000
- accepted_gt20_failures: 0.000000
- accepted_mae_c: 3.819813
- angle_mae_degrees: 41.139768
- center_delta_mean: 0.659052
- center_delta_median: 0.669319
- center_heatmap_peak_mean: 0.996094
- center_heatmap_spread_mean: 44.502274
- center_mae_px_224: 8.406347
- confidence_mean: 0.959275
- count: 47.000000
- guardrail_disagreement_count: 13.000000
- guardrail_disagreements: 10.000000
- percentage_under_10c: 68.085106
- percentage_under_2c: 23.404255
- percentage_under_5c: 57.446809
- temperature_delta_mean: 1.878218
- temperature_delta_median: 1.458893
- temperature_delta_p90: 3.557947
- tip_delta_mean: 12.558416
- tip_delta_median: 9.834512
- tip_heatmap_peak_mean: 0.860622
- tip_heatmap_spread_mean: 47.943966
- tip_mae_px_224: 28.983595
- worst_accepted_error_c: 14.674972
- center_delta_mean: 0.659052
- center_delta_median: 0.669319
- guardrail_disagreement_count: 13.000000
- guardrail_disagreements: 10.000000
- temperature_delta_mean: 1.878218
- temperature_delta_median: 1.458893
- temperature_delta_p90: 3.557947
- tip_delta_mean: 12.558416
- tip_delta_median: 9.834512
- Gate results: accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL

### 07_high_peak_low_floor (07_high_peak_low_floor__pt0.30_c0.05_t0.20_cf0.03_wu5)
- Status: fail
- acceptance_rate: 0.702128
- accepted_count: 33.000000
- accepted_gt20_failures: 0.000000
- accepted_mae_c: 3.916383
- angle_mae_degrees: 34.203491
- center_delta_mean: 0.671054
- center_delta_median: 0.652178
- center_heatmap_peak_mean: 0.996094
- center_heatmap_spread_mean: 44.425683
- center_mae_px_224: 8.420378
- confidence_mean: 0.956283
- count: 47.000000
- guardrail_disagreement_count: 14.000000
- guardrail_disagreements: 8.000000
- percentage_under_10c: 65.957447
- percentage_under_2c: 23.404255
- percentage_under_5c: 55.319149
- temperature_delta_mean: 1.913773
- temperature_delta_median: 1.569081
- temperature_delta_p90: 3.648638
- tip_delta_mean: 11.933051
- tip_delta_median: 9.880031
- tip_heatmap_peak_mean: 0.866273
- tip_heatmap_spread_mean: 49.094318
- tip_mae_px_224: 30.778718
- worst_accepted_error_c: 15.329765
- center_delta_mean: 0.671054
- center_delta_median: 0.652178
- guardrail_disagreement_count: 14.000000
- guardrail_disagreements: 8.000000
- temperature_delta_mean: 1.913773
- temperature_delta_median: 1.569081
- temperature_delta_p90: 3.648638
- tip_delta_mean: 11.933051
- tip_delta_median: 9.880031
- Gate results: accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL

### 01_conservative (01_conservative__pt0.30_c0.10_t0.20_cf0.05_wu5)
- Status: fail
- acceptance_rate: 0.702128
- accepted_count: 33.000000
- accepted_gt20_failures: 0.000000
- accepted_mae_c: 3.919112
- angle_mae_degrees: 34.048217
- center_delta_mean: 0.669021
- center_delta_median: 0.652672
- center_heatmap_peak_mean: 0.996094
- center_heatmap_spread_mean: 44.423412
- center_mae_px_224: 8.424194
- confidence_mean: 0.955951
- count: 47.000000
- guardrail_disagreement_count: 14.000000
- guardrail_disagreements: 8.000000
- percentage_under_10c: 65.957447
- percentage_under_2c: 23.404255
- percentage_under_5c: 55.319149
- temperature_delta_mean: 1.904991
- temperature_delta_median: 1.582023
- temperature_delta_p90: 3.716168
- tip_delta_mean: 11.909428
- tip_delta_median: 9.862854
- tip_heatmap_peak_mean: 0.866523
- tip_heatmap_spread_mean: 49.043765
- tip_mae_px_224: 30.701173
- worst_accepted_error_c: 15.391024
- center_delta_mean: 0.669021
- center_delta_median: 0.652672
- guardrail_disagreement_count: 14.000000
- guardrail_disagreements: 8.000000
- temperature_delta_mean: 1.904991
- temperature_delta_median: 1.582023
- temperature_delta_p90: 3.716168
- tip_delta_mean: 11.909428
- tip_delta_median: 9.862854
- Gate results: accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL

### 05_light_all (05_light_all__pt0.25_c0.05_t0.15_cf0.03_wu5)
- Status: fail
- acceptance_rate: 0.702128
- accepted_count: 33.000000
- accepted_gt20_failures: 0.000000
- accepted_mae_c: 3.920719
- angle_mae_degrees: 34.135781
- center_delta_mean: 0.669752
- center_delta_median: 0.660922
- center_heatmap_peak_mean: 0.996094
- center_heatmap_spread_mean: 44.420901
- center_mae_px_224: 8.424522
- confidence_mean: 0.955951
- count: 47.000000
- guardrail_disagreement_count: 14.000000
- guardrail_disagreements: 8.000000
- percentage_under_10c: 65.957447
- percentage_under_2c: 23.404255
- percentage_under_5c: 55.319149
- temperature_delta_mean: 1.898640
- temperature_delta_median: 1.580306
- temperature_delta_p90: 3.621321
- tip_delta_mean: 11.903601
- tip_delta_median: 9.820466
- tip_heatmap_peak_mean: 0.866107
- tip_heatmap_spread_mean: 49.039066
- tip_mae_px_224: 30.721885
- worst_accepted_error_c: 15.428521
- center_delta_mean: 0.669752
- center_delta_median: 0.660922
- guardrail_disagreement_count: 14.000000
- guardrail_disagreements: 8.000000
- temperature_delta_mean: 1.898640
- temperature_delta_median: 1.580306
- temperature_delta_p90: 3.621321
- tip_delta_mean: 11.903601
- tip_delta_median: 9.820466
- Gate results: accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL

### 03_lighter_shaping (03_lighter_shaping__pt0.30_c0.05_t0.15_cf0.05_wu5)
- Status: fail
- acceptance_rate: 0.702128
- accepted_count: 33.000000
- accepted_gt20_failures: 0.000000
- accepted_mae_c: 3.926611
- angle_mae_degrees: 34.169803
- center_delta_mean: 0.663533
- center_delta_median: 0.664999
- center_heatmap_peak_mean: 0.996094
- center_heatmap_spread_mean: 44.428910
- center_mae_px_224: 8.416473
- confidence_mean: 0.956366
- count: 47.000000
- guardrail_disagreement_count: 14.000000
- guardrail_disagreements: 8.000000
- percentage_under_10c: 65.957447
- percentage_under_2c: 21.276596
- percentage_under_5c: 55.319149
- temperature_delta_mean: 1.901084
- temperature_delta_median: 1.550781
- temperature_delta_p90: 3.657342
- tip_delta_mean: 11.874201
- tip_delta_median: 10.060524
- tip_heatmap_peak_mean: 0.864860
- tip_heatmap_spread_mean: 49.047662
- tip_mae_px_224: 30.706870
- worst_accepted_error_c: 15.308957
- center_delta_mean: 0.663533
- center_delta_median: 0.664999
- guardrail_disagreement_count: 14.000000
- guardrail_disagreements: 8.000000
- temperature_delta_mean: 1.901084
- temperature_delta_median: 1.550781
- temperature_delta_p90: 3.657342
- tip_delta_mean: 11.874201
- tip_delta_median: 10.060524
- Gate results: accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL

### 02_lower_peak_target (02_lower_peak_target__pt0.25_c0.10_t0.20_cf0.05_wu5)
- Status: fail
- acceptance_rate: 0.702128
- accepted_count: 33.000000
- accepted_gt20_failures: 0.000000
- accepted_mae_c: 3.941786
- angle_mae_degrees: 34.182745
- center_delta_mean: 0.664681
- center_delta_median: 0.660080
- center_heatmap_peak_mean: 0.996094
- center_heatmap_spread_mean: 44.428022
- center_mae_px_224: 8.424757
- confidence_mean: 0.956283
- count: 47.000000
- guardrail_disagreement_count: 14.000000
- guardrail_disagreements: 8.000000
- percentage_under_10c: 65.957447
- percentage_under_2c: 23.404255
- percentage_under_5c: 55.319149
- temperature_delta_mean: 1.907034
- temperature_delta_median: 1.582809
- temperature_delta_p90: 3.663737
- tip_delta_mean: 11.919848
- tip_delta_median: 9.853680
- tip_heatmap_peak_mean: 0.866107
- tip_heatmap_spread_mean: 49.027610
- tip_mae_px_224: 30.748697
- worst_accepted_error_c: 15.476594
- center_delta_mean: 0.664681
- center_delta_median: 0.660080
- guardrail_disagreement_count: 14.000000
- guardrail_disagreements: 8.000000
- temperature_delta_mean: 1.907034
- temperature_delta_median: 1.582809
- temperature_delta_p90: 3.663737
- tip_delta_mean: 11.919848
- tip_delta_median: 9.853680
- Gate results: accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL
