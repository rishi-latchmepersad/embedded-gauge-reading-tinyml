# INT8 Decoder Drift Audit Decision

- Generated: 2026-05-24 12:27:40
- Candidates evaluated: 3
- Total decoder+candidate pairs: 18
- Passing pairs: 0

**Decision B**: Decoder-only recovery failed.

No decoder+candidate pair passes all gates.

### Dominant Failing Gates
- **temperature_delta_mean**: failed by 18/18 pairs
- **accepted_mae_c**: failed by 15/18 pairs
- **acceptance_rate**: failed by 15/18 pairs
- **worst_accepted_error_c**: failed by 15/18 pairs
- **accepted_gt20_failures**: failed by 15/18 pairs

### Top 3 Near-Miss Pairs

#### 08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3 / softargmax_w3
- Gates: accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL
- INT8 accepted MAE: 3.7753 C
- INT8 acceptance rate: 0.7234
- Temp drift mean: 1.8405 C
- Tip drift mean: 12.6247 px

#### 04_short_warmup__pt0.30_c0.10_t0.20_cf0.05_wu3 / softargmax_w3
- Gates: accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL
- INT8 accepted MAE: 3.8077 C
- INT8 acceptance rate: 0.7234
- Temp drift mean: 1.8561 C
- Tip drift mean: 12.5244 px

#### 06_aggressive__pt0.25_c0.05_t0.15_cf0.03_wu3 / softargmax_w3
- Gates: accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL
- INT8 accepted MAE: 3.8198 C
- INT8 acceptance rate: 0.7234
- Temp drift mean: 1.8782 C
- Tip drift mean: 12.5584 px

### Next Step
Decoder-only recovery failed. Next step should be a small architecture change: add an INT8-friendly coordinate/offset auxiliary head or replace pure heatmap decoding with a quantization-robust point head.