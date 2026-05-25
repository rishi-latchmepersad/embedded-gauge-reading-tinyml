# INT8 Recovery Decision

- Generated: 2026-05-24 12:08:30
- Candidates: 8
- Passing: 0
- Failing: 8

## No Candidate Passes All Gates

### Dominant Failing Gates

- **temperature_delta_mean**: failed by 8/8 candidates

### Top 2 Near-Miss Candidates

#### 1. 08_tip_focus (08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3)
- Gates: accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL
- Accepted MAE: 3.7753 C
- Acceptance rate: 0.7234
- Temperature drift mean: 1.8405 C

#### 2. 04_short_warmup (04_short_warmup__pt0.30_c0.10_t0.20_cf0.05_wu3)
- Gates: accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL
- Accepted MAE: 3.8077 C
- Acceptance rate: 0.7234
- Temperature drift mean: 1.8561 C

## Decision

**Decision B**: No champion found. All candidates fail one or more gates.
Dominant failure pattern: temperature_delta_mean (8/8)

Recommended next step: Investigate dominant failing gates before escalating to architecture tweaks.
Consider adjusting peak_target, loss weights, or warmup schedule based on failure pattern.