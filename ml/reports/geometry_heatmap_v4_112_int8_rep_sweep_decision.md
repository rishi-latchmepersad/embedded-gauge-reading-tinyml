# Sweep Decision: No Strategy Passes All Gates

## Failing Gates
- Acceptance: A baseline_identity (?)
- Acceptance: C identity_mild_medium (?)
- Acceptance: E spread_boundary (?)
- Temp Drift: Ref Phase 10E (?)
- Temp Drift: A baseline_identity (?)
- Temp Drift: B identity_mild (?)
- Temp Drift: C identity_mild_medium (?)
- Temp Drift: D stratified (?)
- Temp Drift: E spread_boundary (?)
- Temp Drift: F combined (?)

## Recommendations
- Consider widening the representative dataset further (more jitter, more diversity).
- Consider QAT (quantization-aware training) to bake quantization robustness into weights.
- Or deploy the FP32 model instead (7.6 MB, exact Keras match).