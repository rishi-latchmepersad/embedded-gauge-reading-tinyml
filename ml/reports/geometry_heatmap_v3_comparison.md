# Geometry Heatmap v3 Comparison

## Baselines

### Base Keras v2

- test accepted MAE: `3.5554 C`
- acceptance: `0.8136`
- worst accepted error: `17.4588 C`

### Current v2 Full INT8

- test accepted MAE: `3.7062 C`
- acceptance: `0.7119`
- worst accepted error: `14.5879 C`
- temperature drift mean: `1.7350 C`
- tip drift mean: `14.0198 px`
- guardrail disagreements: `12`

### Best v2 Dynamic Range

- test accepted MAE: `3.4466 C`
- acceptance: `0.7288`
- worst accepted error: `13.9186 C`
- temperature drift mean: `1.3152 C`
- tip drift mean: `5.4413 px`
- guardrail disagreements: `15`

### v2 QAT INT8

- validation accepted MAE: `4.1516 C`
- validation acceptance: `0.5957`
- validation worst accepted error: `29.2107 C`
- accepted >20 C failures: `1`

## v3 Validation Result

### Keras v3

- accepted MAE: `3.6002 C`
- acceptance: `0.7021`
- worst accepted error: `13.5297 C`

### v3 INT8

- accepted MAE: `3.3048 C`
- acceptance: `0.5957`
- worst accepted error: `11.8713 C`
- accepted >20 C failures: `0`
- temperature drift mean: `1.9923 C`
- tip drift mean: `14.7833 px`
- guardrail disagreements: `7`

## Interpretation

- v3 Keras is usable, but the exported INT8 model did not clear the validation gate.
- The INT8 model reduced accepted MAE and worst accepted error relative to some baselines, but it missed the required acceptance rate and temperature-drift threshold.
- Tip drift remains materially high relative to the better v2 dynamic-range export.
