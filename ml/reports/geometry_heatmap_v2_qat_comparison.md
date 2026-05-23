# Geometry Heatmap v2 QAT Comparison

## Summary

- Corrected decoder: `softargmax` with `window_size=3`
- QAT training strategy: quantization-noise fine-tuning with fake int8 output round-trips
- Standard TFMOT QAT: not feasible in this environment because `tensorflow_model_optimization` is not installed
- QAT Keras checkpoint selected on validation: frozen stage
- QAT INT8 validation replay: failed the guardrail and drift gates, so test replay was not justified

## Base References

| model | split | accepted MAE | acceptance rate | worst accepted error | temp drift mean | tip drift mean | guardrail disagreements |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base Keras | test | 3.5554 C | 0.8136 | 17.4588 C | - | - | - |
| current full INT8 | test | 3.7062 C | 0.7119 | 14.5879 C | 1.7350 C | 14.0198 px | 12 |
| best dynamic range | test | 3.4466 C | 0.7288 | 13.9186 C | 1.3152 C | 5.4413 px | 15 |

## QAT Results

| model | split | accepted MAE | acceptance rate | worst accepted error | >20 C fails | temp delta mean | temp delta median | temp delta p90 | center delta mean | tip delta mean | guardrail disagreements |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QAT Keras | val | 2.4923 C | 0.7660 | 11.9105 C | 0 | 1.3153 C | 0.8222 C | 2.7485 C | 1.6657 px | 10.7592 px | 7 |
| QAT TFLite FP32 | val | 2.4923 C | 0.7660 | 11.9107 C | 0 | 1.3153 C | 0.8221 C | 2.7485 C | 1.6657 px | 10.7592 px | 7 |
| QAT TFLite INT8 | val | 4.1516 C | 0.5957 | 29.2107 C | 1 | 2.3986 C | 1.6549 C | 5.2812 C | 2.8440 px | 22.6609 px | 9 |

## Interpretation

- The QAT fine-tune preserved the Keras model well.
- The QAT float32 TFLite export also tracked Keras closely.
- The QAT INT8 export did not preserve the replay behavior:
  - acceptance rate fell below the `0.65` gate
  - worst accepted error exceeded `20 C`
  - the INT8 tip drift became much worse than the corrected current INT8 baseline
- The problem is now export-side quantization behavior, not the corrected decoder or the board contract.

## Conclusion

- QAT INT8 is not a Cube.AI packaging candidate yet.
- The most likely next step is to train `geometry_heatmap_v3` with quantization-noise loss baked in from the start.
