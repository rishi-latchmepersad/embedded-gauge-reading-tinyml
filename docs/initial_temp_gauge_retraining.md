# LittleGood temperature-gauge retraining

The v2 candidates were trained from scratch on the original labelled data plus
the `initial_temp_gauge` samples.  Each LittleGood sample appears once in the
merged training set; no oversampling or repeated-gauge weighting was used so
the new gauge family does not dominate the general training distribution.

The archive-level LittleGood split contains 451 training, 72 validation, and
97 held-out test samples.  The ellipse merge contains 7,779 train, 999
validation, and 1,022 test images.  The center/tip merge contains 7,760 train
and 986 validation samples; 97 held-out LittleGood samples are used for the
reported domain test.

## Artifacts

- `ml/artifacts/gauge_ellipse_littlegood_v2/`
- `ml/artifacts/gauge_center_tip_littlegood_v2/`
- `ml/artifacts/gauge_retrained_littlegood_v2_evaluation.json`

Both jobs used the RTX A5500 with TensorFlow's logical GPU memory limit set to
15,000 MB.  Both use TFLite-compatible QAT and full-int8 TFLite export.

## Held-out results

The ellipse model's Keras result is 0.0358 normalized MAE on LittleGood, but
its int8 TFLite result is 0.0596 MAE.  Keras-vs-TFLite mean absolute output
difference is 0.0452, so this candidate must not be packaged for the board
until the conversion mismatch is resolved.

The center/tip model's int8 TFLite result is 2.26/1.93 px coordinate MAE for
center x/y and 5.19/6.38 px for tip x/y in the 160x160 crop.  Mean Euclidean
errors are 3.24 px for center and 8.73 px for tip; 97.9% of centers and 59.8%
of tips are within 8 px.  Keras-vs-TFLite mean output difference is 0.00142.

## Memory and operators

The conservative two-buffer activation bound is 819,200 bytes for both
flatbuffers, below the 1 MiB ellipse and 1.5 MiB center/tip SRAM budgets.
The ellipse graph uses `CONV_2D`, `FULLY_CONNECTED`, `LOGISTIC`, and `MEAN`.
The center/tip graph uses `CONV_2D`, `MAX_POOL_2D`, `RESIZE_NEAREST_NEIGHBOR`,
`CONCATENATION`, `LOGISTIC`, and `QUANTIZE`.  The latter two spatial graph
operations need confirmation in the STM32N6 Edge AI report; the TFLite result
alone is not sufficient to claim every center/tip operation will execute on
the NPU.
