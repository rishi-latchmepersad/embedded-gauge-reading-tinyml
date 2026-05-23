# Geometry Heatmap v2 Quantization Drift Autopsy

## Setup

- Source replay CSV: `/mnt/d/Projects/embedded-gauge-reading-tinyml/ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/quantization_drift_analysis.csv`
- Preprocessing mode: `python_training_rgb_bilinear`

## Root Cause Summary

- Tip heatmap flattening cases: `137`
- Center heatmap spread growth cases: `12`
- Tip heatmap spread growth cases: `217`
- Softargmax-sensitive cases: `52`
- Output order issue likely? `False`
- Input quantization primary? `False`
- Output quantization primary? `True`
- Representative dataset weakness likely? `True`
- The FP32 replay matched Keras in the earlier phase, so raw tensor order / dequantization is not the main culprit.
- The remaining drift is concentrated in the tip branch, where the INT8 heatmap tends to flatten and spread out more than the Keras heatmap.

## Split Summary

### train

| metric | value |
| --- | ---: |
| count | 227 |
| Keras accepted MAE (C) | 2.9753 |
| Keras acceptance rate | 0.7974 |
| INT8 accepted MAE (C) | 3.0627 |
| INT8 acceptance rate | 0.7313 |
| INT8 worst accepted error (C) | 19.6672 |
| Keras-vs-INT8 temp delta mean (C) | nan |
| Keras-vs-INT8 temp delta median (C) | nan |
| Keras-vs-INT8 temp delta p90 (C) | nan |
| Keras-vs-INT8 tip delta mean (px) | 11.6787 |
| Keras-vs-INT8 tip delta median (px) | 8.5989 |
| Keras-vs-INT8 tip delta p90 (px) | 26.4909 |
| guardrail disagreements | 37 |
| accepted >20 C INT8 failures | 0 |

### val

| metric | value |
| --- | ---: |
| count | 47 |
| Keras accepted MAE (C) | 3.2602 |
| Keras acceptance rate | 0.6596 |
| INT8 accepted MAE (C) | 3.3214 |
| INT8 acceptance rate | 0.6383 |
| INT8 worst accepted error (C) | 11.5008 |
| Keras-vs-INT8 temp delta mean (C) | nan |
| Keras-vs-INT8 temp delta median (C) | nan |
| Keras-vs-INT8 temp delta p90 (C) | nan |
| Keras-vs-INT8 tip delta mean (px) | 14.7590 |
| Keras-vs-INT8 tip delta median (px) | 10.9587 |
| Keras-vs-INT8 tip delta p90 (px) | 28.2843 |
| guardrail disagreements | 8 |
| accepted >20 C INT8 failures | 0 |

### test

| metric | value |
| --- | ---: |
| count | 59 |
| Keras accepted MAE (C) | 3.5764 |
| Keras acceptance rate | 0.8136 |
| INT8 accepted MAE (C) | 3.7062 |
| INT8 acceptance rate | 0.7119 |
| INT8 worst accepted error (C) | 14.5879 |
| Keras-vs-INT8 temp delta mean (C) | nan |
| Keras-vs-INT8 temp delta median (C) | nan |
| Keras-vs-INT8 temp delta p90 (C) | nan |
| Keras-vs-INT8 tip delta mean (px) | 14.0826 |
| Keras-vs-INT8 tip delta median (px) | 9.9766 |
| Keras-vs-INT8 tip delta p90 (px) | 32.8538 |
| guardrail disagreements | 12 |
| accepted >20 C INT8 failures | 0 |

## Top 30 Tip Deltas

| split | image_path | tip_delta_px | center_delta_px | temperature_delta_c | keras_tip_heatmap_peak_value | int8_tip_heatmap_peak_value | keras_tip_heatmap_spread_px | int8_tip_heatmap_spread_px | keras_guardrail_status | int8_guardrail_status | disagreement_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | ml/data/raw/PXL_20260125_114555470.jpg | 55.7437 | 1.9367 | nan | 0.3291 | 0.2656 | 26.1842 | 33.3440 | rejected | rejected | both rejected |
| test | ml/data/raw/PXL_20260125_115125409.jpg | 55.6026 | 2.0376 | nan | 0.3794 | 0.3281 | 24.9450 | 31.2925 | accepted | rejected | keras accepted, int8 rejected |
| test | ml/data/raw/PXL_20260125_120228002.jpg | 55.2180 | 2.4226 | nan | 0.4240 | 0.3086 | 30.8134 | 25.6603 | rejected | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_114556873.jpg | 53.0659 | 3.4089 | nan | 0.2915 | 0.3281 | 31.7814 | 32.5039 | rejected | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_115007049.jpg | 52.7675 | 0.5042 | nan | 0.3138 | 0.3750 | 31.6529 | 32.0192 | rejected | rejected | both rejected |
| val | ml/data/raw/PXL_20260125_115540293.jpg | 52.0213 | 4.4176 | nan | 0.4070 | 0.3750 | 28.9051 | 30.1019 | rejected | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_120026205.jpg | 51.2192 | 5.7087 | nan | 0.4223 | 0.5000 | 26.8547 | 22.4801 | rejected | accepted | keras rejected, int8 accepted |
| val | ml/data/raw/PXL_20260125_114924323.jpg | 49.4906 | 6.2643 | nan | 0.4912 | 0.2656 | 22.4195 | 25.6626 | accepted | rejected | keras accepted, int8 rejected |
| train | ml/data/raw/PXL_20260125_120246531.jpg | 45.4893 | 3.5967 | nan | 0.3490 | 0.3750 | 20.9674 | 26.6929 | rejected | accepted | keras rejected, int8 accepted |
| train | ml/data/raw/PXL_20260125_120052049.jpg | 43.3618 | 8.5701 | 12.3459 | 0.3977 | 0.4258 | 24.0450 | 17.7234 | accepted | accepted | both accepted |
| train | ml/data/raw/PXL_20260125_115743603.jpg | 41.8397 | 5.1513 | nan | 0.7009 | 0.3281 | 19.1928 | 28.9114 | accepted | rejected | keras accepted, int8 rejected |
| train | ml/data/raw/PXL_20260125_115541259.jpg | 41.7040 | 6.5258 | nan | 0.6358 | 0.4727 | 18.7228 | 28.5533 | accepted | rejected | keras accepted, int8 rejected |
| test | ml/data/raw/PXL_20260125_115612552.jpg | 41.0805 | 7.3759 | nan | 0.3789 | 0.5273 | 29.6850 | 16.4528 | rejected | accepted | keras rejected, int8 accepted |
| train | ml/data/raw/PXL_20260125_115417966.jpg | 39.7353 | 7.2452 | nan | 0.4224 | 0.3984 | 29.6568 | 29.6277 | rejected | rejected | both rejected |
| test | ml/data/raw/PXL_20260125_120219516.jpg | 38.8823 | 4.4407 | nan | 0.4060 | 0.3750 | 28.7895 | 25.2261 | rejected | accepted | keras rejected, int8 accepted |
| train | ml/data/raw/PXL_20260125_120227628.jpg | 38.6956 | 3.0798 | nan | 0.3985 | 0.3281 | 32.1881 | 28.2287 | rejected | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_120226207.jpg | 38.1861 | 4.0322 | nan | 0.2964 | 0.3281 | 27.7790 | 27.7049 | rejected | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_114901695.jpg | 37.0949 | 3.7746 | 7.2826 | 0.4003 | 0.3750 | 21.3830 | 28.3050 | accepted | accepted | both accepted |
| train | ml/data/raw/PXL_20260125_114554365.jpg | 36.6934 | 4.1284 | nan | 0.3522 | 0.3516 | 25.1058 | 32.1860 | accepted | rejected | keras accepted, int8 rejected |
| test | ml/data/raw/PXL_20260125_120302481.jpg | 35.8093 | 1.4357 | nan | 0.2947 | 0.2852 | 28.3621 | 27.0551 | rejected | rejected | both rejected |
| test | ml/data/raw/PXL_20260125_120119590.jpg | 35.4691 | 1.7622 | nan | 0.3092 | 0.2852 | 25.3360 | 23.8632 | rejected | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_115150984.jpg | 33.5317 | 4.6896 | 10.7283 | 0.4015 | 0.4727 | 24.0276 | 19.4244 | accepted | accepted | both accepted |
| val | ml/data/raw/PXL_20260125_114954780.jpg | 32.9354 | 4.3468 | nan | 0.2960 | 0.2656 | 24.0281 | 30.2351 | rejected | rejected | both rejected |
| val | ml/data/raw/PXL_20260125_120227059.jpg | 32.5634 | 4.4895 | nan | 0.3041 | 0.3750 | 27.0860 | 23.8485 | rejected | clamped | both rejected |
| test | ml/data/raw/PXL_20260125_115542831.jpg | 32.2000 | 1.5238 | nan | 0.5023 | 0.4727 | 27.1018 | 29.7231 | rejected | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_115742712.jpg | 31.5445 | 7.8223 | 7.1662 | 0.6644 | 0.3516 | 19.4139 | 27.2035 | accepted | accepted | both accepted |
| test | ml/data/raw/PXL_20260125_115957052.jpg | 30.1352 | 5.6252 | nan | 0.4778 | 0.3750 | 27.6787 | 30.8662 | accepted | rejected | keras accepted, int8 rejected |
| train | ml/data/raw/PXL_20260125_114922671.jpg | 29.0095 | 1.3233 | nan | 0.2692 | 0.2852 | 27.3673 | 31.0947 | rejected | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_114549307.jpg | 28.8435 | 5.8224 | nan | 0.4431 | 0.3750 | 25.3033 | 27.0179 | rejected | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_115336256.jpg | 28.7339 | 5.2456 | 5.4676 | 0.8818 | 0.3516 | 14.6998 | 27.2222 | accepted | accepted | both accepted |

## Top 30 Temperature Deltas

| split | image_path | temperature_delta_c | tip_delta_px | center_delta_px | keras_tip_heatmap_peak_value | int8_tip_heatmap_peak_value | keras_tip_heatmap_spread_px | int8_tip_heatmap_spread_px | keras_guardrail_status | int8_guardrail_status | disagreement_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | ml/data/raw/PXL_20260125_114524322.jpg | nan | 12.0035 | 1.9057 | 0.4243 | 0.3516 | 23.8483 | 27.2796 | rejected | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_114538442.jpg | nan | 16.6618 | 6.3705 | 0.4472 | 0.2852 | 22.6426 | 25.4726 | rejected | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_114544203.jpg | nan | 8.6492 | 1.3219 | 0.3737 | 0.3281 | 21.3710 | 24.7239 | clamped | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_114546762.jpg | nan | 27.0306 | 4.0003 | 0.3163 | 0.2656 | 24.4684 | 30.3522 | rejected | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_114548536.jpg | nan | 26.4760 | 1.7418 | 0.5034 | 0.3516 | 27.1172 | 28.4411 | rejected | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_114549307.jpg | nan | 28.8435 | 5.8224 | 0.4431 | 0.3750 | 25.3033 | 27.0179 | rejected | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_114552322.jpg | nan | 12.1918 | 1.5743 | 0.4856 | 0.3086 | 17.7302 | 24.4129 | accepted | rejected | keras accepted, int8 rejected |
| train | ml/data/raw/PXL_20260125_114554365.jpg | nan | 36.6934 | 4.1284 | 0.3522 | 0.3516 | 25.1058 | 32.1860 | accepted | rejected | keras accepted, int8 rejected |
| train | ml/data/raw/PXL_20260125_114556873.jpg | nan | 53.0659 | 3.4089 | 0.2915 | 0.3281 | 31.7814 | 32.5039 | rejected | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_114611913.jpg | nan | 4.4035 | 1.6766 | 0.3349 | 0.3086 | 23.5643 | 24.5385 | rejected | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_114631441.jpg | nan | 10.6746 | 1.2105 | 0.3457 | 0.3516 | 21.0498 | 20.7213 | rejected | clamped | both rejected |
| train | ml/data/raw/PXL_20260125_114637155.jpg | nan | 11.0740 | 1.3218 | 0.3164 | 0.3750 | 21.3732 | 19.5056 | rejected | clamped | both rejected |
| train | ml/data/raw/PXL_20260125_114642671.jpg | nan | 9.7194 | 3.2233 | 0.3643 | 0.3750 | 27.4766 | 27.9221 | clamped | rejected | both rejected |
| train | ml/data/raw/PXL_20260125_114844155.jpg | nan | 8.7886 | 1.0305 | 0.3807 | 0.3281 | 21.4412 | 24.4695 | accepted | rejected | keras accepted, int8 rejected |
| train | ml/data/raw/PXL_20260125_114848857.jpg | nan | 21.0567 | 1.8742 | 0.4655 | 0.3086 | 21.3519 | 28.9474 | accepted | rejected | keras accepted, int8 rejected |
| train | ml/data/raw/PXL_20260125_114901695.jpg | 7.2826 | 37.0949 | 3.7746 | 0.4003 | 0.3750 | 21.3830 | 28.3050 | accepted | accepted | both accepted |
| train | ml/data/raw/PXL_20260125_114851923.jpg | 3.8117 | 19.1253 | 2.0487 | 0.4831 | 0.4492 | 19.7804 | 20.9625 | accepted | accepted | both accepted |
| train | ml/data/raw/PXL_20260125_114859876.jpg | 2.5486 | 14.8905 | 1.9946 | 0.6041 | 0.3516 | 17.5513 | 23.7541 | accepted | accepted | both accepted |
| train | ml/data/raw/PXL_20260125_114857049.jpg | 2.3131 | 9.6926 | 0.8845 | 0.8856 | 0.5508 | 14.4859 | 17.4253 | accepted | accepted | both accepted |
| train | ml/data/raw/PXL_20260125_114843309.jpg | 2.2977 | 10.3707 | 2.5194 | 0.4007 | 0.3750 | 21.0331 | 21.5643 | accepted | accepted | both accepted |
| train | ml/data/raw/PXL_20260125_114848028.jpg | 1.8079 | 6.3019 | 2.9038 | 0.7910 | 0.3984 | 15.2077 | 19.8536 | accepted | accepted | both accepted |
| train | ml/data/raw/PXL_20260125_114532964.jpg | 1.6007 | 6.3235 | 1.7929 | 0.5467 | 0.5000 | 19.1709 | 19.2660 | accepted | accepted | both accepted |
| train | ml/data/raw/PXL_20260125_114842502.jpg | 1.5212 | 6.1047 | 0.6205 | 0.6106 | 0.5508 | 15.9171 | 16.0611 | accepted | accepted | both accepted |
| train | ml/data/raw/PXL_20260125_114541776.jpg | 1.4582 | 15.9240 | 3.7949 | 0.4148 | 0.3750 | 23.4806 | 27.5480 | accepted | accepted | both accepted |
| train | ml/data/raw/PXL_20260125_114543193.jpg | 1.2519 | 5.1089 | 1.5429 | 0.5866 | 0.3750 | 19.4157 | 21.9176 | accepted | accepted | both accepted |
| train | ml/data/raw/PXL_20260125_114517176.jpg | 0.9391 | 5.2872 | 1.5601 | 0.3694 | 0.3516 | 20.4134 | 22.1647 | accepted | accepted | both accepted |
| train | ml/data/raw/PXL_20260125_114540819.jpg | 0.8666 | 5.7791 | 1.3599 | 0.4484 | 0.3984 | 19.0032 | 23.4400 | accepted | clamped | both rejected |
| train | ml/data/raw/PXL_20260125_114908265.jpg | 0.7646 | 24.8132 | 2.3053 | 0.7103 | 0.3516 | 15.3340 | 25.9127 | accepted | accepted | both accepted |
| train | ml/data/raw/PXL_20260125_114845174.jpg | 0.7284 | 6.6829 | 1.0404 | 0.4265 | 0.3750 | 20.8160 | 24.6544 | accepted | accepted | both accepted |
| train | ml/data/raw/PXL_20260125_114751509.jpg | 0.5977 | 9.8246 | 2.3360 | 0.4172 | 0.3984 | 27.0469 | 25.8192 | accepted | accepted | both accepted |

## Guardrail Disagreements

| split | image_path | disagreement_type | keras_guardrail_status | int8_guardrail_status | temperature_delta_c | tip_delta_px | keras_tip_heatmap_peak_value | int8_tip_heatmap_peak_value | keras_tip_heatmap_spread_px | int8_tip_heatmap_spread_px | center_delta_px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | ml/data/raw/PXL_20260125_114540819.jpg | both rejected | accepted | clamped | 0.8666 | 5.7791 | 0.4484 | 0.3984 | 19.0032 | 23.4400 | 1.3599 |
| train | ml/data/raw/PXL_20260125_114544203.jpg | both rejected | clamped | rejected | nan | 8.6492 | 0.3737 | 0.3281 | 21.3710 | 24.7239 | 1.3219 |
| train | ml/data/raw/PXL_20260125_114552322.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 12.1918 | 0.4856 | 0.3086 | 17.7302 | 24.4129 | 1.5743 |
| train | ml/data/raw/PXL_20260125_114554365.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 36.6934 | 0.3522 | 0.3516 | 25.1058 | 32.1860 | 4.1284 |
| train | ml/data/raw/PXL_20260125_114631441.jpg | both rejected | rejected | clamped | nan | 10.6746 | 0.3457 | 0.3516 | 21.0498 | 20.7213 | 1.2105 |
| train | ml/data/raw/PXL_20260125_114637155.jpg | both rejected | rejected | clamped | nan | 11.0740 | 0.3164 | 0.3750 | 21.3732 | 19.5056 | 1.3218 |
| train | ml/data/raw/PXL_20260125_114642671.jpg | both rejected | clamped | rejected | nan | 9.7194 | 0.3643 | 0.3750 | 27.4766 | 27.9221 | 3.2233 |
| train | ml/data/raw/PXL_20260125_114844155.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 8.7886 | 0.3807 | 0.3281 | 21.4412 | 24.4695 | 1.0305 |
| train | ml/data/raw/PXL_20260125_114848857.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 21.0567 | 0.4655 | 0.3086 | 21.3519 | 28.9474 | 1.8742 |
| train | ml/data/raw/PXL_20260125_114912074.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 11.6881 | 0.3861 | 0.3086 | 23.1109 | 26.2567 | 1.1483 |
| train | ml/data/raw/PXL_20260125_114912923.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 9.2327 | 0.3797 | 0.3281 | 20.3057 | 24.6430 | 1.4015 |
| train | ml/data/raw/PXL_20260125_114913726.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 12.4941 | 0.5452 | 0.3984 | 25.3063 | 27.8325 | 3.1553 |
| train | ml/data/raw/PXL_20260125_114959197.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 20.0470 | 0.3596 | 0.3086 | 21.9015 | 26.1984 | 4.8329 |
| train | ml/data/raw/PXL_20260125_115143033.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 24.2353 | 0.4466 | 0.2852 | 18.9672 | 25.3520 | 2.4440 |
| train | ml/data/raw/PXL_20260125_115143850.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 2.0185 | 0.6458 | 0.3281 | 15.4199 | 19.2245 | 0.6920 |
| train | ml/data/raw/PXL_20260125_115144597.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 26.5132 | 0.6872 | 0.2852 | 16.3106 | 24.6876 | 1.7564 |
| train | ml/data/raw/PXL_20260125_115207208.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 3.5908 | 0.7620 | 0.3086 | 14.6685 | 20.0488 | 2.6570 |
| train | ml/data/raw/PXL_20260125_115541259.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 41.7040 | 0.6358 | 0.4727 | 18.7228 | 28.5533 | 6.5258 |
| train | ml/data/raw/PXL_20260125_115741590.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 24.8362 | 0.6393 | 0.3086 | 19.7433 | 26.0765 | 2.1556 |
| train | ml/data/raw/PXL_20260125_115743603.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 41.8397 | 0.7009 | 0.3281 | 19.1928 | 28.9114 | 5.1513 |
| train | ml/data/raw/PXL_20260125_115946935.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 13.9037 | 0.3818 | 0.3281 | 21.8332 | 27.2568 | 2.9905 |
| train | ml/data/raw/PXL_20260125_115950638.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 21.9343 | 0.5155 | 0.3516 | 24.8227 | 28.0452 | 4.2720 |
| train | ml/data/raw/PXL_20260125_120021074.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 28.6161 | 0.4650 | 0.3750 | 27.2947 | 30.2769 | 6.4515 |
| train | ml/data/raw/PXL_20260125_120026205.jpg | keras rejected, int8 accepted | rejected | accepted | nan | 51.2192 | 0.4223 | 0.5000 | 26.8547 | 22.4801 | 5.7087 |
| train | ml/data/raw/PXL_20260125_120026856.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 10.3619 | 0.5037 | 0.5508 | 22.8704 | 24.7676 | 1.8880 |
| train | ml/data/raw/PXL_20260125_120050121.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 17.6148 | 0.3547 | 0.3086 | 17.3244 | 22.2241 | 8.9012 |
| train | ml/data/raw/PXL_20260125_120122658.jpg | keras rejected, int8 accepted | rejected | accepted | nan | 13.3598 | 0.3350 | 0.3516 | 22.0296 | 26.4642 | 1.5217 |
| train | ml/data/raw/PXL_20260125_120157736.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 6.5155 | 0.4924 | 0.2852 | 16.5626 | 19.9179 | 2.4530 |
| train | ml/data/raw/PXL_20260125_120205746.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 13.0611 | 0.9834 | 0.9023 | 13.3576 | 13.1851 | 3.4644 |
| train | ml/data/raw/PXL_20260125_120206822.jpg | both rejected | clamped | accepted | 0.2809 | 8.5989 | 0.9233 | 0.5273 | 14.3300 | 18.0125 | 3.7791 |
| train | ml/data/raw/PXL_20260125_120216879.jpg | keras rejected, int8 accepted | rejected | accepted | nan | 20.6888 | 0.3300 | 0.3516 | 21.8843 | 25.9357 | 3.3298 |
| train | ml/data/raw/PXL_20260125_120217738.jpg | both rejected | rejected | clamped | nan | 3.9815 | 0.9400 | 0.7734 | 14.3185 | 14.6627 | 5.4149 |
| train | ml/data/raw/PXL_20260125_120218153.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 6.8676 | 0.4368 | 0.3281 | 20.0201 | 20.5330 | 4.1363 |
| train | ml/data/raw/PXL_20260125_120220417.jpg | keras rejected, int8 accepted | rejected | accepted | nan | 16.2242 | 0.3600 | 0.3750 | 23.8547 | 25.5972 | 4.6605 |
| train | ml/data/raw/PXL_20260125_120228422.jpg | keras rejected, int8 accepted | rejected | accepted | nan | 23.9545 | 0.3098 | 0.3516 | 26.7015 | 23.0121 | 4.1854 |
| train | ml/data/raw/PXL_20260125_120228748.jpg | keras rejected, int8 accepted | rejected | accepted | nan | 26.1738 | 0.2699 | 0.3516 | 21.4637 | 23.2738 | 4.2757 |
| train | ml/data/raw/PXL_20260125_120246531.jpg | keras rejected, int8 accepted | rejected | accepted | nan | 45.4893 | 0.3490 | 0.3750 | 20.9674 | 26.6929 | 3.5967 |
| val | ml/data/raw/PXL_20260125_114924323.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 49.4906 | 0.4912 | 0.2656 | 22.4195 | 25.6626 | 6.2643 |
| val | ml/data/raw/PXL_20260125_115343746.jpg | keras rejected, int8 accepted | rejected | accepted | nan | 3.3792 | 0.3141 | 0.3750 | 19.9848 | 15.7301 | 2.0194 |
| val | ml/data/raw/PXL_20260125_115947232.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 10.9260 | 0.3604 | 0.3086 | 25.3122 | 25.7275 | 4.2478 |
| val | ml/data/raw/PXL_20260125_115948836.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 11.6834 | 0.4029 | 0.3086 | 19.7516 | 22.0838 | 2.2716 |
| val | ml/data/raw/PXL_20260125_115949960.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 12.9933 | 0.4253 | 0.3516 | 23.9245 | 29.7201 | 3.6261 |
| val | ml/data/raw/PXL_20260125_120025564.jpg | keras rejected, int8 accepted | rejected | accepted | nan | 22.6200 | 0.4154 | 0.3750 | 25.5720 | 25.9995 | 9.0141 |
| val | ml/data/raw/PXL_20260125_120227059.jpg | both rejected | rejected | clamped | nan | 32.5634 | 0.3041 | 0.3750 | 27.0860 | 23.8485 | 4.4895 |
| val | ml/data/raw/PXL_20260125_120252819.jpg | both rejected | accepted | clamped | 2.7884 | 15.2491 | 0.5589 | 0.4258 | 17.7934 | 19.3162 | 2.2547 |
| test | ml/data/raw/PXL_20260125_114534732.jpg | keras rejected, int8 accepted | rejected | accepted | nan | 9.7500 | 0.3471 | 0.3750 | 27.5917 | 29.1886 | 3.5070 |
| test | ml/data/raw/PXL_20260125_114536986.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 18.5110 | 0.4001 | 0.3086 | 23.5913 | 28.0656 | 2.3647 |
| test | ml/data/raw/PXL_20260125_114845992.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 14.1899 | 0.5914 | 0.3281 | 20.3191 | 24.6537 | 1.4190 |
| test | ml/data/raw/PXL_20260125_114914961.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 12.7368 | 0.4687 | 0.3281 | 23.1338 | 23.8935 | 3.5281 |
| test | ml/data/raw/PXL_20260125_114925728.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 18.9318 | 0.4850 | 0.2852 | 21.4442 | 27.8504 | 1.5653 |
| test | ml/data/raw/PXL_20260125_115125409.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 55.6026 | 0.3794 | 0.3281 | 24.9450 | 31.2925 | 2.0376 |
| test | ml/data/raw/PXL_20260125_115209157.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 7.6865 | 0.5243 | 0.2852 | 19.2999 | 24.2036 | 5.4189 |
| test | ml/data/raw/PXL_20260125_115612552.jpg | keras rejected, int8 accepted | rejected | accepted | nan | 41.0805 | 0.3789 | 0.5273 | 29.6850 | 16.4528 | 7.3759 |
| test | ml/data/raw/PXL_20260125_115957052.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 30.1352 | 0.4778 | 0.3750 | 27.6787 | 30.8662 | 5.6252 |
| test | ml/data/raw/PXL_20260125_120020529.jpg | keras accepted, int8 rejected | accepted | rejected | nan | 3.6469 | 0.4257 | 0.3086 | 26.4069 | 24.1458 | 4.7855 |
| test | ml/data/raw/PXL_20260125_120159244.jpg | both rejected | clamped | rejected | nan | 9.2090 | 0.4709 | 0.2852 | 20.5508 | 21.2604 | 2.0584 |
| test | ml/data/raw/PXL_20260125_120219516.jpg | keras rejected, int8 accepted | rejected | accepted | nan | 38.8823 | 0.4060 | 0.3750 | 28.7895 | 25.2261 | 4.4407 |

## Notes

- The overlay directories contain the top tip deltas, top temperature deltas, and every guardrail disagreement.
- This autopsy is derived from the saved replay predictions, so it matches the earlier Keras and INT8 replay path.