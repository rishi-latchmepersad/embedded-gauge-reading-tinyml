# LittleGood exact-int8 board candidates

Date: 2026-07-29
Status: validated
Scope: STM32N657 LittleGood ellipse and needle localization pipeline

## Evidence

- Ellipse refiner: `ml/artifacts/ocdet_refiner_256_littlegood_v2/model_int8.tflite`.
- Ellipse evaluation: 97 untouched LittleGood test captures, validation-selected grayscale gamma `0.70`, proposer crop factor `3.2`, center blend `0.85`, center component MAE `2.413594 px`.
- Ellipse package: `firmware/stm32/n657/st_ai_output/packages/gauge_ellipse_littlegood_v2_gamma070_int8_n6_npu/`.
- Required ellipse proposer package: `firmware/stm32/n657/st_ai_output/packages/ocdet_ellipse_320_v2_int8_n6_npu/`.
- Center/tip model: `ml/artifacts/gauge_center_tip_littlegood_unet_v1/model_int8.tflite`.
- Center/tip evaluation: 97 untouched LittleGood test captures, center MAE `1.23649 px`, tip MAE `4.04729 px`, center within 8 px `100%`, tip within 8 px `96.907%`.
- Center/tip package: `firmware/stm32/n657/st_ai_output/packages/gauge_center_tip_littlegood_unet_v1_int8_n6_npu/`.
- ST Edge AI generated `c_info.json`, `network.csv`, relocatable C/H, and xSPI2 blobs for all three package directories. Generated and Windows-copy xSPI2 blobs were byte-for-byte equal.
- Compiler-reported ellipse refiner activation allocation: `786432` bytes. The model-level center/tip bound is `819200` bytes. Both are below the `2.5 MiB` gate.

## Decision

Use these candidates for the next board integration pass. Preserve the existing active packages and flash script until the firmware call sites are updated for the ellipse two-stage contract (`320x320 proposer -> 256x256 refiner`) and the center/tip `160x160x2 -> 80x80x2` heatmap contract. Flashing and post-power-cycle hardware capture validation are still required before declaring board deployment complete.
