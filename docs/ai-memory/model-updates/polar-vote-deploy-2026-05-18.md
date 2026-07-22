# 2026-05-18 Polar-Vote Deploy Integration Note

Goal:
- Package the exact hard-case winner decode behavior into firmware/runtime.
- Rebuild and flash the board with updated model + signatures.

Scope note:
- This file tracks the deploy-specific details.
- The broader runtime + training + integration chronology for this same period is recorded in:
  - `docs/ai-memory/model-updates/ml-experiments.md`
  - section: `2026-05-18 End-to-End Deployment + Firmware Integration Notes`

What was integrated:
- `firmware/stm32/n657/Appli/Src/app_ai.c`
  - Added scalar multi-bin logits decode using top-k expectation:
    - `topk=8`
    - `temperature=1.0`
    - mapped to `[-30C, 50C]`
  - Kept legacy scalar fallback for single-value outputs.
  - Added deterministic zero-fill when input tensor is wider than legacy `224x224x3` to avoid stale-SRAM channels.
  - Updated scalar xSPI2 signature constants to match the regenerated blob:
    - start: `c4f9c7e11eec1458c1296aa84446a93e`
    - tail: `00000000000000000000000000000017`

Model packaging:
- Source checkpoint:
  - `ml/artifacts/training/polar_vote_hardcases_errweighted_v1/best_weights.weights.h5`
- Exported int8 model:
  - `ml/artifacts/deployment/polar_vote_hardcases_errweighted_v1_int8/model_int8.tflite`
- Packaged into existing scalar integration workspace:
  - `firmware/stm32/n657/st_ai_output/packages/prod_model_v0.4_scalar_int8/...`
- Canonical scalar blob refreshed:
  - `firmware/stm32/n657/st_ai_output/atonbuf.xSPI2.raw`

Build + flash:
- Rebuilt Appli successfully via Windows toolchain.
- Flashed FSBL + scalar blob + rectifier blob + OBB blob + signed app successfully using:
  - `firmware/stm32/n657/flash_boot.bat`

Important caveat:
- The deployed polar-vote model input contract is `224x224x7` (`rgb_edge6_vote7`, polar representation), while the historical scalar firmware path was trained around `224x224x3` crop RGB.
- Current firmware now avoids undefined extra-channel data (zero-filled channels), but this is not full training-pipeline feature parity yet.
- If live accuracy remains poor, next required step is full on-device parity for the 7-channel polar feature pipeline (or train/deploy a model on the exact current on-device input representation).

2026-05-19 hot-end parity follow-up:
- The live OBB gate was still too strict for the hot rectified crops in `rectified_crop_boxes_v5_all.csv`, so the firmware was falling back to the fixed training crop on the exact band we care about.
- Relaxing `APP_AI_OBB_TRAINING_CROP_MAX_RATIO` to `1.60f` keeps the tall hot-end OBB crops on the training family so the firmware does not fall back to the fixed crop on the exact frames that need the broader framing.
- The deploy-time affine output calibration is now disabled for the polar-vote path so the board reports the raw model decode, matching the training script instead of a post-hoc fit.
