/**
 * @file    app_ai_config.h
 * @brief   Shared compile-time configuration macros for the AI pipeline.
 *
 * All macros in this header were previously defined in app_ai.c's
 * Private Includes / Private Define sections.  They are now available
 * to every compilation unit that participates in the AI pipeline so
 * that modular .c files do not need their own local copies.
 */

#ifndef __APP_AI_CONFIG_H
#define __APP_AI_CONFIG_H

/* Include dependencies for macros that reference other types */
#include "app_memory_budget.h"  /* for CAMERA_CAPTURE_* macros referenced below */

/* ---- ATON reloc mode ---- */
#ifndef LL_ATON_RT_RELOC
#define LL_ATON_RT_RELOC 1
#endif

/* ---- ATON platform / OSAL ---- */
#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX

/* ---- Feature flags ---- */
/*
 * Keep the very noisy tensor and patch dumps behind a toggle so the normal
 * AI status logs still use the shared debug console directly.
 */
#ifndef APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
#define APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS 0
#endif
#ifndef APP_AI_ENABLE_XSPI2_VERBOSE_LOGS
#define APP_AI_ENABLE_XSPI2_VERBOSE_LOGS 0U
#endif
#undef APP_AI_ENABLE_XSPI2_VERBOSE_LOGS
#define APP_AI_ENABLE_XSPI2_VERBOSE_LOGS 0U
#ifndef APP_AI_ENABLE_RUNTIME_METRICS
#define APP_AI_ENABLE_RUNTIME_METRICS 1U
#endif
/* Keep rectifier diagnostics available even when the rest of the verbose
 * console logging stays off. This is a temporary bring-up aid for crop
 * debugging and can be flipped back to 0 once the rectifier is stable. */
#ifndef APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
#define APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS 0U
#endif
/* The lower inset / dark-blob distractor is now a known live failure mode, so
 * keep the inner-Celsius mask on by default unless we explicitly disable it. */
#ifndef APP_AI_ENABLE_INNER_CELSIUS_MASK
#define APP_AI_ENABLE_INNER_CELSIUS_MASK 1U
#endif
/* Prod v0.8 freeze: keep the 3-frame burst median enabled so the live board
 * path matches the offline recipe and does not jump on glare-heavy captures. */
#ifndef APP_AI_ENABLE_INFERENCE_BURST_SMOOTHING
#define APP_AI_ENABLE_INFERENCE_BURST_SMOOTHING 1U
#endif
/* The ATON runtime has been faulting immediately after the per-frame reset
 * path, so keep that reset behind a switch while we verify whether the model
 * can run cleanly with one-shot initialization only. */
#ifndef APP_AI_RESET_NETWORK_EACH_INFERENCE
#define APP_AI_RESET_NETWORK_EACH_INFERENCE 0
#endif
/* The OBB stage is now fallback-only. The live board inference path routes
 * through the tip-focus UNet heatmap model first, but we keep the old
 * crop front-end behind a switch so it can still be re-enabled for debug. */
#ifndef APP_AI_ENABLE_OBB_STAGE
#define APP_AI_ENABLE_OBB_STAGE 1U
#endif
/* Keep the OBB-to-tip-focus handoff enabled in the live path. The UNet should
 * consume the OBB crop unless the OBB stage itself fails. */
#ifndef APP_AI_ENABLE_OBB_TIP_FOCUS_CROP_HANOFF
#define APP_AI_ENABLE_OBB_TIP_FOCUS_CROP_HANOFF 1U
#endif
/* Optional CPU refinement for the OBB crop.  This keeps the live path tight
 * without bringing back the old rectifier or source-crop-box stages. */
#ifndef APP_AI_ENABLE_LUMA_REFINER
#define APP_AI_ENABLE_LUMA_REFINER 1U
#endif
#ifndef APP_AI_ENABLE_OBB_DECODE_DIAGNOSTICS
#define APP_AI_ENABLE_OBB_DECODE_DIAGNOSTICS 1U
#endif
#ifndef APP_AI_ENABLE_TIP_FOCUS_INPUT_DUMP
#define APP_AI_ENABLE_TIP_FOCUS_INPUT_DUMP 1U
#endif
/* Production path: model images are provisioned via xSPI flash script.
 * Keep SD-based scalar reprovision disabled in live runtime. */
#ifndef APP_AI_ENABLE_SCALAR_SD_REPROVISION
#define APP_AI_ENABLE_SCALAR_SD_REPROVISION 0U
#endif

/* ---- Application constants (legacy Camera / capture wrappers) ---- */
#define APP_AI_CACHE_LINE_BYTES 32U
#define APP_AI_CAPTURE_FRAME_WIDTH_PIXELS CAMERA_CAPTURE_WIDTH_PIXELS
#define APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS CAMERA_CAPTURE_HEIGHT_PIXELS
#define APP_AI_CAPTURE_FRAME_BYTES_PER_PIXEL CAMERA_CAPTURE_BYTES_PER_PIXEL
#define APP_AI_CAPTURE_FRAME_BYTES \
	(APP_AI_CAPTURE_FRAME_WIDTH_PIXELS * APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS * APP_AI_CAPTURE_FRAME_BYTES_PER_PIXEL)
/* Rectified scalar reader: 224x224x3 float RGB input. The offline prod v0.8
 * recipe uses the luma-refined crop to feed this float path, then applies the
 * external calibration/postprocess in firmware. */
#define APP_AI_MODEL_INPUT_FLOAT_COUNT \
	(APP_AI_CAPTURE_FRAME_WIDTH_PIXELS * APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS * 3U)
/* Bright-object detector threshold used to find the gauge face before
 * cropping and resizing the tensor. 80 was a better fit than 60 on captured
 * frames because 60 was too loose and pulled in most of the background. */
#define APP_AI_GAUGE_BRIGHT_THRESHOLD 80U
/* Ignore a thin border while estimating the bright bbox so edge glare does not
 * drag the crop to x=0 or y=0. */
#define APP_AI_GAUGE_CROP_BORDER_PIXELS 16U
/* The newest live board captures want a slightly tighter crop that is biased
 * left and a little above the bright centroid. That keeps more of the dial
 * face and upper markings in frame and reduces the background washout that
 * was pulling the prod model low. */
#define APP_AI_GAUGE_CROP_WIDTH_SCALE_NUMERATOR 17U
#define APP_AI_GAUGE_CROP_WIDTH_SCALE_DENOMINATOR 20U
#define APP_AI_GAUGE_CROP_HEIGHT_SCALE_NUMERATOR 17U
#define APP_AI_GAUGE_CROP_HEIGHT_SCALE_DENOMINATOR 20U
#define APP_AI_GAUGE_CROP_CENTER_X_BIAS_PIXELS 24U
/* Pull the crop upward by a bounded fraction of the crop height so the top
 * of the dial stays visible without drifting off the gauge. */
#define APP_AI_GAUGE_CROP_CENTER_Y_BIAS_RATIO 0.11f
#define APP_AI_GAUGE_CROP_CENTER_Y_BIAS_MIN_PIXELS 8U
#define APP_AI_GAUGE_CROP_CENTER_Y_BIAS_MAX_PIXELS 18U
/* Use a lightweight CPU-side luma heuristic to locate the gauge dial face
 * when the model-driven localizer is unavailable. Replaces the OBB NPU model
 * which had a relocation table conflict (tip-focus reloc was used for OBB,
 * causing MemManage fault). */
/* The deployed board path should stay on the model-driven crop contract.
 * If OBB does not yield a usable box, we fall back to the fixed training crop
 * rather than a heuristic full-frame scan that can diverge from the model
 * replay path on target. */
#define APP_AI_USE_ADAPTIVE_GAUGE_CROP 0U
/* Match the training/replay preprocessing again by keeping the scalar path on
 * the RGB bilinear resize branch. The earlier hard fault was a memory-layout
 * issue, not a problem with this branch itself. */
#define APP_AI_YUV422_INPUT_LUMA_ONLY 0U
/* Use a full affine crop->tensor mapping instead of aspect-preserving
 * letterbox padding. The padded resize was introducing large zero bands on
 * non-square crops and hurting hot-end needle coverage near the edges. */
#define APP_AI_ENABLE_AFFINE_FILL_RESIZE 1U
/* Debug switch: set to 1 to stop the scalar stage after tensor fill. */
#define APP_AI_BYPASS_SCALAR_INFERENCE 0U
#define APP_AI_MODEL_INPUT_FLOAT_BYTES \
	(APP_AI_MODEL_INPUT_FLOAT_COUNT * sizeof(float))
/* Rectified scalar output: a single float temperature value. */
#define APP_AI_MODEL_OUTPUT_FLOAT_COUNT 1U
#define APP_AI_MODEL_OUTPUT_FLOAT_BYTES \
	(APP_AI_MODEL_OUTPUT_FLOAT_COUNT * sizeof(float))
/* Circular decode constants from gauge_calibration_parameters.toml.
 * The gauge sweeps clockwise from 135 deg (2.356 rad) over 270 deg (4.712 rad).
 * Value range: -30 C to +50 C. */
/* Keep the circular vote at 224 bins even as the square image geometry moves
 * to 224x224; the decode helper still expects the original angular resolution.
 */
#define APP_AI_POLAR_VOTE_BINS 224U
#define APP_AI_POLAR_VOTE_MIN_ANGLE_RAD 3.927f   /* 225° — 7:30 o'clock (scale start) */
#define APP_AI_POLAR_VOTE_SWEEP_RAD 4.712f       /* 270° sweep clockwise */
#define APP_AI_POLAR_VOTE_MIN_VALUE_C (-30.0f)   /* inner scale minimum */
#define APP_AI_POLAR_VOTE_MAX_VALUE_C 50.0f      /* inner scale maximum */
/* Match the offline V28 recipe by searching a small center neighborhood
 * before building the polar tensor. The exact training helper uses a
 * 3x3 sweep around the nominal pivot, which is enough to absorb a few
 * pixels of framing drift without turning the preprocess into a full
 * optimizer. */
#define APP_AI_POLAR_CENTER_SEARCH_PIXELS 5U
/* V28 polar input quantisation: float [0,1] -> int8 via q = round(x*255) - 128. */
#define APP_AI_POLAR_INPUT_SCALE 0.0039215687f
#define APP_AI_POLAR_INPUT_ZERO_POINT (-128)
/* V28 polar output quantisation: dequantize via float_val = (q - 16) * 0.093767159. */
#define APP_AI_POLAR_OUTPUT_SCALE 0.093767159f
#define APP_AI_POLAR_OUTPUT_ZERO_POINT 16
/* Dead-zone mask constant: logits outside the gauge sweep are set to this
 * large negative value so they become zero after softmax. */
#define APP_AI_POLAR_MASK_LOGIT (-1.0e9f)
/* Shared inference smoothing and plausibility limits used by both the live
 * tip-focus path and the legacy fallback path. */
#define APP_AI_INFERENCE_BURST_HISTORY_SIZE 3U
#define APP_AI_INFERENCE_BURST_RESET_DELTA_C 12.0f
#define APP_AI_INFERENCE_VALUE_MIN_C (-80.0f)
#define APP_AI_INFERENCE_VALUE_MAX_C (180.0f)
/* Legacy model constants are compiled only when the tip-focus path is off. */
#if !APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
/* Scalar model image path (deprecated — retained for the scalar stage spec). */
#define APP_AI_SCALAR_XSPI2_MODEL_IMAGE_PATH \
	"packages/mobilenetv2_rectified_scalar_finetune_v2/st_ai_output/scalar_full_finetune_from_best_piecewise_calibrated_int8_atonbuf.xSPI2.raw"
#define APP_AI_CENTER_DETECTOR_XSPI2_MODEL_IMAGE_PATH \
	"packages/heatmap_cd_v4s_80/st_ai_output/heatmap_cd_atonbuf.xSPI2.raw" /* DS-CNN v4-S 80×80, ~268 KB */
#define APP_AI_RECTIFIER_XSPI2_MODEL_IMAGE_PATH \
	"atonbuf.rectifier.xSPI2.raw"
#endif
/* Live board models use the recent OBB winner plus the board-fit tip-focus
 * heatmap package. */
#define APP_AI_OBB_XSPI2_MODEL_IMAGE_PATH \
	"packages/obb_box_board_bbox_deploy_candidate/st_ai_output/obb_box_board_bbox_deploy_candidate_atonbuf.xSPI2.raw" /* Board bbox OBB deploy candidate, ~664 KiB */
#define APP_AI_TIP_FOCUS_XSPI2_MODEL_IMAGE_PATH \
	"packages/tip_focus_v18_int8_n6_npu/st_ai_output/tip_focus_v18_int8_atonbuf.xSPI2.raw" /* Tip-focus v18 NPU package, ~815 KiB */
#define APP_AI_XSPI2_MODEL_IMAGE_PATH APP_AI_TIP_FOCUS_XSPI2_MODEL_IMAGE_PATH
#define APP_AI_XSPI2_PROGRAM_CHUNK_BYTES 4096U
#define APP_AI_XSPI2_ERASE_BLOCK_BYTES (64U * 1024U)
#define APP_AI_XSPI2_PROBE_BYTES 16U
/* Keep the rectifier crop slightly larger than the raw box so the scalar head
 * still sees the needle and a bit of surrounding dial context. */
#define APP_AI_RECTIFIER_CROP_SCALE 1.80f
/* Prod v0.8 offline freeze used a 1.20x OBB crop scale before the luma
 * refinement stage constrained the final crop window. */
#define APP_AI_OBB_CROP_SCALE 1.20f
/* Keep the OBB localizer bounded, but give it enough time to finish on the
 * 60 s capture cadence. The earlier 10 s cap forced a fallback before the
 * deployed localizer could converge on harder frames. */
/* Keep the live OBB path aligned with the offline board replay window. The
 * older 0.60 threshold was too strict and pushed valid crops into fallback. */
#define APP_AI_OBB_TRAINING_CROP_MIN_RATIO 0.15f
/* The hot-end rectified crops in the training set can be substantially taller
 * than the fixed fallback crop, so keep the live OBB path from rejecting them
 * back to the wrong framing. */
/* Hot-end training crops can be noticeably taller than the baseline crop.
 * Keep them on the OBB path instead of forcing the fixed fallback crop. */
#define APP_AI_OBB_TRAINING_CROP_MAX_RATIO 1.60f
/* When OBB box size drifts, keep scalar dimensions on training crop but let
 * OBB nudge the crop centre slightly instead of hard-falling to fixed crop. */
#define APP_AI_OBB_CENTER_BLEND_NUMERATOR 1U
#define APP_AI_OBB_CENTER_BLEND_DENOMINATOR 1U
#define APP_AI_OBB_CENTER_MIN_RATIO 0.10f
#define APP_AI_OBB_CENTER_MAX_RATIO 0.90f
/* Match the Python rectifier evaluator: never let the predicted box collapse
 * below a tiny fraction of the canvas, or the scalar stage degenerates to a
 * 1x1 crop. */
#define APP_AI_RECTIFIER_MIN_BOX_RATIO 0.05f
/* Keep the OBB decoder from collapsing into a tiny crop when the localizer
 * gets uncertain. */
#define APP_AI_OBB_MIN_BOX_RATIO 0.05f
/* Legacy decode constants kept for the shared OBB helper. The active board
 * bbox deploy candidate exports one scalar confidence plus a 4-value bbox
 * vector, but the older helper still decodes a 6+2 channel OBB layout. */
#define APP_AI_OBB_OUTPUT_SCALE      0.003921569f      /* input scale (unused for multi-output) */
#define APP_AI_OBB_OUTPUT_ZERO_POINT (-128)            /* input zp (unused for multi-output) */
#define APP_AI_OBB_HEATMAP_SCALE     0.047995351f      /* Transpose_50_out_0 (α=1.25) */
#define APP_AI_OBB_HEATMAP_ZP        14
#define APP_AI_OBB_BOX_SCALE         0.0078125f        /* Transpose_59_out_0 */
#define APP_AI_OBB_BOX_ZP            0
#define APP_AI_OBB_ANGLE_SCALE       0.042166740f      /* Transpose_53_out_0 (α=1.25) */
#define APP_AI_OBB_ANGLE_ZP          1
#define APP_AI_OBB_OUTPUT_CHANNELS   8U  /* 6 bbox channels + 2 centre channels for the legacy helper */
#define APP_AI_OBB_HEATMAP_SIZE      40U
#define APP_AI_OBB_HEATMAP_PIXELS    (APP_AI_OBB_HEATMAP_SIZE * APP_AI_OBB_HEATMAP_SIZE)  /* 1600 */
/* The OBB crop should still be a real crop, not a 1x1 or 8x8 window. */
#define APP_AI_OBB_MIN_CROP_SIZE_PIXELS 48.0f
#define APP_AI_SOURCE_CROP_BOX_MIN_CROP_SIZE_PIXELS 8.0f
/* Accept any box where both w and h are in [0.2, 1.5] of the frame dimension.
 * The rectifier outputs normalised cx/cy/w/h so these limits reject degenerate
 * (near-zero) or wildly oversized predictions while passing plausible dial
 * crops at close-up and normal camera distances. */
#define APP_AI_RECTIFIER_FALLBACK_MIN_BOX_RATIO 0.2f
#define APP_AI_RECTIFIER_FALLBACK_MAX_BOX_RATIO 1.5f
/* Path-1 soft-attention mode: ignore rectifier's predicted (w,h), use prod's
 * fixed training-crop dimensions centred on rectifier's predicted (cx,cy).
 * Keeps scalar input distribution within prod's training distribution while
 * letting framing follow the gauge across camera placements. Set to 0 to
 * restore the original (rectifier-sized box * crop scale) behaviour. */
#define APP_AI_RECTIFIER_FIXED_SCALE_CROP 0U
/* Rectifier center is still a little noisy on the current board captures, so
 * bias the crop center back toward the stable training-crop center instead of
 * trusting the raw rectifier center outright. This keeps us close to the
 * scalar's trained framing while still allowing a small amount of movement. */
#define APP_AI_RECTIFIER_CENTER_BLEND_NUMERATOR 1U
#define APP_AI_RECTIFIER_CENTER_BLEND_DENOMINATOR 5U
/* Reject the rectifier's centre prediction if it falls outside the central
 * 80% of the frame in either axis — that's almost certainly a runaway
 * prediction and we should fall back to the static training crop instead of
 * shifting the scalar's framing into the bezel/background. Only used when
 * APP_AI_RECTIFIER_FIXED_SCALE_CROP is enabled. */
#define APP_AI_RECTIFIER_CENTER_MIN_RATIO 0.10f
#define APP_AI_RECTIFIER_CENTER_MAX_RATIO 0.90f
/* Scalar vote-logit decode settings.
 * Keep these aligned with the training/eval decode path:
 *   mode=topk_expectation, topk=8, temperature=1.0.
 * The current gauge span in gauge_calibration_parameters.toml is [-30, 50] C.
 */
#define APP_AI_SCALAR_DECODE_TOPK 8U
#define APP_AI_SCALAR_DECODE_TEMPERATURE 1.0f
#define APP_AI_SCALAR_DECODE_VALUE_MIN_C (-30.0f)
#define APP_AI_SCALAR_DECODE_VALUE_MAX_C (50.0f)
/* Keep the OBB stage timeout available in the unified live build. */
#define APP_AI_OBB_INFERENCE_TIMEOUT_MS 30000U  /* Give the board-local OBB path room to finish cleanly on hardware. */
/* The current board-local OBB schedule needs more than 1000 epoch steps to
 * finish cleanly on the STM32N6 path, so keep a larger guardrail and let the
 * wall-clock timeout remain the hard safety cap. */
#define APP_AI_OBB_EPOCH_BUDGET_STEPS 6000U
/* xSPI2 window base address (chip address 0). */
#define APP_AI_XSPI2_CHIP_BASE_ADDR 0x70000000UL
/* Tip-focus UNet v18 model: xSPI2 weights at 0x70400000. */
#define APP_AI_XSPI2_TIP_FOCUS_BASE_ADDR 0x70400000UL
#define APP_AI_XSPI2_TIP_FOCUS_CHIP_OFFSET (APP_AI_XSPI2_TIP_FOCUS_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)
/* Board bbox OBB model: xSPI2 weights at 0x71400000. */
#define APP_AI_XSPI2_OBB_BASE_ADDR 0x71400000UL
#define APP_AI_XSPI2_OBB_CHIP_OFFSET (APP_AI_XSPI2_OBB_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)
/* Legacy aliases used by the shared xSPI2 helpers now point at the live
 * tip-focus slot so the generic probe/logging code matches the active model. */
#define APP_AI_XSPI2_MODEL_BASE_ADDR APP_AI_XSPI2_TIP_FOCUS_BASE_ADDR
#define APP_AI_XSPI2_MODEL_CHIP_OFFSET APP_AI_XSPI2_TIP_FOCUS_CHIP_OFFSET
/* Shared runtime types are used by both the live tip-focus path and the
 * compile-guarded legacy fallback helpers, so keep them available in both
 * build modes. */
/* Shared types are now in app_ai_types.h — included above. */
/* Scalar model: immediately after FSBL (0x70000000) + App (0x70100000, 1 MB
 * window). Must match FLASH_SCALAR address in flash_boot.ps1.
 * Size: ~3.07 MB — occupies 0x70200000–0x7051FFFF (50 × 64 KB blocks). */
#if !APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
#define APP_AI_XSPI2_SCALAR_BASE_ADDR 0x70200000UL
#define APP_AI_XSPI2_SCALAR_CHIP_OFFSET (APP_AI_XSPI2_SCALAR_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)
/* Rectifier model: immediately after scalar region (aligned to next 64 KB).
 * Must match FLASH_RECTIFIER address in flash_boot.ps1.
 * Size: ~118 KB — occupies 0x70600000–0x7053FFFF (2 × 64 KB blocks). */
#define APP_AI_XSPI2_RECTIFIER_BASE_ADDR 0x70600000UL
#define APP_AI_XSPI2_RECTIFIER_CHIP_OFFSET (APP_AI_XSPI2_RECTIFIER_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)
/* Center detector model (DS-CNN v4): xSPI2 weights at 0x70200000.
 * The NPU reads weights directly from xSPI2 flash; activation scratch uses
 * xSPI1 hyperRAM (0x90000000) + on-chip AXISRAM2-6. */
#define APP_AI_XSPI2_CENTER_DETECTOR_BASE_ADDR APP_AI_XSPI2_SCALAR_BASE_ADDR
#define APP_AI_XSPI2_CENTER_DETECTOR_CHIP_OFFSET (APP_AI_XSPI2_CENTER_DETECTOR_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)
#endif
/* FileX can take a while to recover from card init retries or media errors.
 * Give the loader a longer window so we do not give up just before the stack
 * settles. */
#define APP_AI_FILEX_MEDIA_READY_TIMEOUT_MS 180000U
/* Tip-focus heatmap guardrail constants.
 * The runtime uses the replay angle convention (negated image Y) so the
 * board-side angle decode stays aligned with the packaged hard-case run.
 * Temperature mapping itself is shared through AppBaselineRuntime_. */
#define APP_AI_TIP_FOCUS_MODEL_INPUT_WIDTH_PIXELS 224U
#define APP_AI_TIP_FOCUS_MODEL_INPUT_HEIGHT_PIXELS 224U
#define APP_AI_TIP_FOCUS_COLD_ANGLE_DEG     135.0f
#define APP_AI_TIP_FOCUS_SWEEP_DEG          270.0f
#define APP_AI_TIP_FOCUS_HEATMAP_SIDE_PIXELS 56U
#define APP_AI_TIP_FOCUS_HEATMAP_PIXELS \
	(APP_AI_TIP_FOCUS_HEATMAP_SIDE_PIXELS * APP_AI_TIP_FOCUS_HEATMAP_SIDE_PIXELS)
#define APP_AI_TIP_FOCUS_SIMCC_BINS         APP_AI_TIP_FOCUS_HEATMAP_SIDE_PIXELS
#define APP_AI_TIP_FOCUS_CONFIDENCE_FLOOR   0.40f
/* The heatmap peaks still need a small floor so we can reject obviously flat
 * activations without over-fitting the gate to one camera session. */
#define APP_AI_TIP_FOCUS_AXIS_PEAK_FLOOR    0.06f
#define APP_AI_TIP_FOCUS_AXIS_SPREAD_MAX_PX  32.0f
#define APP_AI_TIP_FOCUS_TEMP_MIN_C         (-35.0f)
#define APP_AI_TIP_FOCUS_TEMP_MAX_C         55.0f
/* Median smoothing ring buffer for published tip-focus temperature. */
#define APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE 3U
#define APP_AI_TIP_FOCUS_MAX_OUTLIER_DELTA_C  5.0f
#define APP_AI_TIP_FOCUS_OUTLIER_RESET_STREAK 3U
#define APP_AI_TIP_FOCUS_MAX_INVALID_FRAMES   10U

#endif /* __APP_AI_CONFIG_H */
