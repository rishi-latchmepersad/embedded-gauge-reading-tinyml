/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_ai.c
 * @brief   Minimal AI runtime bootstrap helpers.
 ******************************************************************************
 */
/* USER CODE END Header */

/* Keep the legacy tip-focus geometry and source-crop-box blocks disabled.
 * The live board now uses OBB -> center detector -> polar vote ->
 * temperature, but the center detector itself reads the stable training crop
 * directly so it stays in the domain it was trained on. The OBB/luma path is
 * still useful for diagnostics and future fallback logic. These defines must
 * appear before app_ai.h includes the tip-focus header, which uses #ifndef
 * guards that respect pre-include definitions. */
#ifdef APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
#undef APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
#endif
#define APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE 0U
#ifdef APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE
#undef APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE
#endif
#define APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE 0U

#include "app_ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stddef.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#include "debug_console.h"
#include "app_inference_calibration.h"
#include "app_inference_log_utils.h"
#include "app_memory_budget.h"
#include "app_gauge_geometry.h"
#include "app_baseline_runtime.h"
#include "app_inner_celsius_mask.h"
#include "ina219_power.h"
#include "inference_metrics.h"
#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX
#include "tx_api.h"
#include "ll_aton_rt_user_api.h"
#include "ll_aton.h"
#include "ll_aton_runtime.h"
#if defined(__has_include)
#if __has_include("ll_aton_lib_sw_operators.c")
/* ST's relocatable exports still call a few pure software tensor operators.
 * Pull the pack source in here so the generated networks can link without
 * making the firmware project depend on extra build-system edits. */
#include "ll_aton_lib_sw_operators.c"
#else
#error "Missing ST AI software-operator source in the pack include path."
#endif
#else
#include "ll_aton_lib_sw_operators.c"
#endif
#include "app_filex.h"
#include "stm32n6xx_nucleo_xspi.h"
#include "npu_cache.h"
#include "stm32n6xx_hal.h"
#include "app_center_detector.h"

/*
 * The STM32N6 ATON runtime library expects this wait-mask state symbol when
 * it is built in debug mode. We provide it here so the rebuilt runtime object
 * can stay in release mode and still link cleanly with the rest of the app.
 */
#ifndef NDEBUG
uint32_t volatile __ll_current_wait_mask = 0U;
#endif

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
/* The OBB stage is the live crop front-end for the center-detector pipeline.
 * Keep it behind a switch so we can still isolate it quickly if a future board
 * build needs a scalar-only fallback. */
#ifndef APP_AI_ENABLE_OBB_STAGE
#define APP_AI_ENABLE_OBB_STAGE 1U
#endif
/* Optional CPU refinement for the OBB crop.  This keeps the live path tight
 * without bringing back the old rectifier or source-crop-box stages. */
#ifndef APP_AI_ENABLE_LUMA_REFINER
#define APP_AI_ENABLE_LUMA_REFINER 1U
#endif
/* Production path: model images are provisioned via xSPI flash script.
 * Keep SD-based scalar reprovision disabled in live runtime. */
#ifndef APP_AI_ENABLE_SCALAR_SD_REPROVISION
#define APP_AI_ENABLE_SCALAR_SD_REPROVISION 0U
#endif
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define APP_AI_CACHE_LINE_BYTES 32U
#define APP_AI_CAPTURE_FRAME_WIDTH_PIXELS CAMERA_CAPTURE_WIDTH_PIXELS
#define APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS CAMERA_CAPTURE_HEIGHT_PIXELS
#define APP_AI_CAPTURE_FRAME_BYTES_PER_PIXEL CAMERA_CAPTURE_BYTES_PER_PIXEL
#define APP_AI_CAPTURE_FRAME_BYTES \
	(APP_AI_CAPTURE_FRAME_WIDTH_PIXELS * APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS * APP_AI_CAPTURE_FRAME_BYTES_PER_PIXEL)
/* Rectified scalar reader: 320x320x3 float RGB input. The offline prod v0.8
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
#define APP_AI_USE_ADAPTIVE_GAUGE_CROP 1U
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
 * to 320x320; the decode helper still expects the original angular resolution.
 */
#define APP_AI_POLAR_VOTE_BINS 224U
#define APP_AI_POLAR_VOTE_MIN_ANGLE_RAD 2.356f
#define APP_AI_POLAR_VOTE_SWEEP_RAD 4.712f
#define APP_AI_POLAR_VOTE_MIN_VALUE_C (-30.0f)
#define APP_AI_POLAR_VOTE_MAX_VALUE_C 50.0f
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
/* Scalar model image path (deprecated â€” retained for the scalar stage spec). */
#define APP_AI_SCALAR_XSPI2_MODEL_IMAGE_PATH \
	"packages/mobilenetv2_rectified_scalar_finetune_v2/st_ai_output/scalar_full_finetune_from_best_piecewise_calibrated_int8_atonbuf.xSPI2.raw"
#define APP_AI_CENTER_DETECTOR_XSPI2_MODEL_IMAGE_PATH \
	"packages/heatmap_cd_tiny/st_ai_output/heatmap_cd_atonbuf.AXISRAM2.raw"
#define APP_AI_RECTIFIER_XSPI2_MODEL_IMAGE_PATH \
	"atonbuf.rectifier.xSPI2.raw"
#define APP_AI_OBB_XSPI2_MODEL_IMAGE_PATH \
	"packages/prod_model_obb_compact_int8/st_ai_output/prod_model_obb_compact_int8_atonbuf.xSPI2.raw"
#define APP_AI_XSPI2_MODEL_IMAGE_PATH APP_AI_CENTER_DETECTOR_XSPI2_MODEL_IMAGE_PATH
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
#define APP_AI_OBB_INFERENCE_TIMEOUT_MS 45000U
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
/* Compact OBB regressor quantization parameters from the generated ST pack. */
#define APP_AI_OBB_OUTPUT_SCALE 0.00390625f
#define APP_AI_OBB_OUTPUT_ZERO_POINT (-128)
#define APP_AI_OBB_OUTPUT_CHANNELS 6U
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
 * 80% of the frame in either axis ??? that's almost certainly a runaway
 * prediction and we should fall back to the static training crop instead of
 * shifting the scalar's framing into the bezel/background. Only used when
 * APP_AI_RECTIFIER_FIXED_SCALE_CROP is enabled. */
#define APP_AI_RECTIFIER_CENTER_MIN_RATIO 0.10f
#define APP_AI_RECTIFIER_CENTER_MAX_RATIO 0.90f
/* Use a tiny burst history so the user-facing reading can average across a
 * few frames instead of reacting to a single noisy capture. */
#define APP_AI_INFERENCE_BURST_HISTORY_SIZE 3U
/* If the scene jumps by a lot, drop the burst history and re-lock quickly to
 * the new setpoint instead of blending two different gauge positions. */
#define APP_AI_INFERENCE_BURST_RESET_DELTA_C 12.0f
/* Reject scalar outputs that are finite but physically impossible for this
 * gauge. This avoids propagating corrupted tensor reads into logs/control. */
#define APP_AI_INFERENCE_VALUE_MIN_C (-80.0f)
#define APP_AI_INFERENCE_VALUE_MAX_C (180.0f)
/* Scalar vote-logit decode settings.
 * Keep these aligned with the training/eval decode path:
 *   mode=topk_expectation, topk=8, temperature=1.0.
 * The current gauge span in gauge_calibration_parameters.toml is [-30, 50] C.
 */
#define APP_AI_SCALAR_DECODE_TOPK 8U
#define APP_AI_SCALAR_DECODE_TEMPERATURE 1.0f
#define APP_AI_SCALAR_DECODE_VALUE_MIN_C (-30.0f)
#define APP_AI_SCALAR_DECODE_VALUE_MAX_C (50.0f)
/* xSPI2 window base address (chip address 0). */
#define APP_AI_XSPI2_CHIP_BASE_ADDR 0x70000000UL
/* Scalar model: immediately after FSBL (0x70000000) + App (0x70100000, 1 MB
 * window). Must match FLASH_SCALAR address in flash_boot.bat.
 * Size: ~3.07 MB ??? occupies 0x70200000???0x7051FFFF (50 ?? 64 KB blocks). */
#define APP_AI_XSPI2_SCALAR_BASE_ADDR 0x70200000UL
#define APP_AI_XSPI2_SCALAR_CHIP_OFFSET (APP_AI_XSPI2_SCALAR_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)
/* Rectifier model: immediately after scalar region (aligned to next 64 KB).
 * Must match FLASH_RECTIFIER address in flash_boot.bat.
 * Size: ~118 KB ??? occupies 0x70600000???0x7053FFFF (2 ?? 64 KB blocks). */
#define APP_AI_XSPI2_RECTIFIER_BASE_ADDR 0x70600000UL
#define APP_AI_XSPI2_RECTIFIER_CHIP_OFFSET (APP_AI_XSPI2_RECTIFIER_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)
#define APP_AI_XSPI2_OBB_BASE_ADDR 0x70700000UL
#define APP_AI_XSPI2_OBB_CHIP_OFFSET (APP_AI_XSPI2_OBB_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)
/* Center detector model: xSPI2 staging slot at 0x70200000.
 * The runtime copies the staged initializer blob into AXISRAM2 before init. */
#define APP_AI_XSPI2_CENTER_DETECTOR_BASE_ADDR APP_AI_XSPI2_SCALAR_BASE_ADDR
#define APP_AI_XSPI2_CENTER_DETECTOR_CHIP_OFFSET (APP_AI_XSPI2_CENTER_DETECTOR_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)
/* Legacy alias used by the single-stage logging helpers; points to center detector. */
#define APP_AI_XSPI2_MODEL_BASE_ADDR APP_AI_XSPI2_CENTER_DETECTOR_BASE_ADDR
#define APP_AI_XSPI2_MODEL_CHIP_OFFSET APP_AI_XSPI2_CENTER_DETECTOR_CHIP_OFFSET
/* FileX can take a while to recover from card init retries or media errors.
 * Give the loader a longer window so we do not give up just before the stack
 * settles. */
#define APP_AI_FILEX_MEDIA_READY_TIMEOUT_MS 180000U
/* Tip-focus geometry heatmap calibration and guardrail constants.
 * Maps center->tip angle to temperature via robust linear regression. */
#define APP_AI_TIP_FOCUS_COLD_ANGLE_DEG     135.0f
#define APP_AI_TIP_FOCUS_SLOPE              0.3119f
#define APP_AI_TIP_FOCUS_INTERCEPT          (-33.14f)
#define APP_AI_TIP_FOCUS_SWEEP_DEG          270.0f
#define APP_AI_TIP_FOCUS_HEATMAP_SIZE       112U
#define APP_AI_TIP_FOCUS_SOFTARGMAX_WINDOW  3U
#define APP_AI_TIP_FOCUS_CONFIDENCE_FLOOR   0.40f
#define APP_AI_TIP_FOCUS_TEMP_MIN_C         (-35.0f)
#define APP_AI_TIP_FOCUS_TEMP_MAX_C         55.0f
/* Median smoothing ring buffer for published tip-focus temperature. */
#define APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE 3U
#define APP_AI_TIP_FOCUS_MAX_OUTLIER_DELTA_C  5.0f
#define APP_AI_TIP_FOCUS_OUTLIER_RESET_STREAK 3U
#define APP_AI_TIP_FOCUS_MAX_INVALID_FRAMES   10U
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
static bool app_ai_runtime_initialized = false;
static volatile float app_ai_last_inference_value = 0.0f;
static volatile bool app_ai_last_inference_valid = false;
/* EMA-smoothed OBB centre to dampen single-frame x-center jitter. */
static float app_ai_smoothed_obb_cx = -1.0f;
static float app_ai_smoothed_obb_cy = -1.0f;
#define APP_AI_OBB_CENTER_EMA_ALPHA 0.20f
#if APP_AI_ENABLE_INFERENCE_BURST_SMOOTHING
static float app_ai_inference_burst_history
	[APP_AI_INFERENCE_BURST_HISTORY_SIZE] = {0.0f};
static size_t app_ai_inference_burst_history_count = 0U;
static size_t app_ai_inference_burst_history_next_index = 0U;
#endif
static bool app_ai_npu_hw_initialized = false;
static bool app_ai_xspi2_initialized = false;
static bool app_ai_xspi2_mm_enabled = false;
static const struct AppAI_ModelStageSpec *app_ai_loaded_xspi2_stage = NULL;
static float app_ai_tip_focus_median_buffer[APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE] = {0.0f};
static size_t app_ai_tip_focus_median_count = 0U;
static size_t app_ai_tip_focus_median_index = 0U;
static float app_ai_tip_focus_last_published = 0.0f;
static bool app_ai_tip_focus_last_published_valid = false;
static uint32_t app_ai_tip_focus_consecutive_invalid = 0U;
static uint32_t app_ai_tip_focus_outlier_streak = 0U;
static bool app_ai_forced_crop_active = false;
static size_t app_ai_forced_crop_x_min = 0U;
static size_t app_ai_forced_crop_y_min = 0U;
static size_t app_ai_forced_crop_width = 0U;
static size_t app_ai_forced_crop_height = 0U;
static const char *app_ai_forced_crop_label = NULL;
/* Scalar model pool: 32-byte placeholder at 0x70200000 (EXTRAM).
 * The weight blob is pre-flashed to xSPI2 at this address; the NPU reads
 * weights directly from flash, not through this array. */
__attribute__((section(".xspi2_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI2_scalar_full_finetune_from_best_piecewise_calibrated_int8[32U] = {
	0U,
};
/* Rectifier pool placed in its own section so the linker script can map it to
 * the rectifier flash region at 0x70600000 ??? matching FLASH_RECTIFIER in
 * flash_boot.bat.  The NPU resolves all weight addresses as:
 *   _mem_pool_xSPI2_mobilenetv2_rectifier_hardcase_finetune + internal_offset
 * so this symbol MUST live at the base of the flashed blob. */
__attribute__((section(".xspi2_rectifier_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI2_mobilenetv2_rectifier_hardcase_finetune[32U] = {
	0U,
};
/* OBB localizer pool lives in its own section so the linker can map it to the
 * dedicated compact OBB flash slot without disturbing the rectifier blob. */
__attribute__((section(".xspi2_obb_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI2_prod_model_obb_compact_int8[32U] = {
	0U,
};


/* Source-crop-box model pool (mobilenetv2_source_crop_box_v1_stripped_int8). */

__attribute__((section(".xspi2_source_crop_box_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))

uint8_t _mem_pool_xSPI2_mobilenetv2_source_crop_box_v1_stripped_int8[32U] = {

	0U,

};

/* Tip-focus geometry model pool.  The generated network (via
 * LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(Default)) references
 * _mem_pool_xSPI2_Default, and this 32-byte placeholder keeps that xSPI2
 * symbol alive at the address where the NPU weight blob
 * (network_atonbuf.xSPI2.raw) is flashed (0x70400000).
 *
 * IMPORTANT: The actual weights data (2.2MB) is NOT stored in this array.
 * The data lives in xSPI2 flash at 0x70400000, and must be flashed using
 * flash_boot.bat before running inference. This 32-byte symbol is just a
 * linker marker that gets placed at 0x70400000 by the .xspi2_tip_focus_pool
 * section. When the NPU accesses weights, it reads from xSPI2 flash through
 * the memory-mapped window (0x70000000+), not from this RAM array.
 *
 * If you see a HardFault at address 0x8D or similar during inference, it
 * means the xSPI2 flash was not programmed. Run flash_boot.bat to flash
 * network_atonbuf.xSPI2.raw to 0x70400000. */
__attribute__((section(".xspi2_tip_focus_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI2_Default[32U] = { 0U, };
static uint8_t app_ai_xspi2_program_buffer[APP_AI_XSPI2_PROGRAM_CHUNK_BYTES];
__attribute__((aligned(APP_AI_CACHE_LINE_BYTES)))
static uint8_t app_ai_scalar_row_scratch[APP_AI_CAPTURE_FRAME_WIDTH_PIXELS * APP_AI_CAPTURE_FRAME_BYTES_PER_PIXEL];
__attribute__((aligned(APP_AI_CACHE_LINE_BYTES)))
static uint8_t app_ai_scalar_output_row_scratch[APP_AI_CAPTURE_FRAME_WIDTH_PIXELS * 3U * sizeof(float)];
volatile size_t app_ai_scalar_preprocess_last_row = (size_t)SIZE_MAX;
/* Trace the scalar resize loop only every so often so we can tell whether it
 * is progressing without flooding UART in the hot path. */
#define APP_AI_SCALAR_PREPROCESS_ROW_TRACE_INTERVAL_ROWS 32U
/* Keep each preprocessing pass small enough that we can reshape the scalar
 * path without one huge monolithic row loop. */
#define APP_AI_SCALAR_PREPROCESS_ROWS_PER_CHUNK 8U
/* Start/tail signatures for the heatmap center-detector initializer blob.
 * Update these when a new model is exported by running:
 *   python3 -c "
 *     d=open('st_ai_output/packages/heatmap_cd_tiny/st_ai_output/heatmap_cd_atonbuf.AXISRAM2.raw','rb').read()
 *     print('start:', bytes(d[:16]).hex())
 *     print('tail: ', bytes(d[-16:]).hex())" */
/* Heatmap center-detector model: package raw (69,457 bytes) flashed at 0x70200000. */
static const uint8_t app_ai_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES] = {
	0xB5U, 0x3FU, 0x43U, 0x0AU, 0x50U, 0xD3U, 0x0AU, 0x08U,
	0xBCU, 0x1BU, 0xB6U, 0xB1U, 0xD6U, 0xDDU, 0x0FU, 0x0CU,
};
static const uint8_t app_ai_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x90U,
};
/* Rectified scalar v2 xSPI2 signatures used when the board boots with the
 * prod v0.8 scalar blob already flashed at 0x70200000. Size: 3,218,865 bytes. */
static const uint8_t app_ai_rectifier_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES] = {
	0x0FU, 0x11U, 0xF8U, 0x10U, 0xD0U, 0xD8U, 0x0EU, 0x28U,
	0x99U, 0xCEU, 0x98U, 0x7DU, 0xBCU, 0x42U, 0x5EU, 0xF2U,
};
static const uint8_t app_ai_rectifier_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x80U,
};
/* Compact OBB regressor xSPI2 signatures.
 * The board verifies against these bytes when the SD-side cache is not
 * populated, so keep them aligned with the package raw file (117,505 bytes). */
static const uint8_t app_ai_obb_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES] = {
	0xB6U, 0xA6U, 0xD3U, 0xE8U, 0xEAU, 0xD9U, 0xCEU, 0x27U,
	0xE1U, 0xDEU, 0xF8U, 0x0AU, 0xE9U, 0xF4U, 0x2FU, 0x29U,
};
static const uint8_t app_ai_obb_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x7FU,
};

/* Source-crop-box xSPI2 signatures for atonbuf.source_crop_box.xSPI2.raw. */
static const uint8_t app_ai_source_crop_box_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES] = {
	0xF2U, 0x17U, 0x29U, 0xE2U, 0xDCU, 0xEBU, 0xECU, 0x04U,
	0x09U, 0x01U, 0x35U, 0xEBU, 0x14U, 0xDEU, 0x0FU, 0x02U,
};
static const uint8_t app_ai_source_crop_box_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x80U,
};
/* Tip-focus geometry model xSPI2 signatures for network_atonbuf.xSPI2.raw.
 * Flashed to 0x70400000. Size: 2,201,505 bytes. */
static const uint8_t app_ai_tip_focus_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES] = {
	0x04U, 0x2FU, 0x1FU, 0xF2U, 0x62U, 0xE7U, 0x3EU, 0xFDU,
	0x0AU, 0x1EU, 0xF4U, 0x32U, 0xD4U, 0x9AU, 0xFEU, 0xC2U,
};
static const uint8_t app_ai_tip_focus_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x80U,
};
/* Per-stage programmed sizes. Set during provisioning and used by the verify
 * functions for the tail probe offset. Keeping them separate prevents the
 * scalar tail check from using the rectifier's file size (or vice-versa) when
 * stages alternate. */
static ULONG app_ai_scalar_programmed_size = 0UL;
static ULONG app_ai_rectifier_programmed_size = 0UL;
static ULONG app_ai_obb_programmed_size = 0UL;
static ULONG app_ai_source_crop_box_programmed_size = 0UL;
static ULONG app_ai_tip_focus_programmed_size = 0UL;
/* Legacy alias kept so existing references still compile; points to the scalar
 * size which was the only stage before the rectifier was added. */
static ULONG app_ai_xspi2_programmed_size = 0UL;

/* Per-stage signature caches populated from the SD file during provisioning.
 * Using SD-sourced bytes means verify never goes stale when the model blob is
 * replaced, regardless of what the hardcoded fallback constants say. */
static uint8_t app_ai_scalar_sig_start[APP_AI_XSPI2_PROBE_BYTES] = {0U};
static uint8_t app_ai_scalar_sig_tail[APP_AI_XSPI2_PROBE_BYTES] = {0U};
static bool app_ai_scalar_sig_valid = false;
static uint8_t app_ai_rectifier_sig_start[APP_AI_XSPI2_PROBE_BYTES] = {0U};
static uint8_t app_ai_rectifier_sig_tail[APP_AI_XSPI2_PROBE_BYTES] = {0U};
static bool app_ai_rectifier_sig_valid = false;
static uint8_t app_ai_obb_sig_start[APP_AI_XSPI2_PROBE_BYTES] = {0U};
static uint8_t app_ai_obb_sig_tail[APP_AI_XSPI2_PROBE_BYTES] = {0U};
static bool app_ai_obb_sig_valid = false;
static uint8_t app_ai_source_crop_box_sig_start[APP_AI_XSPI2_PROBE_BYTES] = {0U};
static uint8_t app_ai_source_crop_box_sig_tail[APP_AI_XSPI2_PROBE_BYTES] = {0U};
static bool app_ai_source_crop_box_sig_valid = false;
static uint8_t app_ai_tip_focus_sig_start[APP_AI_XSPI2_PROBE_BYTES] = {0U};
static uint8_t app_ai_tip_focus_sig_tail[APP_AI_XSPI2_PROBE_BYTES] = {0U};
static bool app_ai_tip_focus_sig_valid = false;

/* Declare the generated NN instance locally so the dry-run helper can run the
 * AtoNN runtime on the exact network produced by Cube.AI. */
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(
	scalar_full_finetune_from_best_piecewise_calibrated_int8);
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(
	mobilenetv2_rectifier_hardcase_finetune);
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(
	prod_model_obb_compact_int8);
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(
	mobilenetv2_source_crop_box_v1_stripped_int8);
/* Heatmap center detector replaces the scalar CNN as the sole inference
 * authority. Its weight blob lives at 0x70200000 (the old scalar slot). */
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(
	heatmap_cd);

typedef struct AppAI_ModelStageSpec AppAI_ModelStageSpec;

struct AppAI_ModelStageSpec
{
	const char *stage_label;
	const char *model_image_path;
	NN_Instance_TypeDef *nn_instance;
	bool (*network_init_fn)(void);
	bool (*inference_init_fn)(void);
	bool uses_rectifier_box;
	uint32_t xspi2_chip_offset; /* byte offset from chip base (0x70000000) */
	uint32_t xspi2_base_addr;	/* mapped window address for this stage */
};

typedef struct
{
	float center_x;
	float center_y;
	float box_w;
	float box_h;
} AppAI_RectifierBox;

typedef struct
{
	float center_x;
	float center_y;
	float box_w;
	float box_h;
	float angle_rad;
	float confidence;
} AppAI_ObbBox;

typedef struct
{
	size_t x_min;
	size_t y_min;
	size_t width;
	size_t height;
} AppAI_SourceCrop;

static const AppAI_ModelStageSpec app_ai_obb_stage = {
	.stage_label = "obb",
	.model_image_path = APP_AI_OBB_XSPI2_MODEL_IMAGE_PATH,
	.nn_instance = &NN_Instance_prod_model_obb_compact_int8,
	.network_init_fn = LL_ATON_EC_Network_Init_prod_model_obb_compact_int8,
	.inference_init_fn = LL_ATON_EC_Inference_Init_prod_model_obb_compact_int8,
	.uses_rectifier_box = false,
	.xspi2_chip_offset = APP_AI_XSPI2_OBB_CHIP_OFFSET,
	.xspi2_base_addr = APP_AI_XSPI2_OBB_BASE_ADDR,
};

static bool AppAI_ShouldLogStageDiagnostics(
	const AppAI_ModelStageSpec *stage)
{
	(void)stage;
	return true;
}

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/

/* Split helper implementation into a companion include to keep this file manageable. */
#include "app_ai_helpers.inc"

/* Split out the runtime tail into a companion include to keep this file manageable. */
#include "app_ai_runtime_tail.inc"
