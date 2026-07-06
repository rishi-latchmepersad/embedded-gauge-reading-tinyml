/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_ai.c
 * @brief   Minimal AI runtime bootstrap helpers.
 ******************************************************************************
 */
/* USER CODE END Header */

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
#include "app_baseline_runtime.h"
#include "app_inference_log_utils.h"
#include "app_memory_budget.h"
#include "app_gauge_geometry.h"
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
/* The OBB stage is now fallback-only. The live board inference path routes
 * through the tip-focus SimCC coordinate model first, but we keep the old
 * crop front-end behind a switch so it can still be re-enabled for debug. */
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
/* Scalar model image path (deprecated â€” retained for the scalar stage spec). */
#define APP_AI_SCALAR_XSPI2_MODEL_IMAGE_PATH \
	"packages/mobilenetv2_rectified_scalar_finetune_v2/st_ai_output/scalar_full_finetune_from_best_piecewise_calibrated_int8_atonbuf.xSPI2.raw"
#define APP_AI_CENTER_DETECTOR_XSPI2_MODEL_IMAGE_PATH \
	"packages/heatmap_cd_v4s_80/st_ai_output/heatmap_cd_atonbuf.xSPI2.raw" /* DS-CNN v4-S 80×80, ~268 KB */
#define APP_AI_RECTIFIER_XSPI2_MODEL_IMAGE_PATH \
	"atonbuf.rectifier.xSPI2.raw"
#define APP_AI_OBB_XSPI2_MODEL_IMAGE_PATH \
	"packages/obb_face_v2_int8_n6_npu/st_ai_output/obb_face_v2_int8_atonbuf.xSPI2.raw" /* OBB face localizer, ~571 KiB */
#define APP_AI_XSPI2_MODEL_IMAGE_PATH APP_AI_CENTER_DETECTOR_XSPI2_MODEL_IMAGE_PATH
#endif
#define APP_AI_XSPI2_PROGRAM_CHUNK_BYTES 4096U
#define APP_AI_XSPI2_ERASE_BLOCK_BYTES (64U * 1024U)
#define APP_AI_XSPI2_PROBE_BYTES 16U
/* The remaining crop and decode constants only matter to the legacy
 * scalar/OBB/center-detector fallback path. */
#if !APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
/* Keep the rectifier crop slightly larger than the raw box so the scalar head
 * still sees the needle and a bit of surrounding dial context. */
#define APP_AI_RECTIFIER_CROP_SCALE 1.80f
/* Prod v0.8 offline freeze used a 1.20x OBB crop scale before the luma
 * refinement stage constrained the final crop window. */
#define APP_AI_OBB_CROP_SCALE 1.20f
/* Keep the OBB localizer bounded, but give it enough time to finish on the
 * 60 s capture cadence. The earlier 10 s cap forced a fallback before the
 * deployed localizer could converge on harder frames. */
#define APP_AI_OBB_INFERENCE_TIMEOUT_MS 15000U  /* Scaled OBB is a single NPU pass, much faster than YOLO */
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
/* Legacy OBB experiment constants kept for older replay notes and comparison.
 * The live face-localizer path below now uses the 2-output box/conf contract. */
#define APP_AI_OBB_OUTPUT_SCALE      0.003921569f      /* input scale (unused for multi-output) */
#define APP_AI_OBB_OUTPUT_ZERO_POINT (-128)            /* input zp (unused for multi-output) */
#define APP_AI_OBB_HEATMAP_SCALE     0.047995351f      /* Transpose_50_out_0 (α=1.25) */
#define APP_AI_OBB_HEATMAP_ZP        14
#define APP_AI_OBB_BOX_SCALE         0.0078125f        /* Transpose_59_out_0 */
#define APP_AI_OBB_BOX_ZP            0
#define APP_AI_OBB_ANGLE_SCALE       0.042166740f      /* Transpose_53_out_0 (α=1.25) */
#define APP_AI_OBB_ANGLE_ZP          1
#define APP_AI_OBB_OUTPUT_CHANNELS   3U  /* 3 tensors decoded by AppAI_DecodeQarepvggOutput */
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
 * 80% of the frame in either axis ??? that's almost certainly a runaway
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
#endif
/* xSPI2 window base address (chip address 0). */
#define APP_AI_XSPI2_CHIP_BASE_ADDR 0x70000000UL
/* Tip-focus SimCC model: xSPI2 weights at 0x70400000. */
#define APP_AI_XSPI2_TIP_FOCUS_BASE_ADDR 0x70400000UL
#define APP_AI_XSPI2_TIP_FOCUS_CHIP_OFFSET (APP_AI_XSPI2_TIP_FOCUS_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)
/* Legacy aliases used by the shared xSPI2 helpers now point at the live
 * tip-focus slot so the generic probe/logging code matches the active model. */
#define APP_AI_XSPI2_MODEL_BASE_ADDR APP_AI_XSPI2_TIP_FOCUS_BASE_ADDR
#define APP_AI_XSPI2_MODEL_CHIP_OFFSET APP_AI_XSPI2_TIP_FOCUS_CHIP_OFFSET
/* Shared runtime types are used by both the live tip-focus path and the
 * compile-guarded legacy fallback helpers, so keep them available in both
 * build modes. */
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
	float confidence;
	float angle_rad;
	/* Gauge-center prediction used by downstream crop logic (normalised [0,1]).
	 * The face-localizer mirrors the box center here; keep -1.0 for invalid. */
	float gauge_center_x;
	float gauge_center_y;
} AppAI_ObbBox;

typedef struct
{
	size_t x_min;
	size_t y_min;
	size_t width;
	size_t height;
} AppAI_SourceCrop;

#if !APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
/* Scalar model: immediately after FSBL (0x70000000) + App (0x70100000, 1 MB
 * window). Must match FLASH_SCALAR address in flash_boot.ps1.
 * Size: ~3.07 MB ??? occupies 0x70200000???0x7051FFFF (50 ?? 64 KB blocks). */
#define APP_AI_XSPI2_SCALAR_BASE_ADDR 0x70200000UL
#define APP_AI_XSPI2_SCALAR_CHIP_OFFSET (APP_AI_XSPI2_SCALAR_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)
/* Rectifier model: immediately after scalar region (aligned to next 64 KB).
 * Must match FLASH_RECTIFIER address in flash_boot.ps1.
 * Size: ~118 KB ??? occupies 0x70600000???0x7053FFFF (2 ?? 64 KB blocks). */
#define APP_AI_XSPI2_RECTIFIER_BASE_ADDR 0x70600000UL
#define APP_AI_XSPI2_RECTIFIER_CHIP_OFFSET (APP_AI_XSPI2_RECTIFIER_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)
#define APP_AI_XSPI2_OBB_BASE_ADDR 0x70700000UL
#define APP_AI_XSPI2_OBB_CHIP_OFFSET (APP_AI_XSPI2_OBB_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)
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
/* Tip-focus SimCC guardrail constants.
 * The runtime uses the replay angle convention (negated image Y) so the
 * board-side angle decode stays aligned with the packaged hard-case run.
 * Temperature mapping itself is shared through AppBaselineRuntime_. */
#define APP_AI_TIP_FOCUS_MODEL_INPUT_WIDTH_PIXELS 224U
#define APP_AI_TIP_FOCUS_MODEL_INPUT_HEIGHT_PIXELS 224U
#define APP_AI_TIP_FOCUS_COLD_ANGLE_DEG     135.0f
#define APP_AI_TIP_FOCUS_SWEEP_DEG          270.0f
#define APP_AI_TIP_FOCUS_SIMCC_BINS         112U
#define APP_AI_TIP_FOCUS_CONFIDENCE_FLOOR   0.40f
/* The live center SimCC head has been landing just under 0.08 on otherwise
 * good captures, so keep a small margin below that until we retrain or
 * re-tune the head. */
#define APP_AI_TIP_FOCUS_AXIS_PEAK_FLOOR    0.06f
#define APP_AI_TIP_FOCUS_AXIS_SPREAD_MAX_PX  32.0f
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
#if !APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
/* Scalar model pool: 32-byte placeholder at 0x70200000 (EXTRAM).
 * The weight blob is pre-flashed to xSPI2 at this address; the NPU reads
 * weights directly from flash, not through this array. */
__attribute__((section(".xspi2_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI2_scalar_full_finetune_from_best_piecewise_calibrated_int8[32U] = {
	0U,
};
/* Heatmap CD model xSPI2 pool: same slot (0x70200000) as the scalar.
 * DS-CNN v4-S weights are pre-flashed here; the generated heatmap_cd.c
 * references _mem_pool_xSPI2_heatmap_cd for weight addressing. */
__attribute__((section(".xspi2_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI2_heatmap_cd[32U] = {
	0U,
};
/* Compact OBB 320 localizer pool lives in the dedicated OBB flash slot.
 * Keep the linker marker aligned with the flashed blob so the generated
 * reloc runtime can resolve the weight base correctly. */
__attribute__((section(".xspi2_rectifier_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI2_mobilenetv2_rectifier_hardcase_finetune[32U] = {
	0U,
};
/* Board bbox OBB pool lives in the dedicated OBB flash slot.
 * The generated network references:
 *   _mem_pool_xSPI2_obb_face_v2_int8 — weight-addressing.
 * 32-byte placeholder ensures the symbol resolves at link time.
 * Actual weight blob (~571 KiB, obb_face_v2_int8_atonbuf.xSPI2.raw) is
 * flashed separately via flash_boot.ps1. */
__attribute__((section(".xspi2_obb_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI2_obb_face_v2_int8[32U] = {
	0U,
};

/* OBB face-localizer activation overflow (1024 KB) in xSPI1 HyperRAM.
 * The N657 Nucleo has no physical HyperRAM — this symbol resolves
 * at 0x90000000 via the .xspi1_obb_pool linker section.
 * NPU access to 0x90000000 will bus-fault at runtime. */
__attribute__((section(".xspi1_obb_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI1_obb_face_v2_int8[32U] = {
	0U,
};


/* Source-crop-box model pool (mobilenetv2_source_crop_box_v1_stripped_int8). */

__attribute__((section(".xspi2_source_crop_box_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))

uint8_t _mem_pool_xSPI2_mobilenetv2_source_crop_box_v1_stripped_int8[32U] = {

	0U,

};
#endif

/* v16_160 compact UNet needle model pool (2026-06-30).
 * Replaces the old simcc spatial tip-focus.  160x160 input, 40x40 heatmap
 * output, 525K params, 622 KB xSPI2 flash.  NO HyperRAM (xSPI1=0 bytes).
 * The linker places this 32-byte marker at 0x70400000 via .xspi2_tip_focus_pool.
 * Flash the raw blob with flash_boot.ps1 before running inference.
 * Weight blob: tip_focus_v16_160_int8_atonbuf.xSPI2.raw (621,976 bytes). */
__attribute__((section(".xspi2_tip_focus_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI2_tip_focus_v16_160_int8[32U] = { 0U, };
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
#if !APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
/* Start/tail signatures for the heatmap center-detector initializer blob.
 * Update these when a new model is exported by running:
 *   python3 -c "
 *     d=open('st_ai_output/packages/heatmap_cd_tiny/st_ai_output/heatmap_cd_atonbuf.AXISRAM2.raw','rb').read()
 *     print('start:', bytes(d[:16]).hex())
 *     print('tail: ', bytes(d[-16:]).hex())" */
/* Heatmap center-detector model (DS-CNN v4-S): 332,045 bytes flashed at 0x70200000.
 * Signatures unchanged from DS-CNN v4 — same first/last 16 bytes. */
static const uint8_t app_ai_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES] = {
	0x80U, 0x80U, 0x81U, 0x82U, 0x83U, 0x83U, 0x84U, 0x85U,
	0x86U, 0x87U, 0x87U, 0x88U, 0x89U, 0x8AU, 0x8BU, 0x8BU,
};
static const uint8_t app_ai_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x80U,
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
/* OBB face-localizer xSPI2 signatures.  Update after running the matching
 * packaging script for the deployed face-localizer blob.
 * The script prints the 16 start/tail bytes for this raw blob. */
static const uint8_t app_ai_obb_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES] = {
	0x31U, 0xDBU, 0xF1U, 0x9CU, 0xD8U, 0x26U, 0xDEU, 0x07U,
	0x45U, 0x03U, 0xEEU, 0xCEU, 0x21U, 0xD8U, 0x0FU, 0xB3U,
};
static const uint8_t app_ai_obb_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0xFDU,
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
#endif
/* Tip-focus SimCC model xSPI2 signatures for
 * network_atonbuf.xSPI2.raw.
 * Flashed to 0x70400000. Size: 2,201,505 bytes. */
static const uint8_t app_ai_tip_focus_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES] = {
	0x0DU, 0xE2U, 0x24U, 0x39U, 0xFEU, 0xEDU, 0xF1U, 0x12U,
	0xE5U, 0xDFU, 0xF5U, 0x0DU, 0x01U, 0x0CU, 0x19U, 0x1FU,
};
static const uint8_t app_ai_tip_focus_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x80U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
};
#if !APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
/* Per-stage programmed sizes. Set during provisioning and used by the verify
 * functions for the tail probe offset. Keeping them separate prevents the
 * scalar tail check from using the rectifier's file size (or vice-versa) when
 * stages alternate. */
static ULONG app_ai_scalar_programmed_size = 0UL;
static ULONG app_ai_rectifier_programmed_size = 0UL;
static ULONG app_ai_obb_programmed_size = 0UL;
static ULONG app_ai_source_crop_box_programmed_size = 0UL;

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
#endif
/* Legacy alias kept for the shared xSPI2 logging helpers; the live tip-focus
 * build still expects this size tracker to exist. */
static ULONG app_ai_xspi2_programmed_size = 0UL;
static ULONG app_ai_tip_focus_programmed_size = 0UL;
static uint8_t app_ai_tip_focus_sig_start[APP_AI_XSPI2_PROBE_BYTES] = {0U};
static uint8_t app_ai_tip_focus_sig_tail[APP_AI_XSPI2_PROBE_BYTES] = {0U};
static bool app_ai_tip_focus_sig_valid = false;

/* Declare the generated NN instance locally so the dry-run helper can run the
 * AtoNN runtime on the exact network produced by Cube.AI. */
#if !APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(
	scalar_full_finetune_from_best_piecewise_calibrated_int8);
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(
	mobilenetv2_rectifier_hardcase_finetune);
/* OBB face localizer — compact box + confidence head for gauge framing.
 * Outputs 2 int8 tensors: a 4-value box tensor and a 1-value confidence. */
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(
	obb_face_v2_int8);
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(
	mobilenetv2_source_crop_box_v1_stripped_int8);
/* Heatmap center detector replaces the scalar CNN as the sole inference
 * authority. Its weight blob lives at 0x70200000 (the old scalar slot). */
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(
	heatmap_cd);

typedef struct
{
	float center_x;
	float center_y;
	float box_w;
	float box_h;
} AppAI_RectifierBox;

static const AppAI_ModelStageSpec app_ai_obb_stage = {
	.stage_label = "obb_face_v2_int8",
	.model_image_path = APP_AI_OBB_XSPI2_MODEL_IMAGE_PATH,
	.nn_instance = &NN_Instance_obb_face_v2_int8,
	.network_init_fn = LL_ATON_EC_Network_Init_obb_face_v2_int8,
	.inference_init_fn = LL_ATON_EC_Inference_Init_obb_face_v2_int8,
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

static float AppAI_ClampNormalizedUnitFloat(float value)
{
	if (!isfinite(value))
	{
		return 0.0f;
	}
	if (value < 0.0f)
	{
		return 0.0f;
	}
	if (value > 1.0f)
	{
		return 1.0f;
	}
	return value;
}

static float AppAI_DecodeQuantizedScalar(
	const LL_Buffer_InfoTypeDef *buffer_info,
	const int8_t *raw_value_ptr)
{
	const float scale =
		(buffer_info->scale != NULL) ? buffer_info->scale[0] : 0.0f;
	const int32_t zero_point =
		(buffer_info->offset != NULL) ? (int32_t)buffer_info->offset[0] : 0;

	return (((float)(*raw_value_ptr)) - (float)zero_point) * scale;
}

/* OBB face-localizer bridge.
 * output_buffers_info array: [0]=box, [1]=confidence, with a final NULL terminator.
 * The box is [cx, cy, w, h] in normalised [0,1] coordinates. */
static bool AppAI_DecodeObbFaceV2(
	const LL_Buffer_InfoTypeDef *output_info,
	AppAI_SourceCrop            *obb_crop,
	AppAI_ObbBox                *obb_box)
{
	if ((output_info == NULL) || (obb_crop == NULL) || (obb_box == NULL))
	{
		return false;
	}

	const LL_Buffer_InfoTypeDef *box_info = NULL;
	const LL_Buffer_InfoTypeDef *conf_info = NULL;

	for (const LL_Buffer_InfoTypeDef *info = output_info; info->name != NULL; ++info)
	{
		const size_t output_len = (size_t)LL_Buffer_len(info);

		if (output_len == 4U)
		{
			if (box_info != NULL)
			{
				return false;
			}
			box_info = info;
			continue;
		}

		if (output_len == 1U)
		{
			if (conf_info != NULL)
			{
				return false;
			}
			conf_info = info;
			continue;
		}

		/* The live OBB contract should stay at one 4-value box tensor and one
		 * scalar confidence tensor. If the shape changes, fail closed. */
		return false;
	}

	if ((box_info == NULL) || (conf_info == NULL))
	{
		return false;
	}

	if ((box_info->scale == NULL) || (box_info->offset == NULL) ||
		(conf_info->scale == NULL) || (conf_info->offset == NULL))
	{
		return false;
	}

	const int8_t *box_raw = (const int8_t *)LL_Buffer_addr_start(box_info);
	const int8_t *conf_raw = (const int8_t *)LL_Buffer_addr_start(conf_info);
	if ((box_raw == NULL) || (conf_raw == NULL))
	{
		return false;
	}

	float center_x = AppAI_DecodeQuantizedScalar(box_info, &box_raw[0]);
	float center_y = AppAI_DecodeQuantizedScalar(box_info, &box_raw[1]);
	float box_w = AppAI_DecodeQuantizedScalar(box_info, &box_raw[2]);
	float box_h = AppAI_DecodeQuantizedScalar(box_info, &box_raw[3]);
	float confidence = AppAI_DecodeQuantizedScalar(conf_info, &conf_raw[0]);

	if (!isfinite(center_x) || !isfinite(center_y) ||
		!isfinite(box_w) || !isfinite(box_h) ||
		!isfinite(confidence))
	{
		return false;
	}

	center_x = AppAI_ClampNormalizedUnitFloat(center_x);
	center_y = AppAI_ClampNormalizedUnitFloat(center_y);
	box_w = AppAI_ClampNormalizedUnitFloat(box_w);
	box_h = AppAI_ClampNormalizedUnitFloat(box_h);
	confidence = AppAI_ClampNormalizedUnitFloat(confidence);

	if (box_w < APP_AI_OBB_MIN_BOX_RATIO)
	{
		box_w = APP_AI_OBB_MIN_BOX_RATIO;
	}
	if (box_h < APP_AI_OBB_MIN_BOX_RATIO)
	{
		box_h = APP_AI_OBB_MIN_BOX_RATIO;
	}

	obb_box->center_x = center_x;
	obb_box->center_y = center_y;
	obb_box->box_w = box_w;
	obb_box->box_h = box_h;
	obb_box->confidence = confidence;
	obb_box->angle_rad = 0.0f;
	obb_box->gauge_center_x = center_x;
	obb_box->gauge_center_y = center_y;

	/* Populate pixel crop from centre + box size, with OBB crop scale. */
	float cx_px = center_x * (float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	float cy_px = center_y * (float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	float bw_px = box_w    * (float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	float bh_px = box_h    * (float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;

	float crop_w = bw_px * APP_AI_OBB_CROP_SCALE;
	float crop_h = bh_px * APP_AI_OBB_CROP_SCALE;

	/* Clamp crop to frame bounds. */
	float x_min_f = cx_px - crop_w * 0.5f;
	float y_min_f = cy_px - crop_h * 0.5f;

	if (x_min_f < 0.0f) { x_min_f = 0.0f; }
	if (y_min_f < 0.0f) { y_min_f = 0.0f; }
	if ((x_min_f + crop_w) > (float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS)
	{
		crop_w = (float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS - x_min_f;
	}
	if ((y_min_f + crop_h) > (float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS)
	{
		crop_h = (float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS - y_min_f;
	}
	if ((crop_w < APP_AI_OBB_MIN_CROP_SIZE_PIXELS) || (crop_h < APP_AI_OBB_MIN_CROP_SIZE_PIXELS))
	{
		return false;   /* degenerate crop — reject */
	}

	obb_crop->x_min  = (size_t)(x_min_f + 0.5f);
	obb_crop->y_min  = (size_t)(y_min_f + 0.5f);
	obb_crop->width  = (size_t)(crop_w + 0.5f);
	obb_crop->height = (size_t)(crop_h + 0.5f);

	return true;
}
#endif /* !APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE */

/**
 * @brief Return the relocatable runtime base expected in r9 for a network.
 */
uintptr_t AppAI_GetRelocRuntimeR9(const NN_Instance_TypeDef *nn_instance)
{
	if (nn_instance == &NN_Instance_obb_box_board_bbox_deploy_candidate)
	{
		/* The OBB package is flashed as a fixed XIP image, but the generated
		 * epoch blocks still expect a stable runtime base in r9 for their
		 * internal SW helpers. When the reloc handle is intentionally cleared
		 * during init, fall back to the model's fixed runtime window instead of
		 * inheriting whatever junk r9 happened to contain on entry. */
		return (uintptr_t)APP_AI_OBB_RELOC_RAM_BASE_ADDR;
	}

	if ((nn_instance != NULL) && (nn_instance->exec_state.inst_reloc != 0U))
	{
		const struct ai_reloc_rt_ctx *rt_ctx =
			(const struct ai_reloc_rt_ctx *)(uintptr_t)nn_instance->exec_state.inst_reloc;
		if ((rt_ctx != NULL) && (rt_ctx->ram_addr != 0U))
		{
			return (uintptr_t)rt_ctx->ram_addr;
		}
	}

/* Private function prototypes -----------------------------------------------*/

/* Split helper implementation into a companion include to keep this file manageable. */
#include "app_ai_helpers.inc"

/* Split out the runtime tail into a companion include to keep this file manageable. */
#include "app_ai_runtime_tail.inc"
