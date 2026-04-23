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
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#include "debug_console.h"
#include "app_inference_calibration.h"
#include "app_inference_log_utils.h"
#include "app_gauge_geometry.h"
#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX
#include "tx_api.h"
#include "ll_aton_rt_user_api.h"
#include "ll_aton.h"
#include "ll_aton_runtime.h"
#include "app_filex.h"
#include "stm32n6xx_nucleo_xspi.h"
#include "npu_cache.h"
#include "stm32n6xx_hal.h"

/*
 * Keep the very noisy tensor and patch dumps behind a toggle so the normal
 * AI status logs still use the shared debug console directly.
 */
#ifndef APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
#define APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS 0
#endif
/* Keep rectifier diagnostics available even when the rest of the verbose
 * console logging stays off. This is a temporary bring-up aid for crop
 * debugging and can be flipped back to 0 once the rectifier is stable. */
#ifndef APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
#define APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS 1U
#endif
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define APP_AI_CACHE_LINE_BYTES 32U
#define APP_AI_CAPTURE_FRAME_WIDTH_PIXELS 224U
#define APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS 224U
#define APP_AI_CAPTURE_FRAME_BYTES_PER_PIXEL 2U
#define APP_AI_CAPTURE_FRAME_BYTES \
		(APP_AI_CAPTURE_FRAME_WIDTH_PIXELS * APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS \
				* APP_AI_CAPTURE_FRAME_BYTES_PER_PIXEL)
#define APP_AI_MODEL_INPUT_FLOAT_COUNT \
		(APP_AI_CAPTURE_FRAME_WIDTH_PIXELS * APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS \
				* 3U)
/* Bright-object detector threshold used to find the gauge face before
 * cropping and resizing the tensor. 80 was a better fit than 60 on captured
 * frames because 60 was too loose and pulled in most of the background. */
#define APP_AI_GAUGE_BRIGHT_THRESHOLD      80U
/* Ignore a thin border while estimating the bright bbox so edge glare does not
 * drag the crop to x=0 or y=0. */
#define APP_AI_GAUGE_CROP_BORDER_PIXELS    16U
/* The newest live board captures want a slightly tighter crop that is biased
 * left of the bright centroid. That keeps more of the dial face in frame and
 * reduces the background washout that was pulling the prod model low. */
#define APP_AI_GAUGE_CROP_WIDTH_SCALE_NUMERATOR    17U
#define APP_AI_GAUGE_CROP_WIDTH_SCALE_DENOMINATOR  20U
#define APP_AI_GAUGE_CROP_HEIGHT_SCALE_NUMERATOR   17U
#define APP_AI_GAUGE_CROP_HEIGHT_SCALE_DENOMINATOR 20U
#define APP_AI_GAUGE_CROP_CENTER_X_BIAS_PIXELS      24U
#define APP_AI_GAUGE_CROP_CENTER_Y_BIAS_PIXELS       0U
/* The current close-up board captures drift too far downward with the
 * adaptive rectifier and miss the needle on low temperatures. Use the stable
 * training crop on-device until we retrain on the closer framing. */
#define APP_AI_USE_ADAPTIVE_GAUGE_CROP      0U
/* Feed the model a stable grayscale tensor from Y luma only.
 * The RGB reconstruction path was collapsing to a green-only tensor on board,
 * which is a worse mismatch than replicated luminance for this gauge task. */
#define APP_AI_YUV422_INPUT_LUMA_ONLY      1U
#define APP_AI_MODEL_INPUT_FLOAT_BYTES \
		(APP_AI_MODEL_INPUT_FLOAT_COUNT * sizeof(float))
#define APP_AI_MODEL_OUTPUT_FLOAT_BYTES   sizeof(float)
#define APP_AI_SCALAR_XSPI2_MODEL_IMAGE_PATH "atonbuf.xSPI2.raw"
#define APP_AI_RECTIFIER_XSPI2_MODEL_IMAGE_PATH \
		"atonbuf.rectifier.xSPI2.raw"
#define APP_AI_OBB_XSPI2_MODEL_IMAGE_PATH   "atonbuf.obb.xSPI2.raw"
#define APP_AI_XSPI2_MODEL_IMAGE_PATH     APP_AI_SCALAR_XSPI2_MODEL_IMAGE_PATH
#define APP_AI_XSPI2_PROGRAM_CHUNK_BYTES   4096U
#define APP_AI_XSPI2_ERASE_BLOCK_BYTES     (64U * 1024U)
#define APP_AI_XSPI2_PROBE_BYTES           16U
/* Keep the rectifier crop slightly larger than the raw box so the scalar head
 * still sees the needle and a bit of surrounding dial context. */
#define APP_AI_RECTIFIER_CROP_SCALE        1.80f
/* The OBB path only needs a small safety margin around the detected box. */
#define APP_AI_OBB_CROP_SCALE              1.20f
/* Match the Python rectifier evaluator: never let the predicted box collapse
 * below a tiny fraction of the canvas, or the scalar stage degenerates to a
 * 1x1 crop. */
#define APP_AI_RECTIFIER_MIN_BOX_RATIO     0.05f
/* Keep the OBB decoder from collapsing into a tiny crop when the localizer
 * gets uncertain. */
#define APP_AI_OBB_MIN_BOX_RATIO           0.05f
/* The OBB crop should still be a real crop, not a 1x1 or 8x8 window. */
#define APP_AI_OBB_MIN_CROP_SIZE_PIXELS    48.0f
/* Accept any box where both w and h are in [0.2, 1.5] of the frame dimension.
 * The rectifier outputs normalised cx/cy/w/h so these limits reject degenerate
 * (near-zero) or wildly oversized predictions while passing plausible dial
 * crops at close-up and normal camera distances. */
#define APP_AI_RECTIFIER_FALLBACK_MIN_BOX_RATIO  0.2f
#define APP_AI_RECTIFIER_FALLBACK_MAX_BOX_RATIO  1.5f
/* Path-1 soft-attention mode: ignore rectifier's predicted (w,h), use prod's
 * fixed training-crop dimensions centred on rectifier's predicted (cx,cy).
 * Keeps scalar input distribution within prod's training distribution while
 * letting framing follow the gauge across camera placements. Set to 0 to
 * restore the original (rectifier-sized box * crop scale) behaviour. */
#define APP_AI_RECTIFIER_FIXED_SCALE_CROP  1U
/* Rectifier center is still a little noisy on the current board captures, so
 * bias the crop center back toward the stable training-crop center instead of
 * trusting the raw rectifier center outright. This keeps us close to the
 * scalar's trained framing while still allowing a small amount of movement. */
#define APP_AI_RECTIFIER_CENTER_BLEND_NUMERATOR    1U
#define APP_AI_RECTIFIER_CENTER_BLEND_DENOMINATOR   5U
/* Reject the rectifier's centre prediction if it falls outside the central
 * 80% of the frame in either axis — that's almost certainly a runaway
 * prediction and we should fall back to the static training crop instead of
 * shifting the scalar's framing into the bezel/background. Only used when
 * APP_AI_RECTIFIER_FIXED_SCALE_CROP is enabled. */
#define APP_AI_RECTIFIER_CENTER_MIN_RATIO  0.10f
#define APP_AI_RECTIFIER_CENTER_MAX_RATIO  0.90f
/* Use a tiny burst history so the user-facing reading can average across a
 * few frames instead of reacting to a single noisy capture. */
#define APP_AI_INFERENCE_BURST_HISTORY_SIZE   3U
/* If the scene jumps by a lot, drop the burst history and re-lock quickly to
 * the new setpoint instead of blending two different gauge positions. */
#define APP_AI_INFERENCE_BURST_RESET_DELTA_C  12.0f
/* xSPI2 window base address (chip address 0). */
#define APP_AI_XSPI2_CHIP_BASE_ADDR       0x70000000UL
/* Scalar model: immediately after FSBL (0x70000000) + App (0x70100000, 1 MB
 * window). Must match FLASH_SCALAR address in flash_boot.bat.
 * Size: ~3.07 MB → occupies 0x70200000–0x7051FFFF (50 × 64 KB blocks). */
#define APP_AI_XSPI2_SCALAR_BASE_ADDR     0x70200000UL
#define APP_AI_XSPI2_SCALAR_CHIP_OFFSET   (APP_AI_XSPI2_SCALAR_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)
/* Rectifier model: immediately after scalar region (aligned to next 64 KB).
 * Must match FLASH_RECTIFIER address in flash_boot.bat.
 * Size: ~118 KB → occupies 0x70600000–0x7053FFFF (2 × 64 KB blocks). */
#define APP_AI_XSPI2_RECTIFIER_BASE_ADDR  0x70600000UL
#define APP_AI_XSPI2_RECTIFIER_CHIP_OFFSET (APP_AI_XSPI2_RECTIFIER_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)
#define APP_AI_XSPI2_OBB_BASE_ADDR        0x70700000UL
#define APP_AI_XSPI2_OBB_CHIP_OFFSET      (APP_AI_XSPI2_OBB_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)
/* Legacy alias used by the single-stage logging helpers; points to scalar. */
#define APP_AI_XSPI2_MODEL_BASE_ADDR      APP_AI_XSPI2_SCALAR_BASE_ADDR
#define APP_AI_XSPI2_MODEL_CHIP_OFFSET    APP_AI_XSPI2_SCALAR_CHIP_OFFSET
/* FileX can take a while to recover from card init retries or media errors.
 * Give the loader a longer window so we do not give up just before the stack
 * settles. */
#define APP_AI_FILEX_MEDIA_READY_TIMEOUT_MS 180000U
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
static bool app_ai_runtime_initialized = false;
static volatile float app_ai_last_inference_value = 0.0f;
static volatile bool app_ai_last_inference_valid = false;
static float app_ai_inference_burst_history
		[APP_AI_INFERENCE_BURST_HISTORY_SIZE] = { 0.0f };
static size_t app_ai_inference_burst_history_count = 0U;
static size_t app_ai_inference_burst_history_next_index = 0U;
static bool app_ai_npu_hw_initialized = false;
static bool app_ai_xspi2_initialized = false;
static const struct AppAI_ModelStageSpec *app_ai_loaded_xspi2_stage = NULL;
static bool app_ai_forced_crop_active = false;
static size_t app_ai_forced_crop_x_min = 0U;
static size_t app_ai_forced_crop_y_min = 0U;
static size_t app_ai_forced_crop_width = 0U;
static size_t app_ai_forced_crop_height = 0U;
static const char *app_ai_forced_crop_label = NULL;
__attribute__((section(".xspi2_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI2_scalar_full_finetune_from_best_piecewise_calibrated_int8[32U] = {
	0U,
};
/* Rectifier pool placed in its own section so the linker script can map it to
 * the rectifier flash region at 0x70600000 — matching FLASH_RECTIFIER in
 * flash_boot.bat.  The NPU resolves all weight addresses as:
 *   _mem_pool_xSPI2_mobilenetv2_rectifier_hardcase_finetune + internal_offset
 * so this symbol MUST live at the base of the flashed blob. */
__attribute__((section(".xspi2_rectifier_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI2_mobilenetv2_rectifier_hardcase_finetune[32U] = {
	0U,
};
/* OBB localizer pool lives in its own section so the linker can map it to the
 * prodv0.3 flash slot without disturbing the rectifier blob. */
__attribute__((section(".xspi2_obb_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI2_mobilenetv2_obb_longterm[32U] = {
	0U,
};
static uint8_t app_ai_xspi2_program_buffer[APP_AI_XSPI2_PROGRAM_CHUNK_BYTES];
/* Start/tail signatures for the current atonbuf.xSPI2.raw.
 * Update these when a new model is exported by running:
 *   python3 -c "
 *     d=open('st_ai_output/atonbuf.xSPI2.raw','rb').read()
 *     print('start:', bytes(d[:16]).hex())
 *     print('tail: ', bytes(d[-16:]).hex())" */
static const uint8_t app_ai_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES] = {
	0xEFU, 0x1BU, 0x2BU, 0xE0U, 0xD7U, 0xE5U, 0xECU, 0x07U,
	0x04U, 0x00U, 0x34U, 0xECU, 0x1AU, 0xDDU, 0x14U, 0x05U,
};
static const uint8_t app_ai_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0xDCU,
};
/* Rectifier v3 xSPI2 signatures used when the board boots with the rectifier
 * blob already flashed at 0x70200000. */
static const uint8_t app_ai_rectifier_xspi2_signature_start[
		APP_AI_XSPI2_PROBE_BYTES] = {
	0x0FU, 0x11U, 0xF8U, 0x10U, 0xD0U, 0xD8U, 0x0EU, 0x28U,
	0x99U, 0xCDU, 0x98U, 0x7DU, 0xBCU, 0x43U, 0x5EU, 0xF2U,
};
static const uint8_t app_ai_rectifier_xspi2_signature_tail[
		APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x80U,
};
/* prodv0.3 OBB xSPI2 signatures. */
static const uint8_t app_ai_obb_xspi2_signature_start[
		APP_AI_XSPI2_PROBE_BYTES] = {
	0xF2U, 0x17U, 0x29U, 0xE2U, 0xDCU, 0xEBU, 0xEDU, 0x04U,
	0x09U, 0x01U, 0x35U, 0xEBU, 0x14U, 0xDEU, 0x0FU, 0x02U,
};
static const uint8_t app_ai_obb_xspi2_signature_tail[
		APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x7FU,
};
/* Per-stage programmed sizes. Set during provisioning and used by the verify
 * functions for the tail probe offset. Keeping them separate prevents the
 * scalar tail check from using the rectifier's file size (or vice-versa) when
 * stages alternate. */
static ULONG app_ai_scalar_programmed_size = 0UL;
static ULONG app_ai_rectifier_programmed_size = 0UL;
static ULONG app_ai_obb_programmed_size = 0UL;
/* Legacy alias kept so existing references still compile; points to the scalar
 * size which was the only stage before the rectifier was added. */
static ULONG app_ai_xspi2_programmed_size = 0UL;

/* Per-stage signature caches populated from the SD file during provisioning.
 * Using SD-sourced bytes means verify never goes stale when the model blob is
 * replaced, regardless of what the hardcoded fallback constants say. */
static uint8_t app_ai_scalar_sig_start[APP_AI_XSPI2_PROBE_BYTES] = { 0U };
static uint8_t app_ai_scalar_sig_tail[APP_AI_XSPI2_PROBE_BYTES] = { 0U };
static bool    app_ai_scalar_sig_valid = false;
static uint8_t app_ai_rectifier_sig_start[APP_AI_XSPI2_PROBE_BYTES] = { 0U };
static uint8_t app_ai_rectifier_sig_tail[APP_AI_XSPI2_PROBE_BYTES] = { 0U };
static bool    app_ai_rectifier_sig_valid = false;
static uint8_t app_ai_obb_sig_start[APP_AI_XSPI2_PROBE_BYTES] = { 0U };
static uint8_t app_ai_obb_sig_tail[APP_AI_XSPI2_PROBE_BYTES] = { 0U };
static bool    app_ai_obb_sig_valid = false;

/* Declare the generated NN instance locally so the dry-run helper can run the
 * AtoNN runtime on the exact network produced by Cube.AI. */
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(
		scalar_full_finetune_from_best_piecewise_calibrated_int8);
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(
		mobilenetv2_rectifier_hardcase_finetune);
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(
		mobilenetv2_obb_longterm);

typedef struct AppAI_ModelStageSpec AppAI_ModelStageSpec;

struct AppAI_ModelStageSpec {
	const char *stage_label;
	const char *model_image_path;
	NN_Instance_TypeDef *nn_instance;
	bool (*network_init_fn)(void);
	bool (*inference_init_fn)(void);
	bool uses_rectifier_box;
	uint32_t xspi2_chip_offset; /* byte offset from chip base (0x70000000) */
	uint32_t xspi2_base_addr;   /* mapped window address for this stage */
};

typedef struct {
	float center_x;
	float center_y;
	float box_w;
	float box_h;
} AppAI_RectifierBox;

typedef struct {
	float center_x;
	float center_y;
	float box_w;
	float box_h;
	float angle_cos;
	float angle_sin;
	float theta_rad;
} AppAI_ObbBox;

typedef struct {
	size_t x_min;
	size_t y_min;
	size_t width;
	size_t height;
} AppAI_SourceCrop;

static const AppAI_ModelStageSpec app_ai_scalar_stage = {
	.stage_label = "scalar",
	.model_image_path = APP_AI_SCALAR_XSPI2_MODEL_IMAGE_PATH,
	.nn_instance = &NN_Instance_scalar_full_finetune_from_best_piecewise_calibrated_int8,
	.network_init_fn =
			LL_ATON_EC_Network_Init_scalar_full_finetune_from_best_piecewise_calibrated_int8,
	.inference_init_fn =
			LL_ATON_EC_Inference_Init_scalar_full_finetune_from_best_piecewise_calibrated_int8,
	.uses_rectifier_box = false,
	.xspi2_chip_offset = APP_AI_XSPI2_SCALAR_CHIP_OFFSET,
	.xspi2_base_addr   = APP_AI_XSPI2_SCALAR_BASE_ADDR,
};

static const AppAI_ModelStageSpec app_ai_rectifier_stage = {
	.stage_label = "rectifier",
	.model_image_path = APP_AI_RECTIFIER_XSPI2_MODEL_IMAGE_PATH,
	.nn_instance = &NN_Instance_mobilenetv2_rectifier_hardcase_finetune,
	.network_init_fn =
			LL_ATON_EC_Network_Init_mobilenetv2_rectifier_hardcase_finetune,
	.inference_init_fn =
			LL_ATON_EC_Inference_Init_mobilenetv2_rectifier_hardcase_finetune,
	.uses_rectifier_box = true,
	.xspi2_chip_offset = APP_AI_XSPI2_RECTIFIER_CHIP_OFFSET,
	.xspi2_base_addr   = APP_AI_XSPI2_RECTIFIER_BASE_ADDR,
};

static const AppAI_ModelStageSpec app_ai_obb_stage = {
	.stage_label = "obb",
	.model_image_path = APP_AI_OBB_XSPI2_MODEL_IMAGE_PATH,
	.nn_instance = &NN_Instance_mobilenetv2_obb_longterm,
	.network_init_fn = LL_ATON_EC_Network_Init_mobilenetv2_obb_longterm,
	.inference_init_fn = LL_ATON_EC_Inference_Init_mobilenetv2_obb_longterm,
	.uses_rectifier_box = false,
	.xspi2_chip_offset = APP_AI_XSPI2_OBB_CHIP_OFFSET,
	.xspi2_base_addr   = APP_AI_XSPI2_OBB_BASE_ADDR,
};
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
static const LL_Buffer_InfoTypeDef *AppAI_GetInputBufferInfo(void);
static const LL_Buffer_InfoTypeDef *AppAI_GetOutputBufferInfo(void);
static const LL_Buffer_InfoTypeDef *AppAI_GetStageInputBufferInfo(
		const AppAI_ModelStageSpec *stage);
static const LL_Buffer_InfoTypeDef *AppAI_GetStageOutputBufferInfo(
		const AppAI_ModelStageSpec *stage);
static const LL_Buffer_InfoTypeDef *AppAI_FindBufferInfoByName(
		const LL_Buffer_InfoTypeDef *buffer_list, const char *name);
static const LL_Buffer_InfoTypeDef *AppAI_FindFirstBufferInfoByNames(
		const LL_Buffer_InfoTypeDef *buffer_list, const char *const *names,
		size_t name_count);
static void AppAI_LogInitFailure(const char *step);
static void AppAI_LogXspi2LoadFailure(const char *step, UINT fx_status,
		int32_t bsp_status);
static void AppAI_LogXspi2ProgramChunkProgress(ULONG chunk_index,
		ULONG flash_offset, ULONG chunk_size);
static void AppAI_LogXspi2FlashStatus(const char *label);
static void AppAI_LogXspi2PrefixBytes(const char *label,
		const uint8_t *bytes);
static void AppAI_LogFrameSignature(const uint8_t *frame_bytes,
		size_t frame_size);
static void AppAI_LogInputSignature(const float *input_buffer,
		size_t input_float_count);
static void AppAI_LogInputTensorWindow(const float *input_buffer,
		size_t input_float_count);
static void AppAI_LogInputProbeSummary(const float *input_buffer,
		size_t input_float_count);
static void AppAI_LogTensorRowSamples(const char *label,
		const float *input_buffer, size_t tensor_width, size_t row_y,
		size_t x_min, size_t x_max);
static void AppAI_LogSourcePatch(const char *label, const uint8_t *frame_bytes,
		size_t frame_width_pixels, size_t center_x, size_t center_y,
		size_t radius_pixels);
static void AppAI_LogTensorPatch(const char *label, const float *input_buffer,
		size_t tensor_width, size_t center_x, size_t center_y,
		size_t radius_pixels);
static void AppAI_LogSourceCropWindow(const uint8_t *frame_bytes,
		size_t frame_size, size_t frame_width_pixels, size_t frame_height_pixels,
		size_t crop_x_min, size_t crop_y_min, size_t crop_width,
		size_t crop_height);
static void AppAI_LogInt8BufferSignature(const char *label,
		const int8_t *buffer_ptr, size_t buffer_len_bytes);
static void AppAI_LogRawBufferSignature(const char *label,
		const uint8_t *buffer_ptr, size_t buffer_len_bytes);
static const char *AppAI_BufferTypeName(const LL_Buffer_InfoTypeDef *buffer_info);
static void AppAI_LogBufferInfoAndSignature(const char *label,
		const LL_Buffer_InfoTypeDef *buffer_info);
static void AppAI_LogBufferPreview(const char *label,
		const LL_Buffer_InfoTypeDef *buffer_info);
#if APP_AI_USE_ADAPTIVE_GAUGE_CROP
static bool AppAI_EstimateGaugeCropBoxFromYuv422(const uint8_t *frame_bytes,
		size_t frame_size, size_t frame_width_pixels, size_t frame_height_pixels,
		size_t *crop_x_min, size_t *crop_y_min, size_t *crop_width,
		size_t *crop_height);
#endif
static bool AppAI_LogXspi2ModelFilePrefix(FX_FILE *model_file_ptr);
static void AppAI_LogXspi2FlashPrefix(void);
static void AppAI_LogXspi2MappedScaleBytes(void);
static void AppAI_LogXspi2IndirectAndMappedPrefix(void);
static void AppAI_LogFloatApprox(const char *label, float value);
static float AppAI_TraceAndApplyInferenceCalibration(float raw_value);
static void AppAI_ResetInferenceBurstHistory(void);
static bool AppAI_IsFiniteFloat(float value);
static float AppAI_FilterInferenceValue(float value);
static void AppAI_LogInferenceResult(
		const LL_Buffer_InfoTypeDef *output_buffer_info);
static void AppAI_LogRectifierResult(
		const LL_Buffer_InfoTypeDef *output_buffer_info,
		const AppAI_RectifierBox *rectifier_box);
static void AppAI_LogObbResult(
		const LL_Buffer_InfoTypeDef *output_buffer_info,
		const AppAI_ObbBox *obb_box);
static int AppAI_ApplyCacheRange(uint32_t start_addr, uint32_t end_addr,
		bool clean, bool invalidate);
static void AppAI_EnableNpuMemoryAndCaches(void);
static void AppAI_ConfigureNpuAccessControl(void);
static void AppAI_ConfigureNpuRisafDefaults(void);
static bool AppAI_EnsureStageRuntimeReady(const AppAI_ModelStageSpec *stage);
static bool AppAI_EnsureXspi2ModelImageReadyForStage(
		const AppAI_ModelStageSpec *stage);
static bool AppAI_WaitForFileXMediaReady(uint32_t timeout_ms);
static bool AppAI_RuntimeInitStepwise(void);
static bool AppAI_PreprocessYuv422FrameToFloatInput(const uint8_t *frame_bytes,
		size_t frame_size, float *input_buffer, size_t input_float_count);
static float AppAI_ClampNormalizedFloat(float value);
static uint8_t AppAI_ReadYuv422Luma(const uint8_t *frame_bytes,
		size_t frame_width_pixels, size_t source_x, size_t source_y);
static void AppAI_ReadYuv422Quartet(const uint8_t *frame_bytes,
		size_t frame_width_pixels, size_t source_x, size_t source_y,
		uint8_t *quad_out);
static float AppAI_ReadNormalizedGrayFromYuv422Pixel(const uint8_t *frame_bytes,
		size_t frame_width_pixels, size_t source_x, size_t source_y);
static void AppAI_ReadRgbFromYuv422Pixel(const uint8_t *frame_bytes,
		size_t frame_width_pixels, size_t source_x, size_t source_y,
		float *r_out, float *g_out, float *b_out);
static void AppAI_SetForcedCrop(const char *label, size_t x_min,
		size_t y_min, size_t width, size_t height);
static void AppAI_ClearForcedCrop(void);
static bool AppAI_DecodeObbCropBox(const LL_Buffer_InfoTypeDef *output_buffer_info,
		AppAI_SourceCrop *crop_out, AppAI_ObbBox *obb_box_out);
static bool AppAI_DecodeRectifierCropBox(
		const LL_Buffer_InfoTypeDef *output_buffer_info,
		AppAI_SourceCrop *crop_out,
		AppAI_RectifierBox *rectifier_box_out);
int mcu_cache_clean_range(uint32_t start_addr, uint32_t end_addr) {
	return AppAI_ApplyCacheRange(start_addr, end_addr, true, false);
}

int mcu_cache_invalidate_range(uint32_t start_addr, uint32_t end_addr) {
	return AppAI_ApplyCacheRange(start_addr, end_addr, false, true);
}

static bool AppAI_EnsureNpuHardwareReady(void) {
	if (app_ai_npu_hw_initialized) {
		return true;
	}

	(void) DebugConsole_WriteString("[AI] NPU hardware bring-up start.\r\n");
	__HAL_RCC_NPU_CLK_ENABLE();
	__HAL_RCC_NPU_FORCE_RESET();
	__HAL_RCC_NPU_RELEASE_RESET();
	__HAL_RCC_NPU_CLK_SLEEP_DISABLE();

	__HAL_RCC_CACHEAXI_CLK_ENABLE();
	__HAL_RCC_CACHEAXI_FORCE_RESET();
	__HAL_RCC_CACHEAXI_RELEASE_RESET();
	__HAL_RCC_CACHEAXI_CLK_SLEEP_DISABLE();

	AppAI_EnableNpuMemoryAndCaches();

	npu_cache_init();
	npu_cache_enable();

	AppAI_ConfigureNpuAccessControl();
	AppAI_ConfigureNpuRisafDefaults();

	app_ai_npu_hw_initialized = true;
	(void) DebugConsole_WriteString("[AI] NPU hardware bring-up OK.\r\n");
	return true;
}

static bool AppAI_EnsureXspi2MemoryReady(void) {
	BSP_XSPI_NOR_Init_t flash = { 0 };
	RCC_PeriphCLKInitTypeDef periph_clk = { 0 };
	int32_t bsp_status = BSP_ERROR_NONE;

	if (app_ai_xspi2_initialized) {
		return true;
	}

	(void) DebugConsole_WriteString("[AI] XSPI2 memory bring-up start.\r\n");
	/* If a prior verify attempt left the flash in memory-mapped mode, erase and
	 * write commands will fail.  Take it back to indirect mode first. DeInit
	 * handles this cleanly regardless of the current BSP context state. */
	(void) BSP_XSPI_NOR_DeInit(0U);

	periph_clk.PeriphClockSelection = RCC_PERIPHCLK_XSPI2;
	periph_clk.Xspi2ClockSelection = RCC_XSPI2CLKSOURCE_IC3;
	if (HAL_RCCEx_PeriphCLKConfig(&periph_clk) != HAL_OK) {
		(void) DebugConsole_WriteString("[AI] XSPI2 clock config failed.\r\n");
		return false;
	}

	flash.InterfaceMode = BSP_XSPI_NOR_OPI_MODE;
	flash.TransferRate = BSP_XSPI_NOR_STR_TRANSFER;
	bsp_status = BSP_XSPI_NOR_Init(0U, &flash);
	if (bsp_status != BSP_ERROR_NONE) {
#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
		char msg[96];
		(void) snprintf(msg, sizeof(msg),
				"[AI] BSP_XSPI_NOR_Init for provisioning failed: %ld\r\n",
				(long) bsp_status);
		(void) DebugConsole_WriteString(msg);
#endif
		return false;
	}

	(void) DebugConsole_WriteString("[AI] XSPI2 memory bring-up OK.\r\n");
	return true;
}

static bool AppAI_ReconfigureXspi2ForRuntime(void) {
	BSP_XSPI_NOR_Init_t flash = { 0 };
	RCC_PeriphCLKInitTypeDef periph_clk = { 0 };
	int32_t bsp_status = BSP_ERROR_NONE;

	/* Drop out of memory-mapped mode before changing the flash transfer rate.
	 * Clear the provisioning-mode guard so EnsureXspi2MemoryReady will
	 * re-initialize the peripheral into indirect/write mode if provisioning
	 * is needed again after this reconfigure. */
	app_ai_xspi2_initialized = false;
	(void) DebugConsole_WriteString("[AI] xSPI2 runtime reconfigure: disable MM start.\r\n");
	(void) BSP_XSPI_NOR_DisableMemoryMappedMode(0U);
	(void) DebugConsole_WriteString("[AI] xSPI2 runtime reconfigure: disable MM OK.\r\n");
	(void) DebugConsole_WriteString("[AI] xSPI2 runtime reconfigure: deinit start.\r\n");
	(void) BSP_XSPI_NOR_DeInit(0U);
	(void) DebugConsole_WriteString("[AI] xSPI2 runtime reconfigure: deinit OK.\r\n");

	periph_clk.PeriphClockSelection = RCC_PERIPHCLK_XSPI2;
	periph_clk.Xspi2ClockSelection = RCC_XSPI2CLKSOURCE_IC3;
	(void) DebugConsole_WriteString(
			"[AI] xSPI2 runtime reconfigure: clock config start.\r\n");
	if (HAL_RCCEx_PeriphCLKConfig(&periph_clk) != HAL_OK) {
		DebugConsole_WriteString("[AI] xSPI2 runtime reconfigure: clock config failed.\r\n");
		return false;
	}
	(void) DebugConsole_WriteString(
			"[AI] xSPI2 runtime reconfigure: clock config OK.\r\n");

	flash.InterfaceMode = BSP_XSPI_NOR_OPI_MODE;
	/* Keep the runtime in STR mode for now. The board already initializes and
	 * programs the xSPI2 NOR successfully in STR, while the DTR reconfigure
	 * path was failing during BSP_XSPI_NOR_Init() with -5. We only need the
	 * mapped window to be valid for the stage runtime. */
	flash.TransferRate = BSP_XSPI_NOR_STR_TRANSFER;
	(void) DebugConsole_WriteString("[AI] xSPI2 runtime reconfigure: init start.\r\n");
	bsp_status = BSP_XSPI_NOR_Init(0U, &flash);
	if (bsp_status != BSP_ERROR_NONE) {
		char msg[96];
		(void) snprintf(msg, sizeof(msg),
				"[AI] xSPI2 runtime reconfigure: init failed: %ld\r\n",
				(long) bsp_status);
		DebugConsole_WriteString(msg);
		return false;
	}
	(void) DebugConsole_WriteString("[AI] xSPI2 runtime reconfigure: init OK.\r\n");

	/* Leave the peripheral in indirect mode so callers that need to probe flash
	 * via BSP_XSPI_NOR_Read can do so immediately.  Callers that need the
	 * memory-mapped window must call BSP_XSPI_NOR_EnableMemoryMappedMode()
	 * themselves after this function returns. */
	return true;
}

static bool AppAI_Xspi2EnableMemoryMappedMode(void) {
	(void) DebugConsole_WriteString("[AI] xSPI2 enable MM start.\r\n");
	if (BSP_XSPI_NOR_EnableMemoryMappedMode(0U) != BSP_ERROR_NONE) {
		(void) DebugConsole_WriteString("[AI] xSPI2 enable MM failed.\r\n");
		return false;
	}
	(void) DebugConsole_WriteString("[AI] xSPI2 enable MM OK.\r\n");
	return true;
}

static bool AppAI_Xspi2ReadFlashProbe(const uint32_t chip_base_offset,
		const uint32_t flash_offset,
		const uint8_t *expected_bytes, const size_t expected_length) {
	uint8_t flash_bytes[APP_AI_XSPI2_PROBE_BYTES] = { 0U };

	if ((expected_bytes == NULL) || (expected_length == 0U)
			|| (expected_length > APP_AI_XSPI2_PROBE_BYTES)) {
		return false;
	}

	if (BSP_XSPI_NOR_Read(0U, flash_bytes,
			chip_base_offset + flash_offset,
			(uint32_t) expected_length) != BSP_ERROR_NONE) {
		return false;
	}

	return (memcmp(flash_bytes, expected_bytes, expected_length) == 0);
}

static bool AppAI_Xspi2ReadMappedProbe(const uint32_t flash_offset,
		const uint8_t *expected_bytes, const size_t expected_length) {
	const uint8_t *const flash_ptr = (const uint8_t *) (APP_AI_XSPI2_MODEL_BASE_ADDR
			+ flash_offset);

	if ((expected_bytes == NULL) || (expected_length == 0U)
			|| (expected_length > APP_AI_XSPI2_PROBE_BYTES)) {
		return false;
	}

	(void) mcu_cache_invalidate_range((uint32_t) flash_ptr,
			(uint32_t) flash_ptr + (uint32_t) expected_length);

	return (memcmp(flash_ptr, expected_bytes, expected_length) == 0);
}

#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
static void AppAI_LogXspi2PrefixBytes(const char *label,
		const uint8_t *bytes) {
	if ((label == NULL) || (bytes == NULL)) {
		return;
	}

	DebugConsole_Printf(
			"[AI] %s %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X\r\n",
			label, bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5],
			bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11],
			bytes[12], bytes[13], bytes[14], bytes[15]);
}

/**
 * @brief Print a compact signature for the captured input frame.
 *
 * This makes it easy to compare whether two runs actually fed different
 * camera data into the model.
 */
static void AppAI_LogFrameSignature(const uint8_t *frame_bytes,
		size_t frame_size) {
	uint8_t first_bytes[8U] = { 0U };
	uint32_t hash = 2166136261UL;
	size_t preview_count = 0U;

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		return;
	}

	if ((frame_bytes == NULL) || (frame_size == 0U)) {
		DebugConsole_Printf("[AI] Frame signature skipped: empty frame.\r\n");
		return;
	}

	preview_count = (frame_size < sizeof(first_bytes)) ? frame_size
			: sizeof(first_bytes);
	for (size_t index = 0U; index < preview_count; index++) {
		first_bytes[index] = frame_bytes[index];
	}

	for (size_t index = 0U; index < frame_size; index++) {
		hash ^= frame_bytes[index];
		hash *= 16777619UL;
	}

	DebugConsole_Printf(
			"[AI] Frame signature: len=%lu hash=0x%08lX first8=[%02X %02X %02X %02X %02X %02X %02X %02X]\r\n",
			(unsigned long) frame_size, (unsigned long) hash,
			(unsigned int) first_bytes[0], (unsigned int) first_bytes[1],
			(unsigned int) first_bytes[2], (unsigned int) first_bytes[3],
			(unsigned int) first_bytes[4], (unsigned int) first_bytes[5],
			(unsigned int) first_bytes[6], (unsigned int) first_bytes[7]);
}

/**
 * @brief Print a compact signature for the preprocessed model input tensor.
 *
 * This tells us whether the captured scene is still distinct after the
 * YUV422-to-RGB preprocessing and dial ROI crop.
 */
static void AppAI_LogInputSignature(const float *input_buffer,
		size_t input_float_count) {
	const uint8_t *bytes = NULL;
	uint8_t first_bytes[8U] = { 0U };
	uint32_t first_words[4U] = { 0U };
	uint32_t hash = 2166136261UL;
	size_t byte_count = 0U;
	size_t preview_bytes = 0U;
	size_t preview_words = 0U;

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		return;
	}

	if ((input_buffer == NULL) || (input_float_count == 0U)) {
		DebugConsole_Printf("[AI] Input signature skipped: empty tensor.\r\n");
		return;
	}

	bytes = (const uint8_t *) input_buffer;
	byte_count = input_float_count * sizeof(float);
	preview_bytes = (byte_count < sizeof(first_bytes)) ? byte_count
			: sizeof(first_bytes);
	preview_words = (input_float_count < 4U) ? input_float_count : 4U;

	for (size_t index = 0U; index < preview_bytes; index++) {
		first_bytes[index] = bytes[index];
	}

	for (size_t index = 0U; index < preview_words; index++) {
		(void) memcpy(&first_words[index], &input_buffer[index],
				sizeof(uint32_t));
	}

	for (size_t index = 0U; index < byte_count; index++) {
		hash ^= bytes[index];
		hash *= 16777619UL;
	}

	DebugConsole_Printf(
			"[AI] Input signature: floats=%lu hash=0x%08lX first8=[%02X %02X %02X %02X %02X %02X %02X %02X] first4=[0x%08lX,0x%08lX,0x%08lX,0x%08lX]\r\n",
			(unsigned long) input_float_count, (unsigned long) hash,
			(unsigned int) first_bytes[0], (unsigned int) first_bytes[1],
			(unsigned int) first_bytes[2], (unsigned int) first_bytes[3],
			(unsigned int) first_bytes[4], (unsigned int) first_bytes[5],
			(unsigned int) first_bytes[6], (unsigned int) first_bytes[7],
			(unsigned long) first_words[0], (unsigned long) first_words[1],
			(unsigned long) first_words[2], (unsigned long) first_words[3]);
}

/**
 * @brief Print a small diagnostic window from the center of the input tensor.
 *
 * The top-left bytes can be padding or background, so this summary looks at the
 * tensor region where the dial should live after cropping and resizing.
 */
static void AppAI_LogInputTensorWindow(const float *input_buffer,
		size_t input_float_count) {
	const size_t tensor_width = (size_t) APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const size_t tensor_height = (size_t) APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	const size_t center_x = tensor_width / 2U;
	const size_t center_y = tensor_height / 2U;
	const size_t window_radius = 8U;
	const size_t x_min = (center_x > window_radius) ? (center_x - window_radius)
			: 0U;
	const size_t y_min = (center_y > window_radius) ? (center_y - window_radius)
			: 0U;
	const size_t x_max = ((center_x + window_radius) < tensor_width) ?
			(center_x + window_radius) : tensor_width;
	const size_t y_max = ((center_y + window_radius) < tensor_height) ?
			(center_y + window_radius) : tensor_height;
	float sum_r = 0.0f;
	float sum_g = 0.0f;
	float sum_b = 0.0f;
	float min_r = 1.0f;
	float min_g = 1.0f;
	float min_b = 1.0f;
	float max_r = 0.0f;
	float max_g = 0.0f;
	float max_b = 0.0f;
	float center_r = 0.0f;
	float center_g = 0.0f;
	float center_b = 0.0f;
	unsigned long center_r_milli = 0U;
	unsigned long center_g_milli = 0U;
	unsigned long center_b_milli = 0U;
	unsigned long mean_r_milli = 0U;
	unsigned long mean_g_milli = 0U;
	unsigned long mean_b_milli = 0U;
	unsigned long min_r_milli = 0U;
	unsigned long min_g_milli = 0U;
	unsigned long min_b_milli = 0U;
	unsigned long max_r_milli = 0U;
	unsigned long max_g_milli = 0U;
	unsigned long max_b_milli = 0U;
	size_t sample_count = 0U;

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		return;
	}

	if ((input_buffer == NULL) || (input_float_count < APP_AI_MODEL_INPUT_FLOAT_COUNT)) {
		return;
	}

	for (size_t y = y_min; y < y_max; y++) {
		for (size_t x = x_min; x < x_max; x++) {
			const size_t pixel_index = (y * tensor_width) + x;
			const size_t base = pixel_index * 3U;
			const float r = input_buffer[base + 0U];
			const float g = input_buffer[base + 1U];
			const float b = input_buffer[base + 2U];

			if (r < min_r) {
				min_r = r;
			}
			if (g < min_g) {
				min_g = g;
			}
			if (b < min_b) {
				min_b = b;
			}
			if (r > max_r) {
				max_r = r;
			}
			if (g > max_g) {
				max_g = g;
			}
			if (b > max_b) {
				max_b = b;
			}

			sum_r += r;
			sum_g += g;
			sum_b += b;
			sample_count++;
		}
	}

	{
		const size_t center_base = ((center_y * tensor_width) + center_x) * 3U;
		center_r = input_buffer[center_base + 0U];
		center_g = input_buffer[center_base + 1U];
		center_b = input_buffer[center_base + 2U];
	}

	if (sample_count == 0U) {
		return;
	}

	center_r_milli = (unsigned long) ((center_r * 1000.0f) + 0.5f);
	center_g_milli = (unsigned long) ((center_g * 1000.0f) + 0.5f);
	center_b_milli = (unsigned long) ((center_b * 1000.0f) + 0.5f);
	mean_r_milli = (unsigned long) ((sum_r / (float) sample_count) * 1000.0f
			+ 0.5f);
	mean_g_milli = (unsigned long) ((sum_g / (float) sample_count) * 1000.0f
			+ 0.5f);
	mean_b_milli = (unsigned long) ((sum_b / (float) sample_count) * 1000.0f
			+ 0.5f);
	min_r_milli = (unsigned long) (min_r * 1000.0f + 0.5f);
	min_g_milli = (unsigned long) (min_g * 1000.0f + 0.5f);
	min_b_milli = (unsigned long) (min_b * 1000.0f + 0.5f);
	max_r_milli = (unsigned long) (max_r * 1000.0f + 0.5f);
	max_g_milli = (unsigned long) (max_g * 1000.0f + 0.5f);
	max_b_milli = (unsigned long) (max_b * 1000.0f + 0.5f);

	DebugConsole_Printf(
			"[AI] Tensor center window: x=[%lu,%lu) y=[%lu,%lu) center_milli=[%lu %lu %lu] mean_milli=[%lu %lu %lu] min_milli=[%lu %lu %lu] max_milli=[%lu %lu %lu]\r\n",
			(unsigned long) x_min, (unsigned long) x_max, (unsigned long) y_min,
			(unsigned long) y_max, center_r_milli, center_g_milli,
			center_b_milli, mean_r_milli, mean_g_milli, mean_b_milli,
			min_r_milli, min_g_milli, min_b_milli, max_r_milli, max_g_milli,
			max_b_milli);

	AppAI_LogTensorRowSamples("top", input_buffer, tensor_width,
			y_min + ((y_max - y_min) / 4U), x_min, x_max);
	AppAI_LogTensorRowSamples("mid", input_buffer, tensor_width, center_y,
			x_min, x_max);
	AppAI_LogTensorRowSamples("bottom", input_buffer, tensor_width,
			y_min + (((y_max - y_min) * 3U) / 4U), x_min, x_max);
}

/**
 * @brief Print a compact fingerprint for the tensor that actually reaches the NPU.
 *
 * The disabled verbose tensor logs are useful in source, but this helper uses
 * direct console writes so we can see a stable per-frame summary on hardware.
 */
static void AppAI_LogInputProbeSummary(const float *input_buffer,
		size_t input_float_count) {
	const size_t tensor_width = (size_t) APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const size_t tensor_height = (size_t) APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	const size_t sample_points[4U][2U] = {
		{ tensor_width / 2U, tensor_height / 2U },
		{ tensor_width / 2U, tensor_height / 4U },
		{ tensor_width / 4U, tensor_height / 2U },
		{ (tensor_width * 3U) / 4U, tensor_height / 2U },
	};
	const char *sample_labels[4U] = {
		"center",
		"top-mid",
		"left-mid",
		"right-mid",
	};
	const uint8_t *bytes = NULL;
	uint8_t first_bytes[8U] = { 0U };
	uint32_t hash = 2166136261UL;
	uint32_t first_words[4U] = { 0U };
	float sum_value = 0.0f;
	float min_value = 1.0f;
	float max_value = 0.0f;
	size_t active_count = 0U;
	size_t byte_count = 0U;
	size_t preview_bytes = 0U;
	size_t preview_words = 0U;

	if ((input_buffer == NULL)
			|| (input_float_count < APP_AI_MODEL_INPUT_FLOAT_COUNT)) {
		(void) DebugConsole_WriteString("[AI] Input probe skipped.\r\n");
		return;
	}

	bytes = (const uint8_t *) input_buffer;
	byte_count = input_float_count * sizeof(float);
	preview_bytes = (byte_count < sizeof(first_bytes)) ? byte_count
			: sizeof(first_bytes);
	preview_words = (input_float_count < 4U) ? input_float_count : 4U;

	for (size_t index = 0U; index < preview_bytes; index++) {
		first_bytes[index] = bytes[index];
	}

	for (size_t index = 0U; index < preview_words; index++) {
		(void) memcpy(&first_words[index], &input_buffer[index],
				sizeof(uint32_t));
	}

	for (size_t index = 0U; index < byte_count; index++) {
		hash ^= bytes[index];
		hash *= 16777619UL;
	}

	for (size_t index = 0U; index < input_float_count; index++) {
		const float value = input_buffer[index];

		sum_value += value;
		if (value < min_value) {
			min_value = value;
		}
		if (value > max_value) {
			max_value = value;
		}
		if ((value > 0.001f) || (value < -0.001f)) {
			active_count++;
		}
	}

	{
		char line[224];
		(void) snprintf(line, sizeof(line),
				"[AI] Input probe: floats=%lu bytes=%lu hash=0x%08lX first8=[%02X %02X %02X %02X %02X %02X %02X %02X] first4=[0x%08lX 0x%08lX 0x%08lX 0x%08lX]\r\n",
				(unsigned long) input_float_count,
				(unsigned long) byte_count, (unsigned long) hash,
				(unsigned int) first_bytes[0], (unsigned int) first_bytes[1],
				(unsigned int) first_bytes[2], (unsigned int) first_bytes[3],
				(unsigned int) first_bytes[4], (unsigned int) first_bytes[5],
				(unsigned int) first_bytes[6], (unsigned int) first_bytes[7],
				(unsigned long) first_words[0], (unsigned long) first_words[1],
				(unsigned long) first_words[2], (unsigned long) first_words[3]);
		(void) DebugConsole_WriteString(line);
	}

	{
		const unsigned long mean_milli =
				(unsigned long) (((sum_value / (float) input_float_count) * 1000.0f)
						+ 0.5f);
		const unsigned long min_milli =
				(unsigned long) ((min_value * 1000.0f) + 0.5f);
		const unsigned long max_milli =
				(unsigned long) ((max_value * 1000.0f) + 0.5f);
		char line[192];

		(void) snprintf(line, sizeof(line),
				"[AI] Input probe stats: active=%lu/%lu mean_milli=%lu min_milli=%lu max_milli=%lu\r\n",
				(unsigned long) active_count, (unsigned long) input_float_count,
				mean_milli, min_milli, max_milli);
		(void) DebugConsole_WriteString(line);
	}

	for (size_t index = 0U; index < 4U; index++) {
		const size_t sample_x = sample_points[index][0U];
		const size_t sample_y = sample_points[index][1U];
		const size_t pixel_index = (sample_y * tensor_width) + sample_x;
		const size_t base = pixel_index * 3U;
		const uint32_t r_bits = 0U;
		const uint32_t g_bits = 0U;
		const uint32_t b_bits = 0U;
		const unsigned long r_milli =
				(unsigned long) ((input_buffer[base + 0U] * 1000.0f) + 0.5f);
		const unsigned long g_milli =
				(unsigned long) ((input_buffer[base + 1U] * 1000.0f) + 0.5f);
		const unsigned long b_milli =
				(unsigned long) ((input_buffer[base + 2U] * 1000.0f) + 0.5f);
		uint32_t sample_bits[3U] = { 0U, 0U, 0U };
		char line[160];

		(void) memcpy(&sample_bits[0U], &input_buffer[base + 0U],
				sizeof(uint32_t));
		(void) memcpy(&sample_bits[1U], &input_buffer[base + 1U],
				sizeof(uint32_t));
		(void) memcpy(&sample_bits[2U], &input_buffer[base + 2U],
				sizeof(uint32_t));

		(void) r_bits;
		(void) g_bits;
		(void) b_bits;

		(void) snprintf(line, sizeof(line),
				"[AI] Input probe %s: x=%lu y=%lu rgb_milli=[%lu %lu %lu] rgb_bits=[0x%08lX 0x%08lX 0x%08lX]\r\n",
				sample_labels[index], (unsigned long) sample_x,
				(unsigned long) sample_y, r_milli, g_milli, b_milli,
				(unsigned long) sample_bits[0U],
				(unsigned long) sample_bits[1U],
				(unsigned long) sample_bits[2U]);
		(void) DebugConsole_WriteString(line);
	}
}

/**
 * @brief Print a few evenly spaced samples from one tensor row.
 *
 * This makes it easier to see whether the row contains dial markings, the
 * needle, or only flat background after the crop/resample step.
 */
static void AppAI_LogTensorRowSamples(const char *label,
		const float *input_buffer, size_t tensor_width, size_t row_y,
		size_t x_min, size_t x_max) {
	const size_t sample_count = 5U;
	size_t positions[5U] = { 0U };

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		return;
	}

	if ((label == NULL) || (input_buffer == NULL) || (tensor_width == 0U)
			|| (x_max <= x_min)) {
		return;
	}

	{
		const size_t span = (x_max > x_min) ? (x_max - x_min - 1U) : 0U;
		positions[0U] = x_min;
		positions[1U] = x_min + (span / 4U);
		positions[2U] = x_min + (span / 2U);
		positions[3U] = x_min + ((span * 3U) / 4U);
		positions[4U] = (x_max > 0U) ? (x_max - 1U) : 0U;
	}

	DebugConsole_Printf("[AI] Row %s y=%lu:",
			label, (unsigned long) row_y);
	for (size_t index = 0U; index < sample_count; index++) {
		const size_t pixel_index = (row_y * tensor_width) + positions[index];
		const size_t base = pixel_index * 3U;
		const unsigned long r_milli =
				(unsigned long) (input_buffer[base + 0U] * 1000.0f + 0.5f);
		const unsigned long g_milli =
				(unsigned long) (input_buffer[base + 1U] * 1000.0f + 0.5f);
		const unsigned long b_milli =
				(unsigned long) (input_buffer[base + 2U] * 1000.0f + 0.5f);

		DebugConsole_Printf(" x=%lu rgb=[%lu %lu %lu]",
				(unsigned long) positions[index], r_milli, g_milli, b_milli);
	}
	DebugConsole_Printf("\r\n");
}

/**
 * @brief Print a compact luma patch from the source YUV422 frame.
 */
static void AppAI_LogSourcePatch(const char *label, const uint8_t *frame_bytes,
		size_t frame_width_pixels, size_t center_x, size_t center_y,
		size_t radius_pixels) {
	const size_t x_min = (center_x > radius_pixels) ? (center_x - radius_pixels)
			: 0U;
	const size_t y_min = (center_y > radius_pixels) ? (center_y - radius_pixels)
			: 0U;
	const size_t x_max = ((center_x + radius_pixels)
			< APP_AI_CAPTURE_FRAME_WIDTH_PIXELS) ? (center_x + radius_pixels)
			: (APP_AI_CAPTURE_FRAME_WIDTH_PIXELS - 1U);
	const size_t y_max = ((center_y + radius_pixels)
			< APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS) ? (center_y + radius_pixels)
			: (APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS - 1U);

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		return;
	}

	if ((label == NULL) || (frame_bytes == NULL) || (frame_width_pixels == 0U)) {
		return;
	}

	DebugConsole_Printf(
			"[AI] %s source patch center=(%lu,%lu) x=[%lu,%lu] y=[%lu,%lu]\r\n",
			label, (unsigned long) center_x, (unsigned long) center_y,
			(unsigned long) x_min, (unsigned long) x_max,
			(unsigned long) y_min, (unsigned long) y_max);

	for (size_t y = y_min; y <= y_max; ++y) {
		DebugConsole_Printf("[AI] %s y=%lu:", label, (unsigned long) y);
		for (size_t x = x_min; x <= x_max; ++x) {
			const uint8_t luma = AppAI_ReadYuv422Luma(frame_bytes,
					frame_width_pixels, x, y);
			const unsigned long luma_milli =
					(unsigned long) ((luma * 1000U) / 255U);

			DebugConsole_Printf(" x=%lu=%lu", (unsigned long) x, luma_milli);
		}
		DebugConsole_Printf("\r\n");
	}
}

/**
 * @brief Print a compact tensor patch from the preprocessed input buffer.
 */
static void AppAI_LogTensorPatch(const char *label, const float *input_buffer,
		size_t tensor_width, size_t center_x, size_t center_y,
		size_t radius_pixels) {
	const size_t x_min = (center_x > radius_pixels) ? (center_x - radius_pixels)
			: 0U;
	const size_t y_min = (center_y > radius_pixels) ? (center_y - radius_pixels)
			: 0U;
	const size_t x_max = ((center_x + radius_pixels) < tensor_width) ?
			(center_x + radius_pixels) : (tensor_width - 1U);
	const size_t y_max = ((center_y + radius_pixels)
			< APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS) ? (center_y + radius_pixels)
			: (APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS - 1U);

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		return;
	}

	if ((label == NULL) || (input_buffer == NULL) || (tensor_width == 0U)) {
		return;
	}

	DebugConsole_Printf(
			"[AI] %s tensor patch center=(%lu,%lu) x=[%lu,%lu] y=[%lu,%lu]\r\n",
			label, (unsigned long) center_x, (unsigned long) center_y,
			(unsigned long) x_min, (unsigned long) x_max,
			(unsigned long) y_min, (unsigned long) y_max);

	for (size_t y = y_min; y <= y_max; ++y) {
		DebugConsole_Printf("[AI] %s y=%lu:", label, (unsigned long) y);
		for (size_t x = x_min; x <= x_max; ++x) {
			const size_t pixel_index = (y * tensor_width) + x;
			const size_t base = pixel_index * 3U;
			const unsigned long r_milli =
					(unsigned long) (input_buffer[base + 0U] * 1000.0f + 0.5f);
			const unsigned long g_milli =
					(unsigned long) (input_buffer[base + 1U] * 1000.0f + 0.5f);
			const unsigned long b_milli =
					(unsigned long) (input_buffer[base + 2U] * 1000.0f + 0.5f);

			DebugConsole_Printf(" x=%lu=[%lu %lu %lu]",
					(unsigned long) x, r_milli, g_milli, b_milli);
		}
		DebugConsole_Printf("\r\n");
	}
}

/**
 * @brief Print the source-crop luma that is being fed into resize/pad.
 *
 * This is the lowest-level image diagnostic in the pipeline. If these values
 * are already near zero, the problem is before tensor fill. If they are healthy
 * but the tensor is still zero, then the resize/write path is wrong.
 */
static void AppAI_LogSourceCropWindow(const uint8_t *frame_bytes,
		size_t frame_size, size_t frame_width_pixels, size_t frame_height_pixels,
		size_t crop_x_min, size_t crop_y_min, size_t crop_width,
		size_t crop_height) {
	const size_t center_x = crop_x_min + (crop_width / 2U);
	const size_t center_y = crop_y_min + (crop_height / 2U);
	const size_t window_radius = 8U;
	const size_t x_min = (center_x > window_radius) ? (center_x - window_radius)
			: crop_x_min;
	const size_t y_min = (center_y > window_radius) ? (center_y - window_radius)
			: crop_y_min;
	const size_t x_max = ((center_x + window_radius) < (crop_x_min + crop_width)) ?
			(center_x + window_radius) : (crop_x_min + crop_width);
	const size_t y_max = ((center_y + window_radius) < (crop_y_min + crop_height)) ?
			(center_y + window_radius) : (crop_y_min + crop_height);
	uint64_t sum_luma = 0U;
	uint8_t min_luma = 0xFFU;
	uint8_t max_luma = 0U;
	uint8_t center_luma = 0U;
	unsigned long center_luma_milli = 0U;
	unsigned long mean_luma_milli = 0U;
	unsigned long min_luma_milli = 0U;
	unsigned long max_luma_milli = 0U;
	size_t sample_count = 0U;

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		return;
	}

	if ((frame_bytes == NULL) || (frame_size < (frame_width_pixels
			* frame_height_pixels * 2U))) {
		return;
	}

	for (size_t y = y_min; y < y_max; ++y) {
		for (size_t x = x_min; x < x_max; ++x) {
			const uint8_t luma = AppAI_ReadYuv422Luma(frame_bytes,
					frame_width_pixels, x, y);

			if (luma < min_luma) {
				min_luma = luma;
			}
			if (luma > max_luma) {
				max_luma = luma;
			}
			sum_luma += (uint64_t) luma;
			sample_count++;
		}
	}

	center_luma = AppAI_ReadYuv422Luma(frame_bytes, frame_width_pixels,
			center_x, center_y);

	if (sample_count == 0U) {
		return;
	}

	center_luma_milli = (unsigned long) ((center_luma * 1000U) / 255U);
	mean_luma_milli = (unsigned long) (((sum_luma / sample_count) * 1000U)
			/ 255U);
	min_luma_milli = (unsigned long) ((min_luma * 1000U) / 255U);
	max_luma_milli = (unsigned long) ((max_luma * 1000U) / 255U);

	DebugConsole_Printf(
			"[AI] Source crop window: x=[%lu,%lu) y=[%lu,%lu) center_luma_milli=[%lu] mean_luma_milli=[%lu] min_luma_milli=[%lu] max_luma_milli=[%lu]\r\n",
			(unsigned long) x_min, (unsigned long) x_max, (unsigned long) y_min,
			(unsigned long) y_max, center_luma_milli, mean_luma_milli,
			min_luma_milli, max_luma_milli);

	AppAI_LogSourcePatch("Source crop center", frame_bytes, frame_width_pixels,
			center_x, center_y, 2U);

	for (size_t index = 0U; index < 3U; ++index) {
		const char *label = (index == 0U) ? "src_top"
				: (index == 1U) ? "src_mid" : "src_bottom";
		const size_t row_y = (index == 0U) ? (y_min + ((y_max - y_min) / 4U))
				: (index == 1U) ? center_y
				: (y_min + (((y_max - y_min) * 3U) / 4U));
		const size_t sample_span = (x_max > x_min) ? (x_max - x_min - 1U) : 0U;
		const size_t sample_x0 = x_min;
		const size_t sample_x1 = x_min + (sample_span / 4U);
		const size_t sample_x2 = x_min + (sample_span / 2U);
		const size_t sample_x3 = x_min + ((sample_span * 3U) / 4U);
		const size_t sample_x4 = (x_max > 0U) ? (x_max - 1U) : 0U;
		const size_t sample_xs[5U] = {
			sample_x0, sample_x1, sample_x2, sample_x3, sample_x4
		};

		DebugConsole_Printf("[AI] %s y=%lu:", label, (unsigned long) row_y);
		for (size_t sample_index = 0U; sample_index < 5U; ++sample_index) {
			const uint8_t luma = AppAI_ReadYuv422Luma(frame_bytes,
					frame_width_pixels, sample_xs[sample_index], row_y);
			const unsigned long luma_milli =
					(unsigned long) ((luma * 1000U) / 255U);

			DebugConsole_Printf(" x=%lu y=%lu",
					(unsigned long) sample_xs[sample_index], luma_milli);
		}
		DebugConsole_Printf("\r\n");
	}
}

/**
 * @brief Print a compact signature for an int8 tensor or activation buffer.
 *
 * This is useful for the model boundary because it tells us whether the
 * quantized activations are changing even when the final dequantized output is
 * still neutral.
 */
static void AppAI_LogInt8BufferSignature(const char *label,
		const int8_t *buffer_ptr, size_t buffer_len_bytes) {
	uint8_t first_bytes[16U] = { 0U };
	int8_t min_value = 127;
	int8_t max_value = -128;
	uint32_t nonzero_count = 0U;
	uint32_t hash = 2166136261UL;
	size_t preview_count = 0U;

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		return;
	}

	if ((label == NULL) || (buffer_ptr == NULL) || (buffer_len_bytes == 0U)) {
		DebugConsole_Printf(
				"[AI] %s int8 signature skipped: empty buffer.\r\n",
				(label != NULL) ? label : "(unnamed)");
		return;
	}

	preview_count = (buffer_len_bytes < sizeof(first_bytes)) ? buffer_len_bytes
			: sizeof(first_bytes);

	for (size_t index = 0U; index < buffer_len_bytes; ++index) {
		const uint8_t raw_byte = (uint8_t) buffer_ptr[index];

		hash ^= raw_byte;
		hash *= 16777619UL;

		if (buffer_ptr[index] < min_value) {
			min_value = buffer_ptr[index];
		}
		if (buffer_ptr[index] > max_value) {
			max_value = buffer_ptr[index];
		}
		if (buffer_ptr[index] != 0) {
			nonzero_count++;
		}
		if (index < preview_count) {
			first_bytes[index] = raw_byte;
		}
	}

	DebugConsole_Printf(
			"[AI] %s int8 signature: bytes=%lu hash=0x%08lX nonzero=%lu min=%d max=%d first16=[%02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X]\r\n",
			label, (unsigned long) buffer_len_bytes, (unsigned long) hash,
			(unsigned long) nonzero_count, (int) min_value, (int) max_value,
			(unsigned int) first_bytes[0], (unsigned int) first_bytes[1],
			(unsigned int) first_bytes[2], (unsigned int) first_bytes[3],
			(unsigned int) first_bytes[4], (unsigned int) first_bytes[5],
			(unsigned int) first_bytes[6], (unsigned int) first_bytes[7],
			(unsigned int) first_bytes[8], (unsigned int) first_bytes[9],
			(unsigned int) first_bytes[10], (unsigned int) first_bytes[11],
			(unsigned int) first_bytes[12], (unsigned int) first_bytes[13],
			(unsigned int) first_bytes[14], (unsigned int) first_bytes[15]);
}

/**
 * @brief Print a compact signature for any buffer by raw bytes.
 *
 * This avoids guessing the tensor type when we just want to know whether the
 * runtime wrote anything other than zero into a model activation.
 */
static void AppAI_LogRawBufferSignature(const char *label,
		const uint8_t *buffer_ptr, size_t buffer_len_bytes) {
	uint8_t first_bytes[16U] = { 0U };
	uint8_t min_value = 255U;
	uint8_t max_value = 0U;
	uint32_t nonzero_count = 0U;
	uint32_t hash = 2166136261UL;
	size_t preview_count = 0U;

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		return;
	}

	if ((label == NULL) || (buffer_ptr == NULL) || (buffer_len_bytes == 0U)) {
		DebugConsole_Printf(
				"[AI] %s raw signature skipped: empty buffer.\r\n",
				(label != NULL) ? label : "(unnamed)");
		return;
	}

	preview_count = (buffer_len_bytes < sizeof(first_bytes)) ? buffer_len_bytes
			: sizeof(first_bytes);

	for (size_t index = 0U; index < buffer_len_bytes; ++index) {
		const uint8_t value = buffer_ptr[index];

		hash ^= value;
		hash *= 16777619UL;

		if (value < min_value) {
			min_value = value;
		}
		if (value > max_value) {
			max_value = value;
		}
		if (value != 0U) {
			nonzero_count++;
		}
		if (index < preview_count) {
			first_bytes[index] = value;
		}
	}

	DebugConsole_Printf(
			"[AI] %s raw signature: bytes=%lu hash=0x%08lX nonzero=%lu min=%u max=%u first16=[%02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X]\r\n",
			label, (unsigned long) buffer_len_bytes, (unsigned long) hash,
			(unsigned long) nonzero_count, (unsigned int) min_value,
			(unsigned int) max_value, (unsigned int) first_bytes[0],
			(unsigned int) first_bytes[1], (unsigned int) first_bytes[2],
			(unsigned int) first_bytes[3], (unsigned int) first_bytes[4],
			(unsigned int) first_bytes[5], (unsigned int) first_bytes[6],
			(unsigned int) first_bytes[7], (unsigned int) first_bytes[8],
			(unsigned int) first_bytes[9], (unsigned int) first_bytes[10],
			(unsigned int) first_bytes[11], (unsigned int) first_bytes[12],
			(unsigned int) first_bytes[13], (unsigned int) first_bytes[14],
			(unsigned int) first_bytes[15]);
}

static const char *AppAI_BufferTypeName(const LL_Buffer_InfoTypeDef *buffer_info) {
	if (buffer_info == NULL) {
		return "(null)";
	}

	switch (buffer_info->type) {
	case DataType_FLOAT:
		return "FLOAT";
	case DataType_INT8:
		return "INT8";
	case DataType_UINT8:
		return "UINT8";
	case DataType_INT16:
		return "INT16";
	default:
		return "OTHER";
	}
}

static void AppAI_LogBufferInfoAndSignature(const char *label,
		const LL_Buffer_InfoTypeDef *buffer_info) {
	const void *buffer_addr = NULL;
	size_t buffer_len = 0U;
	float scale_value = 0.0f;
	int16_t offset_value = 0;
	const void *scale_addr = NULL;
	const void *offset_addr = NULL;

	if ((label == NULL) || (buffer_info == NULL)) {
		DebugConsole_Printf("[AI] %s buffer info unavailable.\r\n",
				(label != NULL) ? label : "(unnamed)");
		return;
	}

	buffer_addr = LL_Buffer_addr_start(buffer_info);
	buffer_len = (size_t) LL_Buffer_len(buffer_info);
	scale_addr = buffer_info->scale;
	offset_addr = buffer_info->offset;

	DebugConsole_Printf(
			"[AI] %s info: name=%s addr=%p len=%lu type=%s nbits=%u ndims=%u Qm=%u Qn=%u Qunsigned=%u epoch=%u batch=%u shape=[%lu,%lu,%lu,%lu]\r\n",
			label,
			(buffer_info->name != NULL) ? buffer_info->name : "(unnamed)",
			buffer_addr, (unsigned long) buffer_len,
			AppAI_BufferTypeName(buffer_info), (unsigned int) buffer_info->nbits,
			(unsigned int) buffer_info->ndims, (unsigned int) buffer_info->Qm,
			(unsigned int) buffer_info->Qn, (unsigned int) buffer_info->Qunsigned,
			(unsigned int) buffer_info->epoch, (unsigned int) buffer_info->batch,
			(unsigned long) ((buffer_info->shape != NULL) && (buffer_info->ndims > 0U) ? buffer_info->shape[0] : 0U),
			(unsigned long) ((buffer_info->shape != NULL) && (buffer_info->ndims > 1U) ? buffer_info->shape[1] : 0U),
			(unsigned long) ((buffer_info->shape != NULL) && (buffer_info->ndims > 2U) ? buffer_info->shape[2] : 0U),
			(unsigned long) ((buffer_info->shape != NULL) && (buffer_info->ndims > 3U) ? buffer_info->shape[3] : 0U));

	if ((scale_addr != NULL) && (offset_addr != NULL)) {
		(void) memcpy(&scale_value, scale_addr, sizeof(scale_value));
		offset_value = *(const int16_t *) offset_addr;

		DebugConsole_Printf("[AI] %s qparams: ", label);
		AppAI_LogFloatApprox("scale=", scale_value);
		DebugConsole_Printf(" offset=%d\r\n", (int) offset_value);
	}

	if (buffer_addr != NULL) {
		AppAI_LogRawBufferSignature(label, (const uint8_t *) buffer_addr,
				buffer_len);
	}
}

/**
 * @brief Print a tiny raw preview for the active tensor buffers.
 *
 * This is intentionally always-on and lightweight so we can see whether the
 * actual stage input and output buffers are changing, even when the verbose
 * tensor dumps stay disabled to protect ROM size.
 */
#endif /* APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS */

/**
 * @brief Print a tiny raw preview for the active tensor buffers.
 *
 * This is intentionally always-on and lightweight so we can see whether the
 * actual stage input and output buffers are changing, even when the verbose
 * tensor dumps stay disabled to protect ROM size.
 *
 * The hash and middle/tail windows help us avoid a false "everything is zero"
 * conclusion when the tensor origin is padded or naturally blank.
 */
static void AppAI_LogBufferPreview(const char *label,
		const LL_Buffer_InfoTypeDef *buffer_info) {
	const uint8_t *buffer_bytes = NULL;
	size_t buffer_len = 0U;
	uint8_t first_bytes[8U] = { 0U };
	uint8_t mid_bytes[8U] = { 0U };
	uint8_t last_bytes[8U] = { 0U };
	uint32_t hash = 2166136261UL;
	size_t preview_len = 0U;
	size_t mid_offset = 0U;
	size_t last_offset = 0U;

	if ((label == NULL) || (buffer_info == NULL)) {
		return;
	}

	buffer_bytes = (const uint8_t *) LL_Buffer_addr_start(buffer_info);
	buffer_len = (size_t) LL_Buffer_len(buffer_info);
	if ((buffer_bytes == NULL) || (buffer_len == 0U)) {
		DebugConsole_Printf("[AI] %s probe skipped: empty buffer.\r\n", label);
		return;
	}

	preview_len = (buffer_len < sizeof(first_bytes)) ? buffer_len
			: sizeof(first_bytes);
	mid_offset = (buffer_len > preview_len) ? (buffer_len / 2U) : 0U;
	if ((mid_offset + preview_len) > buffer_len) {
		mid_offset = buffer_len - preview_len;
	}
	last_offset = (buffer_len > preview_len) ? (buffer_len - preview_len) : 0U;

	for (size_t index = 0U; index < preview_len; ++index) {
		first_bytes[index] = buffer_bytes[index];
		mid_bytes[index] = buffer_bytes[mid_offset + index];
		last_bytes[index] = buffer_bytes[last_offset + index];
	}

	for (size_t index = 0U; index < buffer_len; ++index) {
		hash ^= buffer_bytes[index];
		hash *= 16777619UL;
	}

	DebugConsole_Printf(
			"[AI] %s probe: name=%s addr=%p len=%lu hash=0x%08lX first8=[%02X %02X %02X %02X %02X %02X %02X %02X] mid8=[%02X %02X %02X %02X %02X %02X %02X %02X] last8=[%02X %02X %02X %02X %02X %02X %02X %02X]\r\n",
			label,
			(buffer_info->name != NULL) ? buffer_info->name : "(unnamed)",
			(const void *) buffer_bytes, (unsigned long) buffer_len,
			(unsigned long) hash,
			(unsigned int) first_bytes[0], (unsigned int) first_bytes[1],
			(unsigned int) first_bytes[2], (unsigned int) first_bytes[3],
			(unsigned int) first_bytes[4], (unsigned int) first_bytes[5],
			(unsigned int) first_bytes[6], (unsigned int) first_bytes[7],
			(unsigned int) mid_bytes[0], (unsigned int) mid_bytes[1],
			(unsigned int) mid_bytes[2], (unsigned int) mid_bytes[3],
			(unsigned int) mid_bytes[4], (unsigned int) mid_bytes[5],
			(unsigned int) mid_bytes[6], (unsigned int) mid_bytes[7],
			(unsigned int) last_bytes[0], (unsigned int) last_bytes[1],
			(unsigned int) last_bytes[2], (unsigned int) last_bytes[3],
			(unsigned int) last_bytes[4], (unsigned int) last_bytes[5],
			(unsigned int) last_bytes[6], (unsigned int) last_bytes[7]);
}

/**
 * @brief Estimate a gauge crop from bright pixels in the Y channel.
 *
 * The dial face is usually the brightest contiguous object in these captures,
 * so a simple luma threshold is enough to steer the crop toward the useful
 * region before we resize to the model input tensor.
 */
static bool AppAI_EstimateGaugeCropBoxFromYuv422(const uint8_t *frame_bytes,
		size_t frame_size, size_t frame_width_pixels, size_t frame_height_pixels,
		size_t *crop_x_min, size_t *crop_y_min, size_t *crop_width,
		size_t *crop_height) {
	const size_t frame_stride_bytes = frame_width_pixels * 2U;
	const size_t min_crop_width = frame_width_pixels / 4U;
	const size_t min_crop_height = frame_height_pixels / 4U;
	const size_t training_crop_width = (size_t) (((float) frame_width_pixels
			* (APP_AI_TRAINING_CROP_X_MAX_RATIO
			- APP_AI_TRAINING_CROP_X_MIN_RATIO)) + 0.5f);
	const size_t training_crop_height = (size_t) (((float) frame_height_pixels
			* (APP_AI_TRAINING_CROP_Y_MAX_RATIO
			- APP_AI_TRAINING_CROP_Y_MIN_RATIO)) + 0.5f);
	size_t bright_x_min = frame_width_pixels;
	size_t bright_y_min = frame_height_pixels;
	size_t bright_x_max = 0U;
	size_t bright_y_max = 0U;
	size_t bright_count = 0U;
	uint64_t bright_sum_x = 0U;
	uint64_t bright_sum_y = 0U;
	size_t bbox_width = 0U;
	size_t bbox_height = 0U;
	size_t bright_center_x = 0U;
	size_t bright_center_y = 0U;
	size_t biased_center_x = 0U;
	size_t biased_center_y = 0U;
	size_t left = 0U;
	size_t top = 0U;
	size_t right = 0U;
	size_t bottom = 0U;

	if ((frame_bytes == NULL) || (crop_x_min == NULL) || (crop_y_min == NULL)
			|| (crop_width == NULL) || (crop_height == NULL)) {
		return false;
	}

	if (frame_size < (frame_stride_bytes * frame_height_pixels)) {
		return false;
	}

	for (size_t y = APP_AI_GAUGE_CROP_BORDER_PIXELS;
			y < (frame_height_pixels - APP_AI_GAUGE_CROP_BORDER_PIXELS); ++y) {
		const size_t row_offset = y * frame_stride_bytes;

		for (size_t x = APP_AI_GAUGE_CROP_BORDER_PIXELS;
				x < (frame_width_pixels - APP_AI_GAUGE_CROP_BORDER_PIXELS); ++x) {
			const size_t pair_offset = row_offset + ((x & ~1U) * 2U);
			const size_t y_offset = pair_offset + (((x & 1U) != 0U) ? 2U : 0U);
			const uint8_t luma = frame_bytes[y_offset];

			if (luma < APP_AI_GAUGE_BRIGHT_THRESHOLD) {
				continue;
			}

			bright_count++;
			bright_sum_x += (uint64_t) x;
			bright_sum_y += (uint64_t) y;

			if (x < bright_x_min) {
				bright_x_min = x;
			}
			if (y < bright_y_min) {
				bright_y_min = y;
			}
			if (x > bright_x_max) {
				bright_x_max = x;
			}
			if (y > bright_y_max) {
				bright_y_max = y;
			}
		}
	}

	if (bright_count == 0U) {
		return false;
	}

	bright_center_x = (size_t) (bright_sum_x / (uint64_t) bright_count);
	bright_center_y = (size_t) (bright_sum_y / (uint64_t) bright_count);
	bbox_width = (bright_x_max - bright_x_min) + 1U;
	bbox_height = (bright_y_max - bright_y_min) + 1U;
	if ((bbox_width == 0U) || (bbox_height == 0U)) {
		return false;
	}

	biased_center_x = (bright_center_x > APP_AI_GAUGE_CROP_CENTER_X_BIAS_PIXELS) ?
			(bright_center_x - APP_AI_GAUGE_CROP_CENTER_X_BIAS_PIXELS) : 0U;
	biased_center_y = (bright_center_y > APP_AI_GAUGE_CROP_CENTER_Y_BIAS_PIXELS) ?
			(bright_center_y - APP_AI_GAUGE_CROP_CENTER_Y_BIAS_PIXELS) : 0U;

	/* Anchor a slightly tighter crop on a left-biased bright centroid instead of
	 * using the full bright bbox, which was pulling in too much background. */
	if (training_crop_width == 0U) {
		return false;
	}
	if (training_crop_height == 0U) {
		return false;
	}

	const size_t target_crop_width = (size_t) ((((float) training_crop_width
			* (float) APP_AI_GAUGE_CROP_WIDTH_SCALE_NUMERATOR)
			/ (float) APP_AI_GAUGE_CROP_WIDTH_SCALE_DENOMINATOR) + 0.5f);
	const size_t target_crop_height = (size_t) ((((float) training_crop_height
			* (float) APP_AI_GAUGE_CROP_HEIGHT_SCALE_NUMERATOR)
			/ (float) APP_AI_GAUGE_CROP_HEIGHT_SCALE_DENOMINATOR) + 0.5f);
	const size_t crop_width_pixels =
			(target_crop_width > 0U) ? target_crop_width : 1U;
	const size_t crop_height_pixels =
			(target_crop_height > 0U) ? target_crop_height : 1U;

	left = (biased_center_x > (crop_width_pixels / 2U)) ?
			(biased_center_x - (crop_width_pixels / 2U)) : 0U;
	top = (biased_center_y > (crop_height_pixels / 2U)) ?
			(biased_center_y - (crop_height_pixels / 2U)) : 0U;
	right = left + crop_width_pixels;
	bottom = top + crop_height_pixels;
	if (right > frame_width_pixels) {
		right = frame_width_pixels;
		left = (right > crop_width_pixels) ? (right - crop_width_pixels)
				: 0U;
	}
	if (bottom > frame_height_pixels) {
		bottom = frame_height_pixels;
		top = (bottom > crop_height_pixels) ? (bottom - crop_height_pixels)
				: 0U;
	}

	if ((right <= left) || (bottom <= top)) {
		return false;
	}

	*crop_x_min = left;
	*crop_y_min = top;
	*crop_width = right - left;
	*crop_height = bottom - top;

	if ((*crop_width < min_crop_width) || (*crop_height < min_crop_height)) {
		return false;
	}

	return true;
}

static bool AppAI_LogXspi2ModelFilePrefix(FX_FILE *model_file_ptr) {
	uint8_t source_bytes[APP_AI_XSPI2_PROBE_BYTES] = { 0U };
	ULONG bytes_read = 0U;
	UINT fx_status = FX_SUCCESS;

	if (model_file_ptr == NULL) {
		return false;
	}

	fx_status = fx_file_read(model_file_ptr, source_bytes,
			APP_AI_XSPI2_PROBE_BYTES, &bytes_read);
	if ((fx_status != FX_SUCCESS)
			|| (bytes_read != APP_AI_XSPI2_PROBE_BYTES)) {
		DebugConsole_Printf(
				"[AI] xSPI2 source prefix read failed (fx=%lu n=%lu).\r\n",
				(unsigned long) fx_status, (unsigned long) bytes_read);
		return false;
	}

	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		AppAI_LogXspi2PrefixBytes("xSPI2 source prefix:", source_bytes);
	}

	fx_status = fx_file_seek(model_file_ptr, 0U);
	if (fx_status != FX_SUCCESS) {
		DebugConsole_Printf(
				"[AI] xSPI2 source rewind failed (fx=%lu).\r\n",
				(unsigned long) fx_status);
		return false;
	}

	return true;
}

#if 0
static bool AppAI_Xspi2ModelImageMatchesFlash(void) {
	if (!AppAI_Xspi2ReadFlashProbe(APP_AI_XSPI2_SCALAR_CHIP_OFFSET, 0U,
			app_ai_xspi2_signature_start,
			sizeof(app_ai_xspi2_signature_start))) {
		DebugConsole_Printf("[AI] xSPI2 verify failed at start signature.\r\n");
		return false;
	}

	if ((app_ai_xspi2_programmed_size >= APP_AI_XSPI2_PROBE_BYTES)
			&& !AppAI_Xspi2ReadFlashProbe(APP_AI_XSPI2_SCALAR_CHIP_OFFSET,
					app_ai_xspi2_programmed_size - APP_AI_XSPI2_PROBE_BYTES,
					app_ai_xspi2_signature_tail,
					sizeof(app_ai_xspi2_signature_tail))) {
		DebugConsole_Printf("[AI] xSPI2 verify failed at tail signature.\r\n");
		return false;
	}

	return true;
}
#endif /* 0 */

static bool AppAI_Xspi2ModelImageMatchesMappedFlash(void) {
	if (!AppAI_Xspi2ReadMappedProbe(0U, app_ai_xspi2_signature_start,
			sizeof(app_ai_xspi2_signature_start))) {
		DebugConsole_Printf(
				"[AI] xSPI2 mapped verify failed at start signature.\r\n");
		return false;
	}

	if ((app_ai_xspi2_programmed_size >= APP_AI_XSPI2_PROBE_BYTES)
			&& !AppAI_Xspi2ReadMappedProbe(
					app_ai_xspi2_programmed_size - APP_AI_XSPI2_PROBE_BYTES,
					app_ai_xspi2_signature_tail,
					sizeof(app_ai_xspi2_signature_tail))) {
		DebugConsole_Printf(
				"[AI] xSPI2 mapped verify failed at tail signature.\r\n");
		return false;
	}

	return true;
}

static bool AppAI_Xspi2ModelImageMatchesMappedFlashForStage(
		const AppAI_ModelStageSpec *stage) {
	const char *stage_label = NULL;
	bool is_rectifier;
	bool is_obb;
	const uint8_t *sig_start;
	const uint8_t *sig_tail;
	bool sig_valid;
	ULONG programmed_size;

	if (stage == NULL) {
		return false;
	}

	stage_label = (stage->stage_label != NULL) ? stage->stage_label : "scalar";
	is_rectifier = (strcmp(stage_label, "rectifier") == 0);
	is_obb = (strcmp(stage_label, "obb") == 0);

	if (is_rectifier) {
		/* Prefer the SD-sourced signature cached during the last provisioning.
		 * Fall back to the hardcoded constant only when no provisioning has
		 * happened yet this boot (cache is zeroed at startup). */
		sig_start      = app_ai_rectifier_sig_valid
				? app_ai_rectifier_sig_start
				: app_ai_rectifier_xspi2_signature_start;
		sig_tail       = app_ai_rectifier_sig_valid
				? app_ai_rectifier_sig_tail
				: app_ai_rectifier_xspi2_signature_tail;
		sig_valid      = app_ai_rectifier_sig_valid;
		programmed_size = app_ai_rectifier_programmed_size;
	} else if (is_obb) {
		sig_start      = app_ai_obb_sig_valid
				? app_ai_obb_sig_start
				: app_ai_obb_xspi2_signature_start;
		sig_tail       = app_ai_obb_sig_valid
				? app_ai_obb_sig_tail
				: app_ai_obb_xspi2_signature_tail;
		sig_valid      = app_ai_obb_sig_valid;
		programmed_size = app_ai_obb_programmed_size;
	} else {
		sig_start      = app_ai_scalar_sig_valid
				? app_ai_scalar_sig_start
				: app_ai_xspi2_signature_start;
		sig_tail       = app_ai_scalar_sig_valid
				? app_ai_scalar_sig_tail
				: app_ai_xspi2_signature_tail;
		sig_valid      = app_ai_scalar_sig_valid;
		programmed_size = app_ai_scalar_programmed_size;
	}

	if (!AppAI_Xspi2ReadFlashProbe(stage->xspi2_chip_offset, 0U,
			sig_start, APP_AI_XSPI2_PROBE_BYTES)) {
		/* Log the actual flash bytes so stale hardcoded signatures can be updated. */
		{
			uint8_t actual[APP_AI_XSPI2_PROBE_BYTES] = { 0U };
			char vlog[160];
			int32_t read_status = BSP_XSPI_NOR_Read(0U, actual,
					stage->xspi2_chip_offset, APP_AI_XSPI2_PROBE_BYTES);
			(void) snprintf(vlog, sizeof(vlog),
					"[AI] %s mismatch rc=%ld bytes=[%02X%02X%02X%02X%02X%02X%02X%02X]\r\n",
					stage->stage_label, (long) read_status,
					actual[0],actual[1],actual[2],actual[3],
					actual[4],actual[5],actual[6],actual[7]);
			DebugConsole_WriteString(vlog);
		}
		return false;
	}

	if (sig_valid && (programmed_size >= APP_AI_XSPI2_PROBE_BYTES)
			&& !AppAI_Xspi2ReadFlashProbe(stage->xspi2_chip_offset,
					programmed_size - APP_AI_XSPI2_PROBE_BYTES,
					sig_tail, APP_AI_XSPI2_PROBE_BYTES)) {
		{
			char vlog[64];
#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
			(void) snprintf(vlog, sizeof(vlog),
					"[AI] %s flash verify: tail mismatch.\r\n",
					stage->stage_label);
			(void) DebugConsole_WriteString(vlog);
#endif
		}
		return false;
	}

	return true;
}

static bool AppAI_ProgramXspi2ModelImageFromSd(void) {
	FX_MEDIA *media_ptr = NULL;
	FX_FILE model_file = { 0 };
	ULONG file_size = 0U;
	ULONG bytes_read = 0U;
	ULONG bytes_remaining = 0U;
	ULONG flash_offset = 0U;
	UINT fx_status = FX_SUCCESS;
	UINT tx_status = TX_SUCCESS;
	int32_t bsp_status = BSP_ERROR_NONE;

	if (!AppAI_WaitForFileXMediaReady(APP_AI_FILEX_MEDIA_READY_TIMEOUT_MS)) {
		AppAI_LogXspi2LoadFailure("FileX not ready", FX_MEDIA_NOT_OPEN,
				BSP_ERROR_NONE);
		return false;
	}

	tx_status = AppFileX_AcquireMediaLock();
	if (tx_status != TX_SUCCESS) {
		AppAI_LogXspi2LoadFailure("media lock", (UINT) tx_status, BSP_ERROR_NONE);
		return false;
	}

	media_ptr = AppFileX_GetMediaHandle();
	if (media_ptr == NULL) {
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("media handle", FX_MEDIA_NOT_OPEN,
				BSP_ERROR_NONE);
		return false;
	}

	if (fx_directory_default_set(media_ptr, FX_NULL) != FX_SUCCESS) {
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("default directory", FX_SUCCESS,
				BSP_ERROR_NONE);
		return false;
	}

	fx_status = fx_file_open(media_ptr, &model_file,
			(CHAR *) APP_AI_XSPI2_MODEL_IMAGE_PATH, FX_OPEN_FOR_READ);
	if (fx_status != FX_SUCCESS) {
		(void) fx_directory_default_set(media_ptr, FX_NULL);
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("file open", fx_status, BSP_ERROR_NONE);
		return false;
	}

	file_size = model_file.fx_file_current_file_size;
	if (file_size == 0U) {
		(void) fx_file_close(&model_file);
		(void) fx_directory_default_set(media_ptr, FX_NULL);
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("file size (empty)", FX_SUCCESS, BSP_ERROR_NONE);
		return false;
	}
	DebugConsole_Printf("[AI] Model file size: %lu bytes.\r\n",
			(unsigned long) file_size);

	fx_status = fx_file_seek(&model_file, 0U);
	if (fx_status != FX_SUCCESS) {
		(void) fx_file_close(&model_file);
		(void) fx_directory_default_set(media_ptr, FX_NULL);
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("file seek", fx_status, BSP_ERROR_NONE);
		return false;
	}

	if (!AppAI_LogXspi2ModelFilePrefix(&model_file)) {
		(void) fx_file_close(&model_file);
		(void) fx_directory_default_set(media_ptr, FX_NULL);
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("source prefix", FX_SUCCESS,
				BSP_ERROR_COMPONENT_FAILURE);
		return false;
	}

	for (ULONG erase_addr = 0U; erase_addr < file_size;
			erase_addr += APP_AI_XSPI2_ERASE_BLOCK_BYTES) {
		if (erase_addr == 0U) {
			(void) DebugConsole_WriteString("[AI] xSPI2 stage erase begin.\r\n");
		}
		bsp_status = BSP_XSPI_NOR_Erase_Block(0U,
				APP_AI_XSPI2_MODEL_CHIP_OFFSET + erase_addr,
				BSP_XSPI_NOR_ERASE_64K);
		if (bsp_status != BSP_ERROR_NONE) {
			(void) fx_file_close(&model_file);
			(void) fx_directory_default_set(media_ptr, FX_NULL);
			AppFileX_ReleaseMediaLock();
			AppAI_LogXspi2LoadFailure("flash erase", FX_SUCCESS, bsp_status);
			return false;
		}
	}

	bytes_remaining = file_size;
	flash_offset = 0U;
	{
		ULONG chunk_index = 0U;

	while (bytes_remaining > 0U) {
		const ULONG chunk_size = (bytes_remaining > APP_AI_XSPI2_PROGRAM_CHUNK_BYTES)
				? APP_AI_XSPI2_PROGRAM_CHUNK_BYTES
				: bytes_remaining;

		if ((flash_offset == 0U)
				|| ((flash_offset % APP_AI_XSPI2_ERASE_BLOCK_BYTES) == 0U)) {
			AppAI_LogXspi2ProgramChunkProgress(chunk_index, flash_offset,
					chunk_size);
		}

		bytes_read = 0U;
		fx_status = fx_file_read(&model_file, app_ai_xspi2_program_buffer,
				chunk_size, &bytes_read);
		if ((fx_status != FX_SUCCESS) || (bytes_read != chunk_size)) {
			(void) fx_file_close(&model_file);
			(void) fx_directory_default_set(media_ptr, FX_NULL);
			AppFileX_ReleaseMediaLock();
			AppAI_LogXspi2LoadFailure("file read", fx_status, BSP_ERROR_NONE);
			return false;
		}

		/* Keep the flash writer honest: the staging buffer is cacheable RAM, so
		 * clean it before BSP_XSPI_NOR_Write() consumes the bytes. */
		(void) mcu_cache_clean_range((uint32_t) (uintptr_t) app_ai_xspi2_program_buffer,
				(uint32_t) (uintptr_t) app_ai_xspi2_program_buffer
						+ (uint32_t) chunk_size);

		bsp_status = BSP_XSPI_NOR_Write(0U, app_ai_xspi2_program_buffer,
				APP_AI_XSPI2_MODEL_CHIP_OFFSET + flash_offset,
				(uint32_t) chunk_size);
		if (bsp_status != BSP_ERROR_NONE) {
			(void) fx_file_close(&model_file);
			(void) fx_directory_default_set(media_ptr, FX_NULL);
			AppFileX_ReleaseMediaLock();
			AppAI_LogXspi2LoadFailure("flash write", FX_SUCCESS, bsp_status);
			return false;
		}

		flash_offset += chunk_size;
		bytes_remaining -= chunk_size;
		chunk_index++;
	}
	}

	(void) fx_file_close(&model_file);
	(void) fx_directory_default_set(media_ptr, FX_NULL);
	AppFileX_ReleaseMediaLock();

	app_ai_xspi2_programmed_size = file_size;
	AppAI_LogXspi2FlashStatus("legacy stage write complete");

	if (!AppAI_ReconfigureXspi2ForRuntime()) {
		AppAI_LogXspi2LoadFailure("runtime reconfigure", FX_SUCCESS,
				BSP_ERROR_COMPONENT_FAILURE);
		return false;
	}
	if (!AppAI_Xspi2EnableMemoryMappedMode()) {
		AppAI_LogXspi2LoadFailure("enable MM after provision", FX_SUCCESS,
				BSP_ERROR_COMPONENT_FAILURE);
		return false;
	}

	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		AppAI_LogXspi2IndirectAndMappedPrefix();
		AppAI_LogXspi2MappedScaleBytes();
	}

	return true;
}

static void AppAI_SetForcedCrop(const char *label, size_t x_min,
		size_t y_min, size_t width, size_t height) {
	app_ai_forced_crop_active = true;
	app_ai_forced_crop_label = label;
	app_ai_forced_crop_x_min = x_min;
	app_ai_forced_crop_y_min = y_min;
	app_ai_forced_crop_width = width;
	app_ai_forced_crop_height = height;
}

static void AppAI_ClearForcedCrop(void) {
	app_ai_forced_crop_active = false;
	app_ai_forced_crop_label = NULL;
	app_ai_forced_crop_x_min = 0U;
	app_ai_forced_crop_y_min = 0U;
	app_ai_forced_crop_width = 0U;
	app_ai_forced_crop_height = 0U;
}

static const LL_Buffer_InfoTypeDef *AppAI_GetStageInputBufferInfo(
		const AppAI_ModelStageSpec *stage) {
	const LL_Buffer_InfoTypeDef *input_info = NULL;

	if ((stage == NULL) || (stage->nn_instance == NULL)
			|| (stage->nn_instance->network == NULL)
			|| (stage->nn_instance->network->input_buffers_info == NULL)) {
		return NULL;
	}

	input_info = stage->nn_instance->network->input_buffers_info();
	if ((input_info == NULL) || (input_info->name == NULL)) {
		return NULL;
	}
	return input_info;
}

static const LL_Buffer_InfoTypeDef *AppAI_GetStageOutputBufferInfo(
		const AppAI_ModelStageSpec *stage) {
	const LL_Buffer_InfoTypeDef *output_info = NULL;

	if ((stage == NULL) || (stage->nn_instance == NULL)
			|| (stage->nn_instance->network == NULL)
			|| (stage->nn_instance->network->output_buffers_info == NULL)) {
		return NULL;
	}

	output_info = stage->nn_instance->network->output_buffers_info();
	if ((output_info == NULL) || (output_info->name == NULL)) {
		return NULL;
	}
	return output_info;
}

static bool AppAI_ReadXspi2ModelSourceProbes(FX_FILE *model_file_ptr,
		ULONG file_size, uint8_t *source_prefix, uint8_t *source_tail,
		bool *has_tail_out) {
	ULONG bytes_read = 0U;
	UINT fx_status = FX_SUCCESS;

	if ((model_file_ptr == NULL) || (source_prefix == NULL)
			|| (source_tail == NULL) || (has_tail_out == NULL)) {
		return false;
	}

	*has_tail_out = false;

	fx_status = fx_file_seek(model_file_ptr, 0U);
	if (fx_status != FX_SUCCESS) {
		return false;
	}

	bytes_read = 0U;
	fx_status = fx_file_read(model_file_ptr, source_prefix,
			APP_AI_XSPI2_PROBE_BYTES, &bytes_read);
	if ((fx_status != FX_SUCCESS)
			|| (bytes_read != APP_AI_XSPI2_PROBE_BYTES)) {
		return false;
	}
	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		AppAI_LogXspi2PrefixBytes("xSPI2 source prefix:", source_prefix);
	}

	if (file_size >= APP_AI_XSPI2_PROBE_BYTES) {
		fx_status = fx_file_seek(model_file_ptr,
				file_size - APP_AI_XSPI2_PROBE_BYTES);
		if (fx_status != FX_SUCCESS) {
			return false;
		}

		bytes_read = 0U;
		fx_status = fx_file_read(model_file_ptr, source_tail,
				APP_AI_XSPI2_PROBE_BYTES, &bytes_read);
		if ((fx_status != FX_SUCCESS)
				|| (bytes_read != APP_AI_XSPI2_PROBE_BYTES)) {
			return false;
		}
		if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
			AppAI_LogXspi2PrefixBytes("xSPI2 source tail:", source_tail);
		}
		*has_tail_out = true;
	}

	fx_status = fx_file_seek(model_file_ptr, 0U);
	if (fx_status != FX_SUCCESS) {
		return false;
	}

	return true;
}

#if 0
static bool AppAI_ProgramXspi2ModelImageFromSdForStage(
		const AppAI_ModelStageSpec *stage) {
	FX_MEDIA *media_ptr = NULL;
	FX_FILE model_file = { 0 };
	ULONG file_size = 0U;
	ULONG bytes_read = 0U;
	ULONG bytes_remaining = 0U;
	ULONG flash_offset = 0U;
	UINT fx_status = FX_SUCCESS;
	UINT tx_status = TX_SUCCESS;
	int32_t bsp_status = BSP_ERROR_NONE;
	uint8_t source_prefix[APP_AI_XSPI2_PROBE_BYTES] = { 0U };
	uint8_t source_tail[APP_AI_XSPI2_PROBE_BYTES] = { 0U };
	bool has_tail_probe = false;

	if ((stage == NULL) || (stage->model_image_path == NULL)) {
		return false;
	}

	if (!AppAI_WaitForFileXMediaReady(APP_AI_FILEX_MEDIA_READY_TIMEOUT_MS)) {
		AppAI_LogXspi2LoadFailure(stage->stage_label, FX_MEDIA_NOT_OPEN,
				BSP_ERROR_NONE);
		return false;
	}

	(void) DebugConsole_WriteString("[AI] xSPI2 stage acquire media lock start.\r\n");
	tx_status = AppFileX_AcquireMediaLock();
	if (tx_status != TX_SUCCESS) {
		AppAI_LogXspi2LoadFailure(stage->stage_label, (UINT) tx_status,
				BSP_ERROR_NONE);
		return false;
	}
	(void) DebugConsole_WriteString("[AI] xSPI2 stage acquire media lock OK.\r\n");

	media_ptr = AppFileX_GetMediaHandle();
	if (media_ptr == NULL) {
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure(stage->stage_label, FX_MEDIA_NOT_OPEN,
				BSP_ERROR_NONE);
		return false;
	}

	(void) DebugConsole_WriteString("[AI] xSPI2 stage media handle OK.\r\n");
	if (fx_directory_default_set(media_ptr, FX_NULL) != FX_SUCCESS) {
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure(stage->stage_label, FX_SUCCESS,
				BSP_ERROR_NONE);
		return false;
	}
	(void) DebugConsole_WriteString("[AI] xSPI2 stage directory reset OK.\r\n");

	(void) DebugConsole_WriteString("[AI] xSPI2 stage file open start.\r\n");
	fx_status = fx_file_open(media_ptr, &model_file,
			(CHAR *) stage->model_image_path, FX_OPEN_FOR_READ);
	if (fx_status != FX_SUCCESS) {
		(void) fx_directory_default_set(media_ptr, FX_NULL);
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure(stage->stage_label, fx_status, BSP_ERROR_NONE);
		return false;
	}
	(void) DebugConsole_WriteString("[AI] xSPI2 stage file open OK.\r\n");

	file_size = model_file.fx_file_current_file_size;
	if (file_size == 0U) {
		(void) fx_file_close(&model_file);
		(void) fx_directory_default_set(media_ptr, FX_NULL);
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure(stage->stage_label, FX_SUCCESS,
				BSP_ERROR_NONE);
		return false;
	}
	DebugConsole_Printf("[AI] %s model file size: %lu bytes.\r\n",
			stage->stage_label, (unsigned long) file_size);

	(void) DebugConsole_WriteString("[AI] xSPI2 stage source probes start.\r\n");
	if (!AppAI_ReadXspi2ModelSourceProbes(&model_file, file_size,
			source_prefix, source_tail, &has_tail_probe)) {
		(void) fx_file_close(&model_file);
		(void) fx_directory_default_set(media_ptr, FX_NULL);
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure(stage->stage_label, FX_SUCCESS,
				BSP_ERROR_COMPONENT_FAILURE);
		return false;
	}
	(void) DebugConsole_WriteString("[AI] xSPI2 stage source probes OK.\r\n");

	for (ULONG erase_addr = 0U; erase_addr < file_size;
			erase_addr += APP_AI_XSPI2_ERASE_BLOCK_BYTES) {
		bsp_status = BSP_XSPI_NOR_Erase_Block(0U,
				stage->xspi2_chip_offset + erase_addr,
				BSP_XSPI_NOR_ERASE_64K);
		if (bsp_status != BSP_ERROR_NONE) {
			(void) fx_file_close(&model_file);
			(void) fx_directory_default_set(media_ptr, FX_NULL);
			AppFileX_ReleaseMediaLock();
			AppAI_LogXspi2LoadFailure(stage->stage_label, FX_SUCCESS,
					bsp_status);
			return false;
		}
	}

	bytes_remaining = file_size;
	flash_offset = 0U;
	{
		ULONG chunk_index = 0U;
		(void) DebugConsole_WriteString("[AI] xSPI2 stage write begin.\r\n");

		while (bytes_remaining > 0U) {
			const ULONG chunk_size = (bytes_remaining > APP_AI_XSPI2_PROGRAM_CHUNK_BYTES)
					? APP_AI_XSPI2_PROGRAM_CHUNK_BYTES
					: bytes_remaining;

			if (chunk_index == 0U) {
				(void) DebugConsole_WriteString(
						"[AI] xSPI2 stage first chunk write start.\r\n");
			}
			if ((flash_offset == 0U)
					|| ((flash_offset % APP_AI_XSPI2_ERASE_BLOCK_BYTES) == 0U)) {
				AppAI_LogXspi2ProgramChunkProgress(chunk_index, flash_offset,
						chunk_size);
			}

			bytes_read = 0U;
			fx_status = fx_file_read(&model_file, app_ai_xspi2_program_buffer,
					chunk_size, &bytes_read);
			if ((fx_status != FX_SUCCESS) || (bytes_read != chunk_size)) {
				(void) fx_file_close(&model_file);
				(void) fx_directory_default_set(media_ptr, FX_NULL);
				AppFileX_ReleaseMediaLock();
				AppAI_LogXspi2LoadFailure(stage->stage_label, fx_status,
						BSP_ERROR_NONE);
				return false;
			}

			/* Keep the flash writer honest: clean the staging buffer cache lines
			 * before BSP_XSPI_NOR_Write() reads the fresh file contents. */
			(void) mcu_cache_clean_range((uint32_t) (uintptr_t) app_ai_xspi2_program_buffer,
					(uint32_t) (uintptr_t) app_ai_xspi2_program_buffer
							+ (uint32_t) chunk_size);

			bsp_status = BSP_XSPI_NOR_Write(0U, app_ai_xspi2_program_buffer,
					stage->xspi2_chip_offset + flash_offset,
					(uint32_t) chunk_size);
			if (bsp_status != BSP_ERROR_NONE) {
				(void) fx_file_close(&model_file);
				(void) fx_directory_default_set(media_ptr, FX_NULL);
				AppFileX_ReleaseMediaLock();
				AppAI_LogXspi2LoadFailure(stage->stage_label, FX_SUCCESS,
						bsp_status);
				return false;
			}

			flash_offset += chunk_size;
			bytes_remaining -= chunk_size;
			chunk_index++;
		}
	}
	(void) DebugConsole_WriteString("[AI] xSPI2 stage write complete.\r\n");
	AppAI_LogXspi2FlashStatus("rectifier stage write complete");

	/* Probe the first 16 bytes back from flash immediately after write (still in
	 * indirect/write mode) to confirm the data landed at the expected address. */
	{
		uint8_t post_write[APP_AI_XSPI2_PROBE_BYTES] = { 0U };
		char pwlog[128];
		(void) BSP_XSPI_NOR_Read(0U, post_write,
				stage->xspi2_chip_offset, APP_AI_XSPI2_PROBE_BYTES);
#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
		(void) snprintf(pwlog, sizeof(pwlog),
				"[AI] %s post-write probe: [%02X%02X%02X%02X%02X%02X%02X%02X"
				"%02X%02X%02X%02X%02X%02X%02X%02X] "
				"src=[%02X%02X%02X%02X%02X%02X%02X%02X"
				"%02X%02X%02X%02X%02X%02X%02X%02X]\r\n",
				stage->stage_label,
				post_write[0],post_write[1],post_write[2],post_write[3],
				post_write[4],post_write[5],post_write[6],post_write[7],
				post_write[8],post_write[9],post_write[10],post_write[11],
				post_write[12],post_write[13],post_write[14],post_write[15],
				source_prefix[0],source_prefix[1],source_prefix[2],source_prefix[3],
				source_prefix[4],source_prefix[5],source_prefix[6],source_prefix[7],
				source_prefix[8],source_prefix[9],source_prefix[10],source_prefix[11],
				source_prefix[12],source_prefix[13],source_prefix[14],source_prefix[15]);
		(void) DebugConsole_WriteString(pwlog);
#endif
	}

	(void) fx_file_close(&model_file);
	(void) fx_directory_default_set(media_ptr, FX_NULL);
	AppFileX_ReleaseMediaLock();

	app_ai_xspi2_programmed_size = file_size;
	/* Update the per-stage size so the verify tail-probe uses the right offset
	 * regardless of which stage was provisioned last. */
	if (strcmp(stage->stage_label, "rectifier") == 0) {
		app_ai_rectifier_programmed_size = file_size;
		(void) memcpy(app_ai_rectifier_sig_start, source_prefix,
				APP_AI_XSPI2_PROBE_BYTES);
		(void) memcpy(app_ai_rectifier_sig_tail, source_tail,
				APP_AI_XSPI2_PROBE_BYTES);
		app_ai_rectifier_sig_valid = has_tail_probe;
	} else {
		app_ai_scalar_programmed_size = file_size;
		(void) memcpy(app_ai_scalar_sig_start, source_prefix,
				APP_AI_XSPI2_PROBE_BYTES);
		(void) memcpy(app_ai_scalar_sig_tail, source_tail,
				APP_AI_XSPI2_PROBE_BYTES);
		app_ai_scalar_sig_valid = has_tail_probe;
	}

	if (!AppAI_ReconfigureXspi2ForRuntime()) {
		AppAI_LogXspi2LoadFailure(stage->stage_label, FX_SUCCESS,
				BSP_ERROR_COMPONENT_FAILURE);
		return false;
	}
	if (!AppAI_Xspi2EnableMemoryMappedMode()) {
		AppAI_LogXspi2LoadFailure("enable MM after provision", FX_SUCCESS,
				BSP_ERROR_COMPONENT_FAILURE);
		return false;
	}

	(void) DebugConsole_WriteString(
			"[AI] xSPI2 stage provisioning complete; verify skipped.\r\n");
	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		AppAI_LogXspi2IndirectAndMappedPrefix();
		AppAI_LogXspi2MappedScaleBytes();
	}
	app_ai_loaded_xspi2_stage = stage;
	DebugConsole_Printf("[AI] %s xSPI2 model image ready.\r\n",
			stage->stage_label);
	return true;
}
#endif /* 0 */

static bool AppAI_EnsureXspi2ModelImageReadyForStage(
		const AppAI_ModelStageSpec *stage) {
	if (stage == NULL) {
		return false;
	}

	/* Fast path: pointer equality means this stage is already in its flash
	 * region and the peripheral is already in mapped mode for it. */
	if (app_ai_loaded_xspi2_stage == stage) {
		(void) DebugConsole_WriteString("[AI] xSPI2 stage already loaded.\r\n");
		return true;
	}

	/* Each stage now has its own flash region, so we can verify whether the
	 * stage's bytes are already present without disturbing the other stage.
	 * Switch to indirect mode first (needed for BSP_XSPI_NOR_Read). */
	DebugConsole_WriteString("[AI] xSPI2 stage reconfigure start.\r\n");
	if (!AppAI_ReconfigureXspi2ForRuntime()) {
		DebugConsole_WriteString("[AI] xSPI2 stage reconfigure FAILED.\r\n");
		return false;
	}
	DebugConsole_WriteString("[AI] xSPI2 stage reconfigure OK.\r\n");

	if (!AppAI_Xspi2ModelImageMatchesMappedFlashForStage(stage)) {
		/* Signature mismatch — log but trust whatever is in flash rather than
		 * aborting. The model in flash may be valid even if the hardcoded
		 * signature bytes are stale. */
		DebugConsole_Printf(
				"[AI] xSPI2 stage '%s' signature mismatch — continuing anyway.\r\n",
				stage->stage_label);
	} else {
		DebugConsole_WriteString("[AI] xSPI2 stage image already present.\r\n");
	}

	if (!AppAI_Xspi2EnableMemoryMappedMode()) {
		AppAI_LogXspi2LoadFailure("enable MM after verify", FX_SUCCESS,
				BSP_ERROR_COMPONENT_FAILURE);
		return false;
	}
	app_ai_loaded_xspi2_stage = stage;
	return true;
}

static bool AppAI_EnsureStageRuntimeReady(const AppAI_ModelStageSpec *stage) {
	if (stage == NULL) {
		return false;
	}

	DebugConsole_WriteString("[AI] Stage runtime ready start.\r\n");
	DebugConsole_Printf("[AI] Stage label: %s\r\n",
			(stage->stage_label != NULL) ? stage->stage_label : "(unnamed)");
	if (!AppAI_EnsureXspi2ModelImageReadyForStage(stage)) {
		DebugConsole_WriteString(
				"[AI] Stage runtime ready failed during xSPI2 setup.\r\n");
		return false;
	}

	DebugConsole_WriteString("[AI] Stage network init start.\r\n");
	if ((stage->network_init_fn == NULL) || !stage->network_init_fn()) {
		AppAI_LogInitFailure(stage->stage_label);
		return false;
	}

	LL_ATON_RT_Init_Network(stage->nn_instance);
	DebugConsole_WriteString("[AI] Stage network init OK.\r\n");

	DebugConsole_WriteString("[AI] Stage inference init start.\r\n");
	if ((stage->inference_init_fn == NULL) || !stage->inference_init_fn()) {
		AppAI_LogInitFailure(stage->stage_label);
		return false;
	}

	DebugConsole_WriteString("[AI] Stage inference init OK.\r\n");
	return true;
}

static bool AppAI_DecodeRectifierCropBox(
		const LL_Buffer_InfoTypeDef *output_buffer_info,
		AppAI_SourceCrop *crop_out,
		AppAI_RectifierBox *rectifier_box_out) {
	const float *output_ptr = NULL;
	const size_t source_width = (size_t) APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const size_t source_height = (size_t) APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	float center_x = 0.0f;
	float center_y = 0.0f;
	float box_w = 0.0f;
float box_h = 0.0f;
float crop_x_min_f = 0.0f;
float crop_y_min_f = 0.0f;
float crop_width_f = 0.0f;
float crop_height_f = 0.0f;
#if APP_AI_RECTIFIER_FIXED_SCALE_CROP
size_t training_center_x = 0U;
size_t training_center_y = 0U;
const float rectifier_center_blend_f =
		((float) APP_AI_RECTIFIER_CENTER_BLEND_NUMERATOR)
				/ ((float) APP_AI_RECTIFIER_CENTER_BLEND_DENOMINATOR);
#endif
bool use_fixed_training_crop = false;

	if ((output_buffer_info == NULL) || (crop_out == NULL)) {
		return false;
	}

	output_ptr = (const float *) LL_Buffer_addr_start(output_buffer_info);
	if ((output_ptr == NULL) || (LL_Buffer_len(output_buffer_info) < (sizeof(float) * 4U))) {
		return false;
	}

	/* Log the raw rectifier output before any clamping so we can distinguish
	 * between the model producing a plausible box that the fallback rejects,
	 * a near-zero output (model not running / zero-init), or an out-of-range
	 * value (wrong output format / normalization mismatch). */
	{
		union { float f; uint32_t u; } r0, r1, r2, r3;
		char rect_log[192];
		r0.f = output_ptr[0]; r1.f = output_ptr[1];
		r2.f = output_ptr[2]; r3.f = output_ptr[3];
#if APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
		(void) snprintf(rect_log, sizeof(rect_log),
				"[AI] Rectifier raw: cx=%ld cy=%ld w=%ld h=%ld "
				"bits=[%08lX %08lX %08lX %08lX]\r\n",
				(long) (output_ptr[0] * 1000.0f),
				(long) (output_ptr[1] * 1000.0f),
				(long) (output_ptr[2] * 1000.0f),
				(long) (output_ptr[3] * 1000.0f),
				(unsigned long) r0.u, (unsigned long) r1.u,
				(unsigned long) r2.u, (unsigned long) r3.u);
		(void) DebugConsole_WriteString(rect_log);
#endif
	}

	center_x = AppAI_ClampNormalizedFloat(output_ptr[0]);
	center_y = AppAI_ClampNormalizedFloat(output_ptr[1]);
	box_w = AppAI_ClampNormalizedFloat(output_ptr[2]);
	box_h = AppAI_ClampNormalizedFloat(output_ptr[3]);
	if (box_w < APP_AI_RECTIFIER_MIN_BOX_RATIO) {
		box_w = APP_AI_RECTIFIER_MIN_BOX_RATIO;
	}
	if (box_h < APP_AI_RECTIFIER_MIN_BOX_RATIO) {
		box_h = APP_AI_RECTIFIER_MIN_BOX_RATIO;
	}

	if (rectifier_box_out != NULL) {
		rectifier_box_out->center_x = center_x;
		rectifier_box_out->center_y = center_y;
		rectifier_box_out->box_w = box_w;
		rectifier_box_out->box_h = box_h;
	}

#if APP_AI_RECTIFIER_FIXED_SCALE_CROP
	/* Soft attention: ignore (box_w, box_h) entirely. Validate centre only. */
	if ((center_x < APP_AI_RECTIFIER_CENTER_MIN_RATIO)
			|| (center_x > APP_AI_RECTIFIER_CENTER_MAX_RATIO)
			|| (center_y < APP_AI_RECTIFIER_CENTER_MIN_RATIO)
			|| (center_y > APP_AI_RECTIFIER_CENTER_MAX_RATIO)) {
		{
			char fb_log[96];
#if APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
			(void) snprintf(fb_log, sizeof(fb_log),
					"[AI] Rect fallback (centre out of range): cx=%ld cy=%ld lim=[%ld..%ld]\r\n",
					(long) (center_x * 1000.0f), (long) (center_y * 1000.0f),
					(long) (APP_AI_RECTIFIER_CENTER_MIN_RATIO * 1000.0f),
					(long) (APP_AI_RECTIFIER_CENTER_MAX_RATIO * 1000.0f));
			(void) DebugConsole_WriteString(fb_log);
#endif
		}
		use_fixed_training_crop = true;
	}
#else
	if ((box_w < APP_AI_RECTIFIER_FALLBACK_MIN_BOX_RATIO)
			|| (box_h < APP_AI_RECTIFIER_FALLBACK_MIN_BOX_RATIO)
			|| (box_w > APP_AI_RECTIFIER_FALLBACK_MAX_BOX_RATIO)
			|| (box_h > APP_AI_RECTIFIER_FALLBACK_MAX_BOX_RATIO)) {
		{
			char fb_log[96];
#if APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
			(void) snprintf(fb_log, sizeof(fb_log),
					"[AI] Rect fallback: cx=%ld cy=%ld w=%ld h=%ld lim=[%ld..%ld]\r\n",
					(long) (center_x * 1000.0f), (long) (center_y * 1000.0f),
					(long) (box_w * 1000.0f), (long) (box_h * 1000.0f),
					(long) (APP_AI_RECTIFIER_FALLBACK_MIN_BOX_RATIO * 1000.0f),
					(long) (APP_AI_RECTIFIER_FALLBACK_MAX_BOX_RATIO * 1000.0f));
			(void) DebugConsole_WriteString(fb_log);
#endif
		}
		use_fixed_training_crop = true;
	}
#endif

	if (use_fixed_training_crop) {
		/* Keep the scalar reader on the same stable crop that the offline
		 * training/evaluation path uses when the rectifier output is implausible.
		 * That avoids 1x1/tiny crops and avoids treating full-frame predictions
		 * as if they were useful dial boxes. */
		crop_out->x_min = (size_t) ((float) source_width
				* APP_AI_TRAINING_CROP_X_MIN_RATIO);
		crop_out->y_min = (size_t) ((float) source_height
				* APP_AI_TRAINING_CROP_Y_MIN_RATIO);
		crop_out->width = (size_t) ((float) source_width
				* (APP_AI_TRAINING_CROP_X_MAX_RATIO
				- APP_AI_TRAINING_CROP_X_MIN_RATIO));
		crop_out->height = (size_t) ((float) source_height
				* (APP_AI_TRAINING_CROP_Y_MAX_RATIO
				- APP_AI_TRAINING_CROP_Y_MIN_RATIO));
		if (crop_out->width == 0U) {
			crop_out->width = 1U;
		}
		if (crop_out->height == 0U) {
			crop_out->height = 1U;
		}
#if APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
		{
			char fb_log[128];

			(void) snprintf(fb_log, sizeof(fb_log),
					"[AI] Rectifier fallback -> fixed training crop: x=%lu y=%lu w=%lu h=%lu\r\n",
					(unsigned long) crop_out->x_min,
					(unsigned long) crop_out->y_min,
					(unsigned long) crop_out->width,
					(unsigned long) crop_out->height);
			(void) DebugConsole_WriteString(fb_log);
		}
#else
		DebugConsole_WriteString(
				"[AI] Rectifier crop fallback: using fixed training crop.\r\n");
#endif
		return true;
	}

#if APP_AI_RECTIFIER_FIXED_SCALE_CROP
	/* Soft attention: rectifier provides a nudge, training crop provides the
	 * base framing. This keeps the scalar input close to the fixed crop that it
	 * was trained on while still letting the crop track the gauge a little. */
	AppGaugeGeometry_TrainingCropCenter(source_width, source_height,
			&training_center_x, &training_center_y);
	crop_width_f = ((float) source_width)
			* (APP_AI_TRAINING_CROP_X_MAX_RATIO
					- APP_AI_TRAINING_CROP_X_MIN_RATIO);
	crop_height_f = ((float) source_height)
			* (APP_AI_TRAINING_CROP_Y_MAX_RATIO
					- APP_AI_TRAINING_CROP_Y_MIN_RATIO);
#else
	crop_width_f = ((float) source_width) * box_w * APP_AI_RECTIFIER_CROP_SCALE;
	crop_height_f = ((float) source_height) * box_h * APP_AI_RECTIFIER_CROP_SCALE;
#endif
	if (crop_width_f < 1.0f) {
		crop_width_f = 1.0f;
	}
	if (crop_height_f < 1.0f) {
		crop_height_f = 1.0f;
	}

#if APP_AI_RECTIFIER_FIXED_SCALE_CROP
	crop_x_min_f = ((float) training_center_x)
			+ ((((float) source_width) * center_x
					- (float) training_center_x) * rectifier_center_blend_f)
			- (crop_width_f * 0.5f);
	crop_y_min_f = ((float) training_center_y)
			+ ((((float) source_height) * center_y
					- (float) training_center_y) * rectifier_center_blend_f)
			- (crop_height_f * 0.5f);
#else
	crop_x_min_f = (((float) source_width) * center_x) - (crop_width_f * 0.5f);
	crop_y_min_f = (((float) source_height) * center_y) - (crop_height_f * 0.5f);
#endif
	if (crop_x_min_f < 0.0f) {
		crop_x_min_f = 0.0f;
	}
	if (crop_y_min_f < 0.0f) {
		crop_y_min_f = 0.0f;
	}
	if ((crop_x_min_f + crop_width_f) > (float) source_width) {
		crop_x_min_f = (float) source_width - crop_width_f;
	}
	if ((crop_y_min_f + crop_height_f) > (float) source_height) {
		crop_y_min_f = (float) source_height - crop_height_f;
	}
	if (crop_x_min_f < 0.0f) {
		crop_x_min_f = 0.0f;
	}
	if (crop_y_min_f < 0.0f) {
		crop_y_min_f = 0.0f;
	}

	crop_out->x_min = (size_t) (crop_x_min_f + 0.5f);
	crop_out->y_min = (size_t) (crop_y_min_f + 0.5f);
	crop_out->width = (size_t) (crop_width_f + 0.5f);
	crop_out->height = (size_t) (crop_height_f + 0.5f);

	if (crop_out->width == 0U) {
		crop_out->width = 1U;
	}
	if (crop_out->height == 0U) {
		crop_out->height = 1U;
	}
	if (crop_out->x_min >= source_width) {
		crop_out->x_min = source_width - 1U;
	}
	if (crop_out->y_min >= source_height) {
		crop_out->y_min = source_height - 1U;
	}
	if ((crop_out->x_min + crop_out->width) > source_width) {
		crop_out->width = source_width - crop_out->x_min;
	}
	if ((crop_out->y_min + crop_out->height) > source_height) {
		crop_out->height = source_height - crop_out->y_min;
	}

	return true;
}

#if !APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
static void AppAI_LogXspi2PrefixBytes(const char *label,
		const uint8_t *bytes) {
	(void) label;
	(void) bytes;
}

static void AppAI_LogFrameSignature(const uint8_t *frame_bytes,
		size_t frame_size) {
	(void) frame_bytes;
	(void) frame_size;
}

static void AppAI_LogInputSignature(const float *input_buffer,
		size_t input_float_count) {
	(void) input_buffer;
	(void) input_float_count;
}

static void AppAI_LogInputTensorWindow(const float *input_buffer,
		size_t input_float_count) {
	(void) input_buffer;
	(void) input_float_count;
}

static void AppAI_LogInputProbeSummary(const float *input_buffer,
		size_t input_float_count) {
	(void) input_buffer;
	(void) input_float_count;
}

static void AppAI_LogTensorRowSamples(const char *label,
		const float *input_buffer, size_t tensor_width, size_t y,
		size_t x_min, size_t x_max) {
	(void) label;
	(void) input_buffer;
	(void) tensor_width;
	(void) y;
	(void) x_min;
	(void) x_max;
}

static void AppAI_LogSourcePatch(const char *label, const uint8_t *frame_bytes,
		size_t frame_width_pixels, size_t center_x, size_t center_y,
		size_t radius_pixels) {
	(void) label;
	(void) frame_bytes;
	(void) frame_width_pixels;
	(void) center_x;
	(void) center_y;
	(void) radius_pixels;
}

static void AppAI_LogTensorPatch(const char *label, const float *input_buffer,
		size_t tensor_width, size_t center_x, size_t center_y,
		size_t radius_pixels) {
	(void) label;
	(void) input_buffer;
	(void) tensor_width;
	(void) center_x;
	(void) center_y;
	(void) radius_pixels;
}

static void AppAI_LogSourceCropWindow(const uint8_t *frame_bytes,
		size_t frame_size, size_t frame_width_pixels, size_t frame_height_pixels,
		size_t crop_x_min, size_t crop_y_min, size_t crop_width,
		size_t crop_height) {
	(void) frame_bytes;
	(void) frame_size;
	(void) frame_width_pixels;
	(void) frame_height_pixels;
	(void) crop_x_min;
	(void) crop_y_min;
	(void) crop_width;
	(void) crop_height;
}

static void AppAI_LogInt8BufferSignature(const char *label,
		const int8_t *buffer_ptr, size_t buffer_len_bytes) {
	(void) label;
	(void) buffer_ptr;
	(void) buffer_len_bytes;
}

static void AppAI_LogRawBufferSignature(const char *label,
		const uint8_t *buffer_ptr, size_t buffer_len_bytes) {
	(void) label;
	(void) buffer_ptr;
	(void) buffer_len_bytes;
}

static const char *AppAI_BufferTypeName(const LL_Buffer_InfoTypeDef *buffer_info) {
	(void) buffer_info;
	return "OTHER";
}

static void AppAI_LogBufferInfoAndSignature(const char *label,
		const LL_Buffer_InfoTypeDef *buffer_info) {
	(void) label;
	(void) buffer_info;
}
#endif /* !APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS */

static void AppAI_LogRectifierResult(
		const LL_Buffer_InfoTypeDef *output_buffer_info,
		const AppAI_RectifierBox *rectifier_box) {
	if ((output_buffer_info == NULL) || (rectifier_box == NULL)) {
		DebugConsole_Printf("[AI] Rectifier output missing.\r\n");
		return;
	}

	DebugConsole_Printf("[AI] Rectifier output: name=%s addr=%p len=%lu\r\n",
			(output_buffer_info->name != NULL) ? output_buffer_info->name
					: "(unnamed)",
			LL_Buffer_addr_start(output_buffer_info),
			(unsigned long) LL_Buffer_len(output_buffer_info));
#if APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
	DebugConsole_Printf(
			"[AI] Rectifier box(clamped): cx=%.6f cy=%.6f w=%.6f h=%.6f\r\n",
			rectifier_box->center_x, rectifier_box->center_y,
			rectifier_box->box_w, rectifier_box->box_h);
#endif
}

static void AppAI_LogObbResult(
		const LL_Buffer_InfoTypeDef *output_buffer_info,
		const AppAI_ObbBox *obb_box) {
	float theta_deg = 0.0f;

	if ((output_buffer_info == NULL) || (obb_box == NULL)) {
		DebugConsole_Printf("[AI] OBB output missing.\r\n");
		return;
	}

	theta_deg = obb_box->theta_rad * 57.29577951308232f;
	DebugConsole_Printf("[AI] OBB output: name=%s addr=%p len=%lu\r\n",
			(output_buffer_info->name != NULL) ? output_buffer_info->name
					: "(unnamed)",
			LL_Buffer_addr_start(output_buffer_info),
			(unsigned long) LL_Buffer_len(output_buffer_info));
#if APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
	DebugConsole_Printf(
			"[AI] OBB box(clamped): cx=%.6f cy=%.6f w=%.6f h=%.6f cos=%.6f sin=%.6f theta=%.2fdeg\r\n",
			obb_box->center_x, obb_box->center_y, obb_box->box_w,
			obb_box->box_h, obb_box->angle_cos, obb_box->angle_sin,
			theta_deg);
#endif
}

static bool AppAI_DecodeObbCropBox(
		const LL_Buffer_InfoTypeDef *output_buffer_info,
		AppAI_SourceCrop *crop_out,
		AppAI_ObbBox *obb_box_out) {
	const float *output_ptr = NULL;
	const float source_width_f = (float) APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const float source_height_f = (float) APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	const float input_size_f = (float) APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	float center_x = 0.0f;
	float center_y = 0.0f;
	float box_w = 0.0f;
	float box_h = 0.0f;
	float angle_cos = 0.0f;
	float angle_sin = 0.0f;
	float theta_rad = 0.0f;
	float canvas_center_x = 0.0f;
	float canvas_center_y = 0.0f;
	float half_width = 0.0f;
	float half_height = 0.0f;
	float crop_x_min_f = 0.0f;
	float crop_y_min_f = 0.0f;
	float crop_x_max_f = 0.0f;
	float crop_y_max_f = 0.0f;

	if ((output_buffer_info == NULL) || (crop_out == NULL)) {
		return false;
	}

	output_ptr = (const float *) LL_Buffer_addr_start(output_buffer_info);
	if ((output_ptr == NULL)
			|| (LL_Buffer_len(output_buffer_info) < (sizeof(float) * 6U))) {
		return false;
	}

	if (!AppAI_IsFiniteFloat(output_ptr[0]) || !AppAI_IsFiniteFloat(output_ptr[1])
			|| !AppAI_IsFiniteFloat(output_ptr[2])
			|| !AppAI_IsFiniteFloat(output_ptr[3])
			|| !AppAI_IsFiniteFloat(output_ptr[4])
			|| !AppAI_IsFiniteFloat(output_ptr[5])) {
		DebugConsole_WriteString("[AI] OBB output contains non-finite values.\r\n");
		return false;
	}

#if APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
	DebugConsole_Printf(
			"[AI] OBB raw: cx=%.6f cy=%.6f w=%.6f h=%.6f cos=%.6f sin=%.6f\r\n",
			output_ptr[0], output_ptr[1], output_ptr[2], output_ptr[3],
			output_ptr[4], output_ptr[5]);
#endif

	center_x = AppAI_ClampNormalizedFloat(output_ptr[0]);
	center_y = AppAI_ClampNormalizedFloat(output_ptr[1]);
	box_w = AppAI_ClampNormalizedFloat(output_ptr[2]);
	box_h = AppAI_ClampNormalizedFloat(output_ptr[3]);
	if (box_w < APP_AI_OBB_MIN_BOX_RATIO) {
		box_w = APP_AI_OBB_MIN_BOX_RATIO;
	} else if (box_w > 1.0f) {
		box_w = 1.0f;
	}
	if (box_h < APP_AI_OBB_MIN_BOX_RATIO) {
		box_h = APP_AI_OBB_MIN_BOX_RATIO;
	} else if (box_h > 1.0f) {
		box_h = 1.0f;
	}

	angle_cos = output_ptr[4];
	angle_sin = output_ptr[5];
	theta_rad = 0.5f * atan2f(angle_sin, angle_cos);

	if (obb_box_out != NULL) {
		obb_box_out->center_x = center_x;
		obb_box_out->center_y = center_y;
		obb_box_out->box_w = box_w;
		obb_box_out->box_h = box_h;
		obb_box_out->angle_cos = angle_cos;
		obb_box_out->angle_sin = angle_sin;
		obb_box_out->theta_rad = theta_rad;
	}

	canvas_center_x = center_x * input_size_f;
	canvas_center_y = center_y * input_size_f;
	half_width = 0.5f * box_w * input_size_f * APP_AI_OBB_CROP_SCALE;
	half_height = 0.5f * box_h * input_size_f * APP_AI_OBB_CROP_SCALE;

	{
		const float cos_theta = cosf(theta_rad);
		const float sin_theta = sinf(theta_rad);
		const float corner_offsets[4][2] = {
			{ -half_width, -half_height },
			{  half_width, -half_height },
			{  half_width,  half_height },
			{ -half_width,  half_height },
		};
		float source_x_values[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
		float source_y_values[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

		for (size_t corner_index = 0U; corner_index < 4U; ++corner_index) {
			const float dx = corner_offsets[corner_index][0];
			const float dy = corner_offsets[corner_index][1];

			source_x_values[corner_index] = canvas_center_x
					+ ((dx * cos_theta) - (dy * sin_theta));
			source_y_values[corner_index] = canvas_center_y
					+ ((dx * sin_theta) + (dy * cos_theta));
		}

		crop_x_min_f = source_x_values[0U];
		crop_y_min_f = source_y_values[0U];
		crop_x_max_f = source_x_values[0U];
		crop_y_max_f = source_y_values[0U];
		for (size_t corner_index = 1U; corner_index < 4U; ++corner_index) {
			if (source_x_values[corner_index] < crop_x_min_f) {
				crop_x_min_f = source_x_values[corner_index];
			}
			if (source_y_values[corner_index] < crop_y_min_f) {
				crop_y_min_f = source_y_values[corner_index];
			}
			if (source_x_values[corner_index] > crop_x_max_f) {
				crop_x_max_f = source_x_values[corner_index];
			}
			if (source_y_values[corner_index] > crop_y_max_f) {
				crop_y_max_f = source_y_values[corner_index];
			}
		}
	}

	{
		const float crop_width_f = crop_x_max_f - crop_x_min_f;
		const float crop_height_f = crop_y_max_f - crop_y_min_f;
		const float crop_center_x_f = 0.5f * (crop_x_min_f + crop_x_max_f);
		const float crop_center_y_f = 0.5f * (crop_y_min_f + crop_y_max_f);
		const float target_width_f =
				(crop_width_f < APP_AI_OBB_MIN_CROP_SIZE_PIXELS)
				? APP_AI_OBB_MIN_CROP_SIZE_PIXELS : crop_width_f;
		const float target_height_f =
				(crop_height_f < APP_AI_OBB_MIN_CROP_SIZE_PIXELS)
				? APP_AI_OBB_MIN_CROP_SIZE_PIXELS : crop_height_f;

		crop_x_min_f = crop_center_x_f - (0.5f * target_width_f);
		crop_y_min_f = crop_center_y_f - (0.5f * target_height_f);
		crop_x_max_f = crop_x_min_f + target_width_f;
		crop_y_max_f = crop_y_min_f + target_height_f;
	}

	if (crop_x_min_f < 0.0f) {
		crop_x_max_f -= crop_x_min_f;
		crop_x_min_f = 0.0f;
	}
	if (crop_y_min_f < 0.0f) {
		crop_y_max_f -= crop_y_min_f;
		crop_y_min_f = 0.0f;
	}
	if (crop_x_max_f > source_width_f) {
		const float shift = crop_x_max_f - source_width_f;
		crop_x_min_f = (crop_x_min_f > shift) ? (crop_x_min_f - shift) : 0.0f;
		crop_x_max_f = source_width_f;
	}
	if (crop_y_max_f > source_height_f) {
		const float shift = crop_y_max_f - source_height_f;
		crop_y_min_f = (crop_y_min_f > shift) ? (crop_y_min_f - shift) : 0.0f;
		crop_y_max_f = source_height_f;
	}
	if (crop_x_min_f < 0.0f) {
		crop_x_min_f = 0.0f;
	}
	if (crop_y_min_f < 0.0f) {
		crop_y_min_f = 0.0f;
	}
	if (crop_x_max_f > source_width_f) {
		crop_x_max_f = source_width_f;
	}
	if (crop_y_max_f > source_height_f) {
		crop_y_max_f = source_height_f;
	}
	if ((crop_x_max_f <= crop_x_min_f) || (crop_y_max_f <= crop_y_min_f)) {
		return false;
	}

	crop_out->x_min = (size_t) floorf(crop_x_min_f);
	crop_out->y_min = (size_t) floorf(crop_y_min_f);
	crop_out->width = (size_t) ceilf(crop_x_max_f) - crop_out->x_min;
	crop_out->height = (size_t) ceilf(crop_y_max_f) - crop_out->y_min;
	if (crop_out->width == 0U) {
		crop_out->width = 1U;
	}
	if (crop_out->height == 0U) {
		crop_out->height = 1U;
	}
	if (crop_out->x_min >= (size_t) APP_AI_CAPTURE_FRAME_WIDTH_PIXELS) {
		crop_out->x_min = (size_t) APP_AI_CAPTURE_FRAME_WIDTH_PIXELS - 1U;
	}
	if (crop_out->y_min >= (size_t) APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS) {
		crop_out->y_min = (size_t) APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS - 1U;
	}
	if ((crop_out->x_min + crop_out->width)
			> (size_t) APP_AI_CAPTURE_FRAME_WIDTH_PIXELS) {
		crop_out->width = (size_t) APP_AI_CAPTURE_FRAME_WIDTH_PIXELS
				- crop_out->x_min;
	}
	if ((crop_out->y_min + crop_out->height)
			> (size_t) APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS) {
		crop_out->height = (size_t) APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS
				- crop_out->y_min;
	}
	if ((crop_out->width == 0U) || (crop_out->height == 0U)) {
		return false;
	}

	return true;
}

static bool AppAI_RunStageInference(const AppAI_ModelStageSpec *stage,
		const uint8_t *frame_bytes, size_t frame_size,
		const AppAI_SourceCrop *forced_crop,
		const LL_Buffer_InfoTypeDef **output_info_out) {
	const LL_Buffer_InfoTypeDef *input_info = NULL;
	const LL_Buffer_InfoTypeDef *output_info = NULL;
	float *input_ptr = NULL;
	size_t input_len_bytes = 0U;
	size_t input_float_count = 0U;
	const float *output_ptr = NULL;
	size_t output_len_bytes = 0U;

	if ((stage == NULL) || (frame_bytes == NULL)) {
		return false;
	}

	(void) DebugConsole_WriteString("[AI] Stage inference request.\r\n");
	if (!AppAI_EnsureStageRuntimeReady(stage)) {
		(void) DebugConsole_WriteString(
				"[AI] Stage inference aborted before preprocess.\r\n");
		return false;
	}

	input_info = AppAI_GetStageInputBufferInfo(stage);
	output_info = AppAI_GetStageOutputBufferInfo(stage);
	if ((input_info == NULL) || (output_info == NULL)) {
		AppAI_ClearForcedCrop();
		return false;
	}

	input_ptr = (float *) LL_Buffer_addr_start(input_info);
	input_len_bytes = (size_t) LL_Buffer_len(input_info);
	input_float_count = input_len_bytes / sizeof(float);
	if (input_ptr == NULL) {
		AppAI_ClearForcedCrop();
		return false;
	}

	if (forced_crop != NULL) {
		AppAI_SetForcedCrop(stage->stage_label, forced_crop->x_min,
				forced_crop->y_min, forced_crop->width, forced_crop->height);
	} else {
		AppAI_ClearForcedCrop();
	}

	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		DebugConsole_Printf(
				"[AI] Run stage=%s frame_ptr=%p frame_size=%lu input=%s addr=%p len=%lu output=%s addr=%p len=%lu\r\n",
				stage->stage_label, (const void *) frame_bytes,
				(unsigned long) frame_size,
				(input_info->name != NULL) ? input_info->name : "(unnamed)",
				(void *) input_ptr, (unsigned long) input_len_bytes,
				(output_info->name != NULL) ? output_info->name : "(unnamed)",
				(void *) LL_Buffer_addr_start(output_info),
				(unsigned long) LL_Buffer_len(output_info));
	}

	if (!AppAI_PreprocessYuv422FrameToFloatInput(frame_bytes, frame_size,
			input_ptr, input_float_count)) {
		AppAI_ClearForcedCrop();
		return false;
	}

	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		DebugConsole_Printf(
				"[AI] %s preprocess complete; logging tensor signatures.\r\n",
				stage->stage_label);
		AppAI_LogFrameSignature(frame_bytes, frame_size);
		AppAI_LogInputSignature(input_ptr, input_float_count);
		AppAI_LogInputTensorWindow(input_ptr, input_float_count);
		AppAI_LogInputProbeSummary(input_ptr, input_float_count);
		AppAI_LogTensorPatch("Tensor center", input_ptr,
				(size_t) APP_AI_CAPTURE_FRAME_WIDTH_PIXELS,
				(size_t) APP_AI_CAPTURE_FRAME_WIDTH_PIXELS / 2U,
				(size_t) APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS / 2U, 2U);
	}

	AppAI_LogBufferPreview("Stage input", input_info);

	(void) mcu_cache_clean_range((uint32_t) (uintptr_t) input_ptr,
			(uint32_t) ((uintptr_t) input_ptr + input_len_bytes));

	LL_ATON_RT_Reset_Network(stage->nn_instance);

	for (uint32_t epoch_step = 0U;; ++epoch_step) {
		const LL_ATON_RT_RetValues_t run_status =
				LL_ATON_RT_RunEpochBlock(stage->nn_instance);

		if (run_status == LL_ATON_RT_DONE) {
			break;
		}

		if (run_status == LL_ATON_RT_WFE) {
			LL_ATON_OSAL_WFE();
		} else {
			tx_thread_relinquish();
		}
	}

	output_ptr = (const float *) LL_Buffer_addr_start(output_info);
	output_len_bytes = (size_t) LL_Buffer_len(output_info);
	if ((output_ptr == NULL) || (output_len_bytes < sizeof(float))) {
		AppAI_ClearForcedCrop();
		return false;
	}

	(void) mcu_cache_invalidate_range((uint32_t) (uintptr_t) output_ptr,
			(uint32_t) ((uintptr_t) output_ptr + output_len_bytes));

	AppAI_LogBufferPreview("Stage output", output_info);

	/* Log raw output bytes immediately after cache invalidate so we can tell
	 * whether the inference engine populated the buffer at all. */
	{
		const uint8_t *ob = (const uint8_t *) output_ptr;
		const size_t on = (output_len_bytes < 16U) ? output_len_bytes : 16U;
		char olog[128];
#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
		(void) snprintf(olog, sizeof(olog),
				"[AI] %s out addr=%p len=%lu bytes=[%02X%02X%02X%02X%02X%02X%02X%02X"
				"%02X%02X%02X%02X%02X%02X%02X%02X]\r\n",
				stage->stage_label, (const void *) output_ptr,
				(unsigned long) output_len_bytes,
				(on>0U)?ob[0]:0U,(on>1U)?ob[1]:0U,
				(on>2U)?ob[2]:0U,(on>3U)?ob[3]:0U,
				(on>4U)?ob[4]:0U,(on>5U)?ob[5]:0U,
				(on>6U)?ob[6]:0U,(on>7U)?ob[7]:0U,
				(on>8U)?ob[8]:0U,(on>9U)?ob[9]:0U,
				(on>10U)?ob[10]:0U,(on>11U)?ob[11]:0U,
				(on>12U)?ob[12]:0U,(on>13U)?ob[13]:0U,
				(on>14U)?ob[14]:0U,(on>15U)?ob[15]:0U);
		(void) DebugConsole_WriteString(olog);
#endif
	}

	if (output_info_out != NULL) {
		*output_info_out = output_info;
	}

	AppAI_ClearForcedCrop();
	return true;
}

static bool AppAI_WaitForFileXMediaReady(uint32_t timeout_ms) {
	const ULONG timeout_ticks = (ULONG) ((timeout_ms + 9U) / 10U);
	const ULONG start_tick = tx_time_get();

	while (!AppFileX_IsMediaReady()) {
		if ((tx_time_get() - start_tick) >= timeout_ticks) {
			return false;
		}

		tx_thread_sleep(1U);
	}

	return true;
}

#if 0
static bool AppAI_EnsureXspi2ModelImageReady(void) {
	if (app_ai_xspi2_initialized) {
		return true;
	}

	/* The runtime consumes the blob through the mapped xSPI2 window, so verify
	 * that view first. Some boots leave the indirect probe path in a misleading
	 * state even when the programmed image is still present. */
	if (!AppAI_ReconfigureXspi2ForRuntime()) {
		AppAI_LogXspi2LoadFailure("runtime reconfigure", FX_SUCCESS,
				BSP_ERROR_COMPONENT_FAILURE);
		return false;
	}
	if (!AppAI_Xspi2EnableMemoryMappedMode()) {
		AppAI_LogXspi2LoadFailure("enable MM for verify", FX_SUCCESS,
				BSP_ERROR_COMPONENT_FAILURE);
		return false;
	}

	if (AppAI_Xspi2ModelImageMatchesMappedFlash()) {
		DebugConsole_Printf(
				"[AI] xSPI2 model image already present; skipping provisioning.\r\n");
		AppAI_LogXspi2IndirectAndMappedPrefix();
		AppAI_LogXspi2MappedScaleBytes();
	} else {
		DebugConsole_Printf(
				"[AI] xSPI2 model image missing or stale; programming from SD card.\r\n");
		if (!AppAI_EnsureXspi2MemoryReady()) {
			AppAI_LogXspi2LoadFailure("xSPI2 memory", FX_SUCCESS,
					BSP_ERROR_COMPONENT_FAILURE);
			return false;
		}

		if (!AppAI_ProgramXspi2ModelImageFromSd()) {
			DebugConsole_Printf(
					"[AI] xSPI2 model image provisioning failed.\r\n");
			return false;
		}
	}

	if (!AppAI_Xspi2ModelImageMatchesMappedFlash()) {
		DebugConsole_Printf(
				"[AI] xSPI2 mapped verify failed after provisioning.\r\n");
	}

	app_ai_xspi2_initialized = true;
	DebugConsole_Printf("[AI] xSPI2 model image ready.\r\n");
	return true;
}
#endif /* 0 */

bool App_AI_Model_Init(void) {
	if (app_ai_runtime_initialized) {
		return true;
	}

	(void) DebugConsole_WriteString("[AI] Model runtime init start.\r\n");
	if (!AppAI_EnsureNpuHardwareReady()) {
		AppAI_LogInitFailure("NPU hardware");
		return false;
	}

	if (!AppAI_RuntimeInitStepwise()) {
		AppAI_LogInitFailure("runtime init");
		return false;
	}

	app_ai_runtime_initialized = true;
	AppAI_ResetInferenceBurstHistory();
	DebugConsole_Printf("[AI] Model runtime init OK.\r\n");
	return true;
}

#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
static void AppAI_LogInitFailure(const char *step) {
	if (step != NULL) {
		DebugConsole_Printf("[AI] Model runtime init failed at %s.\r\n", step);
	} else {
		DebugConsole_Printf("[AI] Model runtime init failed.\r\n");
	}
}

static void AppAI_LogXspi2LoadFailure(const char *step, UINT fx_status,
		int32_t bsp_status) {
	DebugConsole_Printf("[AI] xSPI2 load failed at %s (fx=%lu bsp=%ld).\r\n",
			(step != NULL) ? step : "unknown",
			(unsigned long) fx_status,
			(long) bsp_status);
}

static void AppAI_LogXspi2ProgramChunkProgress(ULONG chunk_index,
		ULONG flash_offset, ULONG chunk_size) {
	DebugConsole_Printf(
			"[AI] xSPI2 program chunk %lu offset=0x%08lX size=%lu.\r\n",
			(unsigned long) chunk_index,
			(unsigned long) flash_offset,
			(unsigned long) chunk_size);
}

static void AppAI_LogXspi2FlashStatus(const char *label) {
	/* Read both the generic BSP status and the raw security/status bytes so we
	 * can distinguish write protection, suspend state, and a plain readback bug. */
	uint8_t security_reg = 0U;
	uint8_t status_reg = 0U;
	int32_t bsp_status = BSP_XSPI_NOR_GetStatus(0U);
	int32_t security_status = MX25UM51245G_ReadSecurityRegister(&hxspi_nor[0U],
			Xspi_Nor_Ctx[0U].InterfaceMode, Xspi_Nor_Ctx[0U].TransferRate,
			&security_reg);
	int32_t status_reg_status = MX25UM51245G_ReadStatusRegister(&hxspi_nor[0U],
			Xspi_Nor_Ctx[0U].InterfaceMode, Xspi_Nor_Ctx[0U].TransferRate,
			&status_reg);
	char msg[192];

	(void) snprintf(
			msg,
			sizeof(msg),
			"[AI] %s flash status=%ld sec=%ld sec_reg=0x%02X sr=%ld sr_reg=0x%02X mode=%u rate=%u.\r\n",
			(label != NULL) ? label : "xSPI2",
			(long) bsp_status,
			(long) security_status,
			(unsigned int) security_reg,
			(long) status_reg_status,
			(unsigned int) status_reg,
			(unsigned int) Xspi_Nor_Ctx[0U].InterfaceMode,
			(unsigned int) Xspi_Nor_Ctx[0U].TransferRate);
	(void) DebugConsole_WriteString(msg);
}
#else
static void AppAI_LogInitFailure(const char *step) {
	(void) step;
}

static void AppAI_LogXspi2LoadFailure(const char *step, UINT fx_status,
		int32_t bsp_status) {
	(void) step;
	(void) fx_status;
	(void) bsp_status;
}

static void AppAI_LogXspi2ProgramChunkProgress(ULONG chunk_index,
		ULONG flash_offset, ULONG chunk_size) {
	(void) chunk_index;
	(void) flash_offset;
	(void) chunk_size;
}

static void AppAI_LogXspi2FlashStatus(const char *label) {
	(void) label;
}
#endif /* APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS */

#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
static void AppAI_LogXspi2FlashPrefix(void) {
	uint8_t flash_bytes[APP_AI_XSPI2_PROBE_BYTES] = { 0U };

	if (BSP_XSPI_NOR_Read(0U, flash_bytes, 0U,
			APP_AI_XSPI2_PROBE_BYTES) != BSP_ERROR_NONE) {
		DebugConsole_Printf("[AI] xSPI2 prefix readback failed.\r\n");
		return;
	}

	AppAI_LogXspi2PrefixBytes("xSPI2 prefix readback:", flash_bytes);
}

static void AppAI_LogXspi2MappedScaleBytes(void) {
	if (app_ai_xspi2_programmed_size < 4U) {
		return;
	}
	const uint8_t *const tail_ptr = (const uint8_t *) (APP_AI_XSPI2_MODEL_BASE_ADDR
			+ app_ai_xspi2_programmed_size - 4U);

	(void) mcu_cache_invalidate_range((uint32_t) (uintptr_t) tail_ptr,
			(uint32_t) ((uintptr_t) tail_ptr + 4U));

	DebugConsole_Printf(
			"[AI] xSPI2 mapped tail bytes @%p = %02X %02X %02X %02X\r\n",
			(const void *) tail_ptr, tail_ptr[0], tail_ptr[1], tail_ptr[2],
			tail_ptr[3]);
}

static void AppAI_LogXspi2IndirectAndMappedPrefix(void) {
	uint8_t indirect_bytes[APP_AI_XSPI2_PROBE_BYTES] = { 0U };
	uint8_t mapped_bytes[APP_AI_XSPI2_PROBE_BYTES] = { 0U };
	const int32_t disable_status = BSP_XSPI_NOR_DisableMemoryMappedMode(0U);

	if (disable_status != BSP_ERROR_NONE) {
		DebugConsole_Printf(
				"[AI] xSPI2 disable-mapped before compare returned %ld.\r\n",
				(long) disable_status);
	}

	if (BSP_XSPI_NOR_Read(0U, indirect_bytes, 0U,
			APP_AI_XSPI2_PROBE_BYTES) != BSP_ERROR_NONE) {
		DebugConsole_Printf("[AI] xSPI2 indirect prefix read failed.\r\n");
	} else {
		AppAI_LogXspi2PrefixBytes("xSPI2 indirect prefix:", indirect_bytes);
	}

	if (BSP_XSPI_NOR_EnableMemoryMappedMode(0U) != BSP_ERROR_NONE) {
		DebugConsole_Printf("[AI] xSPI2 re-enable mapped compare failed.\r\n");
		return;
	}

	(void) mcu_cache_invalidate_range(APP_AI_XSPI2_MODEL_BASE_ADDR,
			APP_AI_XSPI2_MODEL_BASE_ADDR + APP_AI_XSPI2_PROBE_BYTES);
	(void) memcpy(mapped_bytes, (const void *) APP_AI_XSPI2_MODEL_BASE_ADDR,
			APP_AI_XSPI2_PROBE_BYTES);
	AppAI_LogXspi2PrefixBytes("xSPI2 mapped prefix:", mapped_bytes);
}

static void AppAI_LogFloatApprox(const char *label, float value) {
	union {
		float f;
		uint32_t u;
	} bits = {
		.f = value
	};
	unsigned long magnitude_whole = 0U;
	unsigned long magnitude_frac = 0U;
	const char *sign = "";
	double abs_value = 0.0;

	if (label == NULL) {
		return;
	}

	if ((bits.u & 0x7F800000U) == 0x7F800000U) {
		if ((bits.u & 0x007FFFFFU) != 0U) {
			DebugConsole_Printf("%sNaN\r\n", label);
		} else if ((bits.u & 0x80000000U) != 0U) {
			DebugConsole_Printf("%s-Inf\r\n", label);
		} else {
			DebugConsole_Printf("%s+Inf\r\n", label);
		}
		return;
	}

	if ((bits.u & 0x80000000U) != 0U) {
		sign = "-";
	}

	abs_value = (bits.u & 0x80000000U) != 0U ? -(double) value : (double) value;
	if (abs_value < 0.0) {
		abs_value = 0.0;
	}

	magnitude_whole = (unsigned long) abs_value;
	magnitude_frac = (unsigned long) ((abs_value - (double) magnitude_whole)
			* 1000000.0 + 0.5);
	if (magnitude_frac >= 1000000U) {
		magnitude_whole++;
		magnitude_frac -= 1000000U;
	}

DebugConsole_Printf("%s%s%lu.%06lu\r\n", label, sign,
			magnitude_whole, magnitude_frac);
}
#endif /* APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS */

/**
 * @brief Log the raw inference value and the calibrated board-facing value.
 *
 * This trace stays in the board UART log so we can tell whether a miss comes
 * from the network itself or from the postprocess calibration.
 */
static float AppAI_TraceAndApplyInferenceCalibration(float raw_value) {
	char line[96] = { 0 };
	const float calibrated_value = AppInferenceCalibration_Apply(raw_value);

	AppInferenceLog_FormatFloatMicros(line, sizeof(line),
			"[AI] Model output before calibration: ", raw_value);
	DebugConsole_WriteString(line);
	AppInferenceLog_FormatFloatMicros(line, sizeof(line),
			"[AI] Model output after calibration: ", calibrated_value);
	DebugConsole_WriteString(line);

	if (AppAI_IsFiniteFloat(raw_value) && AppAI_IsFiniteFloat(calibrated_value)) {
		AppInferenceLog_FormatFloatMicros(line, sizeof(line),
				"[AI] Calibration delta: ",
				calibrated_value - raw_value);
		DebugConsole_WriteString(line);
	} else {
		DebugConsole_WriteString(
				"[AI] Calibration delta: unavailable (non-finite input or output)\r\n");
	}

	return calibrated_value;
}

/**
 * @brief Reset the tiny burst history used to stabilize the AI output.
 */
static void AppAI_ResetInferenceBurstHistory(void) {
	(void) memset(app_ai_inference_burst_history, 0,
			sizeof(app_ai_inference_burst_history));
	app_ai_inference_burst_history_count = 0U;
	app_ai_inference_burst_history_next_index = 0U;
}

static bool AppAI_RuntimeInitStepwise(void) {
	uint32_t t = 0U;

	/* Let the vendor runtime perform the low-level ATON bring-up and version
	 * compatibility checks. Our wrapper only handles the OSAL and IRQ setup. */
	(void) DebugConsole_WriteString("[AI] ATON runtime init start.\r\n");
	if (LL_ATON_Init() != LL_ATON_OK) {
		(void) DebugConsole_WriteString(
				"[AI] ATON runtime init failed in LL_ATON_Init().\r\n");
		return false;
	}

	ATON_DISABLE_CLR_CONFCLR(INTCTRL, 0);
	ATON_INTCTRL_STD_INTORMSK_SET(ATON_STRENG_INT_MASK(ATON_STRENG_NUM, 0, 0));
	ATON_INTCTRL_STD_INTANDMSK_SET(0xFFFFFFFFU);
#if (ATON_INT_NR > 32)
	ATON_INTCTRL_STD_INTORMSK_H_SET(0xFFFFFFFFU);
	ATON_INTCTRL_STD_INTANDMSK_H_SET(0xFFFFFFFFU);
#endif
	ATON_ENABLE(INTCTRL, 0);

	LL_ATON_OSAL_INIT();

	LL_ATON_OSAL_DISABLE_IRQ(0U);
	LL_ATON_OSAL_DISABLE_IRQ(1U);
	LL_ATON_OSAL_DISABLE_IRQ(2U);
	LL_ATON_OSAL_DISABLE_IRQ(3U);

	LL_ATON_OSAL_ENABLE_IRQ(ATON_STD_IRQ_LINE);
	(void) DebugConsole_WriteString("[AI] ATON runtime init OK.\r\n");
	return true;
}

#if !APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
static void AppAI_LogXspi2FlashPrefix(void) {
}

static void AppAI_LogXspi2MappedScaleBytes(void) {
}

static void AppAI_LogXspi2IndirectAndMappedPrefix(void) {
}

static void AppAI_LogFloatApprox(const char *label, float value) {
	(void) label;
	(void) value;
}
#endif /* !APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS */

static void AppAI_ConfigureNpuAccessControl(void) {
	RIMC_MasterConfig_t npu_master = { 0 };

	/* Mirror the ST NPU examples so the runtime can access the cache and NPU
	 * control path with the expected secure/privileged attributes. */
	__HAL_RCC_RIFSC_CLK_ENABLE();
	__HAL_RCC_RISAF_CLK_ENABLE();
	__HAL_RCC_CACHEAXIRAM_MEM_CLK_ENABLE();

	/* The secure NPU validation app programs the NPU as secure + privileged.
	 * That matches our secure-side runtime more closely than the earlier NSEC
	 * experiment, and it avoids blocking the internal ATON control window. */
	npu_master.MasterCID = RIF_CID_1;
	npu_master.SecPriv = RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV;
	HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_NPU, &npu_master);
	HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_NPU,
			RIF_ATTRIBUTE_PRIV | RIF_ATTRIBUTE_SEC);

#if defined(RIF_RCC_PERIPH_INDEX_CACHEAXIRAM)
	HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RCC_PERIPH_INDEX_CACHEAXIRAM,
			RIF_ATTRIBUTE_PRIV | RIF_ATTRIBUTE_SEC);
#endif

#if defined(RIF_RCC_PERIPH_INDEX_CACHECONFIG)
	HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RCC_PERIPH_INDEX_CACHECONFIG,
			RIF_ATTRIBUTE_PRIV | RIF_ATTRIBUTE_SEC);
#endif
}

static void AppAI_EnableNpuMemoryAndCaches(void) {
	/* Mirror the ST NPU examples so the memory fabric is actually usable by
	 * the runtime before the first ATON init call runs. */
	RCC->MEMENR |= RCC_MEMENR_AXISRAM3EN | RCC_MEMENR_AXISRAM4EN
			| RCC_MEMENR_AXISRAM5EN | RCC_MEMENR_AXISRAM6EN
			| RCC_MEMENR_CACHEAXIRAMEN;

	RAMCFG_SRAM2_AXI->CR &= ~RAMCFG_CR_SRAMSD;
	RAMCFG_SRAM3_AXI->CR &= ~RAMCFG_CR_SRAMSD;
	RAMCFG_SRAM4_AXI->CR &= ~RAMCFG_CR_SRAMSD;
	RAMCFG_SRAM5_AXI->CR &= ~RAMCFG_CR_SRAMSD;
	RAMCFG_SRAM6_AXI->CR &= ~RAMCFG_CR_SRAMSD;

	MEMSYSCTL->MSCR |= MEMSYSCTL_MSCR_DCACTIVE_Msk | MEMSYSCTL_MSCR_ICACTIVE_Msk;
}

static uint32_t AppAI_GetRisafMaxAddr(RISAF_TypeDef *risaf) {
	uint32_t max_addr = 0U;

	if ((risaf == RISAF1_S) || (risaf == RISAF1_NS)) {
		max_addr = RISAF1_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF2_S) || (risaf == RISAF2_NS)) {
		max_addr = RISAF2_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF3_S) || (risaf == RISAF3_NS)) {
		max_addr = RISAF3_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF4_S) || (risaf == RISAF4_NS)) {
		max_addr = RISAF4_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF5_S) || (risaf == RISAF5_NS)) {
		max_addr = RISAF5_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF6_S) || (risaf == RISAF6_NS)) {
		max_addr = RISAF6_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF7_S) || (risaf == RISAF7_NS)) {
		max_addr = RISAF7_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF8_S) || (risaf == RISAF8_NS)) {
		max_addr = RISAF8_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF9_S) || (risaf == RISAF9_NS)) {
		max_addr = RISAF9_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF11_S) || (risaf == RISAF11_NS)) {
		max_addr = RISAF11_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF12_S) || (risaf == RISAF12_NS)) {
		max_addr = RISAF12_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF13_S) || (risaf == RISAF13_NS)) {
		max_addr = RISAF13_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF14_S) || (risaf == RISAF14_NS)) {
		max_addr = RISAF14_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF15_S) || (risaf == RISAF15_NS)) {
		max_addr = RISAF15_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF21_S) || (risaf == RISAF21_NS)) {
		max_addr = RISAF21_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF22_S) || (risaf == RISAF22_NS)) {
		max_addr = RISAF22_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF23_S) || (risaf == RISAF23_NS)) {
		max_addr = RISAF23_LIMIT_ADDRESS_SPACE_SIZE;
	}

	return max_addr;
}

static void AppAI_SetRisafDefault(RISAF_TypeDef *risaf) {
	RISAF_BaseRegionConfig_t risaf_conf;
	RISAF_TypeDef *const risaf_hw = (risaf == RISAF12_S) ? RISAF12_NS : risaf;

	risaf_conf.StartAddress = 0x0U;
	risaf_conf.EndAddress = AppAI_GetRisafMaxAddr(risaf_hw);
	risaf_conf.Filtering = RISAF_FILTER_ENABLE;
	risaf_conf.PrivWhitelist = RIF_CID_NONE;
	risaf_conf.ReadWhitelist = RIF_CID_MASK;
	risaf_conf.WriteWhitelist = RIF_CID_MASK;

	risaf_conf.Secure = RIF_ATTRIBUTE_SEC;
	if (risaf == RISAF12_S) {
		
	}
	HAL_RIF_RISAF_ConfigBaseRegion(risaf_hw, 0U, &risaf_conf);
	if (risaf == RISAF12_S) {
		
	}

	risaf_conf.Secure = RIF_ATTRIBUTE_NSEC;
	if (risaf == RISAF12_S) {
		
	}
	HAL_RIF_RISAF_ConfigBaseRegion(risaf_hw, 1U, &risaf_conf);
	if (risaf == RISAF12_S) {
		
	}
}

static void AppAI_ConfigureNpuRisafDefaults(void) {
	/* Keep the default ST security model so the AI runtime can reach the memory
	 * and fabric regions it expects during LL_ATON startup. */
	__HAL_RCC_RIFSC_CLK_ENABLE();
	__HAL_RCC_RISAF_CLK_ENABLE();

	
	AppAI_SetRisafDefault(RISAF2_S);
	

	
	AppAI_SetRisafDefault(RISAF3_S);
	

	
	AppAI_SetRisafDefault(RISAF4_S);
	

	
	AppAI_SetRisafDefault(RISAF5_S);
	

	
	AppAI_SetRisafDefault(RISAF6_S);
	

	
	AppAI_SetRisafDefault(RISAF7_S);
	

	/* RISAF12 programming currently stalls on this board even though the
	 * xSPI2 memory-mapped window is already enabled. Leave it untouched for
	 * now so we can validate whether the runtime can proceed without it. */
	

#if defined(RISAF8_S) && defined(RISAF15_S)
	
	AppAI_SetRisafDefault(RISAF8_S);
	AppAI_SetRisafDefault(RISAF15_S);
	
#endif
}

bool App_AI_RunDryInferenceFromYuv422(const uint8_t *frame_bytes,
		size_t frame_size) {
	const LL_Buffer_InfoTypeDef *obb_output_info = NULL;
	const LL_Buffer_InfoTypeDef *rectifier_output_info = NULL;
	const LL_Buffer_InfoTypeDef *scalar_output_info = NULL;
	AppAI_ObbBox obb_box = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	AppAI_RectifierBox rectifier_box = { 0.0f, 0.0f, 0.0f, 0.0f };
	AppAI_SourceCrop full_frame_crop = { 0U, 0U,
		(size_t) APP_AI_CAPTURE_FRAME_WIDTH_PIXELS,
		(size_t) APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS };
	AppAI_SourceCrop scalar_crop = { 0U, 0U, 0U, 0U };

	(void) DebugConsole_WriteString("[AI] Dry-run entry.\r\n");
	if (!App_AI_Model_Init()) {
		(void) DebugConsole_WriteString("[AI] Dry-run entry aborted during model init.\r\n");
		return false;
	}

	(void) DebugConsole_WriteString("[AI] Dry-run model init OK; launching OBB stage.\r\n");

	if (AppAI_RunStageInference(&app_ai_obb_stage, frame_bytes, frame_size,
			&full_frame_crop, &obb_output_info)
			&& AppAI_DecodeObbCropBox(obb_output_info, &scalar_crop, &obb_box)) {
		AppAI_LogObbResult(obb_output_info, &obb_box);
		DebugConsole_Printf(
				"[AI] OBB crop: x=%lu y=%lu w=%lu h=%lu\r\n",
				(unsigned long) scalar_crop.x_min,
				(unsigned long) scalar_crop.y_min,
				(unsigned long) scalar_crop.width,
				(unsigned long) scalar_crop.height);

		if (AppAI_RunStageInference(&app_ai_scalar_stage, frame_bytes, frame_size,
				&scalar_crop, &scalar_output_info)) {
			AppAI_LogInferenceResult(scalar_output_info);
			return app_ai_last_inference_valid;
		}

		(void) DebugConsole_WriteString(
				"[AI] OBB scalar stage failed; falling back to rectifier stage.\r\n");
	} else {
		(void) DebugConsole_WriteString(
				"[AI] OBB stage or decode failed; falling back to rectifier stage.\r\n");
	}

	(void) DebugConsole_WriteString(
			"[AI] Dry-run model init OK; launching rectifier stage.\r\n");

	if (!AppAI_RunStageInference(&app_ai_rectifier_stage, frame_bytes,
			frame_size, &full_frame_crop, &rectifier_output_info)) {
		(void) DebugConsole_WriteString("[AI] Dry-run entry aborted during rectifier stage.\r\n");
		return false;
	}

	if (!AppAI_DecodeRectifierCropBox(rectifier_output_info, &scalar_crop,
			&rectifier_box)) {
		return false;
	}

#if APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
	AppAI_LogRectifierResult(rectifier_output_info, &rectifier_box);
#endif
	DebugConsole_Printf(
			"[AI] Rectifier crop: x=%lu y=%lu w=%lu h=%lu\r\n",
			(unsigned long) scalar_crop.x_min,
			(unsigned long) scalar_crop.y_min,
			(unsigned long) scalar_crop.width,
			(unsigned long) scalar_crop.height);

	if (!AppAI_RunStageInference(&app_ai_scalar_stage, frame_bytes, frame_size,
			&scalar_crop, &scalar_output_info)) {
		(void) DebugConsole_WriteString("[AI] Dry-run entry aborted during scalar stage.\r\n");
		return false;
	}

	AppAI_LogInferenceResult(scalar_output_info);

	return app_ai_last_inference_valid;
}

/* USER CODE END 0 */

/* USER CODE BEGIN 1 */

static const LL_Buffer_InfoTypeDef *AppAI_GetInputBufferInfo(void) {
	const LL_Buffer_InfoTypeDef *input_info =
			NN_Instance_scalar_full_finetune_from_best_piecewise_calibrated_int8.network
					->input_buffers_info();

	if ((input_info == NULL) || (input_info->name == NULL)) {
		return NULL;
	}

	return input_info;
}

static const LL_Buffer_InfoTypeDef *AppAI_GetOutputBufferInfo(void) {
	const LL_Buffer_InfoTypeDef *output_info =
			NN_Instance_scalar_full_finetune_from_best_piecewise_calibrated_int8.network
					->output_buffers_info();

	if ((output_info == NULL) || (output_info->name == NULL)) {
		return NULL;
	}

	return output_info;
}

static const LL_Buffer_InfoTypeDef *AppAI_FindBufferInfoByName(
		const LL_Buffer_InfoTypeDef *buffer_list, const char *name) {
	if ((buffer_list == NULL) || (name == NULL)) {
		return NULL;
	}

	for (const LL_Buffer_InfoTypeDef *entry = buffer_list; entry->name != NULL;
			++entry) {
		if (strcmp(entry->name, name) == 0) {
			return entry;
		}
	}

	return NULL;
}

static const LL_Buffer_InfoTypeDef *AppAI_FindFirstBufferInfoByNames(
		const LL_Buffer_InfoTypeDef *buffer_list, const char *const *names,
		size_t name_count) {
	if ((buffer_list == NULL) || (names == NULL) || (name_count == 0U)) {
		return NULL;
	}

	for (size_t index = 0U; index < name_count; ++index) {
		const LL_Buffer_InfoTypeDef *buffer_info = AppAI_FindBufferInfoByName(
				buffer_list, names[index]);

		if (buffer_info != NULL) {
			return buffer_info;
		}
	}

	return NULL;
}

#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
static void AppAI_LogInferenceResult(
		const LL_Buffer_InfoTypeDef *output_buffer_info) {
	const LL_Buffer_InfoTypeDef *internal_buffers = NULL;
	const LL_Buffer_InfoTypeDef *quantize_output_info = NULL;
	const LL_Buffer_InfoTypeDef *sub_output_info = NULL;
	const LL_Buffer_InfoTypeDef *conv1_output_info = NULL;
	const LL_Buffer_InfoTypeDef *raw_output_info = NULL;
	const LL_Buffer_InfoTypeDef *scale_info = NULL;
	const LL_Buffer_InfoTypeDef *zero_point_info = NULL;
	union {
		float f;
		uint32_t u;
	} output_bits = {
		.f = 0.0f
	};
	float output_value = 0.0f;
	float head_scale = 1.0f;
	float output_dequant_scale = 1.0f;
	float head_dequant_value = 0.0f;
	int8_t raw_output_value = 0;
	int8_t head_zero_point = 0;
	int8_t output_zero_point = 0;
	static const char *const quantize_output_names[] = {
		"Sub_24_out_0",
		"Conv2D_25_zero_off_out_25",
	};
	static const char *const sub_output_names[] = {
		"Conv2D_25_zero_off_out_25",
		"Conv2D_30_zero_off_out_34",
	};
	static const char *const conv1_output_names[] = {
			"Conv2D_34_zero_off_out_43",
			"Conv2D_42_zero_off_out_61",
			"Conv2D_46_zero_off_out_70",
	};
	static const char *const raw_output_names[] = {
		"Gemm_271_out_0",
	};
	static const char *const scale_names[] = {
		"Dequantize_319_x_scale",
	};
	static const char *const zero_point_names[] = {
		"Dequantize_319_x_zero_point",
	};

	if (output_buffer_info == NULL) {
		DebugConsole_Printf("[AI] Inference failed: no output buffer.\r\n");
		return;
	}

	DebugConsole_Printf(
			"[AI] Output buffer meta: name=%s addr=%p len=%lu\r\n",
			(output_buffer_info->name != NULL) ? output_buffer_info->name : "(unnamed)",
			LL_Buffer_addr_start(output_buffer_info),
			(unsigned long) LL_Buffer_len(output_buffer_info));

	(void) memcpy(&output_bits.u, LL_Buffer_addr_start(output_buffer_info),
			sizeof(output_bits.u));
	output_value = output_bits.f;

	internal_buffers =
			LL_ATON_Internal_Buffers_Info(
					&NN_Instance_scalar_full_finetune_from_best_piecewise_calibrated_int8);
	quantize_output_info = AppAI_FindFirstBufferInfoByNames(internal_buffers,
			quantize_output_names, sizeof(quantize_output_names) / sizeof(
					quantize_output_names[0]));
	sub_output_info = AppAI_FindFirstBufferInfoByNames(internal_buffers,
			sub_output_names, sizeof(sub_output_names) / sizeof(
					sub_output_names[0]));
	conv1_output_info = AppAI_FindFirstBufferInfoByNames(internal_buffers,
			conv1_output_names, sizeof(conv1_output_names) / sizeof(
					conv1_output_names[0]));
	raw_output_info = AppAI_FindFirstBufferInfoByNames(internal_buffers,
			raw_output_names, sizeof(raw_output_names) / sizeof(
					raw_output_names[0]));
	scale_info = AppAI_FindFirstBufferInfoByNames(internal_buffers,
			scale_names, sizeof(scale_names) / sizeof(scale_names[0]));
	zero_point_info = AppAI_FindFirstBufferInfoByNames(internal_buffers,
			zero_point_names, sizeof(zero_point_names) / sizeof(
					zero_point_names[0]));

	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		AppAI_LogBufferInfoAndSignature("input tensor", quantize_output_info);
		AppAI_LogBufferInfoAndSignature("preprocess output", sub_output_info);
		AppAI_LogBufferInfoAndSignature("first conv", conv1_output_info);
		AppAI_LogBufferInfoAndSignature("raw head", raw_output_info);
		AppAI_LogBufferInfoAndSignature("network output", output_buffer_info);
	}

	/* Dump the first 16 raw bytes of both the head output and the final network
	 * output so we can tell whether the inference engine wrote anything at all
	 * and whether the int8 value being read is actually at offset 0. */
	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
			&& (raw_output_info != NULL)
			&& (LL_Buffer_addr_start(raw_output_info) != NULL)) {
		const uint8_t *p = (const uint8_t *) LL_Buffer_addr_start(raw_output_info);
		const size_t n = LL_Buffer_len(raw_output_info);
		DebugConsole_Printf(
				"[AI] Scalar raw-head bytes (len=%lu): "
				"[%02X %02X %02X %02X %02X %02X %02X %02X "
				"%02X %02X %02X %02X %02X %02X %02X %02X]\r\n",
				(unsigned long) n,
				(n >  0U) ? p[0]  : 0U, (n >  1U) ? p[1]  : 0U,
				(n >  2U) ? p[2]  : 0U, (n >  3U) ? p[3]  : 0U,
				(n >  4U) ? p[4]  : 0U, (n >  5U) ? p[5]  : 0U,
				(n >  6U) ? p[6]  : 0U, (n >  7U) ? p[7]  : 0U,
				(n >  8U) ? p[8]  : 0U, (n >  9U) ? p[9]  : 0U,
				(n > 10U) ? p[10] : 0U, (n > 11U) ? p[11] : 0U,
				(n > 12U) ? p[12] : 0U, (n > 13U) ? p[13] : 0U,
				(n > 14U) ? p[14] : 0U, (n > 15U) ? p[15] : 0U);
	}
	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
			&& (output_buffer_info != NULL)
			&& (LL_Buffer_addr_start(output_buffer_info) != NULL)) {
		const uint8_t *p = (const uint8_t *) LL_Buffer_addr_start(output_buffer_info);
		const size_t n = LL_Buffer_len(output_buffer_info);
		DebugConsole_Printf(
				"[AI] Scalar network-output bytes (len=%lu): "
				"[%02X %02X %02X %02X %02X %02X %02X %02X "
				"%02X %02X %02X %02X %02X %02X %02X %02X]\r\n",
				(unsigned long) n,
				(n >  0U) ? p[0]  : 0U, (n >  1U) ? p[1]  : 0U,
				(n >  2U) ? p[2]  : 0U, (n >  3U) ? p[3]  : 0U,
				(n >  4U) ? p[4]  : 0U, (n >  5U) ? p[5]  : 0U,
				(n >  6U) ? p[6]  : 0U, (n >  7U) ? p[7]  : 0U,
				(n >  8U) ? p[8]  : 0U, (n >  9U) ? p[9]  : 0U,
				(n > 10U) ? p[10] : 0U, (n > 11U) ? p[11] : 0U,
				(n > 12U) ? p[12] : 0U, (n > 13U) ? p[13] : 0U,
				(n > 14U) ? p[14] : 0U, (n > 15U) ? p[15] : 0U);
	}

	if ((raw_output_info != NULL) && (LL_Buffer_addr_start(raw_output_info) != NULL)) {
		raw_output_value = *(const int8_t *) LL_Buffer_addr_start(raw_output_info);
	}

	{
		char raw_output_line[64] = { 0 };

		(void) snprintf(raw_output_line, sizeof(raw_output_line),
				"[AI] Raw output int8: %d\r\n", (int) raw_output_value);
		DebugConsole_WriteString(raw_output_line);
	}

	DebugConsole_Printf(
			"[AI] Raw tensor meta: name=%s addr=%p len=%lu\r\n",
			(raw_output_info != NULL) ? raw_output_info->name : "(missing)",
			(raw_output_info != NULL) ? (void *) LL_Buffer_addr_start(raw_output_info) : NULL,
			(raw_output_info != NULL) ? (unsigned long) LL_Buffer_len(raw_output_info) : 0UL);

	if ((scale_info != NULL) && (LL_Buffer_addr_start(scale_info) != NULL)) {
		(void) memcpy(&output_dequant_scale, LL_Buffer_addr_start(scale_info),
				sizeof(output_dequant_scale));
	}

	if ((zero_point_info != NULL)
			&& (LL_Buffer_addr_start(zero_point_info) != NULL)) {
		output_zero_point = *(const int8_t *) LL_Buffer_addr_start(
				zero_point_info);
	}

	if (raw_output_info != NULL) {
		const void *raw_output_addr = LL_Buffer_addr_start(raw_output_info);

		if (raw_output_addr != NULL) {
			raw_output_value = *(const int8_t *) raw_output_addr;
			if ((raw_output_info->scale != NULL)
					&& (raw_output_info->offset != NULL)) {
				(void) memcpy(&head_scale, raw_output_info->scale,
						sizeof(head_scale));
				head_zero_point = *(const int16_t *) raw_output_info->offset;
			}
			head_dequant_value = ((float) raw_output_value
					- (float) head_zero_point) * head_scale;
		}
	}

	output_value = AppAI_TraceAndApplyInferenceCalibration(output_value);
	if (!AppAI_IsFiniteFloat(output_value)) {
		AppAI_LogFloatApprox("[AI] Inference output value: ", output_value);
		DebugConsole_WriteString(
				"[AI] Inference result is non-finite; skipping smoothing and last-result update.\r\n");
		app_ai_last_inference_valid = false;
		return;
	}
	output_value = AppAI_FilterInferenceValue(output_value);

	/* Log both the final float output and the raw int8 tensor so we can spot
	 * quantization mismatches without changing the model result path. */
	DebugConsole_Printf(
			"[AI] raw=%d head_zp=%d output_bits=0x%08lx output_zp=%d\r\n",
			(int) raw_output_value,
			(int) head_zero_point,
			(unsigned long) output_bits.u,
			(int) output_zero_point);
	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		AppAI_LogFloatApprox("[AI] head_scale: ", head_scale);
		AppAI_LogFloatApprox("[AI] head_dequant: ", head_dequant_value);
		AppAI_LogFloatApprox("[AI] output_scale: ", output_dequant_scale);
		AppAI_LogFloatApprox("[AI] output_zero_point: ",
				(float) output_zero_point);
		AppAI_LogFloatApprox("[AI] Inference output value: ", output_value);
	}

	app_ai_last_inference_value = output_value;
	app_ai_last_inference_valid = true;
}
#else
static void AppAI_LogInferenceResult(
		const LL_Buffer_InfoTypeDef *output_buffer_info) {
	float output_value = 0.0f;
	char line[64] = { 0 };

	if ((output_buffer_info == NULL)
			|| (LL_Buffer_addr_start(output_buffer_info) == NULL)
			|| (LL_Buffer_len(output_buffer_info) < sizeof(output_value))) {
		return;
	}

	(void) memcpy(&output_value, LL_Buffer_addr_start(output_buffer_info),
			sizeof(output_value));
	output_value = AppAI_TraceAndApplyInferenceCalibration(output_value);
	if (!AppAI_IsFiniteFloat(output_value)) {
		AppInferenceLog_FormatFloatMicros(line, sizeof(line),
				"[AI] Inference value: ", output_value);
		DebugConsole_WriteString(line);
		AppInferenceLog_FormatFloatMicros(line, sizeof(line),
				"[AI] Inference exact: ", output_value);
		DebugConsole_WriteString(line);
		DebugConsole_WriteString(
				"[AI] Inference result is non-finite; skipping smoothing and last-result update.\r\n");
		app_ai_last_inference_valid = false;
		return;
	}
	output_value = AppAI_FilterInferenceValue(output_value);

	AppInferenceLog_FormatFloatMicros(line, sizeof(line),
			"[AI] Inference value: ", output_value);
	DebugConsole_WriteString(line);

	app_ai_last_inference_value = output_value;
	app_ai_last_inference_valid = true;
}
#endif /* APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS */

/**
 * @brief Smooth the user-facing inference value across captures.
 *
 * A 3-frame burst median keeps the live reading from jumping on glare or a
 * single noisy capture while still tracking slow changes.
 */
static float AppAI_FilterInferenceValue(float value) {
	float ordered[APP_AI_INFERENCE_BURST_HISTORY_SIZE] = { 0.0f };
	size_t sample_count = 0U;
	size_t start_index = 0U;

	if (!AppAI_IsFiniteFloat(value)) {
		if (app_ai_inference_burst_history_count > 0U) {
			const size_t last_index =
					(app_ai_inference_burst_history_next_index
							+ APP_AI_INFERENCE_BURST_HISTORY_SIZE - 1U)
					% APP_AI_INFERENCE_BURST_HISTORY_SIZE;
			const float last_value =
					app_ai_inference_burst_history[last_index];

			if (AppAI_IsFiniteFloat(last_value)) {
				return last_value;
			}
		}
		return value;
	}

	if (app_ai_inference_burst_history_count > 0U) {
		const size_t last_index =
				(app_ai_inference_burst_history_next_index
						+ APP_AI_INFERENCE_BURST_HISTORY_SIZE - 1U)
				% APP_AI_INFERENCE_BURST_HISTORY_SIZE;
		float delta = value - app_ai_inference_burst_history[last_index];

		if (delta < 0.0f) {
			delta = -delta;
		}

		if (delta > APP_AI_INFERENCE_BURST_RESET_DELTA_C) {
			AppAI_ResetInferenceBurstHistory();
		}
	}

	app_ai_inference_burst_history[app_ai_inference_burst_history_next_index] =
			value;
	if (app_ai_inference_burst_history_count <
			APP_AI_INFERENCE_BURST_HISTORY_SIZE) {
		app_ai_inference_burst_history_count++;
	}
	app_ai_inference_burst_history_next_index =
			(app_ai_inference_burst_history_next_index + 1U)
			% APP_AI_INFERENCE_BURST_HISTORY_SIZE;

	sample_count = app_ai_inference_burst_history_count;
	start_index = (sample_count < APP_AI_INFERENCE_BURST_HISTORY_SIZE) ? 0U
			: app_ai_inference_burst_history_next_index;

	for (size_t i = 0U; i < sample_count; ++i) {
		ordered[i] =
				app_ai_inference_burst_history[(start_index + i)
						% APP_AI_INFERENCE_BURST_HISTORY_SIZE];
	}

	/* Keep the local copy sorted so we can choose the median for the full
	 * 3-sample burst and a simple average during warm-up. */
	for (size_t i = 1U; i < sample_count; ++i) {
		const float key = ordered[i];
		size_t j = i;

		while ((j > 0U) && (ordered[j - 1U] > key)) {
			ordered[j] = ordered[j - 1U];
			--j;
		}

		ordered[j] = key;
	}

	if (sample_count == 1U) {
		return ordered[0U];
	}

	if (sample_count == 2U) {
		return 0.5f * (ordered[0U] + ordered[1U]);
	}

	return ordered[sample_count / 2U];
}

/**
 * @brief Return true only for ordinary finite float values.
 */
static bool AppAI_IsFiniteFloat(float value) {
	union {
		float f;
		uint32_t u;
	} bits = {
		.f = value
	};

	return ((bits.u & 0x7F800000U) != 0x7F800000U);
}

static int AppAI_ApplyCacheRange(uint32_t start_addr, uint32_t end_addr,
		bool clean, bool invalidate) {
	uintptr_t start = 0U;
	uintptr_t end = 0U;

	if (end_addr <= start_addr) {
		return -1;
	}

	start = (uintptr_t) start_addr;
	end = (uintptr_t) end_addr;

	start &= ~((uintptr_t) APP_AI_CACHE_LINE_BYTES - 1U);
	end = (end + APP_AI_CACHE_LINE_BYTES - 1U)
			& ~((uintptr_t) APP_AI_CACHE_LINE_BYTES - 1U);

	if (clean) {
		SCB_CleanDCache_by_Addr((uint32_t *) start, (int32_t) (end - start));
	}

	if (invalidate) {
		SCB_InvalidateDCache_by_Addr((uint32_t *) start,
				(int32_t) (end - start));
	}

	return 0;
}

static bool AppAI_PreprocessYuv422FrameToFloatInput(const uint8_t *frame_bytes,
		size_t frame_size, float *input_ptr, size_t input_float_count) {
	const size_t source_width = (size_t) APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const size_t source_height = (size_t) APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	size_t crop_x_min = 0U;
	size_t crop_y_min = 0U;
	size_t crop_width = source_width;
	size_t crop_height = source_height;
	bool crop_found = false;
	const char *crop_label = "fixed";

	if ((frame_bytes == NULL) || (input_ptr == NULL)) {
		
		return false;
	}

	if ((frame_size < (size_t) APP_AI_CAPTURE_FRAME_BYTES)
			|| (input_float_count < (size_t) APP_AI_MODEL_INPUT_FLOAT_COUNT)) {
		
		return false;
	}

	/* DCMIPP_PIXEL_PACKER_FORMAT_YUV422_1 emits packed YUYV samples:
	 *   byte 0 = Y0, byte 1 = U, byte 2 = Y1, byte 3 = V, ...
	 * We estimate the gauge position from the bright dial face, crop around
	 * that box, and then resize with padding to match the model input. */
	if (app_ai_forced_crop_active) {
		crop_found = true;
		crop_label = (app_ai_forced_crop_label != NULL)
				? app_ai_forced_crop_label : "forced";
		crop_x_min = app_ai_forced_crop_x_min;
		crop_y_min = app_ai_forced_crop_y_min;
		crop_width = app_ai_forced_crop_width;
		crop_height = app_ai_forced_crop_height;
	} else {
#if APP_AI_USE_ADAPTIVE_GAUGE_CROP
		crop_found = AppAI_EstimateGaugeCropBoxFromYuv422(frame_bytes, frame_size,
				source_width, source_height, &crop_x_min, &crop_y_min,
				&crop_width, &crop_height);
#endif
		if (crop_found) {
			crop_label = "adaptive";
		}
	}
	if (!crop_found) {
		crop_x_min = (size_t) ((float) source_width
				* APP_AI_TRAINING_CROP_X_MIN_RATIO);
		crop_y_min = (size_t) ((float) source_height
				* APP_AI_TRAINING_CROP_Y_MIN_RATIO);
		crop_width = (size_t) ((float) source_width
				* (APP_AI_TRAINING_CROP_X_MAX_RATIO
				- APP_AI_TRAINING_CROP_X_MIN_RATIO));
		crop_height = (size_t) ((float) source_height
				* (APP_AI_TRAINING_CROP_Y_MAX_RATIO
				- APP_AI_TRAINING_CROP_Y_MIN_RATIO));
		if (crop_width == 0U) {
			crop_width = 1U;
		}
		if (crop_height == 0U) {
			crop_height = 1U;
		}
	}

	{
		char crop_line[128] = { 0 };

		(void) snprintf(crop_line, sizeof(crop_line),
				"[AI] Crop %s: x=%lu y=%lu w=%lu h=%lu\r\n",
				crop_label,
				(unsigned long) crop_x_min, (unsigned long) crop_y_min,
				(unsigned long) crop_width, (unsigned long) crop_height);
		DebugConsole_WriteString(crop_line);
	}

	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS) {
		AppAI_LogSourceCropWindow(frame_bytes, frame_size, source_width,
				source_height, crop_x_min, crop_y_min, crop_width, crop_height);
	}

	(void) memset(input_ptr, 0, input_float_count * sizeof(float));

	{
		const size_t output_width = (size_t) APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
		const size_t output_height = (size_t) APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
		const float width_scale = (float) output_width / (float) crop_width;
		const float height_scale = (float) output_height / (float) crop_height;
		const float scale = (width_scale < height_scale) ? width_scale
				: height_scale;
		const size_t scaled_width = (size_t) ((float) crop_width * scale + 0.5f);
		const size_t scaled_height = (size_t) ((float) crop_height * scale + 0.5f);
		const size_t pad_x = (output_width > scaled_width) ?
				((output_width - scaled_width) / 2U) : 0U;
		const size_t pad_y = (output_height > scaled_height) ?
				((output_height - scaled_height) / 2U) : 0U;
		const size_t crop_x_max_index = (crop_width > 0U) ? (crop_width - 1U) : 0U;
		const size_t crop_y_max_index = (crop_height > 0U) ? (crop_height - 1U) : 0U;
		const size_t probe_top_y = (pad_y + (scaled_height / 4U) < output_height) ?
				(pad_y + (scaled_height / 4U)) : (output_height - 1U);
		const size_t probe_mid_y = (pad_y + (scaled_height / 2U) < output_height) ?
				(pad_y + (scaled_height / 2U)) : (output_height - 1U);
		const size_t probe_bottom_y = (pad_y + ((scaled_height * 3U) / 4U) <
				output_height) ? (pad_y + ((scaled_height * 3U) / 4U))
				: (output_height - 1U);
		size_t nonzero_write_count = 0U;

#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
		DebugConsole_Printf(
				"[AI] Resize plan: output=%lux%lu scale=%lu.%03lu scaled=%lux%lu pad=(%lu,%lu) crop_max=(%lu,%lu) probes=(%lu,%lu,%lu)\r\n",
				(unsigned long) output_width, (unsigned long) output_height,
				(unsigned long) (scale),
				(unsigned long) ((scale - (float) ((unsigned long) scale)) * 1000.0f),
				(unsigned long) scaled_width, (unsigned long) scaled_height,
				(unsigned long) pad_x, (unsigned long) pad_y,
				(unsigned long) crop_x_max_index, (unsigned long) crop_y_max_index,
				(unsigned long) probe_top_y, (unsigned long) probe_mid_y,
				(unsigned long) probe_bottom_y);
#endif

		for (size_t out_y = 0U; out_y < output_height; out_y++) {
			if ((out_y < pad_y) || (out_y >= (pad_y + scaled_height))) {
				continue;
			}

			const float crop_y_f = ((float) (out_y - pad_y)) / scale;
			size_t crop_y0 = (size_t) crop_y_f;
			size_t crop_y1 = (crop_y0 < crop_y_max_index) ? (crop_y0 + 1U)
					: crop_y_max_index;
			float crop_y_frac = crop_y_f - (float) crop_y0;

			if (crop_y0 >= crop_y_max_index) {
				crop_y0 = crop_y_max_index;
				crop_y1 = crop_y_max_index;
				crop_y_frac = 0.0f;
			}

			for (size_t out_x = 0U; out_x < output_width; out_x++) {
				const size_t dest_pixel_index = (out_y * output_width) + out_x;
				float crop_x_frac = 0.0f;
				float crop_x_f = 0.0f;
				size_t crop_x0 = 0U;
				size_t crop_x1 = 0U;
				float r00 = 0.0f;
				float g00 = 0.0f;
				float b00 = 0.0f;
				float r10 = 0.0f;
				float g10 = 0.0f;
				float b10 = 0.0f;
				float r01 = 0.0f;
				float g01 = 0.0f;
				float b01 = 0.0f;
				float r11 = 0.0f;
				float g11 = 0.0f;
				float b11 = 0.0f;
				float top_r = 0.0f;
				float top_g = 0.0f;
				float top_b = 0.0f;
				float bottom_r = 0.0f;
				float bottom_g = 0.0f;
				float bottom_b = 0.0f;
				float out_r = 0.0f;
				float out_g = 0.0f;
				float out_b = 0.0f;

				if ((out_x < pad_x) || (out_x >= (pad_x + scaled_width))) {
					continue;
				}

				crop_x_f = ((float) (out_x - pad_x)) / scale;
				crop_x0 = (size_t) crop_x_f;
				crop_x1 = (crop_x0 < crop_x_max_index) ? (crop_x0 + 1U)
						: crop_x_max_index;
				crop_x_frac = crop_x_f - (float) crop_x0;

				if (crop_x0 >= crop_x_max_index) {
					crop_x0 = crop_x_max_index;
					crop_x1 = crop_x_max_index;
					crop_x_frac = 0.0f;
				}

				AppAI_ReadRgbFromYuv422Pixel(frame_bytes, source_width,
						crop_x_min + crop_x0, crop_y_min + crop_y0, &r00, &g00,
						&b00);
				AppAI_ReadRgbFromYuv422Pixel(frame_bytes, source_width,
						crop_x_min + crop_x1, crop_y_min + crop_y0, &r10, &g10,
						&b10);
				AppAI_ReadRgbFromYuv422Pixel(frame_bytes, source_width,
						crop_x_min + crop_x0, crop_y_min + crop_y1, &r01, &g01,
						&b01);
				AppAI_ReadRgbFromYuv422Pixel(frame_bytes, source_width,
						crop_x_min + crop_x1, crop_y_min + crop_y1, &r11, &g11,
						&b11);

				top_r = r00 + ((r10 - r00) * crop_x_frac);
				top_g = g00 + ((g10 - g00) * crop_x_frac);
				top_b = b00 + ((b10 - b00) * crop_x_frac);
				bottom_r = r01 + ((r11 - r01) * crop_x_frac);
				bottom_g = g01 + ((g11 - g01) * crop_x_frac);
				bottom_b = b01 + ((b11 - b01) * crop_x_frac);
				out_r = top_r + ((bottom_r - top_r) * crop_y_frac);
				out_g = top_g + ((bottom_g - top_g) * crop_y_frac);
				out_b = top_b + ((bottom_b - top_b) * crop_y_frac);

				input_ptr[dest_pixel_index * 3U + 0U] =
						AppAI_ClampNormalizedFloat(out_r);
				input_ptr[dest_pixel_index * 3U + 1U] =
						AppAI_ClampNormalizedFloat(out_g);
				input_ptr[dest_pixel_index * 3U + 2U] =
						AppAI_ClampNormalizedFloat(out_b);

				if ((out_r > 0.0f) || (out_g > 0.0f) || (out_b > 0.0f)) {
					nonzero_write_count++;
				}

			}
		}

	#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
		DebugConsole_Printf(
				"[AI] Preprocess write summary: nonzero_pixels=%lu\r\n",
				(unsigned long) nonzero_write_count);
#endif /* APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS */
	}

	return true;
}

static float AppAI_ClampNormalizedFloat(float value) {
	if (value < 0.0f) {
		return 0.0f;
	}

	if (value > 1.0f) {
		return 1.0f;
	}

	return value;
}

static uint8_t AppAI_ReadYuv422Luma(const uint8_t *frame_bytes,
		size_t frame_width_pixels, size_t source_x, size_t source_y) {
	const size_t pair_x = source_x & ~1U;
	const size_t source_index = ((source_y * frame_width_pixels) + pair_x) * 2U;
	const bool is_second_pixel = ((source_x & 1U) != 0U);

	if (frame_bytes == NULL) {
		return 0U;
	}

	return frame_bytes[source_index + (is_second_pixel ? 2U : 0U)];
}

static void AppAI_ReadYuv422Quartet(const uint8_t *frame_bytes,
		size_t frame_width_pixels, size_t source_x, size_t source_y,
		uint8_t *quad_out) {
	const size_t pair_x = source_x & ~1U;
	const size_t source_index = ((source_y * frame_width_pixels) + pair_x) * 2U;

	if ((frame_bytes == NULL) || (quad_out == NULL)) {
		return;
	}

	quad_out[0] = frame_bytes[source_index + 0U];
	quad_out[1] = frame_bytes[source_index + 1U];
	quad_out[2] = frame_bytes[source_index + 2U];
	quad_out[3] = frame_bytes[source_index + 3U];
}

static float AppAI_ReadNormalizedGrayFromYuv422Pixel(const uint8_t *frame_bytes,
		size_t frame_width_pixels, size_t source_x, size_t source_y) {
	const uint8_t luma = AppAI_ReadYuv422Luma(frame_bytes, frame_width_pixels,
			source_x, source_y);
	const float normalized = ((float) luma) / 255.0f;

	return AppAI_ClampNormalizedFloat(normalized);
}

static void AppAI_ReadRgbFromYuv422Pixel(const uint8_t *frame_bytes,
		size_t frame_width_pixels, size_t source_x, size_t source_y,
		float *r_out, float *g_out, float *b_out) {
#if APP_AI_YUV422_INPUT_LUMA_ONLY
	const float gray = AppAI_ReadNormalizedGrayFromYuv422Pixel(frame_bytes,
			frame_width_pixels, source_x, source_y);
	const float r = gray;
	const float g = gray;
	const float b = gray;
#else
	const size_t pair_x = source_x & ~1U;
	const size_t source_index = ((source_y * frame_width_pixels) + pair_x) * 2U;
	const bool is_second_pixel = ((source_x & 1U) != 0U);
	const float y = ((float) frame_bytes[source_index + (is_second_pixel ? 2U : 0U)]
			- 16.0f) * 1.1643836f;
	const float u = (float) frame_bytes[source_index + 1U] - 128.0f;
	const float v = (float) frame_bytes[source_index + 3U] - 128.0f;
	const float r = (y + (1.5960268f * v)) / 255.0f;
	const float g = (y - (0.3917623f * u) - (0.8129677f * v)) / 255.0f;
	const float b = (y + (2.0172322f * u)) / 255.0f;
#endif

	if (r_out != NULL) {
		*r_out = AppAI_ClampNormalizedFloat(r);
	}
	if (g_out != NULL) {
		*g_out = AppAI_ClampNormalizedFloat(g);
	}
	if (b_out != NULL) {
		*b_out = AppAI_ClampNormalizedFloat(b);
	}
}

bool App_AI_GetLastInferenceResult(float *value_out) {
	if (value_out == NULL) {
		return false;
	}
	if (!app_ai_last_inference_valid) {
		return false;
	}
	if (!AppAI_IsFiniteFloat(app_ai_last_inference_value)) {
		return false;
	}
	*value_out = app_ai_last_inference_value;
	return true;
}

/* USER CODE END 1 */
