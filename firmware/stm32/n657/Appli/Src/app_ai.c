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
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#include "debug_console.h"
#include "app_inference_calibration.h"
#include "app_baseline_runtime.h"
#include "app_inference_log_config.h"
#include "app_inference_log_utils.h"
#include "app_memory_budget.h"
#include "app_gauge_geometry.h"
#include "app_inner_celsius_mask.h"
#include "ina219_power.h"
#include "inference_metrics.h"
#include "app_center_detector.h"
#include "app_ai_config.h"
#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX
#include "app_ai_types.h"
#include "app_ai_state.h"
#include "app_ai_logging.h"
#include "app_ai_xspi2.h"
#include "app_ai_preprocess.h"
#include "app_ai_inference.h"
#include "app_ai_stage_obb.h"
#include "app_ai_stage_tip_focus.h"
#include "tx_api.h"
#include "ll_aton_rt_user_api.h"
#include "ll_aton.h"
#include "ll_aton_runtime.h"
#include "ll_aton_reloc_network.h"
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

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */


/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
bool app_ai_runtime_initialized = false;
bool app_ai_aton_runtime_initialized = false;
volatile float app_ai_last_inference_value = 0.0f;
volatile bool app_ai_last_inference_valid = false;
/* EMA-smoothed OBB centre to dampen single-frame x-center jitter. */
float app_ai_smoothed_obb_cx = -1.0f;
float app_ai_smoothed_obb_cy = -1.0f;
#define APP_AI_OBB_CENTER_EMA_ALPHA 0.20f
#if APP_AI_ENABLE_INFERENCE_BURST_SMOOTHING
float app_ai_inference_burst_history
	[APP_AI_INFERENCE_BURST_HISTORY_SIZE] = {0.0f};
size_t app_ai_inference_burst_history_count = 0U;
size_t app_ai_inference_burst_history_next_index = 0U;
#endif
bool app_ai_npu_hw_initialized = false;
bool app_ai_xspi2_initialized = false;
bool app_ai_xspi2_mm_enabled = false;
const struct AppAI_ModelStageSpec *app_ai_loaded_xspi2_stage = NULL;
float app_ai_tip_focus_median_buffer[APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE] = {0.0f};
size_t app_ai_tip_focus_median_count = 0U;
size_t app_ai_tip_focus_median_index = 0U;
float app_ai_tip_focus_last_published = 0.0f;
bool app_ai_tip_focus_last_published_valid = false;
uint32_t app_ai_tip_focus_consecutive_invalid = 0U;
uint32_t app_ai_tip_focus_outlier_streak = 0U;
bool app_ai_forced_crop_active = false;
size_t app_ai_forced_crop_x_min = 0U;
size_t app_ai_forced_crop_y_min = 0U;
size_t app_ai_forced_crop_width = 0U;
size_t app_ai_forced_crop_height = 0U;
const char *app_ai_forced_crop_label = NULL;
bool app_ai_tip_focus_input_dump_done = false;
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
 *   _mem_pool_xSPI2_obb_box_board_bbox_deploy_candidate — weight-addressing.
 * 32-byte placeholder ensures the symbol resolves at link time.
 * Actual weight blob (~664 KiB, obb_box_board_bbox_deploy_candidate_atonbuf.xSPI2.raw) is
 * flashed separately via flash_boot.ps1. */
__attribute__((section(".xspi2_obb_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI2_obb_box_board_bbox_deploy_candidate[32U] = {
	0U,
};

/* The current board-bbox OBB package still declares an xSPI1 pool anchor
 * symbol, but the generated C does not actually reference it for this build.
 * Keep the linker marker so the symbol resolves cleanly, but do not treat it
 * as proof that the live deploy candidate requires physical HyperRAM. */
__attribute__((section(".xspi1_obb_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI1_obb_box_board_bbox_deploy_candidate[32U] = {
	0U,
};


/* Source-crop-box model pool (mobilenetv2_source_crop_box_v1_stripped_int8). */

__attribute__((section(".xspi2_source_crop_box_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))

uint8_t _mem_pool_xSPI2_mobilenetv2_source_crop_box_v1_stripped_int8[32U] = {

	0U,

};
#endif

/* Live OBB pool symbols stay unconditional because the tip-focus build still
 * links the generated OBB relocations even though the old scalar branch is off. */
__attribute__((section(".xspi2_obb_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI2_obb_box_board_bbox_deploy_candidate[32U] = { 0U, };
__attribute__((section(".xspi1_obb_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI1_obb_box_board_bbox_deploy_candidate[32U] = { 0U, };

/* Tip-focus v18 model pool. The generated network keeps the xSPI2 pool
 * symbol alive at the address where the NPU weight blob
 * (tip_focus_v18_int8_atonbuf.xSPI2.raw from the NPU package) is flashed
 * (0x70400000).
 *
 * IMPORTANT: The actual weights data (about 2.1MB) is NOT stored in this array.
 * The data lives in xSPI2 flash at 0x70400000, and must be flashed using
 * flash_boot.ps1 before running inference. This 32-byte symbol is just a
 * linker marker that gets placed at 0x70400000 by the .xspi2_tip_focus_pool
 * section. When the NPU accesses weights, it reads from xSPI2 flash through
 * the memory-mapped window (0x70000000+), not from this RAM array.
 *
 * If you see a HardFault at address 0x8D or similar during inference, it
 * means the xSPI2 flash was not programmed. Run flash_boot.ps1 to flash
 * the tip_focus_v18_int8_atonbuf.xSPI2.raw blob to 0x70400000. */
__attribute__((section(".xspi2_tip_focus_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI2_tip_focus_v18_int8[32U] = { 0U, };
/* The board-fit tip-focus package does not need external activations, but it
 * still declares the xSPI1 anchor symbol with size 0. Keep the linker marker
 * so the relocatable build resolves cleanly without implying HyperRAM use. */
__attribute__((section(".xspi1_tip_focus_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI1_tip_focus_v18_int8[32U] = { 0U, };
uint8_t app_ai_xspi2_program_buffer[APP_AI_XSPI2_PROGRAM_CHUNK_BYTES];
__attribute__((aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t app_ai_scalar_row_scratch[APP_AI_CAPTURE_FRAME_WIDTH_PIXELS * APP_AI_CAPTURE_FRAME_BYTES_PER_PIXEL];
__attribute__((aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t app_ai_scalar_output_row_scratch[APP_AI_CAPTURE_FRAME_WIDTH_PIXELS * 3U * sizeof(float)];
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
/* Heatmap center-detector model (DS-CNN v4-S): 332,045 bytes flashed at 0x70200000.
 * Signatures unchanged from DS-CNN v4 — same first/last 16 bytes. */
const uint8_t app_ai_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES] = {
	0x80U, 0x80U, 0x81U, 0x82U, 0x83U, 0x83U, 0x84U, 0x85U,
	0x86U, 0x87U, 0x87U, 0x88U, 0x89U, 0x8AU, 0x8BU, 0x8BU,
};
const uint8_t app_ai_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x80U,
};
/* Rectified scalar v2 xSPI2 signatures used when the board boots with the
 * prod v0.8 scalar blob already flashed at 0x70200000. Size: 3,218,865 bytes. */
const uint8_t app_ai_rectifier_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES] = {
	0x0FU, 0x11U, 0xF8U, 0x10U, 0xD0U, 0xD8U, 0x0EU, 0x28U,
	0x99U, 0xCEU, 0x98U, 0x7DU, 0xBCU, 0x42U, 0x5EU, 0xF2U,
};
const uint8_t app_ai_rectifier_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x80U,
};
/* Board bbox OBB xSPI2 signatures.  Update after running:
 *   python ml/scripts/package_obb_box_board_bbox_deploy_candidate_for_n6.py
 * The script prints the 16 start/tail bytes for this raw blob. */
const uint8_t app_ai_obb_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES] = {
	0x34U, 0x10U, 0x10U, 0x1CU, 0x11U, 0xDFU, 0xFEU, 0x2DU,
	0xD7U, 0xB0U, 0xCFU, 0x0CU, 0x04U, 0x0FU, 0xDBU, 0x0EU,
};
const uint8_t app_ai_obb_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x80U,
};

/* Source-crop-box xSPI2 signatures for atonbuf.source_crop_box.xSPI2.raw. */
const uint8_t app_ai_source_crop_box_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES] = {
	0xF2U, 0x17U, 0x29U, 0xE2U, 0xDCU, 0xEBU, 0xECU, 0x04U,
	0x09U, 0x01U, 0x35U, 0xEBU, 0x14U, 0xDEU, 0x0FU, 0x02U,
};
const uint8_t app_ai_source_crop_box_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x80U,
};
/* Tip-focus v18 xSPI2 signatures for tip_focus_v18_int8_atonbuf.xSPI2.raw.
 * Flashed to 0x70400000. Size: 834,465 bytes. */
const uint8_t app_ai_tip_focus_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES] = {
	0x48U, 0x0AU, 0x13U, 0xC1U, 0x26U, 0xF5U, 0xFCU, 0xE5U,
	0xE9U, 0x37U, 0x1DU, 0xFAU, 0xEAU, 0xC2U, 0xEDU, 0x04U,
};
const uint8_t app_ai_tip_focus_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x80U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
};
/* The shared xSPI2 helper paths should verify against the live UNet blob. */
#define app_ai_xspi2_signature_start app_ai_tip_focus_xspi2_signature_start
#define app_ai_xspi2_signature_tail app_ai_tip_focus_xspi2_signature_tail
/* Per-stage programmed sizes. Set during provisioning and used by the verify
 * functions for the tail probe offset. Keeping them separate prevents the
 * scalar tail check from using the rectifier's file size (or vice-versa) when
 * stages alternate. */
ULONG app_ai_scalar_programmed_size = 0UL;
ULONG app_ai_rectifier_programmed_size = 0UL;
ULONG app_ai_obb_programmed_size = 0UL;
ULONG app_ai_source_crop_box_programmed_size = 0UL;

/* Per-stage signature caches populated from the SD file during provisioning.
 * Using SD-sourced bytes means verify never goes stale when the model blob is
 * replaced, regardless of what the hardcoded fallback constants say. */
uint8_t app_ai_scalar_sig_start[APP_AI_XSPI2_PROBE_BYTES] = {0U};
uint8_t app_ai_scalar_sig_tail[APP_AI_XSPI2_PROBE_BYTES] = {0U};
bool app_ai_scalar_sig_valid = false;
uint8_t app_ai_rectifier_sig_start[APP_AI_XSPI2_PROBE_BYTES] = {0U};
uint8_t app_ai_rectifier_sig_tail[APP_AI_XSPI2_PROBE_BYTES] = {0U};
bool app_ai_rectifier_sig_valid = false;
uint8_t app_ai_obb_sig_start[APP_AI_XSPI2_PROBE_BYTES] = {0U};
uint8_t app_ai_obb_sig_tail[APP_AI_XSPI2_PROBE_BYTES] = {0U};
bool app_ai_obb_sig_valid = false;
uint8_t app_ai_source_crop_box_sig_start[APP_AI_XSPI2_PROBE_BYTES] = {0U};
uint8_t app_ai_source_crop_box_sig_tail[APP_AI_XSPI2_PROBE_BYTES] = {0U};
bool app_ai_source_crop_box_sig_valid = false;
/* Legacy alias kept for the shared xSPI2 logging helpers; the live tip-focus
 * build still expects this size tracker to exist. */
ULONG app_ai_xspi2_programmed_size = 0UL;
ULONG app_ai_tip_focus_programmed_size = 0UL;
uint8_t app_ai_tip_focus_sig_start[APP_AI_XSPI2_PROBE_BYTES] = {0U};
uint8_t app_ai_tip_focus_sig_tail[APP_AI_XSPI2_PROBE_BYTES] = {0U};
bool app_ai_tip_focus_sig_valid = false;

/* NN_Instance variables are now defined in the generated model files and
 * declared extern in app_ai_state.h.  Keep them out of this TU so the
 * linker resolves a single definition per symbol. */

/* OBB reloc helper exported by the wrapper thunk so the stage can install
 * its runtime base before the ATON resize helper runs. */
extern bool AppAI_Obb_InstallRelocContext(NN_Instance_TypeDef *instance, uintptr_t xspi2_base_addr);

const AppAI_ModelStageSpec app_ai_obb_stage = {
	.stage_label = "obb_box_board_bbox_deploy_candidate",
	.model_image_path = APP_AI_OBB_XSPI2_MODEL_IMAGE_PATH,
	.nn_instance = &NN_Instance_obb_box_board_bbox_deploy_candidate,
	.network_init_fn = LL_ATON_EC_Network_Init_obb_box_board_bbox_deploy_candidate,
	.inference_init_fn = LL_ATON_EC_Inference_Init_obb_box_board_bbox_deploy_candidate,
	.uses_rectifier_box = false,
	.xspi2_chip_offset = APP_AI_XSPI2_OBB_CHIP_OFFSET,
	.xspi2_base_addr = APP_AI_XSPI2_OBB_BASE_ADDR,
};

bool AppAI_ShouldLogStageDiagnostics(
	const AppAI_ModelStageSpec *stage)
{
	(void)stage;
	return true;
}

/* Forward declarations for the local helpers that the board-bbox decoder uses
 * before their definitions later in this translation unit. */
bool AppAI_IsFiniteFloat(float value);
float AppAI_ClampNormalizedFloat(float value);
float AppAI_ObbDequantize(int8_t q_value);
bool AppAI_BuildObbDecodeCandidate(
	float x_min, float y_min, float x_max, float y_max,
	AppAI_ObbDecodeCandidate *candidate);
void AppAI_LogObbDecodeDiagnostics(
	float score,
	float raw0,
	float raw1,
	float raw2,
	float raw3,
	const AppAI_ObbDecodeCandidate *corners_candidate,
	const AppAI_ObbDecodeCandidate *center_size_candidate,
	const char *selected_label);
bool AppAI_DumpTipFocusInputTensorOnce(
	const float *input_ptr,
	size_t output_width,
	size_t output_height,
	const char *crop_label,
	const AppAI_SourceCrop *crop_ptr,
	bool obb_crop_valid,
	const AppAI_ObbBox *obb_box);

/* Board-bbox bridge for the deployed OBB candidate.
 * output_buffers_info array: [0]=confidence scalar, [1]=bbox vector, [2]=NULL terminator.
 * The bbox vector has to survive both the historical corner-style export and
 * the newer center/size-style export, so we decode both layouts and keep the
 * first plausible crop we can prove is finite and non-degenerate. */
bool AppAI_DecodeQarepvggObb(
    const LL_Buffer_InfoTypeDef *output_info,
    AppAI_SourceCrop            *obb_crop,
    AppAI_ObbBox                *obb_box)
{
	if ((output_info == NULL) || (obb_crop == NULL) || (obb_box == NULL))
	{
		return false;
	}

	const LL_Buffer_InfoTypeDef *score_info = &output_info[0U];
	const LL_Buffer_InfoTypeDef *box_info = &output_info[1U];
	const int8_t *score_raw = NULL;
	const int8_t *box_raw = NULL;
	float score = 0.0f;
	float raw0 = 0.0f;
	float raw1 = 0.0f;
	float raw2 = 0.0f;
	float raw3 = 0.0f;
	float center_x = 0.0f;
	float center_y = 0.0f;
	float box_w = 0.0f;
	float box_h = 0.0f;
	float center_box_x_min = 0.0f;
	float center_box_y_min = 0.0f;
	float center_box_x_max = 0.0f;
	float center_box_y_max = 0.0f;
	AppAI_ObbDecodeCandidate corners_candidate = {0};
	AppAI_ObbDecodeCandidate center_size_candidate = {0};
	const char *selected_label = "none";
	bool corners_valid = false;
	bool center_size_valid = false;

	/* Verify both outputs are present and large enough for the scalar + bbox head. */
	if ((score_info->name == NULL) || (box_info->name == NULL))
	{
		return false;
	}
	if ((LL_Buffer_len(score_info) < 1U) || (LL_Buffer_len(box_info) < 4U))
	{
		return false;
	}

	score_raw = (const int8_t *)LL_Buffer_addr_start(score_info);
	box_raw = (const int8_t *)LL_Buffer_addr_start(box_info);
	if ((score_raw == NULL) || (box_raw == NULL))
	{
		return false;
	}

	score = AppAI_ClampNormalizedFloat(AppAI_ObbDequantize(score_raw[0]));
	raw0 = AppAI_ClampNormalizedFloat(AppAI_ObbDequantize(box_raw[0]));
	raw1 = AppAI_ClampNormalizedFloat(AppAI_ObbDequantize(box_raw[1]));
	raw2 = AppAI_ClampNormalizedFloat(AppAI_ObbDequantize(box_raw[2]));
	raw3 = AppAI_ClampNormalizedFloat(AppAI_ObbDequantize(box_raw[3]));

	if (!AppAI_IsFiniteFloat(score) || !AppAI_IsFiniteFloat(raw0) ||
		!AppAI_IsFiniteFloat(raw1) || !AppAI_IsFiniteFloat(raw2) ||
		!AppAI_IsFiniteFloat(raw3))
	{
		return false;
	}

	if (score < 0.05f)
	{
		return false;
	}

	/* First try the label-source contract: normalized x_min, y_min, x_max, y_max. */
	corners_valid = AppAI_BuildObbDecodeCandidate(
		raw0, raw1, raw2, raw3, &corners_candidate);

	/* If the export was center/size, recover the actual corners from the same
	 * 4-vector rather than rejecting the model output outright. */
	center_box_x_min = raw0 - (0.5f * raw2);
	center_box_y_min = raw1 - (0.5f * raw3);
	center_box_x_max = raw0 + (0.5f * raw2);
	center_box_y_max = raw1 + (0.5f * raw3);
	if (center_box_x_max < center_box_x_min)
	{
		const float tmp = center_box_x_min;
		center_box_x_min = center_box_x_max;
		center_box_x_max = tmp;
	}
	if (center_box_y_max < center_box_y_min)
	{
		const float tmp = center_box_y_min;
		center_box_y_min = center_box_y_max;
		center_box_y_max = tmp;
	}

	center_size_valid = AppAI_BuildObbDecodeCandidate(
		center_box_x_min, center_box_y_min,
		center_box_x_max, center_box_y_max,
		&center_size_candidate);

	if (!corners_valid && !center_size_valid)
	{
		AppAI_LogObbDecodeDiagnostics(
			score, raw0, raw1, raw2, raw3,
			&corners_candidate, &center_size_candidate, selected_label);
		return false;
	}

	if (corners_valid)
	{
		/* Keep the original corner-style export as the preferred path when it
		 * already produces a sane box. */
		selected_label = "corners";
		center_x = corners_candidate.center_x;
		center_y = corners_candidate.center_y;
		box_w = corners_candidate.box_w;
		box_h = corners_candidate.box_h;
		*obb_crop = corners_candidate.crop;
	}
	else
	{
		selected_label = "center-size";
		center_x = center_size_candidate.center_x;
		center_y = center_size_candidate.center_y;
		box_w = center_size_candidate.box_w;
		box_h = center_size_candidate.box_h;
		*obb_crop = center_size_candidate.crop;
	}
	AppAI_LogObbDecodeDiagnostics(
		score, raw0, raw1, raw2, raw3,
		&corners_candidate, &center_size_candidate, selected_label);

	obb_box->center_x = center_x;
	obb_box->center_y = center_y;
	obb_box->box_w = box_w;
	obb_box->box_h = box_h;
	obb_box->angle_rad = 0.0f;
	obb_box->confidence = score;
	obb_box->gauge_center_x = center_x;
	obb_box->gauge_center_y = center_y;
	return true;
}

/* USER CODE END PV */

bool __attribute__((noinline)) AppAI_PreprocessYuv422FrameToFloatInput(
	const uint8_t *frame_bytes, size_t frame_size, float *input_ptr,
	size_t input_float_count, size_t input_len_bytes,
	size_t output_width, size_t output_height)
{
	const size_t source_width = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const size_t source_height = (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	const size_t required_float_count = output_width * output_height * 3U;
	const size_t required_bytes = required_float_count * sizeof(float);
	size_t crop_x_min = 0U;
	size_t crop_y_min = 0U;
	size_t crop_width = source_width;
	size_t crop_height = source_height;
	bool crop_found = false;
	const char *crop_label = "fixed";

	if ((frame_bytes == NULL) || (input_ptr == NULL))
	{
		return false;
	}
	if ((output_width == 0U) || (output_height == 0U))
	{
		return false;
	}
	if ((frame_size < (size_t)APP_AI_CAPTURE_FRAME_BYTES) ||
		(input_float_count < required_float_count) ||
		(input_len_bytes < required_bytes))
	{
		return false;
	}

	if (app_ai_forced_crop_active)
	{
		crop_found = true;
		crop_label = (app_ai_forced_crop_label != NULL)
						 ? app_ai_forced_crop_label
						 : "forced";
		crop_x_min = app_ai_forced_crop_x_min;
		crop_y_min = app_ai_forced_crop_y_min;
		crop_width = app_ai_forced_crop_width;
		crop_height = app_ai_forced_crop_height;
	}
	else
	{
#if APP_AI_USE_ADAPTIVE_GAUGE_CROP
		crop_found = AppAI_EstimateGaugeCropBoxFromYuv422(
			frame_bytes, frame_size, source_width, source_height,
			&crop_x_min, &crop_y_min, &crop_width, &crop_height);
#endif
		if (crop_found)
		{
			crop_label = "adaptive";
		}
	}

	if (!crop_found)
	{
		crop_x_min = (size_t)((float)source_width * APP_AI_TRAINING_CROP_X_MIN_RATIO);
		crop_y_min = (size_t)((float)source_height * APP_AI_TRAINING_CROP_Y_MIN_RATIO);
		crop_width = (size_t)((float)source_width *
			(APP_AI_TRAINING_CROP_X_MAX_RATIO - APP_AI_TRAINING_CROP_X_MIN_RATIO));
		crop_height = (size_t)((float)source_height *
			(APP_AI_TRAINING_CROP_Y_MAX_RATIO - APP_AI_TRAINING_CROP_Y_MIN_RATIO));
	}

	if (crop_width == 0U)
	{
		crop_width = 1U;
	}
	if (crop_height == 0U)
	{
		crop_height = 1U;
	}
	if (crop_x_min >= source_width)
	{
		crop_x_min = source_width - 1U;
		crop_width = 1U;
	}
	if (crop_y_min >= source_height)
	{
		crop_y_min = source_height - 1U;
		crop_height = 1U;
	}
	if ((crop_x_min + crop_width) > source_width)
	{
		crop_width = source_width - crop_x_min;
	}
	if ((crop_y_min + crop_height) > source_height)
	{
		crop_height = source_height - crop_y_min;
	}

	DebugConsole_Printf("[AI] Crop %s: x=%lu y=%lu w=%lu h=%lu\r\n",
					   crop_label,
					   (unsigned long)crop_x_min,
					   (unsigned long)crop_y_min,
					   (unsigned long)crop_width,
					   (unsigned long)crop_height);
	(void)DebugConsole_WriteString("[AI] Preprocess diagnostics OK.\r\n");
	(void)DebugConsole_WriteString("[AI] Preprocess zero-fill skipped.\r\n");
	(void)DebugConsole_WriteString("[AI] Preprocess resize start.\r\n");

	{
		const float resize_scale =
			fminf((float)output_width / (float)crop_width,
				  (float)output_height / (float)crop_height);
		size_t resized_width = (size_t)(((float)crop_width * resize_scale) + 0.5f);
		size_t resized_height = (size_t)(((float)crop_height * resize_scale) + 0.5f);
		float *dst = input_ptr;

		if (resized_width == 0U)
		{
			resized_width = 1U;
		}
		if (resized_height == 0U)
		{
			resized_height = 1U;
		}
		if (resized_width > output_width)
		{
			resized_width = output_width;
		}
		if (resized_height > output_height)
		{
			resized_height = output_height;
		}

		const size_t resize_pad_x = (output_width - resized_width) / 2U;
		const size_t resize_pad_y = (output_height - resized_height) / 2U;

		(void)DebugConsole_WriteString("[AI] Preprocess row loop enter.\r\n");

		for (size_t out_y = 0U; out_y < output_height; ++out_y)
		{
			for (size_t out_x = 0U; out_x < output_width; ++out_x)
			{
				float out_r = 0.0f;
				float out_g = 0.0f;
				float out_b = 0.0f;

				if ((out_x >= resize_pad_x) &&
					(out_x < (resize_pad_x + resized_width)) &&
					(out_y >= resize_pad_y) &&
					(out_y < (resize_pad_y + resized_height)))
				{
					const float resized_x = (float)(out_x - resize_pad_x);
					const float resized_y = (float)(out_y - resize_pad_y);
					const float crop_x = ((resized_x + 0.5f) / resize_scale) - 0.5f;
					const float crop_y = ((resized_y + 0.5f) / resize_scale) - 0.5f;
					const float src_x = (float)crop_x_min + crop_x;
					const float src_y = (float)crop_y_min + crop_y;

					AppAI_ReadRgbFromYuv422Bilinear(
						frame_bytes, frame_size, source_width, source_height,
						src_x, src_y, &out_r, &out_g, &out_b);
				}

				*dst++ = AppAI_ClampNormalizedFloat(out_r);
				*dst++ = AppAI_ClampNormalizedFloat(out_g);
				*dst++ = AppAI_ClampNormalizedFloat(out_b);
			}
		}
	}

	(void)DebugConsole_WriteString("[AI] Preprocess resize OK.\r\n");
	/* Log the input buffer address, length, and live r9 right after the resize
	 * so we can tell whether the tip-focus model is being handed a valid
	 * tensor pointer and a sane reloc base. A null input_ptr or a zero r9
	 * here is the most common cause of a post-preprocess HardFault. */
	{
		uintptr_t post_resize_r9 = 0U;
		__asm volatile("mov %0, r9" : "=r"(post_resize_r9));
		DebugConsole_Printf(
			"[AI] Preprocess post: input_ptr=%p input_len=%lu floats=%lu r9=%p\r\n",
			(const void *)input_ptr,
			(unsigned long)input_len_bytes,
			(unsigned long)input_float_count,
			(const void *)post_resize_r9);
	}
	return true;
}

bool AppAI_BuildObbDecodeCandidate(
	float x_min, float y_min, float x_max, float y_max,
	AppAI_ObbDecodeCandidate *candidate)
{
	float center_x = 0.0f;
	float center_y = 0.0f;
	float box_w = 0.0f;
	float box_h = 0.0f;
	float cx_px = 0.0f;
	float cy_px = 0.0f;
	float bw_px = 0.0f;
	float bh_px = 0.0f;
	float crop_w = 0.0f;
	float crop_h = 0.0f;
	float crop_x_min = 0.0f;
	float crop_y_min = 0.0f;

	if (candidate == NULL)
	{
		return false;
	}

	(void)memset(candidate, 0, sizeof(*candidate));

	if (x_max < x_min)
	{
		const float tmp = x_min;
		x_min = x_max;
		x_max = tmp;
	}
	if (y_max < y_min)
	{
		const float tmp = y_min;
		y_min = y_max;
		y_max = tmp;
	}

	center_x = 0.5f * (x_min + x_max);
	center_y = 0.5f * (y_min + y_max);
	box_w = x_max - x_min;
	box_h = y_max - y_min;

	if (!AppAI_IsFiniteFloat(x_min) || !AppAI_IsFiniteFloat(y_min) ||
		!AppAI_IsFiniteFloat(x_max) || !AppAI_IsFiniteFloat(y_max) ||
		!AppAI_IsFiniteFloat(center_x) || !AppAI_IsFiniteFloat(center_y) ||
		!AppAI_IsFiniteFloat(box_w) || !AppAI_IsFiniteFloat(box_h) ||
		(box_w <= 0.0f) || (box_h <= 0.0f))
	{
		return false;
	}

	if (box_w < APP_AI_OBB_MIN_BOX_RATIO)
	{
		box_w = APP_AI_OBB_MIN_BOX_RATIO;
	}
	else if (box_w > 1.0f)
	{
		box_w = 1.0f;
	}
	if (box_h < APP_AI_OBB_MIN_BOX_RATIO)
	{
		box_h = APP_AI_OBB_MIN_BOX_RATIO;
	}
	else if (box_h > 1.0f)
	{
		box_h = 1.0f;
	}

	cx_px = center_x * (float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	cy_px = center_y * (float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	bw_px = box_w * (float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	bh_px = box_h * (float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	crop_w = bw_px * APP_AI_OBB_CROP_SCALE;
	crop_h = bh_px * APP_AI_OBB_CROP_SCALE;
	crop_x_min = cx_px - crop_w * 0.5f;
	crop_y_min = cy_px - crop_h * 0.5f;

	if (crop_x_min < 0.0f)
	{
		crop_x_min = 0.0f;
	}
	if (crop_y_min < 0.0f)
	{
		crop_y_min = 0.0f;
	}
	if ((crop_x_min + crop_w) > (float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS)
	{
		crop_w = (float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS - crop_x_min;
	}
	if ((crop_y_min + crop_h) > (float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS)
	{
		crop_h = (float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS - crop_y_min;
	}
	if ((crop_w < APP_AI_OBB_MIN_CROP_SIZE_PIXELS) ||
		(crop_h < APP_AI_OBB_MIN_CROP_SIZE_PIXELS))
	{
		return false;
	}

	candidate->valid = true;
	candidate->x_min = x_min;
	candidate->y_min = y_min;
	candidate->x_max = x_max;
	candidate->y_max = y_max;
	candidate->center_x = center_x;
	candidate->center_y = center_y;
	candidate->box_w = box_w;
	candidate->box_h = box_h;
	candidate->crop.x_min = (size_t)(crop_x_min + 0.5f);
	candidate->crop.y_min = (size_t)(crop_y_min + 0.5f);
	candidate->crop.width = (size_t)(crop_w + 0.5f);
	candidate->crop.height = (size_t)(crop_h + 0.5f);
	return true;
}

void AppAI_LogObbDecodeDiagnostics(
	float score,
	float raw0,
	float raw1,
	float raw2,
	float raw3,
	const AppAI_ObbDecodeCandidate *corners_candidate,
	const AppAI_ObbDecodeCandidate *center_size_candidate,
	const char *selected_label)
{
#if APP_AI_ENABLE_OBB_DECODE_DIAGNOSTICS
	const long score_100 = lroundf(score * 100.0f);
	const long raw0_100 = lroundf(raw0 * 100.0f);
	const long raw1_100 = lroundf(raw1 * 100.0f);
	const long raw2_100 = lroundf(raw2 * 100.0f);
	const long raw3_100 = lroundf(raw3 * 100.0f);

	DebugConsole_Printf(
		"[AI][OBB] score=%ld.%02ld raw=[%ld.%02ld %ld.%02ld %ld.%02ld %ld.%02ld]\r\n",
		labs(score_100) / 100L, labs(score_100) % 100L,
		labs(raw0_100) / 100L, labs(raw0_100) % 100L,
		labs(raw1_100) / 100L, labs(raw1_100) % 100L,
		labs(raw2_100) / 100L, labs(raw2_100) % 100L,
		labs(raw3_100) / 100L, labs(raw3_100) % 100L);

	if ((corners_candidate != NULL) && corners_candidate->valid)
	{
		const long cx_100 = lroundf(corners_candidate->center_x * 100.0f);
		const long cy_100 = lroundf(corners_candidate->center_y * 100.0f);
		const long bw_100 = lroundf(corners_candidate->box_w * 100.0f);
		const long bh_100 = lroundf(corners_candidate->box_h * 100.0f);
		DebugConsole_Printf(
			"[AI][OBB] corners center=(%ld.%02ld,%ld.%02ld) size=(%ld.%02ld,%ld.%02ld) crop=x%lu y%lu w%lu h%lu\r\n",
			labs(cx_100) / 100L, labs(cx_100) % 100L,
			labs(cy_100) / 100L, labs(cy_100) % 100L,
			labs(bw_100) / 100L, labs(bw_100) % 100L,
			labs(bh_100) / 100L, labs(bh_100) % 100L,
			(unsigned long)corners_candidate->crop.x_min,
			(unsigned long)corners_candidate->crop.y_min,
			(unsigned long)corners_candidate->crop.width,
			(unsigned long)corners_candidate->crop.height);
	}
	else
	{
		DebugConsole_WriteString("[AI][OBB] corners invalid\r\n");
	}

	if ((center_size_candidate != NULL) && center_size_candidate->valid)
	{
		const long cx_100 = lroundf(center_size_candidate->center_x * 100.0f);
		const long cy_100 = lroundf(center_size_candidate->center_y * 100.0f);
		const long bw_100 = lroundf(center_size_candidate->box_w * 100.0f);
		const long bh_100 = lroundf(center_size_candidate->box_h * 100.0f);
		DebugConsole_Printf(
			"[AI][OBB] center-size center=(%ld.%02ld,%ld.%02ld) size=(%ld.%02ld,%ld.%02ld) crop=x%lu y%lu w%lu h%lu\r\n",
			labs(cx_100) / 100L, labs(cx_100) % 100L,
			labs(cy_100) / 100L, labs(cy_100) % 100L,
			labs(bw_100) / 100L, labs(bw_100) % 100L,
			labs(bh_100) / 100L, labs(bh_100) % 100L,
			(unsigned long)center_size_candidate->crop.x_min,
			(unsigned long)center_size_candidate->crop.y_min,
			(unsigned long)center_size_candidate->crop.width,
			(unsigned long)center_size_candidate->crop.height);
	}
	else
	{
		DebugConsole_WriteString("[AI][OBB] center-size invalid\r\n");
	}

	DebugConsole_Printf("[AI][OBB] selected=%s\r\n",
		(selected_label != NULL) ? selected_label : "none");
#else
	(void)score;
	(void)raw0;
	(void)raw1;
	(void)raw2;
	(void)raw3;
	(void)corners_candidate;
	(void)center_size_candidate;
	(void)selected_label;
#endif
}

bool AppAI_DumpTipFocusInputTensorOnce(
	const float *input_ptr,
	size_t output_width,
	size_t output_height,
	const char *crop_label,
	const AppAI_SourceCrop *crop_ptr,
	bool obb_crop_valid,
	const AppAI_ObbBox *obb_box)
{
#if APP_AI_ENABLE_TIP_FOCUS_INPUT_DUMP
	FX_MEDIA *media = NULL;
	FX_FILE ppm_file = {0};
	FX_FILE meta_file = {0};
	UINT fx_status = FX_SUCCESS;
	UINT open_status = FX_SUCCESS;
	char ppm_path[48] = {0};
	char meta_path[48] = {0};
	char ppm_header[32] = {0};
	char meta_text[256] = {0};
	float min_value = 0.0f;
	float max_value = 0.0f;
	bool unit_scale = true;
	uint8_t row_bytes[3U * APP_AI_TIP_FOCUS_MODEL_INPUT_WIDTH_PIXELS] = {0U};

	if (app_ai_tip_focus_input_dump_done)
	{
		return true;
	}
	if ((input_ptr == NULL) || (output_width == 0U) || (output_height == 0U) ||
		(output_width > APP_AI_TIP_FOCUS_MODEL_INPUT_WIDTH_PIXELS))
	{
		return false;
	}
	if (!AppFileX_IsMediaReady())
	{
		DebugConsole_WriteString(
			"[AI][TIP_FOCUS] SD not ready; skipping one-shot input dump.\r\n");
		return false;
	}

	min_value = input_ptr[0];
	max_value = input_ptr[0];
	for (size_t i = 1U; i < (output_width * output_height * 3U); ++i)
	{
		const float value = input_ptr[i];
		if (value < min_value)
		{
			min_value = value;
		}
		if (value > max_value)
		{
			max_value = value;
		}
	}
	unit_scale = (min_value >= -0.25f) && (max_value <= 1.25f);

	media = AppFileX_GetMediaHandle();
	if (media == NULL)
	{
		return false;
	}

	fx_status = AppFileX_AcquireMediaLock();
	if (fx_status != TX_SUCCESS)
	{
		DebugConsole_Printf(
			"[AI][TIP_FOCUS] Failed to acquire FileX lock for input dump, status=%lu.\r\n",
			(unsigned long)fx_status);
		return false;
	}

	(void)fx_directory_create(media, INFERENCE_LOG_DIRECTORY_NAME);
	(void)DebugConsole_Snprintf(ppm_path, sizeof(ppm_path),
		"%s/tip_focus_input.ppm", INFERENCE_LOG_DIRECTORY_NAME);
	(void)DebugConsole_Snprintf(meta_path, sizeof(meta_path),
		"%s/tip_focus_input.txt", INFERENCE_LOG_DIRECTORY_NAME);
	(void)fx_file_delete(media, ppm_path);
	(void)fx_file_delete(media, meta_path);
	(void)fx_file_create(media, ppm_path);
	(void)fx_file_create(media, meta_path);

	open_status = fx_file_open(media, &ppm_file, ppm_path, FX_OPEN_FOR_WRITE);
	if (open_status != FX_SUCCESS)
	{
		AppFileX_ReleaseMediaLock();
		DebugConsole_Printf(
			"[AI][TIP_FOCUS] Failed to open %s, status=%lu.\r\n",
			ppm_path, (unsigned long)open_status);
		return false;
	}

	(void)DebugConsole_Snprintf(ppm_header, sizeof(ppm_header),
		"P6\n%lu %lu\n255\n",
		(unsigned long)output_width,
		(unsigned long)output_height);
	(void)fx_file_write(&ppm_file, (VOID *)ppm_header,
		(ULONG)strlen(ppm_header));

	for (size_t y = 0U; y < output_height; ++y)
	{
		for (size_t x = 0U; x < output_width; ++x)
		{
			const size_t pixel_index = ((y * output_width) + x) * 3U;
			for (size_t channel = 0U; channel < 3U; ++channel)
			{
				float value = input_ptr[pixel_index + channel];
				if (unit_scale)
				{
					value *= 255.0f;
				}
				if (value < 0.0f)
				{
					value = 0.0f;
				}
				else if (value > 255.0f)
				{
					value = 255.0f;
				}
				row_bytes[(x * 3U) + channel] = (uint8_t)(value + 0.5f);
			}
		}
		(void)fx_file_write(&ppm_file, row_bytes, (ULONG)(output_width * 3U));
	}
	(void)fx_file_close(&ppm_file);

	open_status = fx_file_open(media, &meta_file, meta_path, FX_OPEN_FOR_WRITE);
	if (open_status == FX_SUCCESS)
	{
		const long conf_100 =
			(obb_box != NULL) ? lroundf(obb_box->confidence * 100.0f) : 0L;
		(void)DebugConsole_Snprintf(
			meta_text, sizeof(meta_text),
			"crop_label=%s\r\n"
			"obb_crop_valid=%lu\r\n"
			"crop_x=%lu\r\n"
			"crop_y=%lu\r\n"
			"crop_w=%lu\r\n"
			"crop_h=%lu\r\n"
			"obb_confidence=%ld.%02ld\r\n"
			"input_min_milli=%ld\r\n"
			"input_max_milli=%ld\r\n"
			"unit_scale=%lu\r\n",
			(crop_label != NULL) ? crop_label : "unknown",
			(unsigned long)(obb_crop_valid ? 1U : 0U),
			(unsigned long)((crop_ptr != NULL) ? crop_ptr->x_min : 0U),
			(unsigned long)((crop_ptr != NULL) ? crop_ptr->y_min : 0U),
			(unsigned long)((crop_ptr != NULL) ? crop_ptr->width : 0U),
			(unsigned long)((crop_ptr != NULL) ? crop_ptr->height : 0U),
			labs(conf_100) / 100L,
			labs(conf_100) % 100L,
			(long)lroundf(min_value * 1000.0f),
			(long)lroundf(max_value * 1000.0f),
			(unsigned long)(unit_scale ? 1U : 0U));
		(void)fx_file_write(&meta_file, (VOID *)meta_text,
			(ULONG)strlen(meta_text));
		(void)fx_file_close(&meta_file);
	}

	(void)fx_media_flush(media);
	AppFileX_ReleaseMediaLock();
	app_ai_tip_focus_input_dump_done = true;
	DebugConsole_Printf(
		"[AI][TIP_FOCUS] dumped exact input to %s and %s (crop=%s x=%lu y=%lu w=%lu h=%lu min_milli=%ld max_milli=%ld unit=%lu)\r\n",
		ppm_path,
		meta_path,
		(crop_label != NULL) ? crop_label : "unknown",
		(unsigned long)((crop_ptr != NULL) ? crop_ptr->x_min : 0U),
		(unsigned long)((crop_ptr != NULL) ? crop_ptr->y_min : 0U),
		(unsigned long)((crop_ptr != NULL) ? crop_ptr->width : 0U),
		(unsigned long)((crop_ptr != NULL) ? crop_ptr->height : 0U),
		(long)lroundf(min_value * 1000.0f),
		(long)lroundf(max_value * 1000.0f),
		(unsigned long)(unit_scale ? 1U : 0U));
	return true;
#else
	(void)input_ptr;
	(void)output_width;
	(void)output_height;
	(void)crop_label;
	(void)crop_ptr;
	(void)obb_crop_valid;
	(void)obb_box;
	return false;
#endif
}

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

	return 0U;
}
