/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_center_detector.c
 * @brief   Heatmap center-detection CNN + polar-vote pipeline.
 ******************************************************************************
 */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <math.h>
#include <string.h>

#include "debug_console.h"
#include "app_ai.h"
#include "app_baseline_runtime.h"
#include "app_memory_budget.h"
#include "app_center_detector.h"

/* The LL_ATON runtime headers require the target platform and OSAL macros
 * before they are included, so keep the local app wrapper explicit here. */
#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX
#ifndef LL_ATON_DBG_BUFFER_INFO_EXCLUDED
#define LL_ATON_DBG_BUFFER_INFO_EXCLUDED 1
#endif
#include "ll_aton_rt_user_api.h"

/* The input/output cache helpers live in app_ai.c, where the NPU bring-up
 * code already owns the cache maintenance policy for ATON buffers. */
extern int mcu_cache_clean_range(uint32_t start_addr, uint32_t end_addr);
extern int mcu_cache_invalidate_range(uint32_t start_addr, uint32_t end_addr);

/* Generated ST Edge AI package for the heatmap center detector. */
#include "../../st_ai_output/packages/heatmap_cd_tiny/st_ai_output/heatmap_cd.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* Center detector model input dimensions. */
#define CENTER_DET_INPUT_WIDTH  CAMERA_CAPTURE_WIDTH_PIXELS
#define CENTER_DET_INPUT_HEIGHT CAMERA_CAPTURE_HEIGHT_PIXELS
#define CENTER_DET_INPUT_CHANS  3U

/* Heatmap output dimensions from the deployed DS-CNN v4 package. */
#define CENTER_DET_HEATMAP_WIDTH  160U
#define CENTER_DET_HEATMAP_HEIGHT 160U

/* Quantization for the heatmap output tensor. */
#define CENTER_DET_HEATMAP_OUTPUT_SCALE 0.00390625f

/* Reject flat or clearly broken heatmaps before we turn them into a pivot. */
#define CENTER_DET_MIN_PEAK_VALUE 0.05f

/* Reject CNN outputs that land far away from the trusted fallback centre.
 * The board-facing failure mode we care about is a collapsed output or a
 * center estimate that is clearly nowhere near the OBB/polar fallback. */
#define CENTER_DET_FALLBACK_MAX_DELTA_PX 48.0f

/* The tiny heatmap model is staged in xSPI2, then copied into AXISRAM2
 * before the network is initialised. */
#define CENTER_DET_MODEL_FLASH_BASE_ADDR 0x70200000UL
#define CENTER_DET_MODEL_RAM_BASE_ADDR   0x34100000UL
#define CENTER_DET_MODEL_IMAGE_BYTES     69457U
#define CENTER_DET_MODEL_SIGNATURE_BYTES 16U

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/** Scratch buffer for the 320x320 RGB crop fed to the NPU. */
static uint8_t center_det_input_buf[CENTER_DET_INPUT_WIDTH * CENTER_DET_INPUT_HEIGHT * CENTER_DET_INPUT_CHANS]
	__attribute__((section(".tip_focus_activations"), aligned(32)));

/** Tiny-model blob signatures used to validate the staged xSPI2 image. */
static const uint8_t center_det_model_signature_start[CENTER_DET_MODEL_SIGNATURE_BYTES] = {
	0xB5U, 0x3FU, 0x43U, 0x0AU, 0x50U, 0xD3U, 0x0AU, 0x08U,
	0xBCU, 0x1BU, 0xB6U, 0xB1U, 0xD6U, 0xDDU, 0x0FU, 0x0CU,
};
static const uint8_t center_det_model_signature_tail[CENTER_DET_MODEL_SIGNATURE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x90U,
};

/** NPU instance and interface declared here via the Cube.AI macro. */
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(heatmap_cd);

static bool center_det_initialized = false;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/**
 * @brief Convert a YUV422 pixel (as stored in the full frame) to raw RGB.
 */
static void AppCenterDetector_YuvToRgb_uint8(
	uint8_t y, uint8_t u, uint8_t v,
	uint8_t *r_out, uint8_t *g_out, uint8_t *b_out);

/**
 * @brief Compute the resize-with-pad geometry used by the training pipeline.
 */
static void AppCenterDetector_ComputeResizeWithPadGeometry(
	size_t crop_w, size_t crop_h,
	float *scale_out, float *pad_x_out, float *pad_y_out);

/**
 * @brief Resample the selected crop from the YUV422 full frame into the
 *        uint8 RGB 320x320 input buffer using resize-with-pad geometry.
 */
static bool AppCenterDetector_FillInputFromCrop(
	const uint8_t *frame_bytes, size_t frame_size,
	size_t frame_width_pixels, size_t frame_height_pixels,
	size_t crop_x, size_t crop_y, size_t crop_w, size_t crop_h);

/**
 * @brief Decode the heatmap output into a sub-pixel center estimate.
 */
static bool AppCenterDetector_DecodeHeatmapOutput(
	const uint8_t *output_ptr, size_t output_len_bytes,
	float *center_row_out, float *center_col_out, float *peak_value_out);

/**
 * @brief One-dimensional parabolic refinement around the argmax peak.
 */
static float AppCenterDetector_ParabolicRefineAxis(
	float left, float center, float right);

/**
 * @brief Decide whether the CNN center output should be rejected.
 */
static bool AppCenterDetector_ShouldFallback(
	float peak_value,
	float ff_cx, float ff_cy,
	size_t frame_width_pixels, size_t frame_height_pixels,
	bool has_fallback_center,
	float fallback_center_x, float fallback_center_y,
	const char **reason_out);

/**
 * @brief Copy the staged tiny-model blob from xSPI2 into AXISRAM2.
 */
static bool AppCenterDetector_LoadModelImage(void);

/* USER CODE END PFP */

/* ---------------------------------------------------------------------------
 *  Public API
 * --------------------------------------------------------------------------- */

bool AppCenterDetector_Init(void)
{
	if (center_det_initialized)
	{
		return true;
	}
	if (!AppAI_Xspi2EnsureMemoryMappedMode())
	{
		return false;
	}
	if (!AppCenterDetector_LoadModelImage())
	{
		return false;
	}
	if (!LL_ATON_EC_Network_Init_heatmap_cd())
	{
		return false;
	}
	/* The generated EC init is a stub; the runtime instance still needs the
	 * generic LL_ATON network initialization before we can run epochs. */
	LL_ATON_RT_Init_Network(&NN_Instance_heatmap_cd);
	center_det_initialized = true;
	return true;
}

bool AppCenterDetector_Run(const uint8_t *frame_bytes, size_t frame_size,
	size_t crop_x_min, size_t crop_y_min, size_t crop_width, size_t crop_height,
	float dial_radius_override_px,
	size_t frame_width_pixels, size_t frame_height_pixels,
	AppCenterDetector_Result_t *result,
	float override_center_x, float override_center_y,
	float fallback_center_x, float fallback_center_y)
{
	/* -----------------------------------------------------------------------
	 *  Validate inputs
	 * ----------------------------------------------------------------------- */
	if ((frame_bytes == NULL) || (result == NULL))
	{
		return false;
	}

	memset(result, 0, sizeof(*result));

	/* -----------------------------------------------------------------------
	 *  Decide whether to use the caller-provided override centre or to run
	 *  the heatmap detector.
	 * ----------------------------------------------------------------------- */
	const bool use_fallback = (override_center_x >= 0.0f)
		&& (override_center_y >= 0.0f);
	const bool has_trusted_fallback = (fallback_center_x >= 0.0f)
		&& (fallback_center_y >= 0.0f);

	float ff_cx = 0.0f;
	float ff_cy = 0.0f;

	if (use_fallback)
	{
		ff_cx = override_center_x;
		ff_cy = override_center_y;

		result->center_x = ff_cx;
		result->center_y = ff_cy;

		DebugConsole_Printf(
			"[AI] Center detector using OBB fallback center: "
			"(%.1f,%.1f)\r\n", ff_cx, ff_cy);
	}
	else
	{
		const char *cnn_failure_reason = NULL;

		if (!center_det_initialized)
		{
			cnn_failure_reason = "model not initialised";
			goto center_detector_use_fallback;
		}
		if ((crop_width == 0U) || (crop_height == 0U))
		{
			cnn_failure_reason = "empty crop";
			goto center_detector_use_fallback;
		}

		/* -------------------------------------------------------------------
		 *  Step 1: fill the uint8 RGB input buffer from the OBB crop region
		 * ------------------------------------------------------------------- */
		if (!AppCenterDetector_FillInputFromCrop(frame_bytes, frame_size,
				frame_width_pixels, frame_height_pixels,
				crop_x_min, crop_y_min, crop_width, crop_height))
		{
			cnn_failure_reason = "preprocess failed";
			goto center_detector_use_fallback;
		}

		/* -------------------------------------------------------------------
		 *  Step 2: run NPU inference
		 * ------------------------------------------------------------------- */
		NN_Instance_TypeDef *instance = &NN_Instance_heatmap_cd;

		/* Re-initialise the network before we touch the runtime buffers. The
		 * generated stage runner expects the network to be in the ready state
		 * before the input buffer is filled. */
		LL_ATON_RT_Init_Network(instance);
		if (!LL_ATON_EC_Inference_Init_heatmap_cd())
		{
			cnn_failure_reason = "inference init failed";
			goto center_detector_use_fallback;
		}

		const LL_Buffer_InfoTypeDef *input_info =
			instance->network->input_buffers_info();
		size_t input_len_bytes = 0U;
		uint8_t *input_ptr = NULL;
		if (input_info == NULL)
		{
			cnn_failure_reason = "input buffer unavailable";
			goto center_detector_use_fallback;
		}
		input_ptr = (uint8_t *)LL_Buffer_addr_start(input_info);
		input_len_bytes = (size_t)LL_Buffer_len(input_info);
		if ((input_ptr == NULL) || (input_len_bytes < sizeof(center_det_input_buf)))
		{
			cnn_failure_reason = "input buffer invalid";
			goto center_detector_use_fallback;
		}
		memcpy(input_ptr, center_det_input_buf, sizeof(center_det_input_buf));
		(void)mcu_cache_clean_range((uint32_t)(uintptr_t)input_ptr,
			(uint32_t)((uintptr_t)input_ptr + input_len_bytes));

		for (;;)
		{
			const LL_ATON_RT_RetValues_t run_status =
				LL_ATON_RT_RunEpochBlock(instance);

			if (run_status == LL_ATON_RT_DONE)
			{
				break;
			}
			if (run_status == LL_ATON_RT_WFE)
			{
				LL_ATON_OSAL_WFE();
				continue;
			}
			if (run_status != LL_ATON_RT_NO_WFE)
			{
				cnn_failure_reason = "epoch run failed";
				goto center_detector_use_fallback;
			}
		}

		const LL_Buffer_InfoTypeDef *output_info =
			instance->network->output_buffers_info();
		size_t output_len_bytes = 0U;
		if (output_info == NULL)
		{
			cnn_failure_reason = "output buffer unavailable";
			goto center_detector_use_fallback;
		}
		output_len_bytes = (size_t)LL_Buffer_len(output_info);
		(void)mcu_cache_invalidate_range(
			(uint32_t)(uintptr_t)LL_Buffer_addr_start(output_info),
			(uint32_t)((uintptr_t)LL_Buffer_addr_start(output_info) + output_len_bytes));

		/* ---------------------------------------------------------------
		 *  Step 3: decode the heatmap output
		 * --------------------------------------------------------------- */
		const uint8_t *output_ptr =
			(const uint8_t *)LL_Buffer_addr_start(output_info);
		if (output_ptr == NULL)
		{
			cnn_failure_reason = "output buffer invalid";
			goto center_detector_use_fallback;
		}

		float heatmap_row = 0.0f;
		float heatmap_col = 0.0f;
		float peak_value = 0.0f;
		if (!AppCenterDetector_DecodeHeatmapOutput(output_ptr, output_len_bytes,
				&heatmap_row, &heatmap_col, &peak_value))
		{
			cnn_failure_reason = "heatmap decode failed";
			goto center_detector_use_fallback;
		}

		float resize_scale = 0.0f;
		float pad_x = 0.0f;
		float pad_y = 0.0f;
		AppCenterDetector_ComputeResizeWithPadGeometry(
			crop_width, crop_height, &resize_scale, &pad_x, &pad_y);

		/* Convert the 160x160 heatmap location back into the 320x320 crop
		 * space, then invert the resize-with-pad transform to recover the
		 * full-frame pixel centre. */
		const float heatmap_to_input_scale =
			((float)CENTER_DET_INPUT_WIDTH - 1.0f) /
			((float)CENTER_DET_HEATMAP_WIDTH - 1.0f);
		const float crop_cx = heatmap_col * heatmap_to_input_scale;
		const float crop_cy = heatmap_row * heatmap_to_input_scale;

		ff_cx = (float)crop_x_min + ((crop_cx - pad_x) / resize_scale);
		ff_cy = (float)crop_y_min + ((crop_cy - pad_y) / resize_scale);

		if (ff_cx < 0.0f)
		{
			ff_cx = 0.0f;
		}
		else if (ff_cx > (float)(frame_width_pixels - 1U))
		{
			ff_cx = (float)(frame_width_pixels - 1U);
		}
		if (ff_cy < 0.0f)
		{
			ff_cy = 0.0f;
		}
		else if (ff_cy > (float)(frame_height_pixels - 1U))
		{
			ff_cy = (float)(frame_height_pixels - 1U);
		}

		{
			const char *fallback_reason = NULL;
			const bool should_fallback = AppCenterDetector_ShouldFallback(
				peak_value, ff_cx, ff_cy,
				frame_width_pixels, frame_height_pixels,
				has_trusted_fallback,
				fallback_center_x, fallback_center_y,
				&fallback_reason);

			if (should_fallback)
			{
				if (has_trusted_fallback)
				{
					DebugConsole_Printf(
						"[CD] CNN output rejected (%s); using fallback centre: "
						"(%.1f,%.1f)\r\n",
						(fallback_reason != NULL) ? fallback_reason : "unknown",
						fallback_center_x, fallback_center_y);
					ff_cx = fallback_center_x;
					ff_cy = fallback_center_y;
				}
				else
				{
					cnn_failure_reason = (fallback_reason != NULL)
						? fallback_reason
						: "cnn output rejected";
					goto center_detector_use_fallback;
				}
			}
		}

		result->center_x = ff_cx;
		result->center_y = ff_cy;
		goto center_detector_center_ready;

center_detector_use_fallback:
		if (has_trusted_fallback)
		{
			DebugConsole_Printf(
				"[CD] CNN path unavailable (%s); using fallback centre: "
				"(%.1f,%.1f)\r\n",
				(cnn_failure_reason != NULL) ? cnn_failure_reason : "unknown",
				fallback_center_x, fallback_center_y);
			ff_cx = fallback_center_x;
			ff_cy = fallback_center_y;
			result->center_x = ff_cx;
			result->center_y = ff_cy;
		}
		else
		{
			return false;
		}

center_detector_center_ready:
		;
	}

	/* -----------------------------------------------------------------------
	 *  Step 4: run polar-vote needle detection around the estimated centre
	 * ----------------------------------------------------------------------- */
	{
		AppBaselineRuntime_Estimate_t polar_est = {0};

		/* Dial radius: prefer caller-provided override (from OBB box), else
		 * fall back to half the smaller crop dimension. */
		const float dial_radius_px = (dial_radius_override_px > 0.0f)
			? dial_radius_override_px
			: (float)((crop_width < crop_height) ? crop_width : crop_height) * 0.5f;
		size_t polar_center_x = (size_t)lroundf(ff_cx);
		size_t polar_center_y = (size_t)lroundf(ff_cy);

		if (polar_center_x >= frame_width_pixels)
		{
			polar_center_x = frame_width_pixels - 1U;
		}
		if (polar_center_y >= frame_height_pixels)
		{
			polar_center_y = frame_height_pixels - 1U;
		}

		{
			const long dr10 = lroundf(dial_radius_px * 10.0f);
			DebugConsole_Printf(
				"[CD] polar vote input: frame=%lux%lu center=(%lu,%lu) "
				"dial_radius=%ld.%01ld scan=(%lu,%lu)-(%lu,%lu)\r\n",
				(unsigned long)frame_width_pixels, (unsigned long)frame_height_pixels,
				(unsigned long)polar_center_x, (unsigned long)polar_center_y,
				labs(dr10) / 10L, labs(dr10) % 10L,
				(unsigned long)0U, (unsigned long)0U,
				(unsigned long)frame_width_pixels, (unsigned long)frame_height_pixels);
		}

		const bool polar_ok = AppBaselineRuntime_EstimatePolarNeedle(
			frame_bytes, frame_size,
			frame_width_pixels, frame_height_pixels,
			0U, 0U, frame_width_pixels, frame_height_pixels,
			polar_center_x, polar_center_y,
			dial_radius_px,
			"center_detector", &polar_est);

		if (polar_ok)
		{
			const long angle_deg = (long)lroundf(
				polar_est.angle_rad * 180.0f / 3.14159265358979f);
			const long t10 = lroundf(polar_est.temperature_c * 10.0f);
			const long c100 = lroundf(polar_est.confidence * 100.0f);
			const long s10 = lroundf(polar_est.best_score);
			const long r10 = lroundf(polar_est.runner_up_score);
			DebugConsole_Printf(
				"[CD] polar vote OK: angle=%lddeg "
				"temp=%s%ld.%01ldC conf=%s%ld.%02ld "
				"score=%ld ru=%ld\r\n",
				angle_deg,
				(t10 < 0L) ? "-" : "", labs(t10) / 10L, labs(t10) % 10L,
				(c100 < 0L) ? "-" : "", labs(c100) / 100L, labs(c100) % 100L,
				s10, r10);
			result->valid = true;
			result->needle_angle_rad = polar_est.angle_rad;
			result->temperature_c = AppBaselineRuntime_ConvertAngleToTemperature(
				polar_est.angle_rad);
			result->confidence = polar_est.confidence;
		}
		else
		{
			DebugConsole_WriteString(
				"[CD] polar vote FAILED (no clear needle)\r\n");
			result->valid = false;
		}
	}

	return true;
}

/* ---------------------------------------------------------------------------
 *  Private helpers
 * --------------------------------------------------------------------------- */

static void AppCenterDetector_YuvToRgb_uint8(
	uint8_t y, uint8_t u, uint8_t v,
	uint8_t *r_out, uint8_t *g_out, uint8_t *b_out)
{
	/* Standard BT.601 YUV -> RGB, then clamp to [0, 255]. */
	const int32_t u_off = (int32_t)u - 128;
	const int32_t v_off = (int32_t)v - 128;
	int32_t r = (int32_t)y + (v_off * 1436) / 1024;
	int32_t g = (int32_t)y - ((u_off * 352) + (v_off * 731)) / 1024;
	int32_t b = (int32_t)y + (u_off * 1814) / 1024;

	if (r < 0) { r = 0; }
	if (r > 255) { r = 255; }
	if (g < 0) { g = 0; }
	if (g > 255) { g = 255; }
	if (b < 0) { b = 0; }
	if (b > 255) { b = 255; }

	if (r_out != NULL)
	{
		*r_out = (uint8_t)r;
	}
	if (g_out != NULL)
	{
		*g_out = (uint8_t)g;
	}
	if (b_out != NULL)
	{
		*b_out = (uint8_t)b;
	}
}

static bool AppCenterDetector_LoadModelImage(void)
{
	const uint8_t *source_ptr =
		(const uint8_t *)CENTER_DET_MODEL_FLASH_BASE_ADDR;
	uint8_t *dest_ptr = (uint8_t *)CENTER_DET_MODEL_RAM_BASE_ADDR;
	const uint32_t source_start = (uint32_t)(uintptr_t)source_ptr;
	const uint32_t source_end = source_start + (uint32_t)CENTER_DET_MODEL_IMAGE_BYTES;
	const uint32_t dest_start = (uint32_t)(uintptr_t)dest_ptr;
	const uint32_t dest_end = dest_start + (uint32_t)CENTER_DET_MODEL_IMAGE_BYTES;

	if ((source_ptr == NULL) || (dest_ptr == NULL))
	{
		return false;
	}

	/* Keep the xSPI2 bytes fresh in case the blob was just reflashed. */
	(void)mcu_cache_invalidate_range(source_start, source_end);

	if ((memcmp(source_ptr, center_det_model_signature_start,
				sizeof(center_det_model_signature_start)) != 0) ||
		(memcmp(source_ptr + CENTER_DET_MODEL_IMAGE_BYTES -
				CENTER_DET_MODEL_SIGNATURE_BYTES,
				center_det_model_signature_tail,
				sizeof(center_det_model_signature_tail)) != 0))
	{
		DebugConsole_WriteString(
			"[CD] heatmap model image signature mismatch in xSPI2.\r\n");
		return false;
	}

	memcpy(dest_ptr, source_ptr, CENTER_DET_MODEL_IMAGE_BYTES);
	(void)mcu_cache_clean_range(dest_start, dest_end);

	DebugConsole_Printf(
		"[CD] heatmap model image copied to AXISRAM2 (%lu bytes).\r\n",
		(unsigned long)CENTER_DET_MODEL_IMAGE_BYTES);
	return true;
}

static bool AppCenterDetector_FillInputFromCrop(
	const uint8_t *frame_bytes, size_t frame_size,
	size_t frame_width_pixels, size_t frame_height_pixels,
	size_t crop_x, size_t crop_y, size_t crop_w, size_t crop_h)
{
	/* Clamp crop to frame dimensions. */
	if (crop_x + crop_w > frame_width_pixels)  { crop_w = frame_width_pixels - crop_x; }
	if (crop_y + crop_h > frame_height_pixels) { crop_h = frame_height_pixels - crop_y; }
	if ((crop_w == 0U) || (crop_h == 0U))
	{
		return false;
	}

	float resize_scale = 0.0f;
	float pad_x = 0.0f;
	float pad_y = 0.0f;
	const size_t stride = frame_width_pixels * 2U;

	AppCenterDetector_ComputeResizeWithPadGeometry(
		crop_w, crop_h, &resize_scale, &pad_x, &pad_y);

	/* Start with a zero-valued canvas in RGB space. */
	(void)memset(center_det_input_buf, 0x00, sizeof(center_det_input_buf));

	for (size_t row = 0U; row < CENTER_DET_INPUT_HEIGHT; row++)
	{
		const float out_row_f = (float)row;
		const float src_row_f =
			(out_row_f - pad_y) / resize_scale + (float)crop_y;
		const float row_lower_bound = pad_y;
		const float row_upper_bound = pad_y + (crop_h * resize_scale);
		size_t src_row;

		if ((out_row_f < row_lower_bound) || (out_row_f >= row_upper_bound))
		{
			continue;
		}

		src_row = (size_t)floorf(src_row_f + 0.5f);
		if (src_row >= frame_height_pixels)
		{
			src_row = frame_height_pixels - 1U;
		}

		for (size_t col = 0U; col < CENTER_DET_INPUT_WIDTH; col++)
		{
			const float out_col_f = (float)col;
			const float src_col_f =
				(out_col_f - pad_x) / resize_scale + (float)crop_x;
			const float col_lower_bound = pad_x;
			const float col_upper_bound = pad_x + (crop_w * resize_scale);
			size_t src_col;

			if ((out_col_f < col_lower_bound) || (out_col_f >= col_upper_bound))
			{
				continue;
			}

			src_col = (size_t)floorf(src_col_f + 0.5f);
			if (src_col >= frame_width_pixels)
			{
				src_col = frame_width_pixels - 1U;
			}

			/* Read YUV422 packed pair at the source pixel.
			 * YUV422 layout for pixel at (src_col, src_row):
			 *   offset = (src_row * stride) + ((src_col / 2) * 4)
			 *   U = frame[offset + 1]
			 *   Y even = frame[offset]
			 *   Y odd  = frame[offset + 2]
			 *   V = frame[offset + 3] */
			const size_t base =
				(src_row * stride) + ((src_col / 2U) * 4U);
			if ((base + 3U) >= frame_size)
			{
				return false;
			}

			const uint8_t y_val = (src_col & 1U)
				? frame_bytes[base + 2U]
				: frame_bytes[base];
			const uint8_t u_val = frame_bytes[base + 1U];
			const uint8_t v_val = frame_bytes[base + 3U];

			const size_t out_idx =
				(row * CENTER_DET_INPUT_WIDTH + col) * CENTER_DET_INPUT_CHANS;
			AppCenterDetector_YuvToRgb_uint8(
				y_val, u_val, v_val,
				&center_det_input_buf[out_idx + 0U],
				&center_det_input_buf[out_idx + 1U],
				&center_det_input_buf[out_idx + 2U]);
		}
	}

	return true;
}

static void AppCenterDetector_ComputeResizeWithPadGeometry(
	size_t crop_w, size_t crop_h,
	float *scale_out, float *pad_x_out, float *pad_y_out)
{
	const float crop_w_f = (float)crop_w;
	const float crop_h_f = (float)crop_h;
	const float output_w_f = (float)CENTER_DET_INPUT_WIDTH;
	const float output_h_f = (float)CENTER_DET_INPUT_HEIGHT;
	const float scale =
		(output_w_f / crop_w_f < output_h_f / crop_h_f)
			? (output_w_f / crop_w_f)
			: (output_h_f / crop_h_f);
	const float resized_w = crop_w_f * scale;
	const float resized_h = crop_h_f * scale;

	if (scale_out != NULL)
	{
		*scale_out = scale;
	}
	if (pad_x_out != NULL)
	{
		*pad_x_out = 0.5f * (output_w_f - resized_w);
	}
	if (pad_y_out != NULL)
	{
		*pad_y_out = 0.5f * (output_h_f - resized_h);
	}
}

static float AppCenterDetector_ParabolicRefineAxis(
	float left, float center, float right)
{
	const float denom = left - (2.0f * center) + right;
	float offset = 0.0f;

	if (fabsf(denom) < 1.0e-6f)
	{
		return 0.0f;
	}

	/* Fit a parabola through the three samples and return the sub-pixel peak
	 * shift relative to the center sample. */
	offset = 0.5f * (left - right) / denom;
	if (offset < -1.0f)
	{
		offset = -1.0f;
	}
	else if (offset > 1.0f)
	{
		offset = 1.0f;
	}
	return offset;
}

static bool AppCenterDetector_DecodeHeatmapOutput(
	const uint8_t *output_ptr, size_t output_len_bytes,
	float *center_row_out, float *center_col_out, float *peak_value_out)
{
	const size_t heatmap_width = CENTER_DET_HEATMAP_WIDTH;
	const size_t heatmap_height = CENTER_DET_HEATMAP_HEIGHT;
	const size_t heatmap_elems = heatmap_width * heatmap_height;
	size_t peak_index = 0U;
	uint8_t peak_q = 0U;
	float peak_value = 0.0f;
	float center_row = 0.0f;
	float center_col = 0.0f;

	if ((output_ptr == NULL) || (center_row_out == NULL) || (center_col_out == NULL))
	{
		return false;
	}
	if (output_len_bytes < heatmap_elems)
	{
		return false;
	}

	for (size_t index = 0U; index < heatmap_elems; ++index)
	{
		const uint8_t value = output_ptr[index];
		if (value > peak_q)
		{
			peak_q = value;
			peak_index = index;
		}
	}

	peak_value = ((float)peak_q) * CENTER_DET_HEATMAP_OUTPUT_SCALE;
	if (peak_value < CENTER_DET_MIN_PEAK_VALUE)
	{
		return false;
	}

	center_row = (float)(peak_index / heatmap_width);
	center_col = (float)(peak_index % heatmap_width);

	if ((peak_index % heatmap_width) > 0U && (peak_index % heatmap_width) < (heatmap_width - 1U))
	{
		const size_t row = peak_index / heatmap_width;
		const size_t col = peak_index % heatmap_width;
		const float left = (float)output_ptr[(row * heatmap_width) + (col - 1U)];
		const float center = (float)output_ptr[(row * heatmap_width) + col];
		const float right = (float)output_ptr[(row * heatmap_width) + (col + 1U)];
		center_col += AppCenterDetector_ParabolicRefineAxis(left, center, right);
	}

	if ((peak_index / heatmap_width) > 0U && (peak_index / heatmap_width) < (heatmap_height - 1U))
	{
		const size_t row = peak_index / heatmap_width;
		const size_t col = peak_index % heatmap_width;
		const float up = (float)output_ptr[((row - 1U) * heatmap_width) + col];
		const float center = (float)output_ptr[(row * heatmap_width) + col];
		const float down = (float)output_ptr[((row + 1U) * heatmap_width) + col];
		center_row += AppCenterDetector_ParabolicRefineAxis(up, center, down);
	}

	if (center_row < 0.0f)
	{
		center_row = 0.0f;
	}
	else if (center_row > (float)(heatmap_height - 1U))
	{
		center_row = (float)(heatmap_height - 1U);
	}
	if (center_col < 0.0f)
	{
		center_col = 0.0f;
	}
	else if (center_col > (float)(heatmap_width - 1U))
	{
		center_col = (float)(heatmap_width - 1U);
	}

	if (peak_value_out != NULL)
	{
		*peak_value_out = peak_value;
	}
	*center_row_out = center_row;
	*center_col_out = center_col;
	return true;
}

static bool AppCenterDetector_ShouldFallback(
	float peak_value,
	float ff_cx, float ff_cy,
	size_t frame_width_pixels, size_t frame_height_pixels,
	bool has_fallback_center,
	float fallback_center_x, float fallback_center_y,
	const char **reason_out)
{
	const float frame_w_f = (float)frame_width_pixels;
	const float frame_h_f = (float)frame_height_pixels;
	const float delta_x = has_fallback_center ? (ff_cx - fallback_center_x) : 0.0f;
	const float delta_y = has_fallback_center ? (ff_cy - fallback_center_y) : 0.0f;

	if (reason_out != NULL)
	{
		*reason_out = NULL;
	}

	if (!isfinite(peak_value) || (peak_value < CENTER_DET_MIN_PEAK_VALUE))
	{
		if (reason_out != NULL)
		{
			*reason_out = "heatmap peak too small";
		}
		return true;
	}

	if ((ff_cx < 0.0f) || (ff_cx > (frame_w_f - 1.0f)) ||
		(ff_cy < 0.0f) || (ff_cy > (frame_h_f - 1.0f)))
	{
		if (reason_out != NULL)
		{
			*reason_out = "center out of frame";
		}
		return true;
	}

	if (has_fallback_center)
	{
		const float delta = sqrtf((delta_x * delta_x) + (delta_y * delta_y));
		if (delta > CENTER_DET_FALLBACK_MAX_DELTA_PX)
		{
			if (reason_out != NULL)
			{
				*reason_out = "center far from fallback";
			}
			return true;
		}
	}

	return false;
}
