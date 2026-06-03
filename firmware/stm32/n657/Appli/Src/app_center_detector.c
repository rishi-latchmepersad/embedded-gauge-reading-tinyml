/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_center_detector.c
 * @brief   Center-detection CNN + polar-vote pipeline for gauge reading.
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

/* Magnet/STM32CubeIDE-generated header must be in the include path.
 * It provides LL_ATON_EC_Network_Init_mobilenetv2_center_detector,
 * LL_ATON_EC_Inference_Init_mobilenetv2_center_detector, and the buffer info
 * accessors via the generated mobilenetv2_center_detector.h. */
#include "../../st_ai_output/packages/center_detector_v4_int8/st_ai_output/mobilenetv2_center_detector.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* Center detector model input dimensions. */
#define CENTER_DET_INPUT_WIDTH  224U
#define CENTER_DET_INPUT_HEIGHT 224U
#define CENTER_DET_INPUT_CHANS  3U

/* Output quantisation (from c_info.json: int8, scale=0.00390625, zp=-128). */
#define CENTER_DET_OUT_SCALE  0.00390625f
#define CENTER_DET_OUT_ZERO_POINT (-128)

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/** Scratch buffer for the 224x224 int8 RGB crop fed to the NPU. */
static int8_t center_det_input_buf[CENTER_DET_INPUT_WIDTH * CENTER_DET_INPUT_HEIGHT * CENTER_DET_INPUT_CHANS]
	__attribute__((aligned(32)));

/** NPU instance and interface declared here via the Cube.AI macro. */
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(
	mobilenetv2_center_detector);

static bool center_det_initialized = false;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/**
 * @brief Convert a YUV422 pixel (as stored in the full frame) to int8 RGB.
 */
static int32_t AppCenterDetector_YuvToRgb_int8(uint8_t y, uint8_t u, uint8_t v);

/**
 * @brief Compute the resize-with-pad geometry used by the training pipeline.
 */
static void AppCenterDetector_ComputeResizeWithPadGeometry(
	size_t crop_w, size_t crop_h,
	float *scale_out, float *pad_x_out, float *pad_y_out);

/**
 * @brief Resample the selected crop from the YUV422 full frame into the
 *        int8 RGB 224x224 input buffer using resize-with-pad geometry.
 */
static bool AppCenterDetector_FillInputFromCrop(
	const uint8_t *frame_bytes, size_t frame_size,
	size_t frame_width_pixels, size_t frame_height_pixels,
	size_t crop_x, size_t crop_y, size_t crop_w, size_t crop_h);

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
	if (!LL_ATON_EC_Network_Init_mobilenetv2_center_detector())
	{
		return false;
	}
	/* The generated EC init is a stub; the runtime instance still needs the
	 * generic LL_ATON network initialization before we can run epochs. */
	LL_ATON_RT_Init_Network(&NN_Instance_mobilenetv2_center_detector);
	center_det_initialized = true;
	return true;
}

bool AppCenterDetector_Run(const uint8_t *frame_bytes, size_t frame_size,
	size_t crop_x_min, size_t crop_y_min, size_t crop_width, size_t crop_height,
	float dial_radius_override_px,
	size_t frame_width_pixels, size_t frame_height_pixels,
	AppCenterDetector_Result_t *result,
	float override_center_x, float override_center_y)
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
	 *  Decide: use the caller-provided override centre (skip CNN) or run the
	 *  centre-detector CNN to estimate it.
	 * ----------------------------------------------------------------------- */
	const bool use_fallback = (override_center_x >= 0.0f)
		&& (override_center_y >= 0.0f);

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
		if (!center_det_initialized)
		{
			return false;
		}
		if ((crop_width == 0U) || (crop_height == 0U))
		{
			return false;
		}

		/* -------------------------------------------------------------------
		 *  Step 1: fill the int8 RGB input buffer from the OBB crop region
		 * ------------------------------------------------------------------- */
		if (!AppCenterDetector_FillInputFromCrop(frame_bytes, frame_size,
				frame_width_pixels, frame_height_pixels,
				crop_x_min, crop_y_min, crop_width, crop_height))
		{
			return false;
		}

		/* -------------------------------------------------------------------
		 *  Step 2: run NPU inference
		 * ------------------------------------------------------------------- */
		NN_Instance_TypeDef *instance =
			&NN_Instance_mobilenetv2_center_detector;

		const LL_Buffer_InfoTypeDef *input_info =
			instance->network->input_buffers_info();
		size_t input_len_bytes = 0U;
		int8_t *input_ptr = NULL;
		if (input_info == NULL)
		{
			return false;
		}
		input_ptr = (int8_t *)LL_Buffer_addr_start(input_info);
		input_len_bytes = (size_t)LL_Buffer_len(input_info);
		if ((input_ptr == NULL) || (input_len_bytes < sizeof(center_det_input_buf)))
		{
			return false;
		}
		memcpy(input_ptr, center_det_input_buf, sizeof(center_det_input_buf));
		(void)mcu_cache_clean_range((uint32_t)(uintptr_t)input_ptr,
			(uint32_t)((uintptr_t)input_ptr + input_len_bytes));

		if (!LL_ATON_EC_Inference_Init_mobilenetv2_center_detector())
		{
			return false;
		}

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
				return false;
			}
		}

		const LL_Buffer_InfoTypeDef *output_info =
			instance->network->output_buffers_info();
		size_t output_len_bytes = 0U;
		if (output_info == NULL)
		{
			return false;
		}
		output_len_bytes = (size_t)LL_Buffer_len(output_info);
		(void)mcu_cache_invalidate_range(
			(uint32_t)(uintptr_t)LL_Buffer_addr_start(output_info),
			(uint32_t)((uintptr_t)LL_Buffer_addr_start(output_info) + output_len_bytes));

		/* ---------------------------------------------------------------
		 *  Step 3: dequantise the 2-byte int8 (cx, cy) output
		 * --------------------------------------------------------------- */
		const int8_t *output_ptr =
			(const int8_t *)LL_Buffer_addr_start(output_info);
		if (output_ptr == NULL)
		{
			return false;
		}
		DebugConsole_Printf("[CD] raw int8 output: %d %d\r\n",
			output_ptr[0], output_ptr[1]);

		const float cx_norm =
			((float)output_ptr[0] - (float)CENTER_DET_OUT_ZERO_POINT)
			* CENTER_DET_OUT_SCALE;
		const float cy_norm =
			((float)output_ptr[1] - (float)CENTER_DET_OUT_ZERO_POINT)
			* CENTER_DET_OUT_SCALE;

		float resize_scale = 0.0f;
		float pad_x = 0.0f;
		float pad_y = 0.0f;
		AppCenterDetector_ComputeResizeWithPadGeometry(
			crop_width, crop_height, &resize_scale, &pad_x, &pad_y);

		const float padded_cx = cx_norm * (float)CENTER_DET_INPUT_WIDTH;
		const float padded_cy = cy_norm * (float)CENTER_DET_INPUT_HEIGHT;
		ff_cx = (float)crop_x_min + ((padded_cx - pad_x) / resize_scale);
		ff_cy = (float)crop_y_min + ((padded_cy - pad_y) / resize_scale);

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

		result->center_x = ff_cx;
		result->center_y = ff_cy;
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

static int32_t AppCenterDetector_YuvToRgb_int8(uint8_t y, uint8_t u, uint8_t v)
{
	/* Standard BT.601 YUV → RGB, then clamp to [0, 255], finally offset to
	 * int8 range by subtracting 128 (the NPU expects zero_point = -128). */
	int32_t r = (int32_t)y + ((int32_t)(v - 128) * 1436) / 1024;
	int32_t g = (int32_t)y - ((int32_t)(u - 128) * 352 + (int32_t)(v - 128) * 731) / 1024;
	int32_t b = (int32_t)y + ((int32_t)(u - 128) * 1814) / 1024;

	if (r < 0) { r = 0; } if (r > 255) { r = 255; }
	if (g < 0) { g = 0; } if (g > 255) { g = 255; }
	if (b < 0) { b = 0; } if (b > 255) { b = 255; }

	/* Shift to int8 [−128, 127]. */
	r -= 128;
	g -= 128;
	b -= 128;

	/* Pack into single int32 (not used as such — we write per-channel below). */
	return (r << 16) | (g << 8) | b;
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

	/* Start with a zero-valued canvas in quantized space. */
	(void)memset(center_det_input_buf, 0x80, sizeof(center_det_input_buf));

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

			/* Convert to int8 RGB. */
			const int32_t rgb = AppCenterDetector_YuvToRgb_int8(y_val, u_val, v_val);

			const size_t out_idx =
				(row * CENTER_DET_INPUT_WIDTH + col) * CENTER_DET_INPUT_CHANS;
			center_det_input_buf[out_idx + 0U] = (int8_t)((rgb >> 16) & 0xFF);
			center_det_input_buf[out_idx + 1U] = (int8_t)((rgb >> 8) & 0xFF);
			center_det_input_buf[out_idx + 2U] = (int8_t)(rgb & 0xFF);
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
