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
#include <string.h>

#include "app_ai.h"
#include "app_baseline_runtime.h"
#include "app_center_detector.h"
#include "ll_aton_rt.h"
#include "ll_aton_rt_user_api.h"

/* Magnet/STM32CubeIDE-generated header must be in the include path.
 * It provides LL_ATON_EC_Network_Init_mobilenetv2_center_detector,
 * LL_ATON_EC_Inference_Init_mobilenetv2_center_detector, and the buffer info
 * accessors via the generated mobilenetv2_center_detector.h. */
#include "mobilenetv2_center_detector.h"

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
	__attribute__((section(".noncacheable")));

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
 * @brief Resample the OBB crop region from the YUV422 full frame into the
 *        int8 RGB 224x224 input buffer (bilinear-ish).
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
	center_det_initialized = true;
	return true;
}

bool AppCenterDetector_Run(const uint8_t *frame_bytes, size_t frame_size,
	size_t crop_x_min, size_t crop_y_min, size_t crop_width, size_t crop_height,
	size_t frame_width_pixels, size_t frame_height_pixels,
	AppCenterDetector_Result_t *result)
{
	/* -----------------------------------------------------------------------
	 *  Validate inputs
	 * ----------------------------------------------------------------------- */
	if ((frame_bytes == NULL) || (result == NULL))
	{
		return false;
	}
	if (!center_det_initialized)
	{
		return false;
	}
	if ((crop_width == 0U) || (crop_height == 0U))
	{
		return false;
	}

	memset(result, 0, sizeof(*result));

	/* -----------------------------------------------------------------------
	 *  Step 1: fill the int8 RGB input buffer from the OBB crop region
	 * ----------------------------------------------------------------------- */
	if (!AppCenterDetector_FillInputFromCrop(frame_bytes, frame_size,
			frame_width_pixels, frame_height_pixels,
			crop_x_min, crop_y_min, crop_width, crop_height))
	{
		return false;
	}

	/* -----------------------------------------------------------------------
	 *  Step 2: run NPU inference
	 * ----------------------------------------------------------------------- */
	NN_Instance_TypeDef *instance =
		&NN_Instance_mobilenetv2_center_detector;

	/* Point the NPU input buffer at our scratch memory. */
	const LL_Buffer_InfoTypeDef *input_info =
		instance->network->input_buffers_info();
	if (input_info == NULL)
	{
		return false;
	}
	int8_t **input_pp = (int8_t **)LL_Buffer_buffer_addr(input_info);
	if (input_pp == NULL)
	{
		return false;
	}
	*input_pp = center_det_input_buf;

	if (!LL_ATON_EC_Inference_Init_mobilenetv2_center_detector())
	{
		return false;
	}

	/* Run the NPU schedule (all epochs). */
	if (!LL_ATON_Run())
	{
		return false;
	}

	/* -----------------------------------------------------------------------
	 *  Step 3: dequantise the 2-byte int8 (cx, cy) output
	 * ----------------------------------------------------------------------- */
	const LL_Buffer_InfoTypeDef *output_info =
		instance->network->output_buffers_info();
	if (output_info == NULL)
	{
		return false;
	}
	const int8_t *output_ptr =
		*(const int8_t **)LL_Buffer_buffer_addr(output_info);
	if (output_ptr == NULL)
	{
		return false;
	}

	/* output_ptr[0] = cx_int8, output_ptr[1] = cy_int8 */
	const float cx_norm =
		((float)output_ptr[0] - (float)CENTER_DET_OUT_ZERO_POINT)
		* CENTER_DET_OUT_SCALE;
	const float cy_norm =
		((float)output_ptr[1] - (float)CENTER_DET_OUT_ZERO_POINT)
		* CENTER_DET_OUT_SCALE;

	/* Map to full-frame pixel coordinates. */
	const float ff_cx =
		(float)crop_x_min + cx_norm * (float)crop_width;
	const float ff_cy =
		(float)crop_y_min + cy_norm * (float)crop_height;

	result->center_x = ff_cx;
	result->center_y = ff_cy;

	/* -----------------------------------------------------------------------
	 *  Step 4: run polar-vote needle detection around the estimated centre
	 * ----------------------------------------------------------------------- */
	{
		AppBaselineRuntime_Estimate_t polar_est = {0};

		/* Dial radius ~half the crop size (gauge face ≈ OBB box). */
		const float dial_radius_px =
			(float)((crop_width < crop_height) ? crop_width : crop_height) * 0.5f;

		const bool polar_ok = AppBaselineRuntime_EstimatePolarNeedle(
			frame_bytes, frame_size,
			frame_width_pixels, frame_height_pixels,
			0U, 0U, frame_width_pixels, frame_height_pixels,
			(size_t)ff_cx, (size_t)ff_cy,
			dial_radius_px,
			"center_detector", &polar_est);

		if (polar_ok)
		{
			result->valid = true;
			result->needle_angle_rad = polar_est.angle_rad;
			result->temperature_c = AppBaselineRuntime_ConvertAngleToTemperature(
				polar_est.angle_rad);
			result->confidence = polar_est.confidence;
		}
		else
		{
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

	/* Each YUV422 pixel occupies 2 bytes: [Y0, U, Y1, V] for 2 luma samples.
	 * Frame stride = frame_width_pixels * 2. */
	const size_t stride = frame_width_pixels * 2U;

	for (size_t row = 0U; row < CENTER_DET_INPUT_HEIGHT; row++)
	{
		/* Map output row → source row (nearest-neighbour for simplicity;
		 * bilinear would be better but adds complexity). */
		const size_t src_row = crop_y + (row * crop_h) / CENTER_DET_INPUT_HEIGHT;
		if (src_row >= frame_height_pixels)
		{
			break;
		}

		for (size_t col = 0U; col < CENTER_DET_INPUT_WIDTH; col++)
		{
			const size_t src_col =
				crop_x + (col * crop_w) / CENTER_DET_INPUT_WIDTH;
			if (src_col >= frame_width_pixels)
			{
				break;
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
			center_det_input_buf[out_idx + 0] = (int8_t)((rgb >> 16) & 0xFF);
			center_det_input_buf[out_idx + 1] = (int8_t)((rgb >> 8) & 0xFF);
			center_det_input_buf[out_idx + 2] = (int8_t)(rgb & 0xFF);
		}
	}

	return true;
}
