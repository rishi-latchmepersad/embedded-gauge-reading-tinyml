/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_baseline_template.c
 * @brief   Classical normalized-template matcher for board gauge captures.
 ******************************************************************************
 */
/* USER CODE END Header */

#include "app_baseline_template.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "app_gauge_geometry.h"

#include "app_baseline_template_bank.inc"

#define APP_BASELINE_TEMPLATE_GRID_SIZE 8U
#define APP_BASELINE_TEMPLATE_FEATURE_COUNT 64U
#define APP_BASELINE_TEMPLATE_QUANT_SCALE 16.0f
#define APP_BASELINE_TEMPLATE_ANGLE_OFFSET_DEG 0.0f
#define APP_BASELINE_TEMPLATE_MIN_ANGLE_DEG 135.0f
#define APP_BASELINE_TEMPLATE_SWEEP_DEG 270.0f
/* Template angles remain in the detector's east-zero frame. Adding 90° to
 * the north-zero TOML endpoints gives the raw labels without the old board
 * offset/gain calibration. */
#define APP_BASELINE_TEMPLATE_HOT_FRACTION (((APP_GAUGE_CALIBRATION_MAX_DEG + 90.0f) - 135.0f) / 270.0f)
#define APP_BASELINE_TEMPLATE_COLD_FRACTION ((((APP_GAUGE_CALIBRATION_MIN_DEG + 90.0f) - 135.0f) + 360.0f) / 270.0f)
#define APP_BASELINE_TEMPLATE_SOURCE_LABEL "classical-template-8x8"
#define APP_BASELINE_TEMPLATE_SCORE_SCALE 10000.0f

/**
 * @brief Read one luma sample from packed YUV422.
 * @param frame_bytes Packed YUV422 frame.
 * @param frame_width_pixels Frame width.
 * @param x Pixel x coordinate.
 * @param y Pixel y coordinate.
 * @return The eight-bit luma sample.
 */
static uint8_t AppBaselineTemplate_ReadLuma(
	const uint8_t *frame_bytes, size_t frame_width_pixels,
	size_t x, size_t y)
{
	const size_t pair_offset =
		(y * frame_width_pixels + (x & ~1U)) * 2U;
	return frame_bytes[pair_offset + (((x & 1U) != 0U) ? 2U : 0U)];
}

/**
 * @brief Quantize the normalized 8x8 mean-luma descriptor for one frame.
 * @param frame_bytes Packed YUV422 frame.
 * @param frame_size Number of valid bytes in the frame.
 * @param frame_width_pixels Frame width.
 * @param frame_height_pixels Frame height.
 * @param feature_out Quantized descriptor destination.
 * @return true when all descriptor cells are valid.
 */
static bool AppBaselineTemplate_BuildFeature(
	const uint8_t *frame_bytes, size_t frame_size,
	size_t frame_width_pixels, size_t frame_height_pixels,
	int8_t feature_out[APP_BASELINE_TEMPLATE_FEATURE_COUNT])
{
	float cell_means[APP_BASELINE_TEMPLATE_FEATURE_COUNT] = {0.0f};
	const size_t cell_width = frame_width_pixels / APP_BASELINE_TEMPLATE_GRID_SIZE;
	const size_t cell_height = frame_height_pixels / APP_BASELINE_TEMPLATE_GRID_SIZE;
	float global_mean = 0.0f;
	float variance = 0.0f;

	if ((frame_bytes == NULL) || (feature_out == NULL) ||
		(frame_width_pixels < APP_BASELINE_TEMPLATE_GRID_SIZE) ||
		(frame_height_pixels < APP_BASELINE_TEMPLATE_GRID_SIZE) ||
		(frame_size < (frame_width_pixels * frame_height_pixels * 2U)) ||
		(cell_width == 0U) || (cell_height == 0U))
	{
		return false;
	}

	for (size_t cell_y = 0U;
		 cell_y < APP_BASELINE_TEMPLATE_GRID_SIZE; ++cell_y)
	{
		for (size_t cell_x = 0U;
			 cell_x < APP_BASELINE_TEMPLATE_GRID_SIZE; ++cell_x)
		{
			const size_t feature_index =
				(cell_y * APP_BASELINE_TEMPLATE_GRID_SIZE) + cell_x;
			uint32_t luma_sum = 0U;

			for (size_t y = cell_y * cell_height;
				 y < (cell_y + 1U) * cell_height; ++y)
			{
				for (size_t x = cell_x * cell_width;
					 x < (cell_x + 1U) * cell_width; ++x)
				{
					luma_sum += AppBaselineTemplate_ReadLuma(
						frame_bytes, frame_width_pixels, x, y);
				}
			}

			cell_means[feature_index] =
				(float)luma_sum / (float)(cell_width * cell_height);
			global_mean += cell_means[feature_index];
		}
	}

	global_mean /= (float)APP_BASELINE_TEMPLATE_FEATURE_COUNT;
	for (size_t index = 0U;
		 index < APP_BASELINE_TEMPLATE_FEATURE_COUNT; ++index)
	{
		const float delta = cell_means[index] - global_mean;
		variance += delta * delta;
	}
	variance = sqrtf(
		(variance / (float)APP_BASELINE_TEMPLATE_FEATURE_COUNT) + 1.0f);

	for (size_t index = 0U;
		 index < APP_BASELINE_TEMPLATE_FEATURE_COUNT; ++index)
	{
		float quantized =
			((cell_means[index] - global_mean) / variance) *
			APP_BASELINE_TEMPLATE_QUANT_SCALE;

		if (quantized > 127.0f)
		{
			quantized = 127.0f;
		}
		else if (quantized < -127.0f)
		{
			quantized = -127.0f;
		}
		feature_out[index] = (int8_t)lroundf(quantized);
	}

	return true;
}

/**
 * @brief Convert a template temperature label to the firmware angle frame.
 * @param temperature_c Template temperature label.
 * @return Equivalent calibrated raw angle in radians.
 */
static float AppBaselineTemplate_TemperatureToAngleRad(float temperature_c)
{
	float fraction = APP_BASELINE_TEMPLATE_HOT_FRACTION +
		(((50.0f - temperature_c) / 80.0f) *
		 (APP_BASELINE_TEMPLATE_COLD_FRACTION -
		  APP_BASELINE_TEMPLATE_HOT_FRACTION));

	if (fraction < 0.0f)
	{
		fraction = 0.0f;
	}
	else if (fraction > 1.0f)
	{
		fraction = 1.0f;
	}

	return (APP_BASELINE_TEMPLATE_MIN_ANGLE_DEG +
			(fraction * APP_BASELINE_TEMPLATE_SWEEP_DEG) -
			APP_BASELINE_TEMPLATE_ANGLE_OFFSET_DEG) *
		(3.14159265358979323846f / 180.0f);
}

/**
 * @brief Estimate temperature using nearest-neighbor classical templates.
 * @param frame_bytes Packed YUV422 frame.
 * @param frame_size Number of valid bytes in the frame.
 * @param frame_width_pixels Frame width.
 * @param frame_height_pixels Frame height.
 * @param estimate_out Destination estimate structure.
 * @return true when the descriptor and bank comparison succeed.
 */
bool AppBaselineTemplate_Estimate(
	const uint8_t *frame_bytes, size_t frame_size,
	size_t frame_width_pixels, size_t frame_height_pixels,
	AppBaselineRuntime_Estimate_t *estimate_out)
{
	int8_t feature[APP_BASELINE_TEMPLATE_FEATURE_COUNT] = {0};
	int32_t best_distance = INT_MAX;
	int32_t second_distance = INT_MAX;
	size_t best_index = 0U;
	size_t center_x = 0U;
	size_t center_y = 0U;

	if ((estimate_out == NULL) ||
		!AppBaselineTemplate_BuildFeature(
			frame_bytes, frame_size, frame_width_pixels,
			frame_height_pixels, feature))
	{
		return false;
	}

	for (size_t bank_index = 0U;
		 bank_index < APP_BASELINE_TEMPLATE_BANK_COUNT; ++bank_index)
	{
		int32_t distance = 0;
		for (size_t feature_index = 0U;
			 feature_index < APP_BASELINE_TEMPLATE_FEATURE_COUNT;
			 ++feature_index)
		{
			const int32_t difference =
				(int32_t)feature[feature_index] -
				(int32_t)app_baseline_template_features[
					bank_index][feature_index];
			distance += difference * difference;
		}

		if (distance < best_distance)
		{
			second_distance = best_distance;
			best_distance = distance;
			best_index = bank_index;
		}
		else if (distance < second_distance)
		{
			second_distance = distance;
		}
	}

	if (second_distance == INT_MAX)
	{
		second_distance = best_distance + 1;
	}

	AppGaugeGeometry_TrainingCropCenter(
		frame_width_pixels, frame_height_pixels, &center_x, &center_y);
	(void)memset(estimate_out, 0, sizeof(*estimate_out));
	estimate_out->valid = true;
	estimate_out->center_x = center_x;
	estimate_out->center_y = center_y;
	estimate_out->temperature_c =
		(float)app_baseline_template_temperature_tenths[best_index] / 10.0f;
	estimate_out->angle_rad =
		AppBaselineTemplate_TemperatureToAngleRad(
			estimate_out->temperature_c);
	/* Why: expose actual nearest-neighbour quality instead of manufacturing a
	 * passing score. A distant or near-tied bank match must reach the Hough
	 * fallback rather than silently becoming a false endpoint temperature. */
	estimate_out->best_score = APP_BASELINE_TEMPLATE_SCORE_SCALE /
		((float)best_distance + 1.0f);
	estimate_out->runner_up_score = APP_BASELINE_TEMPLATE_SCORE_SCALE /
		((float)second_distance + 1.0f);
	estimate_out->confidence = 1.0f +
		(0.75f * ((float)(second_distance - best_distance) /
				  (float)(second_distance + 64)));
	estimate_out->source_label = APP_BASELINE_TEMPLATE_SOURCE_LABEL;
	return true;
}
