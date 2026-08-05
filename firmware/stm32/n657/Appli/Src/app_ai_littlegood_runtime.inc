/* Exact LittleGood proposer -> refiner -> center/tip runtime.
 * This file is included by app_ai_runtime_tail.inc so the existing camera
 * worker and calibration publication path remain unchanged. */

#include "ai_network_littlegood_models.h"

#define LG_PROPOSER_SIDE 320U
#define LG_PROPOSER_GRID 80U
#define LG_PROPOSER_CHANNELS 6U
#define LG_REFINER_SIDE 256U
#define LG_CENTER_TIP_SIDE 160U
#define LG_HEATMAP_SIDE 80U
#define LG_SOURCE_SIDE 640U
#define LG_PI 3.14159265358979323846f

static float AppAI_LittleGood_Clamp(float value, float low, float high)
{
	return fminf(fmaxf(value, low), high);
}

static int8_t AppAI_LittleGood_QSigned(float value)
{
	const int32_t q = (int32_t)lroundf(value);
	return (int8_t)((q < -128) ? -128 : (q > 127) ? 127 : q);
}

static float AppAI_LittleGood_ReadGray(const uint8_t *frame, size_t bytes,
	float x, float y)
{
	return AppAI_ReadNormalizedGrayFromDcmippGray8Bilinear(
		frame, bytes, APP_AI_CAPTURE_FRAME_WIDTH_PIXELS,
		APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS, x, y);
}

static float AppAI_LittleGood_Gamma(float gray)
{
	/* why: validation selected gamma 0.70 before the full-frame proposer. */
	return powf(AppAI_LittleGood_Clamp(gray, 0.0f, 1.0f), 0.70f);
}

static void AppAI_LittleGood_FillProposerInput(int8_t *tensor,
	const uint8_t *frame, size_t bytes, unsigned int flip)
{
	for (size_t y = 0U; y < LG_PROPOSER_SIDE; ++y) {
		for (size_t x = 0U; x < LG_PROPOSER_SIDE; ++x) {
			const float sx = ((float)x * 639.0f) / 319.0f;
			const float sy = ((float)y * 639.0f) / 319.0f;
			const float tx = (flip & 1U) ? 639.0f - sx : sx;
			const float ty = (flip & 2U) ? 639.0f - sy : sy;
			const float value = AppAI_LittleGood_Gamma(
				AppAI_LittleGood_ReadGray(frame, bytes, tx, ty));
			tensor[y * LG_PROPOSER_SIDE + x] =
				AppAI_LittleGood_QSigned(value * 255.0f - 128.0f);
		}
	}
}

static void AppAI_LittleGood_DecodeProposal(const int8_t *output,
	unsigned int flip, float proposal[4])
{
	size_t best = 0U;
	int8_t best_heat = -128;
	for (size_t i = 0U; i < LG_PROPOSER_GRID * LG_PROPOSER_GRID; ++i) {
		const int8_t heat = output[i * LG_PROPOSER_CHANNELS];
		if (heat > best_heat) {
			best_heat = heat;
			best = i;
		}
	}
	const size_t gx = best % LG_PROPOSER_GRID;
	const size_t gy = best / LG_PROPOSER_GRID;
	const size_t base = best * LG_PROPOSER_CHANNELS;
	float cx = ((float)gx + ((float)output[base + 1U] + 128.0f) / 256.0f) /
		(float)LG_PROPOSER_GRID;
	float cy = ((float)gy + ((float)output[base + 2U] + 128.0f) / 256.0f) /
		(float)LG_PROPOSER_GRID;
	proposal[2] = AppAI_LittleGood_Clamp(
		((float)output[base + 3U] + 128.0f) / 256.0f, 0.02f, 0.80f);
	proposal[3] = AppAI_LittleGood_Clamp(
		((float)output[base + 4U] + 128.0f) / 256.0f, 0.02f, 0.80f);
	if (flip & 1U) cx = 1.0f - cx;
	if (flip & 2U) cy = 1.0f - cy;
	proposal[0] = AppAI_LittleGood_Clamp(cx, 0.0f, 1.0f);
	proposal[1] = AppAI_LittleGood_Clamp(cy, 0.0f, 1.0f);
}

static float AppAI_LittleGood_Median4(float a, float b, float c, float d)
{
	float values[4] = {a, b, c, d};
	for (unsigned int i = 1U; i < 4U; ++i) {
		const float value = values[i];
		unsigned int j = i;
		while (j > 0U && values[j - 1U] > value) {
			values[j] = values[j - 1U];
			--j;
		}
		values[j] = value;
	}
	return 0.5f * (values[1] + values[2]);
}

static float AppAI_LittleGood_ReadCrop(const uint8_t *frame, size_t bytes,
	float left, float top, float side, float x, float y)
{
	const float sx = left + ((x + 0.5f) * side / (float)LG_REFINER_SIDE) - 0.5f;
	const float sy = top + ((y + 0.5f) * side / (float)LG_REFINER_SIDE) - 0.5f;
	if (sx < 0.0f || sy < 0.0f ||
		sx > 639.0f || sy > 639.0f)
		return 0.5f;
	return AppAI_LittleGood_ReadGray(frame, bytes, sx, sy);
}

static bool AppAI_LittleGood_RunEllipse(const uint8_t *frame, size_t bytes,
	float ellipse[4])
{
	const LL_Buffer_InfoTypeDef *input = AppAI_LittleGoodProposer_InputInfo();
	const LL_Buffer_InfoTypeDef *output = AppAI_LittleGoodProposer_OutputInfo();
	int8_t *tensor;
	const int8_t *dense;
	float proposals[4][4] = {{0.0f}};
	if (input == NULL || output == NULL)
		return false;
	tensor = (int8_t *)LL_Buffer_addr_start(input);
	dense = (const int8_t *)LL_Buffer_addr_start(output);
	if (tensor == NULL || dense == NULL)
		return false;
	for (unsigned int flip = 0U; flip < 4U; ++flip) {
		AppAI_LittleGood_FillProposerInput(tensor, frame, bytes, flip);
		mcu_cache_clean_range((uint32_t)(uintptr_t)tensor,
			(uint32_t)((uintptr_t)tensor + 102400U));
		if (!AppAI_LittleGoodProposer_Run())
			return false;
		mcu_cache_invalidate_range((uint32_t)(uintptr_t)dense,
			(uint32_t)((uintptr_t)dense + 38400U));
		AppAI_LittleGood_DecodeProposal(dense, flip, proposals[flip]);
	}
	for (unsigned int i = 0U; i < 4U; ++i)
		ellipse[i] = AppAI_LittleGood_Median4(
			proposals[0][i], proposals[1][i], proposals[2][i], proposals[3][i]);

	{
		const LL_Buffer_InfoTypeDef *refiner_input = AppAI_LittleGoodRefiner_InputInfo();
		const LL_Buffer_InfoTypeDef *refiner_output = AppAI_LittleGoodRefiner_OutputInfo();
		int8_t *refiner_tensor;
		const int8_t *refiner_values;
		const float side = AppAI_LittleGood_Clamp(
			3.2f * fmaxf(ellipse[2], ellipse[3]), 0.30f, 1.20f);
		const float left = ellipse[0] - 0.5f * side;
		const float top = ellipse[1] - 0.5f * side;
		float local[4];
		if (refiner_input == NULL || refiner_output == NULL)
			return false;
		refiner_tensor = (int8_t *)LL_Buffer_addr_start(refiner_input);
		refiner_values = (const int8_t *)LL_Buffer_addr_start(refiner_output);
		if (refiner_tensor == NULL || refiner_values == NULL)
			return false;
		for (size_t y = 0U; y < LG_REFINER_SIDE; ++y) {
			for (size_t x = 0U; x < LG_REFINER_SIDE; ++x) {
				const float gray = AppAI_LittleGood_ReadCrop(
					frame, bytes, left * 640.0f, top * 640.0f,
					side * 640.0f, (float)x, (float)y);
				refiner_tensor[y * LG_REFINER_SIDE + x] =
					AppAI_LittleGood_QSigned(gray * 255.0f - 128.0f);
			}
		}
		mcu_cache_clean_range((uint32_t)(uintptr_t)refiner_tensor,
			(uint32_t)((uintptr_t)refiner_tensor + 65536U));
		if (!AppAI_LittleGoodRefiner_Run())
			return false;
		mcu_cache_invalidate_range((uint32_t)(uintptr_t)refiner_values,
			(uint32_t)((uintptr_t)refiner_values + 4U));
		for (unsigned int i = 0U; i < 4U; ++i)
			local[i] = ((float)refiner_values[i] + 128.0f) / 256.0f;
		ellipse[0] = 0.15f * ellipse[0] + 0.85f * (left + local[0] * side);
		ellipse[1] = 0.15f * ellipse[1] + 0.85f * (top + local[1] * side);
		ellipse[2] = local[2] * side;
		ellipse[3] = local[3] * side;
	}
	return true;
}

static int8_t AppAI_LittleGood_QCondition(float value)
{
	/* Input scale is 2/255 and zero-point is zero for both channels. */
	return AppAI_LittleGood_QSigned(value * 127.5f);
}

static bool AppAI_LittleGood_DecodeHeatmap(const int8_t *heatmap,
	unsigned int channel, float *x_norm, float *y_norm, float *peak)
{
	size_t peak_x = 0U, peak_y = 0U;
	float peak_value = 0.0f;
	for (size_t y = 0U; y < LG_HEATMAP_SIDE; ++y) {
		for (size_t x = 0U; x < LG_HEATMAP_SIDE; ++x) {
			const size_t index = (y * LG_HEATMAP_SIDE + x) * 2U + channel;
			const float value = ((float)heatmap[index] + 128.0f) / 256.0f;
			if (value > peak_value) {
				peak_value = value; peak_x = x; peak_y = y;
			}
		}
	}
	*peak = peak_value;
	if (peak_value <= 0.03f)
		return false;
	{
		float wx = 0.0f, wy = 0.0f, total = 0.0f;
		const size_t x0 = (peak_x > 4U) ? peak_x - 4U : 0U;
		const size_t y0 = (peak_y > 4U) ? peak_y - 4U : 0U;
		const size_t x1 = (peak_x + 4U < LG_HEATMAP_SIDE) ?
			peak_x + 4U : LG_HEATMAP_SIDE - 1U;
		const size_t y1 = (peak_y + 4U < LG_HEATMAP_SIDE) ?
			peak_y + 4U : LG_HEATMAP_SIDE - 1U;
		for (size_t y = y0; y <= y1; ++y) {
			for (size_t x = x0; x <= x1; ++x) {
				const size_t index = (y * LG_HEATMAP_SIDE + x) * 2U + channel;
				const float value = ((float)heatmap[index] + 128.0f) / 256.0f;
				const float excess = fmaxf(value - 0.03f, 0.0f);
				const float weight = excess * excess;
				wx += (float)x * weight; wy += (float)y * weight; total += weight;
			}
		}
		if (total <= 0.0f)
			return false;
		*x_norm = (wx / total + 0.5f) / (float)LG_HEATMAP_SIDE;
		*y_norm = (wy / total + 0.5f) / (float)LG_HEATMAP_SIDE;
	}
	return true;
}

static bool __attribute__((unused)) AppAI_LittleGood_RunGaugeModelsFromGray640(const uint8_t *frame,
	size_t bytes)
{
	const LL_Buffer_InfoTypeDef *input = AppAI_LittleGoodCenterTip_New_InputInfo();
	const LL_Buffer_InfoTypeDef *output = AppAI_LittleGoodCenterTip_New_OutputInfo();
	int8_t *tensor;
	const int8_t *heatmap;
	float ellipse[4];
	float center_x, center_y, tip_x, tip_y, center_peak, tip_peak;
	if (frame == NULL || bytes < (size_t)APP_AI_CAPTURE_FRAME_BYTES ||
		input == NULL || output == NULL ||
		!AppAI_LittleGood_RunEllipse(frame, bytes, ellipse))
		return false;
	tensor = (int8_t *)LL_Buffer_addr_start(input);
	if (tensor == NULL)
		return false;
	const float side = AppAI_LittleGood_Clamp(
		1.35f * fmaxf(2.0f * ellipse[2], 2.0f * ellipse[3]), 0.05f, 1.60f);
	const float left = ellipse[0] - 0.5f * side;
	const float top = ellipse[1] - 0.5f * side;
	for (size_t y = 0U; y < LG_CENTER_TIP_SIDE; ++y) {
		for (size_t x = 0U; x < LG_CENTER_TIP_SIDE; ++x) {
			const float sx = left * 640.0f + ((float)x + 0.5f) *
				(side * 640.0f / 160.0f) - 0.5f;
			const float sy = top * 640.0f + ((float)y + 0.5f) *
				(side * 640.0f / 160.0f) - 0.5f;
			const float gray = (sx < 0.0f || sy < 0.0f || sx > 639.0f || sy > 639.0f) ?
				0.5f : AppAI_LittleGood_ReadGray(frame, bytes, sx, sy);
			const float nx = ((float)x + 0.5f) / 160.0f;
			const float ny = ((float)y + 0.5f) / 160.0f;
			const float ellipse_x = (nx * 160.0f - 80.0f) / fmaxf(ellipse[2] * 160.0f / side, 1.0f);
			const float ellipse_y = (ny * 160.0f - 80.0f) / fmaxf(ellipse[3] * 160.0f / side, 1.0f);
			const float mask = (ellipse_x * ellipse_x + ellipse_y * ellipse_y <= 1.0f) ? 1.0f : 0.0f;
			const size_t index = (y * LG_CENTER_TIP_SIDE + x) * 2U;
			tensor[index] = AppAI_LittleGood_QCondition(gray * 2.0f - 1.0f);
			tensor[index + 1U] = AppAI_LittleGood_QCondition(mask * 2.0f - 1.0f);
		}
	}
	mcu_cache_clean_range((uint32_t)(uintptr_t)tensor,
		(uint32_t)((uintptr_t)tensor + 51200U));
	if (!AppAI_LittleGoodCenterTip_New_Run())
		return false;
	heatmap = (const int8_t *)LL_Buffer_addr_start(output);
	if (heatmap == NULL)
		return false;
	mcu_cache_invalidate_range((uint32_t)(uintptr_t)heatmap,
		(uint32_t)((uintptr_t)heatmap + 12800U));
	if (!AppAI_LittleGood_DecodeHeatmap(heatmap, 0U, &center_x, &center_y, &center_peak) ||
		!AppAI_LittleGood_DecodeHeatmap(heatmap, 1U, &tip_x, &tip_y, &tip_peak))
		return false;
	center_x = left + center_x * side;
	center_y = top + center_y * side;
	tip_x = left + tip_x * side;
	tip_y = top + tip_y * side;
	{
		const float dx = tip_x - center_x;
		const float dy = tip_y - center_y;
		float angle_deg = atan2f(dx, -dy) * (180.0f / LG_PI);
		float value;
		while (angle_deg <= -180.0f) angle_deg += 360.0f;
		while (angle_deg > 180.0f) angle_deg -= 360.0f;
		if ((center_peak < APP_AI_TIP_FOCUS_AXIS_PEAK_FLOOR) ||
			(tip_peak < APP_AI_TIP_FOCUS_AXIS_PEAK_FLOOR) ||
			(fabsf(dx) + fabsf(dy) < 0.03f) ||
			(angle_deg < APP_GAUGE_CALIBRATION_MIN_DEG - 5.0f) ||
			(angle_deg > APP_GAUGE_CALIBRATION_MAX_DEG + 5.0f))
			return false;
		angle_deg = AppAI_LittleGood_Clamp(angle_deg,
			APP_GAUGE_CALIBRATION_MIN_DEG, APP_GAUGE_CALIBRATION_MAX_DEG);
		value = AppBaselineRuntime_MapAngleToTemperature(angle_deg);
		if (!AppAI_IsPlausibleInferenceValue(value))
			return false;
		app_ai_last_inference_value = value;
		app_ai_last_inference_valid = true;
		DebugConsole_Printf("[AI][LITTLEGOOD] ellipse_milli=[%ld,%ld,%ld,%ld] center_milli=[%ld,%ld] tip_milli=[%ld,%ld] angle_milli=%ld value_milli=%ld\r\n",
			(long)lroundf(ellipse[0] * 1000.0f), (long)lroundf(ellipse[1] * 1000.0f),
			(long)lroundf(ellipse[2] * 1000.0f), (long)lroundf(ellipse[3] * 1000.0f),
			(long)lroundf(center_x * 1000.0f), (long)lroundf(center_y * 1000.0f),
			(long)lroundf(tip_x * 1000.0f), (long)lroundf(tip_y * 1000.0f),
			(long)lroundf(angle_deg * 1000.0f), (long)lroundf(value * 1000.0f));
		app_ai_tip_focus_last_published = value;
		app_ai_tip_focus_last_published_valid = true;
		(void)INA219_LogReading("AI-DONE");
		Metrics_EndInference("AI", value);
	}
	return true;
}
