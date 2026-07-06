/* Tip-focus postprocess helpers for the v18 UNET board-fit model.
 *
 * Included by app_ai_runtime_tail.inc so the heatmap decode logic can live in
 * a smaller, easier-to-debug unit without changing the CubeIDE project files.
 */

void AppAI_TipFocus_LogHeatmapSummary(
	const char *label,
	const float *heatmap,
	size_t width,
	size_t height)
{
	size_t finite_count = 0U;
	size_t positive_count = 0U;
	float min_value = FLT_MAX;
	float max_value = -FLT_MAX;
	double sum_value = 0.0;
	const size_t element_count = width * height;

	if ((heatmap == NULL) || (width == 0U) || (height == 0U))
	{
		DebugConsole_Printf(
			"[AI][TIP_FOCUS] %s: heatmap unavailable\r\n",
			(label != NULL) ? label : "(null)");
		return;
	}

	for (size_t index = 0U; index < element_count; ++index)
	{
		const float value = heatmap[index];

		if (!isfinite(value))
		{
			continue;
		}

		finite_count++;
		sum_value += (double)value;
		if (value < min_value)
		{
			min_value = value;
		}
		if (value > max_value)
		{
			max_value = value;
		}
		if (value > 0.0f)
		{
			positive_count++;
		}
	}

	if (finite_count == 0U)
	{
		min_value = 0.0f;
		max_value = 0.0f;
		sum_value = 0.0;
	}

	DebugConsole_Printf(
		"[AI][TIP_FOCUS] %s: finite=%lu/%lu positive=%lu min=%ld max=%ld sum=%ld first4=[%ld %ld %ld %ld]\r\n",
		(label != NULL) ? label : "(null)",
		(unsigned long)finite_count,
		(unsigned long)element_count,
		(unsigned long)positive_count,
		(long)lroundf(min_value * 1000.0f),
		(long)lroundf(max_value * 1000.0f),
		(long)lroundf((float)sum_value * 1000.0f),
		(long)((element_count > 0U) ? lroundf(heatmap[0U] * 1000.0f) : 0L),
		(long)((element_count > 1U) ? lroundf(heatmap[1U] * 1000.0f) : 0L),
		(long)((element_count > 2U) ? lroundf(heatmap[2U] * 1000.0f) : 0L),
		(long)((element_count > 3U) ? lroundf(heatmap[3U] * 1000.0f) : 0L));
}

bool AppAI_TipFocus_DecodeHeatmap2D(
	const float *heatmap,
	size_t width,
	size_t height,
	float *coord_x_norm_out,
	float *coord_y_norm_out,
	float *peak_out,
	float *spread_x_px_out,
	float *spread_y_px_out)
{
	const size_t element_count = width * height;
	const size_t radius = 2U;
	size_t peak_index = 0U;
	float peak_value = -FLT_MAX;
	float centroid_x = 0.0f;
	float centroid_y = 0.0f;
	float weight_sum = 0.0f;
	float var_x = 0.0f;
	float var_y = 0.0f;

	if ((heatmap == NULL) || (width < 3U) || (height < 3U))
	{
		return false;
	}

	for (size_t index = 0U; index < element_count; ++index)
	{
		const float value = heatmap[index];
		if (!isfinite(value))
		{
			continue;
		}
		if (value > peak_value)
		{
			peak_value = value;
			peak_index = index;
		}
	}

	if (!(peak_value > 0.0f))
	{
		return false;
	}

	{
		const size_t x0 = peak_index % width;
		const size_t y0 = peak_index / width;
		const size_t x_min = (x0 > radius) ? (x0 - radius) : 0U;
		const size_t x_max = ((x0 + radius) < (width - 1U)) ? (x0 + radius) : (width - 1U);
		const size_t y_min = (y0 > radius) ? (y0 - radius) : 0U;
		const size_t y_max = ((y0 + radius) < (height - 1U)) ? (y0 + radius) : (height - 1U);

		for (size_t y = y_min; y <= y_max; ++y)
		{
			const size_t row_base = y * width;
			for (size_t x = x_min; x <= x_max; ++x)
			{
				const float value = heatmap[row_base + x];
				const float weight = (isfinite(value) && (value > 0.0f)) ? value : 0.0f;
				weight_sum += weight;
				centroid_x += weight * (float)x;
				centroid_y += weight * (float)y;
			}
		}

		if (!(weight_sum > 0.0f))
		{
			centroid_x = (float)x0;
			centroid_y = (float)y0;
		}
		else
		{
			centroid_x /= weight_sum;
			centroid_y /= weight_sum;
		}

		for (size_t y = y_min; y <= y_max; ++y)
		{
			const size_t row_base = y * width;
			for (size_t x = x_min; x <= x_max; ++x)
			{
				const float value = heatmap[row_base + x];
				const float weight = (isfinite(value) && (value > 0.0f)) ? value : 0.0f;
				const float dx = ((float)x) - centroid_x;
				const float dy = ((float)y) - centroid_y;
				var_x += weight * dx * dx;
				var_y += weight * dy * dy;
			}
		}

		if (weight_sum > 0.0f)
		{
			var_x /= weight_sum;
			var_y /= weight_sum;
		}
	}

	if (coord_x_norm_out != NULL)
	{
		*coord_x_norm_out = centroid_x / (float)(width - 1U);
	}
	if (coord_y_norm_out != NULL)
	{
		*coord_y_norm_out = centroid_y / (float)(height - 1U);
	}
	if (peak_out != NULL)
	{
		*peak_out = peak_value;
	}
	if (spread_x_px_out != NULL)
	{
		*spread_x_px_out = sqrtf(fmaxf(var_x, 0.0f));
	}
	if (spread_y_px_out != NULL)
	{
		*spread_y_px_out = sqrtf(fmaxf(var_y, 0.0f));
	}

	return true;
}

bool AppAI_TipFocus_RunDryInferenceFromYuv422(
	const uint8_t *frame_bytes, size_t frame_size)
{
	const LL_Buffer_InfoTypeDef *input_info =
		(const LL_Buffer_InfoTypeDef *)AppAI_TipFocus_GetInputBufferInfo();
	uint8_t *input_buffer = (uint8_t *)AppAI_TipFocus_GetInputBuffer();
	const float *center_heatmap = NULL;
	const float *tip_heatmap = NULL;
	float center_x_norm = 0.0f;
	float center_y_norm = 0.0f;
	float tip_x_norm = 0.0f;
	float tip_y_norm = 0.0f;
	float center_peak = 0.0f;
	float tip_peak = 0.0f;
	float center_spread_x = 0.0f;
	float center_spread_y = 0.0f;
	float tip_spread_x = 0.0f;
	float tip_spread_y = 0.0f;
	float confidence = 0.0f;
	float is_main_needle = 0.0f;
	float angle_rad = 0.0f;
	float angle_degrees = 0.0f;
	float angle_delta_degrees = 0.0f;
	float temperature_c = 0.0f;
	float published_temperature_c = 0.0f;
	const float coord_max_px =
		(float)(APP_AI_TIP_FOCUS_HEATMAP_SIDE_PIXELS - 1U);
	const float expected_center_tip_distance_px =
		APP_GAUGE_INNER_DIAL_RADIUS_FRAME_RATIO * coord_max_px;
	bool heatmap_decode_ok = true;
	bool center_peak_ok = false;
	bool tip_peak_ok = false;
	bool confidence_ok = false;
	bool main_needle_ok = false;
	bool center_spread_ok = false;
	bool tip_spread_ok = false;
	bool edge_margin_ok = false;
	bool distance_ok = false;
	bool angle_ok = false;
	bool temperature_finite_ok = false;
	bool temperature_range_ok = false;

	if ((input_info == NULL) || (input_buffer == NULL))
	{
		DebugConsole_WriteString(
			"[AI] Tip-focus init failed: missing model buffers.\r\n");
		return false;
	}

	if (!AppAI_PreprocessYuv422FrameToInt8Input(
			frame_bytes, frame_size,
			input_buffer,
			(size_t)LL_Buffer_len(input_info),
			input_info))
	{
		DebugConsole_WriteString("[AI] Tip-focus preprocess failed.\r\n");
		return false;
	}

	if (!AppAI_TipFocus_Run())
	{
		DebugConsole_WriteString("[AI] Tip-focus model run failed.\r\n");
		return false;
	}

	center_heatmap = AppAI_TipFocus_GetCenterHeatmap();
	tip_heatmap = AppAI_TipFocus_GetTipHeatmap();
	confidence = AppAI_TipFocus_GetConfidence();
	is_main_needle = AppAI_TipFocus_GetIsMainNeedle();

	if ((center_heatmap == NULL) || (tip_heatmap == NULL))
	{
		DebugConsole_WriteString("[AI] Tip-focus outputs missing.\r\n");
		return false;
	}

	AppAI_TipFocus_LogHeatmapSummary(
		"Heatmap center", center_heatmap,
		APP_AI_TIP_FOCUS_HEATMAP_SIDE_PIXELS,
		APP_AI_TIP_FOCUS_HEATMAP_SIDE_PIXELS);
	AppAI_TipFocus_LogHeatmapSummary(
		"Heatmap tip", tip_heatmap,
		APP_AI_TIP_FOCUS_HEATMAP_SIDE_PIXELS,
		APP_AI_TIP_FOCUS_HEATMAP_SIDE_PIXELS);

	heatmap_decode_ok =
		AppAI_TipFocus_DecodeHeatmap2D(
			center_heatmap,
			APP_AI_TIP_FOCUS_HEATMAP_SIDE_PIXELS,
			APP_AI_TIP_FOCUS_HEATMAP_SIDE_PIXELS,
			&center_x_norm,
			&center_y_norm,
			&center_peak,
			&center_spread_x,
			&center_spread_y) &&
		AppAI_TipFocus_DecodeHeatmap2D(
			tip_heatmap,
			APP_AI_TIP_FOCUS_HEATMAP_SIDE_PIXELS,
			APP_AI_TIP_FOCUS_HEATMAP_SIDE_PIXELS,
			&tip_x_norm,
			&tip_y_norm,
			&tip_peak,
			&tip_spread_x,
			&tip_spread_y);

	if (!heatmap_decode_ok)
	{
		DebugConsole_WriteString(
			"[AI] Tip-focus heatmap decode failed.\r\n");
		goto reject_and_hold;
	}

	{
		const float center_x_px = center_x_norm * coord_max_px;
		const float center_y_px = center_y_norm * coord_max_px;
		const float tip_x_px = tip_x_norm * coord_max_px;
		const float tip_y_px = tip_y_norm * coord_max_px;
		const float dx = tip_x_px - center_x_px;
		const float dy = tip_y_px - center_y_px;
		const float center_tip_distance_px = sqrtf((dx * dx) + (dy * dy));
		const float confidence_floor = APP_AI_TIP_FOCUS_CONFIDENCE_FLOOR;
		const float center_axis_peak_floor = APP_AI_TIP_FOCUS_AXIS_PEAK_FLOOR;
		const float center_axis_spread_max_px = APP_AI_TIP_FOCUS_AXIS_SPREAD_MAX_PX;
		const float edge_margin_px =
			fminf(
				fminf(center_x_px, coord_max_px - center_x_px),
				fminf(center_y_px, coord_max_px - center_y_px));
		const float tip_edge_margin_px =
			fminf(
				fminf(tip_x_px, coord_max_px - tip_x_px),
				fminf(tip_y_px, coord_max_px - tip_y_px));
		const float min_edge_margin_px =
			fminf(edge_margin_px, tip_edge_margin_px);
		const float center_tip_distance_ratio =
			(expected_center_tip_distance_px > 0.0f)
				? (center_tip_distance_px / expected_center_tip_distance_px)
				: 0.0f;

		angle_rad = atan2f(-dy, dx);
		angle_degrees = angle_rad * (180.0f / 3.14159265358979f);
		if (angle_degrees < 0.0f)
		{
			angle_degrees += 360.0f;
		}

		angle_delta_degrees =
			fmodf(angle_degrees - APP_AI_TIP_FOCUS_COLD_ANGLE_DEG, 360.0f);
		if (angle_delta_degrees < 0.0f)
		{
			angle_delta_degrees += 360.0f;
		}

		temperature_c = AppBaselineRuntime_ConvertAngleToTemperature(angle_rad);
		{
			const long raw_angle_100 = lroundf(angle_degrees * 100.0f);
			const long cold_angle_100 = lroundf(APP_AI_TIP_FOCUS_COLD_ANGLE_DEG * 100.0f);
			const long sweep_100 = lroundf(APP_AI_TIP_FOCUS_SWEEP_DEG * 100.0f);
			const long delta_100 = lroundf(angle_delta_degrees * 100.0f);
			const long temp_100 = lroundf(temperature_c * 100.0f);

			DebugConsole_Printf(
				"[AI][TIP_FOCUS][ANGLE] raw=%ld.%02lddeg cold=%ld.%02lddeg sweep=%ld.%02lddeg delta=%ld.%02lddeg temp=%ld.%02ldC\r\n",
				labs(raw_angle_100) / 100L, labs(raw_angle_100) % 100L,
				labs(cold_angle_100) / 100L, labs(cold_angle_100) % 100L,
				labs(sweep_100) / 100L, labs(sweep_100) % 100L,
				labs(delta_100) / 100L, labs(delta_100) % 100L,
				labs(temp_100) / 100L, labs(temp_100) % 100L);
		}

		center_peak_ok = (center_peak >= center_axis_peak_floor);
		tip_peak_ok = (tip_peak >= center_axis_peak_floor);
		confidence_ok = (confidence >= confidence_floor);
		main_needle_ok = (is_main_needle >= 0.5f);
		center_spread_ok = (center_spread_x <= center_axis_spread_max_px);
		tip_spread_ok = (tip_spread_x <= center_axis_spread_max_px);
		edge_margin_ok = (min_edge_margin_px >= 4.0f);
		distance_ok =
			(center_tip_distance_ratio >= 0.35f) &&
			(center_tip_distance_ratio <= 1.40f);
		angle_ok = (angle_delta_degrees <= APP_AI_TIP_FOCUS_SWEEP_DEG);
		temperature_finite_ok = AppAI_IsFiniteFloat(temperature_c);
		temperature_range_ok =
			(temperature_c >= APP_AI_TIP_FOCUS_TEMP_MIN_C) &&
			(temperature_c <= APP_AI_TIP_FOCUS_TEMP_MAX_C);

		if (center_peak_ok && tip_peak_ok && confidence_ok && main_needle_ok &&
			center_spread_ok && tip_spread_ok && edge_margin_ok && distance_ok &&
			angle_ok && temperature_finite_ok && temperature_range_ok)
		{
			temperature_c = AppAI_TipFocus_MedianFilter(temperature_c);
			app_ai_tip_focus_consecutive_invalid = 0U;
			app_ai_last_inference_value = temperature_c;
			app_ai_last_inference_valid = true;
			(void)INA219_LogReading("AI-DONE");
			Metrics_EndInference("AI", temperature_c);
			return true;
		}
	}

	DebugConsole_WriteString(
		"[AI][TIP_FOCUS][REJECT] Heatmaps decoded but quality gates failed.\r\n");

reject_and_hold:
	app_ai_tip_focus_consecutive_invalid++;
	if (app_ai_tip_focus_consecutive_invalid >= APP_AI_TIP_FOCUS_MAX_INVALID_FRAMES)
	{
		app_ai_tip_focus_consecutive_invalid = 0U;
	}

	if (app_ai_tip_focus_last_published_valid)
	{
		published_temperature_c = app_ai_tip_focus_last_published;
		app_ai_last_inference_value = published_temperature_c;
		app_ai_last_inference_valid = true;
		DebugConsole_WriteString(
			"[AI] Tip-focus rejected; holding last published temperature.\r\n");
		(void)INA219_LogReading("AI-DONE");
		Metrics_EndInference("AI", published_temperature_c);
		return true;
	}

	app_ai_last_inference_valid = false;
	DebugConsole_WriteString(
		"[AI] Tip-focus rejected and no stable history is available.\r\n");
	Metrics_EndInference("AI", NAN);
	return false;
}
