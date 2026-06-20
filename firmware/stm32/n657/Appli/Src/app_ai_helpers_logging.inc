/* This file is included by app_ai_helpers.inc. */

#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
static void AppAI_LogXspi2PrefixBytes(const char *label,
									  const uint8_t *bytes)
{
	if ((label == NULL) || (bytes == NULL))
	{
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
									size_t frame_size)
{
	uint8_t first_bytes[8U] = {0U};
	uint32_t hash = 2166136261UL;
	size_t preview_count = 0U;

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
	{
		return;
	}

	if ((frame_bytes == NULL) || (frame_size == 0U))
	{
		DebugConsole_Printf("[AI] Frame signature skipped: empty frame.\r\n");
		return;
	}

	preview_count = (frame_size < sizeof(first_bytes)) ? frame_size
													   : sizeof(first_bytes);
	for (size_t index = 0U; index < preview_count; index++)
	{
		first_bytes[index] = frame_bytes[index];
	}

	for (size_t index = 0U; index < frame_size; index++)
	{
		hash ^= frame_bytes[index];
		hash *= 16777619UL;
	}

	DebugConsole_Printf(
		"[AI] Frame signature: len=%lu hash=0x%08lX first8=[%02X %02X %02X %02X %02X %02X %02X %02X]\r\n",
		(unsigned long)frame_size, (unsigned long)hash,
		(unsigned int)first_bytes[0], (unsigned int)first_bytes[1],
		(unsigned int)first_bytes[2], (unsigned int)first_bytes[3],
		(unsigned int)first_bytes[4], (unsigned int)first_bytes[5],
		(unsigned int)first_bytes[6], (unsigned int)first_bytes[7]);
}

/**
 * @brief Print a compact signature for the preprocessed model input tensor.
 *
 * This tells us whether the captured scene is still distinct after the
 * YUV422-to-RGB preprocessing and dial ROI crop.
 */
static void AppAI_LogInputSignature(const float *input_buffer,
									size_t input_float_count)
{
	const uint8_t *bytes = NULL;
	uint8_t first_bytes[8U] = {0U};
	uint32_t first_words[4U] = {0U};
	uint32_t hash = 2166136261UL;
	size_t byte_count = 0U;
	size_t preview_bytes = 0U;
	size_t preview_words = 0U;

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
	{
		return;
	}

	if ((input_buffer == NULL) || (input_float_count == 0U))
	{
		DebugConsole_Printf("[AI] Input signature skipped: empty tensor.\r\n");
		return;
	}

	bytes = (const uint8_t *)input_buffer;
	byte_count = input_float_count * sizeof(float);
	preview_bytes = (byte_count < sizeof(first_bytes)) ? byte_count
													   : sizeof(first_bytes);
	preview_words = (input_float_count < 4U) ? input_float_count : 4U;

	for (size_t index = 0U; index < preview_bytes; index++)
	{
		first_bytes[index] = bytes[index];
	}

	for (size_t index = 0U; index < preview_words; index++)
	{
		(void)memcpy(&first_words[index], &input_buffer[index],
					 sizeof(uint32_t));
	}

	for (size_t index = 0U; index < byte_count; index++)
	{
		hash ^= bytes[index];
		hash *= 16777619UL;
	}

	DebugConsole_Printf(
		"[AI] Input signature: floats=%lu hash=0x%08lX first8=[%02X %02X %02X %02X %02X %02X %02X %02X] first4=[0x%08lX,0x%08lX,0x%08lX,0x%08lX]\r\n",
		(unsigned long)input_float_count, (unsigned long)hash,
		(unsigned int)first_bytes[0], (unsigned int)first_bytes[1],
		(unsigned int)first_bytes[2], (unsigned int)first_bytes[3],
		(unsigned int)first_bytes[4], (unsigned int)first_bytes[5],
		(unsigned int)first_bytes[6], (unsigned int)first_bytes[7],
		(unsigned long)first_words[0], (unsigned long)first_words[1],
		(unsigned long)first_words[2], (unsigned long)first_words[3]);
}

/**
 * @brief Print a small diagnostic window from the center of the input tensor.
 *
 * The top-left bytes can be padding or background, so this summary looks at the
 * tensor region where the dial should live after cropping and resizing.
 */
static void AppAI_LogInputTensorWindow(const float *input_buffer,
									   size_t input_float_count)
{
	const size_t tensor_width = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const size_t tensor_height = (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	const size_t center_x = tensor_width / 2U;
	const size_t center_y = tensor_height / 2U;
	const size_t window_radius = 8U;
	const size_t x_min = (center_x > window_radius) ? (center_x - window_radius)
													: 0U;
	const size_t y_min = (center_y > window_radius) ? (center_y - window_radius)
													: 0U;
	const size_t x_max = ((center_x + window_radius) < tensor_width) ? (center_x + window_radius) : tensor_width;
	const size_t y_max = ((center_y + window_radius) < tensor_height) ? (center_y + window_radius) : tensor_height;
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

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
	{
		return;
	}

	if ((input_buffer == NULL) || (input_float_count < APP_AI_MODEL_INPUT_FLOAT_COUNT))
	{
		return;
	}

	for (size_t y = y_min; y < y_max; y++)
	{
		for (size_t x = x_min; x < x_max; x++)
		{
			const size_t pixel_index = (y * tensor_width) + x;
			const size_t base = pixel_index * 3U;
			const float r = input_buffer[base + 0U];
			const float g = input_buffer[base + 1U];
			const float b = input_buffer[base + 2U];

			if (r < min_r)
			{
				min_r = r;
			}
			if (g < min_g)
			{
				min_g = g;
			}
			if (b < min_b)
			{
				min_b = b;
			}
			if (r > max_r)
			{
				max_r = r;
			}
			if (g > max_g)
			{
				max_g = g;
			}
			if (b > max_b)
			{
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

	if (sample_count == 0U)
	{
		return;
	}

	center_r_milli = (unsigned long)((center_r * 1000.0f) + 0.5f);
	center_g_milli = (unsigned long)((center_g * 1000.0f) + 0.5f);
	center_b_milli = (unsigned long)((center_b * 1000.0f) + 0.5f);
	mean_r_milli = (unsigned long)((sum_r / (float)sample_count) * 1000.0f + 0.5f);
	mean_g_milli = (unsigned long)((sum_g / (float)sample_count) * 1000.0f + 0.5f);
	mean_b_milli = (unsigned long)((sum_b / (float)sample_count) * 1000.0f + 0.5f);
	min_r_milli = (unsigned long)(min_r * 1000.0f + 0.5f);
	min_g_milli = (unsigned long)(min_g * 1000.0f + 0.5f);
	min_b_milli = (unsigned long)(min_b * 1000.0f + 0.5f);
	max_r_milli = (unsigned long)(max_r * 1000.0f + 0.5f);
	max_g_milli = (unsigned long)(max_g * 1000.0f + 0.5f);
	max_b_milli = (unsigned long)(max_b * 1000.0f + 0.5f);

	DebugConsole_Printf(
		"[AI] Tensor center window: x=[%lu,%lu) y=[%lu,%lu) center_milli=[%lu %lu %lu] mean_milli=[%lu %lu %lu] min_milli=[%lu %lu %lu] max_milli=[%lu %lu %lu]\r\n",
		(unsigned long)x_min, (unsigned long)x_max, (unsigned long)y_min,
		(unsigned long)y_max, center_r_milli, center_g_milli,
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
									   size_t input_float_count)
{
	const size_t tensor_width = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const size_t tensor_height = (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	const size_t sample_points[4U][2U] = {
		{tensor_width / 2U, tensor_height / 2U},
		{tensor_width / 2U, tensor_height / 4U},
		{tensor_width / 4U, tensor_height / 2U},
		{(tensor_width * 3U) / 4U, tensor_height / 2U},
	};
	const char *sample_labels[4U] = {
		"center",
		"top-mid",
		"left-mid",
		"right-mid",
	};
	const uint8_t *bytes = NULL;
	uint8_t first_bytes[8U] = {0U};
	uint32_t hash = 2166136261UL;
	uint32_t first_words[4U] = {0U};
	float sum_value = 0.0f;
	float min_value = 1.0f;
	float max_value = 0.0f;
	size_t active_count = 0U;
	size_t byte_count = 0U;
	size_t preview_bytes = 0U;
	size_t preview_words = 0U;

	if ((input_buffer == NULL) || (input_float_count < APP_AI_MODEL_INPUT_FLOAT_COUNT))
	{
		(void)DebugConsole_WriteString("[AI] Input probe skipped.\r\n");
		return;
	}

	bytes = (const uint8_t *)input_buffer;
	byte_count = input_float_count * sizeof(float);
	preview_bytes = (byte_count < sizeof(first_bytes)) ? byte_count
													   : sizeof(first_bytes);
	preview_words = (input_float_count < 4U) ? input_float_count : 4U;

	for (size_t index = 0U; index < preview_bytes; index++)
	{
		first_bytes[index] = bytes[index];
	}

	for (size_t index = 0U; index < preview_words; index++)
	{
		(void)memcpy(&first_words[index], &input_buffer[index],
					 sizeof(uint32_t));
	}

	for (size_t index = 0U; index < byte_count; index++)
	{
		hash ^= bytes[index];
		hash *= 16777619UL;
	}

	for (size_t index = 0U; index < input_float_count; index++)
	{
		const float value = input_buffer[index];

		sum_value += value;
		if (value < min_value)
		{
			min_value = value;
		}
		if (value > max_value)
		{
			max_value = value;
		}
		if ((value > 0.001f) || (value < -0.001f))
		{
			active_count++;
		}
	}

	{
		char line[224];
		(void)DebugConsole_Snprintf(line, sizeof(line),
					   "[AI] Input probe: floats=%lu bytes=%lu hash=0x%08lX first8=[%02X %02X %02X %02X %02X %02X %02X %02X] first4=[0x%08lX 0x%08lX 0x%08lX 0x%08lX]\r\n",
					   (unsigned long)input_float_count,
					   (unsigned long)byte_count, (unsigned long)hash,
					   (unsigned int)first_bytes[0], (unsigned int)first_bytes[1],
					   (unsigned int)first_bytes[2], (unsigned int)first_bytes[3],
					   (unsigned int)first_bytes[4], (unsigned int)first_bytes[5],
					   (unsigned int)first_bytes[6], (unsigned int)first_bytes[7],
					   (unsigned long)first_words[0], (unsigned long)first_words[1],
					   (unsigned long)first_words[2], (unsigned long)first_words[3]);
		(void)DebugConsole_WriteString(line);
	}

	{
		const unsigned long mean_milli =
			(unsigned long)(((sum_value / (float)input_float_count) * 1000.0f) + 0.5f);
		const unsigned long min_milli =
			(unsigned long)((min_value * 1000.0f) + 0.5f);
		const unsigned long max_milli =
			(unsigned long)((max_value * 1000.0f) + 0.5f);
		char line[192];

		(void)DebugConsole_Snprintf(line, sizeof(line),
					   "[AI] Input probe stats: active=%lu/%lu mean_milli=%lu min_milli=%lu max_milli=%lu\r\n",
					   (unsigned long)active_count, (unsigned long)input_float_count,
					   mean_milli, min_milli, max_milli);
		(void)DebugConsole_WriteString(line);
	}

	for (size_t index = 0U; index < 4U; index++)
	{
		const size_t sample_x = sample_points[index][0U];
		const size_t sample_y = sample_points[index][1U];
		const size_t pixel_index = (sample_y * tensor_width) + sample_x;
		const size_t base = pixel_index * 3U;
		const uint32_t r_bits = 0U;
		const uint32_t g_bits = 0U;
		const uint32_t b_bits = 0U;
		const unsigned long r_milli =
			(unsigned long)((input_buffer[base + 0U] * 1000.0f) + 0.5f);
		const unsigned long g_milli =
			(unsigned long)((input_buffer[base + 1U] * 1000.0f) + 0.5f);
		const unsigned long b_milli =
			(unsigned long)((input_buffer[base + 2U] * 1000.0f) + 0.5f);
		uint32_t sample_bits[3U] = {0U, 0U, 0U};
		char line[160];

		(void)memcpy(&sample_bits[0U], &input_buffer[base + 0U],
					 sizeof(uint32_t));
		(void)memcpy(&sample_bits[1U], &input_buffer[base + 1U],
					 sizeof(uint32_t));
		(void)memcpy(&sample_bits[2U], &input_buffer[base + 2U],
					 sizeof(uint32_t));

		(void)r_bits;
		(void)g_bits;
		(void)b_bits;

		(void)DebugConsole_Snprintf(line, sizeof(line),
					   "[AI] Input probe %s: x=%lu y=%lu rgb_milli=[%lu %lu %lu] rgb_bits=[0x%08lX 0x%08lX 0x%08lX]\r\n",
					   sample_labels[index], (unsigned long)sample_x,
					   (unsigned long)sample_y, r_milli, g_milli, b_milli,
					   (unsigned long)sample_bits[0U],
					   (unsigned long)sample_bits[1U],
					   (unsigned long)sample_bits[2U]);
		(void)DebugConsole_WriteString(line);
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
									  size_t x_min, size_t x_max)
{
	const size_t sample_count = 5U;
	size_t positions[5U] = {0U};

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
	{
		return;
	}

	if ((label == NULL) || (input_buffer == NULL) || (tensor_width == 0U) || (x_max <= x_min))
	{
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
						label, (unsigned long)row_y);
	for (size_t index = 0U; index < sample_count; index++)
	{
		const size_t pixel_index = (row_y * tensor_width) + positions[index];
		const size_t base = pixel_index * 3U;
		const unsigned long r_milli =
			(unsigned long)(input_buffer[base + 0U] * 1000.0f + 0.5f);
		const unsigned long g_milli =
			(unsigned long)(input_buffer[base + 1U] * 1000.0f + 0.5f);
		const unsigned long b_milli =
			(unsigned long)(input_buffer[base + 2U] * 1000.0f + 0.5f);

		DebugConsole_Printf(" x=%lu rgb=[%lu %lu %lu]",
							(unsigned long)positions[index], r_milli, g_milli, b_milli);
	}
	DebugConsole_Printf("\r\n");
}

/**
 * @brief Print a compact luma patch from the source YUV422 frame.
 */
static void AppAI_LogSourcePatch(const char *label, const uint8_t *frame_bytes,
								 size_t frame_width_pixels, size_t center_x, size_t center_y,
								 size_t radius_pixels)
{
	const size_t x_min = (center_x > radius_pixels) ? (center_x - radius_pixels)
													: 0U;
	const size_t y_min = (center_y > radius_pixels) ? (center_y - radius_pixels)
													: 0U;
	const size_t x_max = ((center_x + radius_pixels) < APP_AI_CAPTURE_FRAME_WIDTH_PIXELS) ? (center_x + radius_pixels)
																						  : (APP_AI_CAPTURE_FRAME_WIDTH_PIXELS - 1U);
	const size_t y_max = ((center_y + radius_pixels) < APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS) ? (center_y + radius_pixels)
																						   : (APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS - 1U);

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
	{
		return;
	}

	if ((label == NULL) || (frame_bytes == NULL) || (frame_width_pixels == 0U))
	{
		return;
	}

	DebugConsole_Printf(
		"[AI] %s source patch center=(%lu,%lu) x=[%lu,%lu] y=[%lu,%lu]\r\n",
		label, (unsigned long)center_x, (unsigned long)center_y,
		(unsigned long)x_min, (unsigned long)x_max,
		(unsigned long)y_min, (unsigned long)y_max);

	for (size_t y = y_min; y <= y_max; ++y)
	{
		DebugConsole_Printf("[AI] %s y=%lu:", label, (unsigned long)y);
		for (size_t x = x_min; x <= x_max; ++x)
		{
			const uint8_t luma = AppAI_ReadYuv422Luma(frame_bytes, frame_size,
													  frame_width_pixels, x, y);
			const unsigned long luma_milli =
				(unsigned long)((luma * 1000U) / 255U);

			DebugConsole_Printf(" x=%lu=%lu", (unsigned long)x, luma_milli);
		}
		DebugConsole_Printf("\r\n");
	}
}

/**
 * @brief Print a compact tensor patch from the preprocessed input buffer.
 */
static void AppAI_LogTensorPatch(const char *label, const float *input_buffer,
								 size_t tensor_width, size_t center_x, size_t center_y,
								 size_t radius_pixels)
{
	const size_t x_min = (center_x > radius_pixels) ? (center_x - radius_pixels)
													: 0U;
	const size_t y_min = (center_y > radius_pixels) ? (center_y - radius_pixels)
													: 0U;
	const size_t x_max = ((center_x + radius_pixels) < tensor_width) ? (center_x + radius_pixels) : (tensor_width - 1U);
	const size_t y_max = ((center_y + radius_pixels) < APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS) ? (center_y + radius_pixels)
																						   : (APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS - 1U);

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
	{
		return;
	}

	if ((label == NULL) || (input_buffer == NULL) || (tensor_width == 0U))
	{
		return;
	}

	DebugConsole_Printf(
		"[AI] %s tensor patch center=(%lu,%lu) x=[%lu,%lu] y=[%lu,%lu]\r\n",
		label, (unsigned long)center_x, (unsigned long)center_y,
		(unsigned long)x_min, (unsigned long)x_max,
		(unsigned long)y_min, (unsigned long)y_max);

	for (size_t y = y_min; y <= y_max; ++y)
	{
		DebugConsole_Printf("[AI] %s y=%lu:", label, (unsigned long)y);
		for (size_t x = x_min; x <= x_max; ++x)
		{
			const size_t pixel_index = (y * tensor_width) + x;
			const size_t base = pixel_index * 3U;
			const unsigned long r_milli =
				(unsigned long)(input_buffer[base + 0U] * 1000.0f + 0.5f);
			const unsigned long g_milli =
				(unsigned long)(input_buffer[base + 1U] * 1000.0f + 0.5f);
			const unsigned long b_milli =
				(unsigned long)(input_buffer[base + 2U] * 1000.0f + 0.5f);

			DebugConsole_Printf(" x=%lu=[%lu %lu %lu]",
								(unsigned long)x, r_milli, g_milli, b_milli);
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
									  size_t crop_height)
{
	const size_t center_x = crop_x_min + (crop_width / 2U);
	const size_t center_y = crop_y_min + (crop_height / 2U);
	const size_t window_radius = 8U;
	const size_t x_min = (center_x > window_radius) ? (center_x - window_radius)
													: crop_x_min;
	const size_t y_min = (center_y > window_radius) ? (center_y - window_radius)
													: crop_y_min;
	const size_t x_max = ((center_x + window_radius) < (crop_x_min + crop_width)) ? (center_x + window_radius) : (crop_x_min + crop_width);
	const size_t y_max = ((center_y + window_radius) < (crop_y_min + crop_height)) ? (center_y + window_radius) : (crop_y_min + crop_height);
	uint64_t sum_luma = 0U;
	uint8_t min_luma = 0xFFU;
	uint8_t max_luma = 0U;
	uint8_t center_luma = 0U;
	unsigned long center_luma_milli = 0U;
	unsigned long mean_luma_milli = 0U;
	unsigned long min_luma_milli = 0U;
	unsigned long max_luma_milli = 0U;
	size_t sample_count = 0U;

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
	{
		return;
	}

	if ((frame_bytes == NULL) || (frame_size < (frame_width_pixels * frame_height_pixels * 2U)))
	{
		return;
	}

	for (size_t y = y_min; y < y_max; ++y)
	{
		for (size_t x = x_min; x < x_max; ++x)
		{
			const uint8_t luma = AppAI_ReadYuv422Luma(frame_bytes, frame_size,
													  frame_width_pixels, x, y);

			if (luma < min_luma)
			{
				min_luma = luma;
			}
			if (luma > max_luma)
			{
				max_luma = luma;
			}
			sum_luma += (uint64_t)luma;
			sample_count++;
		}
	}

	center_luma = AppAI_ReadYuv422Luma(frame_bytes, frame_size,
									   center_x, center_y);

	if (sample_count == 0U)
	{
		return;
	}

	center_luma_milli = (unsigned long)((center_luma * 1000U) / 255U);
	mean_luma_milli = (unsigned long)(((sum_luma / sample_count) * 1000U) / 255U);
	min_luma_milli = (unsigned long)((min_luma * 1000U) / 255U);
	max_luma_milli = (unsigned long)((max_luma * 1000U) / 255U);

	DebugConsole_Printf(
		"[AI] Source crop window: x=[%lu,%lu) y=[%lu,%lu) center_luma_milli=[%lu] mean_luma_milli=[%lu] min_luma_milli=[%lu] max_luma_milli=[%lu]\r\n",
		(unsigned long)x_min, (unsigned long)x_max, (unsigned long)y_min,
		(unsigned long)y_max, center_luma_milli, mean_luma_milli,
		min_luma_milli, max_luma_milli);

	AppAI_LogSourcePatch("Source crop center", frame_bytes, frame_width_pixels,
						 center_x, center_y, 2U);

	for (size_t index = 0U; index < 3U; ++index)
	{
		const char *label = (index == 0U)	? "src_top"
							: (index == 1U) ? "src_mid"
											: "src_bottom";
		const size_t row_y = (index == 0U)	 ? (y_min + ((y_max - y_min) / 4U))
							 : (index == 1U) ? center_y
											 : (y_min + (((y_max - y_min) * 3U) / 4U));
		const size_t sample_span = (x_max > x_min) ? (x_max - x_min - 1U) : 0U;
		const size_t sample_x0 = x_min;
		const size_t sample_x1 = x_min + (sample_span / 4U);
		const size_t sample_x2 = x_min + (sample_span / 2U);
		const size_t sample_x3 = x_min + ((sample_span * 3U) / 4U);
		const size_t sample_x4 = (x_max > 0U) ? (x_max - 1U) : 0U;
		const size_t sample_xs[5U] = {
			sample_x0, sample_x1, sample_x2, sample_x3, sample_x4};

		DebugConsole_Printf("[AI] %s y=%lu:", label, (unsigned long)row_y);
		for (size_t sample_index = 0U; sample_index < 5U; ++sample_index)
		{
			const uint8_t luma = AppAI_ReadYuv422Luma(frame_bytes, frame_size,
													  frame_width_pixels, sample_xs[sample_index], row_y);
			const unsigned long luma_milli =
				(unsigned long)((luma * 1000U) / 255U);

			DebugConsole_Printf(" x=%lu y=%lu",
								(unsigned long)sample_xs[sample_index], luma_milli);
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
										 const int8_t *buffer_ptr, size_t buffer_len_bytes)
{
	uint8_t first_bytes[16U] = {0U};
	int8_t min_value = 127;
	int8_t max_value = -128;
	uint32_t nonzero_count = 0U;
	uint32_t hash = 2166136261UL;
	size_t preview_count = 0U;

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
	{
		return;
	}

	if ((label == NULL) || (buffer_ptr == NULL) || (buffer_len_bytes == 0U))
	{
		DebugConsole_Printf(
			"[AI] %s int8 signature skipped: empty buffer.\r\n",
			(label != NULL) ? label : "(unnamed)");
		return;
	}

	preview_count = (buffer_len_bytes < sizeof(first_bytes)) ? buffer_len_bytes
															 : sizeof(first_bytes);

	for (size_t index = 0U; index < buffer_len_bytes; ++index)
	{
		const uint8_t raw_byte = (uint8_t)buffer_ptr[index];

		hash ^= raw_byte;
		hash *= 16777619UL;

		if (buffer_ptr[index] < min_value)
		{
			min_value = buffer_ptr[index];
		}
		if (buffer_ptr[index] > max_value)
		{
			max_value = buffer_ptr[index];
		}
		if (buffer_ptr[index] != 0)
		{
			nonzero_count++;
		}
		if (index < preview_count)
		{
			first_bytes[index] = raw_byte;
		}
	}

	DebugConsole_Printf(
		"[AI] %s int8 signature: bytes=%lu hash=0x%08lX nonzero=%lu min=%d max=%d first16=[%02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X]\r\n",
		label, (unsigned long)buffer_len_bytes, (unsigned long)hash,
		(unsigned long)nonzero_count, (int)min_value, (int)max_value,
		(unsigned int)first_bytes[0], (unsigned int)first_bytes[1],
		(unsigned int)first_bytes[2], (unsigned int)first_bytes[3],
		(unsigned int)first_bytes[4], (unsigned int)first_bytes[5],
		(unsigned int)first_bytes[6], (unsigned int)first_bytes[7],
		(unsigned int)first_bytes[8], (unsigned int)first_bytes[9],
		(unsigned int)first_bytes[10], (unsigned int)first_bytes[11],
		(unsigned int)first_bytes[12], (unsigned int)first_bytes[13],
		(unsigned int)first_bytes[14], (unsigned int)first_bytes[15]);
}

/**
 * @brief Print a compact signature for any buffer by raw bytes.
 *
 * This avoids guessing the tensor type when we just want to know whether the
 * runtime wrote anything other than zero into a model activation.
 */
static void AppAI_LogRawBufferSignature(const char *label,
										const uint8_t *buffer_ptr, size_t buffer_len_bytes)
{
	uint8_t first_bytes[16U] = {0U};
	uint8_t min_value = 255U;
	uint8_t max_value = 0U;
	uint32_t nonzero_count = 0U;
	uint32_t hash = 2166136261UL;
	size_t preview_count = 0U;

	if (!APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
	{
		return;
	}

	if ((label == NULL) || (buffer_ptr == NULL) || (buffer_len_bytes == 0U))
	{
		DebugConsole_Printf(
			"[AI] %s raw signature skipped: empty buffer.\r\n",
			(label != NULL) ? label : "(unnamed)");
		return;
	}

	preview_count = (buffer_len_bytes < sizeof(first_bytes)) ? buffer_len_bytes
															 : sizeof(first_bytes);

	for (size_t index = 0U; index < buffer_len_bytes; ++index)
	{
		const uint8_t value = buffer_ptr[index];

		hash ^= value;
		hash *= 16777619UL;

		if (value < min_value)
		{
			min_value = value;
		}
		if (value > max_value)
		{
			max_value = value;
		}
		if (value != 0U)
		{
			nonzero_count++;
		}
		if (index < preview_count)
		{
			first_bytes[index] = value;
		}
	}

	DebugConsole_Printf(
		"[AI] %s raw signature: bytes=%lu hash=0x%08lX nonzero=%lu min=%u max=%u first16=[%02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X]\r\n",
		label, (unsigned long)buffer_len_bytes, (unsigned long)hash,
		(unsigned long)nonzero_count, (unsigned int)min_value,
		(unsigned int)max_value, (unsigned int)first_bytes[0],
		(unsigned int)first_bytes[1], (unsigned int)first_bytes[2],
		(unsigned int)first_bytes[3], (unsigned int)first_bytes[4],
		(unsigned int)first_bytes[5], (unsigned int)first_bytes[6],
		(unsigned int)first_bytes[7], (unsigned int)first_bytes[8],
		(unsigned int)first_bytes[9], (unsigned int)first_bytes[10],
		(unsigned int)first_bytes[11], (unsigned int)first_bytes[12],
		(unsigned int)first_bytes[13], (unsigned int)first_bytes[14],
		(unsigned int)first_bytes[15]);
}

static const char *AppAI_BufferTypeName(const LL_Buffer_InfoTypeDef *buffer_info)
{
	if (buffer_info == NULL)
	{
		return "(null)";
	}

	switch (buffer_info->type)
	{
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
											const LL_Buffer_InfoTypeDef *buffer_info)
{
	const void *buffer_addr = NULL;
	size_t buffer_len = 0U;
	float scale_value = 0.0f;
	int16_t offset_value = 0;
	const void *scale_addr = NULL;
	const void *offset_addr = NULL;

	if ((label == NULL) || (buffer_info == NULL))
	{
		DebugConsole_Printf("[AI] %s buffer info unavailable.\r\n",
							(label != NULL) ? label : "(unnamed)");
		return;
	}

	buffer_addr = LL_Buffer_addr_start(buffer_info);
	buffer_len = (size_t)LL_Buffer_len(buffer_info);
	scale_addr = buffer_info->scale;
	offset_addr = buffer_info->offset;

	DebugConsole_Printf(
		"[AI] %s info: name=%s addr=%p len=%lu type=%s nbits=%u ndims=%u Qm=%u Qn=%u Qunsigned=%u epoch=%u batch=%u shape=[%lu,%lu,%lu,%lu]\r\n",
		label,
		(buffer_info->name != NULL) ? buffer_info->name : "(unnamed)",
		buffer_addr, (unsigned long)buffer_len,
		AppAI_BufferTypeName(buffer_info), (unsigned int)buffer_info->nbits,
		(unsigned int)buffer_info->ndims, (unsigned int)buffer_info->Qm,
		(unsigned int)buffer_info->Qn, (unsigned int)buffer_info->Qunsigned,
		(unsigned int)buffer_info->epoch, (unsigned int)buffer_info->batch,
		(unsigned long)((buffer_info->shape != NULL) && (buffer_info->ndims > 0U) ? buffer_info->shape[0] : 0U),
		(unsigned long)((buffer_info->shape != NULL) && (buffer_info->ndims > 1U) ? buffer_info->shape[1] : 0U),
		(unsigned long)((buffer_info->shape != NULL) && (buffer_info->ndims > 2U) ? buffer_info->shape[2] : 0U),
		(unsigned long)((buffer_info->shape != NULL) && (buffer_info->ndims > 3U) ? buffer_info->shape[3] : 0U));

	if ((scale_addr != NULL) && (offset_addr != NULL))
	{
		(void)memcpy(&scale_value, scale_addr, sizeof(scale_value));
		offset_value = *(const int16_t *)offset_addr;

		DebugConsole_Printf("[AI] %s qparams: ", label);
		AppAI_LogFloatApprox("scale=", scale_value);
		DebugConsole_Printf(" offset=%d\r\n", (int)offset_value);
	}

	if (buffer_addr != NULL)
	{
		AppAI_LogRawBufferSignature(label, (const uint8_t *)buffer_addr,
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
								   const LL_Buffer_InfoTypeDef *buffer_info)
{
	const uint8_t *buffer_bytes = NULL;
	size_t buffer_len = 0U;
	uint8_t first_bytes[8U] = {0U};
	uint8_t mid_bytes[8U] = {0U};
	uint8_t last_bytes[8U] = {0U};
	uint32_t hash = 2166136261UL;
	size_t preview_len = 0U;
	size_t mid_offset = 0U;
	size_t last_offset = 0U;

	if ((label == NULL) || (buffer_info == NULL))
	{
		return;
	}

	buffer_bytes = (const uint8_t *)LL_Buffer_addr_start(buffer_info);
	buffer_len = (size_t)LL_Buffer_len(buffer_info);
	if ((buffer_bytes == NULL) || (buffer_len == 0U))
	{
		DebugConsole_Printf("[AI] %s probe skipped: empty buffer.\r\n", label);
		return;
	}

	preview_len = (buffer_len < sizeof(first_bytes)) ? buffer_len
													 : sizeof(first_bytes);
	mid_offset = (buffer_len > preview_len) ? (buffer_len / 2U) : 0U;
	if ((mid_offset + preview_len) > buffer_len)
	{
		mid_offset = buffer_len - preview_len;
	}
	last_offset = (buffer_len > preview_len) ? (buffer_len - preview_len) : 0U;

	for (size_t index = 0U; index < preview_len; ++index)
	{
		first_bytes[index] = buffer_bytes[index];
		mid_bytes[index] = buffer_bytes[mid_offset + index];
		last_bytes[index] = buffer_bytes[last_offset + index];
	}

	for (size_t index = 0U; index < buffer_len; ++index)
	{
		hash ^= buffer_bytes[index];
		hash *= 16777619UL;
	}

	DebugConsole_Printf(
		"[AI] %s probe: name=%s addr=%p len=%lu hash=0x%08lX first8=[%02X %02X %02X %02X %02X %02X %02X %02X] mid8=[%02X %02X %02X %02X %02X %02X %02X %02X] last8=[%02X %02X %02X %02X %02X %02X %02X %02X]\r\n",
		label,
		(buffer_info->name != NULL) ? buffer_info->name : "(unnamed)",
		(const void *)buffer_bytes, (unsigned long)buffer_len,
		(unsigned long)hash,
		(unsigned int)first_bytes[0], (unsigned int)first_bytes[1],
		(unsigned int)first_bytes[2], (unsigned int)first_bytes[3],
		(unsigned int)first_bytes[4], (unsigned int)first_bytes[5],
		(unsigned int)first_bytes[6], (unsigned int)first_bytes[7],
		(unsigned int)mid_bytes[0], (unsigned int)mid_bytes[1],
		(unsigned int)mid_bytes[2], (unsigned int)mid_bytes[3],
		(unsigned int)mid_bytes[4], (unsigned int)mid_bytes[5],
		(unsigned int)mid_bytes[6], (unsigned int)mid_bytes[7],
		(unsigned int)last_bytes[0], (unsigned int)last_bytes[1],
		(unsigned int)last_bytes[2], (unsigned int)last_bytes[3],
		(unsigned int)last_bytes[4], (unsigned int)last_bytes[5],
		(unsigned int)last_bytes[6], (unsigned int)last_bytes[7]);
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
												 size_t *crop_height)
{
	const size_t frame_stride_bytes = frame_width_pixels * 2U;
	const size_t min_crop_width = frame_width_pixels / 4U;
	const size_t min_crop_height = frame_height_pixels / 4U;
	const size_t training_crop_width = (size_t)(((float)frame_width_pixels * (APP_AI_TRAINING_CROP_X_MAX_RATIO - APP_AI_TRAINING_CROP_X_MIN_RATIO)) + 0.5f);
	const size_t training_crop_height = (size_t)(((float)frame_height_pixels * (APP_AI_TRAINING_CROP_Y_MAX_RATIO - APP_AI_TRAINING_CROP_Y_MIN_RATIO)) + 0.5f);
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

	if ((frame_bytes == NULL) || (crop_x_min == NULL) || (crop_y_min == NULL) || (crop_width == NULL) || (crop_height == NULL))
	{
		return false;
	}

	if (frame_size < (frame_stride_bytes * frame_height_pixels))
	{
		return false;
	}

	for (size_t y = APP_AI_GAUGE_CROP_BORDER_PIXELS;
		 y < (frame_height_pixels - APP_AI_GAUGE_CROP_BORDER_PIXELS); ++y)
	{
		const size_t row_offset = y * frame_stride_bytes;

		for (size_t x = APP_AI_GAUGE_CROP_BORDER_PIXELS;
			 x < (frame_width_pixels - APP_AI_GAUGE_CROP_BORDER_PIXELS); ++x)
		{
			const size_t pair_offset = row_offset + ((x & ~1U) * 2U);
			const size_t y_offset = pair_offset + (((x & 1U) != 0U) ? 2U : 0U);
			const uint8_t luma = frame_bytes[y_offset];

			if (luma < APP_AI_GAUGE_BRIGHT_THRESHOLD)
			{
				continue;
			}

			bright_count++;
			bright_sum_x += (uint64_t)x;
			bright_sum_y += (uint64_t)y;

			if (x < bright_x_min)
			{
				bright_x_min = x;
			}
			if (y < bright_y_min)
			{
				bright_y_min = y;
			}
			if (x > bright_x_max)
			{
				bright_x_max = x;
			}
			if (y > bright_y_max)
			{
				bright_y_max = y;
			}
		}
	}

	if (bright_count == 0U)
	{
		return false;
	}

	bright_center_x = (size_t)(bright_sum_x / (uint64_t)bright_count);
	bright_center_y = (size_t)(bright_sum_y / (uint64_t)bright_count);
	bbox_width = (bright_x_max - bright_x_min) + 1U;
	bbox_height = (bright_y_max - bright_y_min) + 1U;
	if ((bbox_width == 0U) || (bbox_height == 0U))
	{
		return false;
	}

	/* Anchor a slightly tighter crop on a shifted bright centroid instead of
	 * using the full bright bbox, which was pulling in too much background. */
	if (training_crop_width == 0U)
	{
		return false;
	}
	if (training_crop_height == 0U)
	{
		return false;
	}

	const size_t target_crop_width = (size_t)((((float)training_crop_width * (float)APP_AI_GAUGE_CROP_WIDTH_SCALE_NUMERATOR) / (float)APP_AI_GAUGE_CROP_WIDTH_SCALE_DENOMINATOR) + 0.5f);
	const size_t target_crop_height = (size_t)((((float)training_crop_height * (float)APP_AI_GAUGE_CROP_HEIGHT_SCALE_NUMERATOR) / (float)APP_AI_GAUGE_CROP_HEIGHT_SCALE_DENOMINATOR) + 0.5f);
	const size_t crop_width_pixels =
		(target_crop_width > 0U) ? target_crop_width : 1U;
	const size_t crop_height_pixels =
		(target_crop_height > 0U) ? target_crop_height : 1U;
	{
		const size_t y_bias_pixels = (size_t)(((float)crop_height_pixels * APP_AI_GAUGE_CROP_CENTER_Y_BIAS_RATIO) + 0.5f);
		const size_t bounded_y_bias =
			(y_bias_pixels < APP_AI_GAUGE_CROP_CENTER_Y_BIAS_MIN_PIXELS) ? APP_AI_GAUGE_CROP_CENTER_Y_BIAS_MIN_PIXELS
																		 : ((y_bias_pixels > APP_AI_GAUGE_CROP_CENTER_Y_BIAS_MAX_PIXELS) ? APP_AI_GAUGE_CROP_CENTER_Y_BIAS_MAX_PIXELS
																																		 : y_bias_pixels);

		biased_center_x = (bright_center_x > APP_AI_GAUGE_CROP_CENTER_X_BIAS_PIXELS) ? (bright_center_x - APP_AI_GAUGE_CROP_CENTER_X_BIAS_PIXELS) : 0U;
		biased_center_y = (bright_center_y > bounded_y_bias) ? (bright_center_y - bounded_y_bias) : 0U;
	}

	left = (biased_center_x > (crop_width_pixels / 2U)) ? (biased_center_x - (crop_width_pixels / 2U)) : 0U;
	top = (biased_center_y > (crop_height_pixels / 2U)) ? (biased_center_y - (crop_height_pixels / 2U)) : 0U;
	right = left + crop_width_pixels;
	bottom = top + crop_height_pixels;
	if (right > frame_width_pixels)
	{
		right = frame_width_pixels;
		left = (right > crop_width_pixels) ? (right - crop_width_pixels)
										   : 0U;
	}
	if (bottom > frame_height_pixels)
	{
		bottom = frame_height_pixels;
		top = (bottom > crop_height_pixels) ? (bottom - crop_height_pixels)
											: 0U;
	}

	if ((right <= left) || (bottom <= top))
	{
		return false;
	}

	*crop_x_min = left;
	*crop_y_min = top;
	*crop_width = right - left;
	*crop_height = bottom - top;

	if ((*crop_width < min_crop_width) || (*crop_height < min_crop_height))
	{
		return false;
	}

	return true;
}

