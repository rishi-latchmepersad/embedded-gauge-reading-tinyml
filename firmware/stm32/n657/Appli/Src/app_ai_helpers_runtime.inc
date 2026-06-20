/* This file is included by app_ai_helpers.inc. */

static bool AppAI_WaitForFileXMediaReady(uint32_t timeout_ms)
{
	const ULONG timeout_ticks = (ULONG)((timeout_ms + 9U) / 10U);
	const ULONG start_tick = tx_time_get();

	while (!AppFileX_IsMediaReady())
	{
		if ((tx_time_get() - start_tick) >= timeout_ticks)
		{
			return false;
		}

		tx_thread_sleep(1U);
	}

	return true;
}

/**
 * @brief Preprocess one YUV422 frame into the tip-focus int8 tensor layout.
 */
static bool __attribute__((noinline)) AppAI_PreprocessYuv422FrameToInt8Input(
	const uint8_t *frame_bytes, size_t frame_size, uint8_t *input_ptr,
	size_t input_len_bytes, const LL_Buffer_InfoTypeDef *input_info)
{
	const size_t source_width = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const size_t source_height = (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	/* Derive output dimensions from the model's input tensor memory shape
	 * (NHWC: {batch, height, width, channels}).  Each model stage may have a
	 * different input resolution (for example, the live tip-focus path is
	 * 224x224 while older legacy stages may differ). */
	const size_t output_width =
		(input_info->mem_shape != NULL && input_info->mem_ndims >= 3U)
			? (size_t)input_info->mem_shape[2U]
			: (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const size_t output_height =
		(input_info->mem_shape != NULL && input_info->mem_ndims >= 3U)
			? (size_t)input_info->mem_shape[1U]
			: (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	const size_t required_bytes = output_width * output_height * 3U;
	size_t crop_x_min = 0U;
	size_t crop_y_min = 0U;
	size_t crop_width = source_width;
	size_t crop_height = source_height;
	bool crop_found = false;
	const char *crop_label = "fixed";
	float scale_value = 1.0f / 255.0f;
	int16_t zero_point = 0;
	int32_t q_min = -128;
	int32_t q_max = 127;
	int32_t q_zero = 0;

	if ((frame_bytes == NULL) || (input_ptr == NULL) || (input_info == NULL))
	{
		return false;
	}
	if ((frame_size < (size_t)APP_AI_CAPTURE_FRAME_BYTES) ||
		(input_len_bytes < required_bytes))
	{
		return false;
	}

	if (input_info->scale != NULL)
	{
		(void)memcpy(&scale_value, input_info->scale, sizeof(scale_value));
	}
	if (input_info->offset != NULL)
	{
		(void)memcpy(&zero_point, input_info->offset, sizeof(zero_point));
	}
	if (scale_value <= 0.0f)
	{
		scale_value = 1.0f / 255.0f;
	}
	if (input_info->Qunsigned != 0U)
	{
		q_min = 0;
		q_max = 255;
	}
	q_zero = (int32_t)zero_point;
	if (q_zero < q_min)
	{
		q_zero = q_min;
	}
	if (q_zero > q_max)
	{
		q_zero = q_max;
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
		crop_found = AppAI_EstimateGaugeCropBoxFromYuv422(frame_bytes, frame_size,
														  source_width, source_height, &crop_x_min, &crop_y_min,
														  &crop_width, &crop_height);
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
		crop_width = (size_t)((float)source_width * (APP_AI_TRAINING_CROP_X_MAX_RATIO - APP_AI_TRAINING_CROP_X_MIN_RATIO));
		crop_height = (size_t)((float)source_height * (APP_AI_TRAINING_CROP_Y_MAX_RATIO - APP_AI_TRAINING_CROP_Y_MIN_RATIO));
		if (crop_width == 0U)
		{
			crop_width = 1U;
		}
		if (crop_height == 0U)
		{
			crop_height = 1U;
		}
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
					   (unsigned long)crop_x_min, (unsigned long)crop_y_min,
					   (unsigned long)crop_width, (unsigned long)crop_height);
	(void)DebugConsole_WriteString("[AI] Preprocess diagnostics OK.\r\n");
	(void)DebugConsole_WriteString("[AI] Preprocess zero-fill skipped.\r\n");
	(void)DebugConsole_WriteString("[AI] Preprocess resize start.\r\n");

	{
		const float resize_scale =
			fminf((float)output_width / (float)crop_width,
				  (float)output_height / (float)crop_height);
		size_t resized_width = (size_t)(((float)crop_width * resize_scale) + 0.5f);
		size_t resized_height = (size_t)(((float)crop_height * resize_scale) + 0.5f);
		size_t resize_pad_x = 0U;
		size_t resize_pad_y = 0U;

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
		resize_pad_x = (output_width - resized_width) / 2U;
		resize_pad_y = (output_height - resized_height) / 2U;
		app_ai_scalar_preprocess_last_row = (size_t)SIZE_MAX;

		if (input_info->Qunsigned != 0U)
		{
			(void)memset(input_ptr, (uint8_t)q_zero, required_bytes);
		}
		else
		{
			const int8_t q_zero_s8 = (int8_t)q_zero;
			for (size_t i = 0U; i < required_bytes; ++i)
			{
				((int8_t *)input_ptr)[i] = q_zero_s8;
			}
		}

		(void)DebugConsole_WriteString("[AI] Preprocess row loop enter.\r\n");

		for (size_t out_y = 0U; out_y < output_height; ++out_y)
		{
			app_ai_scalar_preprocess_last_row = out_y;
			if ((out_y < resize_pad_y) || (out_y >= (resize_pad_y + resized_height)))
			{
				continue;
			}

			for (size_t out_x = 0U; out_x < output_width; ++out_x)
			{
				float out_r = 0.0f;
				float out_g = 0.0f;
				float out_b = 0.0f;
				const size_t base = ((out_y * output_width) + out_x) * 3U;
				if ((out_x < resize_pad_x) || (out_x >= (resize_pad_x + resized_width)))
				{
					continue;
				}
				{
					const float resized_x = (float)(out_x - resize_pad_x);
					const float resized_y = (float)(out_y - resize_pad_y);
					float crop_x = ((resized_x + 0.5f) / resize_scale) - 0.5f;
					float crop_y = ((resized_y + 0.5f) / resize_scale) - 0.5f;

					if (crop_x < 0.0f)
					{
						crop_x = 0.0f;
					}
					else if (crop_x > (float)(crop_width - 1U))
					{
						crop_x = (float)(crop_width - 1U);
					}

					if (crop_y < 0.0f)
					{
						crop_y = 0.0f;
					}
					else if (crop_y > (float)(crop_height - 1U))
					{
						crop_y = (float)(crop_height - 1U);
					}

					AppAI_ReadRgbFromYuv422Bilinear(
						frame_bytes, frame_size, source_width, source_height,
						(float)crop_x_min + crop_x, (float)crop_y_min + crop_y,
						&out_r, &out_g, &out_b);
				}
				{
					int32_t q_r = (int32_t)lroundf(out_r / scale_value) + (int32_t)zero_point;
					int32_t q_g = (int32_t)lroundf(out_g / scale_value) + (int32_t)zero_point;
					int32_t q_b = (int32_t)lroundf(out_b / scale_value) + (int32_t)zero_point;
					if (q_r < q_min)
						q_r = q_min;
					if (q_r > q_max)
						q_r = q_max;
					if (q_g < q_min)
						q_g = q_min;
					if (q_g > q_max)
						q_g = q_max;
					if (q_b < q_min)
						q_b = q_min;
					if (q_b > q_max)
						q_b = q_max;

					if (input_info->Qunsigned != 0U)
					{
						input_ptr[base + 0U] = (uint8_t)q_r;
						input_ptr[base + 1U] = (uint8_t)q_g;
						input_ptr[base + 2U] = (uint8_t)q_b;
					}
					else
					{
						((int8_t *)input_ptr)[base + 0U] = (int8_t)q_r;
						((int8_t *)input_ptr)[base + 1U] = (int8_t)q_g;
						((int8_t *)input_ptr)[base + 2U] = (int8_t)q_b;
					}
				}
			}
		}
	}

	(void)DebugConsole_WriteString("[AI] Preprocess resize OK.\r\n");
	return true;
}


