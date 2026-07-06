/* This file is included by app_ai_helpers.inc. */


	if (use_fixed_training_crop)
	{
		/* Keep the scalar reader on the same stable crop that the offline
		 * training/evaluation path uses when the rectifier output is implausible.
		 * That avoids 1x1/tiny crops and avoids treating full-frame predictions
		 * as if they were useful dial boxes. */
		crop_out->x_min = (size_t)((float)source_width * APP_AI_TRAINING_CROP_X_MIN_RATIO);
		crop_out->y_min = (size_t)((float)source_height * APP_AI_TRAINING_CROP_Y_MIN_RATIO);
		crop_out->width = (size_t)((float)source_width * (APP_AI_TRAINING_CROP_X_MAX_RATIO - APP_AI_TRAINING_CROP_X_MIN_RATIO));
		crop_out->height = (size_t)((float)source_height * (APP_AI_TRAINING_CROP_Y_MAX_RATIO - APP_AI_TRAINING_CROP_Y_MIN_RATIO));
		if (crop_out->width == 0U)
		{
			crop_out->width = 1U;
		}
		if (crop_out->height == 0U)
		{
			crop_out->height = 1U;
		}
#if APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
		{
			char fb_log[128];

			(void)DebugConsole_Snprintf(fb_log, sizeof(fb_log),
						   "[AI] Rectifier fallback -> fixed training crop: x=%lu y=%lu w=%lu h=%lu\r\n",
						   (unsigned long)crop_out->x_min,
						   (unsigned long)crop_out->y_min,
						   (unsigned long)crop_out->width,
						   (unsigned long)crop_out->height);
			(void)DebugConsole_WriteString(fb_log);
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
	crop_width_f = ((float)source_width) * (APP_AI_TRAINING_CROP_X_MAX_RATIO - APP_AI_TRAINING_CROP_X_MIN_RATIO);
	crop_height_f = ((float)source_height) * (APP_AI_TRAINING_CROP_Y_MAX_RATIO - APP_AI_TRAINING_CROP_Y_MIN_RATIO);
#else
	crop_width_f = ((float)source_width) * box_w * APP_AI_RECTIFIER_CROP_SCALE;
	crop_height_f = ((float)source_height) * box_h * APP_AI_RECTIFIER_CROP_SCALE;
#endif
	if (crop_width_f < 1.0f)
	{
		crop_width_f = 1.0f;
	}
	if (crop_height_f < 1.0f)
	{
		crop_height_f = 1.0f;
	}

#if APP_AI_RECTIFIER_FIXED_SCALE_CROP
	crop_x_min_f = ((float)training_center_x) + ((((float)source_width) * center_x - (float)training_center_x) * rectifier_center_blend_f) - (crop_width_f * 0.5f);
	crop_y_min_f = ((float)training_center_y) + ((((float)source_height) * center_y - (float)training_center_y) * rectifier_center_blend_f) - (crop_height_f * 0.5f);
#else
	crop_x_min_f = (((float)source_width) * center_x) - (crop_width_f * 0.5f);
	crop_y_min_f = (((float)source_height) * center_y) - (crop_height_f * 0.5f);
#endif
	if (crop_x_min_f < 0.0f)
	{
		crop_x_min_f = 0.0f;
	}
	if (crop_y_min_f < 0.0f)
	{
		crop_y_min_f = 0.0f;
	}
	if ((crop_x_min_f + crop_width_f) > (float)source_width)
	{
		crop_x_min_f = (float)source_width - crop_width_f;
	}
	if ((crop_y_min_f + crop_height_f) > (float)source_height)
	{
		crop_y_min_f = (float)source_height - crop_height_f;
	}
	if (crop_x_min_f < 0.0f)
	{
		crop_x_min_f = 0.0f;
	}
	if (crop_y_min_f < 0.0f)
	{
		crop_y_min_f = 0.0f;
	}

	crop_out->x_min = (size_t)(crop_x_min_f + 0.5f);
	crop_out->y_min = (size_t)(crop_y_min_f + 0.5f);
	crop_out->width = (size_t)(crop_width_f + 0.5f);
	crop_out->height = (size_t)(crop_height_f + 0.5f);

	if (crop_out->width == 0U)
	{
		crop_out->width = 1U;
	}
	if (crop_out->height == 0U)
	{
		crop_out->height = 1U;
	}
	if (crop_out->x_min >= source_width)
	{
		crop_out->x_min = source_width - 1U;
	}
	if (crop_out->y_min >= source_height)
	{
		crop_out->y_min = source_height - 1U;
	}
	if ((crop_out->x_min + crop_out->width) > source_width)
	{
		crop_out->width = source_width - crop_out->x_min;
	}
	if ((crop_out->y_min + crop_out->height) > source_height)
	{
		crop_out->height = source_height - crop_out->y_min;
	}

	return true;
}

#if !APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
void AppAI_LogXspi2PrefixBytes(const char *label,
									  const uint8_t *bytes)
{
	(void)label;
	(void)bytes;
}

void AppAI_LogFrameSignature(const uint8_t *frame_bytes,
									size_t frame_size)
{
	(void)frame_bytes;
	(void)frame_size;
}

void AppAI_LogInputSignature(const float *input_buffer,
									size_t input_float_count)
{
	(void)input_buffer;
	(void)input_float_count;
}

void AppAI_LogInputTensorWindow(const float *input_buffer,
									   size_t input_float_count)
{
	(void)input_buffer;
	(void)input_float_count;
}

void AppAI_LogInputProbeSummary(const float *input_buffer,
									   size_t input_float_count)
{
	(void)input_buffer;
	(void)input_float_count;
}

void AppAI_LogTensorRowSamples(const char *label,
									  const float *input_buffer, size_t tensor_width, size_t y,
									  size_t x_min, size_t x_max)
{
	(void)label;
	(void)input_buffer;
	(void)tensor_width;
	(void)y;
	(void)x_min;
	(void)x_max;
}

void AppAI_LogSourcePatch(const char *label, const uint8_t *frame_bytes,
								 size_t frame_width_pixels, size_t center_x, size_t center_y,
								 size_t radius_pixels)
{
	(void)label;
	(void)frame_bytes;
	(void)frame_width_pixels;
	(void)center_x;
	(void)center_y;
	(void)radius_pixels;
}

void AppAI_LogTensorPatch(const char *label, const float *input_buffer,
								 size_t tensor_width, size_t center_x, size_t center_y,
								 size_t radius_pixels)
{
	(void)label;
	(void)input_buffer;
	(void)tensor_width;
	(void)center_x;
	(void)center_y;
	(void)radius_pixels;
}

void AppAI_LogSourceCropWindow(const uint8_t *frame_bytes,
									  size_t frame_size, size_t frame_width_pixels, size_t frame_height_pixels,
									  size_t crop_x_min, size_t crop_y_min, size_t crop_width,
									  size_t crop_height)
{
	(void)frame_bytes;
	(void)frame_size;
	(void)frame_width_pixels;
	(void)frame_height_pixels;
	(void)crop_x_min;
	(void)crop_y_min;
	(void)crop_width;
	(void)crop_height;
}

void AppAI_LogInt8BufferSignature(const char *label,
										 const int8_t *buffer_ptr, size_t buffer_len_bytes)
{
	(void)label;
	(void)buffer_ptr;
	(void)buffer_len_bytes;
}

void AppAI_LogRawBufferSignature(const char *label,
										const uint8_t *buffer_ptr, size_t buffer_len_bytes)
{
	(void)label;
	(void)buffer_ptr;
	(void)buffer_len_bytes;
}

const char *AppAI_BufferTypeName(const LL_Buffer_InfoTypeDef *buffer_info)
{
	(void)buffer_info;
	return "OTHER";
}

void AppAI_LogBufferInfoAndSignature(const char *label,
											const LL_Buffer_InfoTypeDef *buffer_info)
{
	(void)label;
	(void)buffer_info;
}
#endif /* !APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS */

void AppAI_LogRectifierResult(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	const AppAI_RectifierBox *rectifier_box)
{
	if ((output_buffer_info == NULL) || (rectifier_box == NULL))
	{
		DebugConsole_Printf("[AI] Rectifier output missing.\r\n");
		return;
	}

	DebugConsole_Printf("[AI] Rectifier output: name=%s addr=%p len=%lu\r\n",
						(output_buffer_info->name != NULL) ? output_buffer_info->name
														   : "(unnamed)",
						LL_Buffer_addr_start(output_buffer_info),
						(unsigned long)LL_Buffer_len(output_buffer_info));
#if APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
	DebugConsole_Printf(
		"[AI] Rectifier box(clamped): cx=%.6f cy=%.6f w=%.6f h=%.6f\r\n",
		rectifier_box->center_x, rectifier_box->center_y,
		rectifier_box->box_w, rectifier_box->box_h);
#endif
}

void AppAI_LogObbResult(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	const AppAI_ObbBox *obb_box)
{
	float theta_deg = 0.0f;

	if ((output_buffer_info == NULL) || (obb_box == NULL))
	{
		DebugConsole_Printf("[AI] OBB output missing.\r\n");
		return;
	}

	theta_deg = obb_box->angle_rad * 57.29577951308232f;
	DebugConsole_Printf("[AI] OBB output: name=%s addr=%p len=%lu\r\n",
						(output_buffer_info->name != NULL) ? output_buffer_info->name
														   : "(unnamed)",
						LL_Buffer_addr_start(output_buffer_info),
						(unsigned long)LL_Buffer_len(output_buffer_info));
#if APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
	DebugConsole_Printf(
		"[AI] OBB box(clamped): cx=%.6f cy=%.6f w=%.6f h=%.6f angle=%.2fdeg conf=%.4f\r\n",
		obb_box->center_x, obb_box->center_y, obb_box->box_w,
		obb_box->box_h, theta_deg, obb_box->confidence);
#endif
}

float AppAI_ObbDequantize(int8_t q_value)
{
	return ((float)q_value - (float)APP_AI_OBB_OUTPUT_ZERO_POINT)
		* APP_AI_OBB_OUTPUT_SCALE;
}

bool AppAI_DecodeObbCropBox(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	AppAI_SourceCrop *crop_out,
	AppAI_ObbBox *obb_box_out)
{
	const LL_Buffer_InfoTypeDef *obb_buf = NULL;
	const LL_Buffer_InfoTypeDef *ctr_buf = NULL;
	const int8_t *obb_ptr = NULL;
	const int8_t *ctr_ptr = NULL;
	const float source_width_f = (float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const float source_height_f = (float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	const float input_size_f = (float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const size_t training_crop_width = (size_t)(((float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS * (APP_AI_TRAINING_CROP_X_MAX_RATIO - APP_AI_TRAINING_CROP_X_MIN_RATIO)) + 0.5f);
	const size_t training_crop_height = (size_t)(((float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS * (APP_AI_TRAINING_CROP_Y_MAX_RATIO - APP_AI_TRAINING_CROP_Y_MIN_RATIO)) + 0.5f);
	size_t training_center_x = 0U;
	size_t training_center_y = 0U;
	float center_x = 0.0f;
	float center_y = 0.0f;
	float box_w = 0.0f;
	float box_h = 0.0f;
	float angle_rad = 0.0f;
	float confidence = 0.0f;
	float cos_2theta = 0.0f;
	float sin_2theta = 0.0f;
	float canvas_center_x = 0.0f;
	float canvas_center_y = 0.0f;
	float half_width = 0.0f;
	float half_height = 0.0f;
	float crop_x_min_f = 0.0f;
	float crop_y_min_f = 0.0f;
	float crop_x_max_f = 0.0f;
	float crop_y_max_f = 0.0f;
	float decoded[APP_AI_OBB_OUTPUT_CHANNELS] = {0.0f};

	if ((output_buffer_info == NULL) || (crop_out == NULL))
	{
		return false;
	}

	/* Joint model has two outputs:
	 *   [0] Quantize_318_out_0 — gauge centre (2 bytes, offset 48)
	 *   [1] Quantize_315_out_0 — OBB box (6 bytes, offset 0)
	 * Extract both so we can dequantize all 8 channels. */
	ctr_buf = output_buffer_info;       /* index 0: centre */
	obb_buf = output_buffer_info + 1U;  /* index 1: OBB box */

	obb_ptr = (const int8_t *)LL_Buffer_addr_start(obb_buf);
	ctr_ptr = (const int8_t *)LL_Buffer_addr_start(ctr_buf);
	if ((obb_ptr == NULL) || (ctr_ptr == NULL))
	{
		return false;
	}
	if ((LL_Buffer_len(obb_buf) < 6U) || (LL_Buffer_len(ctr_buf) < 2U))
	{
		return false;
	}

	/* Channels 0-5: OBB (cx, cy, w, h, cos2θ, sin2θ) from the 6-byte OBB buffer */
	for (size_t i = 0U; i < 6U; ++i)
	{
		decoded[i] = AppAI_ObbDequantize(obb_ptr[i]);
	}
	/* Channels 6-7: gauge centre (cx_ctr, cy_ctr) from the 2-byte centre buffer */
	decoded[6U] = AppAI_ObbDequantize(ctr_ptr[0]);
	decoded[7U] = AppAI_ObbDequantize(ctr_ptr[1]);

	center_x = AppAI_ClampNormalizedFloat(decoded[0U]);
	center_y = AppAI_ClampNormalizedFloat(decoded[1U]);
	box_w = AppAI_ClampNormalizedFloat(decoded[2U]);
	box_h = AppAI_ClampNormalizedFloat(decoded[3U]);
	cos_2theta = decoded[4U];
	sin_2theta = decoded[5U];
	if (!AppAI_IsFiniteFloat(center_x) || !AppAI_IsFiniteFloat(center_y) ||
		!AppAI_IsFiniteFloat(box_w) || !AppAI_IsFiniteFloat(box_h) ||
		!AppAI_IsFiniteFloat(cos_2theta) || !AppAI_IsFiniteFloat(sin_2theta))
	{
		return false;
	}

	angle_rad = 0.5f * atan2f(sin_2theta, cos_2theta);
	confidence = sqrtf((cos_2theta * cos_2theta) + (sin_2theta * sin_2theta));
	if (!AppAI_IsFiniteFloat(angle_rad))
	{
		angle_rad = 0.0f;
	}
	if (!AppAI_IsFiniteFloat(confidence))
	{
		confidence = 0.0f;
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

	if (obb_box_out != NULL)
	{
		obb_box_out->center_x = center_x;
		obb_box_out->center_y = center_y;
		obb_box_out->box_w = box_w;
		obb_box_out->box_h = box_h;
		obb_box_out->angle_rad = angle_rad;
		obb_box_out->confidence = confidence;
		/* Channels 6,7: gauge center (needle pivot) from joint OBB+Center model.
		 * Already dequantized into decoded[6..7] above. */
		if (APP_AI_OBB_OUTPUT_CHANNELS >= 8U)
		{
			const float gauge_cx = AppAI_ClampNormalizedFloat(decoded[6U]);
			const float gauge_cy = AppAI_ClampNormalizedFloat(decoded[7U]);
			if (AppAI_IsFiniteFloat(gauge_cx) && AppAI_IsFiniteFloat(gauge_cy))
			{
				obb_box_out->gauge_center_x = gauge_cx;
				obb_box_out->gauge_center_y = gauge_cy;
			}
			else
			{
				obb_box_out->gauge_center_x = -1.0f;  /* invalid */
				obb_box_out->gauge_center_y = -1.0f;
			}
		}
		else
		{
			obb_box_out->gauge_center_x = -1.0f;
			obb_box_out->gauge_center_y = -1.0f;
		}
	}

#if APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
	DebugConsole_Printf(
		"[AI] OBB decoded: cx=%.6f cy=%.6f w=%.6f h=%.6f angle=%.6f conf=%.6f\r\n",
		center_x, center_y, box_w, box_h, angle_rad, confidence);
#endif

	canvas_center_x = center_x * input_size_f;
	canvas_center_y = center_y * input_size_f;
	half_width = 0.5f * box_w * input_size_f * APP_AI_OBB_CROP_SCALE;
	half_height = 0.5f * box_h * input_size_f * APP_AI_OBB_CROP_SCALE;

	{
		const float cos_theta = cosf(angle_rad);
		const float sin_theta = sinf(angle_rad);
		const float corner_offsets[4][2] = {
			{-half_width, -half_height},
			{half_width, -half_height},
			{half_width, half_height},
			{-half_width, half_height},
		};
		float source_x_values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
		float source_y_values[4] = {0.0f, 0.0f, 0.0f, 0.0f};

		for (size_t corner_index = 0U; corner_index < 4U; ++corner_index)
		{
			const float dx = corner_offsets[corner_index][0];
			const float dy = corner_offsets[corner_index][1];

			source_x_values[corner_index] = canvas_center_x + ((dx * cos_theta) - (dy * sin_theta));
			source_y_values[corner_index] = canvas_center_y + ((dx * sin_theta) + (dy * cos_theta));
		}

		crop_x_min_f = source_x_values[0U];
		crop_y_min_f = source_y_values[0U];
		crop_x_max_f = source_x_values[0U];
		crop_y_max_f = source_y_values[0U];
		for (size_t corner_index = 1U; corner_index < 4U; ++corner_index)
		{
			if (source_x_values[corner_index] < crop_x_min_f)
			{
				crop_x_min_f = source_x_values[corner_index];
			}
			if (source_y_values[corner_index] < crop_y_min_f)
			{
				crop_y_min_f = source_y_values[corner_index];
			}
			if (source_x_values[corner_index] > crop_x_max_f)
			{
				crop_x_max_f = source_x_values[corner_index];
			}
			if (source_y_values[corner_index] > crop_y_max_f)
			{
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
				? APP_AI_OBB_MIN_CROP_SIZE_PIXELS
				: crop_width_f;
		const float target_height_f =
			(crop_height_f < APP_AI_OBB_MIN_CROP_SIZE_PIXELS)
				? APP_AI_OBB_MIN_CROP_SIZE_PIXELS
				: crop_height_f;

		crop_x_min_f = crop_center_x_f - (0.5f * target_width_f);
		crop_y_min_f = crop_center_y_f - (0.5f * target_height_f);
		crop_x_max_f = crop_x_min_f + target_width_f;
		crop_y_max_f = crop_y_min_f + target_height_f;
	}

	if (crop_x_min_f < 0.0f)
	{
		crop_x_max_f -= crop_x_min_f;
		crop_x_min_f = 0.0f;
	}
	if (crop_y_min_f < 0.0f)
	{
		crop_y_max_f -= crop_y_min_f;
		crop_y_min_f = 0.0f;
	}
	if (crop_x_max_f > source_width_f)
	{
		const float shift = crop_x_max_f - source_width_f;
		crop_x_min_f = (crop_x_min_f > shift) ? (crop_x_min_f - shift) : 0.0f;
		crop_x_max_f = source_width_f;
	}
	if (crop_y_max_f > source_height_f)
	{
		const float shift = crop_y_max_f - source_height_f;
		crop_y_min_f = (crop_y_min_f > shift) ? (crop_y_min_f - shift) : 0.0f;
		crop_y_max_f = source_height_f;
	}
	if (crop_x_min_f < 0.0f)
	{
		crop_x_min_f = 0.0f;
	}
	if (crop_y_min_f < 0.0f)
	{
		crop_y_min_f = 0.0f;
	}
	if (crop_x_max_f > source_width_f)
	{
		crop_x_max_f = source_width_f;
	}
	if (crop_y_max_f > source_height_f)
	{
		crop_y_max_f = source_height_f;
	}
	if ((crop_x_max_f <= crop_x_min_f) || (crop_y_max_f <= crop_y_min_f))
	{
		return false;
	}

	crop_out->x_min = (size_t)floorf(crop_x_min_f);
	crop_out->y_min = (size_t)floorf(crop_y_min_f);
	crop_out->width = (size_t)ceilf(crop_x_max_f) - crop_out->x_min;
	crop_out->height = (size_t)ceilf(crop_y_max_f) - crop_out->y_min;
	if (crop_out->width == 0U)
	{
		crop_out->width = 1U;
	}
	if (crop_out->height == 0U)
	{
		crop_out->height = 1U;
	}
	if (crop_out->x_min >= (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS)
	{
		crop_out->x_min = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS - 1U;
	}
	if (crop_out->y_min >= (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS)
	{
		crop_out->y_min = (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS - 1U;
	}
	if ((crop_out->x_min + crop_out->width) > (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS)
	{
		crop_out->width = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS - crop_out->x_min;
	}
	if ((crop_out->y_min + crop_out->height) > (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS)
	{
		crop_out->height = (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS - crop_out->y_min;
	}
	if ((crop_out->width == 0U) || (crop_out->height == 0U))
	{
		return false;
	}

	/* Keep the OBB crop inside the stable training-size family; if the box is
	 * too tall or too wide, fall back to the fixed training crop geometry. */
	{
		const float crop_width_ratio =
			((training_crop_width > 0U) ? ((float)crop_out->width / (float)training_crop_width) : 0.0f);
		const float crop_height_ratio =
			((training_crop_height > 0U) ? ((float)crop_out->height / (float)training_crop_height) : 0.0f);
		const long crop_width_ratio_milli =
			(long)(crop_width_ratio * 1000.0f + 0.5f);
		const long crop_height_ratio_milli =
			(long)(crop_height_ratio * 1000.0f + 0.5f);

		if ((crop_width_ratio < APP_AI_OBB_TRAINING_CROP_MIN_RATIO) || (crop_width_ratio > APP_AI_OBB_TRAINING_CROP_MAX_RATIO) || (crop_height_ratio < APP_AI_OBB_TRAINING_CROP_MIN_RATIO) || (crop_height_ratio > APP_AI_OBB_TRAINING_CROP_MAX_RATIO))
		{
			/* When the OBB crop is outside the training window, fall back to the
			 * exact fixed training crop geometry. */
			const AppGaugeGeometry_Crop_t fixed_crop = AppGaugeGeometry_TrainingCrop(
				(size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS,
				(size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS);

			DebugConsole_Printf(
				"[AI] OBB crop outside training window: crop=%lux%lu train=%lux%lu ratio=%ld/%ld -> fixed training crop fallback.\r\n",
				(unsigned long)crop_out->width,
				(unsigned long)crop_out->height,
				(unsigned long)training_crop_width,
				(unsigned long)training_crop_height,
				crop_width_ratio_milli, crop_height_ratio_milli);

			crop_out->x_min = fixed_crop.x_min;
			crop_out->y_min = fixed_crop.y_min;
			crop_out->width = fixed_crop.width;
			crop_out->height = fixed_crop.height;
		}
	}

	return true;
}

#if APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE
void AppAI_LogSourceCropBoxResult(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	const AppAI_SourceCropBox *source_crop_box)
{
	if ((output_buffer_info == NULL) || (source_crop_box == NULL))
	{
		return;
	}
	DebugConsole_Printf(
		"[AI] Source-crop-box raw: x_min=%.6f y_min=%.6f x_max=%.6f y_max=%.6f\r\n",
		source_crop_box->x_min, source_crop_box->y_min,
		source_crop_box->x_max, source_crop_box->y_max);
}
bool __attribute__((noinline)) AppAI_DecodeSourceCropBox(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	AppAI_SourceCrop *crop_out,
	AppAI_SourceCropBox *box_out)
{
	const float *output_ptr = NULL;
	const float source_width_f = (float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const float source_height_f = (float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	float x_min = 0.0f;
	float y_min = 0.0f;
	float x_max = 0.0f;
	float y_max = 0.0f;
	float crop_x_min_f = 0.0f;
	float crop_y_min_f = 0.0f;
	float crop_x_max_f = 0.0f;
	float crop_y_max_f = 0.0f;
	float crop_width_f = 0.0f;
	float crop_height_f = 0.0f;
	if ((output_buffer_info == NULL) || (crop_out == NULL))
	{
		return false;
	}
	output_ptr = (const float *)LL_Buffer_addr_start(output_buffer_info);
	if ((output_ptr == NULL) || (LL_Buffer_len(output_buffer_info) < (sizeof(float) * 4U)))
	{
		return false;
	}
	x_min = AppAI_ClampNormalizedFloat(output_ptr[0]);
	y_min = AppAI_ClampNormalizedFloat(output_ptr[1]);
	x_max = AppAI_ClampNormalizedFloat(output_ptr[2]);
	y_max = AppAI_ClampNormalizedFloat(output_ptr[3]);
	/* Ensure min <= max after clamping. */
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
	if (box_out != NULL)
	{
		box_out->x_min = x_min;
		box_out->y_min = y_min;
		box_out->x_max = x_max;
		box_out->y_max = y_max;
	}
	/* Convert normalized [0,1] to source pixel coordinates. */
	crop_x_min_f = x_min * source_width_f;
	crop_y_min_f = y_min * source_height_f;
	crop_x_max_f = x_max * source_width_f;
	crop_y_max_f = y_max * source_height_f;
	/* Clamp to frame bounds. */
	if (crop_x_min_f < 0.0f)
	{
		crop_x_min_f = 0.0f;
	}
	if (crop_y_min_f < 0.0f)
	{
		crop_y_min_f = 0.0f;
	}
	if (crop_x_max_f > source_width_f)
	{
		crop_x_max_f = source_width_f;
	}
	if (crop_y_max_f > source_height_f)
	{
		crop_y_max_f = source_height_f;
	}
	crop_width_f = crop_x_max_f - crop_x_min_f;
	crop_height_f = crop_y_max_f - crop_y_min_f;
	if ((crop_width_f < APP_AI_SOURCE_CROP_BOX_MIN_CROP_SIZE_PIXELS) ||
	    (crop_height_f < APP_AI_SOURCE_CROP_BOX_MIN_CROP_SIZE_PIXELS))
	{
		return false;
	}
	crop_out->x_min = (size_t)floorf(crop_x_min_f);
	crop_out->y_min = (size_t)floorf(crop_y_min_f);
	crop_out->width = (size_t)ceilf(crop_x_max_f) - crop_out->x_min;
	crop_out->height = (size_t)ceilf(crop_y_max_f) - crop_out->y_min;
	if (crop_out->width == 0U)
	{
		crop_out->width = 1U;
	}
	if (crop_out->height == 0U)
	{
		crop_out->height = 1U;
	}
	if (crop_out->x_min >= (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS)
	{
		crop_out->x_min = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS - 1U;
	}
	if (crop_out->y_min >= (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS)
	{
		crop_out->y_min = (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS - 1U;
	}
	if ((crop_out->x_min + crop_out->width) > (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS)
	{
		crop_out->width = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS - crop_out->x_min;
	}
	if ((crop_out->y_min + crop_out->height) > (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS)
	{
		crop_out->height = (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS - crop_out->y_min;
	}
	return true;
}
#endif /* APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE */

#if !defined(APP_AI_DISABLE_LEGACY_TOPK_DECODE)
bool AppAI_DecodeScalarTopKExpectationFromOutput(
	const LL_Buffer_InfoTypeDef *output_info,
	const uint8_t *output_ptr,
	size_t output_len_bytes,
	float *decoded_value_out)
{
	size_t element_count = 0U;
	size_t topk = 0U;
	float top_logits[APP_AI_SCALAR_DECODE_TOPK];
	size_t top_indices[APP_AI_SCALAR_DECODE_TOPK];
	float scale_value = 1.0f;
	int16_t zero_point = 0;
	float max_logit = 0.0f;
	float weighted_fraction_sum = 0.0f;
	float weight_sum = 0.0f;
	float fraction = 0.0f;
	float decoded_value = 0.0f;
	size_t i = 0U;

	if ((output_info == NULL) || (output_ptr == NULL) || (decoded_value_out == NULL))
	{
		return false;
	}

	if (output_info->nbits <= 8U)
	{
		element_count = output_len_bytes;
		if (output_info->scale != NULL)
		{
			(void)memcpy(&scale_value, output_info->scale, sizeof(scale_value));
		}
		if (output_info->offset != NULL)
		{
			(void)memcpy(&zero_point, output_info->offset, sizeof(zero_point));
		}
	}
	else if (output_info->nbits == 32U)
	{
		element_count = output_len_bytes / sizeof(float);
	}
	else
	{
		return false;
	}

	if (element_count < 2U)
	{
		return false;
	}

	topk = (element_count < (size_t)APP_AI_SCALAR_DECODE_TOPK)
			   ? element_count
			   : (size_t)APP_AI_SCALAR_DECODE_TOPK;
	if (topk == 0U)
	{
		return false;
	}

	for (i = 0U; i < topk; ++i)
	{
		top_logits[i] = -INFINITY;
		top_indices[i] = 0U;
	}

	for (i = 0U; i < element_count; ++i)
	{
		float logit_value = 0.0f;
		size_t min_slot = 0U;
		size_t j = 0U;

		if (output_info->nbits <= 8U)
		{
			int32_t q_value = 0;
			if (output_info->Qunsigned != 0U)
			{
				q_value = (int32_t)output_ptr[i];
			}
			else
			{
				q_value = (int32_t)(*(const int8_t *)&output_ptr[i]);
			}
			logit_value =
				((float)q_value - (float)zero_point) * scale_value;
		}
		else
		{
			(void)memcpy(
				&logit_value,
				output_ptr + (i * sizeof(float)),
				sizeof(logit_value));
		}

		for (j = 1U; j < topk; ++j)
		{
			if (top_logits[j] < top_logits[min_slot])
			{
				min_slot = j;
			}
		}

		if (logit_value > top_logits[min_slot])
		{
			top_logits[min_slot] = logit_value;
			top_indices[min_slot] = i;
		}
	}

	max_logit = top_logits[0];
	for (i = 1U; i < topk; ++i)
	{
		if (top_logits[i] > max_logit)
		{
			max_logit = top_logits[i];
		}
	}

	for (i = 0U; i < topk; ++i)
	{
		const float scaled =
			(top_logits[i] - max_logit) / APP_AI_SCALAR_DECODE_TEMPERATURE;
		const float weight = expf(scaled);
		const float bin_position =
			(float)top_indices[i] / (float)(element_count - 1U);

		weight_sum += weight;
		weighted_fraction_sum += weight * bin_position;
	}

	if ((weight_sum <= 0.0f) || !AppAI_IsFiniteFloat(weight_sum))
	{
		return false;
	}

	fraction = weighted_fraction_sum / weight_sum;
	if (fraction < 0.0f)
	{
		fraction = 0.0f;
	}
	else if (fraction > 1.0f)
	{
		fraction = 1.0f;
	}

	decoded_value = APP_AI_SCALAR_DECODE_VALUE_MIN_C +
					fraction * (APP_AI_SCALAR_DECODE_VALUE_MAX_C - APP_AI_SCALAR_DECODE_VALUE_MIN_C);
	if (!AppAI_IsFiniteFloat(decoded_value))
	{
		return false;
	}

	*decoded_value_out = decoded_value;
	return true;
}
#endif /* !APP_AI_DISABLE_LEGACY_TOPK_DECODE */

/* Decode V28 polar-vote circular output: 224 int8 logits -> gauge temperature.
 *
 * Steps:
 *   1. Dequantize int8 logits to float using output scale/zero_point.
 *   2. Build angle array: angles[i] = 2*pi*i / 224.
 *   3. Compute shifted_angle[i] = fmod(angle[i] - min_angle_rad, 2*pi).
 *   4. Mask: if shifted_angle > sweep_rad + eps, set logit to -1e9.
 *   5. Softmax over 224 masked logits.
 *   6. Circular mean: sin_sum = sum(probs * sin(angles)), cos_sum = sum(probs * cos(angles)).
 *   7. mean_angle = fmod(atan2(sin_sum, cos_sum), 2*pi).
 *   8. fraction = clamp(fmod(mean_angle - min_angle_rad, 2*pi) / sweep_rad, 0, 1).
 *   9. temperature = min_value + fraction * (max_value - min_value).
 */
bool AppAI_DecodeCircularVoteFromOutput(
	const uint8_t *output_ptr,
	size_t output_len_bytes,
	float *decoded_value_out)
{
	/* Bins must match the model output dimension. */
	const size_t bins = (size_t)APP_AI_POLAR_VOTE_BINS;
	const float min_angle_rad = APP_AI_POLAR_VOTE_MIN_ANGLE_RAD;
	const float sweep_rad = APP_AI_POLAR_VOTE_SWEEP_RAD;
	const float min_value_c = APP_AI_POLAR_VOTE_MIN_VALUE_C;
	const float max_value_c = APP_AI_POLAR_VOTE_MAX_VALUE_C;
	const float mask_logit = APP_AI_POLAR_MASK_LOGIT;
	const float scale = APP_AI_POLAR_OUTPUT_SCALE;
	const int zero_point = APP_AI_POLAR_OUTPUT_ZERO_POINT;
	const float two_pi = 2.0f * 3.14159265358979323846f;
	float logits[224U];
	float probs[224U];
	float angles[224U];
	float sin_sum = 0.0f;
	float cos_sum = 0.0f;
	float max_logit = -1.0e30f;
	float sum_exp = 0.0f;
	float mean_angle = 0.0f;
	float fraction = 0.0f;
	float temperature = 0.0f;
	size_t i = 0U;

	if ((output_ptr == NULL) || (decoded_value_out == NULL))
	{
		return false;
	}

	/* We expect exactly 224 int8 output elements. */
	if (output_len_bytes < bins)
	{
		return false;
	}

	/* Step 1: dequantize int8 logits and build angle array. */
	for (i = 0U; i < bins; ++i)
	{
		/* Dequantize: float_val = (q - zero_point) * scale. */
		const int32_t q_val = (int32_t)(*(const int8_t *)&output_ptr[i]);
		const float logit_val = ((float)(q_val - zero_point)) * scale;
		const float angle = (two_pi * (float)i) / (float)bins;
		const float shifted = angle - min_angle_rad;
		float shifted_angle = shifted;
		/* fmod-equivalent: wrap shifted_angle into [0, 2*pi). */
		if (shifted_angle < 0.0f)
		{
			shifted_angle += two_pi;
		}
		if (shifted_angle >= two_pi)
		{
			shifted_angle -= two_pi;
		}
		/* Step 2-3: mask bins outside the gauge sweep. */
		if (shifted_angle > (sweep_rad + 1.0e-6f))
		{
			logits[i] = mask_logit;
		}
		else
		{
			logits[i] = logit_val;
		}
		angles[i] = angle;
	}

	/* Step 4: softmax. Find max for numerical stability. */
	for (i = 0U; i < bins; ++i)
	{
		if (logits[i] > max_logit)
		{
			max_logit = logits[i];
		}
	}

	/* Compute exp(logit - max) and accumulate. */
	for (i = 0U; i < bins; ++i)
	{
		probs[i] = expf(logits[i] - max_logit);
		sum_exp += probs[i];
	}

	/* Normalize and compute circular mean. */
	if ((sum_exp <= 0.0f) || !AppAI_IsFiniteFloat(sum_exp))
	{
		return false;
	}

	for (i = 0U; i < bins; ++i)
	{
		probs[i] /= sum_exp;
		sin_sum += probs[i] * sinf(angles[i]);
		cos_sum += probs[i] * cosf(angles[i]);
	}

	/* Step 5: circular mean angle. */
	mean_angle = atan2f(sin_sum, cos_sum);
	if (mean_angle < 0.0f)
	{
		mean_angle += two_pi;
	}

	/* Step 6: convert mean angle to gauge fraction and temperature. */
	fraction = (mean_angle - min_angle_rad) / sweep_rad;
	if (fraction < 0.0f)
	{
		fraction += (two_pi / sweep_rad);
	}
	/* Clamp fraction to [0, 1]. */
	if (fraction < 0.0f)
	{
		fraction = 0.0f;
	}
	if (fraction > 1.0f)
	{
		fraction = 1.0f;
	}

	temperature = min_value_c + fraction * (max_value_c - min_value_c);

	if (!AppAI_IsFiniteFloat(temperature))
	{
		return false;
	}

	*decoded_value_out = temperature;
	return true;
}

bool AppAI_RunStageInference(const AppAI_ModelStageSpec *stage,
									const uint8_t *frame_bytes, size_t frame_size,
									const AppAI_SourceCrop *forced_crop,
									const LL_Buffer_InfoTypeDef **output_info_out,
									float *output_value_out)
{
	uintptr_t caller_r9 = 0U;
	const LL_Buffer_InfoTypeDef *input_info = NULL;
	const LL_Buffer_InfoTypeDef *output_info = NULL;
	uint8_t *input_bytes_ptr = NULL;
	float *input_ptr = NULL;
	size_t input_len_bytes = 0U;
	size_t input_float_count = 0U;
	const uint8_t *output_ptr = NULL;
	size_t output_len_bytes = 0U;
	float output_value = 0.0f;

	__asm volatile("mov %0, r9" : "=r"(caller_r9));

	if ((stage == NULL) || (frame_bytes == NULL))
	{
		__asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
		return false;
	}

	if (AppAI_ShouldLogStageDiagnostics(stage))
	{
		(void)DebugConsole_WriteString("[AI] Stage inference request.\r\n");
	}
	if (!AppAI_EnsureStageRuntimeReady(stage))
	{
		(void)DebugConsole_WriteString(
			"[AI] Stage inference aborted before preprocess.\r\n");
		__asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
		return false;
	}

	/* Preserve the relocatable runtime base across helper calls. The ATON
	 * reloc package keeps its live model base in r9, and the epoch loop must
	 * always restore the actual ram_addr that the installed instance owns. */
	uintptr_t app_ai_saved_r9 = AppAI_GetRelocRuntimeR9(stage->nn_instance);
	if (app_ai_saved_r9 == 0U)
	{
		__asm volatile("mov %0, r9" : "=r"(app_ai_saved_r9));
	}

	input_info = AppAI_GetStageInputBufferInfo(stage);
	output_info = AppAI_GetStageOutputBufferInfo(stage);
	if ((input_info == NULL) || (output_info == NULL))
	{
		DebugConsole_Printf(
			"[AI][DIAG] Stage %s: input_info=%p output_info=%p\r\n",
			(stage->stage_label != NULL) ? stage->stage_label : "?",
			(const void *)input_info,
			(const void *)output_info);
		AppAI_ClearForcedCrop();
		__asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
		return false;
	}

	input_bytes_ptr = (uint8_t *)LL_Buffer_addr_start(input_info);
	input_ptr = (float *)input_bytes_ptr;
	input_len_bytes = (size_t)LL_Buffer_len(input_info);
	input_float_count = input_len_bytes / sizeof(float);
	const bool emit_stage_diagnostics = AppAI_ShouldLogStageDiagnostics(stage);
	const bool prefer_rgb_int8_preprocess =
		(input_info->nbits <= 8U) &&
		(input_info->mem_shape != NULL) &&
		(input_info->mem_ndims >= 4U) &&
		((size_t)input_info->mem_shape[input_info->mem_ndims - 1U] == 3U);
	static bool app_ai_logged_rgb_int8_preprocess = false;
	if (input_bytes_ptr == NULL)
	{
		DebugConsole_Printf(
			"[AI][DIAG] Stage %s: input_bytes_ptr is NULL.\r\n",
			(stage->stage_label != NULL) ? stage->stage_label : "?");
		AppAI_ClearForcedCrop();
		__asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
		return false;
	}
	if (forced_crop != NULL)
	{
		AppAI_SetForcedCrop(stage->stage_label, forced_crop->x_min,
							forced_crop->y_min, forced_crop->width, forced_crop->height);
	}
	else
	{
		AppAI_ClearForcedCrop();
	}

	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
	{
		DebugConsole_Printf(
			"[AI] Run stage=%s frame_ptr=%p frame_size=%lu input=%s addr=%p len=%lu output=%s addr=%p len=%lu\r\n",
			stage->stage_label, (const void *)frame_bytes,
			(unsigned long)frame_size,
			(input_info->name != NULL) ? input_info->name : "(unnamed)",
			(void *)input_ptr, (unsigned long)input_len_bytes,
			(output_info->name != NULL) ? output_info->name : "(unnamed)",
			(void *)LL_Buffer_addr_start(output_info),
			(unsigned long)LL_Buffer_len(output_info));
	}

	if (emit_stage_diagnostics &&
		prefer_rgb_int8_preprocess &&
		!app_ai_logged_rgb_int8_preprocess)
	{
		DebugConsole_Printf(
			"[AI] Stage preprocess mode: RGB int8 path for %s.\r\n",
			(stage->stage_label != NULL) ? stage->stage_label : "(unnamed)");
		app_ai_logged_rgb_int8_preprocess = true;
	}

#if APP_AI_YUV422_INPUT_LUMA_ONLY
	if (prefer_rgb_int8_preprocess)
	{
		/* Keep 3-channel int8 models on the RGB resize path even when the
		 * legacy scalar build is compiled with luma-only preprocessing. The
		 * live OBB localizer is trained on color and should never silently
		 * regress to Y-only replicated channels. */
		if (!AppAI_PreprocessYuv422FrameToInt8Input(frame_bytes, frame_size,
													input_bytes_ptr, input_len_bytes,
													input_info))
		{
			AppAI_ClearForcedCrop();
			__asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
			return false;
		}
	}
	else if (!AppAI_PreprocessYuv422FrameToFloatInput(frame_bytes, frame_size,
													 input_ptr, input_float_count,
													 input_len_bytes,
													 (input_info->mem_shape != NULL && input_info->mem_ndims >= 3U)
														 ? (size_t)input_info->mem_shape[2U]
														 : (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS,
													 (input_info->mem_shape != NULL && input_info->mem_ndims >= 3U)
														 ? (size_t)input_info->mem_shape[1U]
														 : (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS))
	{
		AppAI_ClearForcedCrop();
		__asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
		return false;
	}
#else
	if (prefer_rgb_int8_preprocess)
	{
		/* 8-bit 3-channel NPU stages use the packed YUV422 -> RGB byte fill
		 * path. This keeps the OBB localizer aligned with its training
		 * contract and avoids accidental grayscale replication. */
		if (emit_stage_diagnostics)
		{
			DebugConsole_Printf(
				"[AI][DIAG] Stage %s: entering int8 preprocess input=%p len=%lu frame=%p frame_sz=%lu\r\n",
				(stage->stage_label != NULL) ? stage->stage_label : "?",
				(void *)input_bytes_ptr, (unsigned long)input_len_bytes,
				(const void *)frame_bytes, (unsigned long)frame_size);
		}
		if (!AppAI_PreprocessYuv422FrameToInt8Input(frame_bytes, frame_size,
													input_bytes_ptr, input_len_bytes,
													input_info))
		{
			if (emit_stage_diagnostics)
			{
				DebugConsole_WriteString("[AI][DIAG] Stage int8 preprocess FAILED.\r\n");
			}
			AppAI_ClearForcedCrop();
			__asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
			return false;
		}
	}
	else if (!AppAI_PreprocessYuv422FrameToFloatInput(frame_bytes, frame_size,
													  input_ptr, input_float_count,
													  input_len_bytes,
													  (input_info->mem_shape != NULL && input_info->mem_ndims >= 3U)
														  ? (size_t)input_info->mem_shape[2U]
														  : (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS,
													  (input_info->mem_shape != NULL && input_info->mem_ndims >= 3U)
														  ? (size_t)input_info->mem_shape[1U]
														  : (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS))
	{

		AppAI_ClearForcedCrop();
		__asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
		return false;
	}
#endif

	/*
	 * Re-load the stage runtime base after preprocessing. The shared resize
	 * helpers make several deep calls, and if any of those paths disturb r9 we
	 * want the follow-on logs and cache maintenance to run with a clean reloc
	 * base instead of inheriting a poisoned value.
	 */
	__asm volatile("mov r9, %0" ::"r"(app_ai_saved_r9) : "r9");

	if (emit_stage_diagnostics)
	{
		(void)DebugConsole_WriteString("[AI] Stage preprocess OK.\r\n");
	}

#if APP_AI_BYPASS_SCALAR_INFERENCE
	/* Keep the tensor fill path intact, but stop before model execution so we
	 * can isolate whether the stall is in preprocessing or in LL_ATON. */
	(void)DebugConsole_WriteString(
		"[AI] Stage inference bypass active; returning after tensor fill.\r\n");
	if (output_info_out != NULL)
	{
		*output_info_out = output_info;
	}
	if (output_value_out != NULL)
	{
		*output_value_out = 0.0f;
	}
	AppAI_ClearForcedCrop();
	__asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
	return true;
#endif

	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS && (input_info->nbits > 8U))
	{
		DebugConsole_Printf(
			"[AI] %s preprocess complete; logging tensor signatures.\r\n",
			stage->stage_label);
		AppAI_LogFrameSignature(frame_bytes, frame_size);
		AppAI_LogInputSignature(input_ptr, input_float_count);
		AppAI_LogInputTensorWindow(input_ptr, input_float_count);
		AppAI_LogInputProbeSummary(input_ptr, input_float_count);
		AppAI_LogTensorPatch("Tensor center", input_ptr,
							 (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS,
							 (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS / 2U,
							 (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS / 2U, 2U);
	}

	if (emit_stage_diagnostics)
	{
		AppAI_LogBufferPreview("Stage input", input_info);
		(void)DebugConsole_WriteString("[AI] Stage input cache clean start.\r\n");
	}
	(void)mcu_cache_clean_range((uint32_t)(uintptr_t)input_ptr,
								(uint32_t)((uintptr_t)input_ptr + input_len_bytes));
	if (emit_stage_diagnostics)
	{
		(void)DebugConsole_WriteString("[AI] Stage input cache clean OK.\r\n");
	}

	/* Keep network reset opt-in. For this scalar package, forcing reset before
	 * every run can leave the output tensor in a constant invalid state even
	 * though the epoch loop reports DONE. */
	const bool force_reset_stage = (APP_AI_RESET_NETWORK_EACH_INFERENCE != 0);
	if (force_reset_stage)
	{
		if (emit_stage_diagnostics)
		{
			(void)DebugConsole_WriteString("[AI] Stage network reset start.\r\n");
		}
		LL_ATON_RT_Reset_Network(stage->nn_instance);
		if (emit_stage_diagnostics)
		{
			(void)DebugConsole_WriteString("[AI] Stage network reset OK.\r\n");
		}
	}
	else
	{
		if (emit_stage_diagnostics)
		{
			(void)DebugConsole_WriteString(
				"[AI] Stage network reset skipped (one-shot runtime).\r\n");
		}
	}

	(void)INA219_LogReading("AI-PRE");

	if (emit_stage_diagnostics && (LL_Buffer_addr_start(output_info) != NULL) &&
		(LL_Buffer_len(output_info) >= sizeof(uint32_t)))
	{
		uint32_t pre_output_bits = 0U;
		(void)memcpy(&pre_output_bits, LL_Buffer_addr_start(output_info),
					 sizeof(pre_output_bits));
		DebugConsole_Printf(
			"[AI] Scalar output pre-run bits=0x%08lX nbits=%lu Qm=%lu Qn=%lu Qu=%lu len=%lu\r\n",
			(unsigned long)pre_output_bits,
			(unsigned long)output_info->nbits,
			(unsigned long)output_info->Qm,
			(unsigned long)output_info->Qn,
			(unsigned long)output_info->Qunsigned,
			(unsigned long)LL_Buffer_len(output_info));
	}

	if (emit_stage_diagnostics)
	{
		(void)DebugConsole_WriteString("[AI] Stage inference run start.\r\n");
	}

	/* Drain any leaked WFE semaphore tokens from a previous inference.
	 * If the last epoch of a prior model completed and signalled the semaphore,
	 * but RunEpochBlock returned DONE before WFE was called, the semaphore
	 * count would be > 0, causing the next inference's WFE to return immediately. */
	extern void LL_ATON_OSAL_DrainWfeSemaphore(void);
	extern UINT LL_ATON_OSAL_GetWfeSemaphoreCount(void);
	if (stage == &app_ai_obb_stage)
	{
		DebugConsole_Printf("[AI] WFE sem count before drain: %lu\r\n",
			(unsigned long)LL_ATON_OSAL_GetWfeSemaphoreCount());
	}
	LL_ATON_OSAL_DrainWfeSemaphore();
	if (stage == &app_ai_obb_stage)
	{
		DebugConsole_Printf("[AI] WFE sem count after drain: %lu\r\n",
			(unsigned long)LL_ATON_OSAL_GetWfeSemaphoreCount());
	}

	/* Clear any leftover streaming engine IRQ flags from a previous inference.
	 * The IRQ register is write-1-to-clear. Writing all-ones clears all flags. */
#if defined(ATON_STRENG_NUM) && ATON_STRENG_NUM > 0
	for (uint32_t se_id = 0U; se_id < ATON_STRENG_NUM; ++se_id)
	{
		ATON_STRENG_IRQ_SET(se_id, 0xFFFFFFFFU);
	}
	/* Log SE2 and SE8 IRQ after clear to confirm the write took effect. */
	if (stage == &app_ai_obb_stage)
	{
		DebugConsole_Printf(
			"[AI][DIAG] SE IRQ after clear: se2_i=0x%08lx se8_i=0x%08lx\r\n",
			(unsigned long)ATON_STRENG_IRQ_GET(2),
			(unsigned long)ATON_STRENG_IRQ_GET(8));
	}
#endif

	bool mid_logged = false;
	bool inference_aborted = false;
	uint32_t obb_stage_start_tick = 0U;
	if (stage == &app_ai_obb_stage)
	{
		obb_stage_start_tick = HAL_GetTick();
	}
	for (uint32_t epoch_step = 0U;; ++epoch_step)
	{
		/* The OBB localizer has a much deeper epoch schedule than the scalar
		 * reader.  Cap runaway loops so a bad runtime state cannot freeze the
		 * board indefinitely. */
		if (epoch_step >= APP_AI_OBB_EPOCH_BUDGET_STEPS)
		{
			(void)DebugConsole_WriteString(
				"[AI] Stage inference aborted: epoch budget exhausted.\r\n");
			inference_aborted = true;
			break;
		}

		if (stage == &app_ai_obb_stage)
		{
			const uint32_t obb_elapsed_ms = HAL_GetTick() - obb_stage_start_tick;
			if (obb_elapsed_ms > APP_AI_OBB_INFERENCE_TIMEOUT_MS)
			{
				DebugConsole_Printf(
					"[AI] OBB inference aborted: time budget exhausted (%lu ms > %lu ms).\r\n",
					(unsigned long)obb_elapsed_ms,
					(unsigned long)APP_AI_OBB_INFERENCE_TIMEOUT_MS);
				inference_aborted = true;
				break;
			}
		}

		/* Log mid-inference power after a few epochs (NPU active) */
		if (!mid_logged && epoch_step == 5U)
		{
#if APP_AI_ENABLE_RUNTIME_METRICS
			(void)INA219_LogReading("AI-MID");
			Metrics_Checkpoint("MID");
#endif
			mid_logged = true;
		}

		/* Reassert xSPI2 memory-mapped mode before each epoch so the NPU's
		 * AXI master sees the weight flash. Epoch-block SW operators that read
		 * xSPI2-resident metadata (e.g. resize scale tables) can stall the
		 * system bus if MM mode drops between epochs. */
		if (!AppAI_Xspi2EnsureMemoryMappedMode())
		{
			(void)DebugConsole_WriteString("[AI] Stage inference aborted: xSPI2 MM lost.\r\n");
			inference_aborted = true;
			break;
		}

		__asm volatile("mov r9, %0" ::"r"(app_ai_saved_r9) : "r9");
		const LL_ATON_RT_RetValues_t run_status =
			LL_ATON_RT_RunEpochBlock(stage->nn_instance);
		const bool log_epoch_detail =
			(stage == &app_ai_obb_stage) && (epoch_step < 8U);

		/* Log the first few OBB epochs in detail, then throttle the rest so we
		 * can still see whether the loop is spinning fast (NO_WFE) or blocked
		 * in WFE without flooding the UART forever. */
		if (log_epoch_detail || ((epoch_step % 50U) == 0U))
		{
			const uint32_t now = HAL_GetTick();
			if (stage == &app_ai_obb_stage)
			{
				DebugConsole_Printf(
					"[AI][DBG] OBB epoch %lu elapsed=%lu status=%ld\r\n",
					(unsigned long)epoch_step,
					(unsigned long)(now - obb_stage_start_tick),
					(long)run_status);
			}
			else
			{
				DebugConsole_Printf(
					"[AI][DBG] %s epoch %lu status=%ld\r\n",
					(stage->stage_label != NULL) ? stage->stage_label : "?",
					(unsigned long)epoch_step,
					(long)run_status);
			}
		}

		if (run_status == LL_ATON_RT_DONE)
		{
			break;
		}

		if (run_status == LL_ATON_RT_WFE)
		{
			/* Probe EC/interrupt-controller state after SetWaitMask so we
			 * can see where the IRQ path breaks. */
#if defined(ATON_EPOCHCTRL_NUM) && defined(ATON_STRENG_NUM)
			if (log_epoch_detail || ((epoch_step % 50U) == 0U))
			{
				const uint32_t ec_ctrl   = ATON_EPOCHCTRL_CTRL_GET(0);
				const uint32_t ec_irq    = ATON_EPOCHCTRL_IRQ_GET(0);
				const uint32_t intreg    = ATON_INTCTRL_INTREG_GET(0);
				const uint32_t ormask    = ATON_INTCTRL_INTORMSK0_GET(0);
				const uint32_t andmask   = ATON_INTCTRL_INTANDMSK0_GET(0);
				uint32_t se2_ctrl=0, se2_irq=0, se8_ctrl=0, se8_irq=0, arith_ctrl=0;
#if ATON_STRENG_NUM > 8
				se2_ctrl = ATON_STRENG_CTRL_GET(2);
				se2_irq  = ATON_STRENG_IRQ_GET(2);
				se8_ctrl = ATON_STRENG_CTRL_GET(8);
				se8_irq  = ATON_STRENG_IRQ_GET(8);
#endif
#if defined(ATON_ARITH_NUM) && ATON_ARITH_NUM > 0
				arith_ctrl = ATON_ARITH_CTRL_GET(0);
#endif
				DebugConsole_Printf(
					"[AI][DIAG] ec_ctrl=0x%08lx ec_irq=0x%08lx"
					" intreg=0x%08lx or=0x%08lx and=0x%08lx"
					" se2_c=0x%08lx se2_i=0x%08lx"
					" se8_c=0x%08lx se8_i=0x%08lx"
					" arith=0x%08lx\r\n",
					(unsigned long)ec_ctrl, (unsigned long)ec_irq,
					(unsigned long)intreg, (unsigned long)ormask,
					(unsigned long)andmask,
					(unsigned long)se2_ctrl, (unsigned long)se2_irq,
					(unsigned long)se8_ctrl, (unsigned long)se8_irq,
					(unsigned long)arith_ctrl);
			}
#endif
			/* Measure early OBB WFE waits directly so we can tell whether the
			 * runtime is genuinely waiting on hardware or spinning on stale
			 * software state. */
			if (log_epoch_detail)
			{
				DebugConsole_Printf(
					"[AI][DIAG] epoch %lu WFE sem count before wait: %lu\r\n",
					(unsigned long)epoch_step,
					(unsigned long)LL_ATON_OSAL_GetWfeSemaphoreCount());
			}
			const uint32_t wfe_before = HAL_GetTick();

			LL_ATON_OSAL_WFE();
			__asm volatile("mov r9, %0" ::"r"(app_ai_saved_r9) : "r9");
			const uint32_t wfe_after = HAL_GetTick();

			if (log_epoch_detail)
			{
				DebugConsole_Printf(
					"[AI][DIAG] epoch %lu WFE returned after %lu ms sem=%lu\r\n",
					(unsigned long)epoch_step,
					(unsigned long)(wfe_after - wfe_before),
					(unsigned long)LL_ATON_OSAL_GetWfeSemaphoreCount());
			}
		}
		else if (run_status == LL_ATON_RT_NO_WFE)
		{
			/* NO_WFE means the runtime has already advanced the epoch chain and
			 * wants the caller to issue the next epoch immediately. Do not add a
			 * ThreadX scheduling gap here; that only stretches the OBB stage. */
		}
		else
		{
			DebugConsole_Printf(
				"[AI] Stage inference aborted: unexpected ATON status=%ld.\r\n",
				(long)run_status);
			inference_aborted = true;
			break;
		}
	}
	if (inference_aborted)
	{
#if APP_AI_ENABLE_RUNTIME_METRICS
		(void)INA219_LogReading("AI-ABORT");
		Metrics_EndInference("AI", NAN);
#endif
		AppAI_ClearForcedCrop();
		__asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
		return false;
	}
	__asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
	if (emit_stage_diagnostics)
	{
		(void)DebugConsole_WriteString("[AI] Stage inference run OK.\r\n");
	}

#if APP_AI_ENABLE_RUNTIME_METRICS
	/* Log post-inference power (peak during NPU activity) */
	(void)INA219_LogReading("AI-POST");
	Metrics_Checkpoint("POST");
#endif

	output_ptr = (const uint8_t *)LL_Buffer_addr_start(output_info);
	output_len_bytes = (size_t)LL_Buffer_len(output_info);
	if (output_ptr == NULL)
	{
#if APP_AI_ENABLE_RUNTIME_METRICS
		(void)INA219_LogReading("AI-ABORT");
		Metrics_EndInference("AI", NAN);
#endif
		AppAI_ClearForcedCrop();
		__asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
		return false;
	}
	if ((output_info->nbits <= 8U) ? (output_len_bytes < 1U) : (output_len_bytes < sizeof(float)))
	{
		DebugConsole_Printf(
			"[AI] Stage output buffer too small: nbits=%lu len=%lu\r\n",
			(unsigned long)output_info->nbits,
			(unsigned long)output_len_bytes);
#if APP_AI_ENABLE_RUNTIME_METRICS
		(void)INA219_LogReading("AI-ABORT");
		Metrics_EndInference("AI", NAN);
#endif
		AppAI_ClearForcedCrop();
		__asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
		return false;
	}

	(void)mcu_cache_invalidate_range((uint32_t)(uintptr_t)output_ptr,
									 (uint32_t)((uintptr_t)output_ptr + output_len_bytes));

	if (emit_stage_diagnostics)
	{
		AppAI_LogBufferPreview("Stage output", output_info);
		AppAI_LogScalarInternalOutputProbe(stage, output_info);
		/* The OBB candidate exports a second output (bbox vector) that the
		 * single-buffer probe above does not cover. Probe it explicitly so we
		 * can tell whether the bbox tensor is being populated and not just
		 * the confidence scalar. */
		if (stage == &app_ai_obb_stage)
		{
			const LL_Buffer_InfoTypeDef *bbox_info = &output_info[1U];
			if ((bbox_info->name != NULL) && (LL_Buffer_len(bbox_info) >= 1U))
			{
				AppAI_LogBufferPreview("Stage output bbox", bbox_info);
			}
		}
	}

	/* Log raw output bytes immediately after cache invalidate so we can tell
	 * whether the inference engine populated the buffer at all. */
	{
		const uint8_t *ob = output_ptr;
		const size_t on = (output_len_bytes < 16U) ? output_len_bytes : 16U;
		char olog[128];
#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
		if (emit_stage_diagnostics)
		{
			(void)DebugConsole_Snprintf(olog, sizeof(olog),
					   "[AI] %s out addr=%p len=%lu bytes=[%02X%02X%02X%02X%02X%02X%02X%02X"
					   "%02X%02X%02X%02X%02X%02X%02X%02X]\r\n",
					   stage->stage_label, (const void *)output_ptr,
					   (unsigned long)output_len_bytes,
					   (on > 0U) ? ob[0] : 0U, (on > 1U) ? ob[1] : 0U,
					   (on > 2U) ? ob[2] : 0U, (on > 3U) ? ob[3] : 0U,
					   (on > 4U) ? ob[4] : 0U, (on > 5U) ? ob[5] : 0U,
					   (on > 6U) ? ob[6] : 0U, (on > 7U) ? ob[7] : 0U,
					   (on > 8U) ? ob[8] : 0U, (on > 9U) ? ob[9] : 0U,
					   (on > 10U) ? ob[10] : 0U, (on > 11U) ? ob[11] : 0U,
					   (on > 12U) ? ob[12] : 0U, (on > 13U) ? ob[13] : 0U,
					   (on > 14U) ? ob[14] : 0U, (on > 15U) ? ob[15] : 0U);
			(void)DebugConsole_WriteString(olog);
		}
#endif
	}

	/* Cache the output value before xSPI2 reconfiguration, because the
	 * internal buffer info pointers will become stale after reconfigure. */
	if (output_info->nbits > 8U)
	{
		(void)memcpy(&output_value, output_ptr, sizeof(output_value));
	}
	else if (emit_stage_diagnostics && (output_info->nbits <= 8U))
	{
		float scale_value = 1.0f;
		int16_t zero_point = 0;
		int32_t q_value = 0;
		char decode_line[96] = {0};

		if (output_info->scale != NULL)
		{
			(void)memcpy(&scale_value, output_info->scale, sizeof(scale_value));
		}
		if (output_info->offset != NULL)
		{
			(void)memcpy(&zero_point, output_info->offset, sizeof(zero_point));
		}

		if (output_info->Qunsigned != 0U)
		{
			q_value = (int32_t)(*output_ptr);
		}
		else
		{
			q_value = (int32_t)(*(const int8_t *)output_ptr);
		}

		output_value = ((float)q_value - (float)zero_point) * scale_value;
		(void)DebugConsole_Snprintf(decode_line, sizeof(decode_line),
				   "[AI] Stage output quantized decode: q=%ld zp=%ld nbits=%lu qu=%lu\r\n",
				   (long)q_value, (long)zero_point,
				   (unsigned long)output_info->nbits,
				   (unsigned long)output_info->Qunsigned);
		(void)DebugConsole_WriteString(decode_line);
	}
	else
	{
		(void)memcpy(&output_value, output_ptr, sizeof(output_value));
	}

	if (output_info_out != NULL)
	{
		*output_info_out = output_info;
	}
	if (output_value_out != NULL)
	{
		*output_value_out = output_value;
	}
	/* Guard the published inference value against corrupted tensor outputs. */
	if (emit_stage_diagnostics)
	{
		if (!AppAI_IsFiniteFloat(output_value))
		{
			(void)DebugConsole_WriteString(
				"[AI] Stage output invalid: non-finite value.\r\n");
#if APP_AI_ENABLE_RUNTIME_METRICS
			(void)INA219_LogReading("AI-ABORT");
			Metrics_EndInference("AI", NAN);
#endif
			AppAI_ClearForcedCrop();
			__asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
			return false;
		}
	}

	AppAI_ClearForcedCrop();
	__asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
	return true;
}

