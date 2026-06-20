/* This file is included by app_ai_helpers.inc. */

static bool AppAI_LogXspi2ModelFilePrefix(FX_FILE *model_file_ptr)
{
	uint8_t source_bytes[APP_AI_XSPI2_PROBE_BYTES] = {0U};
	ULONG bytes_read = 0U;
	UINT fx_status = FX_SUCCESS;

	if (model_file_ptr == NULL)
	{
		return false;
	}

	fx_status = fx_file_read(model_file_ptr, source_bytes,
							 APP_AI_XSPI2_PROBE_BYTES, &bytes_read);
	if ((fx_status != FX_SUCCESS) || (bytes_read != APP_AI_XSPI2_PROBE_BYTES))
	{
		DebugConsole_Printf(
			"[AI] xSPI2 source prefix read failed (fx=%lu n=%lu).\r\n",
			(unsigned long)fx_status, (unsigned long)bytes_read);
		return false;
	}

	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
	{
		AppAI_LogXspi2PrefixBytes("xSPI2 source prefix:", source_bytes);
	}

	fx_status = fx_file_seek(model_file_ptr, 0U);
	if (fx_status != FX_SUCCESS)
	{
		DebugConsole_Printf(
			"[AI] xSPI2 source rewind failed (fx=%lu).\r\n",
			(unsigned long)fx_status);
		return false;
	}

	return true;
}

static bool AppAI_Xspi2ModelImageMatchesMappedFlash(void)
{
	if (!AppAI_Xspi2ReadMappedProbe(0U, app_ai_xspi2_signature_start,
									sizeof(app_ai_xspi2_signature_start)))
	{
		DebugConsole_Printf(
			"[AI] xSPI2 mapped verify failed at start signature.\r\n");
		return false;
	}

	if ((app_ai_xspi2_programmed_size >= APP_AI_XSPI2_PROBE_BYTES) && !AppAI_Xspi2ReadMappedProbe(
																		  app_ai_xspi2_programmed_size - APP_AI_XSPI2_PROBE_BYTES,
																		  app_ai_xspi2_signature_tail,
																		  sizeof(app_ai_xspi2_signature_tail)))
	{
		DebugConsole_Printf(
			"[AI] xSPI2 mapped verify failed at tail signature.\r\n");
		return false;
	}

	return true;
}

static bool AppAI_Xspi2ModelImageMatchesMappedFlashForStage(
	const AppAI_ModelStageSpec *stage)
{
	const char *stage_label = NULL;
	bool is_rectifier;
	bool is_obb;

	if (stage == NULL)
	{
		return false;
	}

	stage_label = (stage->stage_label != NULL) ? stage->stage_label : "scalar";
	is_rectifier = (strcmp(stage_label, "rectifier") == 0);
	is_obb = (strcmp(stage_label, "obb_center") == 0) || (strstr(stage_label, "qarepvgg") != NULL);
	bool is_source_crop_box = (strcmp(stage_label, "source_crop_box") == 0);

	if (is_rectifier)
	{
		DebugConsole_WriteString(
			"[AI] rectifier stage trusted on flash; skipping signature compare.\r\n");
		return true;
	}

	const uint8_t *sig_start;
	const uint8_t *sig_tail;
	bool sig_valid;
	ULONG programmed_size;

	if (is_obb)
	{
		sig_start = app_ai_obb_sig_valid
						? app_ai_obb_sig_start
						: app_ai_obb_xspi2_signature_start;
		sig_tail = app_ai_obb_sig_valid
					   ? app_ai_obb_sig_tail
					   : app_ai_obb_xspi2_signature_tail;
		sig_valid = app_ai_obb_sig_valid;
		programmed_size = app_ai_obb_programmed_size;
	}
	else if (is_source_crop_box)
	{
		sig_start = app_ai_source_crop_box_sig_valid
						? app_ai_source_crop_box_sig_start
						: app_ai_source_crop_box_xspi2_signature_start;
		sig_tail = app_ai_source_crop_box_sig_valid
					   ? app_ai_source_crop_box_sig_tail
					   : app_ai_source_crop_box_xspi2_signature_tail;
		sig_valid = app_ai_source_crop_box_sig_valid;
		programmed_size = app_ai_source_crop_box_programmed_size;
	}
	else
	{
		sig_start = app_ai_scalar_sig_valid
						? app_ai_scalar_sig_start
						: app_ai_xspi2_signature_start;
		sig_tail = app_ai_scalar_sig_valid
					   ? app_ai_scalar_sig_tail
					   : app_ai_xspi2_signature_tail;
		sig_valid = app_ai_scalar_sig_valid;
		programmed_size = app_ai_scalar_programmed_size;
	}

	if (!AppAI_Xspi2ReadStageProbe(stage, 0U,
								   sig_start, APP_AI_XSPI2_PROBE_BYTES))
	{
		/* Log the actual flash bytes so stale hardcoded signatures can be updated. */
		{
			uint8_t actual[APP_AI_XSPI2_PROBE_BYTES] = {0U};
			int32_t read_status = BSP_XSPI_NOR_Read(0U, actual,
													stage->xspi2_chip_offset, APP_AI_XSPI2_PROBE_BYTES);
			{
				char vlog[160];
				(void)DebugConsole_Snprintf(vlog, sizeof(vlog),
					"[AI] %s mismatch rc=%ld actual=[%02X%02X%02X%02X%02X%02X%02X%02X] "
					"expect=[%02X%02X%02X%02X%02X%02X%02X%02X] offset=0x%08lX\r\n",
					stage->stage_label, (long)read_status,
					actual[0], actual[1], actual[2], actual[3],
					actual[4], actual[5], actual[6], actual[7],
					sig_start[0], sig_start[1], sig_start[2], sig_start[3],
					sig_start[4], sig_start[5], sig_start[6], sig_start[7],
					(unsigned long)(stage->xspi2_chip_offset));
				DebugConsole_WriteString(vlog);
			}
		}
		(void)DebugConsole_WriteString(
			"[AI] xSPI2 stage signature mismatch at head probe.\r\n");
		return false;
	}

	if (sig_valid && (programmed_size >= APP_AI_XSPI2_PROBE_BYTES) &&
		!AppAI_Xspi2ReadStageProbe(stage,
								   programmed_size - APP_AI_XSPI2_PROBE_BYTES,
								   sig_tail,
								   APP_AI_XSPI2_PROBE_BYTES))
	{
		{
#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
			char vlog[64];
			(void)DebugConsole_Snprintf(vlog, sizeof(vlog),
						   "[AI] %s flash verify: tail mismatch.\r\n",
						   stage->stage_label);
			(void)DebugConsole_WriteString(vlog);
#endif
		}
		(void)DebugConsole_WriteString(
			"[AI] xSPI2 stage signature mismatch at tail probe.\r\n");
		return false;
	}

	return true;
}

static bool AppAI_ProgramXspi2ModelImageFromSd(void)
{
	FX_MEDIA *media_ptr = NULL;
	FX_FILE model_file = {0};
	ULONG file_size = 0U;
	ULONG bytes_read = 0U;
	ULONG bytes_remaining = 0U;
	ULONG flash_offset = 0U;
	UINT fx_status = FX_SUCCESS;
	UINT tx_status = TX_SUCCESS;
	int32_t bsp_status = BSP_ERROR_NONE;
	uint8_t source_prefix[APP_AI_XSPI2_PROBE_BYTES] = {0U};
	uint8_t source_tail[APP_AI_XSPI2_PROBE_BYTES] = {0U};
	bool has_tail_probe = false;

	if (!AppAI_WaitForFileXMediaReady(APP_AI_FILEX_MEDIA_READY_TIMEOUT_MS))
	{
		AppAI_LogXspi2LoadFailure("FileX not ready", FX_MEDIA_NOT_OPEN,
								  BSP_ERROR_NONE);
		return false;
	}

	tx_status = AppFileX_AcquireMediaLock();
	if (tx_status != TX_SUCCESS)
	{
		AppAI_LogXspi2LoadFailure("media lock", (UINT)tx_status, BSP_ERROR_NONE);
		return false;
	}

	media_ptr = AppFileX_GetMediaHandle();
	if (media_ptr == NULL)
	{
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("media handle", FX_MEDIA_NOT_OPEN,
								  BSP_ERROR_NONE);
		return false;
	}

	if (fx_directory_default_set(media_ptr, FX_NULL) != FX_SUCCESS)
	{
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("default directory", FX_SUCCESS,
								  BSP_ERROR_NONE);
		return false;
	}

	fx_status = fx_file_open(media_ptr, &model_file,
							 (CHAR *)APP_AI_XSPI2_MODEL_IMAGE_PATH, FX_OPEN_FOR_READ);
	if (fx_status != FX_SUCCESS)
	{
		(void)fx_directory_default_set(media_ptr, FX_NULL);
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("file open", fx_status, BSP_ERROR_NONE);
		return false;
	}

	file_size = model_file.fx_file_current_file_size;
	if (file_size == 0U)
	{
		(void)fx_file_close(&model_file);
		(void)fx_directory_default_set(media_ptr, FX_NULL);
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("file size (empty)", FX_SUCCESS, BSP_ERROR_NONE);
		return false;
	}
	DebugConsole_Printf("[AI] Model file size: %lu bytes.\r\n",
						(unsigned long)file_size);

	fx_status = fx_file_seek(&model_file, 0U);
	if (fx_status != FX_SUCCESS)
	{
		(void)fx_file_close(&model_file);
		(void)fx_directory_default_set(media_ptr, FX_NULL);
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("file seek", fx_status, BSP_ERROR_NONE);
		return false;
	}

	if (!AppAI_ReadXspi2ModelSourceProbes(&model_file, file_size,
										  source_prefix, source_tail, &has_tail_probe))
	{
		(void)fx_file_close(&model_file);
		(void)fx_directory_default_set(media_ptr, FX_NULL);
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("source probes", FX_SUCCESS,
								  BSP_ERROR_COMPONENT_FAILURE);
		return false;
	}

	for (ULONG erase_addr = 0U; erase_addr < file_size;
		 erase_addr += APP_AI_XSPI2_ERASE_BLOCK_BYTES)
	{
		if (erase_addr == 0U)
		{
			(void)DebugConsole_WriteString("[AI] xSPI2 stage erase begin.\r\n");
		}
		bsp_status = BSP_XSPI_NOR_Erase_Block(0U,
											  APP_AI_XSPI2_MODEL_CHIP_OFFSET + erase_addr,
											  BSP_XSPI_NOR_ERASE_64K);
		if (bsp_status != BSP_ERROR_NONE)
		{
			(void)fx_file_close(&model_file);
			(void)fx_directory_default_set(media_ptr, FX_NULL);
			AppFileX_ReleaseMediaLock();
			AppAI_LogXspi2LoadFailure("flash erase", FX_SUCCESS, bsp_status);
			return false;
		}
	}

	bytes_remaining = file_size;
	flash_offset = 0U;
	{
		ULONG chunk_index = 0U;

		while (bytes_remaining > 0U)
		{
			const ULONG chunk_size = (bytes_remaining > APP_AI_XSPI2_PROGRAM_CHUNK_BYTES)
										 ? APP_AI_XSPI2_PROGRAM_CHUNK_BYTES
										 : bytes_remaining;

			if ((flash_offset == 0U) || ((flash_offset % APP_AI_XSPI2_ERASE_BLOCK_BYTES) == 0U))
			{
				AppAI_LogXspi2ProgramChunkProgress(chunk_index, flash_offset,
												   chunk_size);
			}

			bytes_read = 0U;
			fx_status = fx_file_read(&model_file, app_ai_xspi2_program_buffer,
									 chunk_size, &bytes_read);
			if ((fx_status != FX_SUCCESS) || (bytes_read != chunk_size))
			{
				(void)fx_file_close(&model_file);
				(void)fx_directory_default_set(media_ptr, FX_NULL);
				AppFileX_ReleaseMediaLock();
				AppAI_LogXspi2LoadFailure("file read", fx_status, BSP_ERROR_NONE);
				return false;
			}

			/* Keep the flash writer honest: the staging buffer is cacheable RAM, so
			 * clean it before BSP_XSPI_NOR_Write() consumes the bytes. */
			(void)mcu_cache_clean_range((uint32_t)(uintptr_t)app_ai_xspi2_program_buffer,
										(uint32_t)(uintptr_t)app_ai_xspi2_program_buffer + (uint32_t)chunk_size);

			bsp_status = BSP_XSPI_NOR_Write(0U, app_ai_xspi2_program_buffer,
											APP_AI_XSPI2_MODEL_CHIP_OFFSET + flash_offset,
											(uint32_t)chunk_size);
			if (bsp_status != BSP_ERROR_NONE)
			{
				(void)fx_file_close(&model_file);
				(void)fx_directory_default_set(media_ptr, FX_NULL);
				AppFileX_ReleaseMediaLock();
				AppAI_LogXspi2LoadFailure("flash write", FX_SUCCESS, bsp_status);
				return false;
			}

			flash_offset += chunk_size;
			bytes_remaining -= chunk_size;
			chunk_index++;
		}
	}

	(void)fx_file_close(&model_file);
	(void)fx_directory_default_set(media_ptr, FX_NULL);
	AppFileX_ReleaseMediaLock();

	app_ai_xspi2_programmed_size = file_size;
	app_ai_scalar_programmed_size = file_size;
	(void)memcpy(app_ai_scalar_sig_start, source_prefix,
				 APP_AI_XSPI2_PROBE_BYTES);
	if (has_tail_probe)
	{
		(void)memcpy(app_ai_scalar_sig_tail, source_tail,
					 APP_AI_XSPI2_PROBE_BYTES);
	}
	else
	{
		(void)memset(app_ai_scalar_sig_tail, 0, APP_AI_XSPI2_PROBE_BYTES);
	}
	app_ai_scalar_sig_valid = has_tail_probe;
	AppAI_LogXspi2FlashStatus("legacy stage write complete");

	if (!AppAI_ReconfigureXspi2ForRuntime())
	{
		AppAI_LogXspi2LoadFailure("runtime reconfigure", FX_SUCCESS,
								  BSP_ERROR_COMPONENT_FAILURE);
		return false;
	}
	if (!AppAI_Xspi2EnableMemoryMappedMode())
	{
		AppAI_LogXspi2LoadFailure("enable MM after provision", FX_SUCCESS,
								  BSP_ERROR_COMPONENT_FAILURE);
		return false;
	}

	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
	{
		AppAI_LogXspi2IndirectAndMappedPrefix();
		AppAI_LogXspi2MappedScaleBytes();
	}

	return true;
}

static void AppAI_SetForcedCrop(const char *label, size_t x_min,
								size_t y_min, size_t width, size_t height)
{
	app_ai_forced_crop_active = true;
	app_ai_forced_crop_label = label;
	app_ai_forced_crop_x_min = x_min;
	app_ai_forced_crop_y_min = y_min;
	app_ai_forced_crop_width = width;
	app_ai_forced_crop_height = height;
}

static void AppAI_ClearForcedCrop(void)
{
	app_ai_forced_crop_active = false;
	app_ai_forced_crop_label = NULL;
	app_ai_forced_crop_x_min = 0U;
	app_ai_forced_crop_y_min = 0U;
	app_ai_forced_crop_width = 0U;
	app_ai_forced_crop_height = 0U;
}

static const LL_Buffer_InfoTypeDef *AppAI_GetStageInputBufferInfo(
	const AppAI_ModelStageSpec *stage)
{
	const LL_Buffer_InfoTypeDef *input_info = NULL;

	if ((stage == NULL) || (stage->nn_instance == NULL) || (stage->nn_instance->network == NULL) || (stage->nn_instance->network->input_buffers_info == NULL))
	{
		return NULL;
	}

	input_info = stage->nn_instance->network->input_buffers_info();
	if ((input_info == NULL) || (input_info->name == NULL))
	{
		return NULL;
	}
	return input_info;
}

static const LL_Buffer_InfoTypeDef *AppAI_GetStageOutputBufferInfo(
	const AppAI_ModelStageSpec *stage)
{
	const LL_Buffer_InfoTypeDef *output_info = NULL;

	if ((stage == NULL) || (stage->nn_instance == NULL) || (stage->nn_instance->network == NULL) || (stage->nn_instance->network->output_buffers_info == NULL))
	{
		return NULL;
	}

	output_info = stage->nn_instance->network->output_buffers_info();
	if ((output_info == NULL) || (output_info->name == NULL))
	{
		return NULL;
	}
	return output_info;
}

static bool AppAI_ReadXspi2ModelSourceProbes(FX_FILE *model_file_ptr,
											 ULONG file_size, uint8_t *source_prefix, uint8_t *source_tail,
											 bool *has_tail_out)
{
	ULONG bytes_read = 0U;
	UINT fx_status = FX_SUCCESS;

	if ((model_file_ptr == NULL) || (source_prefix == NULL) || (source_tail == NULL) || (has_tail_out == NULL))
	{
		return false;
	}

	*has_tail_out = false;

	fx_status = fx_file_seek(model_file_ptr, 0U);
	if (fx_status != FX_SUCCESS)
	{
		return false;
	}

	bytes_read = 0U;
	fx_status = fx_file_read(model_file_ptr, source_prefix,
							 APP_AI_XSPI2_PROBE_BYTES, &bytes_read);
	if ((fx_status != FX_SUCCESS) || (bytes_read != APP_AI_XSPI2_PROBE_BYTES))
	{
		return false;
	}
	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
	{
		AppAI_LogXspi2PrefixBytes("xSPI2 source prefix:", source_prefix);
	}

	if (file_size >= APP_AI_XSPI2_PROBE_BYTES)
	{
		fx_status = fx_file_seek(model_file_ptr,
								 file_size - APP_AI_XSPI2_PROBE_BYTES);
		if (fx_status != FX_SUCCESS)
		{
			return false;
		}

		bytes_read = 0U;
		fx_status = fx_file_read(model_file_ptr, source_tail,
								 APP_AI_XSPI2_PROBE_BYTES, &bytes_read);
		if ((fx_status != FX_SUCCESS) || (bytes_read != APP_AI_XSPI2_PROBE_BYTES))
		{
			return false;
		}
		if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
		{
			AppAI_LogXspi2PrefixBytes("xSPI2 source tail:", source_tail);
		}
		*has_tail_out = true;
	}

	fx_status = fx_file_seek(model_file_ptr, 0U);
	if (fx_status != FX_SUCCESS)
	{
		return false;
	}

	return true;
}

static bool AppAI_EnsureXspi2ModelImageReadyForStage(
	const AppAI_ModelStageSpec *stage)
{
	if (stage == NULL)
	{
		return false;
	}

	/* Fast path: pointer equality means this stage is already in its flash
	 * region and the peripheral is already in mapped mode for it. */
	if (app_ai_loaded_xspi2_stage == stage)
	{
		(void)DebugConsole_WriteString("[AI] xSPI2 stage already loaded.\r\n");
		if (!app_ai_xspi2_mm_enabled)
		{
			(void)DebugConsole_WriteString(
				"[AI] xSPI2 stage was loaded but MM was off; re-enabling MM.\r\n");
			if (!AppAI_Xspi2EnableMemoryMappedMode())
			{
				return false;
			}
		}
		return true;
	}

	/* Each stage now has its own flash region, so we can verify whether the
	 * stage's bytes are already present without disturbing the other stage.
	 * Switch to indirect mode first (needed for BSP_XSPI_NOR_Read). */
#if APP_AI_ENABLE_XSPI2_VERBOSE_LOGS
	DebugConsole_WriteString("[AI] xSPI2 stage reconfigure start.\r\n");
#endif
	if (!AppAI_ReconfigureXspi2ForRuntime())
	{
		DebugConsole_WriteString("[AI] xSPI2 stage reconfigure FAILED.\r\n");
		return false;
	}
	/* Any reconfigure invalidates the previously loaded stage/MM state until
	 * this function verifies and re-enables mapped mode for the target stage. */
	app_ai_loaded_xspi2_stage = NULL;
#if APP_AI_ENABLE_XSPI2_VERBOSE_LOGS
	DebugConsole_WriteString("[AI] xSPI2 stage reconfigure OK.\r\n");
#endif

	if (!AppAI_Xspi2ModelImageMatchesMappedFlashForStage(stage))
	{
		/* The board now treats xSPI2 blobs as flashed-only assets. Any mismatch
		 * means the model package on flash is stale or the wrong stage was
		 * provisioned, so fail fast and let flash_boot.ps1 refresh it. */
		DebugConsole_Printf(
			"[AI] xSPI2 stage '%s' signature mismatch; aborting stage load.\r\n",
			stage->stage_label);
		return false;
	}

	if (AppAI_ShouldLogStageDiagnostics(stage))
	{
		DebugConsole_WriteString("[AI] xSPI2 stage image already present.\r\n");
	}

	/* Reprovision can already leave xSPI2 in memory-mapped mode for this
	 * stage. Avoid a second enable call in that case, which has been
	 * intermittently failing on board. */
	if (!app_ai_xspi2_mm_enabled && !AppAI_Xspi2EnableMemoryMappedMode())
	{
		AppAI_LogXspi2LoadFailure("enable MM after verify", FX_SUCCESS,
								  BSP_ERROR_COMPONENT_FAILURE);
		return false;
	}
	app_ai_loaded_xspi2_stage = stage;
	return true;
}

static bool AppAI_EnsureStageRuntimeReady(const AppAI_ModelStageSpec *stage)
{
	if (stage == NULL)
	{
		return false;
	}

	if (AppAI_ShouldLogStageDiagnostics(stage))
	{
		DebugConsole_WriteString("[AI] Stage runtime ready start.\r\n");
		DebugConsole_Printf("[AI] Stage label: %s\r\n",
					(stage->stage_label != NULL) ? stage->stage_label : "(unnamed)");
	}
	if (!AppAI_EnsureXspi2ModelImageReadyForStage(stage))
	{
		if (AppAI_ShouldLogStageDiagnostics(stage))
		{
			DebugConsole_WriteString(
				"[AI] Stage runtime ready failed during xSPI2 setup.\r\n");
		}
		return false;
	}

	if (AppAI_ShouldLogStageDiagnostics(stage))
	{
		DebugConsole_WriteString("[AI] Stage network init start.\r\n");
	}
	if ((stage->network_init_fn == NULL) || !stage->network_init_fn())
	{
		AppAI_LogInitFailure(stage->stage_label);
		return false;
	}

	LL_ATON_RT_Init_Network(stage->nn_instance);
	if (AppAI_ShouldLogStageDiagnostics(stage))
	{
		DebugConsole_WriteString("[AI] Stage network init OK.\r\n");
	}

	if (AppAI_ShouldLogStageDiagnostics(stage))
	{
		DebugConsole_WriteString("[AI] Stage inference init start.\r\n");
	}
	if ((stage->inference_init_fn == NULL) || !stage->inference_init_fn())
	{
		AppAI_LogInitFailure(stage->stage_label);
		return false;
	}

	if (AppAI_ShouldLogStageDiagnostics(stage))
	{
		DebugConsole_WriteString("[AI] Stage inference init OK.\r\n");
	}
	return true;
}

static bool AppAI_DecodeRectifierCropBox(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	AppAI_SourceCrop *crop_out,
	AppAI_RectifierBox *rectifier_box_out)
{
	const float *output_ptr = NULL;
	const size_t source_width = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const size_t source_height = (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	float center_x = 0.0f;
	float center_y = 0.0f;
	float box_w = 0.0f;
	float box_h = 0.0f;
	float crop_x_min_f = 0.0f;
	float crop_y_min_f = 0.0f;
	float crop_width_f = 0.0f;
	float crop_height_f = 0.0f;
#if APP_AI_RECTIFIER_FIXED_SCALE_CROP
	size_t training_center_x = 0U;
	size_t training_center_y = 0U;
	const float rectifier_center_blend_f =
		((float)APP_AI_RECTIFIER_CENTER_BLEND_NUMERATOR) / ((float)APP_AI_RECTIFIER_CENTER_BLEND_DENOMINATOR);
#endif
	bool use_fixed_training_crop = false;

	if ((output_buffer_info == NULL) || (crop_out == NULL))
	{
		return false;
	}

	output_ptr = (const float *)LL_Buffer_addr_start(output_buffer_info);
	if ((output_ptr == NULL) || (LL_Buffer_len(output_buffer_info) < (sizeof(float) * 4U)))
	{
		return false;
	}

	/* Log the raw rectifier output before any clamping so we can distinguish
	 * between the model producing a plausible box that the fallback rejects,
	 * a near-zero output (model not running / zero-init), or an out-of-range
	 * value (wrong output format / normalization mismatch). */
	{
		union
		{
			float f;
			uint32_t u;
		} r0, r1, r2, r3;
		char rect_log[192];
		r0.f = output_ptr[0];
		r1.f = output_ptr[1];
		r2.f = output_ptr[2];
		r3.f = output_ptr[3];
#if APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
		(void)DebugConsole_Snprintf(rect_log, sizeof(rect_log),
					   "[AI] Rectifier raw: cx=%ld cy=%ld w=%ld h=%ld "
					   "bits=[%08lX %08lX %08lX %08lX]\r\n",
					   (long)(output_ptr[0] * 1000.0f),
					   (long)(output_ptr[1] * 1000.0f),
					   (long)(output_ptr[2] * 1000.0f),
					   (long)(output_ptr[3] * 1000.0f),
					   (unsigned long)r0.u, (unsigned long)r1.u,
					   (unsigned long)r2.u, (unsigned long)r3.u);
		(void)DebugConsole_WriteString(rect_log);
#endif
	}

	center_x = AppAI_ClampNormalizedFloat(output_ptr[0]);
	center_y = AppAI_ClampNormalizedFloat(output_ptr[1]);
	box_w = AppAI_ClampNormalizedFloat(output_ptr[2]);
	box_h = AppAI_ClampNormalizedFloat(output_ptr[3]);
	if (box_w < APP_AI_RECTIFIER_MIN_BOX_RATIO)
	{
		box_w = APP_AI_RECTIFIER_MIN_BOX_RATIO;
	}
	if (box_h < APP_AI_RECTIFIER_MIN_BOX_RATIO)
	{
		box_h = APP_AI_RECTIFIER_MIN_BOX_RATIO;
	}

	if (rectifier_box_out != NULL)
	{
		rectifier_box_out->center_x = center_x;
		rectifier_box_out->center_y = center_y;
		rectifier_box_out->box_w = box_w;
		rectifier_box_out->box_h = box_h;
	}

#if APP_AI_RECTIFIER_FIXED_SCALE_CROP
	/* Soft attention: ignore (box_w, box_h) entirely. Validate centre only. */
	if ((center_x < APP_AI_RECTIFIER_CENTER_MIN_RATIO) || (center_x > APP_AI_RECTIFIER_CENTER_MAX_RATIO) || (center_y < APP_AI_RECTIFIER_CENTER_MIN_RATIO) || (center_y > APP_AI_RECTIFIER_CENTER_MAX_RATIO))
	{
		{
			char fb_log[96];
#if APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
			(void)DebugConsole_Snprintf(fb_log, sizeof(fb_log),
						   "[AI] Rect fallback (centre out of range): cx=%ld cy=%ld lim=[%ld..%ld]\r\n",
						   (long)(center_x * 1000.0f), (long)(center_y * 1000.0f),
						   (long)(APP_AI_RECTIFIER_CENTER_MIN_RATIO * 1000.0f),
						   (long)(APP_AI_RECTIFIER_CENTER_MAX_RATIO * 1000.0f));
			(void)DebugConsole_WriteString(fb_log);
#endif
		}
		use_fixed_training_crop = true;
	}
#else
	if ((box_w < APP_AI_RECTIFIER_FALLBACK_MIN_BOX_RATIO) || (box_h < APP_AI_RECTIFIER_FALLBACK_MIN_BOX_RATIO) || (box_w > APP_AI_RECTIFIER_FALLBACK_MAX_BOX_RATIO) || (box_h > APP_AI_RECTIFIER_FALLBACK_MAX_BOX_RATIO))
	{
		{
			char fb_log[96];
#if APP_AI_ENABLE_RECTIFIER_DIAGNOSTICS
			(void)DebugConsole_Snprintf(fb_log, sizeof(fb_log),
						   "[AI] Rect fallback: cx=%ld cy=%ld w=%ld h=%ld lim=[%ld..%ld]\r\n",
						   (long)(center_x * 1000.0f), (long)(center_y * 1000.0f),
						   (long)(box_w * 1000.0f), (long)(box_h * 1000.0f),
						   (long)(APP_AI_RECTIFIER_FALLBACK_MIN_BOX_RATIO * 1000.0f),
						   (long)(APP_AI_RECTIFIER_FALLBACK_MAX_BOX_RATIO * 1000.0f));
			(void)DebugConsole_WriteString(fb_log);
#endif
		}
		use_fixed_training_crop = true;
	}
#endif
