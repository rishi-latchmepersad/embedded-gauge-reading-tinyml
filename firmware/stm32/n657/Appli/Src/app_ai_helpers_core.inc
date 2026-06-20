/* This file is included by app_ai_helpers.inc. */

static uint32_t AppAI_GrayToFloatBits(uint8_t gray)
{
	/* Convert luma to a normalized float32 payload without reserving extra RAM. */
	union
	{
		float f;
		uint32_t u;
	} gray_value = {
		.f = ((float)gray) * (1.0f / 255.0f)};

	return gray_value.u;
}
static uint8_t AppAI_ReadYuv422Luma(const uint8_t *frame_bytes,
									size_t frame_size_bytes,
									size_t frame_width_pixels, size_t source_x, size_t source_y);
static void AppAI_ReadYuv422Quartet(const uint8_t *frame_bytes,
									size_t frame_size_bytes,
									size_t frame_width_pixels, size_t source_x, size_t source_y,
									uint8_t *quad_out);
static float AppAI_ReadNormalizedGrayFromYuv422Pixel(const uint8_t *frame_bytes,
													 size_t frame_size_bytes,
													 size_t frame_width_pixels, size_t source_x, size_t source_y);
static void AppAI_ReadRgbFromYuv422Pixel(const uint8_t *frame_bytes,
										 size_t frame_size_bytes,
										 size_t frame_width_pixels, size_t source_x, size_t source_y,
										 float *r_out, float *g_out, float *b_out);
static void AppAI_ReadRgbFromYuv422Bilinear(const uint8_t *frame_bytes,
											size_t frame_size_bytes,
											size_t frame_width_pixels,
											size_t frame_height_pixels,
											float source_x, float source_y,
											float *r_out, float *g_out, float *b_out);
static void AppAI_ReadRgbFromRgbBilinear(const uint8_t *rgb_bytes,
										 size_t frame_width_pixels,
										 size_t frame_height_pixels,
										 float source_x, float source_y,
										 float *r_out, float *g_out, float *b_out);
static uint8_t AppAI_FindHistogramPercentile(const uint32_t *histogram,
											 size_t histogram_len,
											 uint32_t total_count,
											 float percentile);
static float AppAI_ScorePolarAlignmentCandidate(
	const uint8_t *resized_rgb,
	size_t output_dim,
	float center_x,
	float center_y,
	float max_radius,
	uint8_t *polar_luma_out);
static void AppAI_SetForcedCrop(const char *label, size_t x_min,
								size_t y_min, size_t width, size_t height);
static void AppAI_ClearForcedCrop(void);
#if !APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
static bool AppAI_DecodeObbCropBox(const LL_Buffer_InfoTypeDef *output_buffer_info,
									   AppAI_SourceCrop *crop_out, AppAI_ObbBox *obb_box_out);
#if APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE
static bool AppAI_DecodeSourceCropBox(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	AppAI_SourceCrop *crop_out,
	AppAI_SourceCropBox *box_out);
#endif /* APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE */
static bool AppAI_DecodeRectifierCropBox(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	AppAI_SourceCrop *crop_out,
	AppAI_RectifierBox *rectifier_box_out);
#if !defined(APP_AI_DISABLE_LEGACY_TOPK_DECODE)
static bool AppAI_DecodeScalarTopKExpectationFromOutput(
	const LL_Buffer_InfoTypeDef *output_info,
	const uint8_t *output_ptr,
	size_t output_len_bytes,
	float *decoded_value_out);
#endif /* !APP_AI_DISABLE_LEGACY_TOPK_DECODE */
static bool AppAI_DecodeCircularVoteFromOutput(
	const uint8_t *output_ptr,
	size_t output_len_bytes,
	float *decoded_value_out);
static bool AppAI_PreprocessYuv422FrameToPolarInput(
	const uint8_t *frame_bytes, size_t frame_size,
	uint8_t *input_ptr, size_t input_len_bytes,
	const LL_Buffer_InfoTypeDef *input_info);
#endif
int mcu_cache_clean_range(uint32_t start_addr, uint32_t end_addr)
{
	return AppAI_ApplyCacheRange(start_addr, end_addr, true, false);
}

int mcu_cache_invalidate_range(uint32_t start_addr, uint32_t end_addr)
{
	return AppAI_ApplyCacheRange(start_addr, end_addr, false, true);
}

int mcu_cache_clean_invalidate_range(uint32_t start_addr, uint32_t end_addr)
{
	/* ST's ll_aton cache helper expects the combined maintenance hook too. */
	return AppAI_ApplyCacheRange(start_addr, end_addr, true, true);
}

static bool AppAI_EnsureNpuHardwareReady(void)
{
	if (app_ai_npu_hw_initialized)
	{
		return true;
	}

	(void)DebugConsole_WriteString("[AI] NPU hardware bring-up start.\r\n");
	__HAL_RCC_NPU_CLK_ENABLE();
	__HAL_RCC_NPU_FORCE_RESET();
	__HAL_RCC_NPU_RELEASE_RESET();
	__HAL_RCC_NPU_CLK_SLEEP_DISABLE();

	__HAL_RCC_CACHEAXI_CLK_ENABLE();
	__HAL_RCC_CACHEAXI_FORCE_RESET();
	__HAL_RCC_CACHEAXI_RELEASE_RESET();
	__HAL_RCC_CACHEAXI_CLK_SLEEP_DISABLE();

	AppAI_EnableNpuMemoryAndCaches();

	npu_cache_init();
	npu_cache_enable();

	AppAI_ConfigureNpuAccessControl();
	AppAI_ConfigureNpuRisafDefaults();

	app_ai_npu_hw_initialized = true;
	(void)DebugConsole_WriteString("[AI] NPU hardware bring-up OK.\r\n");
	return true;
}

static bool AppAI_EnsureXspi2MemoryReady(void)
{
	BSP_XSPI_NOR_Init_t flash = {0};
	RCC_PeriphCLKInitTypeDef periph_clk = {0};
	int32_t bsp_status = BSP_ERROR_NONE;

	if (app_ai_xspi2_initialized)
	{
		return true;
	}

	(void)DebugConsole_WriteString("[AI] XSPI2 memory bring-up start.\r\n");
	/* If a prior verify attempt left the flash in memory-mapped mode, erase and
	 * write commands will fail.  Take it back to indirect mode first. DeInit
	 * handles this cleanly regardless of the current BSP context state. */
	(void)BSP_XSPI_NOR_DeInit(0U);
	app_ai_xspi2_mm_enabled = false;

	periph_clk.PeriphClockSelection = RCC_PERIPHCLK_XSPI2;
	periph_clk.Xspi2ClockSelection = RCC_XSPI2CLKSOURCE_IC3;
	if (HAL_RCCEx_PeriphCLKConfig(&periph_clk) != HAL_OK)
	{
		(void)DebugConsole_WriteString("[AI] XSPI2 clock config failed.\r\n");
		return false;
	}

	/* Ensure the xSPI2 clock is enabled after configuration change.
	 * The deinit may have disabled the clock, and the new clock source
	 * won't be active until the peripheral clock enable is asserted. */
	XSPI_CLK_ENABLE();
	(void)DebugConsole_WriteString("[AI] XSPI2 clock enable OK.\r\n");

	flash.InterfaceMode = BSP_XSPI_NOR_OPI_MODE;
	flash.TransferRate = BSP_XSPI_NOR_STR_TRANSFER;
	bsp_status = BSP_XSPI_NOR_Init(0U, &flash);
	if (bsp_status != BSP_ERROR_NONE)
	{
#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
		char msg[96];
		(void)DebugConsole_Snprintf(msg, sizeof(msg),
					   "[AI] BSP_XSPI_NOR_Init for provisioning failed: %ld\r\n",
					   (long)bsp_status);
		(void)DebugConsole_WriteString(msg);
#endif
		return false;
	}

	(void)DebugConsole_WriteString("[AI] XSPI2 memory bring-up OK.\r\n");
	return true;
}

static bool AppAI_ReconfigureXspi2ForRuntime(void)
{
	BSP_XSPI_NOR_Init_t flash = {0};
	RCC_PeriphCLKInitTypeDef periph_clk = {0};
	int32_t bsp_status = BSP_ERROR_NONE;

	/* Drop out of memory-mapped mode before changing the flash transfer rate.
	 * Clear the provisioning-mode guard so EnsureXspi2MemoryReady will
	 * re-initialize the peripheral into indirect/write mode if provisioning
	 * is needed again after this reconfigure. */
	app_ai_xspi2_initialized = false;
	app_ai_xspi2_mm_enabled = false;
#if APP_AI_ENABLE_XSPI2_VERBOSE_LOGS
	(void)DebugConsole_WriteString("[AI] xSPI2 runtime reconfigure: disable MM start.\r\n");
#endif
	(void)BSP_XSPI_NOR_DisableMemoryMappedMode(0U);
#if APP_AI_ENABLE_XSPI2_VERBOSE_LOGS
	(void)DebugConsole_WriteString("[AI] xSPI2 runtime reconfigure: disable MM OK.\r\n");
	(void)DebugConsole_WriteString("[AI] xSPI2 runtime reconfigure: deinit start.\r\n");
#endif
	(void)BSP_XSPI_NOR_DeInit(0U);
#if APP_AI_ENABLE_XSPI2_VERBOSE_LOGS
	(void)DebugConsole_WriteString("[AI] xSPI2 runtime reconfigure: deinit OK.\r\n");
#endif

	periph_clk.PeriphClockSelection = RCC_PERIPHCLK_XSPI2;
	periph_clk.Xspi2ClockSelection = RCC_XSPI2CLKSOURCE_IC3;
#if APP_AI_ENABLE_XSPI2_VERBOSE_LOGS
	(void)DebugConsole_WriteString(
		"[AI] xSPI2 runtime reconfigure: clock config start.\r\n");
#endif
	if (HAL_RCCEx_PeriphCLKConfig(&periph_clk) != HAL_OK)
	{
		DebugConsole_WriteString("[AI] xSPI2 runtime reconfigure: clock config failed.\r\n");
		return false;
	}
#if APP_AI_ENABLE_XSPI2_VERBOSE_LOGS
	(void)DebugConsole_WriteString(
		"[AI] xSPI2 runtime reconfigure: clock config OK.\r\n");
#endif

	/* Ensure the xSPI2 clock is enabled after configuration change.
	 * The deinit may have disabled the clock, and the new clock source
	 * won't be active until the peripheral clock enable is asserted. */
	XSPI_CLK_ENABLE();
#if APP_AI_ENABLE_XSPI2_VERBOSE_LOGS
	(void)DebugConsole_WriteString("[AI] xSPI2 runtime reconfigure: clock enable OK.\r\n");
#endif

	flash.InterfaceMode = BSP_XSPI_NOR_OPI_MODE;
	/* Keep the runtime in STR mode for now. The board already initializes and
	 * programs the xSPI2 NOR successfully in STR, while the DTR reconfigure
	 * path was failing during BSP_XSPI_NOR_Init() with -5. We only need the
	 * mapped window to be valid for the stage runtime. */
	flash.TransferRate = BSP_XSPI_NOR_STR_TRANSFER;
#if APP_AI_ENABLE_XSPI2_VERBOSE_LOGS
	(void)DebugConsole_WriteString("[AI] xSPI2 runtime reconfigure: init start.\r\n");
#endif
	bsp_status = BSP_XSPI_NOR_Init(0U, &flash);
	if (bsp_status != BSP_ERROR_NONE)
	{
		char msg[96];
		(void)DebugConsole_Snprintf(msg, sizeof(msg),
					   "[AI] xSPI2 runtime reconfigure: init failed: %ld\r\n",
					   (long)bsp_status);
		DebugConsole_WriteString(msg);
		return false;
	}
#if APP_AI_ENABLE_XSPI2_VERBOSE_LOGS
	(void)DebugConsole_WriteString("[AI] xSPI2 runtime reconfigure: init OK.\r\n");
#endif

	/* Leave the peripheral in indirect mode so callers that need to probe flash
	 * via BSP_XSPI_NOR_Read can do so immediately.  Callers that need the
	 * memory-mapped window must call BSP_XSPI_NOR_EnableMemoryMappedMode()
	 * themselves after this function returns. */
	return true;
}

static bool AppAI_Xspi2EnableMemoryMappedMode(void)
{
#if APP_AI_ENABLE_XSPI2_VERBOSE_LOGS
	(void)DebugConsole_WriteString("[AI] xSPI2 enable MM start.\r\n");
#endif
	if (BSP_XSPI_NOR_EnableMemoryMappedMode(0U) != BSP_ERROR_NONE)
	{
		app_ai_xspi2_mm_enabled = false;
		(void)DebugConsole_WriteString("[AI] xSPI2 enable MM failed.\r\n");
		return false;
	}
	app_ai_xspi2_mm_enabled = true;
#if APP_AI_ENABLE_XSPI2_VERBOSE_LOGS
	(void)DebugConsole_WriteString("[AI] xSPI2 enable MM OK.\r\n");
#endif
	return true;
}

static bool AppAI_Xspi2ReadFlashProbe(const uint32_t chip_base_offset,
									  const uint32_t flash_offset,
									  const uint8_t *expected_bytes, const size_t expected_length)
{
	uint8_t flash_bytes[APP_AI_XSPI2_PROBE_BYTES] = {0U};

	if ((expected_bytes == NULL) || (expected_length == 0U) || (expected_length > APP_AI_XSPI2_PROBE_BYTES))
	{
		return false;
	}

	if (BSP_XSPI_NOR_Read(0U, flash_bytes,
						  chip_base_offset + flash_offset,
						  (uint32_t)expected_length) != BSP_ERROR_NONE)
	{
		return false;
	}

	return (memcmp(flash_bytes, expected_bytes, expected_length) == 0);
}

static bool AppAI_Xspi2ReadMappedProbe(const uint32_t flash_offset,
									   const uint8_t *expected_bytes, const size_t expected_length)
{
	const uint8_t *const flash_ptr = (const uint8_t *)(APP_AI_XSPI2_MODEL_BASE_ADDR + flash_offset);

	if ((expected_bytes == NULL) || (expected_length == 0U) || (expected_length > APP_AI_XSPI2_PROBE_BYTES))
	{
		return false;
	}

	(void)mcu_cache_invalidate_range((uint32_t)flash_ptr,
									 (uint32_t)flash_ptr + (uint32_t)expected_length);

	return (memcmp(flash_ptr, expected_bytes, expected_length) == 0);
}

static bool AppAI_Xspi2ReadStageProbe(const AppAI_ModelStageSpec *stage,
									  const uint32_t flash_offset,
									  const uint8_t *expected_bytes,
									  const size_t expected_length)
{
	if ((stage == NULL) || (expected_bytes == NULL) || (expected_length == 0U) ||
		(expected_length > APP_AI_XSPI2_PROBE_BYTES))
	{
		return false;
	}

	/* After stage provisioning we can already be back in memory-mapped mode.
	 * In that state, compare directly through the mapped stage window. */
	if (app_ai_xspi2_mm_enabled)
	{
		const uint8_t *const flash_ptr =
			(const uint8_t *)(stage->xspi2_base_addr + flash_offset);
		(void)mcu_cache_invalidate_range((uint32_t)(uintptr_t)flash_ptr,
										 (uint32_t)((uintptr_t)flash_ptr + (uint32_t)expected_length));
		return (memcmp(flash_ptr, expected_bytes, expected_length) == 0);
	}

	return AppAI_Xspi2ReadFlashProbe(stage->xspi2_chip_offset, flash_offset,
									 expected_bytes, expected_length);
}


