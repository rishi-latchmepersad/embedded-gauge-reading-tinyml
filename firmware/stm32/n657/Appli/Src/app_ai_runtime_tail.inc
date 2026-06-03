/* This file is included by the parent translation unit. */

bool App_AI_Model_Init(void)
{
	if (app_ai_runtime_initialized)
	{
		return true;
	}

	(void)DebugConsole_WriteString("[AI] Model runtime init start.\r\n");
	if (!AppAI_EnsureNpuHardwareReady())
	{
		AppAI_LogInitFailure("NPU hardware");
		return false;
	}

	if (!AppAI_RuntimeInitStepwise())
	{
		AppAI_LogInitFailure("runtime init");
		return false;
	}

	app_ai_runtime_initialized = true;
#if APP_AI_ENABLE_INFERENCE_BURST_SMOOTHING
	AppAI_ResetInferenceBurstHistory();
#endif
	DebugConsole_Printf("[AI] Model runtime init OK.\r\n");
	DebugConsole_Printf("[AI][TIP_FOCUS] compiled, enable=%u\r\n", (unsigned)APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE);
	return true;
}

/* Log the live r9 value at a few dry-run checkpoints. The ST resize helper
 * uses r9 as its ai_array base, so this lets us pinpoint the first caller
 * that hands it a bad register value without disturbing the state. */
static void AppAI_LogR9(const char *label)
{
	uintptr_t r9_before = 0U;

	if (label == NULL)
	{
		return;
	}

	__asm volatile("mov %0, r9" : "=r"(r9_before));
	DebugConsole_Printf("[AI][DRY_RUN][R9] %s=%p\r\n",
						label, (const void *)r9_before);
}

#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
static void AppAI_LogInitFailure(const char *step)
{
	if (step != NULL)
	{
		DebugConsole_Printf("[AI] Model runtime init failed at %s.\r\n", step);
	}
	else
	{
		DebugConsole_Printf("[AI] Model runtime init failed.\r\n");
	}
}

static void AppAI_LogXspi2LoadFailure(const char *step, UINT fx_status,
									  int32_t bsp_status)
{
	DebugConsole_Printf("[AI] xSPI2 load failed at %s (fx=%lu bsp=%ld).\r\n",
						(step != NULL) ? step : "unknown",
						(unsigned long)fx_status,
						(long)bsp_status);
}

static void AppAI_LogXspi2ProgramChunkProgress(ULONG chunk_index,
											   ULONG flash_offset, ULONG chunk_size)
{
	DebugConsole_Printf(
		"[AI] xSPI2 program chunk %lu offset=0x%08lX size=%lu.\r\n",
		(unsigned long)chunk_index,
		(unsigned long)flash_offset,
		(unsigned long)chunk_size);
}

static void AppAI_LogXspi2FlashStatus(const char *label)
{
	/* Read both the generic BSP status and the raw security/status bytes so we
	 * can distinguish write protection, suspend state, and a plain readback bug. */
	uint8_t security_reg = 0U;
	uint8_t status_reg = 0U;
	int32_t bsp_status = BSP_XSPI_NOR_GetStatus(0U);
	int32_t security_status = MX25UM51245G_ReadSecurityRegister(&hxspi_nor[0U],
																Xspi_Nor_Ctx[0U].InterfaceMode, Xspi_Nor_Ctx[0U].TransferRate,
																&security_reg);
	int32_t status_reg_status = MX25UM51245G_ReadStatusRegister(&hxspi_nor[0U],
																Xspi_Nor_Ctx[0U].InterfaceMode, Xspi_Nor_Ctx[0U].TransferRate,
																&status_reg);
	char msg[192];

	(void)DebugConsole_Snprintf(
		msg,
		sizeof(msg),
		"[AI] %s flash status=%ld sec=%ld sec_reg=0x%02X sr=%ld sr_reg=0x%02X mode=%u rate=%u.\r\n",
		(label != NULL) ? label : "xSPI2",
		(long)bsp_status,
		(long)security_status,
		(unsigned int)security_reg,
		(long)status_reg_status,
		(unsigned int)status_reg,
		(unsigned int)Xspi_Nor_Ctx[0U].InterfaceMode,
		(unsigned int)Xspi_Nor_Ctx[0U].TransferRate);
	(void)DebugConsole_WriteString(msg);
}
#else
static void AppAI_LogInitFailure(const char *step)
{
	(void)step;
}

static void AppAI_LogXspi2LoadFailure(const char *step, UINT fx_status,
									  int32_t bsp_status)
{
	(void)step;
	(void)fx_status;
	(void)bsp_status;
}

static void AppAI_LogXspi2ProgramChunkProgress(ULONG chunk_index,
											   ULONG flash_offset, ULONG chunk_size)
{
	(void)chunk_index;
	(void)flash_offset;
	(void)chunk_size;
}

static void AppAI_LogXspi2FlashStatus(const char *label)
{
	(void)label;
}
#endif /* APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS */

#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
static void AppAI_LogXspi2FlashPrefix(void)
{
	uint8_t flash_bytes[APP_AI_XSPI2_PROBE_BYTES] = {0U};

	if (BSP_XSPI_NOR_Read(0U, flash_bytes, 0U,
						  APP_AI_XSPI2_PROBE_BYTES) != BSP_ERROR_NONE)
	{
		DebugConsole_Printf("[AI] xSPI2 prefix readback failed.\r\n");
		return;
	}

	AppAI_LogXspi2PrefixBytes("xSPI2 prefix readback:", flash_bytes);
}

static void AppAI_LogXspi2MappedScaleBytes(void)
{
	if (app_ai_xspi2_programmed_size < 4U)
	{
		return;
	}
	const uint8_t *const tail_ptr = (const uint8_t *)(APP_AI_XSPI2_MODEL_BASE_ADDR + app_ai_xspi2_programmed_size - 4U);

	(void)mcu_cache_invalidate_range((uint32_t)(uintptr_t)tail_ptr,
									 (uint32_t)((uintptr_t)tail_ptr + 4U));

	DebugConsole_Printf(
		"[AI] xSPI2 mapped tail bytes @%p = %02X %02X %02X %02X\r\n",
		(const void *)tail_ptr, tail_ptr[0], tail_ptr[1], tail_ptr[2],
		tail_ptr[3]);
}

static void AppAI_LogXspi2IndirectAndMappedPrefix(void)
{
	uint8_t indirect_bytes[APP_AI_XSPI2_PROBE_BYTES] = {0U};
	uint8_t mapped_bytes[APP_AI_XSPI2_PROBE_BYTES] = {0U};
	const int32_t disable_status = BSP_XSPI_NOR_DisableMemoryMappedMode(0U);

	if (disable_status != BSP_ERROR_NONE)
	{
		DebugConsole_Printf(
			"[AI] xSPI2 disable-mapped before compare returned %ld.\r\n",
			(long)disable_status);
	}

	if (BSP_XSPI_NOR_Read(0U, indirect_bytes, 0U,
						  APP_AI_XSPI2_PROBE_BYTES) != BSP_ERROR_NONE)
	{
		DebugConsole_Printf("[AI] xSPI2 indirect prefix read failed.\r\n");
	}
	else
	{
		AppAI_LogXspi2PrefixBytes("xSPI2 indirect prefix:", indirect_bytes);
	}

	if (BSP_XSPI_NOR_EnableMemoryMappedMode(0U) != BSP_ERROR_NONE)
	{
		DebugConsole_Printf("[AI] xSPI2 re-enable mapped compare failed.\r\n");
		return;
	}

	(void)mcu_cache_invalidate_range(APP_AI_XSPI2_MODEL_BASE_ADDR,
									 APP_AI_XSPI2_MODEL_BASE_ADDR + APP_AI_XSPI2_PROBE_BYTES);
	(void)memcpy(mapped_bytes, (const void *)APP_AI_XSPI2_MODEL_BASE_ADDR,
				 APP_AI_XSPI2_PROBE_BYTES);
	AppAI_LogXspi2PrefixBytes("xSPI2 mapped prefix:", mapped_bytes);
}

static void AppAI_LogFloatApprox(const char *label, float value)
{
	union
	{
		float f;
		uint32_t u;
	} bits = {
		.f = value};
	unsigned long magnitude_whole = 0U;
	unsigned long magnitude_frac = 0U;
	const char *sign = "";
	double abs_value = 0.0;

	if (label == NULL)
	{
		return;
	}

	if ((bits.u & 0x7F800000U) == 0x7F800000U)
	{
		if ((bits.u & 0x007FFFFFU) != 0U)
		{
			DebugConsole_Printf("%sNaN\r\n", label);
		}
		else if ((bits.u & 0x80000000U) != 0U)
		{
			DebugConsole_Printf("%s-Inf\r\n", label);
		}
		else
		{
			DebugConsole_Printf("%s+Inf\r\n", label);
		}
		return;
	}

	if ((bits.u & 0x80000000U) != 0U)
	{
		sign = "-";
	}

	abs_value = (bits.u & 0x80000000U) != 0U ? -(double)value : (double)value;
	if (abs_value < 0.0)
	{
		abs_value = 0.0;
	}

	magnitude_whole = (unsigned long)abs_value;
	magnitude_frac = (unsigned long)((abs_value - (double)magnitude_whole) * 1000000.0 + 0.5);
	if (magnitude_frac >= 1000000U)
	{
		magnitude_whole++;
		magnitude_frac -= 1000000U;
	}

	DebugConsole_Printf("%s%s%lu.%06lu\r\n", label, sign,
						magnitude_whole, magnitude_frac);
}
#endif /* APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS */

static float AppAI_TraceAndApplyInferenceCalibration(float raw_value)
{
	return AppInferenceCalibration_Apply(raw_value);
}

#if APP_AI_ENABLE_INFERENCE_BURST_SMOOTHING
/**
 * @brief Reset the tiny burst history used to stabilize the AI output.
 */
static void AppAI_ResetInferenceBurstHistory(void)
{
	(void)memset(app_ai_inference_burst_history, 0,
				 sizeof(app_ai_inference_burst_history));
	app_ai_inference_burst_history_count = 0U;
	app_ai_inference_burst_history_next_index = 0U;
}
#endif /* APP_AI_ENABLE_INFERENCE_BURST_SMOOTHING */

static bool AppAI_RuntimeInitStepwise(void)
{
	/* Use the same explicit ATON bring-up sequence that the working generated
	 * sample uses.  The lighter LL_ATON_RT_RuntimeInit() path proved too weak
	 * for the OBB stage because it did not leave the NPU interrupt chain in a
	 * state that reliably wakes the runtime from WFE. */
	(void)DebugConsole_WriteString("[AI] ATON runtime init start.\r\n");
	LL_ATON_RT_SetRuntimeCallback(NULL);
	if (LL_ATON_Init() != LL_ATON_OK)
	{
		(void)DebugConsole_WriteString(
			"[AI] ATON runtime init failed in LL_ATON_Init().\r\n");
		return false;
	}

	/* The ATON platform macros use a scratch `t` local exactly like the vendor
	 * runtime sample, so keep that local in scope before invoking them. */
	uint32_t t = 0U;

	ATON_DISABLE_CLR_CONFCLR(INTCTRL, 0);
	ATON_INTCTRL_STD_INTORMSK_SET(ATON_STRENG_INT_MASK(ATON_STRENG_NUM, 0, 0));
	ATON_INTCTRL_STD_INTANDMSK_SET(0xFFFFFFFFU);
#if (ATON_INT_NR > 32)
	ATON_INTCTRL_STD_INTORMSK_H_SET(0xFFFFFFFFU);
	ATON_INTCTRL_STD_INTANDMSK_H_SET(0xFFFFFFFFU);
#endif
	ATON_ENABLE(INTCTRL, 0);

	LL_ATON_OSAL_INIT();
	LL_ATON_OSAL_DISABLE_IRQ(0U);
	LL_ATON_OSAL_DISABLE_IRQ(1U);
	LL_ATON_OSAL_DISABLE_IRQ(2U);
	LL_ATON_OSAL_DISABLE_IRQ(3U);
	LL_ATON_OSAL_INSTALL_IRQ(ATON_STD_IRQ_LINE, ATON_STD_IRQHandler);
	LL_ATON_OSAL_ENABLE_IRQ(ATON_STD_IRQ_LINE);
	(void)DebugConsole_WriteString("[AI] ATON runtime init OK.\r\n");
	return true;
}

#if !APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
static void AppAI_LogXspi2FlashPrefix(void)
{
}

static void AppAI_LogXspi2MappedScaleBytes(void)
{
}

static void AppAI_LogXspi2IndirectAndMappedPrefix(void)
{
}

static void AppAI_LogFloatApprox(const char *label, float value)
{
	(void)label;
	(void)value;
}
#endif /* !APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS */

static void AppAI_ConfigureNpuAccessControl(void)
{
	RIMC_MasterConfig_t npu_master = {0};

	/* Mirror the ST NPU examples so the runtime can access the cache and NPU
	 * control path with the expected secure/privileged attributes. */
	__HAL_RCC_RIFSC_CLK_ENABLE();
	__HAL_RCC_RISAF_CLK_ENABLE();
	__HAL_RCC_CACHEAXIRAM_MEM_CLK_ENABLE();

	/* The secure NPU validation app programs the NPU as secure + privileged.
	 * That matches our secure-side runtime more closely than the earlier NSEC
	 * experiment, and it avoids blocking the internal ATON control window. */
	npu_master.MasterCID = RIF_CID_1;
	npu_master.SecPriv = RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV;
	HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_NPU, &npu_master);
	HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_NPU,
										  RIF_ATTRIBUTE_PRIV | RIF_ATTRIBUTE_SEC);

#if defined(RIF_RCC_PERIPH_INDEX_CACHEAXIRAM)
	HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RCC_PERIPH_INDEX_CACHEAXIRAM,
										  RIF_ATTRIBUTE_PRIV | RIF_ATTRIBUTE_SEC);
#endif

#if defined(RIF_RCC_PERIPH_INDEX_CACHECONFIG)
	HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RCC_PERIPH_INDEX_CACHECONFIG,
										  RIF_ATTRIBUTE_PRIV | RIF_ATTRIBUTE_SEC);
#endif
}

static void AppAI_EnableNpuMemoryAndCaches(void)
{
	/* Mirror the ST NPU examples so the memory fabric is actually usable by
	 * the runtime before the first ATON init call runs.  Enable ALL AXI SRAM
	 * banks that the ATON virtual memory pool (pool 11, spanning
	 * AXISRAM2-6) might touch.  Without AXISRAM2 the NPU's AXI master can
	 * fault at 0x00018800 when the runtime maps an activation buffer into
	 * the unmapped region. */
	RCC->MEMENR |= RCC_MEMENR_AXISRAM2EN | RCC_MEMENR_AXISRAM3EN | RCC_MEMENR_AXISRAM4EN | RCC_MEMENR_AXISRAM5EN | RCC_MEMENR_AXISRAM6EN | RCC_MEMENR_CACHEAXIRAMEN;

	RAMCFG_SRAM2_AXI->CR &= ~RAMCFG_CR_SRAMSD;
	RAMCFG_SRAM3_AXI->CR &= ~RAMCFG_CR_SRAMSD;
	RAMCFG_SRAM4_AXI->CR &= ~RAMCFG_CR_SRAMSD;
	RAMCFG_SRAM5_AXI->CR &= ~RAMCFG_CR_SRAMSD;
	RAMCFG_SRAM6_AXI->CR &= ~RAMCFG_CR_SRAMSD;

	MEMSYSCTL->MSCR |= MEMSYSCTL_MSCR_DCACTIVE_Msk | MEMSYSCTL_MSCR_ICACTIVE_Msk;
}

static uint32_t AppAI_GetRisafMaxAddr(RISAF_TypeDef *risaf)
{
	uint32_t max_addr = 0U;

	if ((risaf == RISAF1_S) || (risaf == RISAF1_NS))
	{
		max_addr = RISAF1_LIMIT_ADDRESS_SPACE_SIZE;
	}
	else if ((risaf == RISAF2_S) || (risaf == RISAF2_NS))
	{
		max_addr = RISAF2_LIMIT_ADDRESS_SPACE_SIZE;
	}
	else if ((risaf == RISAF3_S) || (risaf == RISAF3_NS))
	{
		max_addr = RISAF3_LIMIT_ADDRESS_SPACE_SIZE;
	}
	else if ((risaf == RISAF4_S) || (risaf == RISAF4_NS))
	{
		max_addr = RISAF4_LIMIT_ADDRESS_SPACE_SIZE;
	}
	else if ((risaf == RISAF5_S) || (risaf == RISAF5_NS))
	{
		max_addr = RISAF5_LIMIT_ADDRESS_SPACE_SIZE;
	}
	else if ((risaf == RISAF6_S) || (risaf == RISAF6_NS))
	{
		max_addr = RISAF6_LIMIT_ADDRESS_SPACE_SIZE;
	}
	else if ((risaf == RISAF7_S) || (risaf == RISAF7_NS))
	{
		max_addr = RISAF7_LIMIT_ADDRESS_SPACE_SIZE;
	}
	else if ((risaf == RISAF8_S) || (risaf == RISAF8_NS))
	{
		max_addr = RISAF8_LIMIT_ADDRESS_SPACE_SIZE;
	}
	else if ((risaf == RISAF9_S) || (risaf == RISAF9_NS))
	{
		max_addr = RISAF9_LIMIT_ADDRESS_SPACE_SIZE;
	}
	else if ((risaf == RISAF11_S) || (risaf == RISAF11_NS))
	{
		max_addr = RISAF11_LIMIT_ADDRESS_SPACE_SIZE;
	}
	else if ((risaf == RISAF12_S) || (risaf == RISAF12_NS))
	{
		max_addr = RISAF12_LIMIT_ADDRESS_SPACE_SIZE;
	}
	else if ((risaf == RISAF13_S) || (risaf == RISAF13_NS))
	{
		max_addr = RISAF13_LIMIT_ADDRESS_SPACE_SIZE;
	}
	else if ((risaf == RISAF14_S) || (risaf == RISAF14_NS))
	{
		max_addr = RISAF14_LIMIT_ADDRESS_SPACE_SIZE;
	}
	else if ((risaf == RISAF15_S) || (risaf == RISAF15_NS))
	{
		max_addr = RISAF15_LIMIT_ADDRESS_SPACE_SIZE;
	}
	else if ((risaf == RISAF21_S) || (risaf == RISAF21_NS))
	{
		max_addr = RISAF21_LIMIT_ADDRESS_SPACE_SIZE;
	}
	else if ((risaf == RISAF22_S) || (risaf == RISAF22_NS))
	{
		max_addr = RISAF22_LIMIT_ADDRESS_SPACE_SIZE;
	}
	else if ((risaf == RISAF23_S) || (risaf == RISAF23_NS))
	{
		max_addr = RISAF23_LIMIT_ADDRESS_SPACE_SIZE;
	}

	return max_addr;
}

static void AppAI_SetRisafDefault(RISAF_TypeDef *risaf)
{
	RISAF_BaseRegionConfig_t risaf_conf;
	RISAF_TypeDef *const risaf_hw = (risaf == RISAF12_S) ? RISAF12_NS : risaf;

	risaf_conf.StartAddress = 0x0U;
	risaf_conf.EndAddress = AppAI_GetRisafMaxAddr(risaf_hw);
	risaf_conf.Filtering = RISAF_FILTER_ENABLE;
	risaf_conf.PrivWhitelist = RIF_CID_NONE;
	risaf_conf.ReadWhitelist = RIF_CID_MASK;
	risaf_conf.WriteWhitelist = RIF_CID_MASK;

	risaf_conf.Secure = RIF_ATTRIBUTE_SEC;
	if (risaf == RISAF12_S)
	{
	}
	HAL_RIF_RISAF_ConfigBaseRegion(risaf_hw, 0U, &risaf_conf);
	if (risaf == RISAF12_S)
	{
	}

	risaf_conf.Secure = RIF_ATTRIBUTE_NSEC;
	if (risaf == RISAF12_S)
	{
	}
	HAL_RIF_RISAF_ConfigBaseRegion(risaf_hw, 1U, &risaf_conf);
	if (risaf == RISAF12_S)
	{
	}
}

static void AppAI_ConfigureNpuRisafDefaults(void)
{
	/* Keep the default ST security model so the AI runtime can reach the memory
	 * and fabric regions it expects during LL_ATON startup. */
	__HAL_RCC_RIFSC_CLK_ENABLE();
	__HAL_RCC_RISAF_CLK_ENABLE();

	AppAI_SetRisafDefault(RISAF2_S);

	AppAI_SetRisafDefault(RISAF3_S);

	AppAI_SetRisafDefault(RISAF4_S);

	AppAI_SetRisafDefault(RISAF5_S);

	AppAI_SetRisafDefault(RISAF6_S);

	AppAI_SetRisafDefault(RISAF7_S);

	/* RISAF12 programming currently stalls on this board even though the
	 * xSPI2 memory-mapped window is already enabled. Leave it untouched for
	 * now so we can validate whether the runtime can proceed without it. */

#if defined(RISAF8_S) && defined(RISAF15_S)

	AppAI_SetRisafDefault(RISAF8_S);
	AppAI_SetRisafDefault(RISAF15_S);

#endif
}

bool App_AI_RunDryInferenceFromYuv422(const uint8_t *frame_bytes,
									  size_t frame_size)
{
	const uint8_t *safe_frame_bytes = frame_bytes;
	const LL_Buffer_InfoTypeDef *obb_output_info = NULL;
	AppAI_ObbBox obb_box = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	AppAI_SourceCrop full_frame_crop = {0U, 0U,
										(size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS,
										(size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS};
	const AppGaugeGeometry_Crop_t fixed_training_geometry =
		AppGaugeGeometry_TrainingCrop(
			(size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS,
			(size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS);
	const AppAI_SourceCrop fixed_training_crop = {
		fixed_training_geometry.x_min,
		fixed_training_geometry.y_min,
		fixed_training_geometry.width,
		fixed_training_geometry.height,
	};
	AppAI_SourceCrop scalar_crop = fixed_training_crop;
	bool scalar_crop_from_obb = false;
	float obb_output_value = 0.0f;

	(void)DebugConsole_WriteString("[AI] Dry-run entry.\r\n");
	AppAI_LogR9("entry");
	if ((frame_bytes == NULL) || (frame_size < (size_t)APP_AI_CAPTURE_FRAME_BYTES))
	{
		(void)DebugConsole_WriteString("[AI] Dry-run entry aborted: invalid frame buffer.\r\n");
		return false;
	}

	memcpy(app_ai_dry_run_frame_scratch, frame_bytes, (size_t)APP_AI_CAPTURE_FRAME_BYTES);
	safe_frame_bytes = app_ai_dry_run_frame_scratch;

	if (!App_AI_Model_Init())
	{
		(void)DebugConsole_WriteString("[AI] Dry-run entry aborted during model init.\r\n");
		return false;
	}

	if (!AppCenterDetector_Init())
	{
		(void)DebugConsole_WriteString("[AI] Dry-run entry aborted during center-detector init.\r\n");
		return false;
	}
	AppAI_LogR9("post_model_init");

	/* -----------------------------------------------------------------------
	 *  Pipeline: OBB → (optional luma refinement) → center-detector crop
	 *
	 *  Use fully independent copies at each stage so the crop that reaches
	 *  the center detector cannot be corrupted by fallback logic in the
	 *  OBB decode or any other path.
	 * ----------------------------------------------------------------------- */
	AppAI_SourceCrop obb_crop = fixed_training_crop;  /* set iff obb produces a valid box */
	bool obb_crop_valid = false;

#if APP_AI_ENABLE_OBB_STAGE
	{
		const bool obb_ok = AppAI_RunStageInference(
			&app_ai_obb_stage, safe_frame_bytes, frame_size,
			&full_frame_crop, &obb_output_info, &obb_output_value);

		/* Decode the OBB output into obb_crop (may still fall back to fixed
		 * training crop inside DecodeObbCropBox — that is OK). */
		if (obb_ok &&
			AppAI_DecodeObbCropBox(obb_output_info, &obb_crop, &obb_box))
		{
			obb_crop_valid = true;
			AppAI_LogObbResult(obb_output_info, &obb_box);
			DebugConsole_Printf(
				"[AI] OBB crop: x=%lu y=%lu w=%lu h=%lu\r\n",
				(unsigned long)obb_crop.x_min,
				(unsigned long)obb_crop.y_min,
				(unsigned long)obb_crop.width,
				(unsigned long)obb_crop.height);
		}
		else
		{
			obb_crop_valid = false;
			DebugConsole_WriteString(
				"[AI] OBB stage failed; using fixed training crop.\r\n");
		}
	}
#else
	(void)obb_output_info;
	(void)obb_output_value;
	obb_crop_valid = false;
	DebugConsole_WriteString(
		"[AI] OBB stage disabled; using fixed training crop.\r\n");
#endif

	/* -----------------------------------------------------------------------
	 *  Training-crop-sized rectangle centred on OBB for centre-detector crop
	 *
	 *  Uses the OBB centre to shift the training-crop-sized window so the
	 *  centre-detector model sees the same gauge framing it was trained on,
	 *  while still tracking the OBB-detected gauge position.
	 * ----------------------------------------------------------------------- */
	AppAI_SourceCrop cd_crop;

	if (obb_crop_valid)
	{
		const AppGaugeGeometry_Crop_t tc =
			AppGaugeGeometry_TrainingCrop(
				(size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS,
				(size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS);

		const float obb_cx = obb_box.center_x
			* (float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
		const float obb_cy = obb_box.center_y
			* (float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;

		float cd_x_min_f = obb_cx - (float)tc.width * 0.5f;
		float cd_y_min_f = obb_cy - (float)tc.height * 0.5f;

		if (cd_x_min_f < 0.0f)
		{
			cd_x_min_f = 0.0f;
		}
		if (cd_y_min_f < 0.0f)
		{
			cd_y_min_f = 0.0f;
		}
		if ((cd_x_min_f + (float)tc.width) > (float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS)
		{
			cd_x_min_f = (float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS - (float)tc.width;
		}
		if ((cd_y_min_f + (float)tc.height) > (float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS)
		{
			cd_y_min_f = (float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS - (float)tc.height;
		}

		cd_crop.x_min = (size_t)(cd_x_min_f + 0.5f);
		cd_crop.y_min = (size_t)(cd_y_min_f + 0.5f);
		cd_crop.width  = tc.width;
		cd_crop.height = tc.height;
	}
	else
	{
		/* OBB did not produce a valid box → fall back to fixed training crop. */
		cd_crop = fixed_training_crop;
	}

	/* Update scalar_crop so any downstream reader that still references the
	 * original name sees the same crop that reaches the centre detector. */
	scalar_crop = cd_crop;
	scalar_crop_from_obb = obb_crop_valid;

	DebugConsole_Printf(
		"[AI] Center detector crop: x=%lu y=%lu w=%lu h=%lu%s\r\n",
		(unsigned long)cd_crop.x_min,
		(unsigned long)cd_crop.y_min,
		(unsigned long)cd_crop.width,
		(unsigned long)cd_crop.height,
		obb_crop_valid ? " (obb-centered)" : " (fixed-training)");

	{
		/* Dial radius comes from a fixed gauge-design ratio, not from
		 * the OBB box size.  The OBB box width drifts with lighting
		 * (darker captures → wider box) and would shift the polar-vote
		 * scan ring away from the needle.  The baseline already uses
		 * the same frame-ratio approach via the training crop. */
		const float dial_radius_px =
			APP_GAUGE_INNER_DIAL_RADIUS_FRAME_RATIO
			* (float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
		{
			const long dr10 = lroundf(dial_radius_px * 10.0f);
			const long bw10000 = lroundf(obb_box.box_w * 10000.0f);
			const long bcx10000 = lroundf(obb_box.center_x * 10000.0f);
			const long bcy10000 = lroundf(obb_box.center_y * 10000.0f);
			DebugConsole_Printf(
				"[DBG] AI dial_radius=%ld.%01ldpx obb_valid=%d "
				"obb_box_w=%ld.%04ld obb_cx=%ld.%04ld obb_cy=%ld.%04ld "
				"fixed_tc=(%lu,%lu,%lu,%lu) "
				"cd_crop=(%lu,%lu,%lu,%lu)\r\n",
				labs(dr10) / 10L, labs(dr10) % 10L,
				obb_crop_valid ? 1 : 0,
				labs(bw10000) / 10000L, labs(bw10000) % 10000L,
				labs(bcx10000) / 10000L, labs(bcx10000) % 10000L,
				labs(bcy10000) / 10000L, labs(bcy10000) % 10000L,
				(unsigned long)fixed_training_geometry.x_min,
				(unsigned long)fixed_training_geometry.y_min,
				(unsigned long)fixed_training_geometry.width,
				(unsigned long)fixed_training_geometry.height,
				(unsigned long)cd_crop.x_min,
				(unsigned long)cd_crop.y_min,
				(unsigned long)cd_crop.width,
				(unsigned long)cd_crop.height);
		}

	/* Primary: rim-vote centre search (same algorithm as baseline's
	 * AppBaselineRuntime_EstimateDialCenterFromRimVotes).  Finds the
	 * circular gauge rim by edge alignment, giving the same accuracy
	 * as the baseline rim-geometry seed.  OBB pivot is the fallback. */
	float override_cx;
	float override_cy;
	if (obb_crop_valid)
	{
		size_t rim_cx = 0U;
		size_t rim_cy = 0U;
		float rim_quality = 0.0f;
		const bool rim_ok =
			AppBaselineRuntime_EstimateDialCenterFromRimVotes(
				safe_frame_bytes, frame_size,
				APP_AI_CAPTURE_FRAME_WIDTH_PIXELS,
				APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS,
				0U, 0U,
				APP_AI_CAPTURE_FRAME_WIDTH_PIXELS,
				APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS,
				dial_radius_px,
				&rim_cx, &rim_cy, &rim_quality);
		if (rim_ok)
		{
			override_cx = (float)rim_cx;
			override_cy = (float)rim_cy;
			{
				const long cx10 = lroundf(override_cx * 10.0f);
				const long cy10 = lroundf(override_cy * 10.0f);
				const long rq100 = lroundf(rim_quality * 100.0f);
				DebugConsole_Printf(
					"[AI] Rim-vote centre: (%ld.%01ld,%ld.%01ld) "
					"quality=%ld.%02ld\r\n",
					labs(cx10) / 10L, labs(cx10) % 10L,
					labs(cy10) / 10L, labs(cy10) % 10L,
					labs(rq100) / 100L, labs(rq100) % 100L);
			}
		}
		else
		{
			/* Fallback: EMA-smoothed OBB pivot. */
			const float raw_cx =
				obb_box.center_x * (float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
			const float raw_cy =
				obb_box.center_y * (float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
			if (app_ai_smoothed_obb_cx < 0.0f)
			{
				app_ai_smoothed_obb_cx = raw_cx;
				app_ai_smoothed_obb_cy = raw_cy;
			}
			else
			{
				app_ai_smoothed_obb_cx =
					(APP_AI_OBB_CENTER_EMA_ALPHA * raw_cx) +
					((1.0f - APP_AI_OBB_CENTER_EMA_ALPHA) * app_ai_smoothed_obb_cx);
				app_ai_smoothed_obb_cy =
					(APP_AI_OBB_CENTER_EMA_ALPHA * raw_cy) +
					((1.0f - APP_AI_OBB_CENTER_EMA_ALPHA) * app_ai_smoothed_obb_cy);
			}
			const float x_off =
				APP_GAUGE_OBB_PIVOT_X_OFFSET_RATIO
				* (float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
			const float y_off =
				APP_GAUGE_OBB_PIVOT_Y_OFFSET_RATIO
				* (float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
			override_cx = app_ai_smoothed_obb_cx + x_off;
			override_cy = app_ai_smoothed_obb_cy + y_off;
			{
				const long cx10 = lroundf(override_cx * 10.0f);
				const long cy10 = lroundf(override_cy * 10.0f);
				DebugConsole_Printf(
					"[AI] Rim-vote failed; OBB pivot fallback: "
					"(%ld.%01ld,%ld.%01ld)\r\n",
					labs(cx10) / 10L, labs(cx10) % 10L,
					labs(cy10) / 10L, labs(cy10) % 10L);
			}
		}
	}
	else
	{
		/* No OBB → fixed training-crop centre. */
		override_cx =
			(float)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS
			* APP_GAUGE_INNER_DIAL_CENTER_X_RATIO;
		override_cy =
			(float)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS
			* APP_GAUGE_INNER_DIAL_CENTER_Y_RATIO;
	}
		AppCenterDetector_Result_t cd_result = {0};

		if (AppCenterDetector_Run(safe_frame_bytes, frame_size,
				cd_crop.x_min, cd_crop.y_min,
				cd_crop.width, cd_crop.height,
				dial_radius_px,
				APP_AI_CAPTURE_FRAME_WIDTH_PIXELS,
				APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS,
				&cd_result,
				override_cx, override_cy))
		{
			if (cd_result.valid)
			{
				const long center_x_px = lroundf(cd_result.center_x);
				const long center_y_px = lroundf(cd_result.center_y);
				const long angle_deg_hundredths =
					lroundf((float)(cd_result.needle_angle_rad * 180.0 / 3.14159) * 100.0f);
				const long temp_tenths = lroundf(cd_result.temperature_c * 10.0f);
				const long conf_hundredths = lroundf(cd_result.confidence * 100.0f);
				const long angle_abs_hundredths = labs(angle_deg_hundredths);
				const long temp_abs_tenths = labs(temp_tenths);
				const long conf_abs_hundredths = labs(conf_hundredths);

				app_ai_last_inference_value = cd_result.temperature_c;
				app_ai_last_inference_valid = true;
				DebugConsole_Printf(
					"[AI] Center detector: center=(%ld,%ld) "
					"angle=%s%ld.%02lddeg temp=%s%ld.%01ldC conf=%s%ld.%02ld\r\n",
					center_x_px,
					center_y_px,
					(angle_deg_hundredths < 0L) ? "-" : "",
					angle_abs_hundredths / 100L,
					angle_abs_hundredths % 100L,
					(temp_tenths < 0L) ? "-" : "",
					temp_abs_tenths / 10L,
					temp_abs_tenths % 10L,
					(conf_hundredths < 0L) ? "-" : "",
					conf_abs_hundredths / 100L,
					conf_abs_hundredths % 100L);
				(void)INA219_LogReading("AI-DONE");
				Metrics_EndInference("AI", cd_result.temperature_c);
			}
			else
			{
				DebugConsole_WriteString(
					"[AI] Center detector: polar vote failed (no clear needle).\r\n");
				app_ai_last_inference_valid = false;
			Metrics_EndInference("AI", NAN);
		}
	}
	else
	{
		DebugConsole_WriteString(
			"[AI] Center detector pipeline failed.\r\n");
		app_ai_last_inference_valid = false;
		Metrics_EndInference("AI", NAN);
		}
	}

	return true;
}
/* USER CODE END 0 */

/* USER CODE BEGIN 1 */

static const LL_Buffer_InfoTypeDef *AppAI_GetInputBufferInfo(void)
{
	const LL_Buffer_InfoTypeDef *input_info =
		NN_Instance_scalar_full_finetune_from_best_piecewise_calibrated_int8.network
			->input_buffers_info();

	if ((input_info == NULL) || (input_info->name == NULL))
	{
		return NULL;
	}

	return input_info;
}

static const LL_Buffer_InfoTypeDef *AppAI_GetOutputBufferInfo(void)
{
	const LL_Buffer_InfoTypeDef *output_info =
		NN_Instance_scalar_full_finetune_from_best_piecewise_calibrated_int8.network
			->output_buffers_info();

	if ((output_info == NULL) || (output_info->name == NULL))
	{
		return NULL;
	}

	return output_info;
}

static const LL_Buffer_InfoTypeDef *AppAI_FindBufferInfoByName(
	const LL_Buffer_InfoTypeDef *buffer_list, const char *name)
{
	if ((buffer_list == NULL) || (name == NULL))
	{
		return NULL;
	}

	for (const LL_Buffer_InfoTypeDef *entry = buffer_list; entry->name != NULL;
		 ++entry)
	{
		if (strcmp(entry->name, name) == 0)
		{
			return entry;
		}
	}

	return NULL;
}

static const LL_Buffer_InfoTypeDef *AppAI_FindFirstBufferInfoByNames(
	const LL_Buffer_InfoTypeDef *buffer_list, const char *const *names,
	size_t name_count)
{
	if ((buffer_list == NULL) || (names == NULL) || (name_count == 0U))
	{
		return NULL;
	}

	for (size_t index = 0U; index < name_count; ++index)
	{
		const LL_Buffer_InfoTypeDef *buffer_info = AppAI_FindBufferInfoByName(
			buffer_list, names[index]);

		if (buffer_info != NULL)
		{
			return buffer_info;
		}
	}

	return NULL;
}

static void AppAI_LogScalarInternalOutputProbe(
	const AppAI_ModelStageSpec *stage,
	const LL_Buffer_InfoTypeDef *stage_output_info)
{
	(void)stage;
	(void)stage_output_info;
}

#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
static void AppAI_LogInferenceResult(
	const LL_Buffer_InfoTypeDef *output_buffer_info)
{
	const LL_Buffer_InfoTypeDef *internal_buffers = NULL;
	const LL_Buffer_InfoTypeDef *quantize_output_info = NULL;
	const LL_Buffer_InfoTypeDef *sub_output_info = NULL;
	const LL_Buffer_InfoTypeDef *conv1_output_info = NULL;
	const LL_Buffer_InfoTypeDef *raw_output_info = NULL;
	const LL_Buffer_InfoTypeDef *scale_info = NULL;
	const LL_Buffer_InfoTypeDef *zero_point_info = NULL;
	union
	{
		float f;
		uint32_t u;
	} output_bits = {
		.f = 0.0f};
	float output_value = 0.0f;
	float head_scale = 1.0f;
	float output_dequant_scale = 1.0f;
	float head_dequant_value = 0.0f;
	int8_t raw_output_value = 0;
	int8_t head_zero_point = 0;
	int8_t output_zero_point = 0;
	static const char *const quantize_output_names[] = {
		"Sub_24_out_0",
		"Conv2D_25_zero_off_out_25",
	};
	static const char *const sub_output_names[] = {
		"Conv2D_25_zero_off_out_25",
		"Conv2D_30_zero_off_out_34",
	};
	static const char *const conv1_output_names[] = {
		"Conv2D_34_zero_off_out_43",
		"Conv2D_42_zero_off_out_61",
		"Conv2D_46_zero_off_out_70",
	};
	static const char *const raw_output_names[] = {
		"Quantize_261_out_0",
		"Quantize_390_out_0",
		"Gemm_271_out_0",
	};
	static const char *const scale_names[] = {
		"Dequantize_319_x_scale",
	};
	static const char *const zero_point_names[] = {
		"Dequantize_319_x_zero_point",
	};

	if (output_buffer_info == NULL)
	{
		DebugConsole_Printf("[AI] Inference failed: no output buffer.\r\n");
		return;
	}

	DebugConsole_Printf(
		"[AI] Output buffer meta: name=%s addr=%p len=%lu\r\n",
		(output_buffer_info->name != NULL) ? output_buffer_info->name : "(unnamed)",
		LL_Buffer_addr_start(output_buffer_info),
		(unsigned long)LL_Buffer_len(output_buffer_info));

	(void)memcpy(&output_bits.u, LL_Buffer_addr_start(output_buffer_info),
				 sizeof(output_bits.u));
	output_value = output_bits.f;

		internal_buffers =
		LL_ATON_Internal_Buffers_Info(
			&NN_Instance_scalar_full_finetune_from_best_piecewise_calibrated_int8);
	quantize_output_info = AppAI_FindFirstBufferInfoByNames(internal_buffers,
															quantize_output_names, sizeof(quantize_output_names) / sizeof(quantize_output_names[0]));
	sub_output_info = AppAI_FindFirstBufferInfoByNames(internal_buffers,
													   sub_output_names, sizeof(sub_output_names) / sizeof(sub_output_names[0]));
	conv1_output_info = AppAI_FindFirstBufferInfoByNames(internal_buffers,
														 conv1_output_names, sizeof(conv1_output_names) / sizeof(conv1_output_names[0]));
	raw_output_info = AppAI_FindFirstBufferInfoByNames(internal_buffers,
													   raw_output_names, sizeof(raw_output_names) / sizeof(raw_output_names[0]));
	scale_info = AppAI_FindFirstBufferInfoByNames(internal_buffers,
												  scale_names, sizeof(scale_names) / sizeof(scale_names[0]));
	zero_point_info = AppAI_FindFirstBufferInfoByNames(internal_buffers,
													   zero_point_names, sizeof(zero_point_names) / sizeof(zero_point_names[0]));

	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
	{
		AppAI_LogBufferInfoAndSignature("input tensor", quantize_output_info);
		AppAI_LogBufferInfoAndSignature("preprocess output", sub_output_info);
		AppAI_LogBufferInfoAndSignature("first conv", conv1_output_info);
		AppAI_LogBufferInfoAndSignature("raw head", raw_output_info);
		AppAI_LogBufferInfoAndSignature("network output", output_buffer_info);
	}

	/* Dump the first 16 raw bytes of both the head output and the final network
	 * output so we can tell whether the inference engine wrote anything at all
	 * and whether the int8 value being read is actually at offset 0. */
	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS && (raw_output_info != NULL) && (LL_Buffer_addr_start(raw_output_info) != NULL))
	{
		const uint8_t *p = (const uint8_t *)LL_Buffer_addr_start(raw_output_info);
		const size_t n = LL_Buffer_len(raw_output_info);
		DebugConsole_Printf(
			"[AI] Scalar raw-head bytes (len=%lu): "
			"[%02X %02X %02X %02X %02X %02X %02X %02X "
			"%02X %02X %02X %02X %02X %02X %02X %02X]\r\n",
			(unsigned long)n,
			(n > 0U) ? p[0] : 0U, (n > 1U) ? p[1] : 0U,
			(n > 2U) ? p[2] : 0U, (n > 3U) ? p[3] : 0U,
			(n > 4U) ? p[4] : 0U, (n > 5U) ? p[5] : 0U,
			(n > 6U) ? p[6] : 0U, (n > 7U) ? p[7] : 0U,
			(n > 8U) ? p[8] : 0U, (n > 9U) ? p[9] : 0U,
			(n > 10U) ? p[10] : 0U, (n > 11U) ? p[11] : 0U,
			(n > 12U) ? p[12] : 0U, (n > 13U) ? p[13] : 0U,
			(n > 14U) ? p[14] : 0U, (n > 15U) ? p[15] : 0U);
	}
	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS && (output_buffer_info != NULL) && (LL_Buffer_addr_start(output_buffer_info) != NULL))
	{
		const uint8_t *p = (const uint8_t *)LL_Buffer_addr_start(output_buffer_info);
		const size_t n = LL_Buffer_len(output_buffer_info);
		DebugConsole_Printf(
			"[AI] Scalar network-output bytes (len=%lu): "
			"[%02X %02X %02X %02X %02X %02X %02X %02X "
			"%02X %02X %02X %02X %02X %02X %02X %02X]\r\n",
			(unsigned long)n,
			(n > 0U) ? p[0] : 0U, (n > 1U) ? p[1] : 0U,
			(n > 2U) ? p[2] : 0U, (n > 3U) ? p[3] : 0U,
			(n > 4U) ? p[4] : 0U, (n > 5U) ? p[5] : 0U,
			(n > 6U) ? p[6] : 0U, (n > 7U) ? p[7] : 0U,
			(n > 8U) ? p[8] : 0U, (n > 9U) ? p[9] : 0U,
			(n > 10U) ? p[10] : 0U, (n > 11U) ? p[11] : 0U,
			(n > 12U) ? p[12] : 0U, (n > 13U) ? p[13] : 0U,
			(n > 14U) ? p[14] : 0U, (n > 15U) ? p[15] : 0U);
	}

	if ((raw_output_info != NULL) && (LL_Buffer_addr_start(raw_output_info) != NULL))
	{
		raw_output_value = *(const int8_t *)LL_Buffer_addr_start(raw_output_info);
	}

	{
		char raw_output_line[64] = {0};

		(void)DebugConsole_Snprintf(raw_output_line, sizeof(raw_output_line),
					   "[AI] Raw output int8: %d\r\n", (int)raw_output_value);
		DebugConsole_WriteString(raw_output_line);
	}

	DebugConsole_Printf(
		"[AI] Raw tensor meta: name=%s addr=%p len=%lu\r\n",
		(raw_output_info != NULL) ? raw_output_info->name : "(missing)",
		(raw_output_info != NULL) ? (void *)LL_Buffer_addr_start(raw_output_info) : NULL,
		(raw_output_info != NULL) ? (unsigned long)LL_Buffer_len(raw_output_info) : 0UL);

	if ((scale_info != NULL) && (LL_Buffer_addr_start(scale_info) != NULL))
	{
		(void)memcpy(&output_dequant_scale, LL_Buffer_addr_start(scale_info),
					 sizeof(output_dequant_scale));
	}

	if ((zero_point_info != NULL) && (LL_Buffer_addr_start(zero_point_info) != NULL))
	{
		output_zero_point = *(const int8_t *)LL_Buffer_addr_start(
			zero_point_info);
	}

	if (raw_output_info != NULL)
	{
		const void *raw_output_addr = LL_Buffer_addr_start(raw_output_info);

		if (raw_output_addr != NULL)
		{
			raw_output_value = *(const int8_t *)raw_output_addr;
			if ((raw_output_info->scale != NULL) && (raw_output_info->offset != NULL))
			{
				(void)memcpy(&head_scale, raw_output_info->scale,
							 sizeof(head_scale));
				head_zero_point = *(const int16_t *)raw_output_info->offset;
			}
			head_dequant_value = ((float)raw_output_value - (float)head_zero_point) * head_scale;
		}
	}

	output_value = AppAI_TraceAndApplyInferenceCalibration(output_value);
	if (!AppAI_IsFiniteFloat(output_value))
	{
		AppAI_LogFloatApprox("[AI] Inference output value: ", output_value);
		DebugConsole_WriteString(
			"[AI] Inference result is non-finite; skipping smoothing and last-result update.\r\n");
		app_ai_last_inference_valid = false;
		return;
	}
	if (!AppAI_IsPlausibleInferenceValue(output_value))
	{
		AppAI_LogFloatApprox("[AI] Inference output value: ", output_value);
		DebugConsole_WriteString(
			"[AI] Inference result out of plausible range; skipping publish/update.\r\n");
		app_ai_last_inference_valid = false;
		return;
	}
#if APP_AI_ENABLE_INFERENCE_BURST_SMOOTHING
	output_value = AppAI_FilterInferenceValue(output_value);
#endif

	/* Log both the final float output and the raw int8 tensor so we can spot
	 * quantization mismatches without changing the model result path. */
	DebugConsole_Printf(
		"[AI] raw=%d head_zp=%d output_bits=0x%08lx output_zp=%d\r\n",
		(int)raw_output_value,
		(int)head_zero_point,
		(unsigned long)output_bits.u,
		(int)output_zero_point);
	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
	{
		AppAI_LogFloatApprox("[AI] head_scale: ", head_scale);
		AppAI_LogFloatApprox("[AI] head_dequant: ", head_dequant_value);
		AppAI_LogFloatApprox("[AI] output_scale: ", output_dequant_scale);
		AppAI_LogFloatApprox("[AI] output_zero_point: ",
							 (float)output_zero_point);
		AppAI_LogFloatApprox("[AI] Inference output value: ", output_value);
	}

	app_ai_last_inference_value = output_value;
	app_ai_last_inference_valid = true;
}
#else
static void AppAI_LogInferenceResult(
	const LL_Buffer_InfoTypeDef *output_buffer_info)
{
	float output_value = 0.0f;
	char line[64] = {0};

	if ((output_buffer_info == NULL) || (LL_Buffer_addr_start(output_buffer_info) == NULL) || (LL_Buffer_len(output_buffer_info) < sizeof(output_value)))
	{
		return;
	}

	(void)memcpy(&output_value, LL_Buffer_addr_start(output_buffer_info),
				 sizeof(output_value));
	output_value = AppAI_TraceAndApplyInferenceCalibration(output_value);
	if (!AppAI_IsFiniteFloat(output_value))
	{
		AppInferenceLog_FormatFloatMicros(line, sizeof(line),
										  "[AI] Inference value: ", output_value);
		DebugConsole_WriteString(line);
		AppInferenceLog_FormatFloatMicros(line, sizeof(line),
										  "[AI] Inference exact: ", output_value);
		DebugConsole_WriteString(line);
		DebugConsole_WriteString(
			"[AI] Inference result is non-finite; skipping smoothing and last-result update.\r\n");
		app_ai_last_inference_valid = false;
		return;
	}
	if (!AppAI_IsPlausibleInferenceValue(output_value))
	{
		AppInferenceLog_FormatFloatMicros(line, sizeof(line),
										  "[AI] Inference value: ", output_value);
		DebugConsole_WriteString(line);
		DebugConsole_WriteString(
			"[AI] Inference result out of plausible range; skipping smoothing and last-result update.\r\n");
		app_ai_last_inference_valid = false;
		return;
	}
#if APP_AI_ENABLE_INFERENCE_BURST_SMOOTHING
	output_value = AppAI_FilterInferenceValue(output_value);
#endif

	AppInferenceLog_FormatFloatMicros(line, sizeof(line),
									  "[AI] Inference value: ", output_value);
	DebugConsole_WriteString(line);

	app_ai_last_inference_value = output_value;
	app_ai_last_inference_valid = true;

#if APP_AI_ENABLE_RUNTIME_METRICS
	/* Log final power consumption and end metrics tracking */
	(void)INA219_LogReading("AI-DONE");
	Metrics_EndInference("AI", output_value);
#endif
}
#endif /* APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS */

#if APP_AI_ENABLE_INFERENCE_BURST_SMOOTHING
/**
 * @brief Smooth the user-facing inference value across captures.
 *
 * A 3-frame burst median keeps the live reading from jumping on glare or a
 * single noisy capture while still tracking slow changes.
 */
static float AppAI_FilterInferenceValue(float value)
{
	float ordered[APP_AI_INFERENCE_BURST_HISTORY_SIZE] = {0.0f};
	size_t sample_count = 0U;
	size_t start_index = 0U;

	if (!AppAI_IsFiniteFloat(value))
	{
		if (app_ai_inference_burst_history_count > 0U)
		{
			const size_t last_index =
				(app_ai_inference_burst_history_next_index + APP_AI_INFERENCE_BURST_HISTORY_SIZE - 1U) % APP_AI_INFERENCE_BURST_HISTORY_SIZE;
			const float last_value =
				app_ai_inference_burst_history[last_index];

			if (AppAI_IsFiniteFloat(last_value))
			{
				return last_value;
			}
		}
		return value;
	}

	if (app_ai_inference_burst_history_count > 0U)
	{
		const size_t last_index =
			(app_ai_inference_burst_history_next_index + APP_AI_INFERENCE_BURST_HISTORY_SIZE - 1U) % APP_AI_INFERENCE_BURST_HISTORY_SIZE;
		float delta = value - app_ai_inference_burst_history[last_index];

		if (delta < 0.0f)
		{
			delta = -delta;
		}

		if (delta > APP_AI_INFERENCE_BURST_RESET_DELTA_C)
		{
			AppAI_ResetInferenceBurstHistory();
		}
	}

	app_ai_inference_burst_history[app_ai_inference_burst_history_next_index] =
		value;
	if (app_ai_inference_burst_history_count <
		APP_AI_INFERENCE_BURST_HISTORY_SIZE)
	{
		app_ai_inference_burst_history_count++;
	}
	app_ai_inference_burst_history_next_index =
		(app_ai_inference_burst_history_next_index + 1U) % APP_AI_INFERENCE_BURST_HISTORY_SIZE;

	sample_count = app_ai_inference_burst_history_count;
	start_index = (sample_count < APP_AI_INFERENCE_BURST_HISTORY_SIZE) ? 0U
																	   : app_ai_inference_burst_history_next_index;

	for (size_t i = 0U; i < sample_count; ++i)
	{
		ordered[i] =
			app_ai_inference_burst_history[(start_index + i) % APP_AI_INFERENCE_BURST_HISTORY_SIZE];
	}

	/* Keep the local copy sorted so we can choose the median for the full
	 * 3-sample burst and a simple average during warm-up. */
	for (size_t i = 1U; i < sample_count; ++i)
	{
		const float key = ordered[i];
		size_t j = i;

		while ((j > 0U) && (ordered[j - 1U] > key))
		{
			ordered[j] = ordered[j - 1U];
			--j;
		}

		ordered[j] = key;
	}

	if (sample_count == 1U)
	{
		return ordered[0U];
	}

	if (sample_count == 2U)
	{
		return 0.5f * (ordered[0U] + ordered[1U]);
	}

	return ordered[sample_count / 2U];
}
#endif /* APP_AI_ENABLE_INFERENCE_BURST_SMOOTHING */

/**
 * @brief Return true only for ordinary finite float values.
 */
static bool AppAI_IsFiniteFloat(float value)
{
	union
	{
		float f;
		uint32_t u;
	} bits = {
		.f = value};

	return ((bits.u & 0x7F800000U) != 0x7F800000U);
}

/**
 * @brief Bound-check the calibrated scalar inference before publish/log.
 */
static bool AppAI_IsPlausibleInferenceValue(float value)
{
	if (!AppAI_IsFiniteFloat(value))
	{
		return false;
	}

	if (value < APP_AI_INFERENCE_VALUE_MIN_C)
	{
		return false;
	}

	if (value > APP_AI_INFERENCE_VALUE_MAX_C)
	{
		return false;
	}

	return true;
}

static int AppAI_ApplyCacheRange(uint32_t start_addr, uint32_t end_addr,
								 bool clean, bool invalidate)
{
	uintptr_t start = 0U;
	uintptr_t end = 0U;

	if (end_addr <= start_addr)
	{
		return -1;
	}

	start = (uintptr_t)start_addr;
	end = (uintptr_t)end_addr;

	start &= ~((uintptr_t)APP_AI_CACHE_LINE_BYTES - 1U);
	end = (end + APP_AI_CACHE_LINE_BYTES - 1U) & ~((uintptr_t)APP_AI_CACHE_LINE_BYTES - 1U);

	if (clean)
	{
		SCB_CleanDCache_by_Addr((uint32_t *)start, (int32_t)(end - start));
	}

	if (invalidate)
	{
		SCB_InvalidateDCache_by_Addr((uint32_t *)start,
									 (int32_t)(end - start));
	}

return 0;
}

static uint8_t AppAI_ReadYuv422Luma(const uint8_t *frame_bytes,
									size_t frame_size_bytes,
									size_t frame_width_pixels, size_t source_x, size_t source_y);

static float AppAI_ReadNormalizedGrayFromYuv422Bilinear(
	const uint8_t *frame_bytes, size_t frame_size_bytes,
	size_t frame_width_pixels, size_t frame_height_pixels,
	float source_x, float source_y)
{
	/* Add additional bounds checking */
	if ((frame_bytes == NULL) || (frame_size_bytes < 4U) ||
		(frame_width_pixels == 0U) || (frame_height_pixels == 0U))
	{
		DebugConsole_Printf("[AI] ReadNormalizedGrayFromYuv422Bilinear: Invalid parameters\r\n");
		return 0.0f;
	}

	/* Add bounds checking for source coordinates */
	if (source_x < 0.0f || source_y < 0.0f)
	{
		DebugConsole_Printf("[AI] ReadNormalizedGrayFromYuv422Bilinear: Negative coordinates: x=%f, y=%f\r\n", 
						   (double)source_x, (double)source_y);
		source_x = (source_x < 0.0f) ? 0.0f : source_x;
		source_y = (source_y < 0.0f) ? 0.0f : source_y;
	}

	const float max_x = (frame_width_pixels > 0U) ? (float)(frame_width_pixels - 1U)
												 : 0.0f;
	const float max_y = (frame_height_pixels > 0U) ? (float)(frame_height_pixels - 1U)
												  : 0.0f;

	/* Add additional bounds checking for max values */
	if (max_x < 0.0f || max_y < 0.0f)
	{
		DebugConsole_Printf("[AI] ReadNormalizedGrayFromYuv422Bilinear: Invalid max values: max_x=%f, max_y=%f\r\n", 
						   (double)max_x, (double)max_y);
		return 0.0f;
	}

	float clamped_x = source_x;
	float clamped_y = source_y;

	if (clamped_x > max_x)
	{
		clamped_x = max_x;
	}

	if (clamped_y > max_y)
	{
		clamped_y = max_y;
	}

	/* Add additional validation for clamped coordinates */
	if (clamped_x < 0.0f || clamped_y < 0.0f)
	{
		DebugConsole_Printf("[AI] ReadNormalizedGrayFromYuv422Bilinear: Clamped coordinates negative: x=%f, y=%f\r\n", 
						   (double)clamped_x, (double)clamped_y);
		return 0.0f;
	}

	const size_t x0 = (size_t)floorf(clamped_x);
	const size_t y0 = (size_t)floorf(clamped_y);
	
	/* Add bounds checking for x0 and y0 */
	if (x0 >= frame_width_pixels || y0 >= frame_height_pixels)
	{
		DebugConsole_Printf("[AI] ReadNormalizedGrayFromYuv422Bilinear: x0/y0 out of bounds: x0=%lu, y0=%lu, width=%lu, height=%lu\r\n", 
						   (unsigned long)x0, (unsigned long)y0, 
						   (unsigned long)frame_width_pixels, (unsigned long)frame_height_pixels);
		return 0.0f;
	}

	const size_t x1 = (x0 + 1U < frame_width_pixels) ? (x0 + 1U) : x0;
	const size_t y1 = (y0 + 1U < frame_height_pixels) ? (y0 + 1U) : y0;
	const float fx = clamped_x - (float)x0;
	const float fy = clamped_y - (float)y0;
	
	/* Add bounds checking for fx and fy */
	if (fx < 0.0f || fx > 1.0f || fy < 0.0f || fy > 1.0f)
	{
		DebugConsole_Printf("[AI] ReadNormalizedGrayFromYuv422Bilinear: Invalid fx/fy: fx=%f, fy=%f\r\n", 
						   (double)fx, (double)fy);
	}

	/* Add bounds checking for AppAI_ReadYuv422Luma calls */
	const uint8_t luma00 = AppAI_ReadYuv422Luma(frame_bytes, frame_size_bytes,
												frame_width_pixels, x0, y0);
	const uint8_t luma10 = AppAI_ReadYuv422Luma(frame_bytes, frame_size_bytes,
												frame_width_pixels, x1, y0);
	const uint8_t luma01 = AppAI_ReadYuv422Luma(frame_bytes, frame_size_bytes,
												frame_width_pixels, x0, y1);
	const uint8_t luma11 = AppAI_ReadYuv422Luma(frame_bytes, frame_size_bytes,
												frame_width_pixels, x1, y1);
	const float l00 = (float)luma00 / 255.0f;
	const float l10 = (float)luma10 / 255.0f;
	const float l01 = (float)luma01 / 255.0f;
	const float l11 = (float)luma11 / 255.0f;
	const float top = l00 + (fx * (l10 - l00));
	const float bottom = l01 + (fx * (l11 - l01));
	const float gray = top + (fy * (bottom - top));

	/* Add bounds checking for final gray value */
	if (gray < 0.0f)
	{
		return 0.0f;
	}

	if (gray > 1.0f)
	{
		return 1.0f;
	}

	return gray;
}

static bool __attribute__((noinline)) AppAI_PreprocessScalarRow(
	const uint8_t *frame_bytes, size_t frame_size, size_t source_width,
	size_t source_height, size_t crop_x_min, size_t crop_y_min,
	size_t crop_width, size_t crop_height, size_t output_width,
	size_t output_height, float resize_scale, size_t resized_width,
	size_t resized_height, size_t resize_pad_x, size_t resize_pad_y,
	size_t out_y, float *input_ptr, size_t input_len_bytes)
{
	/* Add bounds checking for parameters */
	if ((frame_bytes == NULL) || (input_ptr == NULL) || (source_width == 0) || (source_height == 0))
	{
		DebugConsole_Printf("[AI] PreprocessScalarRow: Invalid parameters\r\n");
		return false;
	}

#if APP_AI_YUV422_INPUT_LUMA_ONLY
	if ((out_y < resize_pad_y) || (out_y >= (resize_pad_y + resized_height)))
	{
		return true;
	}
#endif

	const size_t row_len = output_width * 3U * sizeof(float);
	const size_t row_offset = (out_y * output_width) * 3U * sizeof(float);
	
	/* Add additional bounds checking for row_offset */
	if (row_offset >= input_len_bytes)
	{
		DebugConsole_Printf("[AI] PreprocessScalarRow: row_offset out of bounds: %lu >= %lu\r\n", 
						   (unsigned long)row_offset, (unsigned long)input_len_bytes);
		return false;
	}
	
	/* Add additional bounds checking for row_len */
	if ((row_offset + row_len) > input_len_bytes)
	{
		DebugConsole_Printf("[AI] PreprocessScalarRow: row_len out of bounds: %lu + %lu > %lu\r\n", 
						   (unsigned long)row_offset, (unsigned long)row_len, (unsigned long)input_len_bytes);
		return false;
	}
	
	uint8_t *const row_bytes = (uint8_t *)input_ptr + row_offset;

	if ((row_bytes == NULL) || (frame_bytes == NULL))
	{
		DebugConsole_Printf("[AI] PreprocessScalarRow: NULL pointer detected\r\n");
		return false;
	}
	if ((row_offset + row_len) > input_len_bytes)
	{
		DebugConsole_Printf("[AI] PreprocessScalarRow: Buffer overflow detected\r\n");
		return false;
	}

	for (size_t out_x = 0U; out_x < output_width; out_x++)
	{
		/* Add bounds checking for out_x */
		if (out_x >= output_width)
		{
			DebugConsole_Printf("[AI] PreprocessScalarRow: out_x out of bounds: %lu >= %lu\r\n", 
							   (unsigned long)out_x, (unsigned long)output_width);
			return false;
		}
		
#if APP_AI_YUV422_INPUT_LUMA_ONLY
		const size_t pixel_base = out_x * 3U * sizeof(float);
		
		/* Add bounds checking for pixel_base */
		if ((pixel_base + 11U) >= (output_width * 3U * sizeof(float)))
		{
			DebugConsole_Printf("[AI] PreprocessScalarRow: pixel_base out of bounds: %lu >= %lu\r\n", 
							   (unsigned long)(pixel_base + 11U), (unsigned long)(output_width * 3U * sizeof(float)));
			return false;
		}
		
		uint32_t gray_bits = 0U;

		if ((out_x >= resize_pad_x) &&
			(out_x < (resize_pad_x + resized_width)))
		{
			const size_t resized_x = out_x - resize_pad_x;
			const size_t resized_y = out_y - resize_pad_y;
			size_t sample_x = crop_x_min + ((resized_x * crop_width) / resized_width);
			size_t sample_y = crop_y_min + ((resized_y * crop_height) / resized_height);

			/* Clamp the nearest-neighbor sample point to the crop bounds. */
			if (sample_x >= (crop_x_min + crop_width))
			{
				sample_x = crop_x_min + crop_width - 1U;
			}
			if (sample_y >= (crop_y_min + crop_height))
			{
				sample_y = crop_y_min + crop_height - 1U;
			}

			/* Read the luma sample directly so the hot row loop stays in the
			 * lightweight nearest-neighbor path. */
			const size_t source_index =
				(((sample_y * source_width) + (sample_x & ~1U)) * 2U) +
				((sample_x & 1U) ? 2U : 0U);
			const uint8_t gray = frame_bytes[source_index];

			{
				union
				{
					float f;
					uint32_t u;
				} gray_value = {
					.f = ((float)gray) / 255.0f};
				gray_bits = gray_value.u;
			}
		}
		{
			const uint8_t *const gray_bytes = (const uint8_t *)&gray_bits;

			/* Write the float bytes one at a time so the compiler cannot turn the
			 * store into an alignment-sensitive wide access on this board. */
			row_bytes[pixel_base + 0U] = gray_bytes[0];
			row_bytes[pixel_base + 1U] = gray_bytes[1];
			row_bytes[pixel_base + 2U] = gray_bytes[2];
			row_bytes[pixel_base + 3U] = gray_bytes[3];

			row_bytes[pixel_base + 4U] = gray_bytes[0];
			row_bytes[pixel_base + 5U] = gray_bytes[1];
			row_bytes[pixel_base + 6U] = gray_bytes[2];
			row_bytes[pixel_base + 7U] = gray_bytes[3];

			row_bytes[pixel_base + 8U] = gray_bytes[0];
			row_bytes[pixel_base + 9U] = gray_bytes[1];
			row_bytes[pixel_base + 10U] = gray_bytes[2];
			row_bytes[pixel_base + 11U] = gray_bytes[3];
		}
#else
	if ((out_y < resize_pad_y) || (out_y >= (resize_pad_y + resized_height)))
	{
		/* Add bounds checking for memset */
		const size_t memset_len = output_width * 3U * sizeof(float);
		if ((row_offset + memset_len) <= input_len_bytes)
		{
			(void)memset(row_bytes, 0, memset_len);
		}
		else
		{
			DebugConsole_Printf("[AI] PreprocessScalarRow: memset would overflow buffer\r\n");
		}
		return true;
	}

	const size_t pixel_base = out_x * 3U * sizeof(float);
	
	/* Add bounds checking for pixel_base */
	if ((pixel_base + 11U) >= (output_width * 3U * sizeof(float)))
	{
		DebugConsole_Printf("[AI] PreprocessScalarRow: pixel_base out of bounds: %lu >= %lu\r\n", 
						   (unsigned long)(pixel_base + 11U), (unsigned long)(output_width * 3U * sizeof(float)));
		return false;
	}
	
	float out_r_bits = 0.0f;
	float out_g_bits = 0.0f;
	float out_b_bits = 0.0f;

		if ((out_x >= resize_pad_x) &&
			(out_x < (resize_pad_x + resized_width)))
		{
			const float resized_x = (float)(out_x - resize_pad_x);
			const float resized_y = (float)(out_y - resize_pad_y);
			float crop_x = ((resized_x + 0.5f) / resize_scale) - 0.5f;
			float crop_y = ((resized_y + 0.5f) / resize_scale) - 0.5f;

			/* Add bounds checking for crop coordinates */
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

			/* Add bounds checking for crop_x_min and crop_y_min */
			const float total_crop_x = (float)crop_x_min + crop_x;
			const float total_crop_y = (float)crop_y_min + crop_y;
			
			if (total_crop_x >= 0.0f && total_crop_x < (float)source_width &&
				total_crop_y >= 0.0f && total_crop_y < (float)source_height)
			{
				AppAI_ReadRgbFromYuv422Bilinear(
					frame_bytes, frame_size, source_width, source_height,
					total_crop_x, total_crop_y,
					&out_r_bits, &out_g_bits, &out_b_bits);
			}
			else
			{
				DebugConsole_Printf("[AI] PreprocessScalarRow: Crop coordinates out of bounds: x=%f, y=%f\r\n", 
								   (double)total_crop_x, (double)total_crop_y);
				out_r_bits = 0.0f;
				out_g_bits = 0.0f;
				out_b_bits = 0.0f;
			}
		}

		out_r_bits = AppAI_ClampNormalizedFloat(out_r_bits);
		out_g_bits = AppAI_ClampNormalizedFloat(out_g_bits);
		out_b_bits = AppAI_ClampNormalizedFloat(out_b_bits);
		{
			const uint8_t *const r_bytes = (const uint8_t *)&out_r_bits;
			const uint8_t *const g_bytes = (const uint8_t *)&out_g_bits;
			const uint8_t *const b_bytes = (const uint8_t *)&out_b_bits;

			row_bytes[pixel_base + 0U] = r_bytes[0];
			row_bytes[pixel_base + 1U] = r_bytes[1];
			row_bytes[pixel_base + 2U] = r_bytes[2];
			row_bytes[pixel_base + 3U] = r_bytes[3];

			row_bytes[pixel_base + 4U] = g_bytes[0];
			row_bytes[pixel_base + 5U] = g_bytes[1];
			row_bytes[pixel_base + 6U] = g_bytes[2];
			row_bytes[pixel_base + 7U] = g_bytes[3];

			row_bytes[pixel_base + 8U] = b_bytes[0];
			row_bytes[pixel_base + 9U] = b_bytes[1];
			row_bytes[pixel_base + 10U] = b_bytes[2];
			row_bytes[pixel_base + 11U] = b_bytes[3];
		}
#endif
	}

	return true;
}

static bool __attribute__((noinline)) AppAI_PreprocessYuv422FrameToFloatInput(
	const uint8_t *frame_bytes, size_t frame_size, float *input_ptr,
	size_t input_float_count, size_t input_len_bytes)
{
	/* Add additional bounds checking and validation */
	if ((frame_bytes == NULL) || (input_ptr == NULL))
	{
		DebugConsole_Printf("[AI] PreprocessYuv422FrameToFloatInput: NULL pointers\r\n");
		return false;
	}

	const size_t source_width = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const size_t source_height = (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;

	/* Add additional validation for source dimensions */
	if (source_width == 0U || source_height == 0U)
	{
		DebugConsole_Printf("[AI] PreprocessYuv422FrameToFloatInput: Invalid source dimensions: width=%lu, height=%lu\r\n", 
						   (unsigned long)source_width, (unsigned long)source_height);
		return false;
	}

	/* Minimum 3-channel float input (224*224*3) for RGB models like the
	 * rectified scalar reader and the OBB front-end. Legacy int8 polar-path
	 * tensors stay on their separate branch below. */
	const size_t min_float_count = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS * (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS * 3U;
	const size_t min_float_bytes = min_float_count * sizeof(float);
	if ((frame_size < (size_t)APP_AI_CAPTURE_FRAME_BYTES) ||
		(input_float_count < min_float_count) ||
		(input_len_bytes < min_float_bytes))
	{
		DebugConsole_Printf("[AI] PreprocessYuv422FrameToFloatInput: Buffer size mismatch: frame_size=%lu, input_float_count=%lu, input_len_bytes=%lu\r\n", 
						   (unsigned long)frame_size, (unsigned long)input_float_count, (unsigned long)input_len_bytes);
		return false;
	}

	/* Add additional validation for expected frame size */
	const size_t expected_frame_size = source_width * source_height * APP_AI_CAPTURE_FRAME_BYTES_PER_PIXEL;
	if (frame_size != expected_frame_size)
	{
		DebugConsole_Printf("[AI] PreprocessYuv422FrameToFloatInput: Frame size mismatch: expected=%lu, actual=%lu\r\n", 
						   (unsigned long)expected_frame_size, (unsigned long)frame_size);
		/* Continue processing but log the mismatch */
	}

	size_t crop_x_min = 0U;
	size_t crop_y_min = 0U;
	size_t crop_width = source_width;
	size_t crop_height = source_height;
	bool crop_found = false;
	const char *crop_label = "fixed";

	/* DCMIPP_PIXEL_PACKER_FORMAT_YUV422_1 emits packed YUYV samples:
	 *   byte 0 = Y0, byte 1 = U, byte 2 = Y1, byte 3 = V, ...
	 * We estimate the gauge position from the bright dial face, crop around
	 * that box, and then map the crop into the model input tensor. */
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
		
		/* Add bounds checking for forced crop */
		if (crop_x_min >= source_width || crop_y_min >= source_height)
		{
			DebugConsole_Printf("[AI] PreprocessYuv422FrameToFloatInput: Forced crop origin out of bounds: x=%lu, y=%lu, width=%lu, height=%lu\r\n", 
							   (unsigned long)crop_x_min, (unsigned long)crop_y_min, 
							   (unsigned long)source_width, (unsigned long)source_height);
			crop_x_min = 0U;
			crop_y_min = 0U;
			crop_width = source_width;
			crop_height = source_height;
		}
		
		if ((crop_x_min + crop_width) > source_width)
		{
			crop_width = source_width - crop_x_min;
		}
		
		if ((crop_y_min + crop_height) > source_height)
		{
			crop_height = source_height - crop_y_min;
		}
	}
	else
	{
#if APP_AI_USE_ADAPTIVE_GAUGE_CROP
		crop_found = AppAI_EstimateGaugeCropBoxFromYuv422(frame_bytes, frame_size,
														  source_width, source_height, &crop_x_min, &crop_y_min,
														  &crop_width, &crop_height);
		
		/* Add bounds checking for adaptive crop */
		if (crop_found)
		{
			if (crop_x_min >= source_width || crop_y_min >= source_height)
			{
				DebugConsole_Printf("[AI] PreprocessYuv422FrameToFloatInput: Adaptive crop origin out of bounds: x=%lu, y=%lu, width=%lu, height=%lu\r\n", 
								   (unsigned long)crop_x_min, (unsigned long)crop_y_min, 
								   (unsigned long)source_width, (unsigned long)source_height);
				crop_found = false;
			}
			else
			{
				if ((crop_x_min + crop_width) > source_width)
				{
					crop_width = source_width - crop_x_min;
				}
				
				if ((crop_y_min + crop_height) > source_height)
				{
					crop_height = source_height - crop_y_min;
				}
				
				if (crop_width == 0U || crop_height == 0U)
				{
					crop_found = false;
				}
			}
		}
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
		
		/* Add bounds checking for default crop */
		if (crop_x_min >= source_width)
		{
			crop_x_min = source_width - 1U;
		}
		if (crop_y_min >= source_height)
		{
			crop_y_min = source_height - 1U;
		}
		if ((crop_x_min + crop_width) > source_width)
		{
			crop_width = source_width - crop_x_min;
		}
		if ((crop_y_min + crop_height) > source_height)
		{
			crop_height = source_height - crop_y_min;
		}
	}

	/* Clamp the crop before we hand it to the resize path.
	 * The scalar reader expects a crop that stays inside the captured frame,
	 * and clamping here makes the preprocessing path resilient even if a future
	 * crop heuristic overshoots by a pixel. */
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

	/* Add additional validation for final crop dimensions */
	if (crop_width == 0U || crop_height == 0U)
	{
		DebugConsole_Printf("[AI] PreprocessYuv422FrameToFloatInput: Invalid crop dimensions: width=%lu, height=%lu\r\n", 
						   (unsigned long)crop_width, (unsigned long)crop_height);
		return false;
	}

	DebugConsole_Printf("[AI] Crop %s: x=%lu y=%lu w=%lu h=%lu\r\n",
					   crop_label,
					   (unsigned long)crop_x_min, (unsigned long)crop_y_min,
					   (unsigned long)crop_width, (unsigned long)crop_height);

	if (APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS)
	{
		AppAI_LogSourceCropWindow(frame_bytes, frame_size, source_width,
								  source_height, crop_x_min, crop_y_min, crop_width, crop_height);
	}

	(void)DebugConsole_WriteString("[AI] Preprocess diagnostics OK.\r\n");
	/* Keep legacy speed for 3-channel models, but if a wider tensor is
	 * provisioned (for example, feature-augmented channels), clear it first so
	 * unwritten channels are deterministic instead of stale SRAM. */
	if (input_len_bytes > (size_t)APP_AI_MODEL_INPUT_FLOAT_BYTES)
	{
		(void)memset(input_ptr, 0, input_len_bytes);
		DebugConsole_Printf(
			"[AI] Preprocess zero-fill applied for extended input: bytes=%lu\r\n",
			(unsigned long)input_len_bytes);
	}
	else
	{
		(void)DebugConsole_WriteString("[AI] Preprocess zero-fill skipped.\r\n");
	}
	(void)DebugConsole_WriteString("[AI] Preprocess resize start.\r\n");

#if APP_AI_YUV422_INPUT_LUMA_ONLY
	{
		const size_t output_width = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
		const size_t output_height = (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
		const size_t output_float_count = output_width * output_height * 3U;
		const size_t output_row_bytes = output_width * 3U * sizeof(float);
		const size_t output_total_bytes = output_float_count * sizeof(float);
		size_t resized_width = output_width;
		size_t resized_height = output_height;
		size_t resize_pad_x = 0U;
		size_t resize_pad_y = 0U;

		if ((output_width == 0U) || (output_height == 0U))
		{
			DebugConsole_Printf("[AI] PreprocessYuv422FrameToFloatInput: Invalid output dimensions: width=%lu, height=%lu\r\n",
							   (unsigned long)output_width, (unsigned long)output_height);
			return false;
		}

		if (output_total_bytes > input_len_bytes)
		{
			DebugConsole_Printf("[AI] PreprocessYuv422FrameToFloatInput: Output buffer too small: required=%lu, available=%lu\r\n",
							   (unsigned long)output_total_bytes,
							   (unsigned long)input_len_bytes);
			return false;
		}

		/* Keep the original aspect ratio using integer math only. */
		app_ai_scalar_preprocess_last_row = (size_t)SIZE_MAX;

		/* Fast path: when the crop already spans the output tensor, fill it
		 * directly without extra resize math or padding checks. */
		(void)DebugConsole_WriteString("[AI] Preprocess row loop enter (luma).\r\n");
		if ((crop_x_min == 0U) && (crop_y_min == 0U) &&
			(crop_width == output_width) && (crop_height == output_height))
		{
			resized_width = output_width;
			resized_height = output_height;
			resize_pad_x = 0U;
			resize_pad_y = 0U;

			for (size_t out_y = 0U; out_y < output_height; ++out_y)
			{
				uint8_t *const row_bytes =
					(uint8_t *)input_ptr + (out_y * output_row_bytes);

				app_ai_scalar_preprocess_last_row = out_y;
				if ((out_y == 0U) || ((out_y & 0x3FU) == 0U))
				{
					DebugConsole_Printf(
						"[AI] Preprocess luma row %lu/223\r\n",
						(unsigned long)out_y);
				}

				for (size_t out_x = 0U; out_x < output_width; ++out_x)
				{
					const size_t pixel_base = out_x * 3U * sizeof(float);
					const size_t read_index = ((out_y * source_width) + out_x) * 2U;
					const uint8_t gray = frame_bytes[read_index];
					const uint32_t gray_bits = AppAI_GrayToFloatBits(gray);

					{
						const uint8_t *const gray_bytes =
							(const uint8_t *)&gray_bits;

						row_bytes[pixel_base + 0U] = gray_bytes[0];
						row_bytes[pixel_base + 1U] = gray_bytes[1];
						row_bytes[pixel_base + 2U] = gray_bytes[2];
						row_bytes[pixel_base + 3U] = gray_bytes[3];

						row_bytes[pixel_base + 4U] = gray_bytes[0];
						row_bytes[pixel_base + 5U] = gray_bytes[1];
						row_bytes[pixel_base + 6U] = gray_bytes[2];
						row_bytes[pixel_base + 7U] = gray_bytes[3];

						row_bytes[pixel_base + 8U] = gray_bytes[0];
						row_bytes[pixel_base + 9U] = gray_bytes[1];
						row_bytes[pixel_base + 10U] = gray_bytes[2];
						row_bytes[pixel_base + 11U] = gray_bytes[3];
					}
				}
			}
		}
		else
		{
			if ((output_width * crop_height) <= (output_height * crop_width))
			{
				resized_width = output_width;
				resized_height = (size_t)(((crop_height * output_width) + (crop_width / 2U)) / crop_width);
			}
			else
			{
				resized_height = output_height;
				resized_width = (size_t)(((crop_width * output_height) + (crop_height / 2U)) / crop_height);
			}

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

			for (size_t out_y = 0U; out_y < output_height; ++out_y)
			{
				uint8_t *const row_bytes =
					(uint8_t *)input_ptr + (out_y * output_row_bytes);

				app_ai_scalar_preprocess_last_row = out_y;
				if ((out_y == 0U) || ((out_y & 0x3FU) == 0U))
				{
					DebugConsole_Printf(
						"[AI] Preprocess luma row %lu/223\r\n",
						(unsigned long)out_y);
				}

				if ((out_y < resize_pad_y) ||
					(out_y >= (resize_pad_y + resized_height)))
				{
					(void)memset(row_bytes, 0, output_row_bytes);
					continue;
				}

				for (size_t out_x = 0U; out_x < output_width; ++out_x)
				{
					const size_t pixel_base = out_x * 3U * sizeof(float);
					uint32_t gray_bits = 0U;

					if ((out_x >= resize_pad_x) &&
						(out_x < (resize_pad_x + resized_width)))
					{
						const size_t resized_x = out_x - resize_pad_x;
						const size_t resized_y = out_y - resize_pad_y;
						size_t sample_x =
							crop_x_min + ((resized_x * crop_width) / resized_width);
						size_t sample_y =
							crop_y_min + ((resized_y * crop_height) / resized_height);

						if (sample_x >= source_width)
						{
							sample_x = source_width - 1U;
						}
						if (sample_y >= source_height)
						{
							sample_y = source_height - 1U;
						}

						if ((sample_x < source_width) && (sample_y < source_height))
						{
							const size_t pair_x = sample_x & ~1U;
							const size_t source_index =
								((sample_y * source_width) + pair_x) * 2U;
							const size_t read_index =
								source_index + ((sample_x & 1U) ? 2U : 0U);

							if (read_index < frame_size)
							{
								const uint8_t gray = frame_bytes[read_index];
								gray_bits = AppAI_GrayToFloatBits(gray);
							}
						}
					}

					{
						const uint8_t *const gray_bytes =
							(const uint8_t *)&gray_bits;

						row_bytes[pixel_base + 0U] = gray_bytes[0];
						row_bytes[pixel_base + 1U] = gray_bytes[1];
						row_bytes[pixel_base + 2U] = gray_bytes[2];
						row_bytes[pixel_base + 3U] = gray_bytes[3];

						row_bytes[pixel_base + 4U] = gray_bytes[0];
						row_bytes[pixel_base + 5U] = gray_bytes[1];
						row_bytes[pixel_base + 6U] = gray_bytes[2];
						row_bytes[pixel_base + 7U] = gray_bytes[3];

						row_bytes[pixel_base + 8U] = gray_bytes[0];
						row_bytes[pixel_base + 9U] = gray_bytes[1];
						row_bytes[pixel_base + 10U] = gray_bytes[2];
						row_bytes[pixel_base + 11U] = gray_bytes[3];
					}
				}
			}
		}
	}
#else
	{
		const size_t output_width = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
		const size_t output_height = (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
		const size_t output_float_count = output_width * output_height * 3U;
		
		/* Add bounds checking for output dimensions */
		if (output_width == 0U || output_height == 0U)
		{
			DebugConsole_Printf("[AI] PreprocessYuv422FrameToFloatInput: Invalid output dimensions: width=%lu, height=%lu\r\n", 
							   (unsigned long)output_width, (unsigned long)output_height);
			return false;
		}
		
		const float resize_scale =
			fminf((float)output_width / (float)crop_width,
				  (float)output_height / (float)crop_height);
		
		/* Add bounds checking for resize_scale */
		if (resize_scale <= 0.0f || !isfinite(resize_scale))
		{
			DebugConsole_Printf("[AI] PreprocessYuv422FrameToFloatInput: Invalid resize_scale: %f\r\n", (double)resize_scale);
			return false;
		}
		
		size_t resized_width = (size_t)(((float)crop_width * resize_scale) + 0.5f);
		size_t resized_height = (size_t)(((float)crop_height * resize_scale) + 0.5f);
		size_t resize_pad_x = 0U;
		size_t resize_pad_y = 0U;
		
#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
		DebugConsole_Printf(
			"[AI] Resize plan: mode=bilinear-pad output=%lux%lu crop=%lux%lu resized=%lux%lu scale=%.6f\r\n",
			(unsigned long)output_width, (unsigned long)output_height,
			(unsigned long)crop_width, (unsigned long)crop_height,
			(unsigned long)resized_width, (unsigned long)resized_height,
			(double)resize_scale);
#endif

		if ((output_width == 0U) || (output_height == 0U))
		{
			return false;
		}

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

		/* Row writer handles both active pixels and padded areas, so we do not
		 * bulk-clear the full tensor up-front. */
		if ((output_float_count * sizeof(float)) > input_len_bytes)
		{
			DebugConsole_Printf("[AI] PreprocessYuv422FrameToFloatInput: Output buffer too small: required=%lu, available=%lu\r\n", 
							   (unsigned long)(output_float_count * sizeof(float)), (unsigned long)input_len_bytes);
			return false;
		}

		(void)DebugConsole_WriteString("[AI] Preprocess row loop enter.\r\n");

		for (size_t chunk_start = 0U; chunk_start < output_height;
			 chunk_start += APP_AI_SCALAR_PREPROCESS_ROWS_PER_CHUNK)
		{
			const size_t chunk_end =
				((chunk_start + APP_AI_SCALAR_PREPROCESS_ROWS_PER_CHUNK) < output_height)
					? (chunk_start + APP_AI_SCALAR_PREPROCESS_ROWS_PER_CHUNK)
					: output_height;

			/* Add bounds checking for chunk_end */
			if (chunk_end > output_height)
			{
				DebugConsole_Printf("[AI] PreprocessYuv422FrameToFloatInput: chunk_end out of bounds: %lu > %lu\r\n", 
								   (unsigned long)chunk_end, (unsigned long)output_height);
				return false;
			}

			for (size_t out_y = chunk_start; out_y < chunk_end; out_y++)
			{
				/* Add bounds checking for out_y */
				if (out_y >= output_height)
				{
					DebugConsole_Printf("[AI] PreprocessYuv422FrameToFloatInput: out_y out of bounds: %lu >= %lu\r\n", 
									   (unsigned long)out_y, (unsigned long)output_height);
					return false;
				}
				
				app_ai_scalar_preprocess_last_row = out_y;

				if (!AppAI_PreprocessScalarRow(frame_bytes, frame_size, source_width,
											 source_height, crop_x_min, crop_y_min,
												crop_width, crop_height, output_width,
													 output_height, resize_scale, resized_width,
													 resized_height, resize_pad_x, resize_pad_y,
													 out_y, input_ptr, input_len_bytes))
				{
					DebugConsole_Printf("[AI] PreprocessYuv422FrameToFloatInput: AppAI_PreprocessScalarRow failed at row %lu\r\n", 
									   (unsigned long)out_y);
					return false;
				}
			}
		}

#if APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS
		DebugConsole_WriteString("[AI] Preprocess write summary skipped.\r\n");
#endif /* APP_AI_ENABLE_VERBOSE_CONSOLE_LOGS */
	}

	(void)DebugConsole_WriteString("[AI] Preprocess resize OK.\r\n");

	return true;
}

static bool __attribute__((noinline)) AppAI_PreprocessYuv422FrameToInt8Input(
	const uint8_t *frame_bytes, size_t frame_size, uint8_t *input_ptr,
	size_t input_len_bytes, const LL_Buffer_InfoTypeDef *input_info)
{
	const size_t source_width = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const size_t source_height = (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	const size_t output_width = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const size_t output_height = (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
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
#endif

	(void)DebugConsole_WriteString("[AI] Preprocess resize OK.\r\n");
	return true;
}

/* V28 polar-vote preprocessing: YUV422 -> 224x224x7 int8 polar input.
 *
 * Channels 0-2: polar-projected RGB (normalized 0-1, mapped to int8).
 * Channels 3-5: Sobel edge channels on polar luma (angular, radial gradients).
 * Channel 6: vote prior (weighted darkness + edge evidence with radial mask).
 *
 * Polar projection maps each (row, col) in the 224x224 output to source (x, y):
 *   angle = 2 * pi * col / 224
 *   radius = (row / 224) * max_radius
 *   source_y = center_y - radius * sin(angle)   [Y axis inverted]

 *
 * Sobel edges are computed on the polar luma:
 *   Channel 4: |dI/dtheta| (horizontal Sobel = angular derivative)
 *   Channel 5: |dI/dr|     (vertical Sobel = radial derivative)
 *   Each normalized by its 99th percentile.
 *
 * Input quantisation: float [0,1] -> int8 via q = round(x*255) - 128, clamped [-128, 127].
 */
static float __attribute__((noinline)) AppAI_ScorePolarAlignmentCandidate(
	const uint8_t *resized_rgb,
	size_t output_dim,
	float center_x,
	float center_y,
	float max_radius,
	uint8_t *polar_luma_out)
{
	const float two_pi = 2.0f * 3.14159265358979323846f;
	const size_t bins = (size_t)APP_AI_POLAR_VOTE_BINS;
	uint32_t luma_hist[256U] = {0U};
	const uint32_t total_luma_samples = (uint32_t)(output_dim * output_dim);
	const float row_denominator = (output_dim > 1U) ? (float)(output_dim - 1U) : 1.0f;
	float continuity_sum = 0.0f;

	if ((resized_rgb == NULL) || (polar_luma_out == NULL) || (output_dim == 0U))
	{
		return 0.0f;
	}

	for (size_t row = 0U; row < output_dim; ++row)
	{
		for (size_t col = 0U; col < output_dim; ++col)
		{
			const float angle = two_pi * (float)col / (float)bins;
			const float radius = ((float)row / (float)output_dim) * max_radius;
			const float src_x = center_x + (radius * cosf(angle));
			const float src_y = center_y - (radius * sinf(angle));
			float out_r = 0.0f;
			float out_g = 0.0f;
			float out_b = 0.0f;
			float luma_f = 0.0f;
			int32_t luma_q = 0;

			if ((src_x >= 0.0f) && (src_x < (float)output_dim) &&
			    (src_y >= 0.0f) && (src_y < (float)output_dim))
			{
				AppAI_ReadRgbFromRgbBilinear(
					resized_rgb, output_dim, output_dim,
					src_x, src_y, &out_r, &out_g, &out_b);
			}

			luma_f = (0.299f * out_r) + (0.587f * out_g) + (0.114f * out_b);
			luma_q = (int32_t)lroundf(luma_f * 255.0f);
			if (luma_q < 0)
			{
				luma_q = 0;
			}
			else if (luma_q > 255)
			{
				luma_q = 255;
			}

			polar_luma_out[row * output_dim + col] = (uint8_t)luma_q;
			luma_hist[(uint8_t)luma_q]++;
		}
	}

	{
		const uint8_t luma_threshold = AppAI_FindHistogramPercentile(
			luma_hist, 256U, total_luma_samples, 0.38f);

		for (size_t col = 0U; col < output_dim; ++col)
		{
			const float radial_mask_column = 1.0f;
			float best_run = 0.0f;
			float run = 0.0f;
			for (size_t row = 0U; row < output_dim; ++row)
			{
				const float row_position = (float)row / row_denominator;
				const float radial_mask =
					((row_position >= 0.18f) && (row_position <= 0.95f)) ? radial_mask_column : 0.0f;
				const uint8_t luma_q = polar_luma_out[row * output_dim + col];
				if ((radial_mask > 0.5f) && (luma_q <= luma_threshold))
				{
					run += 1.0f;
					if (run > best_run)
					{
						best_run = run;
					}
				}
				else
				{
					run = 0.0f;
				}
			}
			continuity_sum += best_run;
		}
	}

	{
		const uint32_t total_cells = (uint32_t)(output_dim * bins);
		return continuity_sum / (float)((total_cells > 0U) ? total_cells : 1U);
	}
}

static bool __attribute__((noinline)) AppAI_PreprocessYuv422FrameToPolarInput(
	const uint8_t *frame_bytes, size_t frame_size,
	uint8_t *input_ptr, size_t input_len_bytes,
	const LL_Buffer_InfoTypeDef *input_info)
{
	const size_t source_width = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const size_t source_height = (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS;
	const size_t output_dim = (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS;
	const size_t channels = 7U;
	const size_t required_bytes = output_dim * output_dim * channels;
	const float two_pi = 2.0f * 3.14159265358979323846f;
	const size_t bins = (size_t)APP_AI_POLAR_VOTE_BINS;

	/* Polar projection center and radius in the resized 224x224 coordinate
	 * space (matching the training pipeline crop -> resize_with_pad -> polar
	 * flow).  The inverse resize_with_pad transform maps each polar pixel
	 * back to the correct source frame coordinate. */
	float center_x = 0.0f;
	float center_y = 0.0f;
	float max_radius = 0.0f;
	float resize_scale = 1.0f;   /* 224 / max(crop_w, crop_h) */
	float pad_x_f = 0.0f;       /* horizontal padding in resized space */
	float pad_y_f = 0.0f;       /* vertical padding in resized space */
	float crop_x_min_f = 0.0f;  /* crop origin in frame coords */
	float crop_y_min_f = 0.0f;
	float crop_width_f = 0.0f;   /* crop dims needed for padded-area check */
	float crop_height_f = 0.0f;
	float resized_w = 0.0f;     /* crop dims after resize_with_pad (in 224 space) */
	float resized_h = 0.0f;
	/* Scratch for polar luma plane: 224x224 uint8 (~49 KB in BSS).
	 * Stores quantised luma as uint8 [0..255] instead of float [0..1]
	 * to save ~148 KB per plane.  Gradients are recomputed on-the-fly
	 * in a second pass instead of being stored in separate arrays,
	 * eliminating two ~197 KB float arrays (~394 KB total savings). */
	static uint8_t resized_rgb[224U * 224U * 3U] __attribute__((section(".npusram6"))); /* training-style resized crop */
	static uint8_t polar_luma[224U * 224U] __attribute__((section(".npusram6")));  /* exact V28 polar luma scratch */

	if ((frame_bytes == NULL) || (input_ptr == NULL) || (input_info == NULL))
	{
		return false;
	}
	if ((frame_size < (size_t)APP_AI_CAPTURE_FRAME_BYTES) ||
		(input_len_bytes < required_bytes))
	{
		DebugConsole_Printf(
			"[AI] PolarInput: buffer size mismatch frame=%lu input=%lu required=%lu\r\n",
			(unsigned long)frame_size, (unsigned long)input_len_bytes,
			(unsigned long)required_bytes);
		return false;
	}

	/* Determine crop region from forced crop or training defaults. */
	{
		size_t crop_x_min = 0U;
		size_t crop_y_min = 0U;
		size_t crop_width = source_width;
		size_t crop_height = source_height;

		if (app_ai_forced_crop_active)
		{
			crop_x_min = app_ai_forced_crop_x_min;
			crop_y_min = app_ai_forced_crop_y_min;
			crop_width = app_ai_forced_crop_width;
			crop_height = app_ai_forced_crop_height;
		}
		else
		{
			/* Use the training crop ratios as defaults. */
			crop_x_min = (size_t)((float)source_width * APP_AI_TRAINING_CROP_X_MIN_RATIO);
			crop_y_min = (size_t)((float)source_height * APP_AI_TRAINING_CROP_Y_MIN_RATIO);
			crop_width = (size_t)((float)source_width * (APP_AI_TRAINING_CROP_X_MAX_RATIO - APP_AI_TRAINING_CROP_X_MIN_RATIO));
			crop_height = (size_t)((float)source_height * (APP_AI_TRAINING_CROP_Y_MAX_RATIO - APP_AI_TRAINING_CROP_Y_MIN_RATIO));
		}

		/* Clamp crop to frame bounds. */
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

		/* Training pipeline: crop -> tf.image.resize_with_pad(224x224) ->
		 * polar(center=112,112, r=112).  We replicate this by computing
		 * the inverse resize_with_pad transform so each polar pixel maps
		 * back to the correct source frame coordinate. */
		crop_x_min_f = (float)crop_x_min;
		crop_y_min_f = (float)crop_y_min;
		crop_width_f = (float)crop_width;
		crop_height_f = (float)crop_height;
		resize_scale = 224.0f / fmaxf(crop_width_f, crop_height_f);
		resized_w = crop_width_f * resize_scale;
		resized_h = crop_height_f * resize_scale;
		pad_x_f = (224.0f - resized_w) * 0.5f;
		pad_y_f = (224.0f - resized_h) * 0.5f;
		/* Polar center and max_radius in the resized 224x224 space. */
		center_x = 112.0f;
		center_y = 112.0f;
		max_radius = 112.0f;

		/* newlib-nano does not support %f in printf; use integer formatting. */
		{
			const long scale_milli = (long)(resize_scale * 1000.0f + 0.5f);
			const long padx_milli = (long)(pad_x_f * 1000.0f + 0.5f);
			const long pady_milli = (long)(pad_y_f * 1000.0f + 0.5f);
			DebugConsole_Printf("[AI] Polar crop: cx=%ld cy=%ld r=%ld scale=%ld.%03ld pad=(%ld.%03ld,%ld.%03ld)\r\n",
				(long)center_x, (long)center_y, (long)max_radius,
				scale_milli / 1000L, scale_milli % 1000L,
				padx_milli / 1000L, padx_milli % 1000L,
				pady_milli / 1000L, pady_milli % 1000L);
		}
	}

	/* Step 1: build the resized 224x224 crop exactly like the offline
	 * resize_with_pad path, using a temporary uint8 RGB buffer. */
	(void)memset(input_ptr, 0, required_bytes);
	(void)memset(resized_rgb, 0, sizeof(resized_rgb));
	(void)memset(polar_luma, 0, sizeof(polar_luma));

	for (size_t out_y = 0U; out_y < output_dim; ++out_y)
	{
		for (size_t out_x = 0U; out_x < output_dim; ++out_x)
		{
			const float resized_x = (float)out_x;
			const float resized_y = (float)out_y;
			uint8_t out_r = 0U;
			uint8_t out_g = 0U;
			uint8_t out_b = 0U;

			if ((resized_x >= pad_x_f) && (resized_x < (pad_x_f + resized_w)) &&
			    (resized_y >= pad_y_f) && (resized_y < (pad_y_f + resized_h)))
			{
				float crop_x = ((resized_x + 0.5f) / resize_scale) - 0.5f;
				float crop_y = ((resized_y + 0.5f) / resize_scale) - 0.5f;
				if (crop_x < 0.0f)
				{
					crop_x = 0.0f;
				}
				else if (crop_x > (crop_width_f - 1.0f))
				{
					crop_x = crop_width_f - 1.0f;
				}
				if (crop_y < 0.0f)
				{
					crop_y = 0.0f;
				}
				else if (crop_y > (crop_height_f - 1.0f))
				{
					crop_y = crop_height_f - 1.0f;
				}

				const float total_crop_x = crop_x_min_f + crop_x;
				const float total_crop_y = crop_y_min_f + crop_y;
				float sample_r = 0.0f;
				float sample_g = 0.0f;
				float sample_b = 0.0f;

				AppAI_ReadRgbFromYuv422Bilinear(
					frame_bytes, frame_size, source_width, source_height,
					total_crop_x, total_crop_y,
					&sample_r, &sample_g, &sample_b);

				out_r = (uint8_t)lroundf(sample_r * 255.0f);
				out_g = (uint8_t)lroundf(sample_g * 255.0f);
				out_b = (uint8_t)lroundf(sample_b * 255.0f);
			}

			{
				const size_t base = (out_y * output_dim + out_x) * 3U;
				resized_rgb[base + 0U] = out_r;
				resized_rgb[base + 1U] = out_g;
				resized_rgb[base + 2U] = out_b;
			}
		}
	}

	/* Step 1.5: search a small 3x3 neighborhood around the nominal polar
	 * pivot. This matches the offline V28 helper and keeps the live needle
	 * vote from drifting when the crop lands a few pixels off-center. */
	{
		const int32_t offsets[] = {
			0,
			-(int32_t)APP_AI_POLAR_CENTER_SEARCH_PIXELS,
			(int32_t)APP_AI_POLAR_CENTER_SEARCH_PIXELS,
		};
		float best_score = -1.0f;
		float best_center_x = center_x;
		float best_center_y = center_y;

		for (size_t idx_y = 0U; idx_y < (sizeof(offsets) / sizeof(offsets[0])); ++idx_y)
		{
			for (size_t idx_x = 0U; idx_x < (sizeof(offsets) / sizeof(offsets[0])); ++idx_x)
			{
				const float candidate_center_x = center_x + (float)offsets[idx_x];
				const float candidate_center_y = center_y + (float)offsets[idx_y];
				const float candidate_score = AppAI_ScorePolarAlignmentCandidate(
					resized_rgb,
					output_dim,
					candidate_center_x,
					candidate_center_y,
					max_radius,
					polar_luma);
				if (candidate_score > best_score)
				{
					best_score = candidate_score;
					best_center_x = candidate_center_x;
					best_center_y = candidate_center_y;
				}
			}
		}

		center_x = best_center_x;
		center_y = best_center_y;

		{
			char search_log[128];
			const long score_milli = (long)(best_score * 1000.0f + 0.5f);
			(void)DebugConsole_Snprintf(
				search_log,
				sizeof(search_log),
				"[AI] Polar center search: best=(%ld,%ld) score=%ld.%03ld\r\n",
				(long)center_x,
				(long)center_y,
				score_milli / 1000L,
				labs(score_milli % 1000L));
			(void)DebugConsole_WriteString(search_log);
		}
	}

	/* Step 2: polar-project the resized crop into the live 7-channel tensor. */
	{
		for (size_t row = 0U; row < output_dim; ++row)
		{
			for (size_t col = 0U; col < output_dim; ++col)
			{
				const float angle = two_pi * (float)col / (float)bins;
				const float radius = ((float)row / (float)output_dim) * max_radius;
				const float src_x = center_x + radius * cosf(angle);
				const float src_y = center_y - radius * sinf(angle);
				float out_r = 0.0f;
				float out_g = 0.0f;
				float out_b = 0.0f;

				if ((src_x >= 0.0f) && (src_x < (float)output_dim) &&
				    (src_y >= 0.0f) && (src_y < (float)output_dim))
				{
					AppAI_ReadRgbFromRgbBilinear(
						resized_rgb, output_dim, output_dim,
						src_x, src_y, &out_r, &out_g, &out_b);
				}

				{
					const size_t base = (row * output_dim + col) * channels;
					int32_t q_r = (int32_t)lroundf(out_r * 255.0f) - 128;
					int32_t q_g = (int32_t)lroundf(out_g * 255.0f) - 128;
					int32_t q_b = (int32_t)lroundf(out_b * 255.0f) - 128;
					if (q_r < -128) q_r = -128;
					if (q_r > 127) q_r = 127;
					if (q_g < -128) q_g = -128;
					if (q_g > 127) q_g = 127;
					if (q_b < -128) q_b = -128;
					if (q_b > 127) q_b = 127;
					((int8_t *)input_ptr)[base + 0U] = (int8_t)q_r;
					((int8_t *)input_ptr)[base + 1U] = (int8_t)q_g;
					((int8_t *)input_ptr)[base + 2U] = (int8_t)q_b;
				}

				{
					const float luma_f = 0.299f * out_r + 0.587f * out_g + 0.114f * out_b;
					int32_t luma_q = (int32_t)lroundf(luma_f * 255.0f);
					if (luma_q < 0) luma_q = 0;
					if (luma_q > 255) luma_q = 255;
					polar_luma[row * output_dim + col] = (uint8_t)luma_q;
				}
			}
		}
	}

	/* Step 3: derive edge and vote channels from the polar luma plane using
	 * the same percentile/continuity logic as the offline V28 pipeline. */
	{
		uint32_t angular_hist[256U] = {0U};
		uint32_t radial_hist[256U] = {0U};
		uint32_t luma_hist[256U] = {0U};
		float sum_evidence[APP_AI_POLAR_VOTE_BINS] = {0.0f};
		float continuity[APP_AI_POLAR_VOTE_BINS] = {0.0f};
		float score[APP_AI_POLAR_VOTE_BINS] = {0.0f};
		const uint32_t total_luma_samples = (uint32_t)(output_dim * output_dim);
		const uint32_t total_gradient_samples = (uint32_t)((output_dim - 2U) * output_dim);
		const float row_denominator = (output_dim > 1U) ? (float)(output_dim - 1U) : 1.0f;
		uint32_t radial_mask_rows = 0U;

		for (size_t row = 0U; row < output_dim; ++row)
		{
			for (size_t col = 0U; col < output_dim; ++col)
			{
				luma_hist[polar_luma[row * output_dim + col]]++;
			}
		}

		for (size_t row = 1U; row + 1U < output_dim; ++row)
		{
			for (size_t col = 0U; col < output_dim; ++col)
			{
				const size_t col_prev = (col == 0U) ? (output_dim - 1U) : (col - 1U);
				const size_t col_next = (col == output_dim - 1U) ? 0U : (col + 1U);
				const int32_t gx = (int32_t)polar_luma[row * output_dim + col_next]
					- (int32_t)polar_luma[row * output_dim + col_prev];
				const int32_t gy = (int32_t)polar_luma[(row + 1U) * output_dim + col]
					- (int32_t)polar_luma[(row - 1U) * output_dim + col];
				const uint8_t abs_gx = (uint8_t)((gx >= 0) ? gx : -gx);
				const uint8_t abs_gy = (uint8_t)((gy >= 0) ? gy : -gy);
				angular_hist[abs_gx]++;
				radial_hist[abs_gy]++;
			}
		}

		const uint8_t luma_threshold = AppAI_FindHistogramPercentile(
			luma_hist, 256U, total_luma_samples, 0.38f);
		const uint8_t angular_threshold = AppAI_FindHistogramPercentile(
			angular_hist, 256U, total_gradient_samples, 0.99f);
		const uint8_t radial_threshold = AppAI_FindHistogramPercentile(
			radial_hist, 256U, total_gradient_samples, 0.99f);
		const float angular_inv = (angular_threshold > 0U) ? (1.0f / (float)angular_threshold) : 1.0f;
		const float radial_inv = (radial_threshold > 0U) ? (1.0f / (float)radial_threshold) : 1.0f;
		const float luma_threshold_f = (float)luma_threshold / 255.0f;

		for (size_t row = 0U; row < output_dim; ++row)
		{
			const float row_position = (float)row / row_denominator;
			const float radial_mask = ((row_position >= 0.18f) && (row_position <= 0.95f)) ? 1.0f : 0.0f;
			if (radial_mask > 0.5f)
			{
				radial_mask_rows++;
			}
		}
		if (radial_mask_rows == 0U)
		{
			radial_mask_rows = 1U;
		}

		for (size_t row = 0U; row < output_dim; ++row)
		{
			for (size_t col = 0U; col < output_dim; ++col)
			{
				const size_t base = (row * output_dim + col) * channels;
				const float luma_val = (float)polar_luma[row * output_dim + col] / 255.0f;
				float ang_val = 0.0f;
				float rad_val = 0.0f;
				if ((row > 0U) && (row + 1U < output_dim))
				{
					const size_t col_prev = (col == 0U) ? (output_dim - 1U) : (col - 1U);
					const size_t col_next = (col == output_dim - 1U) ? 0U : (col + 1U);
					const int32_t gx = (int32_t)polar_luma[row * output_dim + col_next]
						- (int32_t)polar_luma[row * output_dim + col_prev];
					const int32_t gy = (int32_t)polar_luma[(row + 1U) * output_dim + col]
						- (int32_t)polar_luma[(row - 1U) * output_dim + col];
					const int32_t abs_gx = (gx >= 0) ? gx : -gx;
					const int32_t abs_gy = (gy >= 0) ? gy : -gy;
					ang_val = (float)abs_gx * angular_inv;
					rad_val = (float)abs_gy * radial_inv;
					if (row == 1U)
					{
						/* Keep the compiler honest about the normalized range. */
						if (ang_val < 0.0f) ang_val = 0.0f;
						if (rad_val < 0.0f) rad_val = 0.0f;
					}
				}

				{
					int32_t q_luma = (int32_t)polar_luma[row * output_dim + col] - 128;
					((int8_t *)input_ptr)[base + 3U] = (int8_t)q_luma;
				}
				{
					int32_t q_ang = (int32_t)lroundf(ang_val * 255.0f) - 128;
					if (q_ang < -128) q_ang = -128;
					if (q_ang > 127) q_ang = 127;
					((int8_t *)input_ptr)[base + 4U] = (int8_t)q_ang;
				}
				{
					int32_t q_rad = (int32_t)lroundf(rad_val * 255.0f) - 128;
					if (q_rad < -128) q_rad = -128;
					if (q_rad > 127) q_rad = 127;
					((int8_t *)input_ptr)[base + 5U] = (int8_t)q_rad;
				}

				{
					const float row_position = (float)row / row_denominator;
					const float radial_mask = ((row_position >= 0.18f) && (row_position <= 0.95f)) ? 1.0f : 0.0f;
					const float darkness = (0.62f - luma_val) / 0.62f;
					const float clipped_darkness = (darkness > 0.0f) ? ((darkness < 1.0f) ? darkness : 1.0f) : 0.0f;
					const float edge = (ang_val > 1.0f) ? 1.0f : ang_val;
					const float evidence = (0.65f * clipped_darkness + 0.35f * edge) * radial_mask;
					sum_evidence[col] += evidence;
				}
			}
		}

		{
			float score_min = 0.0f;
			float score_max = 0.0f;
			for (size_t col = 0U; col < output_dim; ++col)
			{
				float best_run = 0.0f;
				float run = 0.0f;
				for (size_t row = 0U; row < output_dim; ++row)
				{
					const float row_position = (float)row / row_denominator;
					const float radial_mask = ((row_position >= 0.18f) && (row_position <= 0.95f)) ? 1.0f : 0.0f;
					const uint8_t luma_q = polar_luma[row * output_dim + col];
					if ((radial_mask > 0.5f) && (luma_q <= luma_threshold))
					{
						run += 1.0f;
						if (run > best_run)
						{
							best_run = run;
						}
					}
					else
					{
						run = 0.0f;
					}
				}
				continuity[col] = best_run / (float)output_dim;
				score[col] = (0.70f * (sum_evidence[col] / (float)radial_mask_rows)) + (0.30f * continuity[col]);
			}

			for (size_t col = 0U; col < output_dim; ++col)
			{
				const size_t prev = (col == 0U) ? (output_dim - 1U) : (col - 1U);
				const size_t next = (col == (output_dim - 1U)) ? 0U : (col + 1U);
				score[col] = (0.50f * score[col]) + (0.25f * score[prev]) + (0.25f * score[next]);
			}
			score_min = score[0U];
			score_max = score[0U];
			for (size_t col = 1U; col < output_dim; ++col)
			{
				if (score[col] < score_min)
				{
					score_min = score[col];
				}
				if (score[col] > score_max)
				{
					score_max = score[col];
				}
			}
			for (size_t col = 0U; col < output_dim; ++col)
			{
				float prior = score[col] - score_min;
				if (score_max > score_min)
				{
					prior = prior / (score_max - score_min);
				}
				if (prior < 0.0f)
				{
					prior = 0.0f;
				}
				else if (prior > 1.0f)
				{
					prior = 1.0f;
				}

				{
					const int32_t q_prior = (int32_t)lroundf(prior * 255.0f) - 128;
					for (size_t row = 0U; row < output_dim; ++row)
					{
						const size_t pixel_base = (row * output_dim + col) * channels;
						int32_t clipped = q_prior;
						if (clipped < -128) clipped = -128;
						if (clipped > 127) clipped = 127;
						((int8_t *)input_ptr)[pixel_base + 6U] = (int8_t)clipped;
					}
				}
			}
		}
	}

	(void)DebugConsole_WriteString("[AI] Polar preprocess OK.\r\n");
	return true;
}

static float AppAI_ClampNormalizedFloat(float value)
{
	/* Add additional validation for NaN and infinity */
	if (!isfinite(value))
	{
		DebugConsole_Printf("[AI] ClampNormalizedFloat: Invalid value: %f\r\n", (double)value);
		return 0.0f;
	}

	if (value < 0.0f)
	{
		return 0.0f;
	}

	if (value > 1.0f)
	{
		return 1.0f;
	}

	return value;
}

static uint8_t __attribute__((noinline)) AppAI_ReadYuv422Luma(
	const uint8_t *frame_bytes, size_t frame_size_bytes,
	size_t frame_width_pixels, size_t source_x, size_t source_y)
{
	if (frame_bytes == NULL)
	{
		return 0U;
	}

	if (frame_size_bytes < 4U)
	{
		return 0U;
	}

	if (frame_width_pixels == 0U)
	{
		return 0U;
	}

	if (source_x >= frame_width_pixels)
	{
		return 0U;
	}

	if (source_y >= (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS)
	{
		return 0U;
	}

	const size_t pair_x = source_x & ~1U;
	const size_t source_index = ((source_y * frame_width_pixels) + pair_x) * 2U;
	if (source_index >= frame_size_bytes)
	{
		return 0U;
	}

	const bool is_second_pixel = ((source_x & 1U) != 0U);
	const size_t read_index = source_index + (is_second_pixel ? 2U : 0U);
	if (read_index >= frame_size_bytes)
	{
		return 0U;
	}

	return frame_bytes[read_index];
}

static void AppAI_ReadYuv422Quartet(const uint8_t *frame_bytes,
									size_t frame_size_bytes,
									size_t frame_width_pixels, size_t source_x, size_t source_y,
									uint8_t *quad_out)
{
	if ((frame_bytes == NULL) || (quad_out == NULL) ||
		(frame_size_bytes < 4U) ||
		(source_x >= frame_width_pixels) ||
		(source_y >= (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS))
	{
		return;
	}

	const size_t pair_x = source_x & ~1U;
	const size_t source_index = ((source_y * frame_width_pixels) + pair_x) * 2U;
	if ((source_index + 3U) >= frame_size_bytes)
	{
		return;
	}

	quad_out[0] = frame_bytes[source_index + 0U];
	quad_out[1] = frame_bytes[source_index + 1U];
	quad_out[2] = frame_bytes[source_index + 2U];
	quad_out[3] = frame_bytes[source_index + 3U];
}

static float AppAI_ReadNormalizedGrayFromYuv422Pixel(const uint8_t *frame_bytes,
													 size_t frame_size_bytes,
													 size_t frame_width_pixels, size_t source_x, size_t source_y)
{
	/* Add additional bounds checking and validation */
	if ((frame_bytes == NULL) || (frame_size_bytes < 4U) ||
		(frame_width_pixels == 0U) || (source_x >= frame_width_pixels) ||
		(source_y >= (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS))
	{
		DebugConsole_Printf("[AI] ReadNormalizedGrayFromYuv422Pixel: Invalid parameters or out of bounds\r\n");
		return 0.0f;
	}

	const uint8_t luma = AppAI_ReadYuv422Luma(frame_bytes, frame_size_bytes, frame_width_pixels,
											  source_x, source_y);
	const float normalized = ((float)luma) / 255.0f;

	return AppAI_ClampNormalizedFloat(normalized);
}

static void AppAI_ReadRgbFromYuv422Pixel(const uint8_t *frame_bytes,
										 size_t frame_size_bytes,
										 size_t frame_width_pixels, size_t source_x, size_t source_y,
										 float *r_out, float *g_out, float *b_out)
{
	/* Add additional bounds checking and validation */
	if (frame_bytes == NULL)
	{
		DebugConsole_Printf("[AI] ReadRgbFromYuv422Pixel: NULL frame_bytes pointer\r\n");
		if (r_out != NULL)
		{
			*r_out = 0.0f;
		}
		if (g_out != NULL)
		{
			*g_out = 0.0f;
		}
		if (b_out != NULL)
		{
			*b_out = 0.0f;
		}
		return;
	}

	if (frame_size_bytes < 4U)
	{
		DebugConsole_Printf("[AI] ReadRgbFromYuv422Pixel: frame_size_bytes too small: %lu\r\n", (unsigned long)frame_size_bytes);
		if (r_out != NULL)
		{
			*r_out = 0.0f;
		}
		if (g_out != NULL)
		{
			*g_out = 0.0f;
		}
		if (b_out != NULL)
		{
			*b_out = 0.0f;
		}
		return;
	}

	if (frame_width_pixels == 0U)
	{
		DebugConsole_Printf("[AI] ReadRgbFromYuv422Pixel: frame_width_pixels is zero\r\n");
		if (r_out != NULL)
		{
			*r_out = 0.0f;
		}
		if (g_out != NULL)
		{
			*g_out = 0.0f;
		}
		if (b_out != NULL)
		{
			*b_out = 0.0f;
		}
		return;
	}

	if (source_x >= frame_width_pixels)
	{
		DebugConsole_Printf("[AI] ReadRgbFromYuv422Pixel: source_x out of bounds: %lu >= %lu\r\n", 
						   (unsigned long)source_x, (unsigned long)frame_width_pixels);
		if (r_out != NULL)
		{
			*r_out = 0.0f;
		}
		if (g_out != NULL)
		{
			*g_out = 0.0f;
		}
		if (b_out != NULL)
		{
			*b_out = 0.0f;
		}
		return;
	}

	if (source_y >= (size_t)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS)
	{
		DebugConsole_Printf("[AI] ReadRgbFromYuv422Pixel: source_y out of bounds: %lu >= %lu\r\n", 
						   (unsigned long)source_y, (unsigned long)APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS);
		if (r_out != NULL)
		{
			*r_out = 0.0f;
		}
		if (g_out != NULL)
		{
			*g_out = 0.0f;
		}
		if (b_out != NULL)
		{
			*b_out = 0.0f;
		}
		return;
	}

	/* Add additional validation for frame dimensions */
	if (frame_width_pixels != (size_t)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS)
	{
		DebugConsole_Printf("[AI] ReadRgbFromYuv422Pixel: frame_width_pixels mismatch: %lu != %lu\r\n", 
						   (unsigned long)frame_width_pixels, (unsigned long)APP_AI_CAPTURE_FRAME_WIDTH_PIXELS);
		/* Continue processing but log the mismatch */
	}

#if APP_AI_YUV422_INPUT_LUMA_ONLY
	const float gray = AppAI_ReadNormalizedGrayFromYuv422Pixel(frame_bytes,
															   frame_size_bytes,
															   frame_width_pixels, source_x, source_y);
	const float r = gray;
	const float g = gray;
	const float b = gray;
#else
	const size_t pair_x = source_x & ~1U;
	const size_t source_index = ((source_y * frame_width_pixels) + pair_x) * 2U;
	
	/* Add bounds checking for source_index calculation */
	if (source_index >= frame_size_bytes)
	{
		DebugConsole_Printf("[AI] ReadRgbFromYuv422Pixel: source_index out of bounds: %lu >= %lu\r\n", 
						   (unsigned long)source_index, (unsigned long)frame_size_bytes);
		if (r_out != NULL)
		{
			*r_out = 0.0f;
		}
		if (g_out != NULL)
		{
			*g_out = 0.0f;
		}
		if (b_out != NULL)
		{
			*b_out = 0.0f;
		}
		return;
	}

	const bool is_second_pixel = ((source_x & 1U) != 0U);

	/* The RGB path touches three adjacent bytes per pixel, so make the read
	 * bounds explicit before we let the compiler emit the loads. */
	if ((source_index + 3U) >= frame_size_bytes)
	{
		DebugConsole_Printf("[AI] ReadRgbFromYuv422Pixel: source_index+3 out of bounds: %lu >= %lu\r\n", 
						   (unsigned long)(source_index + 3U), (unsigned long)frame_size_bytes);
		if (r_out != NULL)
		{
			*r_out = 0.0f;
		}
		if (g_out != NULL)
		{
			*g_out = 0.0f;
		}
		if (b_out != NULL)
		{
			*b_out = 0.0f;
		}
		return;
	}

	/* Add additional validation that we're not reading beyond the end of the buffer */
	const size_t y_index = source_index + (is_second_pixel ? 2U : 0U);
	const size_t u_index = source_index + 1U;
	const size_t v_index = source_index + 3U;
	
	if (y_index >= frame_size_bytes || u_index >= frame_size_bytes || v_index >= frame_size_bytes)
	{
		DebugConsole_Printf("[AI] ReadRgbFromYuv422Pixel: Individual indices out of bounds: y=%lu, u=%lu, v=%lu, frame_size=%lu\r\n", 
						   (unsigned long)y_index, (unsigned long)u_index, (unsigned long)v_index, 
						   (unsigned long)frame_size_bytes);
		if (r_out != NULL)
		{
			*r_out = 0.0f;
		}
		if (g_out != NULL)
		{
			*g_out = 0.0f;
		}
		if (b_out != NULL)
		{
			*b_out = 0.0f;
		}
		return;
	}

	const float y = ((float)frame_bytes[y_index] - 16.0f) * 1.1643836f;
	const float u = (float)frame_bytes[u_index] - 128.0f;
	const float v = (float)frame_bytes[v_index] - 128.0f;
	const float r = (y + (1.5960268f * v)) / 255.0f;
	const float g = (y - (0.3917623f * u) - (0.8129677f * v)) / 255.0f;
	const float b = (y + (2.0172322f * u)) / 255.0f;
#endif

	if (r_out != NULL)
	{
		*r_out = AppAI_ClampNormalizedFloat(r);
	}
	if (g_out != NULL)
	{
		*g_out = AppAI_ClampNormalizedFloat(g);
	}
	if (b_out != NULL)
	{
		*b_out = AppAI_ClampNormalizedFloat(b);
	}
}

static void AppAI_BlankTipFocusLowerInset(int8_t *input, size_t width,
	size_t height)
{
	if ((input == NULL) || (width == 0U) || (height == 0U))
	{
		return;
	}

	/* Blank the lower inset gauge and the dark blob in the lower center so the
	 * CNN has to pay attention to the main dial. */
	const int32_t mask_center_x = 112;
	const int32_t mask_center_y = 150;
	const int32_t mask_radius_x = 54;
	const int32_t mask_radius_y = 42;
	const int64_t radius_x_sq = (int64_t)mask_radius_x * (int64_t)mask_radius_x;
	const int64_t radius_y_sq = (int64_t)mask_radius_y * (int64_t)mask_radius_y;
	const int64_t ellipse_limit = radius_x_sq * radius_y_sq;

	for (size_t py = 0U; py < height; ++py)
	{
		const int64_t dy = (int64_t)(int32_t)py - (int64_t)mask_center_y;
		const int64_t dy_term = dy * dy * radius_x_sq;

		for (size_t px = 0U; px < width; ++px)
		{
			const int64_t dx = (int64_t)(int32_t)px - (int64_t)mask_center_x;
			const int64_t dx_term = dx * dx * radius_y_sq;

			if ((dx_term + dy_term) <= ellipse_limit)
			{
				const size_t idx = (py * width + px) * 3U;
				input[idx] = 0;
				input[idx + 1U] = 0;
				input[idx + 2U] = 0;
			}
		}
	}
}

static void AppAI_FormatFixedFloat(char *dst, size_t dst_len, float value,
	unsigned decimals)
{
	if ((dst == NULL) || (dst_len == 0U))
	{
		return;
	}

	if (!isfinite(value))
	{
		(void)DebugConsole_Snprintf(dst, dst_len, "NaN");
		return;
	}

	const bool negative = (value < 0.0f);
	const double magnitude = negative ? -(double)value : (double)value;
	unsigned long whole = (unsigned long)magnitude;
	double fraction = magnitude - (double)whole;
	unsigned long scale = 1UL;

	for (unsigned i = 0U; i < decimals; ++i)
	{
		scale *= 10UL;
	}

	unsigned long scaled = (unsigned long)(fraction * (double)scale + 0.5);
	if (scaled >= scale)
	{
		scaled = 0UL;
		whole += 1UL;
	}

	if (decimals == 0U)
	{
		(void)DebugConsole_Snprintf(dst, dst_len, "%s%lu",
			negative ? "-" : "", whole);
		return;
	}

	(void)DebugConsole_Snprintf(dst, dst_len, "%s%lu.%0*lu",
		negative ? "-" : "", whole, (int)decimals, scaled);
}

static void AppAI_ReadRgbFromYuv422Bilinear(const uint8_t *frame_bytes,
											size_t frame_size_bytes,
											size_t frame_width_pixels,
											size_t frame_height_pixels,
											float source_x, float source_y,
											float *r_out, float *g_out, float *b_out)
{
	/* Add additional bounds checking and validation */
	if ((frame_bytes == NULL) || (frame_size_bytes < 4U) ||
		(frame_width_pixels == 0U) || (frame_height_pixels == 0U))
	{
		DebugConsole_Printf("[AI] ReadRgbFromYuv422Bilinear: Invalid parameters\r\n");
		if (r_out != NULL)
		{
			*r_out = 0.0f;
		}
		if (g_out != NULL)
		{
			*g_out = 0.0f;
		}
		if (b_out != NULL)
		{
			*b_out = 0.0f;
		}
		return;
	}

	/* Add bounds checking for source coordinates */
	if (source_x < 0.0f || source_y < 0.0f)
	{
		DebugConsole_Printf("[AI] ReadRgbFromYuv422Bilinear: Negative coordinates: x=%f, y=%f\r\n", 
						   (double)source_x, (double)source_y);
		source_x = (source_x < 0.0f) ? 0.0f : source_x;
		source_y = (source_y < 0.0f) ? 0.0f : source_y;
	}

	const float max_x = (frame_width_pixels > 0U) ? (float)(frame_width_pixels - 1U)
												 : 0.0f;
	const float max_y = (frame_height_pixels > 0U) ? (float)(frame_height_pixels - 1U)
												  : 0.0f;

	/* Add additional validation for max values */
	if (max_x < 0.0f || max_y < 0.0f)
	{
		DebugConsole_Printf("[AI] ReadRgbFromYuv422Bilinear: Invalid max values: max_x=%f, max_y=%f\r\n", 
						   (double)max_x, (double)max_y);
		if (r_out != NULL)
		{
			*r_out = 0.0f;
		}
		if (g_out != NULL)
		{
			*g_out = 0.0f;
		}
		if (b_out != NULL)
		{
			*b_out = 0.0f;
		}
		return;
	}

	float clamped_x = source_x;
	float clamped_y = source_y;
	float r00 = 0.0f;
	float g00 = 0.0f;
	float b00 = 0.0f;
	float r10 = 0.0f;
	float g10 = 0.0f;
	float b10 = 0.0f;
	float r01 = 0.0f;
	float g01 = 0.0f;
	float b01 = 0.0f;
	float r11 = 0.0f;
	float g11 = 0.0f;
	float b11 = 0.0f;

	if (clamped_x > max_x)
	{
		clamped_x = max_x;
	}

	if (clamped_y > max_y)
	{
		clamped_y = max_y;
	}

	/* Add additional validation for clamped coordinates */
	if (clamped_x < 0.0f || clamped_y < 0.0f)
	{
		DebugConsole_Printf("[AI] ReadRgbFromYuv422Bilinear: Clamped coordinates negative: x=%f, y=%f\r\n", 
						   (double)clamped_x, (double)clamped_y);
		if (r_out != NULL)
		{
			*r_out = 0.0f;
		}
		if (g_out != NULL)
		{
			*g_out = 0.0f;
		}
		if (b_out != NULL)
		{
			*b_out = 0.0f;
		}
		return;
	}

	const size_t x0 = (size_t)floorf(clamped_x);
	const size_t y0 = (size_t)floorf(clamped_y);
	
	/* Add bounds checking for x0 and y0 */
	if (x0 >= frame_width_pixels || y0 >= frame_height_pixels)
	{
		DebugConsole_Printf("[AI] ReadRgbFromYuv422Bilinear: x0/y0 out of bounds: x0=%lu, y0=%lu, width=%lu, height=%lu\r\n", 
						   (unsigned long)x0, (unsigned long)y0, 
						   (unsigned long)frame_width_pixels, (unsigned long)frame_height_pixels);
		if (r_out != NULL)
		{
			*r_out = 0.0f;
		}
		if (g_out != NULL)
		{
			*g_out = 0.0f;
		}
		if (b_out != NULL)
		{
			*b_out = 0.0f;
		}
		return;
	}

	const size_t x1 = (x0 + 1U < frame_width_pixels) ? (x0 + 1U) : x0;
	const size_t y1 = (y0 + 1U < frame_height_pixels) ? (y0 + 1U) : y0;
	const float fx = clamped_x - (float)x0;
	const float fy = clamped_y - (float)y0;
	
	/* Add bounds checking for fx and fy */
	if (fx < 0.0f || fx > 1.0f || fy < 0.0f || fy > 1.0f)
	{
		DebugConsole_Printf("[AI] ReadRgbFromYuv422Bilinear: Invalid fx/fy: fx=%f, fy=%f\r\n", 
						   (double)fx, (double)fy);
	}

	/* Add bounds checking for AppAI_ReadRgbFromYuv422Pixel calls */
	AppAI_ReadRgbFromYuv422Pixel(frame_bytes, frame_size_bytes, frame_width_pixels,
								 x0, y0, &r00, &g00, &b00);
	AppAI_ReadRgbFromYuv422Pixel(frame_bytes, frame_size_bytes, frame_width_pixels,
								 x1, y0, &r10, &g10, &b10);
	AppAI_ReadRgbFromYuv422Pixel(frame_bytes, frame_size_bytes, frame_width_pixels,
								 x0, y1, &r01, &g01, &b01);
	AppAI_ReadRgbFromYuv422Pixel(frame_bytes, frame_size_bytes, frame_width_pixels,
								 x1, y1, &r11, &g11, &b11);

	if (r_out != NULL)
	{
		const float top_r = r00 + (fx * (r10 - r00));
		const float bottom_r = r01 + (fx * (r11 - r01));
		*r_out = AppAI_ClampNormalizedFloat(top_r + (fy * (bottom_r - top_r)));
	}

	if (g_out != NULL)
	{
		const float top_g = g00 + (fx * (g10 - g00));
		const float bottom_g = g01 + (fx * (g11 - g01));
		*g_out = AppAI_ClampNormalizedFloat(top_g + (fy * (bottom_g - top_g)));
	}

	if (b_out != NULL)
	{
		const float top_b = b00 + (fx * (b10 - b00));
		const float bottom_b = b01 + (fx * (b11 - b01));
		*b_out = AppAI_ClampNormalizedFloat(top_b + (fy * (bottom_b - top_b)));
	}
}

static void AppAI_ReadRgbFromRgbBilinear(const uint8_t *rgb_bytes,
										size_t frame_width_pixels,
										size_t frame_height_pixels,
										float source_x, float source_y,
										float *r_out, float *g_out, float *b_out)
{
	if ((rgb_bytes == NULL) || (frame_width_pixels == 0U) || (frame_height_pixels == 0U))
	{
		if (r_out != NULL) { *r_out = 0.0f; }
		if (g_out != NULL) { *g_out = 0.0f; }
		if (b_out != NULL) { *b_out = 0.0f; }
		return;
	}

	if (source_x < 0.0f)
	{
		source_x = 0.0f;
	}
	if (source_y < 0.0f)
	{
		source_y = 0.0f;
	}

	const float max_x = (frame_width_pixels > 0U) ? (float)(frame_width_pixels - 1U) : 0.0f;
	const float max_y = (frame_height_pixels > 0U) ? (float)(frame_height_pixels - 1U) : 0.0f;
	float clamped_x = source_x;
	float clamped_y = source_y;
	if (clamped_x > max_x)
	{
		clamped_x = max_x;
	}
	if (clamped_y > max_y)
	{
		clamped_y = max_y;
	}

	const size_t x0 = (size_t)floorf(clamped_x);
	const size_t y0 = (size_t)floorf(clamped_y);
	const size_t x1 = (x0 + 1U < frame_width_pixels) ? (x0 + 1U) : x0;
	const size_t y1 = (y0 + 1U < frame_height_pixels) ? (y0 + 1U) : y0;
	const float fx = clamped_x - (float)x0;
	const float fy = clamped_y - (float)y0;

	const size_t base00 = ((y0 * frame_width_pixels) + x0) * 3U;
	const size_t base10 = ((y0 * frame_width_pixels) + x1) * 3U;
	const size_t base01 = ((y1 * frame_width_pixels) + x0) * 3U;
	const size_t base11 = ((y1 * frame_width_pixels) + x1) * 3U;

	const float r00 = ((float)rgb_bytes[base00 + 0U]) / 255.0f;
	const float g00 = ((float)rgb_bytes[base00 + 1U]) / 255.0f;
	const float b00 = ((float)rgb_bytes[base00 + 2U]) / 255.0f;
	const float r10 = ((float)rgb_bytes[base10 + 0U]) / 255.0f;
	const float g10 = ((float)rgb_bytes[base10 + 1U]) / 255.0f;
	const float b10 = ((float)rgb_bytes[base10 + 2U]) / 255.0f;
	const float r01 = ((float)rgb_bytes[base01 + 0U]) / 255.0f;
	const float g01 = ((float)rgb_bytes[base01 + 1U]) / 255.0f;
	const float b01 = ((float)rgb_bytes[base01 + 2U]) / 255.0f;
	const float r11 = ((float)rgb_bytes[base11 + 0U]) / 255.0f;
	const float g11 = ((float)rgb_bytes[base11 + 1U]) / 255.0f;
	const float b11 = ((float)rgb_bytes[base11 + 2U]) / 255.0f;

	if (r_out != NULL)
	{
		const float top_r = r00 + (fx * (r10 - r00));
		const float bottom_r = r01 + (fx * (r11 - r01));
		*r_out = AppAI_ClampNormalizedFloat(top_r + (fy * (bottom_r - top_r)));
	}
	if (g_out != NULL)
	{
		const float top_g = g00 + (fx * (g10 - g00));
		const float bottom_g = g01 + (fx * (g11 - g01));
		*g_out = AppAI_ClampNormalizedFloat(top_g + (fy * (bottom_g - top_g)));
	}
	if (b_out != NULL)
	{
		const float top_b = b00 + (fx * (b10 - b00));
		const float bottom_b = b01 + (fx * (b11 - b01));
		*b_out = AppAI_ClampNormalizedFloat(top_b + (fy * (bottom_b - top_b)));
	}
}

static uint8_t AppAI_FindHistogramPercentile(const uint32_t *histogram,
											 size_t histogram_len,
											 uint32_t total_count,
											 float percentile)
{
	uint32_t target = 0U;
	uint32_t cumulative = 0U;
	size_t bin = 0U;

	if ((histogram == NULL) || (histogram_len == 0U) || (total_count == 0U))
	{
		return 0U;
	}

	if (percentile < 0.0f)
	{
		percentile = 0.0f;
	}
	if (percentile > 1.0f)
	{
		percentile = 1.0f;
	}

	target = (uint32_t)ceilf(percentile * (float)total_count);
	if (target == 0U)
	{
		target = 1U;
	}

	for (bin = 0U; bin < histogram_len; ++bin)
	{
		cumulative += histogram[bin];
		if (cumulative >= target)
		{
			return (uint8_t)bin;
		}
	}

	return (uint8_t)(histogram_len - 1U);
}

/**
 * @brief Global softargmax on an int8 heatmap.
 *
 * Matches the Python training decode: softargmax_2d() which computes
 * the global weighted centroid over the entire heatmap.
 *
 * Formula: normalized = heatmap / sum(heatmap)
 *          expected_x = sum(normalized * x_coords)
 *          expected_y = sum(normalized * y_coords)
 *
 * Returns false if the heatmap has no positive activations.
 */
static bool AppAI_SoftArgmaxGlobal(const int8_t *heatmap, int32_t size,
	float *x_out, float *y_out)
{
	if ((heatmap == NULL) || (x_out == NULL) || (y_out == NULL) || (size <= 0))
	{
		return false;
	}

	/* First pass: compute global max for numerical stability, then sum */
	float global_max = -128.0f;
	for (int32_t i = 0; i < size * size; i++)
	{
		const float val = (float)heatmap[i];
		if (val > global_max) { global_max = val; }
	}

	/* Second pass: compute exp(heatmap - global_max) and accumulate */
	float total_weight = 0.0f;
	float sum_x = 0.0f;
	float sum_y = 0.0f;

	for (int32_t py = 0; py < size; py++)
	{
		for (int32_t px = 0; px < size; px++)
		{
			const float v = (float)heatmap[py * size + px] - global_max;
			const float e = expf(v);
			total_weight += e;
			sum_x += e * (float)px;
			sum_y += e * (float)py;
		}
	}

	if (total_weight < 1e-6f)
	{
		return false;
	}

	*x_out = sum_x / total_weight;
	*y_out = sum_y / total_weight;
	return true;
}

/**
 * @brief Reset the tip-focus median history to a fresh seed value.
 *
 * When the reading has drifted far from the last published value for several
 * consecutive frames, the historical median needs a hard re-seed so the live
 * value can catch up instead of being pinned forever by an old stable state.
 */
static void AppAI_TipFocus_ResetMedianHistory(float seed_value)
{
	size_t i;

	for (i = 0U; i < APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE; i++)
	{
		app_ai_tip_focus_median_buffer[i] = seed_value;
	}

	app_ai_tip_focus_median_count = APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE;
	app_ai_tip_focus_median_index = 0U;
	app_ai_tip_focus_last_published = seed_value;
	app_ai_tip_focus_last_published_valid = true;
	app_ai_tip_focus_outlier_streak = 0U;
}

/**
 * @brief 3-sample median filter with outlier rejection.
 *
 * Maintains a ring buffer of accepted tip-focus temperatures.  If the new
 * value deviates from the current median by more than the outlier threshold,
 * it is rejected and the last published value is returned unchanged.
 */
static float AppAI_TipFocus_MedianFilter(float new_value)
{
	/* If no buffer history yet, initialise with the first value */
	if (app_ai_tip_focus_median_count < APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE)
	{
		app_ai_tip_focus_median_buffer[app_ai_tip_focus_median_count] = new_value;
		app_ai_tip_focus_median_count++;
		app_ai_tip_focus_last_published = new_value;
		app_ai_tip_focus_outlier_streak = 0U;
		return new_value;
	}

	/* Compute current median */
	float sorted[APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE];
	(void)memcpy(sorted, app_ai_tip_focus_median_buffer, sizeof(sorted));
	/* Simple bubble sort for 3 elements */
	for (size_t i = 0U; i < APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE; i++)
	{
		for (size_t j = i + 1U; j < APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE; j++)
		{
			if (sorted[i] > sorted[j])
			{
				const float tmp = sorted[i];
				sorted[i] = sorted[j];
				sorted[j] = tmp;
			}
		}
	}
	const float median = sorted[APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE / 2U];

	/* Outlier rejection: if new value is too far from median, hold last published */
	const float delta = (new_value > median) ? (new_value - median) : (median - new_value);
	if (delta > APP_AI_TIP_FOCUS_MAX_OUTLIER_DELTA_C)
	{
		app_ai_tip_focus_outlier_streak++;
		if (app_ai_tip_focus_outlier_streak >= APP_AI_TIP_FOCUS_OUTLIER_RESET_STREAK)
		{
			DebugConsole_Printf(
				"[AI][TIP_FOCUS] Outlier %.2f C delta=%.2f > %.1f; re-seeding median with %.2f\r\n",
				new_value, delta, APP_AI_TIP_FOCUS_MAX_OUTLIER_DELTA_C,
				new_value);
			AppAI_TipFocus_ResetMedianHistory(new_value);
			return new_value;
		}

		DebugConsole_Printf(
			"[AI][TIP_FOCUS] Outlier %.2f C delta=%.2f > %.1f; holding %.2f (streak=%lu/%u)\r\n",
			new_value, delta, APP_AI_TIP_FOCUS_MAX_OUTLIER_DELTA_C,
			app_ai_tip_focus_last_published,
			(unsigned long)app_ai_tip_focus_outlier_streak,
			(unsigned)APP_AI_TIP_FOCUS_OUTLIER_RESET_STREAK);
		return app_ai_tip_focus_last_published;
	}

	app_ai_tip_focus_outlier_streak = 0U;

	/* Insert new value into ring buffer */
	app_ai_tip_focus_median_buffer[app_ai_tip_focus_median_index] = new_value;
	app_ai_tip_focus_median_index = (app_ai_tip_focus_median_index + 1U)
		% APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE;

	/* Recompute median after insertion */
	(void)memcpy(sorted, app_ai_tip_focus_median_buffer, sizeof(sorted));
	for (size_t i = 0U; i < APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE; i++)
	{
		for (size_t j = i + 1U; j < APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE; j++)
		{
			if (sorted[i] > sorted[j])
			{
				const float tmp = sorted[i];
				sorted[i] = sorted[j];
				sorted[j] = tmp;
			}
		}
	}
	const float new_median = sorted[APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE / 2U];
	app_ai_tip_focus_last_published = new_median;
	return new_median;
}

bool App_AI_GetLastInferenceResult(float *value_out)
{
	if (value_out == NULL)
	{
		return false;
	}
	if (!app_ai_last_inference_valid)
	{
		return false;
	}
	if (!AppAI_IsFiniteFloat(app_ai_last_inference_value))
	{
		return false;
	}
	if (!AppAI_IsPlausibleInferenceValue(app_ai_last_inference_value))
	{
		return false;
	}
	*value_out = app_ai_last_inference_value;
	return true;
}

/**
 * @brief Verify that the rectifier weights have been flashed to xSPI2.
 * @retval true xSPI2 flash contains valid rectifier weights.
 * @retval false xSPI2 flash is empty or corrupted - run flash_boot.bat.
 */
bool AppAI_VerifyRectifierWeights(void)
{
	uint8_t actual_start[APP_AI_XSPI2_PROBE_BYTES] = {0U};
	int32_t read_status;
	size_t i;
	bool match;

	/* Ensure xSPI2 is initialized before probing. */
	if (!app_ai_xspi2_initialized)
	{
		if (!AppAI_ReconfigureXspi2ForRuntime())
		{
			DebugConsole_WriteString(
				"[AI][RECTIFIER] xSPI2 reconfigure for verification failed.\r\n");
			return false;
		}
	}

	/* Read the first 16 bytes from xSPI2 flash at 0x70600000.
	 * The chip offset for rectifier is 0x70600000 - 0x70000000 = 0x600000. */
	const uint32_t rectifier_chip_offset = 0x00600000U;

	read_status = BSP_XSPI_NOR_Read(0U, actual_start,
									rectifier_chip_offset,
									APP_AI_XSPI2_PROBE_BYTES);
	if (read_status != BSP_ERROR_NONE)
	{
		DebugConsole_Printf(
			"[AI][RECTIFIER] xSPI2 read failed: status=%ld\r\n",
			(long)read_status);
		return false;
	}

	/* Compare against expected signature. */
	match = true;
	for (i = 0U; i < APP_AI_XSPI2_PROBE_BYTES; i++)
	{
		if (actual_start[i] != app_ai_rectifier_xspi2_signature_start[i])
		{
			match = false;
			break;
		}
	}

	if (!match)
	{
		DebugConsole_WriteString(
			"[AI][RECTIFIER] xSPI2 flash signature mismatch!\r\n");
		DebugConsole_WriteString(
			"[AI][RECTIFIER] Expected: ");
		for (i = 0U; i < APP_AI_XSPI2_PROBE_BYTES; i++)
		{
			DebugConsole_Printf("%02X", app_ai_rectifier_xspi2_signature_start[i]);
		}
		DebugConsole_WriteString("\r\n[AI][RECTIFIER] Actual:   ");
		for (i = 0U; i < APP_AI_XSPI2_PROBE_BYTES; i++)
		{
			DebugConsole_Printf("%02X", actual_start[i]);
		}
		DebugConsole_WriteString("\r\n");
		DebugConsole_WriteString(
			"[AI][RECTIFIER] CRITICAL: Run flash_boot.bat to flash atonbuf.rectifier.xSPI2.raw\r\n");
		DebugConsole_WriteString(
			"[AI][RECTIFIER] to 0x70600000 before running inference.\r\n");
		return false;
	}

	DebugConsole_WriteString(
		"[AI][RECTIFIER] xSPI2 weights signature OK.\r\n");
	return true;
}

/**
 * @brief Verify that the tip-focus weights have been flashed to xSPI2.
 * @retval true xSPI2 flash contains valid tip-focus weights.
 * @retval false xSPI2 flash is empty or corrupted - run flash_boot.bat.
 */
bool AppAI_VerifyTipFocusWeights(void)
{
	uint8_t actual_start[APP_AI_XSPI2_PROBE_BYTES] = {0U};
	int32_t read_status;
	size_t i;
	bool match;

	/* Ensure xSPI2 is initialized before probing. */
	if (!app_ai_xspi2_initialized)
	{
		if (!AppAI_ReconfigureXspi2ForRuntime())
		{
			DebugConsole_WriteString(
				"[AI][TIP_FOCUS] xSPI2 reconfigure for verification failed.\r\n");
			return false;
		}
	}

	/* Read the first 16 bytes from xSPI2 flash at 0x70400000.
	 * The chip offset for tip-focus is 0x70400000 - 0x70000000 = 0x400000. */
	const uint32_t tip_focus_chip_offset = 0x00400000U;

	read_status = BSP_XSPI_NOR_Read(0U, actual_start,
									tip_focus_chip_offset,
									APP_AI_XSPI2_PROBE_BYTES);
	if (read_status != BSP_ERROR_NONE)
	{
		DebugConsole_Printf(
			"[AI][TIP_FOCUS] xSPI2 read failed: status=%ld\r\n",
			(long)read_status);
		return false;
	}

	/* Compare against expected signature. */
	match = true;
	for (i = 0U; i < APP_AI_XSPI2_PROBE_BYTES; i++)
	{
		if (actual_start[i] != app_ai_tip_focus_xspi2_signature_start[i])
		{
			match = false;
			break;
		}
	}

	if (!match)
	{
		DebugConsole_WriteString(
			"[AI][TIP_FOCUS] xSPI2 flash signature mismatch!\r\n");
		DebugConsole_WriteString(
			"[AI][TIP_FOCUS] Expected: ");
		for (i = 0U; i < APP_AI_XSPI2_PROBE_BYTES; i++)
		{
			DebugConsole_Printf("%02X", app_ai_tip_focus_xspi2_signature_start[i]);
		}
		DebugConsole_WriteString("\r\n[AI][TIP_FOCUS] Actual:   ");
		for (i = 0U; i < APP_AI_XSPI2_PROBE_BYTES; i++)
		{
			DebugConsole_Printf("%02X", actual_start[i]);
		}
		DebugConsole_WriteString("\r\n");
		DebugConsole_WriteString(
			"[AI][TIP_FOCUS] CRITICAL: Run flash_boot.bat to flash atonbuf.tip_focus.xSPI2.raw\r\n");
		DebugConsole_WriteString(
			"[AI][TIP_FOCUS] to 0x70400000 before running inference.\r\n");
		return false;
	}

	DebugConsole_WriteString(
		"[AI][TIP_FOCUS] xSPI2 weights signature OK.\r\n");
	return true;
}

bool AppAI_Xspi2EnsureMemoryMappedMode(void)
{
	/* Fast-path: if xSPI2 is already in memory-mapped mode, skip directly. */
	if (app_ai_xspi2_mm_enabled)
	{
		return true;
	}

	/* Before enabling MM mode, make sure the xSPI2 peripheral is initialized
	 * and the flash is accessible.  If the peripheral was left de-inited by a
	 * prior stage, re-init it first. */
	if (!app_ai_xspi2_initialized)
	{
		if (!AppAI_ReconfigureXspi2ForRuntime())
		{
			(void)DebugConsole_WriteString("[AI] xSPI2 runtime reconfigure failed.\r\n");
			return false;
		}
	}

	if (!AppAI_Xspi2EnableMemoryMappedMode())
	{
		return false;
	}

	/* Ensure the memory-mapped window is visible to all bus masters before
	 * the NPU starts reading weight data from xSPI2 flash. */
	__DSB();
	__ISB();
	return true;
}

/* USER CODE END 1 */
