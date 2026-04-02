/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_ai.c
 * @brief   Minimal AI runtime bootstrap helpers.
 ******************************************************************************
 */
/* USER CODE END Header */

#include "app_ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stddef.h>
#include <string.h>
#include <stdint.h>

#include "debug_console.h"
#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX
#include "tx_api.h"
#include "ll_aton_rt_user_api.h"
#include "ll_aton.h"
#include "ll_aton_runtime.h"
#include "app_filex.h"
#include "stm32n6xx_nucleo_xspi.h"
#include "npu_cache.h"
#include "stm32n6xx_hal.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define APP_AI_CACHE_LINE_BYTES 32U
#define APP_AI_CAPTURE_FRAME_WIDTH_PIXELS 224U
#define APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS 224U
#define APP_AI_CAPTURE_FRAME_BYTES_PER_PIXEL 2U
#define APP_AI_CAPTURE_FRAME_BYTES \
		(APP_AI_CAPTURE_FRAME_WIDTH_PIXELS * APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS \
				* APP_AI_CAPTURE_FRAME_BYTES_PER_PIXEL)
#define APP_AI_MODEL_INPUT_FLOAT_COUNT \
		(APP_AI_CAPTURE_FRAME_WIDTH_PIXELS * APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS \
				* 3U)
#define APP_AI_MODEL_INPUT_FLOAT_BYTES \
		(APP_AI_MODEL_INPUT_FLOAT_COUNT * sizeof(float))
#define APP_AI_MODEL_OUTPUT_FLOAT_BYTES   sizeof(float)
#define APP_AI_XSPI2_MODEL_IMAGE_PATH     "atonbuf.xSPI2.raw"
#define APP_AI_XSPI2_MODEL_IMAGE_SIZE     3218865UL
#define APP_AI_XSPI2_PROGRAM_CHUNK_BYTES   4096U
#define APP_AI_XSPI2_ERASE_BLOCK_BYTES     (64U * 1024U)
#define APP_AI_XSPI2_PROBE_BYTES           16U
#define APP_AI_XSPI2_SCALE_OFFSET          3218160UL
#define APP_AI_XSPI2_ZERO_POINT_OFFSET     3218864UL
#define APP_AI_FILEX_MEDIA_READY_TIMEOUT_MS 60000U
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
static bool app_ai_runtime_initialized = false;
static bool app_ai_npu_hw_initialized = false;
static bool app_ai_xspi2_initialized = false;
__attribute__((section(".xspi2_pool"), aligned(APP_AI_CACHE_LINE_BYTES)))
uint8_t _mem_pool_xSPI2_mobilenetv2_scalar_hardcase_warmstart_int8[32U] = {
	0U,
};
static uint8_t app_ai_xspi2_program_buffer[APP_AI_XSPI2_PROGRAM_CHUNK_BYTES];
static const uint8_t app_ai_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES] = {
	0xEFU, 0x1BU, 0x2BU, 0xE0U, 0xD7U, 0xE6U, 0xECU, 0x06U,
	0x04U, 0xFFU, 0x34U, 0xECU, 0x1AU, 0xDDU, 0x14U, 0x05U,
};
static const uint8_t app_ai_xspi2_signature_scale[4] = {
	0xF9U, 0x1EU, 0x8DU, 0x3EU,
};
static const uint8_t app_ai_xspi2_signature_zero_point[1] = {
	0xD8U,
};
static const uint8_t app_ai_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES] = {
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U,
	0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0xD8U,
};

/* Declare the generated NN instance locally so the dry-run helper can run the
 * AtoNN runtime on the exact network produced by Cube.AI. */
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(
		mobilenetv2_scalar_hardcase_warmstart_int8);
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
static const LL_Buffer_InfoTypeDef *AppAI_GetInputBufferInfo(void);
static const LL_Buffer_InfoTypeDef *AppAI_GetOutputBufferInfo(void);
static const LL_Buffer_InfoTypeDef *AppAI_FindBufferInfoByName(
		const LL_Buffer_InfoTypeDef *buffer_list, const char *name);
static void AppAI_LogInitFailure(const char *step);
static void AppAI_LogXspi2LoadFailure(const char *step, UINT fx_status,
		int32_t bsp_status);
static void AppAI_LogXspi2ProgramChunkProgress(ULONG chunk_index,
		ULONG flash_offset, ULONG chunk_size);
static void AppAI_LogInferenceResult(
		const LL_Buffer_InfoTypeDef *output_buffer_info);
static int AppAI_ApplyCacheRange(uint32_t start_addr, uint32_t end_addr,
		bool clean, bool invalidate);
static void AppAI_EnableNpuMemoryAndCaches(void);
static void AppAI_ConfigureNpuAccessControl(void);
static void AppAI_ConfigureNpuRisafDefaults(void);
static bool AppAI_EnsureXspi2ModelImageReady(void);
static bool AppAI_Xspi2ModelImageMatchesFlash(void);
static bool AppAI_ProgramXspi2ModelImageFromSd(void);
static bool AppAI_WaitForFileXMediaReady(uint32_t timeout_ms);
static bool AppAI_RuntimeInitStepwise(void);
static bool AppAI_PreprocessYuv422FrameToFloatInput(const uint8_t *frame_bytes,
		size_t frame_size, float *input_buffer, size_t input_float_count);
int mcu_cache_clean_range(uint32_t start_addr, uint32_t end_addr) {
	return AppAI_ApplyCacheRange(start_addr, end_addr, true, false);
}

int mcu_cache_invalidate_range(uint32_t start_addr, uint32_t end_addr) {
	return AppAI_ApplyCacheRange(start_addr, end_addr, false, true);
}

static bool AppAI_EnsureNpuHardwareReady(void) {
	if (app_ai_npu_hw_initialized) {
		return true;
	}

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
	return true;
}

static bool AppAI_EnsureXspi2MemoryReady(void) {
	BSP_XSPI_NOR_Init_t flash = { 0 };
	RCC_PeriphCLKInitTypeDef periph_clk = { 0 };
	int32_t bsp_status = BSP_ERROR_NONE;

	if (app_ai_xspi2_initialized) {
		return true;
	}

	periph_clk.PeriphClockSelection = RCC_PERIPHCLK_XSPI2;
	periph_clk.Xspi2ClockSelection = RCC_XSPI2CLKSOURCE_HCLK;
	if (HAL_RCCEx_PeriphCLKConfig(&periph_clk) != HAL_OK) {
		return false;
	}

	flash.InterfaceMode = BSP_XSPI_NOR_OPI_MODE;
	flash.TransferRate = BSP_XSPI_NOR_DTR_TRANSFER;
	bsp_status = BSP_XSPI_NOR_Init(0U, &flash);
	if (bsp_status != BSP_ERROR_NONE) {
		return false;
	}

	return true;
}

static bool AppAI_Xspi2ReadFlashProbe(const uint32_t flash_offset,
		const uint8_t *expected_bytes, const size_t expected_length) {
	uint8_t flash_bytes[APP_AI_XSPI2_PROBE_BYTES] = { 0U };

	if ((expected_bytes == NULL) || (expected_length == 0U)
			|| (expected_length > APP_AI_XSPI2_PROBE_BYTES)) {
		return false;
	}

	if (BSP_XSPI_NOR_Read(0U, flash_bytes, flash_offset,
			(uint32_t) expected_length) != BSP_ERROR_NONE) {
		return false;
	}

	return (memcmp(flash_bytes, expected_bytes, expected_length) == 0);
}

static bool AppAI_Xspi2ModelImageMatchesFlash(void) {
	return AppAI_Xspi2ReadFlashProbe(0U, app_ai_xspi2_signature_start,
			sizeof(app_ai_xspi2_signature_start))
			&& AppAI_Xspi2ReadFlashProbe(APP_AI_XSPI2_SCALE_OFFSET,
					app_ai_xspi2_signature_scale,
					sizeof(app_ai_xspi2_signature_scale))
			&& AppAI_Xspi2ReadFlashProbe(APP_AI_XSPI2_ZERO_POINT_OFFSET,
					app_ai_xspi2_signature_zero_point,
					sizeof(app_ai_xspi2_signature_zero_point))
			&& AppAI_Xspi2ReadFlashProbe(
					APP_AI_XSPI2_MODEL_IMAGE_SIZE
							- APP_AI_XSPI2_PROBE_BYTES,
					app_ai_xspi2_signature_tail,
					sizeof(app_ai_xspi2_signature_tail));
}

static bool AppAI_ProgramXspi2ModelImageFromSd(void) {
	FX_MEDIA *media_ptr = NULL;
	FX_FILE model_file = { 0 };
	ULONG file_size = 0U;
	ULONG bytes_read = 0U;
	ULONG bytes_remaining = 0U;
	ULONG flash_offset = 0U;
	UINT fx_status = FX_SUCCESS;
	UINT tx_status = TX_SUCCESS;
	int32_t bsp_status = BSP_ERROR_NONE;

	if (!AppAI_WaitForFileXMediaReady(APP_AI_FILEX_MEDIA_READY_TIMEOUT_MS)) {
		AppAI_LogXspi2LoadFailure("FileX not ready", FX_MEDIA_NOT_OPEN,
				BSP_ERROR_NONE);
		return false;
	}

	tx_status = AppFileX_AcquireMediaLock();
	if (tx_status != TX_SUCCESS) {
		AppAI_LogXspi2LoadFailure("media lock", (UINT) tx_status, BSP_ERROR_NONE);
		return false;
	}

	media_ptr = AppFileX_GetMediaHandle();
	if (media_ptr == NULL) {
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("media handle", FX_MEDIA_NOT_OPEN,
				BSP_ERROR_NONE);
		return false;
	}

	if (fx_directory_default_set(media_ptr, FX_NULL) != FX_SUCCESS) {
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("default directory", FX_SUCCESS,
				BSP_ERROR_NONE);
		return false;
	}

	fx_status = fx_file_open(media_ptr, &model_file,
			(CHAR *) APP_AI_XSPI2_MODEL_IMAGE_PATH, FX_OPEN_FOR_READ);
	if (fx_status != FX_SUCCESS) {
		(void) fx_directory_default_set(media_ptr, FX_NULL);
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("file open", fx_status, BSP_ERROR_NONE);
		return false;
	}

	file_size = model_file.fx_file_current_file_size;
	if (file_size != APP_AI_XSPI2_MODEL_IMAGE_SIZE) {
		(void) fx_file_close(&model_file);
		(void) fx_directory_default_set(media_ptr, FX_NULL);
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("file size", FX_SUCCESS, BSP_ERROR_NONE);
		return false;
	}

	fx_status = fx_file_seek(&model_file, 0U);
	if (fx_status != FX_SUCCESS) {
		(void) fx_file_close(&model_file);
		(void) fx_directory_default_set(media_ptr, FX_NULL);
		AppFileX_ReleaseMediaLock();
		AppAI_LogXspi2LoadFailure("file seek", fx_status, BSP_ERROR_NONE);
		return false;
	}

	for (ULONG erase_addr = 0U; erase_addr < file_size;
			erase_addr += APP_AI_XSPI2_ERASE_BLOCK_BYTES) {
		bsp_status = BSP_XSPI_NOR_Erase_Block(0U, erase_addr,
				BSP_XSPI_NOR_ERASE_64K);
		if (bsp_status != BSP_ERROR_NONE) {
			(void) fx_file_close(&model_file);
			(void) fx_directory_default_set(media_ptr, FX_NULL);
			AppFileX_ReleaseMediaLock();
			AppAI_LogXspi2LoadFailure("flash erase", FX_SUCCESS, bsp_status);
			return false;
		}
	}

	bytes_remaining = file_size;
	flash_offset = 0U;
	{
		ULONG chunk_index = 0U;

	while (bytes_remaining > 0U) {
		const ULONG chunk_size = (bytes_remaining > APP_AI_XSPI2_PROGRAM_CHUNK_BYTES)
				? APP_AI_XSPI2_PROGRAM_CHUNK_BYTES
				: bytes_remaining;

		if ((flash_offset == 0U)
				|| ((flash_offset % APP_AI_XSPI2_ERASE_BLOCK_BYTES) == 0U)) {
			AppAI_LogXspi2ProgramChunkProgress(chunk_index, flash_offset,
					chunk_size);
		}

		bytes_read = 0U;
		fx_status = fx_file_read(&model_file, app_ai_xspi2_program_buffer,
				chunk_size, &bytes_read);
		if ((fx_status != FX_SUCCESS) || (bytes_read != chunk_size)) {
			(void) fx_file_close(&model_file);
			(void) fx_directory_default_set(media_ptr, FX_NULL);
			AppFileX_ReleaseMediaLock();
			AppAI_LogXspi2LoadFailure("file read", fx_status, BSP_ERROR_NONE);
			return false;
		}

		bsp_status = BSP_XSPI_NOR_Write(0U, app_ai_xspi2_program_buffer,
				flash_offset, (uint32_t) chunk_size);
		if (bsp_status != BSP_ERROR_NONE) {
			(void) fx_file_close(&model_file);
			(void) fx_directory_default_set(media_ptr, FX_NULL);
			AppFileX_ReleaseMediaLock();
			AppAI_LogXspi2LoadFailure("flash write", FX_SUCCESS, bsp_status);
			return false;
		}

		flash_offset += chunk_size;
		bytes_remaining -= chunk_size;
		chunk_index++;
	}
	}

	(void) fx_file_close(&model_file);
	(void) fx_directory_default_set(media_ptr, FX_NULL);
	AppFileX_ReleaseMediaLock();

	if (!AppAI_Xspi2ModelImageMatchesFlash()) {
		AppAI_LogXspi2LoadFailure("flash verify", FX_SUCCESS, BSP_ERROR_NONE);
		return false;
	}

	return true;
}

static bool AppAI_WaitForFileXMediaReady(uint32_t timeout_ms) {
	const ULONG timeout_ticks = (ULONG) ((timeout_ms + 9U) / 10U);
	const ULONG start_tick = tx_time_get();

	while (!AppFileX_IsMediaReady()) {
		if ((tx_time_get() - start_tick) >= timeout_ticks) {
			return false;
		}

		tx_thread_sleep(1U);
	}

	return true;
}

static bool AppAI_EnsureXspi2ModelImageReady(void) {
	if (app_ai_xspi2_initialized) {
		return true;
	}

	if (!AppAI_Xspi2ModelImageMatchesFlash()) {
		DebugConsole_Printf("[AI] xSPI2 model image missing; programming from SD card.\r\n");
		if (!AppAI_ProgramXspi2ModelImageFromSd()) {
			return false;
		}
	}

	if (BSP_XSPI_NOR_EnableMemoryMappedMode(0U) != BSP_ERROR_NONE) {
		return false;
	}

	app_ai_xspi2_initialized = true;
	DebugConsole_Printf("[AI] xSPI2 model image ready.\r\n");
	return true;
}

bool App_AI_Model_Init(void) {
	if (app_ai_runtime_initialized) {
		return true;
	}

	if (!AppAI_EnsureNpuHardwareReady()) {
		AppAI_LogInitFailure("NPU hardware");
		return false;
	}

	if (!AppAI_EnsureXspi2MemoryReady()) {
		AppAI_LogInitFailure("XSPI2 memory");
		return false;
	}

	if (!AppAI_EnsureXspi2ModelImageReady()) {
		AppAI_LogInitFailure("xSPI2 model image");
		return false;
	}

	if (!AppAI_RuntimeInitStepwise()) {
		AppAI_LogInitFailure("runtime init");
		return false;
	}

	if (!LL_ATON_EC_Network_Init_mobilenetv2_scalar_hardcase_warmstart_int8()) {
		AppAI_LogInitFailure("network init");
		return false;
	}
	LL_ATON_RT_Init_Network(
			&NN_Instance_mobilenetv2_scalar_hardcase_warmstart_int8);

	if (!LL_ATON_EC_Inference_Init_mobilenetv2_scalar_hardcase_warmstart_int8()) {
		AppAI_LogInitFailure("inference init");
		return false;
	}

	app_ai_runtime_initialized = true;
	DebugConsole_Printf("[AI] Model runtime init OK.\r\n");
	return true;
}

static void AppAI_LogInitFailure(const char *step) {
	if (step != NULL) {
		DebugConsole_Printf("[AI] Model runtime init failed at %s.\r\n", step);
	} else {
		DebugConsole_Printf("[AI] Model runtime init failed.\r\n");
	}
}

static void AppAI_LogXspi2LoadFailure(const char *step, UINT fx_status,
		int32_t bsp_status) {
	DebugConsole_Printf("[AI] xSPI2 load failed at %s (fx=%lu bsp=%ld).\r\n",
			(step != NULL) ? step : "unknown",
			(unsigned long) fx_status,
			(long) bsp_status);
}

static void AppAI_LogXspi2ProgramChunkProgress(ULONG chunk_index,
		ULONG flash_offset, ULONG chunk_size) {
	DebugConsole_Printf(
			"[AI] xSPI2 program chunk %lu offset=0x%08lX size=%lu.\r\n",
			(unsigned long) chunk_index,
			(unsigned long) flash_offset,
			(unsigned long) chunk_size);
}

static bool AppAI_RuntimeInitStepwise(void) {
	uint32_t t = 0U;

	/* Let the vendor runtime perform the low-level ATON bring-up and version
	 * compatibility checks. Our wrapper only handles the OSAL and IRQ setup. */
	if (LL_ATON_Init() != LL_ATON_OK) {
		return false;
	}

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

	LL_ATON_OSAL_ENABLE_IRQ(ATON_STD_IRQ_LINE);
	return true;
}

static void AppAI_ConfigureNpuAccessControl(void) {
	RIMC_MasterConfig_t npu_master = { 0 };

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

static void AppAI_EnableNpuMemoryAndCaches(void) {
	/* Mirror the ST NPU examples so the memory fabric is actually usable by
	 * the runtime before the first ATON init call runs. */
	RCC->MEMENR |= RCC_MEMENR_AXISRAM3EN | RCC_MEMENR_AXISRAM4EN
			| RCC_MEMENR_AXISRAM5EN | RCC_MEMENR_AXISRAM6EN
			| RCC_MEMENR_CACHEAXIRAMEN;

	RAMCFG_SRAM2_AXI->CR &= ~RAMCFG_CR_SRAMSD;
	RAMCFG_SRAM3_AXI->CR &= ~RAMCFG_CR_SRAMSD;
	RAMCFG_SRAM4_AXI->CR &= ~RAMCFG_CR_SRAMSD;
	RAMCFG_SRAM5_AXI->CR &= ~RAMCFG_CR_SRAMSD;
	RAMCFG_SRAM6_AXI->CR &= ~RAMCFG_CR_SRAMSD;

	MEMSYSCTL->MSCR |= MEMSYSCTL_MSCR_DCACTIVE_Msk | MEMSYSCTL_MSCR_ICACTIVE_Msk;
}

static uint32_t AppAI_GetRisafMaxAddr(RISAF_TypeDef *risaf) {
	uint32_t max_addr = 0U;

	if ((risaf == RISAF1_S) || (risaf == RISAF1_NS)) {
		max_addr = RISAF1_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF2_S) || (risaf == RISAF2_NS)) {
		max_addr = RISAF2_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF3_S) || (risaf == RISAF3_NS)) {
		max_addr = RISAF3_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF4_S) || (risaf == RISAF4_NS)) {
		max_addr = RISAF4_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF5_S) || (risaf == RISAF5_NS)) {
		max_addr = RISAF5_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF6_S) || (risaf == RISAF6_NS)) {
		max_addr = RISAF6_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF7_S) || (risaf == RISAF7_NS)) {
		max_addr = RISAF7_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF8_S) || (risaf == RISAF8_NS)) {
		max_addr = RISAF8_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF9_S) || (risaf == RISAF9_NS)) {
		max_addr = RISAF9_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF11_S) || (risaf == RISAF11_NS)) {
		max_addr = RISAF11_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF12_S) || (risaf == RISAF12_NS)) {
		max_addr = RISAF12_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF13_S) || (risaf == RISAF13_NS)) {
		max_addr = RISAF13_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF14_S) || (risaf == RISAF14_NS)) {
		max_addr = RISAF14_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF15_S) || (risaf == RISAF15_NS)) {
		max_addr = RISAF15_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF21_S) || (risaf == RISAF21_NS)) {
		max_addr = RISAF21_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF22_S) || (risaf == RISAF22_NS)) {
		max_addr = RISAF22_LIMIT_ADDRESS_SPACE_SIZE;
	} else if ((risaf == RISAF23_S) || (risaf == RISAF23_NS)) {
		max_addr = RISAF23_LIMIT_ADDRESS_SPACE_SIZE;
	}

	return max_addr;
}

static void AppAI_SetRisafDefault(RISAF_TypeDef *risaf) {
	RISAF_BaseRegionConfig_t risaf_conf;
	RISAF_TypeDef *const risaf_hw = (risaf == RISAF12_S) ? RISAF12_NS : risaf;

	risaf_conf.StartAddress = 0x0U;
	risaf_conf.EndAddress = AppAI_GetRisafMaxAddr(risaf_hw);
	risaf_conf.Filtering = RISAF_FILTER_ENABLE;
	risaf_conf.PrivWhitelist = RIF_CID_NONE;
	risaf_conf.ReadWhitelist = RIF_CID_MASK;
	risaf_conf.WriteWhitelist = RIF_CID_MASK;

	risaf_conf.Secure = RIF_ATTRIBUTE_SEC;
	if (risaf == RISAF12_S) {
		
	}
	HAL_RIF_RISAF_ConfigBaseRegion(risaf_hw, 0U, &risaf_conf);
	if (risaf == RISAF12_S) {
		
	}

	risaf_conf.Secure = RIF_ATTRIBUTE_NSEC;
	if (risaf == RISAF12_S) {
		
	}
	HAL_RIF_RISAF_ConfigBaseRegion(risaf_hw, 1U, &risaf_conf);
	if (risaf == RISAF12_S) {
		
	}
}

static void AppAI_ConfigureNpuRisafDefaults(void) {
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
		size_t frame_size) {
	const LL_Buffer_InfoTypeDef *input_info = NULL;
	const LL_Buffer_InfoTypeDef *output_info = NULL;
	float *input_ptr = NULL;
	const float *output_ptr = NULL;
	size_t input_len_bytes = 0U;
	size_t input_float_count = 0U;
	size_t output_len_bytes = 0U;

	if (!App_AI_Model_Init()) {
		
		return false;
	}

	input_info = AppAI_GetInputBufferInfo();
	output_info = AppAI_GetOutputBufferInfo();
	if ((input_info == NULL) || (output_info == NULL)) {
		
		return false;
	}

	input_ptr = (float *) LL_Buffer_addr_start(input_info);
	input_len_bytes = (size_t) LL_Buffer_len(input_info);
	input_float_count = input_len_bytes / sizeof(float);
	if (input_ptr == NULL) {
		
		return false;
	}

	if (!AppAI_PreprocessYuv422FrameToFloatInput(frame_bytes, frame_size,
			input_ptr, input_float_count)) {
		return false;
	}

	/* Make sure the NPU sees the freshly prepared float input. */
	(void) mcu_cache_clean_range((uint32_t) (uintptr_t) input_ptr,
			(uint32_t) ((uintptr_t) input_ptr + input_len_bytes));

	LL_ATON_RT_Reset_Network(&NN_Instance_mobilenetv2_scalar_hardcase_warmstart_int8);

	for (uint32_t epoch_step = 0U;; ++epoch_step) {
		

		const LL_ATON_RT_RetValues_t run_status = LL_ATON_RT_RunEpochBlock(
				&NN_Instance_mobilenetv2_scalar_hardcase_warmstart_int8);

		

		if (run_status == LL_ATON_RT_DONE) {
			break;
		}

		/* Follow the ATON runtime contract so the NPU can wait on its own event. */
		if (run_status == LL_ATON_RT_WFE) {
			
			LL_ATON_OSAL_WFE();
			
		} else {
			tx_thread_relinquish();
		}
	}

	output_ptr = (const float *) LL_Buffer_addr_start(output_info);
	output_len_bytes = (size_t) LL_Buffer_len(output_info);
	if ((output_ptr == NULL) || (output_len_bytes < sizeof(float))) {
		
		return false;
	}

	/* Read the scalar output back after the runtime finishes. */
	(void) mcu_cache_invalidate_range((uint32_t) (uintptr_t) output_ptr,
			(uint32_t) ((uintptr_t) output_ptr + output_len_bytes));

	AppAI_LogInferenceResult(output_info);

	return true;
}

/* USER CODE END 0 */

/* USER CODE BEGIN 1 */

static const LL_Buffer_InfoTypeDef *AppAI_GetInputBufferInfo(void) {
	const LL_Buffer_InfoTypeDef *input_info =
			LL_ATON_Input_Buffers_Info(
					&NN_Instance_mobilenetv2_scalar_hardcase_warmstart_int8);

	if ((input_info == NULL) || (input_info->name == NULL)) {
		return NULL;
	}

	return input_info;
}

static const LL_Buffer_InfoTypeDef *AppAI_GetOutputBufferInfo(void) {
	const LL_Buffer_InfoTypeDef *output_info =
			LL_ATON_Output_Buffers_Info(
					&NN_Instance_mobilenetv2_scalar_hardcase_warmstart_int8);

	if ((output_info == NULL) || (output_info->name == NULL)) {
		return NULL;
	}

	return output_info;
}

static const LL_Buffer_InfoTypeDef *AppAI_FindBufferInfoByName(
		const LL_Buffer_InfoTypeDef *buffer_list, const char *name) {
	if ((buffer_list == NULL) || (name == NULL)) {
		return NULL;
	}

	for (const LL_Buffer_InfoTypeDef *entry = buffer_list; entry->name != NULL;
			++entry) {
		if (strcmp(entry->name, name) == 0) {
			return entry;
		}
	}

	return NULL;
}

static void AppAI_LogInferenceResult(
		const LL_Buffer_InfoTypeDef *output_buffer_info) {
	const LL_Buffer_InfoTypeDef *internal_buffers = NULL;
	const LL_Buffer_InfoTypeDef *raw_output_info = NULL;
	const LL_Buffer_InfoTypeDef *scale_info = NULL;
	const LL_Buffer_InfoTypeDef *zero_point_info = NULL;
	const uint8_t *scale_bytes = NULL;
	union {
		float f;
		uint32_t u;
	} output_bits = {
		.f = 0.0f
	};
	union {
		float f;
		uint32_t u;
	} dequant_scale_bits = {
		.f = 0.0f
	};
	int8_t raw_output_value = 0;
	int8_t dequant_zero_point = 0;

	if (output_buffer_info == NULL) {
		DebugConsole_Printf("[AI] Inference failed: no output buffer.\r\n");
		return;
	}

	output_bits.f = *(const float *) LL_Buffer_addr_start(output_buffer_info);

	internal_buffers =
			LL_ATON_Internal_Buffers_Info(
					&NN_Instance_mobilenetv2_scalar_hardcase_warmstart_int8);
	raw_output_info = AppAI_FindBufferInfoByName(internal_buffers,
			"Gemm_259_out_0");
	scale_info = AppAI_FindBufferInfoByName(internal_buffers,
			"Dequantize_261_x_scale");
	zero_point_info = AppAI_FindBufferInfoByName(internal_buffers,
			"Dequantize_261_x_zero_point");

	if ((raw_output_info != NULL) && (LL_Buffer_addr_start(raw_output_info) != NULL)) {
		raw_output_value = *(const int8_t *) LL_Buffer_addr_start(raw_output_info);
	}

	if ((scale_info != NULL) && (LL_Buffer_addr_start(scale_info) != NULL)) {
		scale_bytes = (const uint8_t *) LL_Buffer_addr_start(scale_info);
		dequant_scale_bits.f = *(const float *) LL_Buffer_addr_start(scale_info);
	}

	if ((zero_point_info != NULL)
			&& (LL_Buffer_addr_start(zero_point_info) != NULL)) {
		dequant_zero_point = *(const int8_t *) LL_Buffer_addr_start(
				zero_point_info);
	}

	DebugConsole_Printf(
			"[AI] Inference OK; raw=0x%02x output=0x%08lx scale=0x%08lx zp=%d\r\n",
			(unsigned int) (uint8_t) raw_output_value,
			(unsigned long) output_bits.u,
			(unsigned long) dequant_scale_bits.u,
			(int) dequant_zero_point);

	if (scale_bytes != NULL) {
		DebugConsole_Printf("[AI] Dequant scale bytes @%p = %02x %02x %02x %02x\r\n",
				(const void *) scale_bytes,
				(unsigned int) scale_bytes[0],
				(unsigned int) scale_bytes[1],
				(unsigned int) scale_bytes[2],
				(unsigned int) scale_bytes[3]);
	}
}

static int AppAI_ApplyCacheRange(uint32_t start_addr, uint32_t end_addr,
		bool clean, bool invalidate) {
	uintptr_t start = 0U;
	uintptr_t end = 0U;

	if (end_addr <= start_addr) {
		return -1;
	}

	start = (uintptr_t) start_addr;
	end = (uintptr_t) end_addr;

	start &= ~((uintptr_t) APP_AI_CACHE_LINE_BYTES - 1U);
	end = (end + APP_AI_CACHE_LINE_BYTES - 1U)
			& ~((uintptr_t) APP_AI_CACHE_LINE_BYTES - 1U);

	if (clean) {
		SCB_CleanDCache_by_Addr((uint32_t *) start, (int32_t) (end - start));
	}

	if (invalidate) {
		SCB_InvalidateDCache_by_Addr((uint32_t *) start,
				(int32_t) (end - start));
	}

	return 0;
}

static bool AppAI_PreprocessYuv422FrameToFloatInput(const uint8_t *frame_bytes,
		size_t frame_size, float *input_ptr, size_t input_float_count) {
	if ((frame_bytes == NULL) || (input_ptr == NULL)) {
		
		return false;
	}

	if ((frame_size < (size_t) APP_AI_CAPTURE_FRAME_BYTES)
			|| (input_float_count < (size_t) APP_AI_MODEL_INPUT_FLOAT_COUNT)) {
		
		return false;
	}

	/* Convert each 2-byte YUV422 sample pair into a grayscale RGB triplet.
	 * This keeps the first dry-run simple while still exercising the real model
	 * runtime on the captured frame contents. */
	for (size_t pixel_index = 0U;
			pixel_index < (size_t) (APP_AI_CAPTURE_FRAME_WIDTH_PIXELS
					* APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS); pixel_index++) {
		const size_t source_index = pixel_index * 2U;
		const float gray = ((float) frame_bytes[source_index]
				+ (float) frame_bytes[source_index + 1U]) * 0.5f
				/ 255.0f;
		const size_t dest_index = pixel_index * 3U;

		input_ptr[dest_index + 0U] = gray;
		input_ptr[dest_index + 1U] = gray;
		input_ptr[dest_index + 2U] = gray;
	}

	return true;
}

/* USER CODE END 1 */
