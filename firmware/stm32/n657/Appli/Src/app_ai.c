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
#include <stdint.h>

#include "debug_console.h"
#include "stm32n6xx_hal.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define APP_AI_CACHE_LINE_BYTES 32U
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
static bool app_ai_runtime_initialized = false;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
static void AppAI_MaintainCacheRange(void *address, size_t length, bool clean,
		bool invalidate);
/* USER CODE END PFP */

/* USER CODE BEGIN 0 */

/* The generated AI object expects this xSPI2 pool symbol. We only need a small
 * placeholder for the init-only smoke test; later inference will swap this for
 * the real external memory pool wiring. */
__attribute__((used, aligned(APP_AI_CACHE_LINE_BYTES)))
unsigned char _mem_pool_xSPI2_mobilenetv2_scalar_hardcase_warmstart_int8[32];

extern bool LL_ATON_EC_Network_Init_mobilenetv2_scalar_hardcase_warmstart_int8(
		void);
extern bool LL_ATON_EC_Inference_Init_mobilenetv2_scalar_hardcase_warmstart_int8(
		void);

void mcu_cache_clean_range(void *address, size_t length) {
	AppAI_MaintainCacheRange(address, length, true, false);
}

void mcu_cache_invalidate_range(void *address, size_t length) {
	AppAI_MaintainCacheRange(address, length, false, true);
}

bool App_AI_Model_Init(void) {
	if (app_ai_runtime_initialized) {
		return true;
	}

	DebugConsole_Printf("[AI] Initializing model runtime package...\r\n");

	if (!LL_ATON_EC_Network_Init_mobilenetv2_scalar_hardcase_warmstart_int8()) {
		DebugConsole_Printf("[AI] Network init failed.\r\n");
		return false;
	}

	if (!LL_ATON_EC_Inference_Init_mobilenetv2_scalar_hardcase_warmstart_int8()) {
		DebugConsole_Printf("[AI] Inference init failed.\r\n");
		return false;
	}

	app_ai_runtime_initialized = true;
	DebugConsole_Printf("[AI] Model runtime init OK.\r\n");
	return true;
}

/* USER CODE END 0 */

/* USER CODE BEGIN 1 */

static void AppAI_MaintainCacheRange(void *address, size_t length,
		bool clean, bool invalidate) {
	uintptr_t start = 0U;
	uintptr_t end = 0U;

	if ((address == NULL) || (length == 0U)) {
		return;
	}

	start = (uintptr_t) address;
	end = start + length;

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
}

/* USER CODE END 1 */
