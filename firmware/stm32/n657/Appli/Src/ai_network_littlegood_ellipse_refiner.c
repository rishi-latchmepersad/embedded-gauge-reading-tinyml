/**
 * @file ai_network_littlegood_ellipse_refiner.c
 * @brief Relocatable wrapper for the 256x256 local ellipse refiner.
 */

#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX
#define LL_ATON_RT_RELOC 1
#define BUILD_AI_NETWORK_RELOC 1
#define _network_rt_ctx _network_rt_ctx_gauge_ellipse_littlegood_v2_gamma070
#define _network_entries _network_entries_gauge_ellipse_littlegood_v2_gamma070
#define _network_flags _network_flags_gauge_ellipse_littlegood_v2_gamma070
#define _itf_network _itf_network_gauge_ellipse_littlegood_v2_gamma070
#define LL_ATON_Internal_Buffers_Info_Default_Empty LL_ATON_Internal_Buffers_Info_Default_Empty_gauge_ellipse_littlegood_v2_gamma070
#ifndef NDEBUG
#define NDEBUG
#endif
#ifndef LL_ATON_DBG_BUFFER_INFO_EXCLUDED
#define LL_ATON_DBG_BUFFER_INFO_EXCLUDED 1
#endif

#define MODEL_CONF "../../st_ai_output/packages/gauge_ellipse_littlegood_v2_gamma070_int8_n6_npu/st_ai_ws/build_network/gauge_ellipse_littlegood_v2_gamma070_reloc_conf.h"
#include "../../st_ai_output/packages/gauge_ellipse_littlegood_v2_gamma070_int8_n6_npu/st_ai_output/gauge_ellipse_littlegood_v2_gamma070.h"
#include "ll_aton_NN_interface.h"
#include "ll_aton_rt_user_api.h"
#include "ll_aton_reloc_network.h"
#include "mcu_cache.h"
#include "main.h"
#include <string.h>

extern bool AppAI_Xspi2EnsureMemoryMappedMode(void);
extern bool LL_ATON_OSAL_WfeGuardExpired(void);
extern void LL_ATON_OSAL_DrainWfeSemaphore(void);

LL_ATON_DECLARE_NAMED_NN_INTERFACE(gauge_ellipse_littlegood_v2_gamma070);
#include "../../st_ai_output/packages/gauge_ellipse_littlegood_v2_gamma070_int8_n6_npu/st_ai_ws/build_network/gauge_ellipse_littlegood_v2_gamma070_reloc.c"

extern struct ai_reloc_rt_ctx _network_rt_ctx_gauge_ellipse_littlegood_v2_gamma070;
NN_Instance_TypeDef NN_Instance_gauge_ellipse_littlegood_v2_gamma070 = {
	.network = &NN_Interface_gauge_ellipse_littlegood_v2_gamma070,
	.exec_state = {0}
};

static bool app_ai_littlegood_refiner_ready = false;
static const uint8_t app_ai_littlegood_refiner_signature_start[16] = {
	0xB5U, 0x00U, 0x52U, 0xCEU, 0x15U, 0xF8U, 0xE5U, 0xCEU,
	0xBDU, 0x0AU, 0x56U, 0xA5U, 0xFDU, 0xF9U, 0x43U, 0xFCU
};

/** Verify the exact local-refiner image before NPU execution. */
static bool AppAI_LittleGoodRefiner_VerifyFlashImage(void)
{
	const uint8_t *image = (const uint8_t *)0x70400000UL;
	(void)mcu_cache_invalidate_range((uint32_t)(uintptr_t)image,
		(uint32_t)((uintptr_t)image + 233025U));
	return memcmp(image, app_ai_littlegood_refiner_signature_start, 16U) == 0;
}

/** Restore the generated relocatable runtime context after initialization. */
static void AppAI_LittleGoodRefiner_InstallRelocContext(void)
{
	_network_rt_ctx_gauge_ellipse_littlegood_v2_gamma070.ram_addr = 0x34100000UL;
	_network_rt_ctx_gauge_ellipse_littlegood_v2_gamma070.file_addr = 0x70400000UL;
	_network_rt_ctx_gauge_ellipse_littlegood_v2_gamma070.state = AI_RELOC_RT_STATE_INITIALIZED |
		AI_RELOC_RT_STATE_XIP_MODE;
	NN_Instance_gauge_ellipse_littlegood_v2_gamma070.exec_state.inst_reloc =
		(uint32_t)(uintptr_t)&_network_rt_ctx_gauge_ellipse_littlegood_v2_gamma070;
}

/** Initialize the local refiner and validate its xSPI2 image. */
bool AppAI_LittleGoodRefiner_Init(void)
{
	if (!AppAI_Xspi2EnsureMemoryMappedMode() ||
		!AppAI_LittleGoodRefiner_VerifyFlashImage())
		return false;
	if (!LL_ATON_EC_Network_Init_gauge_ellipse_littlegood_v2_gamma070())
		return false;
	LL_ATON_RT_Init_Network(&NN_Instance_gauge_ellipse_littlegood_v2_gamma070);
	AppAI_LittleGoodRefiner_InstallRelocContext();
	app_ai_littlegood_refiner_ready = true;
	return true;
}

/** Return the generated 256x256x1 refiner input descriptor. */
const LL_Buffer_InfoTypeDef *AppAI_LittleGoodRefiner_InputInfo(void)
{
	return LL_ATON_Input_Buffers_Info_gauge_ellipse_littlegood_v2_gamma070();
}

/** Return the generated four-value refiner output descriptor. */
const LL_Buffer_InfoTypeDef *AppAI_LittleGoodRefiner_OutputInfo(void)
{
	return LL_ATON_Output_Buffers_Info_gauge_ellipse_littlegood_v2_gamma070();
}

/** Run one local-refiner inference while preserving the relocatable r9 base. */
bool AppAI_LittleGoodRefiner_Run(void)
{
	LL_ATON_RT_RetValues_t status = LL_ATON_RT_DONE;
	uintptr_t caller_r9 = 0U;
	const uintptr_t runtime_r9 = (uintptr_t)_network_rt_ctx_gauge_ellipse_littlegood_v2_gamma070.ram_addr;
	const uint32_t start_tick = HAL_GetTick();
	if (!app_ai_littlegood_refiner_ready || !AppAI_Xspi2EnsureMemoryMappedMode())
		return false;
	__asm volatile("mov %0, r9" : "=r"(caller_r9));
	if (!LL_ATON_EC_Network_Init_gauge_ellipse_littlegood_v2_gamma070())
		return false;
	LL_ATON_RT_Init_Network(&NN_Instance_gauge_ellipse_littlegood_v2_gamma070);
	AppAI_LittleGoodRefiner_InstallRelocContext();
	if (!LL_ATON_EC_Inference_Init_gauge_ellipse_littlegood_v2_gamma070())
		return false;
	LL_ATON_OSAL_DrainWfeSemaphore();
	for (;;) {
		if ((HAL_GetTick() - start_tick) >= 10000U) {
			__asm volatile("mov r9, %0" : : "r"(caller_r9) : "r9");
			return false;
		}
		__asm volatile("mov r9, %0" : : "r"(runtime_r9));
		status = LL_ATON_RT_RunEpochBlock(&NN_Instance_gauge_ellipse_littlegood_v2_gamma070);
		if (status == LL_ATON_RT_DONE)
			break;
		if (status == LL_ATON_RT_WFE) {
			LL_ATON_OSAL_WFE();
			__asm volatile("mov r9, %0" : : "r"(runtime_r9));
			if (LL_ATON_OSAL_WfeGuardExpired()) {
				__asm volatile("mov r9, %0" : : "r"(caller_r9) : "r9");
				return false;
			}
			continue;
		}
		if (status == LL_ATON_RT_NO_WFE)
			continue;
		__asm volatile("mov r9, %0" : : "r"(caller_r9) : "r9");
		return false;
	}
	__asm volatile("mov r9, %0" : : "r"(caller_r9) : "r9");
	return true;
}
