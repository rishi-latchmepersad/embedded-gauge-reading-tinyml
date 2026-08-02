/**
 * @file ai_network_littlegood_ellipse_proposer.c
 * @brief Relocatable wrapper for the 320x320 full-frame ellipse proposer.
 */

#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX
#define LL_ATON_RT_RELOC 1
#define BUILD_AI_NETWORK_RELOC 1
#define _network_rt_ctx _network_rt_ctx_ocdet_ellipse_320_v2
#define _network_entries _network_entries_ocdet_ellipse_320_v2
#define _network_flags _network_flags_ocdet_ellipse_320_v2
#define _itf_network _itf_network_ocdet_ellipse_320_v2
#define LL_ATON_Internal_Buffers_Info_Default_Empty LL_ATON_Internal_Buffers_Info_Default_Empty_ocdet_ellipse_320_v2
#ifndef NDEBUG
#define NDEBUG
#endif
#ifndef LL_ATON_DBG_BUFFER_INFO_EXCLUDED
#define LL_ATON_DBG_BUFFER_INFO_EXCLUDED 1
#endif

#define MODEL_CONF "../../st_ai_output/packages/ocdet_ellipse_320_v2_int8_n6_npu/st_ai_ws/build_network/ocdet_ellipse_320_v2_reloc_conf.h"
#include "../../st_ai_output/packages/ocdet_ellipse_320_v2_int8_n6_npu/st_ai_output/ocdet_ellipse_320_v2.h"
#include "ll_aton_NN_interface.h"
#include "ll_aton_rt_user_api.h"
#include "ll_aton_reloc_network.h"
#include "mcu_cache.h"
#include "main.h"
#include <string.h>

extern bool AppAI_Xspi2EnsureMemoryMappedMode(void);
extern bool LL_ATON_OSAL_WfeGuardExpired(void);
extern void LL_ATON_OSAL_DrainWfeSemaphore(void);

LL_ATON_DECLARE_NAMED_NN_INTERFACE(ocdet_ellipse_320_v2);
#include "../../st_ai_output/packages/ocdet_ellipse_320_v2_int8_n6_npu/st_ai_ws/build_network/ocdet_ellipse_320_v2_reloc.c"

extern struct ai_reloc_rt_ctx _network_rt_ctx_ocdet_ellipse_320_v2;
NN_Instance_TypeDef NN_Instance_ocdet_ellipse_320_v2 = {
	.network = &NN_Interface_ocdet_ellipse_320_v2,
	.exec_state = {0}
};

static bool app_ai_littlegood_proposer_ready = false;
static const uint8_t app_ai_littlegood_proposer_signature_start[16] = {
	0x0FU, 0x1BU, 0x0DU, 0x2CU, 0xE3U, 0x1EU, 0xE8U, 0x0FU,
	0x28U, 0xF4U, 0x18U, 0x02U, 0xDEU, 0x0DU, 0x18U, 0x00U
};

/** Verify the exact proposer image before any NPU execution. */
static bool AppAI_LittleGoodProposer_VerifyFlashImage(void)
{
	const uint8_t *image = (const uint8_t *)0x70C00000UL;
	(void)mcu_cache_invalidate_range((uint32_t)(uintptr_t)image,
		(uint32_t)((uintptr_t)image + 352401U));
	return memcmp(image, app_ai_littlegood_proposer_signature_start, 16U) == 0;
}

/** Restore the generated relocatable runtime context after initialization. */
static void AppAI_LittleGoodProposer_InstallRelocContext(void)
{
	_network_rt_ctx_ocdet_ellipse_320_v2.ram_addr = 0x34100000UL;
	_network_rt_ctx_ocdet_ellipse_320_v2.file_addr = 0x70C00000UL;
	_network_rt_ctx_ocdet_ellipse_320_v2.state = AI_RELOC_RT_STATE_INITIALIZED |
		AI_RELOC_RT_STATE_XIP_MODE;
	NN_Instance_ocdet_ellipse_320_v2.exec_state.inst_reloc =
		(uint32_t)(uintptr_t)&_network_rt_ctx_ocdet_ellipse_320_v2;
}

/** Initialize the full-frame proposer and validate its xSPI2 image. */
bool AppAI_LittleGoodProposer_Init(void)
{
	if (!AppAI_Xspi2EnsureMemoryMappedMode() ||
		!AppAI_LittleGoodProposer_VerifyFlashImage())
		return false;
	if (!LL_ATON_EC_Network_Init_ocdet_ellipse_320_v2())
		return false;
	LL_ATON_RT_Init_Network(&NN_Instance_ocdet_ellipse_320_v2);
	AppAI_LittleGoodProposer_InstallRelocContext();
	app_ai_littlegood_proposer_ready = true;
	return true;
}

/** Return the generated 320x320x1 proposer input descriptor. */
const LL_Buffer_InfoTypeDef *AppAI_LittleGoodProposer_InputInfo(void)
{
	return LL_ATON_Input_Buffers_Info_ocdet_ellipse_320_v2();
}

/** Return the generated 80x80x6 dense proposer output descriptor. */
const LL_Buffer_InfoTypeDef *AppAI_LittleGoodProposer_OutputInfo(void)
{
	return LL_ATON_Output_Buffers_Info_ocdet_ellipse_320_v2();
}

/** Run one proposer inference while preserving the relocatable r9 base. */
bool AppAI_LittleGoodProposer_Run(void)
{
	LL_ATON_RT_RetValues_t status = LL_ATON_RT_DONE;
	uintptr_t caller_r9 = 0U;
	const uintptr_t runtime_r9 = (uintptr_t)_network_rt_ctx_ocdet_ellipse_320_v2.ram_addr;
	const uint32_t start_tick = HAL_GetTick();
	if (!app_ai_littlegood_proposer_ready || !AppAI_Xspi2EnsureMemoryMappedMode())
		return false;
	__asm volatile("mov %0, r9" : "=r"(caller_r9));
	if (!LL_ATON_EC_Network_Init_ocdet_ellipse_320_v2())
		return false;
	LL_ATON_RT_Init_Network(&NN_Instance_ocdet_ellipse_320_v2);
	AppAI_LittleGoodProposer_InstallRelocContext();
	if (!LL_ATON_EC_Inference_Init_ocdet_ellipse_320_v2())
		return false;
	LL_ATON_OSAL_DrainWfeSemaphore();
	for (;;) {
		if ((HAL_GetTick() - start_tick) >= 10000U) {
			__asm volatile("mov r9, %0" : : "r"(caller_r9) : "r9");
			return false;
		}
		__asm volatile("mov r9, %0" : : "r"(runtime_r9));
		status = LL_ATON_RT_RunEpochBlock(&NN_Instance_ocdet_ellipse_320_v2);
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
