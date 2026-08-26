/* Relocatable wrapper for the 224x224 wide-augmentation center/tip U-Net. */

#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX
#define LL_ATON_RT_RELOC 1
#define BUILD_AI_NETWORK_RELOC 1
#define _network_rt_ctx _network_rt_ctx_keypoint_unet_224g_wide_aug_int8
#define _network_entries _network_entries_keypoint_unet_224g_wide_aug_int8
#define _network_flags _network_flags_keypoint_unet_224g_wide_aug_int8
#define _itf_network _itf_network_keypoint_unet_224g_wide_aug_int8
#define LL_ATON_Internal_Buffers_Info_Default_Empty LL_ATON_Internal_Buffers_Info_Default_Empty_keypoint_unet_224g_wide_aug_int8
#ifndef NDEBUG
#define NDEBUG
#endif
#ifndef LL_ATON_DBG_BUFFER_INFO_EXCLUDED
#define LL_ATON_DBG_BUFFER_INFO_EXCLUDED 1
#endif

#define MODEL_CONF "../../st_ai_output/packages/keypoint_unet_224g_wide_aug_int8_n6_npu/st_ai_ws/build_network/keypoint_unet_224g_wide_aug_int8_reloc_conf.h"
#include "../../st_ai_output/packages/keypoint_unet_224g_wide_aug_int8_n6_npu/st_ai_output/keypoint_unet_224g_wide_aug_int8.h"
#include "ll_aton_NN_interface.h"
#include "ll_aton_rt_user_api.h"
#include "ll_aton_reloc_network.h"
#include "ll_aton.h"
#include "mcu_cache.h"
#include "main.h"
#include "debug_console.h"
#include <string.h>

extern bool LL_ATON_OSAL_WfeGuardExpired(void);
extern void LL_ATON_OSAL_DrainWfeSemaphore(void);
extern bool AppAI_Xspi2EnsureMemoryMappedMode(void);

static uintptr_t AppAI_GaugeKeypoint_GetRelocImageBase(void)
{
	/* The generated relocatable code uses r9 as the start of the initialized
	 * reloc image.  The GOT entries are relative to this base, not to the
	 * .got subsection itself. */
	return 0x34099400UL;
}

/**
 * Use the generated epoch link setup without ST's unbounded switch-clear poll.
 *
 * Each generated epoch has a matching LL_Switch_Deinit() that removes every
 * route it created.  The no-reset variant is therefore sufficient here and
 * avoids stranding the AI worker if the N6 switch CLR bit does not self-clear.
 */
static int AppAI_GaugeKeypoint_SwitchInit(const LL_Switch_InitTypeDef *config,
	int count)
{
	return LL_Switch_Init_NoReset(config, count);
}

LL_ATON_DECLARE_NAMED_NN_INTERFACE(keypoint_unet_224g_wide_aug_int8);
/* Redirect only this compile-in model.  The generated source remains intact,
 * while the live wrapper avoids the vendor function's unbounded reset poll. */
#define LL_Switch_Init AppAI_GaugeKeypoint_SwitchInit
#include "../../st_ai_output/packages/keypoint_unet_224g_wide_aug_int8_n6_npu/st_ai_ws/build_network/keypoint_unet_224g_wide_aug_int8_reloc.c"
#undef LL_Switch_Init

extern struct ai_reloc_rt_ctx _network_rt_ctx_keypoint_unet_224g_wide_aug_int8;
NN_Instance_TypeDef NN_Instance_keypoint_unet_224g_wide_aug_int8 = {
	.network = &NN_Interface_keypoint_unet_224g_wide_aug_int8,
	.exec_state = {0}
};

static bool app_ai_gauge_keypoint_ready = false;
static const uint8_t app_ai_gauge_keypoint_signature_start[16] = {
		0x14U, 0x32U, 0xE8U, 0xF9U, 0xFCU, 0xF7U, 0x03U, 0x27U,
		0x09U, 0x1BU, 0xF4U, 0xE7U, 0x0DU, 0x33U, 0x11U, 0x17U
};
static const uint8_t app_ai_gauge_keypoint_signature_tail[16] = {
	0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0x80U
};

/** Verify the exact stride-2 keypoint weight image before NPU execution. */
static bool AppAI_GaugeKeypoint_VerifyFlashImage(void)
{
	const uint8_t *image = (const uint8_t *)0x70800000UL;
	(void)mcu_cache_invalidate_range((uint32_t)(uintptr_t)image,
		(uint32_t)((uintptr_t)image + 2246369U));
	return (memcmp(image, app_ai_gauge_keypoint_signature_start, 16U) == 0) &&
		(memcmp(image + 2246369U - 16U, app_ai_gauge_keypoint_signature_tail, 16U) == 0);
}

/** Prepare the linked-image context used only to preserve r9 in epoch calls. */
static void AppAI_GaugeKeypoint_PrepareRelocContext(void)
{
	/* Point r9 at the linked GOT. 0x34100000 is the activation pool and is
	 * not a valid base for the generated SW operator GOT lookups. */
	_network_rt_ctx_keypoint_unet_224g_wide_aug_int8.ram_addr =
		(uint32_t)AppAI_GaugeKeypoint_GetRelocImageBase();
	_network_rt_ctx_keypoint_unet_224g_wide_aug_int8.file_addr = 0x70800000UL;
	_network_rt_ctx_keypoint_unet_224g_wide_aug_int8.state = AI_RELOC_RT_STATE_INITIALIZED | AI_RELOC_RT_STATE_XIP_MODE;
}

/** Clear software and hardware completion state left by an earlier epoch. */
static void AppAI_GaugeKeypoint_ClearPendingAttonEvents(void)
{
	/* A completed epoch can signal before the runtime returns DONE.  Drain that
	 * software token before the next model so it cannot satisfy a later wait. */
	LL_ATON_OSAL_DrainWfeSemaphore();

#if defined(ATON_STRENG_NUM) && (ATON_STRENG_NUM > 0)
	/* The streaming-engine IRQ register is write-one-to-clear. */
	for (uint32_t se_id = 0U; se_id < ATON_STRENG_NUM; ++se_id)
	{
		ATON_STRENG_IRQ_SET(se_id, 0xFFFFFFFFU);
	}
#endif

	/* Clear the combined controller status after clearing its source flags. */
	{
		const uint32_t pending = ATON_INTCTRL_INTREG_GET(0);
		if (pending != 0U)
		{
			ATON_INTCTRL_INTCLR_SET(0, pending);
		}
	}
#if (ATON_INT_NR > 32)
	{
		const uint32_t pending_high = ATON_INTCTRL_INTREG_H_GET(0);
		if (pending_high != 0U)
		{
			ATON_INTCTRL_INTCLR_H_SET(0, pending_high);
		}
	}
#endif
}

/** Initialize the keypoint network and validate its flash image. */
bool AppAI_GaugeCenterTip_Init(void)
{
	if (!AppAI_Xspi2EnsureMemoryMappedMode() || !AppAI_GaugeKeypoint_VerifyFlashImage()) return false;
	if (!LL_ATON_EC_Network_Init_keypoint_unet_224g_wide_aug_int8()) return false;
	/* This is compile-in relocatable C, not an ll_aton_reloc_install() binary.
	 * Keep inst_reloc clear while the stock runtime builds its direct epoch list. */
	NN_Instance_keypoint_unet_224g_wide_aug_int8.exec_state.inst_reloc = 0U;
	LL_ATON_RT_Init_Network(&NN_Instance_keypoint_unet_224g_wide_aug_int8);
	AppAI_GaugeKeypoint_PrepareRelocContext();
	app_ai_gauge_keypoint_ready = true;
	return true;
}

/** Return the generated single grayscale input descriptor. */
const LL_Buffer_InfoTypeDef *AppAI_GaugeCenterTip_InputInfo(void)
{
	return LL_ATON_Input_Buffers_Info_keypoint_unet_224g_wide_aug_int8();
}

/** Return the generated 56x56x2 center/tip heatmap descriptor. */
const LL_Buffer_InfoTypeDef *AppAI_GaugeCenterTip_OutputInfo(void)
{
	return LL_ATON_Output_Buffers_Info_keypoint_unet_224g_wide_aug_int8();
}

/** Run one keypoint inference while preserving the generated reloc base in r9. */
bool AppAI_GaugeCenterTip_Run(void)
{
	LL_ATON_RT_RetValues_t status = LL_ATON_RT_DONE;
	uintptr_t caller_r9 = 0U;
	uintptr_t runtime_r9 = 0U;
	const uint32_t start_tick = HAL_GetTick();
	const char *failure_stage = "unknown";
	if (!app_ai_gauge_keypoint_ready) {
		failure_stage = "not-ready";
		goto fail;
	}
	if (!AppAI_Xspi2EnsureMemoryMappedMode()) {
		failure_stage = "xspi2-mm";
		goto fail;
	}
	__asm volatile("mov %0, r9" : "=r"(caller_r9));
	if (!LL_ATON_EC_Network_Init_keypoint_unet_224g_wide_aug_int8()) {
		failure_stage = "network-init";
		goto fail;
	}
	/* Let LL_ATON_RT_Init_Network use the compile-in network interface. */
	NN_Instance_keypoint_unet_224g_wide_aug_int8.exec_state.inst_reloc = 0U;
	LL_ATON_RT_Init_Network(&NN_Instance_keypoint_unet_224g_wide_aug_int8);
	AppAI_GaugeKeypoint_PrepareRelocContext();
	if (!LL_ATON_EC_Inference_Init_keypoint_unet_224g_wide_aug_int8()) {
		failure_stage = "inference-init";
		goto fail;
	}
	/* Direct generated inference init has done the binary handler's setup.
	 * Enable the context only now so epoch calls preserve r9 without making the
	 * runtime interpret this compile-in context as a flashed reloc binary. */
	NN_Instance_keypoint_unet_224g_wide_aug_int8.exec_state.inference_started = true;
	NN_Instance_keypoint_unet_224g_wide_aug_int8.exec_state.inst_reloc =
		(uint32_t)(uintptr_t)&_network_rt_ctx_keypoint_unet_224g_wide_aug_int8;
	runtime_r9 = (uintptr_t)_network_rt_ctx_keypoint_unet_224g_wide_aug_int8.ram_addr;
	AppAI_GaugeKeypoint_ClearPendingAttonEvents();
	for (;;) {
		if ((HAL_GetTick() - start_tick) >= 10000U) {
			failure_stage = "timeout";
			goto fail;
		}
		__asm volatile("mov r9, %0" : : "r"(runtime_r9));
		status = LL_ATON_RT_RunEpochBlock(&NN_Instance_keypoint_unet_224g_wide_aug_int8);
		if (status == LL_ATON_RT_DONE) break;
		if (status == LL_ATON_RT_WFE) {
			LL_ATON_OSAL_WFE();
			if (LL_ATON_OSAL_WfeGuardExpired()) {
				failure_stage = "wfe-guard";
				goto fail;
			}
			continue;
		}
		if (status != LL_ATON_RT_NO_WFE) {
			failure_stage = "epoch-status";
			goto fail;
		}
	}
	DebugConsole_WriteString("[AI][CENTER_TIP] NPU inference complete.\r\n");
	__asm volatile("mov r9, %0" : : "r"(caller_r9) : "r9");
	return true;
fail:
	__asm volatile("mov r9, %0" : : "r"(caller_r9) : "r9");
	DebugConsole_Printf(
		"[AI][CENTER_TIP] run failed stage=%s status=%d elapsed_ms=%lu.\r\n",
		failure_stage, (int)status,
		(unsigned long)(HAL_GetTick() - start_tick));
	return false;
}
