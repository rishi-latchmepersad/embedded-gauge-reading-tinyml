/**
 * @file    ai_network_qarepvgg_pro_a175_int8.c
 * @brief   Include the Cube.AI-generated board-bbox OBB int8 model.
 *
 * STM32CubeIDE builds the generated .c directly from the package directory.
 * This thunk forwards the include with the required ATON platform macros and
 * installs the relocatable runtime context that the generated OBB package
 * expects in r9 at inference time.
 */
#include <stdint.h>

#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX
#define LL_ATON_RT_RELOC 1
#ifndef NDEBUG
#define NDEBUG
#endif
#ifndef LL_ATON_DBG_BUFFER_INFO_EXCLUDED
#define LL_ATON_DBG_BUFFER_INFO_EXCLUDED 1
#endif

#include "ll_aton_NN_interface.h"
#include "ll_aton_rt_user_api.h"
#include "ll_aton_reloc_network.h"
#include "debug_console.h"
#include "C:/Users/rishi_latchmepersad/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/Middlewares/ST/AI/Npu/ll_aton/ll_aton_reloc_network.c"
#include "../../st_ai_output/packages/obb_box_board_bbox_deploy_candidate/st_ai_output/obb_box_board_bbox_deploy_candidate.h"

LL_ATON_DECLARE_NAMED_NN_INTERFACE(obb_box_board_bbox_deploy_candidate);

NN_Instance_TypeDef NN_Instance_obb_box_board_bbox_deploy_candidate = {
    .network = &NN_Interface_obb_box_board_bbox_deploy_candidate,
    .exec_state = {0}
};

#include "../../st_ai_output/packages/obb_box_board_bbox_deploy_candidate/st_ai_ws/build_obb_box_board_bbox_deploy_candidate/obb_box_board_bbox_deploy_candidate_reloc.c"

extern struct ai_reloc_rt_ctx _network_rt_ctx;

/* The generated reloc C only gives us the epoch code and runtime metadata.
 * The actual reloc binary must be aligned so the ST installer can read the
 * binary header and section table without hitting an unaligned access path. */
__attribute__((aligned(32)))
static const uint8_t app_ai_obb_reloc_bin[] = {
#include "obb_box_board_bbox_deploy_candidate_rel_bin.inc"
};
static uintptr_t app_ai_obb_reloc_handle = 0U;

/* The relocatable OBB package keeps its live runtime tables in the shared
 * AXISRAM window that the board init code exposes to the NPU. Use the actual
 * on-chip base here, not the virtual reloc alias from the generated ELF. */
#define APP_AI_OBB_RELOC_RAM_BASE_ADDR 0x34100000UL
#define APP_AI_OBB_RELOC_RAM_SIZE      2883576UL
#ifndef APP_AI_XSPI2_OBB_BASE_ADDR
#define APP_AI_XSPI2_OBB_BASE_ADDR     0x71400000UL
#endif

bool AppAI_Obb_InstallRelocContext(NN_Instance_TypeDef *instance, uintptr_t xspi2_base_addr)
{
	ll_aton_reloc_config reloc_cfg = {0};
	int reloc_status = AI_RELOC_RT_ERR_ARG;

	if (instance == NULL)
	{
		return false;
	}

	/* Install the generated reloc binary once and keep the returned handle.
	 * `LL_ATON_RT_Init_Network()` clears `inst_reloc`, so later calls only
	 * need to restore the stored handle instead of copying the image again. */
	if (app_ai_obb_reloc_handle == 0U)
	{
		if (xspi2_base_addr == 0U)
		{
			xspi2_base_addr = APP_AI_XSPI2_OBB_BASE_ADDR;
		}

		reloc_cfg.exec_ram_addr = APP_AI_OBB_RELOC_RAM_BASE_ADDR;
		reloc_cfg.exec_ram_size = APP_AI_OBB_RELOC_RAM_SIZE;
		reloc_cfg.ext_ram_addr = 0U;
		reloc_cfg.ext_ram_size = 0U;
		/* The OBB reloc package expects the parameter/weight pool to come from
		 * the flashed xSPI2 image, not from a scratch RAM buffer. */
		reloc_cfg.ext_param_addr = xspi2_base_addr;
		reloc_cfg.mode = AI_RELOC_RT_LOAD_MODE_XIP;

		reloc_status = ll_aton_reloc_install((uintptr_t)app_ai_obb_reloc_bin,
											 &reloc_cfg, instance);
		if (reloc_status != AI_RELOC_RT_ERR_NONE)
		{
			DebugConsole_Printf(
				"[AI][OBB] reloc install failed: status=%d ram=0x%08lX size=%lu param=0x%08lX\r\n",
				reloc_status,
				(unsigned long)APP_AI_OBB_RELOC_RAM_BASE_ADDR,
				(unsigned long)APP_AI_OBB_RELOC_RAM_SIZE,
				(unsigned long)xspi2_base_addr);
			return false;
		}

		app_ai_obb_reloc_handle = (uintptr_t)instance->exec_state.inst_reloc;
	}
	else
	{
		instance->exec_state.inst_reloc = (uint32_t)app_ai_obb_reloc_handle;
	}

	if ((instance->exec_state.inst_reloc == 0U) && (app_ai_obb_reloc_handle != 0U))
	{
		instance->exec_state.inst_reloc = (uint32_t)app_ai_obb_reloc_handle;
	}

	return instance->exec_state.inst_reloc != 0U;
}
