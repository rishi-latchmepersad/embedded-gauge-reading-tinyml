/**
 * @file    app_ai_logging.c
 * @brief   Compilation unit that exports the logging helpers.
 *
 * This is a thin wrapper that includes the logging .inc file so the
 * function bodies live in their own translation unit.  Other .c files
 * include app_ai_logging.h and link against this TU at build time.
 */

#include "app_ai.h"
#include <stddef.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include "debug_console.h"
#include "app_memory_budget.h"
#include "app_gauge_geometry.h"
#include "app_inner_celsius_mask.h"
#include "app_ai_config.h"
#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX
#include "tx_api.h"
#include "ll_aton_rt_user_api.h"
#include "ll_aton.h"
#include "ll_aton_runtime.h"
#include "ll_aton_reloc_network.h"
#include "app_filex.h"
#include "stm32n6xx_nucleo_xspi.h"
#include "npu_cache.h"
#include "stm32n6xx_hal.h"

#include "app_ai_logging.h"
#include "app_ai_state.h"
#include "app_ai_types.h"
#include "app_ai_xspi2.h"
#include "app_ai_preprocess.h"
#include "app_ai_stage_obb.h"
#include "app_ai_stage_tip_focus.h"
#include "ina219_power.h"
#include "inference_metrics.h"
#include "app_ai_helpers_logging.inc"
