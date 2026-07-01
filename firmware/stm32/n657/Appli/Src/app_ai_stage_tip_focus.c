/**
 * @file    app_ai_stage_tip_focus.c
 * @brief   Tip-focus UNet geometry stage module.
 *
 * This compilation unit includes the shared runtime-tail implementations
 * and exposes the tip-focus public surface via app_ai_stage_tip_focus.h.
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

#include "app_ai_stage_tip_focus.h"
#include "app_ai_state.h"
#include "app_ai_types.h"
#include "app_ai_inference.h"
#include "app_ai_preprocess.h"
#include "app_ai_xspi2.h"
#include "app_ai_logging.h"
#include "app_ai_stage_obb.h"
#include "app_center_detector.h"
#include "app_inference_log_utils.h"
#include "ina219_power.h"
#include "inference_metrics.h"
#include "app_inference_calibration.h"
#include "app_baseline_runtime.h"
#include "app_ai_runtime_tail.inc"
