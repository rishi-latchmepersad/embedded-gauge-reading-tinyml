/**
 * @file    app_ai_stage_obb.h
 * @brief   Extern declarations for the OBB localizer stage.
 *
 * Collects the public surface of QARepVGG-Pro OBB decode, bounding-box
 * candidate construction, diagnostic logging, and value-scale utilities
 * so they can be called from other compilation units (e.g.
 * app_ai_stage_obb.c) without duplicating definitions.
 */

#ifndef __APP_AI_STAGE_OBB_H
#define __APP_AI_STAGE_OBB_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "app_ai_types.h"

/* ------------------------------------------------------------------ */
/* OBB decode                                                         */
/* ------------------------------------------------------------------ */

extern bool AppAI_DecodeQarepvggObb(
	const LL_Buffer_InfoTypeDef *output_info,
	AppAI_SourceCrop *obb_crop,
	AppAI_ObbBox *obb_box);

/* ------------------------------------------------------------------ */
/* OBB candidate builder                                              */
/* ------------------------------------------------------------------ */

extern bool AppAI_BuildObbDecodeCandidate(
	float x_min, float y_min, float x_max, float y_max,
	AppAI_ObbDecodeCandidate *candidate);

/* ------------------------------------------------------------------ */
/* OBB diagnostics                                                    */
/* ------------------------------------------------------------------ */

extern void AppAI_LogObbDecodeDiagnostics(
	float score,
	float raw0,
	float raw1,
	float raw2,
	float raw3,
	const AppAI_ObbDecodeCandidate *corners_candidate,
	const AppAI_ObbDecodeCandidate *center_size_candidate,
	const char *selected_label);

/* ------------------------------------------------------------------ */
/* Value-scale utilities (shared with the tip-focus stage)            */
/* ------------------------------------------------------------------ */

extern float AppAI_ObbDequantize(int8_t q_value);

extern float AppAI_ClampNormalizedFloat(float value);

extern bool AppAI_IsFiniteFloat(float value);

/* ------------------------------------------------------------------ */
/* Relocatable runtime helper                                         */
/* ------------------------------------------------------------------ */

extern uintptr_t AppAI_GetRelocRuntimeR9(
	const NN_Instance_TypeDef *nn_instance);

/* ------------------------------------------------------------------ */
/* Generated OBB model init / inference declarations                   */
/* ------------------------------------------------------------------ */
extern bool LL_ATON_EC_Network_Init_obb_box_board_bbox_deploy_candidate(void);
extern bool LL_ATON_EC_Inference_Init_obb_box_board_bbox_deploy_candidate(void);

/**
 * @brief Install the OBB relocatable runtime context.
 *
 * The OBB package uses a relocatable binary with a zero ec_network_init
 * vector.  Call this after LL_ATON_RT_Init_Network() to restore the
 * per-instance inst_reloc pointer so the epoch loop and SW resize
 * helper see the correct runtime base.
 */
extern bool AppAI_Obb_InstallRelocContext(NN_Instance_TypeDef *instance,
                                           uintptr_t xspi2_base_addr);

#ifdef __cplusplus
}
#endif

#endif /* __APP_AI_STAGE_OBB_H */
