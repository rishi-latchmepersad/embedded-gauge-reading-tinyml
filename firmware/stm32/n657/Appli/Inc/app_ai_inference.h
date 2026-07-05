/**
 * @file    app_ai_inference.h
 * @brief   Extern declarations for the inference module.
 *
 * Collects the public surface of model-stage management, inference
 * execution, crop-box decode, and result-logging helpers so they can
 * be called from other compilation units (e.g. a dedicated inference
 * .c file) without duplicating definitions.
 */

#ifndef __APP_AI_INFERENCE_H
#define __APP_AI_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "app_ai_types.h"

/* Forward-declare the source-crop-box struct — the full definition lives
 * alongside the functions that populate it (app_ai_helpers_decode.inc). */
#ifndef APP_AI_SOURCE_CROP_BOX_DECLARED
#define APP_AI_SOURCE_CROP_BOX_DECLARED
typedef struct {
	float x_min;
	float y_min;
	float x_max;
	float y_max;
} AppAI_SourceCropBox;
#endif

/* ------------------------------------------------------------------ */
/* Forced crop helpers                                                */
/* ------------------------------------------------------------------ */

extern void AppAI_SetForcedCrop(const char *label, size_t x_min,
								size_t y_min, size_t width, size_t height);

extern void AppAI_ClearForcedCrop(void);

/* ------------------------------------------------------------------ */
/* Stage buffer info                                                  */
/* ------------------------------------------------------------------ */

extern const LL_Buffer_InfoTypeDef *AppAI_GetStageInputBufferInfo(
	const AppAI_ModelStageSpec *stage);

extern const LL_Buffer_InfoTypeDef *AppAI_GetStageOutputBufferInfo(
	const AppAI_ModelStageSpec *stage);

/* ------------------------------------------------------------------ */
/* Stage lifecycle                                                    */
/* ------------------------------------------------------------------ */

extern bool AppAI_InitStage(const AppAI_ModelStageSpec *stage);

extern bool AppAI_EnsureStageRuntimeReady(const AppAI_ModelStageSpec *stage);

/* ------------------------------------------------------------------ */
/* Crop-box decoders                                                  */
/* ------------------------------------------------------------------ */

extern bool AppAI_DecodeRectifierCropBox(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	AppAI_SourceCrop *crop_out,
	AppAI_RectifierBox *rectifier_box_out);

extern bool AppAI_DecodeObbCropBox(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	AppAI_SourceCrop *crop_out,
	AppAI_ObbBox *obb_box_out);

extern bool AppAI_DecodeSourceCropBox(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	AppAI_SourceCrop *crop_out,
	AppAI_SourceCropBox *box_out);

/* ------------------------------------------------------------------ */
/* Inference engine                                                   */
/* ------------------------------------------------------------------ */

extern bool AppAI_RunStageInference(const AppAI_ModelStageSpec *stage,
	const uint8_t *frame_bytes, size_t frame_size,
	const AppAI_SourceCrop *forced_crop,
	const LL_Buffer_InfoTypeDef **output_info_out,
	float *output_value_out);

/* ------------------------------------------------------------------ */
/* Result logging                                                     */
/* ------------------------------------------------------------------ */

extern void AppAI_LogRectifierResult(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	const AppAI_RectifierBox *rectifier_box);

extern void AppAI_LogObbResult(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	const AppAI_ObbBox *obb_box);

extern void AppAI_LogSourceCropBoxResult(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	const AppAI_SourceCropBox *source_crop_box);

extern void AppAI_LogScalarInternalOutputProbe(
	const AppAI_ModelStageSpec *stage,
	const LL_Buffer_InfoTypeDef *stage_output_info);

extern void AppAI_LogInferenceResult(
	const LL_Buffer_InfoTypeDef *output_buffer_info);

extern void AppAI_LogR9(const char *label);

extern void AppAI_LogInputSignature(const float *input_buffer,
	size_t input_float_count);

extern void AppAI_LogFrameSignature(const uint8_t *frame_bytes,
	size_t frame_size);

/* ------------------------------------------------------------------ */
/* Value-scale utilities                                              */
/* ------------------------------------------------------------------ */

extern float AppAI_ObbDequantize(int8_t q_value);

extern bool AppAI_DecodeScalarTopKExpectationFromOutput(
	const LL_Buffer_InfoTypeDef *output_info,
	const uint8_t *output_ptr,
	size_t output_len_bytes,
	float *decoded_value_out);

extern bool AppAI_DecodeCircularVoteFromOutput(
	const uint8_t *output_ptr,
	size_t output_len_bytes,
	float *decoded_value_out);

extern bool AppAI_IsFiniteFloat(float value);

extern bool AppAI_IsPlausibleInferenceValue(float value);

extern float AppAI_ClampNormalizedFloat(float value);

/* ------------------------------------------------------------------ */
/* Cache / stack probes                                               */
/* ------------------------------------------------------------------ */

extern int AppAI_ApplyCacheRange(uint32_t start_addr, uint32_t end_addr,
	bool clean, bool invalidate);

extern void AppAI_StackWatermark_Log(const char *tag);

#ifdef __cplusplus
}
#endif

#endif /* __APP_AI_INFERENCE_H */
