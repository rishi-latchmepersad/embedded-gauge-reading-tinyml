/**
 * @file    app_ai_stage_tip_focus.h
 * @brief   Extern declarations for the tip-focus UNet geometry stage.
 *
 * Collects the public surface of tip-focus heatmap decode, median
 * filtering, geometry scoring, polar preprocessing, NPU/RISAF config,
 * and inference calibration so they can be called from other
 * compilation units (e.g. app_ai_stage_tip_focus.c) without
 * duplicating definitions.
 */

#ifndef __APP_AI_STAGE_TIP_FOCUS_H
#define __APP_AI_STAGE_TIP_FOCUS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "app_ai_types.h"

/** RISAF_TypeDef is provided by the STM32 HAL headers. */

/* ------------------------------------------------------------------ */
/* Tip-focus main entry points                                        */
/* ------------------------------------------------------------------ */

extern bool AppAI_TipFocus_RunDryInferenceFromYuv422(
	const uint8_t *frame_bytes, size_t frame_size);

extern bool AppAI_TipFocus_ResolveObbCropForLogging(
	const uint8_t *frame_bytes,
	size_t frame_size,
	AppAI_SourceCrop *obb_crop_out,
	AppAI_ObbBox *obb_box_out,
	const LL_Buffer_InfoTypeDef **obb_output_info_out,
	float *obb_output_value_out);

/* ------------------------------------------------------------------ */
/* Tip-focus heatmap decode                                           */
/* ------------------------------------------------------------------ */

extern bool AppAI_TipFocus_DecodeHeatmap2D(
	const float *heatmap, size_t width, size_t height,
	float *coord_x_norm_out, float *coord_y_norm_out,
	float *peak_out, float *spread_x_px_out, float *spread_y_px_out);

extern void AppAI_TipFocus_LogHeatmapSummary(
	const char *label,
	const float *heatmap,
	size_t width,
	size_t height);

/* ------------------------------------------------------------------ */
/* Tip-focus median filtering                                         */
/* ------------------------------------------------------------------ */

extern void AppAI_TipFocus_ResetMedianHistory(float seed_value);

extern float AppAI_TipFocus_MedianFilter(float new_value);

/* ------------------------------------------------------------------ */
/* Input blanking / formatting utilities                              */
/* ------------------------------------------------------------------ */

extern void AppAI_BlankTipFocusLowerInset(int8_t *input, size_t width,
	size_t height);

extern void AppAI_FormatFixedFloat(char *dst, size_t dst_len, float value,
	unsigned decimals);

/* ------------------------------------------------------------------ */
/* Scalar / polar preprocessing                                       */
/* ------------------------------------------------------------------ */

extern bool AppAI_PreprocessScalarRow(
	const uint8_t *frame_bytes, size_t frame_size, size_t source_width,
	size_t source_height, size_t crop_x_min, size_t crop_y_min,
	size_t crop_width, size_t crop_height, size_t output_width,
	size_t output_height, float resize_scale, size_t resized_width,
	size_t resized_height, size_t resize_pad_x, size_t resize_pad_y,
	size_t out_y, float *input_ptr, size_t input_len_bytes);

extern uint8_t AppAI_FindHistogramPercentile(const uint32_t *histogram,
	size_t histogram_len, uint32_t total_count, float percentile);

extern float AppAI_ScorePolarAlignmentCandidate(
	const uint8_t *resized_rgb,
	size_t output_dim,
	float center_x,
	float center_y,
	float max_radius,
	uint8_t *polar_luma_out);

extern bool AppAI_PreprocessYuv422FrameToPolarInput(
	const uint8_t *frame_bytes, size_t frame_size,
	uint8_t *input_ptr, size_t input_len_bytes,
	const LL_Buffer_InfoTypeDef *input_info);

/* ------------------------------------------------------------------ */
/* Runtime / NPU / RISAF bootstrapping                                */
/* ------------------------------------------------------------------ */

extern bool AppAI_RuntimeInitStepwise(void);

extern void AppAI_EnableNpuMemoryAndCaches(void);

extern void AppAI_ConfigureNpuAccessControl(void);

extern void AppAI_ConfigureNpuRisafDefaults(void);

extern void AppAI_SetRisafDefault(RISAF_TypeDef *risaf);

extern uint32_t AppAI_GetRisafMaxAddr(RISAF_TypeDef *risaf);

/* ------------------------------------------------------------------ */
/* Inference calibration & burst smoothing                            */
/* ------------------------------------------------------------------ */

extern float AppAI_TraceAndApplyInferenceCalibration(float raw_value);

extern void AppAI_ResetInferenceBurstHistory(void);

extern float AppAI_FilterInferenceValue(float value);

/* ------------------------------------------------------------------ */
/* Public result access                                               */
/* ------------------------------------------------------------------ */

extern bool App_AI_GetLastInferenceResult(float *value_out);

/* ------------------------------------------------------------------ */
/* Weight verification                                                */
/* ------------------------------------------------------------------ */

extern bool AppAI_VerifyRectifierWeights(void);

extern bool AppAI_VerifyTipFocusWeights(void);

/* ------------------------------------------------------------------ */
/* xSPI2 memory-mapped mode                                           */
/* ------------------------------------------------------------------ */

extern bool AppAI_Xspi2EnsureMemoryMappedMode(void);

/* ------------------------------------------------------------------ */
/* Tip-focus input dump (one-shot SD card export)                     */
/* ------------------------------------------------------------------ */

extern bool AppAI_DumpTipFocusInputTensorOnce(
	const float *input_ptr,
	size_t output_width,
	size_t output_height,
	const char *crop_label,
	const AppAI_SourceCrop *crop_ptr,
	bool obb_crop_valid,
	const AppAI_ObbBox *obb_box);

/* ------------------------------------------------------------------ */
/* Utility / logging helpers defined in the runtime tail              */
/* ------------------------------------------------------------------ */

extern void AppAI_LogInitFailure(const char *step);

extern void AppAI_LogXspi2LoadFailure(const char *step, unsigned int fx_status,
									  int32_t bsp_status);

extern void AppAI_LogXspi2ProgramChunkProgress(unsigned long chunk_index,
											   unsigned long flash_offset,
											   unsigned long chunk_size);

extern void AppAI_LogXspi2FlashStatus(const char *label);

extern void AppAI_LogXspi2IndirectAndMappedPrefix(void);

extern void AppAI_LogXspi2MappedScaleBytes(void);

extern void AppAI_LogFloatApprox(const char *label, float value);

extern void AppAI_LogXspi2FlashPrefix(void);

/* ------------------------------------------------------------------ */
/* Stage diagnostics toggle (always true in debug)                    */
/* ------------------------------------------------------------------ */

extern bool AppAI_ShouldLogStageDiagnostics(const struct AppAI_ModelStageSpec *stage);

#ifdef __cplusplus
}
#endif

#endif /* __APP_AI_STAGE_TIP_FOCUS_H */
