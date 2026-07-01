/**
 * @file    app_ai_preprocess.h
 * @brief   Extern declarations for the preprocess module.
 *
 * This header collects the public surface of the YUV422 frame preprocess
 * helpers so they can be called from other compilation units (e.g. a
 * dedicated preprocess .c file) without duplicating definitions.
 */

#ifndef __APP_AI_PREPROCESS_H
#define __APP_AI_PREPROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "app_ai_types.h"

extern bool AppAI_WaitForFileXMediaReady(uint32_t timeout_ms);

extern bool AppAI_PreprocessYuv422FrameToInt8Input(
	const uint8_t *frame_bytes, size_t frame_size, uint8_t *input_ptr,
	size_t input_len_bytes, const LL_Buffer_InfoTypeDef *input_info);

extern bool AppAI_PreprocessYuv422FrameToFloatInput(
	const uint8_t *frame_bytes, size_t frame_size, float *input_ptr,
	size_t input_float_count, size_t input_len_bytes,
	size_t output_width, size_t output_height);

extern void AppAI_ReadRgbFromYuv422Bilinear(const uint8_t *frame_bytes,
	size_t frame_size_bytes,
	size_t frame_width_pixels, size_t frame_height_pixels,
	float source_x, float source_y,
	float *r_out, float *g_out, float *b_out);

extern void AppAI_ReadRgbFromRgbBilinear(const uint8_t *rgb_bytes,
	size_t frame_width_pixels, size_t frame_height_pixels,
	float source_x, float source_y,
	float *r_out, float *g_out, float *b_out);

extern uint8_t AppAI_ReadYuv422Luma(const uint8_t *frame_bytes,
	size_t frame_size_bytes,
	size_t frame_width_pixels, size_t source_x, size_t source_y);

extern void AppAI_ReadYuv422Quartet(const uint8_t *frame_bytes,
	size_t frame_size_bytes,
	size_t frame_width_pixels, size_t source_x, size_t source_y,
	uint8_t *quad_out);

extern void AppAI_ReadYuv422Pixel(const uint8_t *frame_bytes,
	size_t frame_size_bytes,
	size_t frame_width_pixels, size_t source_x, size_t source_y,
	float *r_out, float *g_out, float *b_out);

extern float AppAI_ReadNormalizedGrayFromYuv422Pixel(const uint8_t *frame_bytes,
	size_t frame_size_bytes,
	size_t frame_width_pixels, size_t source_x, size_t source_y);

#ifdef __cplusplus
}
#endif

#endif /* __APP_AI_PREPROCESS_H */
