/**
 * @file    app_ai_logging.h
 * @brief   Public logging API for the AI pipeline.
 *
 * Every log/trace helper that was previously reachable only through the
 * monolithic helpers.inc include chain is now declared here so translation
 * units can include a lightweight header instead of pulling in the full
 * implementation.
 */

#ifndef __APP_AI_LOGGING_H
#define __APP_AI_LOGGING_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "app_ai_types.h"

/* ------------------------------------------------------------------ */
/* Verbose tensor / buffer diagnostics                                */
/* Implemented in app_ai_helpers_logging.inc (real) and               */
/* app_ai_helpers_decode.inc (stubs when VERBOSE_LOGS is off).       */
/* ------------------------------------------------------------------ */

extern void AppAI_LogXspi2PrefixBytes(const char *label,
                                      const uint8_t *bytes);

extern void AppAI_LogFrameSignature(const uint8_t *frame_bytes,
                                    size_t frame_size);

extern void AppAI_LogInputSignature(const float *input_buffer,
                                    size_t input_float_count);

extern void AppAI_LogInputTensorWindow(const float *input_buffer,
                                       size_t input_float_count);

extern void AppAI_LogInputProbeSummary(const float *input_buffer,
                                       size_t input_float_count);

extern void AppAI_LogTensorRowSamples(const char *label,
                                      const float *input_buffer,
                                      size_t tensor_width, size_t row_y,
                                      size_t x_min, size_t x_max);

extern void AppAI_LogSourcePatch(const char *label,
                                 const uint8_t *frame_bytes,
                                 size_t frame_width_pixels,
                                 size_t center_x, size_t center_y,
                                 size_t radius_pixels);

extern void AppAI_LogTensorPatch(const char *label,
                                 const float *input_buffer,
                                 size_t tensor_width, size_t center_x,
                                 size_t center_y, size_t radius_pixels);

extern void AppAI_LogSourceCropWindow(const uint8_t *frame_bytes,
                                      size_t frame_size,
                                      size_t frame_width_pixels,
                                      size_t frame_height_pixels,
                                      size_t crop_x_min,
                                      size_t crop_y_min,
                                      size_t crop_width,
                                      size_t crop_height);

extern void AppAI_LogInt8BufferSignature(const char *label,
                                         const int8_t *buffer_ptr,
                                         size_t buffer_len_bytes);

extern void AppAI_LogRawBufferSignature(const char *label,
                                        const uint8_t *buffer_ptr,
                                        size_t buffer_len_bytes);

extern const char *AppAI_BufferTypeName(
    const LL_Buffer_InfoTypeDef *buffer_info);

extern void AppAI_LogBufferInfoAndSignature(
    const char *label,
    const LL_Buffer_InfoTypeDef *buffer_info);

/* ------------------------------------------------------------------ */
/* Always-on buffer preview                                           */
/* Implemented in app_ai_helpers_logging.inc (unconditional).         */
/* ------------------------------------------------------------------ */

extern void AppAI_LogBufferPreview(const char *label,
                                   const LL_Buffer_InfoTypeDef *buffer_info);

/* ------------------------------------------------------------------ */
/* Gauge crop heuristic                                               */
/* Implemented in app_ai_helpers_logging.inc (unconditional).         */
/* ------------------------------------------------------------------ */

extern bool AppAI_EstimateGaugeCropBoxFromYuv422(
    const uint8_t *frame_bytes, size_t frame_size,
    size_t frame_width_pixels, size_t frame_height_pixels,
    size_t *crop_x_min, size_t *crop_y_min, size_t *crop_width,
    size_t *crop_height);

#ifdef __cplusplus
}
#endif

#endif /* __APP_AI_LOGGING_H */
