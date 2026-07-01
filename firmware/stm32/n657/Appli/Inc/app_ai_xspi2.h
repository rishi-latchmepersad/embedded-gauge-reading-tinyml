/**
 * @file    app_ai_xspi2.h
 * @brief   Public xSPI2 / NPU / cache / YUV422 helpers API.
 *
 * Declares every helper from app_ai_helpers_core.inc plus the xSPI2
 * provisioning functions from app_ai_helpers_model.inc so that the
 * pipeline stages and the logging module can call them directly.
 */

#ifndef __APP_AI_XSPI2_H
#define __APP_AI_XSPI2_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "app_ai_types.h"

/* Forward declaration for FileX types.  Include fx_api.h in .c files
 * that need to dereference these. */
typedef struct FX_FILE_STRUCT FX_FILE;

/* ------------------------------------------------------------------ */
/* YUV422 pixel helpers                                               */
/* Forward-declared in app_ai_helpers_core.inc; implemented in       */
/* app_ai_helpers_decode.inc and app_ai_helpers_resize.inc.          */
/* ------------------------------------------------------------------ */

extern uint32_t AppAI_GrayToFloatBits(uint8_t gray);

extern uint8_t AppAI_ReadYuv422Luma(const uint8_t *frame_bytes,
                                    size_t frame_size_bytes,
                                    size_t frame_width_pixels,
                                    size_t source_x,
                                    size_t source_y);

extern void AppAI_ReadYuv422Quartet(const uint8_t *frame_bytes,
                                    size_t frame_size_bytes,
                                    size_t frame_width_pixels,
                                    size_t source_x,
                                    size_t source_y,
                                    uint8_t *quad_out);

extern float AppAI_ReadNormalizedGrayFromYuv422Pixel(
    const uint8_t *frame_bytes, size_t frame_size_bytes,
    size_t frame_width_pixels, size_t source_x,
    size_t source_y);

extern void AppAI_ReadRgbFromYuv422Pixel(const uint8_t *frame_bytes,
                                         size_t frame_size_bytes,
                                         size_t frame_width_pixels,
                                         size_t source_x,
                                         size_t source_y,
                                         float *r_out, float *g_out,
                                         float *b_out);

extern void AppAI_ReadRgbFromYuv422Bilinear(
    const uint8_t *frame_bytes, size_t frame_size_bytes,
    size_t frame_width_pixels, size_t frame_height_pixels,
    float source_x, float source_y, float *r_out, float *g_out,
    float *b_out);

extern void AppAI_ReadRgbFromRgbBilinear(const uint8_t *rgb_bytes,
                                         size_t frame_width_pixels,
                                         size_t frame_height_pixels,
                                         float source_x,
                                         float source_y,
                                         float *r_out, float *g_out,
                                         float *b_out);

extern uint8_t AppAI_FindHistogramPercentile(const uint32_t *histogram,
                                             size_t histogram_len,
                                             uint32_t total_count,
                                             float percentile);

extern float AppAI_ScorePolarAlignmentCandidate(
    const uint8_t *resized_rgb, size_t output_dim, float center_x,
    float center_y, float max_radius, uint8_t *polar_luma_out);

/* ------------------------------------------------------------------ */
/* Decode helpers (forward-declared in app_ai_helpers_core.inc)       */
/* Implemented in app_ai_helpers_decode.inc.                          */
/* ------------------------------------------------------------------ */

extern bool AppAI_DecodeObbCropBox(
    const LL_Buffer_InfoTypeDef *output_buffer_info,
    AppAI_SourceCrop *crop_out, AppAI_ObbBox *obb_box_out);

#if APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE
extern bool AppAI_DecodeSourceCropBox(
    const LL_Buffer_InfoTypeDef *output_buffer_info,
    AppAI_SourceCrop *crop_out,
    AppAI_SourceCropBox *box_out);
#endif /* APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE */

extern bool AppAI_DecodeRectifierCropBox(
    const LL_Buffer_InfoTypeDef *output_buffer_info,
    AppAI_SourceCrop *crop_out,
    AppAI_RectifierBox *rectifier_box_out);

#ifndef APP_AI_DISABLE_LEGACY_TOPK_DECODE
extern bool AppAI_DecodeScalarTopKExpectationFromOutput(
    const LL_Buffer_InfoTypeDef *output_info,
    const uint8_t *output_ptr, size_t output_len_bytes,
    float *decoded_value_out);
#endif /* !APP_AI_DISABLE_LEGACY_TOPK_DECODE */

extern bool AppAI_DecodeCircularVoteFromOutput(
    const uint8_t *output_ptr, size_t output_len_bytes,
    float *decoded_value_out);

extern bool AppAI_PreprocessYuv422FrameToPolarInput(
    const uint8_t *frame_bytes, size_t frame_size,
    uint8_t *input_ptr, size_t input_len_bytes,
    const LL_Buffer_InfoTypeDef *input_info);

/* ------------------------------------------------------------------ */
/* Forced crop helpers                                                */
/* Implemented in app_ai_helpers_model.inc.                           */
/* ------------------------------------------------------------------ */

extern void AppAI_SetForcedCrop(const char *label, size_t x_min,
                                size_t y_min, size_t width,
                                size_t height);

extern void AppAI_ClearForcedCrop(void);

/* ------------------------------------------------------------------ */
/* Cache maintenance wrappers                                         */
/* Implemented in app_ai_helpers_core.inc.                            */
/* ------------------------------------------------------------------ */

extern int mcu_cache_clean_range(uint32_t start_addr,
                                 uint32_t end_addr);

extern int mcu_cache_invalidate_range(uint32_t start_addr,
                                      uint32_t end_addr);

extern int mcu_cache_clean_invalidate_range(uint32_t start_addr,
                                            uint32_t end_addr);

/* ------------------------------------------------------------------ */
/* NPU / xSPI2 hardware bring-up                                      */
/* Implemented in app_ai_helpers_core.inc.                            */
/* ------------------------------------------------------------------ */

extern bool AppAI_EnsureNpuHardwareReady(void);
extern bool AppAI_EnsureXspi2MemoryReady(void);
extern bool AppAI_ReconfigureXspi2ForRuntime(void);
extern bool AppAI_Xspi2EnableMemoryMappedMode(void);

/* ------------------------------------------------------------------ */
/* xSPI2 flash probes                                                 */
/* Implemented in app_ai_helpers_core.inc.                            */
/* ------------------------------------------------------------------ */

extern void AppAI_LogXspi2ProbeBytes(const char *label,
                                     const uint8_t *bytes);

extern bool AppAI_Xspi2ReadFlashProbe(uint32_t chip_base_offset,
                                      uint32_t flash_offset,
                                      const uint8_t *expected_bytes,
                                      size_t expected_length);

extern bool AppAI_Xspi2ReadMappedProbe(uint32_t flash_offset,
                                       const uint8_t *expected_bytes,
                                       size_t expected_length);

extern bool AppAI_Xspi2ReadStageProbe(const AppAI_ModelStageSpec *stage,
                                      uint32_t flash_offset,
                                      const uint8_t *expected_bytes,
                                      size_t expected_length);

/* ------------------------------------------------------------------ */
/* xSPI2 model-flash provisioning                                     */
/* Implemented in app_ai_helpers_model.inc.                           */
/* ------------------------------------------------------------------ */

extern bool AppAI_LogXspi2ModelFilePrefix(FX_FILE *model_file_ptr);

extern bool AppAI_Xspi2ModelImageMatchesMappedFlash(void);

extern bool AppAI_Xspi2ModelImageMatchesMappedFlashForStage(
    const AppAI_ModelStageSpec *stage);

extern bool AppAI_ProgramXspi2ModelImageFromSd(void);

extern bool AppAI_ReadXspi2ModelSourceProbes(
    FX_FILE *model_file_ptr, unsigned long file_size,
    uint8_t *source_prefix, uint8_t *source_tail,
    bool *has_tail_out);

extern bool AppAI_EnsureXspi2ModelImageReadyForStage(
    const AppAI_ModelStageSpec *stage);

/* ------------------------------------------------------------------ */
/* xSPI2 runtime guard                                                */
/* Implemented in app_ai_runtime_tail.inc.                            */
/* ------------------------------------------------------------------ */

extern bool AppAI_Xspi2EnsureMemoryMappedMode(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_AI_XSPI2_H */
