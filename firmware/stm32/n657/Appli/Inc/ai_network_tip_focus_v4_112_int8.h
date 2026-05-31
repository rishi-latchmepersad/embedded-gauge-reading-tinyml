/* Guarded wrapper API for the tip-focus geometry heatmap model
 * (tip_focus_v4_112_int8, LL_ATON NPU runtime).
 *
 * The underlying generated package uses the LL_ATON NPU runtime API family
 * (matching the other models in this project).
 * This wrapper manages the NN instance and exposes buffer accessors.
 *
 * Output order (generated NPU order):
 *   output[0] = tip_heatmap    [1,112,112,1] int8
 *   output[1] = center_heatmap [1,112,112,1] int8
 *   output[2] = confidence     [1,1]          int8
 *
 * NOTE: The ML replay semantic order is center/tip/confidence.
 * The wrapper intentionally keeps the generated order and names the accessors
 * explicitly to prevent accidental reversal.
 */

#ifndef AI_NETWORK_TIP_FOCUS_V4_112_INT8_H
#define AI_NETWORK_TIP_FOCUS_V4_112_INT8_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ---------------------------------------------------------------------------
 * Guard flag — default OFF.  Set to 1U in the build or a prefixed header to
 * enable the tip-focus geometry stage.
 * ---------------------------------------------------------------------------
 */
#ifndef APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
#define APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE 1U
#endif

/* Boot-time dry-run flag — default OFF.  Set to 1U to run a single
 * deterministic self-test inference during camera init (see app_threadx.c). */
#ifndef APP_AI_ENABLE_TIP_FOCUS_BOOT_DRY_RUN
#define APP_AI_ENABLE_TIP_FOCUS_BOOT_DRY_RUN 0U
#endif

#if APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE

/* ---------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------------
 */

/**
 * @brief Initialize the tip-focus NPU network interface.
 *
 * Calls LL_ATON_EC_Network_Init_Default() and
 * LL_ATON_EC_Inference_Init_Default() to prepare the NPU runtime.
 *
 * @retval true  NPU interface is ready for inference.
 * @retval false Initialisation failed (see UART log for details).
 */
bool AppAI_TipFocus_Init(void);

/**
 * @brief Run a single inference on a pre-filled int8 input.
 *
 * The caller must have filled the input buffer with int8 NHWC data
 * [1,224,224,3] quantised with scale=0.003921569, zp=-128.
 *
 * @note  The input buffer is managed by the NPU runtime.
 *        Use AppAI_TipFocus_GetInputBuffer() to get the address.
 *
 * @retval true  Inference completed successfully on the NPU.
 * @retval false LL_ATON_RT_RunEpochBlock() returned an error.
 */
bool AppAI_TipFocus_Run(void);

/**
 * @brief Get the tip heatmap output buffer.
 * @return Pointer to int8 [112,112] tip-heatmap values (row-major).
 *         NULL if no inference has completed yet.
 */
const int8_t *AppAI_TipFocus_GetTipHeatmap(void);

/**
 * @brief Get the center heatmap output buffer.
 * @return Pointer to int8 [112,112] center-heatmap values (row-major).
 *         NULL if no inference has completed yet.
 */
const int8_t *AppAI_TipFocus_GetCenterHeatmap(void);

/**
 * @brief Get the raw int8 confidence value.
 * @return The int8 confidence output (dequantised: val * 0.00390625 - 128).
 *         -128 if no inference has completed yet.
 */
int8_t AppAI_TipFocus_GetConfidenceRaw(void);

/**
 * @brief Get a pointer to the input buffer for writing.
 *
 * The caller fills this buffer with int8 NHWC [224,224,3] data before
 * calling AppAI_TipFocus_Run().
 *
 * @return Pointer to the input buffer (150528 bytes), or NULL if not inited.
 */
int8_t *AppAI_TipFocus_GetInputBuffer(void);

/**
 * @brief Get the model's input buffer info (scale, zero-point, size, etc.)
 *        so the caller can use the shared YUV422 -> int8 preprocessing
 *        pipeline (AppAI_PreprocessYuv422FrameToInt8Input).
 * @return Pointer to the input buffer descriptor, or NULL if not inited.
 */
const void *AppAI_TipFocus_GetInputBufferInfo(void);

/**
 * @brief Dry-run self-test: fills input with determinisitic values, runs one
 *        inference, and logs heatmap statistics.
 *
 * Call once after boot to verify the model loads and produces repeatable
 * output without requiring camera frames.
 *
 * @retval true  Inference completed; statistics logged via DebugConsole.
 * @retval false Initialisation or run failed.
 */
bool AppAI_TipFocus_DryRun(void);

#else  /* APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE */

/* Stub declarations so callers can compile unconditionally. */
static inline bool AppAI_TipFocus_Init(void) { return true; }
static inline bool AppAI_TipFocus_Run(void) { return true; }
static inline const int8_t *AppAI_TipFocus_GetTipHeatmap(void) { return NULL; }
static inline const int8_t *AppAI_TipFocus_GetCenterHeatmap(void) { return NULL; }
static inline int8_t  AppAI_TipFocus_GetConfidenceRaw(void) { return -128; }
static inline int8_t *AppAI_TipFocus_GetInputBuffer(void) { return NULL; }
static inline const void *AppAI_TipFocus_GetInputBufferInfo(void) { return NULL; }
static inline bool AppAI_TipFocus_DryRun(void) { return true; }

#endif /* APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE */

#endif /* AI_NETWORK_TIP_FOCUS_V4_112_INT8_H */
