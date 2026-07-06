/* Guarded wrapper API for the board-fit tip-focus heatmap model
 * (`tip_focus_v18_int8`, LL_ATON NPU runtime).
 *
 * The generated package uses quantized internal tensors with float32 I/O at
 * the firmware boundary. This wrapper keeps the higher-level firmware API
 * stable while swapping the live model contract to the latest board-fit
 * package.
 *
 * Output order:
 *   output[0] = center_heatmap [1,56,56,1] float
 *   output[1] = confidence     [1,1]      float
 *   output[2] = tip_heatmap    [1,56,56,1] float
 *   output[3] = is_main_needle [1,1]      float
 */

#ifndef AI_NETWORK_TIP_FOCUS_V4_112_INT8_H
#define AI_NETWORK_TIP_FOCUS_V4_112_INT8_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
#define APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE 1U
#endif

#ifndef APP_AI_ENABLE_TIP_FOCUS_BOOT_DRY_RUN
#define APP_AI_ENABLE_TIP_FOCUS_BOOT_DRY_RUN 0U
#endif

#if APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE

bool AppAI_TipFocus_Init(void);
bool AppAI_TipFocus_Run(void);
float *AppAI_TipFocus_GetInputBuffer(void);
const void *AppAI_TipFocus_GetInputBufferInfo(void);
const float *AppAI_TipFocus_GetCenterHeatmap(void);
const float *AppAI_TipFocus_GetTipHeatmap(void);
float AppAI_TipFocus_GetConfidence(void);
float AppAI_TipFocus_GetIsMainNeedle(void);
bool AppAI_TipFocus_DryRun(void);

#else

static inline bool AppAI_TipFocus_Init(void) { return true; }
static inline bool AppAI_TipFocus_Run(void) { return true; }
static inline float *AppAI_TipFocus_GetInputBuffer(void) { return NULL; }
static inline const void *AppAI_TipFocus_GetInputBufferInfo(void) { return NULL; }
static inline const float *AppAI_TipFocus_GetCenterHeatmap(void) { return NULL; }
static inline const float *AppAI_TipFocus_GetTipHeatmap(void) { return NULL; }
static inline float AppAI_TipFocus_GetConfidence(void) { return 0.0f; }
static inline float AppAI_TipFocus_GetIsMainNeedle(void) { return 0.0f; }
static inline bool AppAI_TipFocus_DryRun(void) { return true; }

#endif

#endif /* AI_NETWORK_TIP_FOCUS_V4_112_INT8_H */
