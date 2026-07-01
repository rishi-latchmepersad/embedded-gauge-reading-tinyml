/**
 * @file    app_ai_state.h
 * @brief   Shared global state declarations for the AI pipeline.
 *
 * All variables previously declared `static` in app_ai.c's Private Variables
 * section are declared here as `extern` so other compilation units (e.g.
 * app_center_detector.c, app_baseline_runtime.c) can access them directly
 * without duplicating definitions or relying on getter/setter trampolines.
 *
 * Memory-pool symbols with __attribute__((section(...))) remain in app_ai.c
 * and are NOT declared here — the linker resolves those via the section
 * layout, not through extern references.
 */

#ifndef __APP_AI_STATE_H
#define __APP_AI_STATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "tx_api.h"
#include "app_ai_types.h"
#include "app_memory_budget.h"

/* ------------------------------------------------------------------ */
/* Macros kept here so referencing TUs can use them                   */
/* ------------------------------------------------------------------ */
#define APP_AI_OBB_CENTER_EMA_ALPHA        0.20f
#define APP_AI_INFERENCE_BURST_HISTORY_SIZE 3U
#define APP_AI_INFERENCE_BURST_RESET_DELTA_C 12.0f
#define APP_AI_INFERENCE_VALUE_MIN_C (-80.0f)
#define APP_AI_INFERENCE_VALUE_MAX_C (180.0f)
#define APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE 3U
#define APP_AI_TIP_FOCUS_MAX_OUTLIER_DELTA_C 5.0f
#define APP_AI_TIP_FOCUS_OUTLIER_RESET_STREAK 3U
#define APP_AI_TIP_FOCUS_MAX_INVALID_FRAMES  10U
#define APP_AI_TIP_FOCUS_HEATMAP_SIDE_PIXELS 56U
#define APP_AI_XSPI2_PROBE_BYTES            16U
#define APP_AI_XSPI2_PROGRAM_CHUNK_BYTES    4096U
#define APP_AI_CACHE_LINE_BYTES             32U
#define APP_AI_CAPTURE_FRAME_WIDTH_PIXELS   CAMERA_CAPTURE_WIDTH_PIXELS
#define APP_AI_CAPTURE_FRAME_HEIGHT_PIXELS  CAMERA_CAPTURE_HEIGHT_PIXELS
#define APP_AI_CAPTURE_FRAME_BYTES_PER_PIXEL CAMERA_CAPTURE_BYTES_PER_PIXEL
#define APP_AI_SCALAR_PREPROCESS_ROW_TRACE_INTERVAL_ROWS 32U
#define APP_AI_SCALAR_PREPROCESS_ROWS_PER_CHUNK 8U

/* ------------------------------------------------------------------ */
/* Runtime state flags                                                */
/* ------------------------------------------------------------------ */
extern bool app_ai_runtime_initialized;
extern bool app_ai_aton_runtime_initialized;

/* ------------------------------------------------------------------ */
/* Last inference result (written from ISR / thread context)          */
/* ------------------------------------------------------------------ */
extern volatile float app_ai_last_inference_value;
extern volatile bool  app_ai_last_inference_valid;

/* ------------------------------------------------------------------ */
/* OBB centre EMA smoothing                                           */
/* ------------------------------------------------------------------ */
extern float app_ai_smoothed_obb_cx;
extern float app_ai_smoothed_obb_cy;

/* ------------------------------------------------------------------ */
/* Inference burst smoothing (3-frame median)                         */
/* ------------------------------------------------------------------ */
#if APP_AI_ENABLE_INFERENCE_BURST_SMOOTHING
extern float  app_ai_inference_burst_history[APP_AI_INFERENCE_BURST_HISTORY_SIZE];
extern size_t app_ai_inference_burst_history_count;
extern size_t app_ai_inference_burst_history_next_index;
#endif

/* ------------------------------------------------------------------ */
/* Hardware / xSPI2 init flags                                        */
/* ------------------------------------------------------------------ */
extern bool app_ai_npu_hw_initialized;
extern bool app_ai_xspi2_initialized;
extern bool app_ai_xspi2_mm_enabled;

/* ------------------------------------------------------------------ */
/* Currently loaded xSPI2 stage                                       */
/* ------------------------------------------------------------------ */
extern const struct AppAI_ModelStageSpec *app_ai_loaded_xspi2_stage;

/* ------------------------------------------------------------------ */
/* Tip-focus median smoothing ring buffer                             */
/* ------------------------------------------------------------------ */
extern float  app_ai_tip_focus_median_buffer[APP_AI_TIP_FOCUS_MEDIAN_BUFFER_SIZE];
extern size_t app_ai_tip_focus_median_count;
extern size_t app_ai_tip_focus_median_index;
extern float  app_ai_tip_focus_last_published;
extern bool   app_ai_tip_focus_last_published_valid;
extern uint32_t app_ai_tip_focus_consecutive_invalid;
extern uint32_t app_ai_tip_focus_outlier_streak;

/* ------------------------------------------------------------------ */
/* Forced (debug) crop override                                       */
/* ------------------------------------------------------------------ */
extern bool        app_ai_forced_crop_active;
extern size_t      app_ai_forced_crop_x_min;
extern size_t      app_ai_forced_crop_y_min;
extern size_t      app_ai_forced_crop_width;
extern size_t      app_ai_forced_crop_height;
extern const char *app_ai_forced_crop_label;

/* ------------------------------------------------------------------ */
/* Tip-focus input dump guard (one-shot)                              */
/* ------------------------------------------------------------------ */
extern bool app_ai_tip_focus_input_dump_done;

/* ------------------------------------------------------------------ */
/* Scalar preprocessing scratch buffers                               */
/* ------------------------------------------------------------------ */
extern uint8_t app_ai_xspi2_program_buffer[APP_AI_XSPI2_PROGRAM_CHUNK_BYTES];
extern uint8_t app_ai_scalar_row_scratch[APP_AI_CAPTURE_FRAME_WIDTH_PIXELS * APP_AI_CAPTURE_FRAME_BYTES_PER_PIXEL];
extern uint8_t app_ai_scalar_output_row_scratch[APP_AI_CAPTURE_FRAME_WIDTH_PIXELS * 3U * sizeof(float)];
extern volatile size_t app_ai_scalar_preprocess_last_row;

/* ------------------------------------------------------------------ */
/* xSPI2 flash signatures (const, compiled-in model ID bytes)         */
/* ------------------------------------------------------------------ */

/* Shared / legacy aliases (tip-focus is the live model) */
extern const uint8_t app_ai_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES];
extern const uint8_t app_ai_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES];

/* Rectified scalar v2 */
extern const uint8_t app_ai_rectifier_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES];
extern const uint8_t app_ai_rectifier_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES];

/* Board bbox OBB */
extern const uint8_t app_ai_obb_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES];
extern const uint8_t app_ai_obb_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES];

/* Source-crop-box */
extern const uint8_t app_ai_source_crop_box_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES];
extern const uint8_t app_ai_source_crop_box_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES];

/* Tip-focus v18 (live model) */
extern const uint8_t app_ai_tip_focus_xspi2_signature_start[APP_AI_XSPI2_PROBE_BYTES];
extern const uint8_t app_ai_tip_focus_xspi2_signature_tail[APP_AI_XSPI2_PROBE_BYTES];

/* ------------------------------------------------------------------ */
/* Per-stage programmed sizes (set during provisioning)               */
/* ------------------------------------------------------------------ */
extern ULONG app_ai_scalar_programmed_size;
extern ULONG app_ai_rectifier_programmed_size;
extern ULONG app_ai_obb_programmed_size;
extern ULONG app_ai_source_crop_box_programmed_size;
extern ULONG app_ai_xspi2_programmed_size;       /* legacy alias */
extern ULONG app_ai_tip_focus_programmed_size;

/* ------------------------------------------------------------------ */
/* Per-stage SD-sourced signature caches (populated at provisioning)  */
/* ------------------------------------------------------------------ */

/* Scalar */
extern uint8_t app_ai_scalar_sig_start[APP_AI_XSPI2_PROBE_BYTES];
extern uint8_t app_ai_scalar_sig_tail[APP_AI_XSPI2_PROBE_BYTES];
extern bool    app_ai_scalar_sig_valid;

/* Rectifier */
extern uint8_t app_ai_rectifier_sig_start[APP_AI_XSPI2_PROBE_BYTES];
extern uint8_t app_ai_rectifier_sig_tail[APP_AI_XSPI2_PROBE_BYTES];
extern bool    app_ai_rectifier_sig_valid;

/* OBB */
extern uint8_t app_ai_obb_sig_start[APP_AI_XSPI2_PROBE_BYTES];
extern uint8_t app_ai_obb_sig_tail[APP_AI_XSPI2_PROBE_BYTES];
extern bool    app_ai_obb_sig_valid;

/* Source-crop-box */
extern uint8_t app_ai_source_crop_box_sig_start[APP_AI_XSPI2_PROBE_BYTES];
extern uint8_t app_ai_source_crop_box_sig_tail[APP_AI_XSPI2_PROBE_BYTES];
extern bool    app_ai_source_crop_box_sig_valid;

/* Tip-focus */
extern uint8_t app_ai_tip_focus_sig_start[APP_AI_XSPI2_PROBE_BYTES];
extern uint8_t app_ai_tip_focus_sig_tail[APP_AI_XSPI2_PROBE_BYTES];
extern bool    app_ai_tip_focus_sig_valid;

/* ------------------------------------------------------------------ */
/* OBB stage spec                                                     */
/* ------------------------------------------------------------------ */
extern const AppAI_ModelStageSpec app_ai_obb_stage;

/* ------------------------------------------------------------------ */
/* Generated NN instances (from LL_ATON_DECLARE_NAMED macro)          */
/* ------------------------------------------------------------------ */
#include "ll_aton_rt_user_api.h"

extern NN_Instance_TypeDef NN_Instance_scalar_full_finetune_from_best_piecewise_calibrated_int8;
extern NN_Instance_TypeDef NN_Instance_mobilenetv2_rectifier_hardcase_finetune;
extern NN_Instance_TypeDef NN_Instance_obb_box_board_bbox_deploy_candidate;
extern NN_Instance_TypeDef NN_Instance_mobilenetv2_source_crop_box_v1_stripped_int8;
extern NN_Instance_TypeDef NN_Instance_heatmap_cd;

#endif /* __APP_AI_STATE_H */
