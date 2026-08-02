/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_memory_budget.h
 * @brief   Shared memory sizing constants for the application.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __APP_MEMORY_BUDGET_H
#define __APP_MEMORY_BUDGET_H

#ifdef __cplusplus
extern "C" {
#endif

/* Shared memory budgets ---------------------------------------------------- */
/* Keep these in one place so we can tune the app footprint without digging
 * through the thread and capture logic. */
#define NEWLIB_HEAP_LIMIT_ADDR          0x34100000UL
#define INFERENCE_LOG_THREAD_STACK_SIZE_BYTES   8192U
#define INFERENCE_LOG_QUEUE_DEPTH               8U

#define CAMERA_INIT_THREAD_STACK_SIZE_BYTES     16384U
#define CAMERA_ISP_THREAD_STACK_SIZE_BYTES      4096U
#define CAMERA_HEARTBEAT_THREAD_STACK_SIZE_BYTES 1024U
/* Keep the AI worker stack large enough for preprocessing and two sequential
 * ATON calls while leaving the shared reloc runtime window untouched. */
#define CAMERA_AI_THREAD_STACK_SIZE_BYTES      16384U
#define BASELINE_RUNTIME_THREAD_STACK_SIZE_BYTES 16384U
#define IMAGE_CLEANUP_THREAD_STACK_SIZE_BYTES    4096U

/* Capture geometry --------------------------------------------------------- */
/* DCMIPP resizes the complete IMX335 frame to the live ellipse contract.
 * This is a packed one-byte MONO_Y8 image, not a cropped YUV422 frame. */
#define CAMERA_CAPTURE_WIDTH_PIXELS             640U
#define CAMERA_CAPTURE_HEIGHT_PIXELS            640U
#define CAMERA_CAPTURE_BUFFER_COUNT             1U
#define CAMERA_CAPTURE_BYTES_PER_PIXEL          1U
#define CAMERA_CAPTURE_BUFFER_SIZE_BYTES        (CAMERA_CAPTURE_WIDTH_PIXELS * CAMERA_CAPTURE_HEIGHT_PIXELS * CAMERA_CAPTURE_BYTES_PER_PIXEL)

/* New sequential gauge-model contracts. DCMIPP supplies a complete 640x640
 * grayscale frame; firmware resizes it to the 384x384 ellipse input, then
 * the ellipse selects the center/tip crop.
 * The NPU activation pools are reused per model. */
#define GAUGE_ELLIPSE_INPUT_WIDTH_PIXELS        384U
#define GAUGE_ELLIPSE_INPUT_HEIGHT_PIXELS       384U
#define GAUGE_ELLIPSE_INPUT_CHANNELS            1U
#define GAUGE_ELLIPSE_INPUT_SIZE_BYTES          (GAUGE_ELLIPSE_INPUT_WIDTH_PIXELS * GAUGE_ELLIPSE_INPUT_HEIGHT_PIXELS * GAUGE_ELLIPSE_INPUT_CHANNELS)
#define GAUGE_CENTER_TIP_INPUT_WIDTH_PIXELS     224U
#define GAUGE_CENTER_TIP_INPUT_HEIGHT_PIXELS    224U
#define GAUGE_CENTER_TIP_INPUT_CHANNELS         1U
#define GAUGE_CENTER_TIP_INPUT_SIZE_BYTES       (GAUGE_CENTER_TIP_INPUT_WIDTH_PIXELS * GAUGE_CENTER_TIP_INPUT_HEIGHT_PIXELS * GAUGE_CENTER_TIP_INPUT_CHANNELS)
#define GAUGE_CENTER_TIP_OUTPUT_SIZE_BYTES      (56U * 56U * 2U)
#define GAUGE_MODEL_NPU_POOL_USED_BYTES         (400U * 1024U)

#ifdef __cplusplus
}
#endif

#endif /* __APP_MEMORY_BUDGET_H */
