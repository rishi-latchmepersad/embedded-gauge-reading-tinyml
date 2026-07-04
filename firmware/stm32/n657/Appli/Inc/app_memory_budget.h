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
/* Keep the AI worker stack large enough for the OBB->UNet cascade while
 * leaving room in RAM for the OBB reloc image and the rest of the app. */
#define CAMERA_AI_THREAD_STACK_SIZE_BYTES      16384U
#define BASELINE_RUNTIME_THREAD_STACK_SIZE_BYTES 16384U
#define IMAGE_CLEANUP_THREAD_STACK_SIZE_BYTES    4096U

/* Capture geometry --------------------------------------------------------- */
/* Standardize the live capture budget on 224x224 so the AI and baseline
 * pipelines see the same square frame that the current student models use. */
#define CAMERA_CAPTURE_WIDTH_PIXELS             224U
#define CAMERA_CAPTURE_HEIGHT_PIXELS            224U
#define CAMERA_CAPTURE_BUFFER_COUNT             1U
#define CAMERA_CAPTURE_BYTES_PER_PIXEL          2U
#define CAMERA_CAPTURE_BUFFER_SIZE_BYTES        (CAMERA_CAPTURE_WIDTH_PIXELS * CAMERA_CAPTURE_HEIGHT_PIXELS * CAMERA_CAPTURE_BYTES_PER_PIXEL)

#ifdef __cplusplus
}
#endif

#endif /* __APP_MEMORY_BUDGET_H */
