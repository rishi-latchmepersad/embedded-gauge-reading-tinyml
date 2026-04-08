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
#define INFERENCE_LOG_THREAD_STACK_SIZE_BYTES   8192U
#define INFERENCE_LOG_QUEUE_DEPTH               8U

#define CAMERA_INIT_THREAD_STACK_SIZE_BYTES     8192U
#define CAMERA_ISP_THREAD_STACK_SIZE_BYTES      4096U
#define CAMERA_HEARTBEAT_THREAD_STACK_SIZE_BYTES 1024U
#define CAMERA_AI_THREAD_STACK_SIZE_BYTES       16384U

/* Capture geometry --------------------------------------------------------- */
/* Use the same 224x224 frame budget everywhere so the pipeline stays simple. */
#define CAMERA_CAPTURE_WIDTH_PIXELS             224U
#define CAMERA_CAPTURE_HEIGHT_PIXELS            224U
#define CAMERA_CAPTURE_BUFFER_COUNT             1U
#define CAMERA_CAPTURE_BYTES_PER_PIXEL          2U
#define CAMERA_CAPTURE_BUFFER_SIZE_BYTES        (CAMERA_CAPTURE_WIDTH_PIXELS * CAMERA_CAPTURE_HEIGHT_PIXELS * CAMERA_CAPTURE_BYTES_PER_PIXEL)

#ifdef __cplusplus
}
#endif

#endif /* __APP_MEMORY_BUDGET_H */
