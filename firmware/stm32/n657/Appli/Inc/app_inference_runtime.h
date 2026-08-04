/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_inference_runtime.h
 * @brief   Inference worker and logging runtime helpers.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __APP_INFERENCE_RUNTIME_H
#define __APP_INFERENCE_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

#include "tx_api.h"

UINT AppInferenceRuntime_Init(void);
UINT AppInferenceRuntime_Start(void);
bool AppInferenceRuntime_RequestDryInference(const uint8_t *frame_ptr,
		ULONG frame_length);

/** @brief State of the single-owner AI worker and its current request. */
typedef enum {
	APP_INFERENCE_WORKER_UNINITIALIZED = 0,
	APP_INFERENCE_WORKER_WAITING,
	APP_INFERENCE_WORKER_QUEUED,
	APP_INFERENCE_WORKER_EXECUTING,
	APP_INFERENCE_WORKER_PUBLISHING,
	APP_INFERENCE_WORKER_FAILED,
} AppInferenceRuntime_WorkerState_t;

/** @brief Return the AI worker state for bounded-wait diagnostics. */
AppInferenceRuntime_WorkerState_t AppInferenceRuntime_GetWorkerState(void);

/** @brief Return the monotonically increasing AI request generation. */
ULONG AppInferenceRuntime_GetRequestGeneration(void);

/** @brief Return the last AI worker progress tick. */
ULONG AppInferenceRuntime_GetWorkerProgressTick(void);

/** @brief Convert an AI worker state to a stable diagnostic label. */
const char *AppInferenceRuntime_WorkerStateName(
		AppInferenceRuntime_WorkerState_t state);

/**
 * @brief Report whether the AI worker still owns the capture buffer.
 * @retval true while the queued request or model execution is active.
 */
bool AppInferenceRuntime_IsInferenceInFlight(void);

/** @brief Return whether the most recently completed AI request succeeded. */
bool AppInferenceRuntime_WasLastRequestSuccessful(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_INFERENCE_RUNTIME_H */
