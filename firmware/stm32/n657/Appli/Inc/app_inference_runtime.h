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

/**
 * @brief Report whether the AI worker still owns the capture buffer.
 * @retval true while the queued request or model execution is active.
 */
bool AppInferenceRuntime_IsInferenceInFlight(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_INFERENCE_RUNTIME_H */
