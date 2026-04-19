/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_baseline_runtime.h
 * @brief   Classical CV baseline worker for temperature estimation.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __APP_BASELINE_RUNTIME_H
#define __APP_BASELINE_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

#include "tx_api.h"

/**
 * @brief Initialize the baseline runtime synchronization objects.
 *
 * The baseline worker runs independently from the learned AI runtime so the
 * classical comparator can stay alive even when the model worker is busy.
 *
 * @retval TX_SUCCESS on success.
 */
UINT AppBaselineRuntime_Init(void);

/**
 * @brief Start the baseline worker thread.
 *
 * The worker consumes copied YUV422 frames from a private snapshot buffer and
 * emits a temperature estimate for each accepted camera frame.
 *
 * @retval TX_SUCCESS on success.
 */
UINT AppBaselineRuntime_Start(void);

/**
 * @brief Queue a frame for the baseline temperature estimate.
 *
 * @param frame_ptr Pointer to the captured frame bytes.
 * @param frame_length Number of valid bytes in the frame.
 * @retval true when the request was queued successfully.
 * @retval false when the runtime is unavailable or the frame is invalid.
 */
bool AppBaselineRuntime_RequestEstimate(const uint8_t *frame_ptr,
		ULONG frame_length);

#ifdef __cplusplus
}
#endif

#endif /* __APP_BASELINE_RUNTIME_H */
