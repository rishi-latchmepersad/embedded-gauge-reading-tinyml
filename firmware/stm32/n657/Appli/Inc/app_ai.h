/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_ai.h
 * @brief   Minimal AI runtime bootstrap helpers.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __APP_AI_H
#define __APP_AI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Initialize the generated AI runtime package.
 *
 * This is step 1 only: we bring the runtime up and validate the generated
 * network package links correctly, but we do not execute inference yet.
 *
 * @retval true when the runtime init calls succeed.
 * @retval false when the model package fails to initialize.
 */
bool App_AI_Model_Init(void);

/**
 * @brief Run a one-shot inference using a captured 224x224 YUV422 frame.
 *
 * The helper converts the frame into the model's float32 RGB input buffer,
 * runs the generated LL_ATON runtime once, and logs the output summary.
 *
 * @param frame_bytes Pointer to the captured frame bytes.
 * @param frame_size Number of valid bytes in the captured frame.
 * @retval true when the runtime run completes successfully.
 * @retval false when preprocessing or runtime execution fails.
 */
bool App_AI_RunDryInferenceFromYuv422(const uint8_t *frame_bytes,
		size_t frame_size);

#ifdef __cplusplus
}
#endif

#endif /* __APP_AI_H */
