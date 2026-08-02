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

/* Keep the live geometry-stage route enabled even when the generated CubeIDE
 * makefile leaves its legacy switch at the default 0. The implementation now
 * runs the 384g ellipse encoder followed by keypoint U-Net v4. */
#ifdef APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
#undef APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
#endif
#define APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE 1U

/* The retired tip-focus wrapper is not part of the production header.  A
 * replay-only target must explicitly enable APP_AI_ENABLE_LEGACY_TIP_FOCUS_COMPAT
 * and include its compatibility header directly. */

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
 * @brief Run a one-shot inference using a captured 640x640 MONO_Y8 frame.
 *
 * The helper consumes the 640x640 DCMIPP luma, resizes it for the 384x384
 * ellipse encoder, builds the 224x224 keypoint tensor, runs both generated
 * networks, and maps the
 * final needle angle through the gauge-1 north-zero calibration endpoints.
 *
 * @param frame_bytes Pointer to the captured frame bytes.
 * @param frame_size Number of valid bytes in the captured frame.
 * @retval true when the runtime run completes successfully.
 * @retval false when preprocessing or runtime execution fails.
 */
bool App_AI_RunDryInferenceFromGray640(const uint8_t *frame_bytes,
		size_t frame_size);

/**
 * @brief Ensure xSPI2 flash is in memory-mapped mode for NPU weight access.
 *
 * The generated LL_ATON code dereferences the xSPI2 model pool directly to
 * read coefficient vectors from flash. If a prior stage left xSPI2 in
 * indirect mode, those CPU-side reads will hang the bus.
 *
 * Call this before any inference that uses pool-7+xSPI2 weight data.
 *
 * @retval true when xSPI2 is in memory-mapped mode (or was already).
 * @retval false when the MM-mode switch failed.
 */
bool AppAI_Xspi2EnsureMemoryMappedMode(void);

/**
 * @brief Retrieve the most recent inference scalar result.
 *
 * @param[out] value_out Receives the last dequantized inference value.
 * @retval true when a valid result has been produced since boot.
 * @retval false when no inference has completed yet or value_out is NULL.
 */
bool App_AI_GetLastInferenceResult(float *value_out);

/**
 * @brief Verify that the active gauge-model weights are programmed in xSPI2.
 *
 * The active model signatures are checked by the generated stage wrappers and
 * the flash script; this compatibility API remains for older callers.
 *
 * @retval true xSPI2 contains valid tip-focus weights.
 * @retval false xSPI2 is empty or corrupted - run flash_boot.ps1.
 */
bool AppAI_VerifyTipFocusWeights(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_AI_H */
