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

/* Force the live firmware onto the sc128 tip-focus geometry path even when
 * the generated CubeIDE makefile leaves the stage switch at its default 0.
 * This keeps app_ai.c, app_threadx.c, and the runtime tail aligned on the
 * same compile-time route. */
#ifdef APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
#undef APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE
#endif
#define APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE 1U

/* Tip-focus SimCC coordinate stage.
 * This is the active live-board geometry model wrapper, so expose it through
 * the public AI header for the runtime and the ThreadX bootstrap. */
#include "ai_network_tip_focus_v4_112_int8.h"

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
 * The helper converts the colour frame into the model int8 RGB input buffer,
 * runs the generated LL_ATON runtime once, and logs the SimCC output
 * summary.
 *
 * @param frame_bytes Pointer to the captured frame bytes.
 * @param frame_size Number of valid bytes in the captured frame.
 * @retval true when the runtime run completes successfully.
 * @retval false when preprocessing or runtime execution fails.
 */
bool App_AI_RunDryInferenceFromYuv422(const uint8_t *frame_bytes,
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
 * @brief Verify that tip-focus weights are programmed in xSPI2 flash.
 *
 * Reads the signature bytes from xSPI2 at 0x70400000 and compares
 * against the expected simcc_gauge_v2_spatial_qat_sc128_int8_atonbuf.xSPI2.raw header.
 *
 * @retval true xSPI2 contains valid tip-focus weights.
 * @retval false xSPI2 is empty or corrupted - run flash_boot.ps1.
 */
bool AppAI_VerifyTipFocusWeights(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_AI_H */
