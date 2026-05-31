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

/* Prod v0.8: source-crop-box stage is disabled by default.
 * The OBB + luma + polar-vote cascade replaces it.
 * Define APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE=1 from the compiler command line
 * to re-enable for debug comparisons. */
#ifndef APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE
#define APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE 0U
#endif

#if APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE
/** @brief Path to the source-crop-box model image on xSPI2 flash (legacy debug-only). */
#define APP_AI_SOURCE_CROP_BOX_XSPI2_MODEL_IMAGE_PATH \
	"atonbuf.source_crop_box.xSPI2.raw"

/** @brief Base address for the source-crop-box model in xSPI2 mapped window (legacy). */
#define APP_AI_XSPI2_SOURCE_CROP_BOX_BASE_ADDR 0x70B00000UL

/** @brief Chip offset for the source-crop-box model from xSPI2 chip base (legacy). */
#define APP_AI_XSPI2_SOURCE_CROP_BOX_CHIP_OFFSET (APP_AI_XSPI2_SOURCE_CROP_BOX_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)

/** @brief Source-space xyxy crop box produced by the source-crop-box localizer (legacy). */
typedef struct
{
	float x_min;
	float y_min;
	float x_max;
	float y_max;
} AppAI_SourceCropBox;
#endif /* APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE */

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
 * The helper converts the frame into the model float32 RGB input buffer,
 * runs the generated LL_ATON runtime once, and logs the output summary.
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
 * The generated LL_ATON code dereferences _mem_pool_xSPI2_Default pointers
 * to read coefficient vectors directly from xSPI2 flash.  If a prior stage
 * left xSPI2 in indirect mode, those CPU-side reads will hang the bus.
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
 * @brief Verify that tip-focus weights are programmed in xSPI2 flash
 *        (legacy debug-only; tip-focus stage is disabled by default in prod v0.8).
 *
 * Reads the signature bytes from xSPI2 at 0x70400000 and compares
 * against the expected network_atonbuf.xSPI2.raw header.
 *
 * @retval true xSPI2 contains valid tip-focus weights.
 * @retval false xSPI2 is empty or corrupted - run flash_boot.bat.
 */
bool AppAI_VerifyTipFocusWeights(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_AI_H */
