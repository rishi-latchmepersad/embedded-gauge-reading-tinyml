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

/** @brief Path to the source-crop-box model image on xSPI2 flash. */
#define APP_AI_SOURCE_CROP_BOX_XSPI2_MODEL_IMAGE_PATH \
	"atonbuf.source_crop_box.xSPI2.raw"

/** @brief Base address for the source-crop-box model in xSPI2 mapped window. */
#define APP_AI_XSPI2_SOURCE_CROP_BOX_BASE_ADDR 0x70B00000UL

/** @brief Chip offset for the source-crop-box model from xSPI2 chip base. */
#define APP_AI_XSPI2_SOURCE_CROP_BOX_CHIP_OFFSET (APP_AI_XSPI2_SOURCE_CROP_BOX_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)

/** @brief Source-space xyxy crop box produced by the source-crop-box localizer. */
typedef struct
{
	float x_min;
	float y_min;
	float x_max;
	float y_max;
} AppAI_SourceCropBox;

#ifndef APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE
#define APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE 1U
#endif

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
 * @brief Retrieve the most recent inference scalar result.
 *
 * @param[out] value_out Receives the last dequantized inference value.
 * @retval true when a valid result has been produced since boot.
 * @retval false when no inference has completed yet or value_out is NULL.
 */
bool App_AI_GetLastInferenceResult(float *value_out);

#ifdef __cplusplus
}
#endif

#endif /* __APP_AI_H */
