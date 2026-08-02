/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_baseline_template.h
 * @brief   Classical normalized-template matcher for board gauge captures.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __APP_BASELINE_TEMPLATE_H
#define __APP_BASELINE_TEMPLATE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "app_baseline_runtime.h"

/**
 * @brief Estimate one board reading using normalized luma template matching.
 * @param frame_bytes Packed YUV422 frame.
 * @param frame_size Number of valid bytes in @p frame_bytes.
 * @param frame_width_pixels Frame width.
 * @param frame_height_pixels Frame height.
 * @param estimate_out Destination estimate structure.
 * @return true when the frame descriptor and template comparison succeed.
 */
bool AppBaselineTemplate_Estimate(
	const uint8_t *frame_bytes, size_t frame_size,
	size_t frame_width_pixels, size_t frame_height_pixels,
	AppBaselineRuntime_Estimate_t *estimate_out);

#ifdef __cplusplus
}
#endif

#endif /* __APP_BASELINE_TEMPLATE_H */
