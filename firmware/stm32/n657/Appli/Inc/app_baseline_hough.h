/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_baseline_hough.h
 * @brief   Small board baseline: calibrated center plus radial Hough vote.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __APP_BASELINE_HOUGH_H
#define __APP_BASELINE_HOUGH_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "app_baseline_runtime.h"

/**
 * @brief Estimate one gauge reading with the fresh simple baseline.
 *
 * The detector uses the calibrated inner-dial center and scores 360 radial
 * Hough bins by dark-line contrast plus along-ray continuity. It deliberately
 * has no rim override, hot-zone rescue, or multi-family selector.
 *
 * @param frame_bytes Packed YUV422 frame buffer.
 * @param frame_size Number of valid bytes in @p frame_bytes.
 * @param frame_width_pixels Frame width in pixels.
 * @param frame_height_pixels Frame height in pixels.
 * @param estimate_out Destination estimate structure.
 * @return true when a separated radial peak is found.
 */
bool AppBaselineHough_Estimate(
	const uint8_t *frame_bytes, size_t frame_size,
	size_t frame_width_pixels, size_t frame_height_pixels,
	AppBaselineRuntime_Estimate_t *estimate_out);

#ifdef __cplusplus
}
#endif

#endif /* __APP_BASELINE_HOUGH_H */
