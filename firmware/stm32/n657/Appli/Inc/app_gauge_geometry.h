/* USER CODE BEGIN Header */
/**
 *******************************************************************************
 * @file    app_gauge_geometry.h
 * @brief   Shared gauge crop geometry for the AI and classical baseline paths.
 *******************************************************************************
 */
/* USER CODE END Header */

#ifndef __APP_GAUGE_GEOMETRY_H
#define __APP_GAUGE_GEOMETRY_H

#include <stddef.h>

/* Shared crop ratios for the stable gauge framing used during training. */
#define APP_GAUGE_TRAINING_CROP_X_MIN_RATIO   0.1027f
#define APP_GAUGE_TRAINING_CROP_Y_MIN_RATIO   0.2573f
#define APP_GAUGE_TRAINING_CROP_X_MAX_RATIO   0.7987f
#define APP_GAUGE_TRAINING_CROP_Y_MAX_RATIO   0.8071f

/* Keep the legacy per-module names so the existing call sites stay readable. */
#define APP_AI_TRAINING_CROP_X_MIN_RATIO \
		APP_GAUGE_TRAINING_CROP_X_MIN_RATIO
#define APP_AI_TRAINING_CROP_Y_MIN_RATIO \
		APP_GAUGE_TRAINING_CROP_Y_MIN_RATIO
#define APP_AI_TRAINING_CROP_X_MAX_RATIO \
		APP_GAUGE_TRAINING_CROP_X_MAX_RATIO
#define APP_AI_TRAINING_CROP_Y_MAX_RATIO \
		APP_GAUGE_TRAINING_CROP_Y_MAX_RATIO

#define APP_BASELINE_TRAINING_CROP_X_MIN_RATIO \
		APP_GAUGE_TRAINING_CROP_X_MIN_RATIO
#define APP_BASELINE_TRAINING_CROP_Y_MIN_RATIO \
		APP_GAUGE_TRAINING_CROP_Y_MIN_RATIO
#define APP_BASELINE_TRAINING_CROP_X_MAX_RATIO \
		APP_GAUGE_TRAINING_CROP_X_MAX_RATIO
#define APP_BASELINE_TRAINING_CROP_Y_MAX_RATIO \
		APP_GAUGE_TRAINING_CROP_Y_MAX_RATIO

typedef struct {
	size_t x_min;
	size_t y_min;
	size_t width;
	size_t height;
} AppGaugeGeometry_Crop_t;

/**
 * @brief Build the stable training crop for a frame with the given size.
 */
static inline AppGaugeGeometry_Crop_t AppGaugeGeometry_TrainingCrop(
		size_t frame_width_pixels, size_t frame_height_pixels) {
	AppGaugeGeometry_Crop_t crop = { 0U, 0U, 0U, 0U };
	const size_t crop_x_min = (size_t) ((float) frame_width_pixels
			* APP_GAUGE_TRAINING_CROP_X_MIN_RATIO);
	const size_t crop_y_min = (size_t) ((float) frame_height_pixels
			* APP_GAUGE_TRAINING_CROP_Y_MIN_RATIO);
	const size_t crop_width = (size_t) ((float) frame_width_pixels
			* (APP_GAUGE_TRAINING_CROP_X_MAX_RATIO
					- APP_GAUGE_TRAINING_CROP_X_MIN_RATIO));
	const size_t crop_height = (size_t) ((float) frame_height_pixels
			* (APP_GAUGE_TRAINING_CROP_Y_MAX_RATIO
					- APP_GAUGE_TRAINING_CROP_Y_MIN_RATIO));

	crop.x_min = crop_x_min;
	crop.y_min = crop_y_min;
	crop.width = (crop_width > 0U) ? crop_width : 1U;
	crop.height = (crop_height > 0U) ? crop_height : 1U;
	return crop;
}

/**
 * @brief Return the center point of the stable training crop.
 */
static inline void AppGaugeGeometry_TrainingCropCenter(
		size_t frame_width_pixels, size_t frame_height_pixels,
		size_t *center_x_out, size_t *center_y_out) {
	const AppGaugeGeometry_Crop_t crop =
			AppGaugeGeometry_TrainingCrop(frame_width_pixels,
					frame_height_pixels);

	if (center_x_out != NULL) {
		*center_x_out = crop.x_min + (crop.width / 2U);
	}
	if (center_y_out != NULL) {
		*center_y_out = crop.y_min + (crop.height / 2U);
	}
}

#endif /* __APP_GAUGE_GEOMETRY_H */
