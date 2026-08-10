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

#include "app_ai_config.h"

/* Shared crop ratios for the stable gauge framing used during training. */
#define APP_GAUGE_TRAINING_CROP_X_MIN_RATIO 0.1027f
#define APP_GAUGE_TRAINING_CROP_Y_MIN_RATIO 0.2573f
#define APP_GAUGE_TRAINING_CROP_X_MAX_RATIO 0.7987f
#define APP_GAUGE_TRAINING_CROP_Y_MAX_RATIO 0.8071f

/* Inner Celsius dial center ratios — calibrated from board captures.
 * Gives center ~(129,121) on a 224x224 frame, which is the correct pivot
 * for the polar needle vote on the inner Celsius scale. */
#define APP_GAUGE_INNER_DIAL_CENTER_X_RATIO 0.576f
#define APP_GAUGE_INNER_DIAL_CENTER_Y_RATIO 0.540f

/* Celsius needle pivot offset from the OBB bounding-box geometric centre.
 * Expressed as fraction of the frame dimension so it scales with resolution.
 * Positive values move the pivot right / down from the OBB centre.
 * Calibrated from the rim-geometry detector on the live board:
 *   pivot_x = obb_centre_abs_x + frame_width  * APP_GAUGE_OBB_PIVOT_X_OFFSET_RATIO
 *   pivot_y = obb_centre_abs_y + frame_height * APP_GAUGE_OBB_PIVOT_Y_OFFSET_RATIO
 */
/* Inner Celsius dial radius as a fraction of the frame dimension.
 * The baseline derives this from training-crop height × 0.56, which
 * on a 224x224 frame produces about 68.9 px.  Using a fixed frame ratio
 * keeps the polar-vote scan ring stable regardless of lighting.
 * The OBB box size is lighting-dependent and cannot supply a
 * reliable radius for the polar annulus. */
#define APP_GAUGE_INNER_DIAL_RADIUS_FRAME_RATIO 0.3076f

/* Celsius needle pivot offset from the OBB bounding-box geometric centre.
 * Expressed as fraction of the frame dimension so it scales with resolution.
 * Positive values move the pivot right / down from the OBB centre.
 * Calibrated from the rim-geometry detector on the live board:
 *   pivot_x = obb_centre_abs_x + frame_width  * APP_GAUGE_OBB_PIVOT_X_OFFSET_RATIO
 *   pivot_y = obb_centre_abs_y + frame_height * APP_GAUGE_OBB_PIVOT_Y_OFFSET_RATIO
 */
#define APP_GAUGE_OBB_PIVOT_X_OFFSET_RATIO (-0.0089f)  /* ≈ −2 px */
#define APP_GAUGE_OBB_PIVOT_Y_OFFSET_RATIO  0.0625f    /* ≈ +14 px */

/* The active gauge profile mirrors the matching TOML section.  Both current
 * profiles use the same signed north-zero endpoint angles; the thermometer
 * changes only the physical value range. */
#if APP_GAUGE_ACTIVE_PROFILE == APP_GAUGE_PROFILE_COMPASS_HYGROMETER_SILVER
#define APP_GAUGE_CALIBRATION_MIN_DEG       (-135.0f)
#define APP_GAUGE_CALIBRATION_MAX_DEG       (135.0f)
#define APP_GAUGE_CALIBRATION_MIN_VALUE     (0.0f)
#define APP_GAUGE_CALIBRATION_MAX_VALUE     (100.0f)
#elif APP_GAUGE_ACTIVE_PROFILE == APP_GAUGE_PROFILE_COMPASS_BAROMETER_SILVER
#define APP_GAUGE_CALIBRATION_MIN_DEG       (-157.5f)
#define APP_GAUGE_CALIBRATION_MAX_DEG       (157.5f)
#define APP_GAUGE_CALIBRATION_MIN_VALUE     (940.0f)
#define APP_GAUGE_CALIBRATION_MAX_VALUE     (1080.0f)
#elif APP_GAUGE_ACTIVE_PROFILE == APP_GAUGE_PROFILE_COMPASS_THERMOMETER_SILVER
#define APP_GAUGE_CALIBRATION_MIN_DEG       (-135.0f)
#define APP_GAUGE_CALIBRATION_MAX_DEG       (135.0f)
#define APP_GAUGE_CALIBRATION_MIN_VALUE     (-35.0f)
#define APP_GAUGE_CALIBRATION_MAX_VALUE     (55.0f)
#else
#define APP_GAUGE_CALIBRATION_MIN_DEG       (-135.0f)
#define APP_GAUGE_CALIBRATION_MAX_DEG       (135.0f)
#define APP_GAUGE_CALIBRATION_MIN_VALUE     (-30.0f)
#define APP_GAUGE_CALIBRATION_MAX_VALUE     (50.0f)
#endif

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

typedef struct
{
	size_t x_min;
	size_t y_min;
	size_t width;
	size_t height;
} AppGaugeGeometry_Crop_t;

/**
 * @brief Build the stable training crop for a frame with the given size.
 */
static inline AppGaugeGeometry_Crop_t AppGaugeGeometry_TrainingCrop(
	size_t frame_width_pixels, size_t frame_height_pixels)
{
	AppGaugeGeometry_Crop_t crop = {0U, 0U, 0U, 0U};
	const size_t crop_x_min = (size_t)((float)frame_width_pixels * APP_GAUGE_TRAINING_CROP_X_MIN_RATIO);
	const size_t crop_y_min = (size_t)((float)frame_height_pixels * APP_GAUGE_TRAINING_CROP_Y_MIN_RATIO);
	const size_t crop_width = (size_t)((float)frame_width_pixels * (APP_GAUGE_TRAINING_CROP_X_MAX_RATIO - APP_GAUGE_TRAINING_CROP_X_MIN_RATIO));
	const size_t crop_height = (size_t)((float)frame_height_pixels * (APP_GAUGE_TRAINING_CROP_Y_MAX_RATIO - APP_GAUGE_TRAINING_CROP_Y_MIN_RATIO));

	crop.x_min = crop_x_min;
	crop.y_min = crop_y_min;
	crop.width = (crop_width > 0U) ? crop_width : 1U;
	crop.height = (crop_height > 0U) ? crop_height : 1U;
	return crop;
}

/**
 * @brief Return the center point of the inner Celsius dial.
 *
 * The inner dial sits in the bottom-left of the full gauge crop. Using this
 * center as the polar-vote pivot gives correct needle angles for the Celsius
 * scale, rather than the outer Fahrenheit dial center.
 */
static inline void AppGaugeGeometry_TrainingCropCenter(
	size_t frame_width_pixels, size_t frame_height_pixels,
	size_t *center_x_out, size_t *center_y_out)
{

	if (center_x_out != NULL)
	{
		*center_x_out = (size_t)((float)frame_width_pixels * APP_GAUGE_INNER_DIAL_CENTER_X_RATIO);
	}
	if (center_y_out != NULL)
	{
		*center_y_out = (size_t)((float)frame_height_pixels * APP_GAUGE_INNER_DIAL_CENTER_Y_RATIO);
	}
}

/**
 * @brief Map a north-zero gauge angle to the gauge-1 calibrated value.
 *
 * The endpoints are the TOML values, so this function deliberately contains
 * no board-specific offset or gain stage.
 */
static inline float AppGaugeGeometry_AngleToGauge1Value(float angle_deg)
{
	const float fraction = (angle_deg - APP_GAUGE_CALIBRATION_MIN_DEG) /
		(APP_GAUGE_CALIBRATION_MAX_DEG - APP_GAUGE_CALIBRATION_MIN_DEG);
	return APP_GAUGE_CALIBRATION_MIN_VALUE + fraction *
		(APP_GAUGE_CALIBRATION_MAX_VALUE - APP_GAUGE_CALIBRATION_MIN_VALUE);
}

#endif /* __APP_GAUGE_GEOMETRY_H */
