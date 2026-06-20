/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_center_detector.h
 * @brief   Heatmap center-detection CNN + polar-vote pipeline for gauge reading.
 *
 * Pipeline:
 *   1. Crop the full-frame YUV422 to the stable training crop family.
 *   2. Resize / convert to 224x224 uint8 RGB.
 *   3. Run center detector NPU inference -> 80x80 heatmap.
 *   4. Decode the heatmap with soft-argmax (intensity-weighted centroid).
 *   5. Map the centre back to full-frame pixel coordinates.
 *   6. Run polar vote around the full-frame centre -> needle angle.
 *   7. Convert angle -> temperature (deg C).
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __APP_CENTER_DETECTOR_H
#define __APP_CENTER_DETECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/** @brief Result from the center-detector pipeline. */
typedef struct
{
	bool valid;
	float center_x;      /**< Full-frame gauge centre column (pixels). */
	float center_y;      /**< Full-frame gauge centre row (pixels). */
	float needle_angle_rad; /**< Polar-vote needle angle (radians). */
	float temperature_c; /**< Converted temperature (deg C). */
	float confidence;    /**< Polar-vote confidence [0, 1]. */
} AppCenterDetector_Result_t;

/**
 * @brief Initialise the center-detector model (load NPU weights, init runtime).
 *
 * Call once after App_AI_Model_Init(). The staged heatmap blob is copied from
 * xSPI2 into AXISRAM2 before the network is initialised.
 *
 * @retval true  Model initialised and network init succeeded.
 * @retval false Initialisation failed.
 */
bool AppCenterDetector_Init(void);

/**
 * @brief Run the full center-detector pipeline on one frame.
 *
 * @param frame_bytes      Full-frame YUV422 buffer (must be valid for frame_size bytes).
 * @param frame_size       Size of the frame buffer in bytes.
 * @param crop_x_min       Left edge of the selected crop (pixels, full-frame coords).
 * @param crop_y_min       Top edge of the selected crop.
 * @param crop_width       Width of the selected crop (pixels).
 * @param crop_height      Height of the selected crop (pixels).
 * @param dial_radius_override_px  When > 0, overrides the internal polar-vote
 *                                 dial radius.  Pass 0 to use the legacy
 *                                 min(crop_width, crop_height) / 2 heuristic.
 * @param frame_width_pixels  Full-frame width in pixels (e.g. 224).
 * @param frame_height_pixels Full-frame height in pixels (e.g. 224).
 * @param[out] result      Filled with centre, angle, temperature, confidence.
 * @param override_center_x  When >= 0, skip the CNN and use this as the
 *                           full-frame polar-vote pivot (pixels).
 * @param override_center_y  When >= 0, skip the CNN and use this as the
 *                           full-frame polar-vote pivot (pixels).
 *                           Pass -1, -1 to run the CNN normally.
 * @param fallback_center_x  When the CNN output looks suspicious, or the CNN
 *                           path cannot run cleanly, fall back to this
 *                           full-frame pivot (pixels) instead.
 * @param fallback_center_y  When the CNN output looks suspicious, or the CNN
 *                           path cannot run cleanly, fall back to this
 *                           full-frame pivot (pixels) instead.
 * @retval true  Pipeline completed successfully (result.valid may still be false
 *               if the polar vote failed to find a clear needle).
 * @retval false Preprocessing or NPU inference failed.
 */
bool AppCenterDetector_Run(const uint8_t *frame_bytes, size_t frame_size,
	size_t crop_x_min, size_t crop_y_min, size_t crop_width, size_t crop_height,
	float dial_radius_override_px,
	size_t frame_width_pixels, size_t frame_height_pixels,
	AppCenterDetector_Result_t *result,
	float override_center_x, float override_center_y,
	float fallback_center_x, float fallback_center_y);

#ifdef __cplusplus
}
#endif

#endif /* __APP_CENTER_DETECTOR_H */
