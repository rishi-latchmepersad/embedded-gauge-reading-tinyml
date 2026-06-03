/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_center_detector.h
 * @brief   Center-detection CNN + polar-vote pipeline for gauge reading.
 *
 * Pipeline:
 *   1. Crop the full-frame YUV422 to the stable training crop family.
 *   2. Resize / convert to 224x224 int8 RGB.
 *   3. Run center detector NPU inference → predicted (cx, cy) in crop space.
 *   4. Map (cx, cy) back to full-frame pixel coordinates.
 *   5. Run polar vote around the full-frame center → needle angle.
 *   6. Convert angle → temperature (°C).
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
	float temperature_c; /**< Converted temperature (°C). */
	float confidence;    /**< Polar-vote confidence [0, 1]. */
} AppCenterDetector_Result_t;

/**
 * @brief Initialise the center-detector model (load NPU weights, init runtime).
 *
 * Call once after App_AI_Model_Init().  The center detector reuses the xSPI2
 * flash slot formerly occupied by the scalar model (0x70200000).
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
 * @param crop_height      Height of the selected crop.
 * @param dial_radius_override_px  When > 0, overrides the internal polar-vote
 *                                 dial radius.  Pass 0 to use the legacy
 *                                 min(crop_width, crop_height) / 2 heuristic.
 * @param frame_width_pixels  Full-frame width in pixels (e.g. 2592).
 * @param frame_height_pixels Full-frame height in pixels (e.g. 1944).
 * @param[out] result      Filled with centre, angle, temperature, confidence.
 * @param override_center_x  When >= 0, skip the CNN and use this as the
 *                           full-frame polar-vote pivot (pixels).
 * @param override_center_y  When >= 0, skip the CNN and use this as the
 *                           full-frame polar-vote pivot (pixels).
 *                           Pass -1, -1 to run the CNN normally.
 * @retval true  Pipeline completed successfully (result.valid may still be false
 *               if the polar vote failed to find a clear needle).
 * @retval false Preprocessing or NPU inference failed.
 */
bool AppCenterDetector_Run(const uint8_t *frame_bytes, size_t frame_size,
	size_t crop_x_min, size_t crop_y_min, size_t crop_width, size_t crop_height,
	float dial_radius_override_px,
	size_t frame_width_pixels, size_t frame_height_pixels,
	AppCenterDetector_Result_t *result,
	float override_center_x, float override_center_y);

#ifdef __cplusplus
}
#endif

#endif /* __APP_CENTER_DETECTOR_H */
