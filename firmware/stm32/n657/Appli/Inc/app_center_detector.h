/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    app_center_detector.h
  * @brief   Center-detector + polar-vote pipeline for the gauge-reading AI.
  *
  * This module replaces the scalar CNN with a two-step pipeline:
  *   1. MobileNetV2 center detector (NPU) → normalized (cx, cy) needle pivot
  *   2. CPU polar-vote → needle angle → temperature
  *
  * Pipeline: OBB → center detector → polar vote → temperature
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

/* ── xSPI2 flash layout for the center detector model ─────────────────── */

/** @brief Path to the center-detector model image on xSPI2 flash. */
#define APP_AI_CENTER_DETECTOR_XSPI2_MODEL_IMAGE_PATH \
    "atonbuf.center_detector.xSPI2.raw"

/** @brief xSPI2 base address for the center-detector model weights.
 *  Reuses the former scalar-CNN flash region (0x70200000).
 */
#define APP_AI_XSPI2_CENTER_DETECTOR_BASE_ADDR      0x70200000UL

/** @brief Chip offset from xSPI2 chip base. */
#define APP_AI_XSPI2_CENTER_DETECTOR_CHIP_OFFSET    \
    (APP_AI_XSPI2_CENTER_DETECTOR_BASE_ADDR - APP_AI_XSPI2_CHIP_BASE_ADDR)

/* ── Pipeline constants ───────────────────────────────────────────────── */

/** @brief Dial radius in 224x224 crop pixels (0.56 × 224). */
#define APP_CENTER_DETECTOR_DIAL_RADIUS_PX          125.0f

/** @brief Crop size expected by the center detector. */
#define APP_CENTER_DETECTOR_CROP_WIDTH_PIXELS       224U
#define APP_CENTER_DETECTOR_CROP_HEIGHT_PIXELS      224U

/** @brief Number of int8 bytes per input pixel (RGB). */
#define APP_CENTER_DETECTOR_BYTES_PER_PIXEL         3U

/** @brief Input buffer size in int8 elements. */
#define APP_CENTER_DETECTOR_INPUT_SIZE              \
    (APP_CENTER_DETECTOR_CROP_WIDTH_PIXELS *         \
     APP_CENTER_DETECTOR_CROP_HEIGHT_PIXELS *        \
     APP_CENTER_DETECTOR_BYTES_PER_PIXEL)

/** @brief Output buffer size in int8 elements (2 values: cx, cy). */
#define APP_CENTER_DETECTOR_OUTPUT_SIZE             2U

/* ── Acceptance gate ──────────────────────────────────────────────────── */

/** @brief Maximum centre-to-inner-dial distance (px in full frame) before
 *         we reject the detection and fall back to the fixed crop centre. */
#define APP_CENTER_DETECTOR_MAX_CENTER_DIST_PX      80.0f

/* ── Public API ───────────────────────────────────────────────────────── */

/**
 * @brief Run the full centre-detector + polar-vote pipeline.
 *
 * Call this after the OBB stage has produced a crop box.  The pipeline:
 *   1. Crops the YUV422 frame to the OBB region, resizes to 224×224,
 *      converts to int8 RGB.
 *   2. Runs the centre-detector model on the NPU.
 *   3. Decodes the int8 output → normalized (cx, cy).
 *   4. Converts crop-space (cx, cy) to full-frame pixel coordinates.
 *   5. Runs the CPU polar-vote detector at that centre.
 *   6. Converts the winning needle angle to temperature.
 *
 * @param frame_bytes      Full YUV422 frame buffer.
 * @param frame_size       Size of frame_bytes in bytes.
 * @param crop_x_min       OBB crop region left (full-frame pixels).
 * @param crop_y_min       OBB crop region top (full-frame pixels).
 * @param crop_width       OBB crop region width (full-frame pixels).
 * @param crop_height      OBB crop region height (full-frame pixels).
 * @param temperature_c_out  Receives the estimated temperature in °C.
 * @param confidence_out     Receives the polar-vote confidence.
 * @param angle_deg_out      Receives the needle angle in degrees.
 *
 * @retval true  A valid temperature was produced.
 * @retval false The pipeline failed (likely NPU init or polar vote fail).
 */
bool AppCenterDetector_RunPipeline(
    const uint8_t *frame_bytes, size_t frame_size,
    size_t crop_x_min, size_t crop_y_min,
    size_t crop_width, size_t crop_height,
    float *temperature_c_out,
    float *confidence_out,
    float *angle_deg_out);

/**
 * @brief Check whether the centre-detector NPU model is available.
 *
 * @retval true  The model weights are programmed and init succeeded.
 * @retval false Hardware or model init failed.
 */
bool AppCenterDetector_IsAvailable(void);

/**
 * @brief De-initialize and release NPU resources for the centre detector.
 */
void AppCenterDetector_Deinit(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_CENTER_DETECTOR_H */
