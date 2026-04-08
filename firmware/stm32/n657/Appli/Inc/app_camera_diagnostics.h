/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_camera_diagnostics.h
 * @brief   Camera capture diagnostic helpers.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __APP_CAMERA_DIAGNOSTICS_H
#define __APP_CAMERA_DIAGNOSTICS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

/* Summarize the active capture frame without changing the camera pipeline. */
void AppCameraDiagnostics_LogCaptureBufferSummary(const uint8_t *buffer_ptr,
		uint32_t captured_bytes, bool use_cmw_pipeline);

#ifdef __cplusplus
}
#endif

#endif /* __APP_CAMERA_DIAGNOSTICS_H */
