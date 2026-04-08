/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_camera_capture.h
 * @brief   High-level camera capture and save helpers.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __APP_CAMERA_CAPTURE_H
#define __APP_CAMERA_CAPTURE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Start the processed/raw camera background service for the current frame path. */
bool AppCameraCapture_RunImx335Background(void);

/* Capture a frame, save it to storage, and queue dry-run inference if needed. */
bool AppCameraCapture_CaptureAndStoreSingleFrame(void);

/* Internal capture helpers now owned by the capture module. */
bool AppCameraCapture_CaptureSingleFrame(uint32_t *captured_bytes_ptr);
void AppCameraCapture_LogCaptureState(const char *reason);

#ifdef __cplusplus
}
#endif

#endif /* __APP_CAMERA_CAPTURE_H */
