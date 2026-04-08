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

#ifdef __cplusplus
extern "C" {
#endif

/* Start the processed/raw camera background service for the current frame path. */
bool AppCameraCapture_RunImx335Background(void);

/* Capture a frame, save it to storage, and queue dry-run inference if needed. */
bool AppCameraCapture_CaptureAndStoreSingleFrame(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_CAMERA_CAPTURE_H */
