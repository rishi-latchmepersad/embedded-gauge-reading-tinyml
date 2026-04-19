/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_camera_buffers.h
 * @brief   Shared camera buffer storage for capture and inference.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __APP_CAMERA_BUFFERS_H
#define __APP_CAMERA_BUFFERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "app_memory_budget.h"

/* Shared camera buffer state ------------------------------------------------ */
extern uint32_t camera_capture_active_buffer_index;
extern uint8_t *camera_capture_result_buffer;
extern uint8_t camera_capture_buffers[CAMERA_CAPTURE_BUFFER_COUNT][CAMERA_CAPTURE_BUFFER_SIZE_BYTES];
extern uint8_t camera_ai_frame_snapshot[CAMERA_CAPTURE_BUFFER_SIZE_BYTES];
extern uint8_t camera_baseline_frame_snapshot[CAMERA_CAPTURE_BUFFER_SIZE_BYTES];
extern uint32_t camera_capture_write_probe_words[2U];
extern uint32_t camera_capture_raw_level_histogram[1024U];

/* Shared camera buffer helpers --------------------------------------------- */
void AppCameraBuffers_PrepareForDma(void);
void AppCameraBuffers_InvalidateCaptureRegion(uint32_t captured_bytes);
uint32_t AppCameraBuffers_CountNonZeroBytes(const uint8_t *buffer_ptr,
		uint32_t length_bytes);

#ifdef __cplusplus
}
#endif

#endif /* __APP_CAMERA_BUFFERS_H */
