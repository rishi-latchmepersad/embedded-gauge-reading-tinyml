/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_camera_buffers.c
 * @brief   Camera capture buffers and shared frame snapshot storage.
 ******************************************************************************
 */
/* USER CODE END Header */

#include "app_camera_buffers.h"
#include "main.h"

/* Keep the live capture buffer in the noncacheable window so DMA and CPU
 * access stay coherent without extra cache maintenance on the write path. */
uint32_t camera_capture_active_buffer_index = 0U;
uint8_t *camera_capture_result_buffer = NULL;
uint8_t camera_capture_buffers[CAMERA_CAPTURE_BUFFER_COUNT][CAMERA_CAPTURE_BUFFER_SIZE_BYTES]
		__attribute__((section(".noncacheable"), aligned(__SCB_DCACHE_LINE_SIZE)));

/* Keep a private AI snapshot so preprocessing can run without racing the
 * capture DMA buffer that the camera thread continues to own. */
uint8_t camera_ai_frame_snapshot[CAMERA_CAPTURE_BUFFER_SIZE_BYTES]
		__attribute__((aligned(__SCB_DCACHE_LINE_SIZE)));

/* Keep the CPU write-probe scratch separate from the live DMA frame. */
uint32_t camera_capture_write_probe_words[2U];

/* Histogram bins for the RAW10 level summary. Keeping this out of the thread
 * file makes the camera storage block easier to scale independently. */
uint32_t camera_capture_raw_level_histogram[1024U];
