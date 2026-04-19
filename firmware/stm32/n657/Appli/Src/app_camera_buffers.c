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
#include <string.h>

#include "debug_console.h"

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

/* Keep a second private snapshot for the classical baseline worker so the two
 * inference paths never race over the same copied frame. */
uint8_t camera_baseline_frame_snapshot[CAMERA_CAPTURE_BUFFER_SIZE_BYTES]
		__attribute__((aligned(__SCB_DCACHE_LINE_SIZE)));

/* Keep the CPU write-probe scratch separate from the live DMA frame. */
uint32_t camera_capture_write_probe_words[2U];

/* Histogram bins for the RAW10 level summary. Keeping this out of the thread
 * file makes the camera storage block easier to scale independently. */
uint32_t camera_capture_raw_level_histogram[1024U];

void AppCameraBuffers_PrepareForDma(void) {
	for (uint32_t buffer_index = 0U; buffer_index < CAMERA_CAPTURE_BUFFER_COUNT;
			buffer_index++) {
		volatile uint32_t *probe_words =
				(volatile uint32_t*) camera_capture_write_probe_words;
		probe_words[0] = 0xDEADBEEFU;
		probe_words[1] = 0xCAFEBABEU;
		__DSB();
		(void) memset(camera_capture_buffers[buffer_index], 0xAA,
				CAMERA_CAPTURE_BUFFER_SIZE_BYTES);
		__DSB();
	}
}

void AppCameraBuffers_InvalidateCaptureRegion(uint32_t captured_bytes) {
	uint32_t invalidate_bytes = captured_bytes;

	if ((invalidate_bytes == 0U)
			|| (invalidate_bytes > CAMERA_CAPTURE_BUFFER_SIZE_BYTES)) {
		invalidate_bytes = CAMERA_CAPTURE_BUFFER_SIZE_BYTES;
	}

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] Invalidate capture buffer cache: ptr=%p bytes=%lu\r\n",
			(void *) camera_capture_result_buffer, (unsigned long) invalidate_bytes);

	SCB_InvalidateDCache_by_Addr((void*) camera_capture_result_buffer,
			(int32_t) invalidate_bytes);
}

uint32_t AppCameraBuffers_CountNonZeroBytes(const uint8_t *buffer_ptr,
		uint32_t length_bytes) {
	uint32_t nonzero_count = 0U;

	if (buffer_ptr == NULL) {
		return 0U;
	}

	for (uint32_t byte_index = 0U; byte_index < length_bytes; byte_index++) {
		if (buffer_ptr[byte_index] != 0U) {
			nonzero_count++;
		}
	}

	return nonzero_count;
}
