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
#include <stdio.h>
#include <string.h>

#include "debug_console.h"

/* Keep the live capture buffer in the noncacheable window so DMA and CPU
 * access stay coherent without extra cache maintenance on the write path. */
uint32_t camera_capture_active_buffer_index = 0U;
uint8_t *camera_capture_result_buffer = NULL;
/* Why the pad: the DMA buffer base MUST stay at 0x2419C000 (physical
 * 0x3419C000). The ellipse and keypoint packages' npuRAM3/npuRAM4 pools are
 * written by the NPU from 0x34100000 up to ~0x3419A800, so a buffer based at
 * 0x24160000/0x2416C400 gets its crop rows overwritten between the ellipse
 * fill and the crop fill (both stages read the frame after the ellipse NPU
 * run). 0x3419C000 sits past that usage and the buffer ends exactly at the
 * npuRAM5 pool base (0x34200000), which is never written below its base. */
__attribute__((section(".camera_buffer_pad"), aligned(__SCB_DCACHE_LINE_SIZE)))
uint8_t camera_capture_noncacheable_pad[0x3C000U];
uint8_t camera_capture_buffers[CAMERA_CAPTURE_BUFFER_COUNT][CAMERA_CAPTURE_BUFFER_SIZE_BYTES]
		__attribute__((section(".noncacheable"), aligned(__SCB_DCACHE_LINE_SIZE)));

/* Keep a private frame snapshot so preprocessing can run without racing the
 * capture DMA buffer that the camera thread continues to own. The AI worker
 * owns this copy, and the baseline worker reuses it only after the AI path
 * has finished with the frame. */
uint8_t camera_inference_frame_snapshot[CAMERA_CAPTURE_BUFFER_SIZE_BYTES]
		__attribute__((section(".tip_focus_activations_snapshot"), aligned(__SCB_DCACHE_LINE_SIZE)));

/* Keep the CPU write-probe scratch separate from the live DMA frame. */
uint32_t camera_capture_write_probe_words[2U];

/* Histogram bins for the RAW10 level summary. Keeping this out of the thread
 * file makes the camera storage block easier to scale independently. */
uint32_t camera_capture_raw_level_histogram[1024U];

/* The console uses this flag to drop unrelated output while the snapshot
 * copy owns the CPU for a bounded transfer. A word-sized volatile is used so
 * the flag is naturally observable by the other Cortex-M thread context. */
volatile uint32_t camera_snapshot_copy_active = 0U;
volatile uint32_t camera_snapshot_copy_progress_words = 0U;

/**
 * @brief Reports whether the camera snapshot transfer is in progress.
 * @param None.
 * @return Non-zero while the private snapshot copy is active.
 * @sideeffect Reads a RAM-only diagnostic flag; no I/O is performed.
 */
bool AppCameraBuffers_IsSnapshotCopyActive(void) {
	return camera_snapshot_copy_active != 0U;
}

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

#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] Capture buffer is non-cacheable: ptr=%p bytes=%lu; cache invalidate skipped.\r\n",
			(void *) camera_capture_result_buffer, (unsigned long) invalidate_bytes);
#endif
	/* why: camera_capture_buffers lives in RAM_NC (0x24160000), so a D-cache
	 * invalidate on this address is unnecessary and can fault on this MPU
	 * configuration. The barrier still orders the completed DMA writes before
	 * the CPU copies the frame into the private snapshot. */
	__DSB();
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

/**
 * @brief Log a compact signature for a frame without emitting frame bytes.
 * @param label Short point-in-time label for the signature.
 * @param buffer_ptr Frame storage to inspect.
 * @param length_bytes Number of valid frame bytes.
 * @sideeffect Scans the frame and writes one diagnostic line to the console.
 */
void AppCameraBuffers_LogFrameSignature(const char *label,
		const uint8_t *buffer_ptr, uint32_t length_bytes) {
	uint32_t hash = 2166136261UL;
	uint32_t sum = 0U;
	uint8_t min_value = 255U;
	uint8_t max_value = 0U;
	uint8_t first_value = 0U;
	uint8_t middle_value = 0U;
	uint8_t last_value = 0U;

	if ((buffer_ptr == NULL) || (length_bytes == 0U)) {
		DebugConsole_Printf(
			"[CAMERA][FRAME] signature skipped label=%s ptr=%p len=%lu\r\n",
			(label != NULL) ? label : "none", (const void *)buffer_ptr,
			(unsigned long)length_bytes);
		return;
	}

	first_value = buffer_ptr[0U];
	middle_value = buffer_ptr[length_bytes / 2U];
	last_value = buffer_ptr[length_bytes - 1U];
	for (uint32_t index = 0U; index < length_bytes; ++index) {
		const uint8_t value = buffer_ptr[index];
		hash ^= (uint32_t)value;
		hash *= 16777619UL;
		sum += (uint32_t)value;
		if (value < min_value) {
			min_value = value;
		}
		if (value > max_value) {
			max_value = value;
		}
	}

	DebugConsole_Printf(
		"[CAMERA][FRAME] label=%s ptr=%p len=%lu first=%u mid=%u last=%u "
		"min=%u max=%u mean_milli=%lu hash=0x%08lX\r\n",
		(label != NULL) ? label : "none", (const void *)buffer_ptr,
		(unsigned long)length_bytes, (unsigned int)first_value,
		(unsigned int)middle_value, (unsigned int)last_value,
		(unsigned int)min_value, (unsigned int)max_value,
		(unsigned long)(((uint64_t)sum * 1000ULL) / (uint64_t)length_bytes),
		(unsigned long)hash);
}

bool AppCameraBuffers_CopyCaptureToSnapshot(const uint8_t *source_ptr,
		uint32_t length_bytes) {
	/* The AI worker must never read the live DMA buffer. The sensor and receiver
	 * are stopped before this function is called, but the private copy is still
	 * required because the board's camera path previously produced tensors that
	 * did not match the corresponding gray8 file on the SD card. */
	if ((source_ptr == NULL)
			|| (length_bytes != CAMERA_CAPTURE_BUFFER_SIZE_BYTES)) {
		DebugConsole_Printf(
			"[CAMERA][FRAME] snapshot-copy rejected src=%p len=%lu\r\n",
			(const void *)source_ptr, (unsigned long)length_bytes);
		return false;
	}
	camera_snapshot_copy_progress_words = 0U;
	camera_snapshot_copy_active = 1U;
	__DMB();
	/* Do not format strings, print, or take the UART lock in this transfer.
	 * The old progress-marker path was the source of the binary UART spam and
	 * could block the camera thread while the copy was in progress. The
	 * destination is now CPU-cacheable, so copy one image row at a time: this
	 * avoids a single long AXI burst while retaining libc's fast cache path. */
	uint8_t *destination_ptr = camera_inference_frame_snapshot;
	for (uint32_t row = 0U; row < CAMERA_CAPTURE_HEIGHT_PIXELS; ++row) {
		(void)memcpy(destination_ptr +
				(row * CAMERA_CAPTURE_WIDTH_PIXELS),
				source_ptr + (row * CAMERA_CAPTURE_WIDTH_PIXELS),
				CAMERA_CAPTURE_WIDTH_PIXELS * CAMERA_CAPTURE_BYTES_PER_PIXEL);
		/* Keep fault-dump progress in RAM only; never emit frame data on UART. */
		camera_snapshot_copy_progress_words =
				((row + 1U) * CAMERA_CAPTURE_WIDTH_PIXELS) / sizeof(uint32_t);
	}
	__DSB();
	camera_snapshot_copy_active = 0U;
	__DMB();
	return true;
}
