/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_camera_capture.c
 * @brief   High-level camera capture and save helpers.
 ******************************************************************************
 */
/* USER CODE END Header */

#include "app_camera_capture.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "main.h"
#include "app_camera_buffers.h"
#include "app_camera_config.h"
#include "app_camera_diagnostics.h"
#include "app_camera_platform.h"
#include "app_filex.h"
#include "app_inference_runtime.h"
#include "app_storage.h"
#include "debug_console.h"
#include "imx335.h"
#include "cmw_camera.h"
/* USER CODE END Includes */

/* USER CODE BEGIN PV */
extern bool camera_capture_use_cmw_pipeline;
extern bool camera_cmw_initialized;
extern bool camera_stream_started;
extern volatile uint32_t camera_capture_isp_run_count;
/* USER CODE END PV */

/**
 * @brief Service ST's IMX335 middleware background process for ISP state updates.
 * @retval true when the background step succeeded or is not used by this driver.
 */
bool AppCameraCapture_RunImx335Background(void) {
	/* The ISP background loop is only needed for the processed image path.
	 * Raw Pipe0 diagnostics bypass the ISP output path, so running AWB/AEC
	 * updates there can trip middleware code that expects the YUV pipeline. */
	if (camera_capture_use_cmw_pipeline && camera_cmw_initialized) {
		camera_capture_isp_run_count++;

		if (CMW_CAMERA_Run() != CMW_ERROR_NONE) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] CMW_CAMERA_Run() failed on ISP wake %lu.\r\n",
					(unsigned long) camera_capture_isp_run_count);
			return false;
		}

		return true;
	}

	return true;
}

/**
 * @brief Capture a single frame, save it to the SD card, and queue inference.
 * @retval true when the frame reaches storage successfully.
 */
bool AppCameraCapture_CaptureAndStoreSingleFrame(void) {
	uint32_t captured_bytes = 0U;
	UINT filex_status = FX_SUCCESS;
	CHAR capture_file_name[CAMERA_CAPTURE_FILE_NAME_LENGTH] = { 0 };
	uint8_t *image_ptr = NULL;
	ULONG image_length = captured_bytes;
	const CHAR *file_extension = camera_capture_use_cmw_pipeline ? "yuv422"
			: "raw16";

	if (!AppStorage_WaitForMediaReady(CAMERA_STORAGE_WAIT_TIMEOUT_MS)) {
		return false;
	}

	if (!CameraPlatform_CaptureSingleFrame(&captured_bytes)) {
		return false;
	}

	image_length = captured_bytes;
	image_ptr = camera_capture_result_buffer;
	if (image_ptr == NULL) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Capture buffer pointer is NULL after frame completion.\r\n");
		return false;
	}

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] Frame ready for save: ptr=%p length=%lu pipeline=%s\r\n",
			(void *) image_ptr, (unsigned long) image_length,
			camera_capture_use_cmw_pipeline ? "processed" : "raw");

	AppCameraDiagnostics_LogCaptureBufferPreview("ready-to-save", image_ptr,
			(uint32_t) image_length);

	if (!AppStorage_BuildCaptureFileName(capture_file_name,
			sizeof(capture_file_name), file_extension)) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to build capture filename.\r\n");
		return false;
	}

	if (camera_capture_use_cmw_pipeline) {
		CameraPlatform_LogCaptureState("processed-capture");
		AppCameraDiagnostics_LogProcessedFrameDiagnostics("processed-capture",
				image_ptr, (uint32_t) image_length);
	}

	DebugConsole_Printf(
			camera_capture_use_cmw_pipeline ?
					"[CAMERA][CAPTURE] Saving YUV422 capture to /captured_images/%s (%lu bytes)...\r\n"
					: "[CAMERA][CAPTURE] Saving raw Pipe0 capture to /captured_images/%s (%lu bytes)...\r\n",
			capture_file_name, (unsigned long) image_length);

	filex_status = AppFileX_WriteCapturedImage(capture_file_name,
			image_ptr, image_length);
	if (filex_status != FX_SUCCESS) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to write image to SD card, status=%lu.\r\n",
				(unsigned long) filex_status);
		return false;
	}

	DebugConsole_Printf(
			camera_capture_use_cmw_pipeline ?
					"[CAMERA][CAPTURE] Stored %lu-byte YUV422 image at /captured_images/%s.\r\n"
					: "[CAMERA][CAPTURE] Stored %lu-byte raw Pipe0 frame at /captured_images/%s.\r\n",
			(unsigned long) image_length, capture_file_name);

	if (camera_capture_use_cmw_pipeline) {
		if (!AppInferenceRuntime_RequestDryInference(
					(const uint8_t *) image_ptr, (ULONG) image_length)) {
			DebugConsole_Printf(
					"[AI] Failed to queue one-shot dry-run inference.\r\n");
		}
	}

	return true;
}
