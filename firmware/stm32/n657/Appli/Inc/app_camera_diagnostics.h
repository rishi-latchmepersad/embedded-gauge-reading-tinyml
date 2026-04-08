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

#include "stm32n6xx_hal.h"
#include "stm32n6xx_hal_dcmipp.h"

/* Snapshot of the camera/ISP/pipe state used for bring-up diagnostics. */
typedef struct {
	const char *reason;
	DCMIPP_HandleTypeDef *capture_dcmipp;
	uint32_t capture_pipe;
	uintptr_t pipe_memory_address;
	const uint8_t *buffer0;
	const uint8_t *buffer1;
	const uint8_t *result_buffer;
	uint32_t pipe_mode;
	uint32_t pipe_state;
	uint32_t pipe_counter;
	bool snapshot_armed;
	bool stream_started;
	bool use_cmw_pipeline;
	bool cmw_initialized;
	uint32_t frame_event_count;
	uint32_t vsync_event_count;
	uint32_t isp_run_count;
	uint32_t csi_irq_count;
	uint32_t dcmipp_irq_count;
	uint32_t reported_byte_count;
	uint32_t counter_status;
	bool sof_seen;
	bool eof_seen;
	bool failed;
	uint32_t error_code;
	uint32_t line_error_count;
	uint32_t line_error_mask;
	uint32_t active_buffer_index;
	bool cmw_state_ok;
	int32_t cmw_exposure_mode;
	uint8_t cmw_aec_enabled;
	int32_t cmw_exposure;
	int32_t cmw_gain;
	int32_t cmw_test_pattern;
	const char *sensor_name;
	uint32_t sensor_width;
	uint32_t sensor_height;
	uint32_t sensor_gain_min;
	uint32_t sensor_gain_max;
	uint32_t sensor_again_max;
	uint32_t sensor_exposure_min;
	uint32_t sensor_exposure_max;
	bool sensor_regs_ok;
	uint8_t mode_select;
	uint8_t lane_mode_reg_3050;
	uint8_t lane_mode_reg_319d;
	uint8_t lane_mode_reg_341c;
	uint8_t lane_mode_reg_341d;
	uint8_t lane_mode_reg_3a01;
	uint8_t hold_reg;
	uint8_t tpg_reg;
	uint16_t gain_reg;
	uint32_t shutter_reg;
	uint32_t vmax_reg;
	uint32_t csi_linebyte_event_count;
} AppCameraDiagnostics_CaptureState_t;

/* Summarize the active capture frame without changing the camera pipeline. */
void AppCameraDiagnostics_LogCaptureBufferSummary(const uint8_t *buffer_ptr,
		uint32_t captured_bytes, bool use_cmw_pipeline);

/* Print a small byte/word preview of the current frame buffer. */
void AppCameraDiagnostics_LogCaptureBufferPreview(const char *reason,
		const uint8_t *buffer_ptr, uint32_t length_bytes);

/* Print a compact ROI summary for the processed YUV frame. */
void AppCameraDiagnostics_LogProcessedFrameDiagnostics(const char *reason,
		const uint8_t *buffer_ptr, uint32_t length_bytes);

/* Print the CSI line/byte counter state for capture bring-up. */
void AppCameraDiagnostics_LogCsiLineByteCounters(const char *reason,
		uint32_t event_count);

/* Print the active DCMIPP pipe registers for capture bring-up. */
void AppCameraDiagnostics_LogDcmippPipeRegisters(const char *reason,
		DCMIPP_HandleTypeDef *capture_dcmipp);

/* Print the full camera/ISP/pipe state snapshot used during capture bring-up. */
void AppCameraDiagnostics_LogCaptureState(
		const AppCameraDiagnostics_CaptureState_t *snapshot);

#ifdef __cplusplus
}
#endif

#endif /* __APP_CAMERA_DIAGNOSTICS_H */
