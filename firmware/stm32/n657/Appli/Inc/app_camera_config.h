/*
 *******************************************************************************
 * @file    app_camera_config.h
 * @brief   Shared camera tuning constants for the ThreadX app.
 *******************************************************************************
 */

#ifndef __APP_CAMERA_CONFIG_H
#define __APP_CAMERA_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

#include "app_memory_budget.h"
#include "stm32n6xx_hal_dcmipp.h"

/* Camera control / sensor bring-up ---------------------------------------- */
#define BCAMS_IMX_I2C_ADDRESS_7BIT          0x1AU
#define BCAMS_IMX_I2C_ADDRESS_HAL           (BCAMS_IMX_I2C_ADDRESS_7BIT << 1U)
#define BCAMS_IMX_I2C_PROBE_TRIALS          5U
#define BCAMS_IMX_I2C_PROBE_TIMEOUT_MS      50U
#define BCAMS_IMX_POWER_SETTLE_DELAY_MS     10U
#define BCAMS_IMX_RESET_ASSERT_DELAY_MS     5U
#define BCAMS_IMX_RESET_RELEASE_DELAY_MS    10U
#define IMX335_SENSOR_WIDTH_PIXELS          2592U
#define IMX335_SENSOR_HEIGHT_LINES          1944U
#define CAMERA_INIT_STARTUP_DELAY_MS        200U

/* Capture and timing knobs ------------------------------------------------- */
/* Use the processed CMW/ISP path so AE/AWB and demosaicing can converge on a
 * usable live image. Set to 1 only if we need raw Pipe0 diagnostics. */
#define CAMERA_CAPTURE_FORCE_RAW_DIAGNOSTIC 0
#define CAMERA_CAPTURE_TARGET_FRAME_COUNT   4U
/* Brightness gate: reject frames that are still effectively black or blown out
 * and nudge the sensor before trying again. */
#define CAMERA_CAPTURE_BRIGHTNESS_GATE_ROI_SIZE_PIXELS      32U
#define CAMERA_CAPTURE_BRIGHTNESS_DARK_MEAN_THRESHOLD      24U
#define CAMERA_CAPTURE_BRIGHTNESS_DARK_MAX_THRESHOLD       48U
#define CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_MEAN_THRESHOLD   232U
#define CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_MIN_THRESHOLD    208U
#define CAMERA_CAPTURE_BRIGHTNESS_RETRY_LIMIT               5U
#define CAMERA_CAPTURE_BRIGHTNESS_EXPOSURE_STEP_NUMERATOR   1U
#define CAMERA_CAPTURE_BRIGHTNESS_EXPOSURE_STEP_DENOMINATOR 3U
#define CAMERA_CAPTURE_BRIGHTNESS_GAIN_STEP_NUMERATOR       1U
#define CAMERA_CAPTURE_BRIGHTNESS_GAIN_STEP_DENOMINATOR     6U
#define CAMERA_CAPTURE_BRIGHTNESS_SETTLE_DELAY_MS         250U
/* Capture crop is expressed directly in pixels/lines. */
#define CAMERA_CAPTURE_CROP_HSTART_PIXELS   0U
#define CAMERA_CAPTURE_CROP_VSTART_LINES    0U
/* Arm one CSI line/byte counter on VC0 so we can tell whether the receiver
 * is observing line progress even when the captured payload stays all zeros. */
#define CAMERA_CAPTURE_CSI_LB_PROBE_COUNTER      DCMIPP_CSI_COUNTER0
#define CAMERA_CAPTURE_CSI_LB_PROBE_LINE_COUNTER  1U
#define CAMERA_CAPTURE_CSI_LB_PROBE_BYTE_COUNTER  (CAMERA_CAPTURE_WIDTH_PIXELS * CAMERA_CAPTURE_BYTES_PER_PIXEL)
/* Use a centered ROI for the raw diagnostic path so we do not accidentally
 * sample a blank top-left margin from the sensor frame. */
#define CAMERA_CAPTURE_RAW_CROP_HSTART_PIXELS   ((IMX335_SENSOR_WIDTH_PIXELS - CAMERA_CAPTURE_WIDTH_PIXELS) / 2U)
#define CAMERA_CAPTURE_RAW_CROP_VSTART_LINES    ((IMX335_SENSOR_HEIGHT_LINES - CAMERA_CAPTURE_HEIGHT_PIXELS) / 2U)
/* Pipe0 raw-capture frames store one 16-bit padded pixel per sample, so the
 * preview code should read them as a 224x224 source image and upscale only the view. */
#define CAMERA_CAPTURE_RAW_SOURCE_WIDTH_PIXELS    CAMERA_CAPTURE_WIDTH_PIXELS
#define CAMERA_CAPTURE_RAW_SOURCE_HEIGHT_LINES    CAMERA_CAPTURE_HEIGHT_PIXELS
#define CAMERA_CAPTURE_RAW_BMP_PREVIEW_SCALE      2U
#define CAMERA_CAPTURE_RAW_BMP_PREVIEW_WIDTH_PIXELS   (CAMERA_CAPTURE_RAW_SOURCE_WIDTH_PIXELS * CAMERA_CAPTURE_RAW_BMP_PREVIEW_SCALE)
#define CAMERA_CAPTURE_RAW_BMP_PREVIEW_HEIGHT_LINES   (CAMERA_CAPTURE_RAW_SOURCE_HEIGHT_LINES * CAMERA_CAPTURE_RAW_BMP_PREVIEW_SCALE)
#define CAMERA_CAPTURE_RAW_BMP_PREVIEW_PIXEL_COUNT    (CAMERA_CAPTURE_RAW_BMP_PREVIEW_WIDTH_PIXELS * CAMERA_CAPTURE_RAW_BMP_PREVIEW_HEIGHT_LINES)
#define CAMERA_CAPTURE_RAW_BMP_FILE_HEADER_BYTES  14U
#define CAMERA_CAPTURE_RAW_BMP_INFO_HEADER_BYTES   40U
#define CAMERA_CAPTURE_RAW_BMP_PALETTE_BYTES      (256U * 4U)
#define CAMERA_CAPTURE_RAW_BMP_HEADER_BYTES       (CAMERA_CAPTURE_RAW_BMP_FILE_HEADER_BYTES + CAMERA_CAPTURE_RAW_BMP_INFO_HEADER_BYTES + CAMERA_CAPTURE_RAW_BMP_PALETTE_BYTES)
/* IMX335 color-bar bring-up consistently shows four blank top lines before the
 * active test-pattern data starts, so we skip them in the raw Pipe0 view. */
#define CAMERA_CAPTURE_RAW_TOP_SKIP_LINES       4U
/* Give the ISP/AEC loop time to move the sensor away from its black-frame
 * startup state before we give up on the first saved capture. */
#define CAMERA_CAPTURE_TIMEOUT_MS           8000U
#define CAMERA_STORAGE_WAIT_TIMEOUT_MS      70000U
#define CAMERA_CAPTURE_RETRY_DELAY_MS       50U
#define CAMERA_FIRST_FRAME_WARMUP_DELAY_MS  1500U
#define CAMERA_STREAM_WARMUP_DELAY_MS       250U
#define IMX335_CAPTURE_FRAMERATE_FPS        10
#define CAMERA_CAPTURE_FILE_NAME_LENGTH     64U
#define CAMERA_STORAGE_READY_EVENT_FLAG     0x00000001U
/* Seed IMX335 with a brighter starting point so AEC has a usable frame to
 * converge from when the scene is dark. */
#define CAMERA_IMX335_SEED_EXPOSURE_FRACTION_NUMERATOR    2U
#define CAMERA_IMX335_SEED_EXPOSURE_FRACTION_DENOMINATOR  3U
#define CAMERA_IMX335_SEED_GAIN_FRACTION_NUMERATOR        1U
#define CAMERA_IMX335_SEED_GAIN_FRACTION_DENOMINATOR      8U
/* Match ST's IMX335 middleware and upstream Linux driver ID check. */
#define IMX335_CHIP_ID_REG                 0x3912U
#define IMX335_CHIP_ID_VALUE               0x00U
/* IMX335 test-pattern selection.
 * -1 = disabled (live image), 0 = disabled (same as -1 in driver),
 *  1 = solid color (default color regs = 0x000 = black, all-zero pixels, NOT useful),
 * 10 = color bars (non-zero pixel values, use this to verify DMA path). */
/* Return to live optical input so the raw capture reflects the real gauge
 * scene instead of a synthetic test pattern. */
#define IMX335_TEST_PATTERN_MODE           -1

/*
 * Expensive frame diagnostics can monopolize the UART and make the board look
 * stuck even when the camera path is fine. Keep them off by default so the
 * capture/save/inference breadcrumbs stay visible.
 */
#define CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS  0U

/* ST treats PIPE0 as the raw dump pipe and PIPE1 as the processed/YUV pipe.
 * Use PIPE0 only while the raw diagnostic branch is enabled. */
#if CAMERA_CAPTURE_FORCE_RAW_DIAGNOSTIC
#define CAMERA_CAPTURE_PIPE                 DCMIPP_PIPE0
#else
#define CAMERA_CAPTURE_PIPE                 DCMIPP_PIPE1
#endif

/* Prevent accidentally using mode 1 (solid black = all-zero pixels) during
 * raw diagnostic; it is indistinguishable from a broken DMA path. */
#if CAMERA_CAPTURE_FORCE_RAW_DIAGNOSTIC && (IMX335_TEST_PATTERN_MODE == 1)
#error "IMX335_TEST_PATTERN_MODE=1 produces all-zero pixels in raw diag mode. Use mode 10 (color bars)."
#endif

#ifdef __cplusplus
}
#endif

#endif /* __APP_CAMERA_CONFIG_H */
