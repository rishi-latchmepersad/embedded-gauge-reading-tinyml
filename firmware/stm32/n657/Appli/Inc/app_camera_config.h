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
/* Brightness gate: reject frames that are still too dim for the gauge face or
 * blown out, then nudge the sensor before trying again.
 *
 * Step size history: 1/12 caused a hard oscillation between two exposure
 * values that had no stable midpoint — the step was larger than the gap
 * between the dark threshold and the bright threshold at that scene.
 * Reduced to 1/20 so the loop can converge instead of bouncing.
 *
 * BRIGHT_MIN_THRESHOLD lowered from 80 to 40: at borderline exposures the
 * dial face center is bright (mean ~200) but there are always some dim pixels
 * in the crop from the needle and dial markings.  A min of 80 rejected frames
 * that were genuinely usable for inference.
 *
 * BRIGHT_MEAN_THRESHOLD raised from 200 to 210: gives a small guard band
 * above the dark limit so a borderline mean=200 frame is accepted as OK
 * rather than being right on the too-bright edge.
 *
 * Brightness gate now measures the full training crop (155x123 = ~19k pixels)
 * rather than a small centre ROI.  A specular reflection on the gauge glass
 * at frame centre was making the old 32x32 ROI read as "bright enough" while
 * the rest of the dial face was still underexposed, causing systematic
 * under-reading.  Crop-mean thresholds are calibrated from captured frames:
 * good frames (13:xx session, model reading ~31C) had crop mean 97-156;
 * bad frames (18:xx, model reading 14-20C) had crop mean 43-87.
 * DARK threshold=100 rejects the dim frames; BRIGHT threshold=200 with
 * min=20 catches blown-out overexposure (min=20 because the needle and
 * markings will always produce some dark pixels in the crop). */
#define CAMERA_CAPTURE_BRIGHTNESS_DARK_MEAN_THRESHOLD     100U
#define CAMERA_CAPTURE_BRIGHTNESS_DARK_MAX_THRESHOLD      240U
#define CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_MEAN_THRESHOLD   200U
#define CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_MIN_THRESHOLD     20U
#define CAMERA_CAPTURE_BRIGHTNESS_RETRY_LIMIT              16U
/* Multiplicative step: exposure multiplied/divided by 2 per nudge (1 stop).
 * Linear steps across the full sensor range (26–33333 µs) are too coarse at
 * low exposures — a 1/20-range step (1662 µs) straddled the entire acceptable
 * window and caused infinite bright↔dark oscillation. */
/* Active sensor nudge: use a 25% fractional step instead of a 2x jump. */
#define CAMERA_CAPTURE_BRIGHTNESS_EXPOSURE_STEP_FRACTION_SHIFT  2U
#define CAMERA_CAPTURE_BRIGHTNESS_GAIN_STEP_FRACTION_SHIFT      2U
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
/* Seed IMX335 at ~2/3 of the exposure range (~22177 us at max=33266 us).
 * The 1/5 seed (6659 us) required 10+ nudge steps to reach the crop-mean
 * threshold of 100 for dimmer evening/indoor scenes.  2/3 overshoots for
 * bright scenes but the brightness gate will descend quickly from there. */
#define CAMERA_IMX335_SEED_EXPOSURE_FRACTION_NUMERATOR    2U
#define CAMERA_IMX335_SEED_EXPOSURE_FRACTION_DENOMINATOR  3U
#define CAMERA_IMX335_SEED_GAIN_FRACTION_NUMERATOR        1U
#define CAMERA_IMX335_SEED_GAIN_FRACTION_DENOMINATOR      2U
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
