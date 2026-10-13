/*
 *******************************************************************************
 * @file    app_threadx_config.h
 * @brief   Shared ThreadX scheduling and heartbeat constants.
 *******************************************************************************
 */

#ifndef __APP_THREADX_CONFIG_H
#define __APP_THREADX_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32n6xx_hal_gpio.h"

/* Thread priorities -------------------------------------------------------- */
#define CAMERA_INIT_THREAD_PRIORITY          9U
#define CAMERA_ISP_THREAD_PRIORITY          11U
#define CAMERA_HEARTBEAT_THREAD_PRIORITY    10U
#define CAMERA_AI_THREAD_PRIORITY           11U
/* Snapshot the classical frame early, but keep its CPU-heavy CV pass below
 * the NPU worker so a baseline calculation cannot delay model progress. */
#define BASELINE_RUNTIME_THREAD_PRIORITY    15U
#define IMAGE_CLEANUP_THREAD_PRIORITY       16U

/* Heartbeat timing --------------------------------------------------------- */
#define CAMERA_HEARTBEAT_PERIOD_MS          120000U
#define CAMERA_HEARTBEAT_PULSE_MS           1000U
#ifndef CAMERA_HEARTBEAT_ENABLE_UART_PULSES
#define CAMERA_HEARTBEAT_ENABLE_UART_PULSES 0U
#endif
#define CAMERA_HEARTBEAT_LED_GPIO_PORT      GPIOG
#define CAMERA_HEARTBEAT_LED_PIN            GPIO_PIN_0

/* Camera capture cadence --------------------------------------------------- */
/* Normal operation uses three frames per wake, then five minutes in Standby.
 * The disabled diagnostic override takes one frame every 30 seconds while
 * awake; set it to 1 only for a temporary capture investigation. */
#ifndef CAMERA_CAPTURE_DIAGNOSTIC_30S_ENABLE
#define CAMERA_CAPTURE_DIAGNOSTIC_30S_ENABLE    0U
#endif
#if CAMERA_CAPTURE_DIAGNOSTIC_30S_ENABLE
#define CAMERA_CAPTURE_PERIOD_MS           30000U
#define CAMERA_CAPTURE_BURST_COUNT              1U
#else
#define CAMERA_CAPTURE_PERIOD_MS           60000U
/* Serial frames release AI ownership before the next capture. Later frames
 * reuse the preceding ellipse crop and support two-of-three value consensus. */
#define CAMERA_CAPTURE_BURST_COUNT              3U
#endif

/* Stage 2 low-power cycle: enter real STM32N6 Stop mode for one minute using
 * the LPTIM1/LSE wake route, only after the camera and consumers are quiescent. */
#ifndef CAMERA_STOP_MODE_PROOF_ENABLE
#define CAMERA_STOP_MODE_PROOF_ENABLE           1U
#endif
#define CAMERA_STOP_MODE_PROOF_DURATION_MS     60000U
#define CAMERA_STOP_MODE_PROOF_TIMEOUT_MS       3000U

/* Temporary Standby experiment. Standby removes VCORE and loses most RAM;
 * keep this separate from the validated Stop switch so the test can be
 * disabled without deleting the Stop-mode proof path. The current LRUN image
 * is larger than the default retained SRAM, so the wake-loader problem remains
 * a separate follow-up from validating the automatic wake event. */
#ifndef CAMERA_STANDBY_MODE_PROOF_ENABLE
#define CAMERA_STANDBY_MODE_PROOF_ENABLE        1U
#endif

/* Standby uses the RTC's CK_SPRE wake-timer source, nominally 1 Hz from the
 * RTC prescalers, so the 16-bit counter can cover intervals far longer than
 * the original 30-second wake-path test. */
#ifndef CAMERA_STANDBY_WAKE_INTERVAL_MS
#define CAMERA_STANDBY_WAKE_INTERVAL_MS        300000U
#endif

/* Storage maintenance timing ---------------------------------------------- */
#define IMAGE_CLEANUP_PERIOD_MS            600000U

/* Camera middleware coordination ------------------------------------------ */
#define CAMERA_MIDDLEWARE_LOCK_TIMEOUT_MS    5000U

#ifdef __cplusplus
}
#endif

#endif /* __APP_THREADX_CONFIG_H */
