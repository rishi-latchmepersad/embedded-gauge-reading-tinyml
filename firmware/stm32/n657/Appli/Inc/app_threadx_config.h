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
/* Keep the board on a simple one-frame-per-minute duty cycle now that the
 * save path is no longer the bottleneck. */
#define CAMERA_CAPTURE_PERIOD_MS           60000U
/* Capture three sequential frames per scheduled observation. The capture
 * owner waits for the AI worker to release each frame before the next one. */
#define CAMERA_CAPTURE_BURST_COUNT              3U

/* Stage 2 low-power cycle: enter real STM32N6 Stop mode for one minute using
 * the LPTIM1/LSE wake route, only after the camera and consumers are quiescent. */
#ifndef CAMERA_STOP_MODE_PROOF_ENABLE
#define CAMERA_STOP_MODE_PROOF_ENABLE           1U
#endif
#define CAMERA_STOP_MODE_PROOF_DURATION_MS     60000U
#define CAMERA_STOP_MODE_PROOF_TIMEOUT_MS       3000U

/* Storage maintenance timing ---------------------------------------------- */
#define IMAGE_CLEANUP_PERIOD_MS            600000U

/* Camera middleware coordination ------------------------------------------ */
#define CAMERA_MIDDLEWARE_LOCK_TIMEOUT_MS    5000U

#ifdef __cplusplus
}
#endif

#endif /* __APP_THREADX_CONFIG_H */
