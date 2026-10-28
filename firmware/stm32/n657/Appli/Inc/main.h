/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined ( __ICCARM__ )
#  define CMSE_NS_CALL  __cmse_nonsecure_call
#  define CMSE_NS_ENTRY __cmse_nonsecure_entry
#else
#  define CMSE_NS_CALL  __attribute((cmse_nonsecure_call))
#  define CMSE_NS_ENTRY __attribute((cmse_nonsecure_entry))
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32n6xx_hal.h"
#include <stdbool.h>

#include "stm32n6xx_nucleo.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "ds3231_clock.h"

/* USER CODE END Includes */

/* why: Keep one visible build identifier for the UART and SD boot records so
 * the board log can be compared with the firmware image that was built. */
#define APP_FIRMWARE_BUILD_ID "2026-10-28-standby-irq-order-v1"

/* Exported types ------------------------------------------------------------*/
/* Function pointer declaration in non-secure*/
#if defined ( __ICCARM__ )
typedef void (CMSE_NS_CALL *funcptr)(void);
#else
typedef void CMSE_NS_CALL (*funcptr)(void);
#endif

/* typedef for non-secure callback functions */
typedef funcptr funcptr_NS;

/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/** Return the reset flags captured before the boot code clears them. */
const char *App_GetBootResetLogLine(void);
/** Return the previous reset's retained Cortex fault record, when present. */
const char *App_GetPreviousFaultLogLine(void);
/** Describe the progress markers retained across a low-power transition. */
typedef enum {
	LP_PHASE_IDLE = 0U,
	LP_PHASE_LOADER_PREPARE,
	LP_PHASE_LOADER_READY,
	LP_PHASE_RTC_ARMED,
	LP_PHASE_BEFORE_WFI,
	LP_PHASE_RTC_IRQ_SEEN,
	LP_PHASE_STANDBY_RETURNED,
	LP_PHASE_WAKE_LOADER_BRANCH,
	LP_PHASE_ERROR_LOADER_PREPARE,
	LP_PHASE_ERROR_RTC_LSI_START,
	LP_PHASE_ERROR_RTC_CLOCK_SELECT,
	LP_PHASE_ERROR_RTC_TICK_RANGE,
	LP_PHASE_ERROR_RTC_WUTWF,
	LP_PHASE_ERROR_WAKE_LOADER_VECTOR,
	LP_PHASE_STOP_INFERENCE_DRAIN_BEGIN,
	LP_PHASE_STOP_INFERENCE_DRAIN_DONE,
	LP_PHASE_STOP_DEBUG_DRAIN_BEGIN,
	LP_PHASE_STOP_DEBUG_DRAIN_DONE,
	LP_PHASE_STOP_CAPTURE_FLUSH_BEGIN,
	LP_PHASE_STOP_CAPTURE_FLUSH_DONE,
	LP_PHASE_STOP_FILEX_FLUSH_BEGIN,
	LP_PHASE_STOP_FILEX_FLUSH_DONE,
	LP_PHASE_STOP_POWER_WINDOW_CLOSE_BEGIN,
	LP_PHASE_STOP_POWER_WINDOW_CLOSE_DONE,
	LP_PHASE_STOP_AWAKE_LOG_DRAIN_BEGIN,
	LP_PHASE_STOP_AWAKE_LOG_DRAIN_DONE,
	LP_PHASE_STOP_AWAKE_FILEX_FLUSH_BEGIN,
	LP_PHASE_STOP_AWAKE_FILEX_FLUSH_DONE,
	LP_PHASE_STOP_DEBUGLOG_FLUSH_BEGIN,
	LP_PHASE_STOP_DEBUGLOG_FLUSH_DONE,
	LP_PHASE_STOP_BARRIERS_COMPLETE
} AppLowPowerPhase;
/** Retain one low-power phase and two diagnostic values without using the SD card. */
void AppDiagnostics_RecordLowPowerPhase(AppLowPowerPhase phase,
		uint32_t detail0, uint32_t detail1);
/** Return a formatted low-power journal record captured before reset flags clear. */
const char *App_GetPreviousLowPowerLogLine(void);
/** Clear the low-power journal only after its record has been durably flushed. */
void AppDiagnostics_AcknowledgePersistedLowPower(void);
/** @brief Clear the retained fault marker only after its SD record is flushed. */
void AppDiagnostics_AcknowledgePersistedFault(void);
/** Save a minimal fault snapshot into the RTC backup domain for the next boot. */
void AppDiagnostics_RecordFault(uint32_t stacked_pc, uint32_t stacked_lr,
		uint32_t cfsr, uint32_t hfsr, uint32_t bfar, uint32_t mmfar,
		uint32_t exc_lr, uint32_t exception_number);

/**
 * @brief Restore the application PLL and bus clock tree after STM32N6 Stop.
 */
void App_SystemClock_Config(void);

/**
 * @brief Restore the DCMIPP and CSI kernel clocks after STM32N6 Stop.
 */
void App_CameraKernelClock_Config(void);

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define CAM_NRST_Pin GPIO_PIN_5
#define CAM_NRST_GPIO_Port GPIOO
#define I2C1_SDA_Pin GPIO_PIN_1
#define I2C1_SDA_GPIO_Port GPIOC
#define I2C1_SCL_Pin GPIO_PIN_9
#define I2C1_SCL_GPIO_Port GPIOH
#define I2C2_SDA_Pin GPIO_PIN_11
#define I2C2_SDA_GPIO_Port GPIOB
#define SPI5_CS_Pin GPIO_PIN_3
#define SPI5_CS_GPIO_Port GPIOA
#define CAM1_Pin GPIO_PIN_0
#define CAM1_GPIO_Port GPIOA
#define I2C2_SCL_Pin GPIO_PIN_10
#define I2C2_SCL_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
