/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    i2c_scanner.h
 * @brief   Simple I2C bus scanner for testing INA219 and other devices.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __I2C_SCANNER_H
#define __I2C_SCANNER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include <stdbool.h>
#include <stdint.h>
#include "stm32n6xx_hal.h"

/* Function prototypes -------------------------------------------------------*/

/**
 * @brief Run I2C scanner on both I2C1 and I2C2 buses.
 * @retval Total number of devices found across both buses
 */
uint8_t I2CScanner_Run(void);

/**
 * @brief Quick probe for INA219 on expected address.
 * @param hi2c Pointer to I2C handle to test
 * @retval true if INA219 found, false otherwise
 */
bool I2CScanner_ProbeINA219(I2C_HandleTypeDef *hi2c);

/**
 * @brief Find which I2C bus has the INA219 connected.
 * @retval Pointer to I2C handle with INA219, or NULL if not found
 */
I2C_HandleTypeDef* I2CScanner_FindINA219(void);

#ifdef __cplusplus
}
#endif

#endif /* __I2C_SCANNER_H */
