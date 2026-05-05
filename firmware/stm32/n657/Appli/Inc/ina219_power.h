/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    ina219_power.h
 * @brief   INA219 power/current/voltage monitor driver and logging thread.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __INA219_POWER_H
#define __INA219_POWER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include <stdbool.h>
#include <stdint.h>
#include "stm32n6xx_hal.h"

/* Public defines ------------------------------------------------------------*/
#define INA219_I2C_ADDRESS_7BIT     0x40U
#define INA219_I2C_ADDRESS_HAL      (INA219_I2C_ADDRESS_7BIT << 1U)

/* INA219 Register addresses */
#define INA219_REG_CONFIG           0x00U
#define INA219_REG_SHUNT_VOLTAGE   0x01U
#define INA219_REG_BUS_VOLTAGE      0x02U
#define INA219_REG_POWER            0x03U
#define INA219_REG_CURRENT          0x04U
#define INA219_REG_CALIBRATION      0x05U

/* Configuration register default (32V, 2A, 12-bit, 1 sample, continuous) */
#define INA219_CONFIG_DEFAULT       0x399FU

/* Public typedefs -----------------------------------------------------------*/
/**
 * @brief INA219 measurement data structure
 */
typedef struct {
    float bus_voltage_v;      /**< Bus voltage in Volts */
    float shunt_voltage_mv;   /**< Shunt voltage in millivolts */
    float current_ma;         /**< Current in milliamps */
    float power_mw;           /**< Power in milliwatts */
    uint32_t timestamp_ms;    /**< Timestamp in milliseconds */
    bool valid;               /**< True if reading is valid */
} INA219_Measurement_t;

/* Public function prototypes ------------------------------------------------*/

/**
 * @brief Initialize the INA219 power monitor.
 * @param hi2c Pointer to I2C handle (I2C1)
 * @retval true if initialization successful, false otherwise
 */
bool INA219_Init(I2C_HandleTypeDef *hi2c);

/**
 * @brief Read all measurements from INA219.
 * @param measurement Pointer to measurement structure to fill
 * @retval true if read successful, false otherwise
 */
bool INA219_ReadMeasurement(INA219_Measurement_t *measurement);

/**
 * @brief Get the last valid measurement.
 * @param measurement Pointer to measurement structure to fill
 * @retval true if a valid measurement exists, false otherwise
 */
bool INA219_GetLastMeasurement(INA219_Measurement_t *measurement);

/**
 * @brief Start the power monitoring thread.
 * @retval true if thread started successfully, false otherwise
 */
bool INA219_StartMonitoringThread(void);

/**
 * @brief Trigger a power reading and log it with a label.
 * @param label Label for the reading (e.g., "CNN", "BASELINE")
 * @retval true if reading triggered successfully, false otherwise
 */
bool INA219_LogReading(const char *label);

/**
 * @brief Check if INA219 is detected and responding.
 * @retval true if INA219 is ready, false otherwise
 */
bool INA219_IsReady(void);

#ifdef __cplusplus
}
#endif

#endif /* __INA219_POWER_H */
