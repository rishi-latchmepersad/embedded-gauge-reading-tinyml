/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    ina219_power.c
 * @brief   INA219 power/current/voltage monitor driver and logging thread.
 ******************************************************************************
 */
/* USER CODE END Header */

#include "ina219_power.h"

/* Private includes ----------------------------------------------------------*/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "debug_console.h"
#include "inference_metrics.h"
#include "main.h"
#include "tx_api.h"

/* Private defines -----------------------------------------------------------*/
#define INA219_TIMEOUT_MS           100U
#define INA219_PROBE_TRIALS         3U

/* Calibration values for 0.1 ohm shunt, 3.2A max current */
#define INA219_CALIBRATION_VALUE    0x1000U  /* Adjust based on your shunt */
#define INA219_CURRENT_LSB_MA       0.1f     /* 0.1 mA per bit */
#define INA219_POWER_LSB_W          0.002f   /* 2 mW per bit, reported here as 0.002 W per bit */

/* Thread configuration */
#define INA219_THREAD_STACK_SIZE    1024U
#define INA219_THREAD_PRIORITY      10U      /* Must outrank the pipeline workers */
#define INA219_SAMPLE_PERIOD_MS     250U     /* Sample every 250 ms for power stats */

/* Private variables ---------------------------------------------------------*/
static I2C_HandleTypeDef *g_hi2c = NULL;
static bool g_initialized = false;
static INA219_Measurement_t g_last_measurement = {0};
static TX_THREAD g_ina219_thread;
static TX_SEMAPHORE g_ina219_semaphore;
static uint8_t g_ina219_thread_stack[INA219_THREAD_STACK_SIZE];
static volatile bool g_thread_running = false;

static float INA219_ConvertSignedRaw(uint16_t raw_value, float scale)
{
    const int32_t signed_raw =
        (raw_value & 0x8000U) ? -((int32_t)((~raw_value + 1U) & 0xFFFFU)) : (int32_t)raw_value;
    return (float)signed_raw * scale;
}

static long INA219_ToTenths(float value)
{
    return (long)lroundf(value * 10.0f);
}

/* Private function prototypes -----------------------------------------------*/
static bool INA219_WriteRegister(uint8_t reg, uint16_t value);
static bool INA219_ReadRegister(uint8_t reg, uint16_t *value);
static void INA219_ThreadEntry(ULONG thread_input);

/**
 * @brief Write a 16-bit value to an INA219 register.
 */
static bool INA219_WriteRegister(uint8_t reg, uint16_t value)
{
    if (g_hi2c == NULL) {
        return false;
    }
    
    uint8_t data[3];
    data[0] = reg;
    data[1] = (uint8_t)((value >> 8) & 0xFFU);  /* MSB first */
    data[2] = (uint8_t)(value & 0xFFU);
    
    HAL_StatusTypeDef status = HAL_I2C_Master_Transmit(
        g_hi2c, INA219_I2C_ADDRESS_HAL, data, 3, INA219_TIMEOUT_MS);
    
    return (status == HAL_OK);
}

/**
 * @brief Read a 16-bit value from an INA219 register.
 */
static bool INA219_ReadRegister(uint8_t reg, uint16_t *value)
{
    if (g_hi2c == NULL || value == NULL) {
        return false;
    }
    
    uint8_t data[2];
    
    /* Write register address */
    HAL_StatusTypeDef status = HAL_I2C_Master_Transmit(
        g_hi2c, INA219_I2C_ADDRESS_HAL, &reg, 1, INA219_TIMEOUT_MS);
    if (status != HAL_OK) {
        return false;
    }
    
    /* Read 2 bytes */
    status = HAL_I2C_Master_Receive(
        g_hi2c, INA219_I2C_ADDRESS_HAL, data, 2, INA219_TIMEOUT_MS);
    if (status != HAL_OK) {
        return false;
    }
    
    /* MSB first */
    *value = ((uint16_t)data[0] << 8) | (uint16_t)data[1];
    return true;
}

/**
 * @brief Initialize the INA219 power monitor.
 */
bool INA219_Init(I2C_HandleTypeDef *hi2c)
{
    if (hi2c == NULL) {
        return false;
    }
    
    g_hi2c = hi2c;
    
    /* Check if device is present */
    HAL_StatusTypeDef status = HAL_I2C_IsDeviceReady(
        g_hi2c, INA219_I2C_ADDRESS_HAL, INA219_PROBE_TRIALS, INA219_TIMEOUT_MS);
    if (status != HAL_OK) {
        DebugConsole_Printf("[INA219] Device not found at 0x%02X\r\n", INA219_I2C_ADDRESS_7BIT);
        return false;
    }
    
    /* Configure INA219 */
    /* 0x399F = 32V bus range, +/-320mV shunt, 12-bit ADC, continuous mode */
    if (!INA219_WriteRegister(INA219_REG_CONFIG, INA219_CONFIG_DEFAULT)) {
        DebugConsole_Printf("[INA219] Failed to write config\r\n");
        return false;
    }
    
    /* Set calibration register */
    if (!INA219_WriteRegister(INA219_REG_CALIBRATION, INA219_CALIBRATION_VALUE)) {
        DebugConsole_Printf("[INA219] Failed to write calibration\r\n");
        return false;
    }
    
    g_initialized = true;
    DebugConsole_Printf("[INA219] Initialized at 0x%02X\r\n", INA219_I2C_ADDRESS_7BIT);

    /* Seed the measurement cache so Metrics_ReadPower doesn't return 0 W
     * on the very first pipeline start. */
    {
        INA219_Measurement_t seed = {0};
        (void) INA219_ReadMeasurement(&seed);
    }

    return true;
}

/**
 * @brief Read all measurements from INA219.
 */
bool INA219_ReadMeasurement(INA219_Measurement_t *measurement)
{
    if (!g_initialized || measurement == NULL) {
        return false;
    }
    
    uint16_t raw_bus, raw_shunt, raw_current, raw_power;
    
    /* Read bus voltage register */
    if (!INA219_ReadRegister(INA219_REG_BUS_VOLTAGE, &raw_bus)) {
        return false;
    }
    
    /* Read shunt voltage register */
    if (!INA219_ReadRegister(INA219_REG_SHUNT_VOLTAGE, &raw_shunt)) {
        return false;
    }
    
    /* Read current register */
    if (!INA219_ReadRegister(INA219_REG_CURRENT, &raw_current)) {
        return false;
    }
    
    /* Read power register */
    if (!INA219_ReadRegister(INA219_REG_POWER, &raw_power)) {
        return false;
    }
    
    /* Convert raw values */
    /* Bus voltage: 4mV per bit, shift right 3 bits */
    measurement->bus_voltage_v = ((float)(raw_bus >> 3)) * 0.004f;
    
    /* Shunt voltage: 10uV per bit, signed. Flip the sign so board draw is positive. */
    measurement->shunt_voltage_mv =
        -INA219_ConvertSignedRaw(raw_shunt, 0.01f);

    /* Current: based on calibration. Flip the sign so board draw is positive. */
    measurement->current_ma =
        -INA219_ConvertSignedRaw(raw_current, INA219_CURRENT_LSB_MA);

    /* Power: based on calibration, reported in watts. */
    measurement->power_w = ((float)raw_power) * INA219_POWER_LSB_W;
    
    measurement->timestamp_ms = tx_time_get();
    measurement->valid = true;
    
    /* Store last valid measurement */
    memcpy(&g_last_measurement, measurement, sizeof(INA219_Measurement_t));
    
    return true;
}

/**
 * @brief Get the last valid measurement.
 */
bool INA219_GetLastMeasurement(INA219_Measurement_t *measurement)
{
    if (measurement == NULL) {
        return false;
    }
    
    if (!g_last_measurement.valid) {
        return false;
    }
    
    memcpy(measurement, &g_last_measurement, sizeof(INA219_Measurement_t));
    return true;
}

/**
 * @brief INA219 monitoring thread entry function.
 */
static void INA219_ThreadEntry(ULONG thread_input)
{
    (void)thread_input;
    
    INA219_Measurement_t measurement;
    ULONG actual_flags;
    
    DebugConsole_Printf("[INA219] Monitoring thread started\r\n");
    
    while (g_thread_running) {
        /* Wait for semaphore or timeout (1 second) */
        (void) tx_semaphore_get(&g_ina219_semaphore, INA219_SAMPLE_PERIOD_MS);

        /* Read sensor and feed power (mW) to the metrics subsystem so
         * min/avg/max can be reported per-pipeline after latency ends. */
        if (INA219_ReadMeasurement(&measurement)) {
            Metrics_PowerSample(measurement.power_w * 1000.0f);
        }
    }
    
    DebugConsole_Printf("[INA219] Monitoring thread exiting\r\n");
}

/**
 * @brief Start the power monitoring thread.
 */
bool INA219_StartMonitoringThread(void)
{
    if (!g_initialized) {
        DebugConsole_Printf("[INA219] Cannot start thread - not initialized\r\n");
        return false;
    }
    
    if (g_thread_running) {
        return true;  /* Already running */
    }
    
    /* Create semaphore */
    UINT status = tx_semaphore_create(&g_ina219_semaphore, "INA219 Sem", 0);
    if (status != TX_SUCCESS) {
        DebugConsole_Printf("[INA219] Failed to create semaphore\r\n");
        return false;
    }
    
    /* Create thread */
    g_thread_running = true;
    status = tx_thread_create(
        &g_ina219_thread,
        "INA219 Thread",
        INA219_ThreadEntry,
        0,
        g_ina219_thread_stack,
        INA219_THREAD_STACK_SIZE,
        INA219_THREAD_PRIORITY,
        INA219_THREAD_PRIORITY,
        TX_NO_TIME_SLICE,
        TX_AUTO_START);
    
    if (status != TX_SUCCESS) {
        g_thread_running = false;
        DebugConsole_Printf("[INA219] Failed to create thread\r\n");
        return false;
    }
    
    DebugConsole_Printf("[INA219] Monitoring thread started\r\n");
    return true;
}

/**
 * @brief Trigger a power reading and log it with a label (silent).
 */
bool INA219_LogReading(const char *label)
{
	INA219_Measurement_t measurement;

	if (!g_initialized)
	{
		return false;
	}

	if (!INA219_ReadMeasurement(&measurement))
	{
		return false;
	}

	(void)label;

	/* Feed the current measurement into the active pipeline window so the
	 * min/avg/max summary includes explicit stage checkpoints as well as the
	 * background sampler thread. */
	Metrics_PowerSample(measurement.power_w * 1000.0f);

	return true;
}

/**
 * @brief Check if INA219 is detected and responding.
 */
bool INA219_IsReady(void)
{
    return g_initialized;
}
