/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    i2c_scanner.c
 * @brief   Simple I2C bus scanner for testing INA219 and other devices.
 ******************************************************************************
 */
/* USER CODE END Header */

#include "i2c_scanner.h"

/* Private includes ----------------------------------------------------------*/
#include <stdio.h>
#include <string.h>
#include "debug_console.h"
#include "main.h"

/* Private defines -----------------------------------------------------------*/
#define I2C_SCANNER_TIMEOUT_MS      100U
#define I2C_SCANNER_PROBE_TRIALS    3U

/* Known device addresses */
#define INA219_I2C_ADDRESS_7BIT     0x40U
#define INA219_I2C_ADDRESS_HAL      (INA219_I2C_ADDRESS_7BIT << 1U)
#define DS3231_I2C_ADDRESS_7BIT   0x68U
#define DS3231_I2C_ADDRESS_HAL      (DS3231_I2C_ADDRESS_7BIT << 1U)

/* External I2C handles from main.c */
extern I2C_HandleTypeDef hi2c1;
extern I2C_HandleTypeDef hi2c2;

/**
 * @brief Scan a single I2C bus and report found devices.
 * @param hi2c Pointer to I2C handle
 * @param bus_name String name for the bus (e.g., "I2C1")
 * @retval Number of devices found
 */
static uint8_t I2CScanner_ScanBus(I2C_HandleTypeDef *hi2c, const char *bus_name)
{
    uint8_t found_count = 0;
    
    DebugConsole_Printf("\r\n[I2C SCAN] Scanning %s...\r\n", bus_name);
    DebugConsole_Printf("[I2C SCAN] Address | Status\r\n");
    DebugConsole_Printf("[I2C SCAN] --------|--------\r\n");
    
    /* Scan all 7-bit addresses (0x08 to 0x77) */
    for (uint8_t addr_7bit = 0x08U; addr_7bit <= 0x77U; addr_7bit++)
    {
        uint16_t addr_hal = (uint16_t)(addr_7bit << 1U);
        HAL_StatusTypeDef status;
        
        /* Try to probe the device */
        status = HAL_I2C_IsDeviceReady(hi2c, addr_hal, 
                                       I2C_SCANNER_PROBE_TRIALS, 
                                       I2C_SCANNER_TIMEOUT_MS);
        
        if (status == HAL_OK)
        {
            found_count++;
            
            /* Identify known devices */
            const char *device_name = "Unknown";
            if (addr_7bit == INA219_I2C_ADDRESS_7BIT)
            {
                device_name = "INA219";
            }
            else if (addr_7bit == DS3231_I2C_ADDRESS_7BIT)
            {
                device_name = "DS3231";
            }
            
            DebugConsole_Printf("[I2C SCAN]  0x%02X   | FOUND (%s)\r\n", 
                         addr_7bit, device_name);
        }
    }
    
    DebugConsole_Printf("[I2C SCAN] %s: %d device(s) found\r\n", 
                 bus_name, found_count);
    
    return found_count;
}

/**
 * @brief Run I2C scanner on both I2C1 and I2C2 buses.
 * @retval Total number of devices found across both buses
 */
uint8_t I2CScanner_Run(void)
{
    uint8_t total_found = 0;
    
    DebugConsole_Printf("\r\n========================================\r\n");
    DebugConsole_Printf("[I2C SCAN] Starting I2C bus scan...\r\n");
    DebugConsole_Printf("[I2C SCAN] Looking for:\r\n");
    DebugConsole_Printf("[I2C SCAN]   - INA219 at 0x%02X\r\n", INA219_I2C_ADDRESS_7BIT);
    DebugConsole_Printf("[I2C SCAN]   - DS3231 at 0x%02X\r\n", DS3231_I2C_ADDRESS_7BIT);
    
    /* Scan I2C1 (PC1/PH9) */
    total_found += I2CScanner_ScanBus(&hi2c1, "I2C1 (PC1/PH9)");
    
    /* Scan I2C2 (PB11/PB10) */
    total_found += I2CScanner_ScanBus(&hi2c2, "I2C2 (PB11/PB10)");
    
    DebugConsole_Printf("\r\n[I2C SCAN] Total devices found: %d\r\n", total_found);
    DebugConsole_Printf("========================================\r\n\r\n");
    
    return total_found;
}

/**
 * @brief Quick probe for INA219 on expected address.
 * @param hi2c Pointer to I2C handle to test
 * @retval true if INA219 found, false otherwise
 */
bool I2CScanner_ProbeINA219(I2C_HandleTypeDef *hi2c)
{
    HAL_StatusTypeDef status;
    
    status = HAL_I2C_IsDeviceReady(hi2c, INA219_I2C_ADDRESS_HAL,
                                   I2C_SCANNER_PROBE_TRIALS,
                                   I2C_SCANNER_TIMEOUT_MS);
    
    if (status == HAL_OK)
    {
        DebugConsole_Printf("[I2C SCAN] INA219 found at 0x%02X\r\n",
                            INA219_I2C_ADDRESS_7BIT);
        return true;
    }
    
    return false;
}

/**
 * @brief Find which I2C bus has the INA219 connected.
 * @retval Pointer to I2C handle with INA219, or NULL if not found
 */
I2C_HandleTypeDef* I2CScanner_FindINA219(void)
{
    DebugConsole_Printf("[I2C SCAN] Probing for INA219...\r\n");
    
    /* Try I2C1 first */
    if (I2CScanner_ProbeINA219(&hi2c1))
    {
        DebugConsole_Printf("[I2C SCAN] INA219 is on I2C1\r\n");
        return &hi2c1;
    }
    
    /* Try I2C2 */
    if (I2CScanner_ProbeINA219(&hi2c2))
    {
        DebugConsole_Printf("[I2C SCAN] INA219 is on I2C2\r\n");
        return &hi2c2;
    }
    
    DebugConsole_Printf("[I2C SCAN] ERROR: INA219 not found on either bus!\r\n");
    return NULL;
}
