/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : ds3231_clock.h
 * @brief          : DS3231 RTC helper declarations.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __DS3231_CLOCK_H
#define __DS3231_CLOCK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

bool App_Clock_GetCaptureTimestamp(char *buffer, uint32_t buffer_length);
bool App_Clock_GetCurrentTimestamp(char *buffer, uint32_t buffer_length);
void DS3231_LogI2c1LineState(void);
void DS3231_ScanI2C1Bus(void);
void DS3231_LogBootTime(void);

#ifdef __cplusplus
}
#endif

#endif /* __DS3231_CLOCK_H */
