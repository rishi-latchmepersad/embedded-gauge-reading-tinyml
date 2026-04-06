/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    ds3231_clock.c
 * @brief   DS3231 RTC helpers and boot-time timestamp handling.
 ******************************************************************************
 */
/* USER CODE END Header */

#include "ds3231_clock.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>

#include "debug_console.h"
#include "main.h"
/* USER CODE END Includes */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#define DS3231_I2C_ADDRESS_7BIT      0x68U
#define DS3231_I2C_ADDRESS_HAL       (DS3231_I2C_ADDRESS_7BIT << 1U)
#define DS3231_I2C_PROBE_TRIALS      3U
#define DS3231_I2C_PROBE_TIMEOUT_MS  50U
#define DS3231_READ_RETRY_ATTEMPTS    5U
#define DS3231_READ_RETRY_DELAY_MS   10U
/* Leave this at 0 for normal boots. Flip it to 1 only when you intentionally
 * want to force a one-time DS3231 update from the firmware build timestamp. */
#define DS3231_ENABLE_BUILD_TIME_SEED 0

/* USER CODE END PD */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

static char g_ds3231_last_timestamp[32];
static bool g_ds3231_last_timestamp_valid = false;

extern I2C_HandleTypeDef hi2c1;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

static uint8_t DS3231_BcdToBin(uint8_t value);
static uint8_t DS3231_DecodeHour(uint8_t value);
void DS3231_LogI2c1LineState(void);
void DS3231_ScanI2C1Bus(void);
static bool DS3231_ReadDateTime(char *buffer, size_t buffer_length);
static bool DS3231_ReadDateTimeWithRetry(char *buffer, size_t buffer_length);
#if DS3231_ENABLE_BUILD_TIME_SEED
static void DS3231_SetBuildTime(void);
#endif

/* USER CODE END PFP */

/* USER CODE BEGIN 0 */

/**
 * @brief Convert a packed BCD byte into binary.
 * @param value Packed BCD value from the DS3231.
 * @retval Converted binary value.
 */
static uint8_t DS3231_BcdToBin(uint8_t value) {
	return (uint8_t) (((value >> 4U) * 10U) + (value & 0x0FU));
}

/**
 * @brief Decode a DS3231 hour register into 24-hour binary form.
 * @param value Raw hour register byte.
 * @retval Hour in the range 0-23.
 */
static uint8_t DS3231_DecodeHour(uint8_t value) {
	if ((value & 0x40U) == 0U) {
		return DS3231_BcdToBin((uint8_t) (value & 0x3FU));
	}

	uint8_t hour = DS3231_BcdToBin((uint8_t) (value & 0x1FU));
	if ((value & 0x20U) != 0U) {
		if (hour != 12U) {
			hour = (uint8_t) (hour + 12U);
		}
	} else if (hour == 12U) {
		hour = 0U;
	}

	return hour;
}

/**
 * @brief Log the I2C1 pin state so we can spot wiring or pull-up issues.
 */
void DS3231_LogI2c1LineState(void) {
	uint32_t pc1_moder = (GPIOC->MODER >> (1U * 2U)) & 0x3U;
	uint32_t ph9_moder = (GPIOH->MODER >> (9U * 2U)) & 0x3U;
	uint32_t pc1_pupdr = (GPIOC->PUPDR >> (1U * 2U)) & 0x3U;
	uint32_t ph9_pupdr = (GPIOH->PUPDR >> (9U * 2U)) & 0x3U;
	uint32_t pc1_afr = (GPIOC->AFR[0] >> (1U * 4U)) & 0xFU;
	uint32_t ph9_afr = (GPIOH->AFR[1] >> ((9U - 8U) * 4U)) & 0xFU;
	GPIO_PinState sda_state = HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_1);
	GPIO_PinState scl_state = HAL_GPIO_ReadPin(GPIOH, GPIO_PIN_9);

	DebugConsole_Printf(
			"[RTC] I2C1 lines: PC1(MODE=%lu PUPD=%lu AF=%lu state=%u) PH9(MODE=%lu PUPD=%lu AF=%lu state=%u)\r\n",
			(unsigned long) pc1_moder, (unsigned long) pc1_pupdr,
			(unsigned long) pc1_afr, (unsigned int) sda_state,
			(unsigned long) ph9_moder, (unsigned long) ph9_pupdr,
			(unsigned long) ph9_afr, (unsigned int) scl_state);
}

/**
 * @brief Scan the I2C1 bus for any responding device addresses.
 */
void DS3231_ScanI2C1Bus(void) {
	uint32_t found_count = 0U;
	char found_list[128] = { 0 };
	size_t used = 0U;

	DebugConsole_Printf("[RTC] I2C1 scan starting.\r\n");

	for (uint32_t address = 0x03U; address <= 0x77U; ++address) {
		if (HAL_I2C_IsDeviceReady(&hi2c1, (uint16_t) (address << 1U), 1U, 10U)
				== HAL_OK) {
			if (found_count == 0U) {
				used = (size_t) snprintf(found_list, sizeof(found_list),
						"[RTC] I2C1 ACKs:");
			}

			if (used < sizeof(found_list)) {
				int written = snprintf(&found_list[used],
						sizeof(found_list) - used, " 0x%02lX",
						(unsigned long) address);
				if (written > 0) {
					used += (size_t) written;
				}
			}

			++found_count;
		}
	}

	if (found_count > 0U) {
		DebugConsole_Printf("%s\r\n", found_list);
	} else {
		DebugConsole_Printf("[RTC] I2C1 scan found no ACKs.\r\n");
	}
}

/**
 * @brief Read the DS3231 clock registers and format a timestamp.
 */
static bool DS3231_ReadDateTime(char *buffer, size_t buffer_length) {
	uint8_t raw_registers[7] = { 0 };
	HAL_StatusTypeDef status;
	int printed = 0;

	if ((buffer == NULL) || (buffer_length == 0U)) {
		return false;
	}

	hi2c1.ErrorCode = HAL_I2C_ERROR_NONE;
	hi2c1.State = HAL_I2C_STATE_READY;
	hi2c1.PreviousState = 0U;

	status = HAL_I2C_IsDeviceReady(&hi2c1, DS3231_I2C_ADDRESS_HAL,
	DS3231_I2C_PROBE_TRIALS,
	DS3231_I2C_PROBE_TIMEOUT_MS);
	if (status != HAL_OK) {
		return false;
	}

	status = HAL_I2C_Mem_Read(&hi2c1, DS3231_I2C_ADDRESS_HAL, 0x00U,
	I2C_MEMADD_SIZE_8BIT, raw_registers, sizeof(raw_registers), 100U);
	if (status != HAL_OK) {
		return false;
	}

	printed = snprintf(buffer, buffer_length, "%04u-%02u-%02u %02u:%02u:%02u",
			(unsigned int) (2000U + DS3231_BcdToBin(raw_registers[6])),
			(unsigned int) DS3231_BcdToBin(
					(uint8_t) (raw_registers[5] & 0x1FU)),
			(unsigned int) DS3231_BcdToBin(raw_registers[4]),
			(unsigned int) DS3231_DecodeHour(raw_registers[2]),
			(unsigned int) DS3231_BcdToBin(raw_registers[1]),
			(unsigned int) DS3231_BcdToBin(
					(uint8_t) (raw_registers[0] & 0x7FU)));
	return (printed > 0) && ((size_t) printed < buffer_length);
}

/**
 * @brief Try to read the DS3231 clock registers a few times before giving up.
 */
static bool DS3231_ReadDateTimeWithRetry(char *buffer, size_t buffer_length) {
	for (uint32_t attempt = 0U; attempt < DS3231_READ_RETRY_ATTEMPTS;
			++attempt) {
		if (DS3231_ReadDateTime(buffer, buffer_length)) {
			return true;
		}

		if ((attempt + 1U) < DS3231_READ_RETRY_ATTEMPTS) {
			HAL_Delay(DS3231_READ_RETRY_DELAY_MS);
		}
	}

	return false;
}

#if DS3231_ENABLE_BUILD_TIME_SEED
/**
 * @brief Write the firmware build timestamp to the DS3231 registers.
 */
static void DS3231_SetBuildTime(void) {
	static const char *const months = "JanFebMarAprMayJunJulAugSepOctNovDec";
	char mon_str[4] = { __DATE__[0], __DATE__[1], __DATE__[2], '\0' };
	uint8_t month = 1U;
	for (uint8_t m = 0U; m < 12U; m++) {
		if (strncmp(mon_str, &months[m * 3U], 3U) == 0) {
			month = (uint8_t) (m + 1U);
			break;
		}
	}

	uint8_t day = (uint8_t) ((
			__DATE__[4] == ' ' ? 0U : (uint8_t) (__DATE__[4] - '0')) * 10U
			+ (uint8_t) (__DATE__[5] - '0'));
	uint8_t year = (uint8_t) (((uint8_t) (__DATE__[9] - '0')) * 10U
			+ (uint8_t) (__DATE__[10] - '0'));
	uint8_t hour = (uint8_t) (((uint8_t) (__TIME__[0] - '0')) * 10U
			+ (uint8_t) (__TIME__[1] - '0'));
	uint8_t min = (uint8_t) (((uint8_t) (__TIME__[3] - '0')) * 10U
			+ (uint8_t) (__TIME__[4] - '0'));
	uint8_t sec = (uint8_t) (((uint8_t) (__TIME__[6] - '0')) * 10U
			+ (uint8_t) (__TIME__[7] - '0'));

#define TO_BCD(v)  ((uint8_t)((((v) / 10U) << 4U) | ((v) % 10U)))
	uint8_t regs[8] = { 0x00U, TO_BCD(sec), TO_BCD(min), TO_BCD(hour), 0x01U,
			TO_BCD(day), TO_BCD(month), TO_BCD(year), };
#undef TO_BCD

	HAL_StatusTypeDef status = HAL_I2C_Master_Transmit(&hi2c1,
			DS3231_I2C_ADDRESS_HAL, regs, sizeof(regs), 100U);
	if (status == HAL_OK) {
		DebugConsole_Printf(
				"[RTC] Set DS3231 to build time: 20%02u-%02u-%02u %02u:%02u:%02u\r\n",
				(unsigned int) year, (unsigned int) month, (unsigned int) day,
				(unsigned int) hour, (unsigned int) min, (unsigned int) sec);
	} else {
		DebugConsole_Printf(
				"[RTC] DS3231 set-time write failed (status=%d).\r\n",
				(int) status);
	}
}
#endif

/**
 * @brief Print the DS3231 time once during boot so we can confirm the module.
 */
void DS3231_LogBootTime(void) {
	char rtc_text[32] = { 0 };

#if DS3231_ENABLE_BUILD_TIME_SEED
	DebugConsole_Printf(
			"[RTC] DS3231 build-time seed enabled; writing compile-time timestamp now.\r\n");
	DS3231_SetBuildTime();
#endif

	if (DS3231_ReadDateTimeWithRetry(rtc_text, sizeof(rtc_text))) {
		if (strncmp(rtc_text, "2000-", 5U) == 0) {
			DebugConsole_Printf(
					"[RTC] DS3231 year=2000 (power-on default); build-time seed is disabled.\r\n");
		}

		(void) snprintf(g_ds3231_last_timestamp,
				sizeof(g_ds3231_last_timestamp), "%s", rtc_text);
		g_ds3231_last_timestamp_valid = true;
		DebugConsole_Printf("[RTC] DS3231 time: %s\r\n", rtc_text);
	} else {
		DebugConsole_Printf("[RTC] DS3231 probe/read failed on I2C1.\r\n");
	}
}

/**
 * @brief Format the current DS3231 time for use in a capture filename.
 */
bool App_Clock_GetCaptureTimestamp(char *buffer, uint32_t buffer_length) {
	char rtc_text[32] = { 0 };
	uint32_t source_index = 0U;
	uint32_t dest_index = 0U;

	if ((buffer == NULL) || (buffer_length == 0U)) {
		return false;
	}

	if (!DS3231_ReadDateTimeWithRetry(rtc_text, sizeof(rtc_text))) {
		if (!g_ds3231_last_timestamp_valid) {
			return false;
		}

		(void) snprintf(rtc_text, sizeof(rtc_text), "%s",
				g_ds3231_last_timestamp);
	}

	(void) snprintf(g_ds3231_last_timestamp, sizeof(g_ds3231_last_timestamp),
			"%s", rtc_text);
	g_ds3231_last_timestamp_valid = true;

	while ((rtc_text[source_index] != '\0')
			&& ((dest_index + 1U) < buffer_length)) {
		char ch = rtc_text[source_index++];

		if (ch == ' ') {
			ch = '_';
		} else if (ch == ':') {
			ch = '-';
		}

		buffer[dest_index++] = ch;
	}

	buffer[dest_index] = '\0';
	return (rtc_text[source_index] == '\0');
}

/* USER CODE END 0 */
