#ifndef DEBUG_CONSOLE_H
#define DEBUG_CONSOLE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "stm32n6xx_hal.h"

/* Optional lock callbacks for thread-safety (FreeRTOS mutex, critical section, etc.) */
typedef void (*DebugConsole_LockCallback_t)(void);
typedef void (*DebugConsole_UnlockCallback_t)(void);

typedef struct {
	UART_HandleTypeDef *uart_handle_pointer;
	uint32_t uart_transmit_timeout_milliseconds;
	DebugConsole_LockCallback_t lock_callback;
	DebugConsole_UnlockCallback_t unlock_callback;
} DebugConsole_Configuration_t;

/* Public API */
bool DebugConsole_Init(
		const DebugConsole_Configuration_t *configuration_pointer);

bool DebugConsole_IsInitialized(void);

bool DebugConsole_WriteBytes(const uint8_t *byte_array_pointer,
		size_t byte_array_length);

bool DebugConsole_Printf(const char *format_string_pointer, ...);

bool DebugConsole_VPrintf(const char *format_string_pointer,
		void *va_list_pointer);

/* Convenience: write a C string (no formatting) */
bool DebugConsole_WriteString(const char *null_terminated_string_pointer);

#ifdef __cplusplus
}
#endif

#endif /* DEBUG_CONSOLE_H */
