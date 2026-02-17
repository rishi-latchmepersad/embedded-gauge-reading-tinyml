/*
 * debug_console.c
 *
 *  Created on: 10 Feb 2026
 *      Author: rishi_latchmepersad
 */
#include "debug_console.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#ifndef DEBUG_CONSOLE_FORMAT_BUFFER_SIZE_BYTES
#define DEBUG_CONSOLE_FORMAT_BUFFER_SIZE_BYTES (256U)
#endif

static DebugConsole_Configuration_t g_debug_console_configuration;
static bool g_debug_console_is_initialized = false;

static void DebugConsole_InternalLock(void);
static void DebugConsole_InternalUnlock(void);
static bool DebugConsole_InternalUartTransmitBlocking(
		const uint8_t *byte_array_pointer, size_t byte_array_length);

/**
 * @brief  Initializes the debug console module.
 * @param  configuration_pointer Pointer to the debug console configuration.
 * @return true if initialization succeeded, false otherwise.
 * @details
 * Purpose:
 *   Configures the debug console to use the provided UART handle and optional
 *   lock callbacks.
 * Params:
 *   configuration_pointer: must not be NULL, uart_handle_pointer must not be NULL.
 * Returns:
 *   true on success, false on invalid parameters.
 * Side effects:
 *   Stores configuration globally and enables module usage.
 * Preconditions:
 *   UART must already be initialized (CubeMX HAL init or BSP COM init).
 * Concurrency:
 *   Safe if optional lock callbacks are provided and used.
 * Timing:
 *   Constant time.
 * Errors:
 *   Returns false for NULL config or NULL UART handle pointer.
 * Notes:
 *   Call this after MX_USARTx_UART_Init() or BSP_COM_Init().
 */
bool DebugConsole_Init(
		const DebugConsole_Configuration_t *configuration_pointer) {
	if (configuration_pointer == NULL) {
		return false;
	}

	if (configuration_pointer->uart_handle_pointer == NULL) {
		return false;
	}

	g_debug_console_configuration = *configuration_pointer;

	if (g_debug_console_configuration.uart_transmit_timeout_milliseconds
			== 0U) {
		g_debug_console_configuration.uart_transmit_timeout_milliseconds = 100U;
	}

	g_debug_console_is_initialized = true;
	return true;
}

/**
 * @brief  Checks whether the debug console has been initialized.
 * @param  None.
 * @return true if initialized, false otherwise.
 * @details
 * Purpose:
 *   Allows callers to guard logging calls before init.
 * Preconditions:
 *   None.
 * Concurrency:
 *   Safe.
 */
bool DebugConsole_IsInitialized(void) {
	return g_debug_console_is_initialized;
}

/**
 * @brief  Writes raw bytes to the configured debug output.
 * @param  byte_array_pointer Pointer to bytes to send.
 * @param  byte_array_length Number of bytes to send.
 * @return true if send request accepted, false otherwise.
 * @details
 * Purpose:
 *   Provides a single transport function for all debug output.
 * Side effects:
 *   Performs a blocking UART transmit.
 * Preconditions:
 *   DebugConsole_Init() called and UART initialized.
 * Concurrency:
 *   Use lock callbacks if multiple tasks can log concurrently.
 * Timing:
 *   O(n), blocking on UART transmission.
 * Errors:
 *   Returns false if not initialized or invalid inputs or HAL transmit fails.
 * Notes:
 *   Avoid calling this from interrupt context if lock callbacks block.
 */
bool DebugConsole_WriteBytes(const uint8_t *byte_array_pointer,
		size_t byte_array_length) {
	bool transmit_successful = false;

	if (g_debug_console_is_initialized == false) {
		return false;
	}

	if (byte_array_pointer == NULL) {
		return false;
	}

	if (byte_array_length == 0U) {
		return true;
	}

	DebugConsole_InternalLock();
	transmit_successful = DebugConsole_InternalUartTransmitBlocking(
			byte_array_pointer, byte_array_length);
	DebugConsole_InternalUnlock();

	return transmit_successful;
}

/**
 * @brief  Writes a null-terminated string to the debug output.
 * @param  null_terminated_string_pointer Pointer to a null-terminated string.
 * @return true if written, false otherwise.
 * @details
 * Purpose:
 *   Convenience wrapper around DebugConsole_WriteBytes().
 * Preconditions:
 *   DebugConsole_Init() called.
 */
bool DebugConsole_WriteString(const char *null_terminated_string_pointer) {
	size_t string_length = 0U;

	if (null_terminated_string_pointer == NULL) {
		return false;
	}

	string_length = strlen(null_terminated_string_pointer);
	return DebugConsole_WriteBytes(
			(const uint8_t*) null_terminated_string_pointer, string_length);
}

/**
 * @brief  Prints formatted output to the debug console.
 * @param  format_string_pointer printf-style format string.
 * @param  ... Variadic arguments for the format string.
 * @return true if written, false otherwise.
 * @details
 * Purpose:
 *   Provides printf-like formatting without requiring libc printf retargeting.
 * Preconditions:
 *   DebugConsole_Init() called.
 * Concurrency:
 *   Protected by optional lock callbacks inside DebugConsole_WriteBytes().
 * Timing:
 *   O(n) formatting plus UART transmit.
 * Notes:
 *   Keeps formatting buffer on stack, adjust DEBUG_CONSOLE_FORMAT_BUFFER_SIZE_BYTES if needed.
 */
bool DebugConsole_Printf(const char *format_string_pointer, ...) {
	va_list variadic_argument_list;
	bool result_value = false;

	if (format_string_pointer == NULL) {
		return false;
	}

	va_start(variadic_argument_list, format_string_pointer);
	result_value = DebugConsole_VPrintf(format_string_pointer,
			(void*) &variadic_argument_list);
	va_end(variadic_argument_list);

	return result_value;
}

/**
 * @brief  Prints formatted output using a va_list to the debug console.
 * @param  format_string_pointer printf-style format string.
 * @param  va_list_pointer Pointer to a va_list (passed as void* to keep header minimal).
 * @return true if written, false otherwise.
 * @details
 * Purpose:
 *   Lets you build wrappers or integrate with other logging systems.
 * Preconditions:
 *   DebugConsole_Init() called.
 * Errors:
 *   Returns false on formatting error.
 */
bool DebugConsole_VPrintf(const char *format_string_pointer,
		void *va_list_pointer) {
	char formatted_output_buffer[DEBUG_CONSOLE_FORMAT_BUFFER_SIZE_BYTES];
	int formatted_character_count = 0;

	if (format_string_pointer == NULL) {
		return false;
	}

	if (va_list_pointer == NULL) {
		return false;
	}

	formatted_output_buffer[0] = '\0';

	formatted_character_count = vsnprintf(formatted_output_buffer,
			sizeof(formatted_output_buffer), format_string_pointer,
			*(va_list*) va_list_pointer);

	if (formatted_character_count < 0) {
		return false;
	}

	/* vsnprintf returns number of chars that would have been written, excluding null terminator */
	if ((size_t) formatted_character_count >= sizeof(formatted_output_buffer)) {
		/* Output was truncated, still transmit what we have */
		formatted_output_buffer[sizeof(formatted_output_buffer) - 1U] = '\0';
	}

	return DebugConsole_WriteString(formatted_output_buffer);
}

static void DebugConsole_InternalLock(void) {
	if (g_debug_console_configuration.lock_callback != NULL) {
		g_debug_console_configuration.lock_callback();
	}
}

static void DebugConsole_InternalUnlock(void) {
	if (g_debug_console_configuration.unlock_callback != NULL) {
		g_debug_console_configuration.unlock_callback();
	}
}

static bool DebugConsole_InternalUartTransmitBlocking(
		const uint8_t *byte_array_pointer, size_t byte_array_length) {
	HAL_StatusTypeDef transmit_status = HAL_ERROR;

	if (g_debug_console_configuration.uart_handle_pointer == NULL) {
		return false;
	}

	transmit_status = HAL_UART_Transmit(
			g_debug_console_configuration.uart_handle_pointer,
			(uint8_t*) byte_array_pointer, (uint16_t) byte_array_length,
			g_debug_console_configuration.uart_transmit_timeout_milliseconds);

	return (transmit_status == HAL_OK);
}
