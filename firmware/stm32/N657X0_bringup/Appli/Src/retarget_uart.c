/*
 * retarget_uart.c
 *
 *  Created on: 9 Feb 2026
 *      Author: rishi_latchmepersad
 */

#include "debug_console.h"
#include <unistd.h>

int _write(int file, char *ptr, int len) {
	(void) file;

	if ((ptr == NULL) || (len <= 0)) {
		return 0;
	}

	(void) DebugConsole_WriteBytes((const uint8_t*) ptr, (size_t) len);
	return len;
}
