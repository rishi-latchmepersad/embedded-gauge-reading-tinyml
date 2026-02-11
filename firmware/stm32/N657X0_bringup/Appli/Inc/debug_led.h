#ifndef DEBUG_LED_H
#define DEBUG_LED_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

/* BSP header providing Led_TypeDef, LED1/LED2/LED3, BSP_LED_* prototypes */
#include "stm32n6xx_nucleo.h"

typedef enum {
	DEBUG_LED_COLOR_RED = 0,
	DEBUG_LED_COLOR_GREEN,
	DEBUG_LED_COLOR_BLUE,
	DEBUG_LED_COLOR_COUNT
} DebugLed_Color_t;

typedef void (*DebugLed_DelayMillisecondsCallback_t)(
		uint32_t delay_time_milliseconds);

typedef struct {
	/* Maps each logical color to a BSP LED index (LED1, LED2, LED3). */
	Led_TypeDef bsp_led_for_color_array[DEBUG_LED_COLOR_COUNT];

	/* Optional delay callback used by blink helpers. */
	DebugLed_DelayMillisecondsCallback_t delay_milliseconds_callback;
} DebugLed_Configuration_t;

bool DebugLed_Initialize(const DebugLed_Configuration_t *configuration_pointer);

bool DebugLed_IsInitialized(void);

bool DebugLed_SetColorToBspLedMapping(DebugLed_Color_t color,
		Led_TypeDef bsp_led);

bool DebugLed_TurnOn(DebugLed_Color_t color);

bool DebugLed_TurnOff(DebugLed_Color_t color);

bool DebugLed_Toggle(DebugLed_Color_t color);

bool DebugLed_BlinkBlocking(DebugLed_Color_t color,
		uint32_t on_time_milliseconds, uint32_t off_time_milliseconds,
		uint32_t blink_count);

bool DebugLed_BlinkRedBlocking(uint32_t on_time_milliseconds,
		uint32_t off_time_milliseconds, uint32_t blink_count);

bool DebugLed_BlinkGreenBlocking(uint32_t on_time_milliseconds,
		uint32_t off_time_milliseconds, uint32_t blink_count);

bool DebugLed_BlinkBlueBlocking(uint32_t on_time_milliseconds,
		uint32_t off_time_milliseconds, uint32_t blink_count);

#ifdef __cplusplus
}
#endif

#endif /* DEBUG_LED_H */
