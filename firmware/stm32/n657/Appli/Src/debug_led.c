#include "debug_led.h"

#include <stddef.h>

static DebugLed_Configuration_t g_debug_led_configuration;
static bool g_debug_led_is_initialized = false;

static bool DebugLed_InternalIsColorValid(DebugLed_Color_t color);
static bool DebugLed_InternalDelayMilliseconds(uint32_t delay_time_milliseconds);
static bool DebugLed_InternalInitializeMappedBspLeds(void);

/**
 * @brief  Initialize the debug LED abstraction layer.
 * @param  configuration_pointer Pointer to configuration describing color mapping and delay callback.
 * @return true on success, false on invalid configuration or BSP init failure.
 * @sideeffects Initializes BSP LEDs using BSP_LED_Init() for mapped LEDs.
 * @preconditions BSP is available; stm32n6xx_nucleo.h is in include path.
 * @concurrency Not thread-safe; call once during startup before concurrent threads.
 * @timing O(1).
 * @errors Returns false if configuration_pointer is NULL or mapping contains invalid LED IDs.
 * @notes Default mappings are not assumed here, you must provide mapping explicitly.
 */
bool DebugLed_Initialize(const DebugLed_Configuration_t *configuration_pointer) {
	if (configuration_pointer == NULL) {
		return false;
	}

	g_debug_led_configuration = *configuration_pointer;

	if (DebugLed_InternalInitializeMappedBspLeds() == false) {
		g_debug_led_is_initialized = false;
		return false;
	}

	g_debug_led_is_initialized = true;
	return true;
}

/**
 * @brief  Check whether DebugLed has been initialized.
 * @param  None.
 * @return true if initialized, false otherwise.
 * @sideeffects None.
 * @preconditions None.
 * @concurrency Safe.
 * @timing O(1).
 * @errors None.
 * @notes None.
 */
bool DebugLed_IsInitialized(void) {
	return g_debug_led_is_initialized;
}

/**
 * @brief  Update mapping from a logical LED color to a BSP LED.
 * @param  color The logical color to map.
 * @param  bsp_led The BSP LED identifier (LED1, LED2, LED3).
 * @return true on success, false on invalid input or if not initialized.
 * @sideeffects Updates internal mapping used by subsequent calls.
 * @preconditions DebugLed_Initialize() has been called.
 * @concurrency Not thread-safe with concurrent LED operations.
 * @timing O(1).
 * @errors Returns false if invalid color or invalid bsp_led.
 * @notes This does not call BSP_LED_Init again; call this before DebugLed_Initialize for best behavior.
 */
bool DebugLed_SetColorToBspLedMapping(DebugLed_Color_t color,
		Led_TypeDef bsp_led) {
	if (g_debug_led_is_initialized == false) {
		return false;
	}

	if (DebugLed_InternalIsColorValid(color) == false) {
		return false;
	}

	if ((bsp_led != LED1) && (bsp_led != LED2) && (bsp_led != LED3)) {
		return false;
	}

	g_debug_led_configuration.bsp_led_for_color_array[(uint32_t) color] =
			bsp_led;
	return true;
}

/**
 * @brief  Turn on the specified logical LED color.
 * @param  color Logical color to turn on.
 * @return true on success, false otherwise.
 * @sideeffects Calls BSP_LED_On() for mapped BSP LED.
 * @preconditions DebugLed_Initialize() has been called and BSP LED is initialized.
 * @concurrency Not thread-safe without external synchronization.
 * @timing O(1).
 * @errors Returns false if color invalid or module not initialized.
 * @notes None.
 */
bool DebugLed_TurnOn(DebugLed_Color_t color) {
	if (g_debug_led_is_initialized == false) {
		return false;
	}

	if (DebugLed_InternalIsColorValid(color) == false) {
		return false;
	}

	BSP_LED_On(
			g_debug_led_configuration.bsp_led_for_color_array[(uint32_t) color]);
	return true;
}

/**
 * @brief  Turn off the specified logical LED color.
 * @param  color Logical color to turn off.
 * @return true on success, false otherwise.
 * @sideeffects Calls BSP_LED_Off() for mapped BSP LED.
 * @preconditions DebugLed_Initialize() has been called and BSP LED is initialized.
 * @concurrency Not thread-safe without external synchronization.
 * @timing O(1).
 * @errors Returns false if color invalid or module not initialized.
 * @notes None.
 */
bool DebugLed_TurnOff(DebugLed_Color_t color) {
	if (g_debug_led_is_initialized == false) {
		return false;
	}

	if (DebugLed_InternalIsColorValid(color) == false) {
		return false;
	}

	BSP_LED_Off(
			g_debug_led_configuration.bsp_led_for_color_array[(uint32_t) color]);
	return true;
}

/**
 * @brief  Toggle the specified logical LED color.
 * @param  color Logical color to toggle.
 * @return true on success, false otherwise.
 * @sideeffects Calls BSP_LED_Toggle() for mapped BSP LED.
 * @preconditions DebugLed_Initialize() has been called and BSP LED is initialized.
 * @concurrency Not thread-safe without external synchronization.
 * @timing O(1).
 * @errors Returns false if color invalid or module not initialized.
 * @notes None.
 */
bool DebugLed_Toggle(DebugLed_Color_t color) {
	if (g_debug_led_is_initialized == false) {
		return false;
	}

	if (DebugLed_InternalIsColorValid(color) == false) {
		return false;
	}

	BSP_LED_Toggle(
			g_debug_led_configuration.bsp_led_for_color_array[(uint32_t) color]);
	return true;
}

/**
 * @brief  Blink a logical LED color using the configured delay callback.
 * @param  color Logical color to blink.
 * @param  on_time_milliseconds Time LED stays on per blink.
 * @param  off_time_milliseconds Time LED stays off per blink.
 * @param  blink_count Number of blink cycles.
 * @return true on success, false otherwise.
 * @sideeffects Repeatedly toggles the LED with delays, blocks the calling thread.
 * @preconditions DebugLed_Initialize() called; delay callback configured if delays are non-zero.
 * @concurrency Not thread-safe without external synchronization.
 * @timing Blocking; total duration approximately blink_count*(on+off) rounded to system timing.
 * @errors Returns false if delay callback is NULL and a non-zero delay is requested.
 * @notes For ThreadX, set delay callback to a wrapper around tx_thread_sleep(ticks).
 */
bool DebugLed_BlinkBlocking(DebugLed_Color_t color,
		uint32_t on_time_milliseconds, uint32_t off_time_milliseconds,
		uint32_t blink_count) {
	uint32_t blink_index = 0U;

	if (g_debug_led_is_initialized == false) {
		return false;
	}

	if (DebugLed_InternalIsColorValid(color) == false) {
		return false;
	}

	for (blink_index = 0U; blink_index < blink_count; blink_index++) {
		(void) DebugLed_TurnOn(color);

		if (DebugLed_InternalDelayMilliseconds(on_time_milliseconds) == false) {
			return false;
		}

		(void) DebugLed_TurnOff(color);

		if (DebugLed_InternalDelayMilliseconds(off_time_milliseconds) == false) {
			return false;
		}
	}

	return true;
}

/**
 * @brief  Blink the red debug LED.
 * @param  on_time_milliseconds Time LED stays on per blink.
 * @param  off_time_milliseconds Time LED stays off per blink.
 * @param  blink_count Number of blink cycles.
 * @return true on success, false otherwise.
 * @sideeffects Blocks caller; toggles LED.
 * @preconditions DebugLed_Initialize() called.
 * @concurrency Not thread-safe without external synchronization.
 * @timing Blocking.
 * @errors Returns false on invalid state.
 * @notes None.
 */
bool DebugLed_BlinkRedBlocking(uint32_t on_time_milliseconds,
		uint32_t off_time_milliseconds, uint32_t blink_count) {
	return DebugLed_BlinkBlocking(DEBUG_LED_COLOR_RED, on_time_milliseconds,
			off_time_milliseconds, blink_count);
}

/**
 * @brief  Blink the green debug LED.
 * @param  on_time_milliseconds Time LED stays on per blink.
 * @param  off_time_milliseconds Time LED stays off per blink.
 * @param  blink_count Number of blink cycles.
 * @return true on success, false otherwise.
 * @sideeffects Blocks caller; toggles LED.
 * @preconditions DebugLed_Initialize() called.
 * @concurrency Not thread-safe without external synchronization.
 * @timing Blocking.
 * @errors Returns false on invalid state.
 * @notes None.
 */
bool DebugLed_BlinkGreenBlocking(uint32_t on_time_milliseconds,
		uint32_t off_time_milliseconds, uint32_t blink_count) {
	return DebugLed_BlinkBlocking(DEBUG_LED_COLOR_GREEN, on_time_milliseconds,
			off_time_milliseconds, blink_count);
}

/**
 * @brief  Blink the blue debug LED.
 * @param  on_time_milliseconds Time LED stays on per blink.
 * @param  off_time_milliseconds Time LED stays off per blink.
 * @param  blink_count Number of blink cycles.
 * @return true on success, false otherwise.
 * @sideeffects Blocks caller; toggles LED.
 * @preconditions DebugLed_Initialize() called.
 * @concurrency Not thread-safe without external synchronization.
 * @timing Blocking.
 * @errors Returns false on invalid state.
 * @notes None.
 */
bool DebugLed_BlinkBlueBlocking(uint32_t on_time_milliseconds,
		uint32_t off_time_milliseconds, uint32_t blink_count) {
	return DebugLed_BlinkBlocking(DEBUG_LED_COLOR_BLUE, on_time_milliseconds,
			off_time_milliseconds, blink_count);
}

static bool DebugLed_InternalIsColorValid(DebugLed_Color_t color) {
	return ((uint32_t) color < (uint32_t) DEBUG_LED_COLOR_COUNT);
}

static bool DebugLed_InternalDelayMilliseconds(uint32_t delay_time_milliseconds) {
	if (delay_time_milliseconds == 0U) {
		return true;
	}

	if (g_debug_led_configuration.delay_milliseconds_callback == NULL) {
		return false;
	}

	g_debug_led_configuration.delay_milliseconds_callback(
			delay_time_milliseconds);
	return true;
}

static bool DebugLed_InternalInitializeMappedBspLeds(void) {
	uint32_t color_index = 0U;
	int32_t bsp_status = BSP_ERROR_NONE;

	for (color_index = 0U; color_index < (uint32_t) DEBUG_LED_COLOR_COUNT;
			color_index++) {
		const Led_TypeDef bsp_led =
				g_debug_led_configuration.bsp_led_for_color_array[color_index];

		if ((bsp_led != LED1) && (bsp_led != LED2) && (bsp_led != LED3)) {
			return false;
		}

		bsp_status = BSP_LED_Init(bsp_led);
		if (bsp_status != BSP_ERROR_NONE) {
			return false;
		}
	}

	return true;
}
