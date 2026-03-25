/**
 * @file cmw_camera_vendor.c
 * @brief Build wrapper for ST's camera middleware core.
 *
 * The application already owns the board-specific DCMIPP MSP hooks, so we
 * rename ST's default MSP helpers here while keeping the public CMW camera
 * APIs and pipe callbacks intact.
 */

#include <stdint.h>
#include "stm32n6xx_hal.h"

#define USE_IMX335_SENSOR

#define HAL_DCMIPP_MspInit   CMW_VENDOR_HAL_DCMIPP_MspInit_unused
#define HAL_DCMIPP_MspDeInit CMW_VENDOR_HAL_DCMIPP_MspDeInit_unused

static int32_t BSP_GetTick(void)
{
  return (int32_t)HAL_GetTick();
}

#include "../Middlewares/Third_Party/Camera_Middleware/cmw_camera.c"
