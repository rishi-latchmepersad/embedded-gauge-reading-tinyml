/**
 * @file    imx335_driver_vendor.c
 * @brief   Build the ST IMX335 sensor driver from the copied vendor source.
 *
 * The full Camera_Middleware tree is excluded from the managed build to avoid
 * pulling in ISP and USB dependencies. This thin wrapper lets us compile only
 * the IMX335 sensor driver we need for bring-up.
 */

#include "../Middlewares/Third_Party/Camera_Middleware/sensors/imx335/imx335.c"
