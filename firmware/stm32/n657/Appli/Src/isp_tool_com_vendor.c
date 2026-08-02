/*
 * Product-build transport policy.
 *
 * The ST tuning layer is intentionally not compiled into the gauge image.
 * Its dump command sends the complete DCMIPP frame through USBX, which is the
 * exact data-plane path that produced ASCII-looking camera bytes on the human
 * UART session.  Keep this wrapper as a visible build boundary: a future
 * engineer must opt into a separate tuning image instead of re-enabling the
 * transport by changing a middleware header.
 */
#if defined(APP_ISP_TUNING_IMAGE) && (APP_ISP_TUNING_IMAGE != 0)
#include "../Middlewares/Third_Party/Camera_Middleware/ISP_Library/isp/Src/isp_tool_com.c"
#endif
