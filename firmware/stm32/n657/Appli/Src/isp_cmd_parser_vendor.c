/*
 * The product image has no host tuning command channel.  In particular, do
 * not compile the parser's preview/ISP/raw-frame dump commands into the
 * firmware that shares LPUART1 with the human-readable console.  A dedicated
 * tuning image can define APP_ISP_TUNING_IMAGE and rebuild this wrapper.
 */
#if defined(APP_ISP_TUNING_IMAGE) && (APP_ISP_TUNING_IMAGE != 0)
#include "../Middlewares/Third_Party/Camera_Middleware/ISP_Library/isp/Src/isp_cmd_parser.c"
#endif
