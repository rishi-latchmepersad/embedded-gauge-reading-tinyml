/* Thin build wrapper for the board rectifier ST Edge AI model source.
 *
 * The firmware links this wrapper so it can initialize and run the rectifier
 * stage before switching to the scalar reader stage.
 */

#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX
/* Keep the generated ST AI model out of debug-buffer and assert-heavy paths. */
#ifndef NDEBUG
#define NDEBUG
#endif
#ifndef LL_ATON_DBG_BUFFER_INFO_EXCLUDED
#define LL_ATON_DBG_BUFFER_INFO_EXCLUDED 1
#endif

#include "../../../../../st_ai_output/packages/mobilenetv2_rectifier_zoom_aug_v4/st_ai_output/mobilenetv2_rectifier_hardcase_finetune.c"
