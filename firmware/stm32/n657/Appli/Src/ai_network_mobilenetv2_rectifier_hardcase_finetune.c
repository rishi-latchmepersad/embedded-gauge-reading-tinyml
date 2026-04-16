/* Thin build wrapper for the board rectifier ST Edge AI model source.
 *
 * The firmware links this wrapper so it can initialize and run the rectifier
 * stage before switching to the scalar reader stage.
 */

#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX

#include "../../../../../st_ai_output/packages/mobilenetv2_rectifier_hardcase_finetune_v3/st_ai_output/mobilenetv2_rectifier_hardcase_finetune.c"
