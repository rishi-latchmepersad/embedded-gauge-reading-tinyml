/* Thin build wrapper for the generated ST Edge AI model source.
 *
 * Step 1 only needs the model init entry points so App_AI_Model_Init() can
 * smoke-test the package without running inference yet.
 */

#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX

#include "../../../../../st_ai_output/packages/st_ai_ws_mnv2_scalar_warmstart/neural_art__mobilenetv2_scalar_hardcase_warmstart_int8/mobilenetv2_scalar_hardcase_warmstart_int8.c"
