/* Thin build wrapper for the current production ST Edge AI model source.
 *
 * Step 1 only needs the model init entry points so App_AI_Model_Init() can
 * smoke-test the package without running inference yet.
 */

#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX

#include "../../../../../st_ai_output/packages/scalar_full_finetune_from_best_piecewise_calibrated_int8/st_ai_output/scalar_full_finetune_from_best_piecewise_calibrated_int8.c"
