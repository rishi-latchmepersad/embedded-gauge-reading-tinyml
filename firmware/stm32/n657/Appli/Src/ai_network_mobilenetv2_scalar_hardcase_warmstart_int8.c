/* Thin build wrapper for the current production ST Edge AI model source.
 *
 * Step 1 only needs the model init entry points so App_AI_Model_Init() can
 * smoke-test the package without running inference yet.
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

#include "../../../../../st_ai_output/packages/scalar_full_finetune_from_best_piecewise_calibrated_int8/st_ai_output/scalar_full_finetune_from_best_piecewise_calibrated_int8.c"
