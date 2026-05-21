/* Thin build wrapper for the source-crop-box ST Edge AI model source.
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

#include "../../../../../st_ai_output/packages/mobilenetv2_source_crop_box_v1_stripped_int8/st_ai_output/mobilenetv2_source_crop_box_v1_stripped_int8.c"
