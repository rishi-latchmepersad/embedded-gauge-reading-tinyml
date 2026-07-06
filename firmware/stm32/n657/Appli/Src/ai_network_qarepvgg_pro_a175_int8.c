/**
 * @file    ai_network_qarepvgg_pro_a175_int8.c
 * @brief   Include the Cube.AI-generated OBB face-localizer int8 model.
 *
 * STM32CubeIDE builds the generated .c directly from the package directory.
 * This thunk forwards the include with the required ATON platform macros.
 */
#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX
#ifndef NDEBUG
#define NDEBUG
#endif
#ifndef LL_ATON_DBG_BUFFER_INFO_EXCLUDED
#define LL_ATON_DBG_BUFFER_INFO_EXCLUDED 1
#endif

#include "../../st_ai_output/packages/obb_face_v2_int8_n6_npu/st_ai_output/obb_face_v2_int8.c"
