/* Compatibility shim for the v18 tip-focus relocatable model configuration.
 *
 * The generated CubeIDE build still passes the old MODEL_CONF define on the
 * command line, so the wrapper overrides it to this header name. We keep the
 * real generated config under the package directory and include it from here so
 * the existing include search path can resolve the model metadata without
 * touching any generated build files. */

#include "../../st_ai_output/packages/tip_focus_v18_int8_n6_npu/st_ai_ws/build_tip_focus_v18_int8/tip_focus_v18_int8_reloc_conf.h"
