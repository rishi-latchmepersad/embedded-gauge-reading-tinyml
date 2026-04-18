/* Thin wrapper around the MX25UM51245G component driver from the installed pack.
 * The actual driver stays untouched; we only anchor it in the project build.
 *
 * We include our project-local mx25um51245g_conf.h FIRST so its include guards
 * block the pack's own conf file when mx25um51245g.h pulls it in.  This lets
 * us override DUMMY_CYCLES_READ_OCTAL (and friends) without touching vendor
 * files.  See Appli/Inc/mx25um51245g_conf.h for the rationale. */

#include "stm32n657xx.h"
#include "stm32n6xx_hal_xspi.h"
#include "mx25um51245g_conf.h"  /* project override — must come before the pack .c */
#include "C:/Users/rishi_latchmepersad/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/Projects/STM32N6570-DK/Applications/NPU_Validation/Drivers/BSP/Components/mx25um51245g/mx25um51245g.c"
