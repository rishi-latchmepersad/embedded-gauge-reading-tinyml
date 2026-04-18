/* Thin wrapper around the STM32N6 Nucleo XSPI BSP driver from the installed pack.
 * This keeps the memory-mapped XSPI bring-up in sync with the vendor reference.
 *
 * Include our project-local mx25um51245g_conf.h FIRST so its include guards
 * block the pack's own conf file when stm32n6xx_nucleo_xspi.h → mx25um51245g.h
 * pulls it in.  See Appli/Inc/mx25um51245g_conf.h for the rationale. */

#include "stm32n657xx.h"
#include "stm32n6xx_hal_xspi.h"
#include "mx25um51245g_conf.h"  /* project override — must come before the pack .c */
#include "C:/Users/rishi_latchmepersad/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/Projects/STM32N6570-DK/Applications/NPU_Validation/Drivers/BSP/STM32N6xx_Nucleo/stm32n6xx_nucleo_xspi.c"
