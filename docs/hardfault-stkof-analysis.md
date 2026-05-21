## 2026-05-19 HardFault STKOF Root Cause Analysis

### Fault Details

**First boot:**
- HardFault PC=0x340065F8 LR=0x34007553 CFSR=0x00100000 HFSR=0x40000000
- CFSR=0x00100000 = UFSR bit 4 (STKOF, stack overflow on ARMv8-M)
- HFSR=0x40000000 = FORCED bit (UsageFault escalated to HardFault)
- SP=0x340FAA68 is below MSPLIM=0x340FC000

**Second boot (latched):**
- MemManage PC=0xAAAAAAAA CFSR=0x00000001 (IACCVIOL from erased-flash pattern, cascading fault)

### Root Cause

The fault is an **MSP (main stack) overflow**. Key evidence:

1. SP=0x340FAA68 is 5528 bytes below MSPLIM=0x340FC000. The 16KB MSP stack is insufficient.
2. PC=0x340065F8 is inside LL_RCC_HSI_SetDivider, called from HAL_RCC_ClockConfig.
3. The main() init path through App_SystemClock_Config() with its large RCC_OscInitTypeDef (148B) and RCC_ClkInitTypeDef (64B) plus deep HAL call chains exceeds 16KB.
4. Second fault at PC=0xAAAAAAAA is a cascading failure after STKOF corrupted MSP.

### Fix

1. Increase _Min_Stack_Size in linker script from 0x4000 (16KB) to at least 0x8000 (32KB).
2. Add EXC_LR and MSPLIM/PSPLIM diagnostics to HardFault dump.
3. Consider moving heavy init to a ThreadX thread with a dedicated large stack.
