/**
 * @file app_ai_aton_cache.c
 * @brief Shared cache callbacks required by relocatable ATON networks.
 *
 * Generated relocatable model code references these callbacks when
 * BUILD_AI_NETWORK_RELOC is enabled. They are shared runtime services, not
 * model-specific behavior, so they live outside the model wrappers.
 */

#include "mcu_cache.h"
#include "npu_cache.h"

/** Clean a range from the MCU-side cache before NPU access. */
void LL_ATON_Cache_MCU_Clean_Range(uintptr_t virtual_addr, uint32_t size)
{
	(void)mcu_cache_clean_range((uint32_t)virtual_addr,
		(uint32_t)(virtual_addr + size));
}

/** Invalidate a range from the MCU-side cache after NPU access. */
void LL_ATON_Cache_MCU_Invalidate_Range(uintptr_t virtual_addr, uint32_t size)
{
	(void)mcu_cache_invalidate_range((uint32_t)virtual_addr,
		(uint32_t)(virtual_addr + size));
}

/** Clean and invalidate an MCU-side cache range. */
void LL_ATON_Cache_MCU_Clean_Invalidate_Range(uintptr_t virtual_addr,
	uint32_t size)
{
	(void)mcu_cache_clean_invalidate_range((uint32_t)virtual_addr,
		(uint32_t)(virtual_addr + size));
}

/** Clean a range from the NPU-side cache before reuse. */
void LL_ATON_Cache_NPU_Clean_Range(uintptr_t virtual_addr, uint32_t size)
{
	npu_cache_clean_range((uint32_t)virtual_addr,
		(uint32_t)(virtual_addr + size));
}

/** Clean and invalidate an NPU-side cache range. */
void LL_ATON_Cache_NPU_Clean_Invalidate_Range(uintptr_t virtual_addr,
	uint32_t size)
{
	npu_cache_clean_invalidate_range((uint32_t)virtual_addr,
		(uint32_t)(virtual_addr + size));
}

/** Invalidate the complete NPU cache. */
void LL_ATON_Cache_NPU_Invalidate(void)
{
	npu_cache_invalidate();
}
