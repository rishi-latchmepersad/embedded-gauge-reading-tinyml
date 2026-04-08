/*
 *******************************************************************************
 * @file    app_storage.h
 * @brief   Storage coordination helpers for FileX/RTC file naming.
 *******************************************************************************
 */

#ifndef __APP_STORAGE_H
#define __APP_STORAGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include "fx_api.h"
#include "tx_api.h"

UINT AppStorage_Init(void);

bool AppStorage_WaitForMediaReady(uint32_t timeout_ms);
void AppStorage_NotifyMediaReady(void);
bool AppStorage_BuildCaptureFileName(CHAR *file_name_ptr,
		ULONG file_name_length, const CHAR *file_extension_ptr);

#ifdef __cplusplus
}
#endif

#endif /* __APP_STORAGE_H */
