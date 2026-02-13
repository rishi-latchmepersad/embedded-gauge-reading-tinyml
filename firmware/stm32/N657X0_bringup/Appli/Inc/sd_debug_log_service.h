#ifndef SD_DEBUG_LOG_SERVICE_H
#define SD_DEBUG_LOG_SERVICE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "fx_api.h"
#include "tx_api.h"
#include "sd_debug_log_core.h"
#include <stdint.h>

#define SD_DEBUG_LOG_SERVICE_MAX_LINE_LENGTH_BYTES          (256U)
#define SD_DEBUG_LOG_SERVICE_QUEUE_DEPTH                    (64U)
#define SD_DEBUG_LOG_SERVICE_ROLLOVER_THRESHOLD_BYTES       (5UL * 1024UL * 1024UL)

/*==============================================================================
 * Function: SdDebugLogService_Initialize
 *
 * Purpose:
 *   Initialize the logging queue and core, bind to a mounted FX_MEDIA instance.
 *
 * Parameters:
 *   byte_pool_ptr - ThreadX byte pool used to allocate queue storage.
 *   media_ptr     - Mounted FileX media instance (SD card).
 *
 * Returns:
 *   TX_SUCCESS on success, otherwise a ThreadX error code.
 *==============================================================================*/
UINT SdDebugLogService_Initialize(TX_BYTE_POOL *byte_pool_ptr,
		FX_MEDIA *media_ptr);

/*==============================================================================
 * Function: SdDebugLogService_EnqueueLine
 *
 * Purpose:
 *   Enqueue a log line for the FileX thread to write to debug.log.
 *
 * Parameters:
 *   line_ptr - Null terminated C string.
 *
 * Returns:
 *   TX_SUCCESS on success, otherwise a ThreadX error code.
 *==============================================================================*/
UINT SdDebugLogService_EnqueueLine(const CHAR *line_ptr);

/*==============================================================================
 * Function: SdDebugLogService_ServiceQueue
 *
 * Purpose:
 *   Drain up to max_messages_to_process messages from the queue and write them.
 *
 * Parameters:
 *   max_messages_to_process - Maximum messages to process this call.
 *
 * Returns:
 *   None.
 *==============================================================================*/
void SdDebugLogService_ServiceQueue(ULONG max_messages_to_process);

/*==============================================================================
 * Function: SdDebugLogService_ForceFlush
 *
 * Purpose:
 *   Force flush and close of the active file.
 *
 * Returns:
 *   None.
 *==============================================================================*/
void SdDebugLogService_ForceFlush(void);

#ifdef __cplusplus
}
#endif

#endif /* SD_DEBUG_LOG_SERVICE_H */
