#ifndef SD_SPI_LL_H
#define SD_SPI_LL_H

#include "stm32n6xx_hal.h"                                   /* HAL types for SPI handle, GPIO types, etc. */
#include "fx_api.h"                                          /* FileX types for FX_MEDIA and driver signature. */
#include <stdint.h>                                          /* uint8_t/uint32_t types. */

/*==============================================================================
 * Type: Sd_FileX_DriverContext
 *
 * Purpose:
 *   Context block passed to the FileX media driver so it can translate
 *   FileX logical sectors (0..N-1) into physical SD sectors on the card.
 *
 * Fields:
 *   partition_start_lba      - First physical LBA of the FAT partition (from MBR).
 *   partition_sector_count   - Number of sectors in that partition.
 *   is_initialized           - Debug flag to confirm bringup reached a milestone.
 *==============================================================================*/
typedef struct {
	uint32_t partition_start_lba; /* Physical start sector of the FAT partition on the SD card. */
	uint32_t partition_sector_count; /* Total sectors in that partition. */
	uint8_t is_initialized; /* Debug flag for visibility. */
} Sd_FileX_DriverContext;

/* SD SPI bringup helpers */
uint8_t SPI_SendCMD0_GetR1(void);
uint8_t SPI_SendCMD8_ReadR7(uint8_t r7_out[4]);
uint8_t SPI_SendACMD41_UntilReady(uint8_t *cmd55_r1_out);
uint8_t SPI_SendCMD58_ReadOCR(uint8_t ocr_out[4]);
void SPI_Test_Run(void);

/* SD block I/O (512-byte sectors) */
uint8_t SPI_ReadSingleBlock512(uint32_t block_lba,
		uint8_t data_out_512_bytes[512]);
uint8_t SPI_WriteSingleBlock512(uint32_t block_lba,
		const uint8_t data_in_512_bytes[512]);

/* Partition parsing helper (MBR entry 0) */
uint8_t SPI_ReadPartition0Info(uint32_t *partition_start_lba_out,
		uint32_t *partition_sector_count_out);

/* FileX media driver (calls the SD block I/O above) */
VOID SPI_FileX_SdSpiMediaDriver(FX_MEDIA *media_ptr);

#endif /* SD_SPI_LL_H */
