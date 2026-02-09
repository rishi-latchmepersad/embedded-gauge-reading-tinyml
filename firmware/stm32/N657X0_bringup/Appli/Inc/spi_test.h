#ifndef SPI_TEST_H
#define SPI_TEST_H

#include "stm32n6xx_hal.h"
#include <stdint.h>

uint8_t SPI_Test_SendCMD0_GetR1(void);
uint8_t SPI_Test_SendCMD8_ReadR7(uint8_t r7_out[4]);
uint8_t SPI_Test_SendACMD41_UntilReady(uint8_t *cmd55_r1_out);
uint8_t SPI_Test_SendCMD58_ReadOCR(uint8_t ocr_out[4]);
void SPI_Test_TryCMD55_ACMD41(uint8_t *cmd55_r1_out, uint8_t *acmd41_r1_out);
uint8_t SPI_Test_SendCMD17_ReadSingleBlock(uint32_t block_lba, uint8_t data_out[512]);
uint8_t SPI_Test_ReadBlock_CheckSignature(uint32_t block_lba, uint8_t *signature_byte0_out, uint8_t *signature_byte1_out);
uint8_t SPI_Test_FindFirstPartitionStartLba(uint32_t *partition_start_lba_out, uint8_t *partition_type_out);
uint8_t SPI_Test_ReadVolumeBootSectorSignature(uint32_t *volume_boot_sector_lba_out, uint8_t *signature_byte0_out, uint8_t *signature_byte1_out);

#endif
