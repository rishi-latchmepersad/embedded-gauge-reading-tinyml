#ifndef SD_SPI_PROTOCOL_H
#define SD_SPI_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

/*==============================================================================
 * Type: SdSpiProtocol_TransferByteFunction
 *
 * Purpose:
 *   Abstract the SPI byte exchange so protocol code can run on host or target.
 *
 * Parameters:
 *   transfer_context - Opaque pointer owned by the caller (may be NULL).
 *   transmit_byte    - Byte to clock out on MOSI.
 *
 * Returns:
 *   Byte sampled on MISO during the same clock edges.
 *==============================================================================*/
typedef uint8_t (*SdSpiProtocol_TransferByteFunction)(void *transfer_context,
		uint8_t transmit_byte);

/*==============================================================================
 * Type: SdSpiProtocol_DataTokenWaitStatus
 *
 * Purpose:
 *   Report outcome of waiting for an SD data token (for example 0xFE on CMD17).
 *==============================================================================*/
typedef enum {
	SD_SPI_PROTOCOL_DATA_TOKEN_WAIT_STATUS_OK = 0,
	SD_SPI_PROTOCOL_DATA_TOKEN_WAIT_STATUS_TIMEOUT = 1,
	SD_SPI_PROTOCOL_DATA_TOKEN_WAIT_STATUS_UNEXPECTED_TOKEN = 2,
	SD_SPI_PROTOCOL_DATA_TOKEN_WAIT_STATUS_NULL_ARGUMENT = 3
} SdSpiProtocol_DataTokenWaitStatus;

/*==============================================================================
 * Function: SdSpiProtocol_ComputeCrc7ForCommandPacket
 *
 * Purpose:
 *   Compute SD CRC7 over the first 5 bytes of a command packet.
 *
 * Parameters:
 *   command_packet_five_bytes - 5 bytes: [0]=0x40|cmd, [1..4]=arg.
 *
 * Returns:
 *   CRC byte formatted for SD commands: (crc7 << 1) | 1.
 *==============================================================================*/
uint8_t SdSpiProtocol_ComputeCrc7ForCommandPacket(
		const uint8_t command_packet_five_bytes[5]);

/*==============================================================================
 * Function: SdSpiProtocol_BuildCommandFrame
 *
 * Purpose:
 *   Build a full 6-byte SD SPI command frame.
 *
 * Parameters:
 *   command_index                   - SD command number (0..63).
 *   argument                        - 32-bit argument.
 *   crc7_with_end_bit_or_zero_auto  - If 0, compute CRC automatically.
 *                                    Otherwise transmit this CRC byte as-is.
 *   command_frame_six_bytes_out     - Output buffer for 6 bytes.
 *
 * Returns:
 *   None.
 *==============================================================================*/
void SdSpiProtocol_BuildCommandFrame(uint8_t command_index, uint32_t argument,
		uint8_t crc7_with_end_bit_or_zero_auto,
		uint8_t command_frame_six_bytes_out[6]);

/*==============================================================================
 * Function: SdSpiProtocol_SendCommandAndGetR1
 *
 * Purpose:
 *   Send a 6-byte command frame then poll for the first non-0xFF R1 response.
 *
 * Parameters:
 *   transfer_byte_function          - Byte exchange function.
 *   transfer_context                - Opaque context passed to transfer function.
 *   command_index                   - SD command number (0..63).
 *   argument                        - 32-bit argument.
 *   crc7_with_end_bit_or_zero_auto  - If 0, compute CRC automatically.
 *   max_response_poll_bytes         - How many bytes to poll for R1.
 *
 * Returns:
 *   R1 byte on success, or 0xFF if no response observed.
 *==============================================================================*/
uint8_t SdSpiProtocol_SendCommandAndGetR1(
		SdSpiProtocol_TransferByteFunction transfer_byte_function,
		void *transfer_context, uint8_t command_index, uint32_t argument,
		uint8_t crc7_with_end_bit_or_zero_auto,
		uint32_t max_response_poll_bytes);

/*==============================================================================
 * Function: SdSpiProtocol_ReadResponseBytes
 *
 * Purpose:
 *   Read N response bytes by clocking 0xFF bytes.
 *
 * Parameters:
 *   transfer_byte_function - Byte exchange function.
 *   transfer_context       - Opaque context passed to transfer function.
 *   response_buffer        - Output buffer.
 *   response_length_bytes  - Number of bytes to read.
 *
 * Returns:
 *   None.
 *==============================================================================*/
void SdSpiProtocol_ReadResponseBytes(
		SdSpiProtocol_TransferByteFunction transfer_byte_function,
		void *transfer_context, uint8_t *response_buffer,
		uint32_t response_length_bytes);

/*==============================================================================
 * Function: SdSpiProtocol_ParseIsHighCapacityCardFromOcr
 *
 * Purpose:
 *   Parse OCR and return whether the card uses block addressing (SDHC/SDXC).
 *
 * Parameters:
 *   ocr_bytes - 4 OCR bytes returned by CMD58, MSB first.
 *
 * Returns:
 *   1 if SDHC/SDXC (CCS bit set), otherwise 0.
 *==============================================================================*/
uint8_t SdSpiProtocol_ParseIsHighCapacityCardFromOcr(const uint8_t ocr_bytes[4]);

/*==============================================================================
 * Function: SdSpiProtocol_ComputeCmd17Cmd24ArgumentFromBlockLba
 *
 * Purpose:
 *   Convert a logical block index to the argument used by CMD17/CMD24.
 *
 * Parameters:
 *   block_lba               - Logical block index (512-byte sector).
 *   is_high_capacity_card   - 1 for SDHC/SDXC, 0 for SDSC.
 *
 * Returns:
 *   Argument for CMD17/CMD24.
 *==============================================================================*/
uint32_t SdSpiProtocol_ComputeCmd17Cmd24ArgumentFromBlockLba(uint32_t block_lba,
		uint8_t is_high_capacity_card);

/*==============================================================================
 * Function: SdSpiProtocol_WaitForDataToken
 *
 * Purpose:
 *   Poll the bus until a non-0xFF token arrives, then verify it matches expected.
 *
 * Parameters:
 *   transfer_byte_function  - Byte exchange function.
 *   transfer_context        - Opaque context passed to transfer function.
 *   expected_token          - Token byte expected (for CMD17, 0xFE).
 *   max_poll_bytes          - Max bytes to poll.
 *   observed_token_out      - Optional output of the observed token.
 *
 * Returns:
 *   SdSpiProtocol_DataTokenWaitStatus.
 *==============================================================================*/
SdSpiProtocol_DataTokenWaitStatus SdSpiProtocol_WaitForDataToken(
		SdSpiProtocol_TransferByteFunction transfer_byte_function,
		void *transfer_context, uint8_t expected_token, uint32_t max_poll_bytes,
		uint8_t *observed_token_out);

#endif /* SD_SPI_PROTOCOL_H */
