#include "sd_spi_protocol.h"

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
		const uint8_t command_packet_five_bytes[5]) {
	uint8_t crc7 = 0U;

	if (command_packet_five_bytes == NULL) {
		return 0xFFU;
	}

	for (uint32_t byte_index = 0U; byte_index < 5U; byte_index++) {
		uint8_t data = command_packet_five_bytes[byte_index];

		for (uint32_t bit_index = 0U; bit_index < 8U; bit_index++) {
			crc7 <<= 1U;

			if (((data & 0x80U) ^ (crc7 & 0x80U)) != 0U) {
				crc7 ^= 0x09U;
			}

			data <<= 1U;
		}
	}

	return (uint8_t) ((crc7 << 1U) | 0x01U);
}

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
 *   command_frame_six_bytes_out     - Output buffer for 6 bytes.
 *
 * Returns:
 *   None.
 *==============================================================================*/
void SdSpiProtocol_BuildCommandFrame(uint8_t command_index, uint32_t argument,
		uint8_t crc7_with_end_bit_or_zero_auto,
		uint8_t command_frame_six_bytes_out[6]) {
	uint8_t command_packet_five_bytes[5];

	if (command_frame_six_bytes_out == NULL) {
		return;
	}

	command_packet_five_bytes[0] = (uint8_t) (0x40U | (command_index & 0x3FU));
	command_packet_five_bytes[1] = (uint8_t) ((argument >> 24) & 0xFFU);
	command_packet_five_bytes[2] = (uint8_t) ((argument >> 16) & 0xFFU);
	command_packet_five_bytes[3] = (uint8_t) ((argument >> 8) & 0xFFU);
	command_packet_five_bytes[4] = (uint8_t) (argument & 0xFFU);

	if (crc7_with_end_bit_or_zero_auto == 0x00U) {
		crc7_with_end_bit_or_zero_auto =
				SdSpiProtocol_ComputeCrc7ForCommandPacket(
						command_packet_five_bytes);
	}

	command_frame_six_bytes_out[0] = command_packet_five_bytes[0];
	command_frame_six_bytes_out[1] = command_packet_five_bytes[1];
	command_frame_six_bytes_out[2] = command_packet_five_bytes[2];
	command_frame_six_bytes_out[3] = command_packet_five_bytes[3];
	command_frame_six_bytes_out[4] = command_packet_five_bytes[4];
	command_frame_six_bytes_out[5] = crc7_with_end_bit_or_zero_auto;
}

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
		uint32_t max_response_poll_bytes) {
	uint8_t command_frame_six_bytes[6];

	if (transfer_byte_function == NULL) {
		return 0xFFU;
	}

	SdSpiProtocol_BuildCommandFrame(command_index, argument,
			crc7_with_end_bit_or_zero_auto, command_frame_six_bytes);

	(void) transfer_byte_function(transfer_context, command_frame_six_bytes[0]);
	(void) transfer_byte_function(transfer_context, command_frame_six_bytes[1]);
	(void) transfer_byte_function(transfer_context, command_frame_six_bytes[2]);
	(void) transfer_byte_function(transfer_context, command_frame_six_bytes[3]);
	(void) transfer_byte_function(transfer_context, command_frame_six_bytes[4]);
	(void) transfer_byte_function(transfer_context, command_frame_six_bytes[5]);

	for (uint32_t attempt = 0U; attempt < max_response_poll_bytes; attempt++) {
		uint8_t r1 = transfer_byte_function(transfer_context, 0xFFU);
		if (r1 != 0xFFU) {
			return r1;
		}
	}

	return 0xFFU;
}

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
		uint32_t response_length_bytes) {
	if ((transfer_byte_function == NULL) || (response_buffer == NULL)) {
		return;
	}

	for (uint32_t i = 0U; i < response_length_bytes; i++) {
		response_buffer[i] = transfer_byte_function(transfer_context, 0xFFU);
	}
}

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
uint8_t SdSpiProtocol_ParseIsHighCapacityCardFromOcr(const uint8_t ocr_bytes[4]) {
	if (ocr_bytes == NULL) {
		return 0U;
	}

	return (uint8_t) ((ocr_bytes[0] & 0x40U) ? 1U : 0U);
}

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
		uint8_t is_high_capacity_card) {
	if (is_high_capacity_card != 0U) {
		return block_lba;
	}

	return (uint32_t) (block_lba * 512U);
}

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
		uint8_t *observed_token_out) {
	if (transfer_byte_function == NULL) {
		return SD_SPI_PROTOCOL_DATA_TOKEN_WAIT_STATUS_NULL_ARGUMENT;
	}

	for (uint32_t attempt = 0U; attempt < max_poll_bytes; attempt++) {
		uint8_t token = transfer_byte_function(transfer_context, 0xFFU);

		if (token == 0xFFU) {
			continue;
		}

		if (observed_token_out != NULL) {
			*observed_token_out = token;
		}

		if (token == expected_token) {
			return SD_SPI_PROTOCOL_DATA_TOKEN_WAIT_STATUS_OK;
		}

		return SD_SPI_PROTOCOL_DATA_TOKEN_WAIT_STATUS_UNEXPECTED_TOKEN;
	}

	return SD_SPI_PROTOCOL_DATA_TOKEN_WAIT_STATUS_TIMEOUT;
}
