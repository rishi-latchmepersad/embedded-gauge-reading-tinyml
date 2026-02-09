/*
 * spi_test.c
 *
 * Purpose:
 *   Simple SD card SPI bring-up helpers for STM32, using HAL SPI and a GPIO chip select.
 *
 * Notes:
 *   - This is intentionally "bring-up" style code, not a production SD driver.
 *   - Init sequence supported: CMD0, CMD8, ACMD41 loop, CMD58 OCR read.
 *   - Designed for SPI Mode 0, low speed during init (<= 400 kHz), higher after init.
 */

#include "spi_test.h"                         // Include matching header for prototypes and types.
#include "main.h"                             // Include main for SPI5_CS_GPIO_Port / SPI5_CS_Pin definitions.

extern SPI_HandleTypeDef hspi5;               // Use CubeMX-generated SPI handle, provided by STM32 HAL startup code.

/* CMD17 (READ_SINGLE_BLOCK) data token for a valid data block. */
#define SD_SPI_DATA_START_TOKEN_SINGLE_BLOCK_READ   (0xFEU)   /* 0xFE indicates the next 512 bytes are data. */
#define SD_SPI_CMD17_STATUS_OK                       (0x00U)  /* Success. */
#define SD_SPI_CMD17_STATUS_NULL_BUFFER              (0xE0U)  /* Caller passed NULL pointer. */
#define SD_SPI_CMD17_STATUS_CMD_R1_NOT_READY         (0xE1U)  /* CMD17 returned non-zero R1. */
#define SD_SPI_CMD17_STATUS_TOKEN_TIMEOUT            (0xE2U)  /* Timed out waiting for 0xFE token. */
#define SD_SPI_CMD17_STATUS_UNEXPECTED_TOKEN         (0xE3U)  /* Received non-0xFF token that is not 0xFE. */


/*==============================================================================
 * File-scope shared sector buffer.
 * Purpose:
 *   Avoid stack usage by using one reusable 512-byte buffer for SD block reads.
 *==============================================================================*/
static uint8_t g_sd_test_sector_buffer[512];                                 // Stored in .bss, not on stack, safe for small stack projects.

/*==============================================================================
 * Function: SD_Select
 *
 * Purpose:
 *   Assert the SD card chip select line (active low) to start an SPI transaction.
 *
 * Parameters:
 *   None.
 *
 * Returns:
 *   None.
 *
 * Side Effects:
 *   Drives the GPIO CS pin low.
 *
 * Preconditions:
 *   - CS pin configured as a push-pull output.
 *
 * Concurrency:
 *   Not thread-safe, caller must serialize SPI access.
 *
 * Timing:
 *   GPIO write is immediate.
 *
 * Errors:
 *   None.
 *
 * Notes:
 *   SD cards sample commands only when CS is low in SPI mode.
 *==============================================================================*/
static void SD_Select(void)
{
    HAL_GPIO_WritePin(SPI5_CS_GPIO_Port, SPI5_CS_Pin, GPIO_PIN_RESET);  // Pull CS low to select the SD card.
}

/*==============================================================================
 * Function: SD_Deselect
 *
 * Purpose:
 *   Deassert the SD card chip select line to end an SPI transaction.
 *
 * Parameters:
 *   None.
 *
 * Returns:
 *   None.
 *
 * Side Effects:
 *   Drives the GPIO CS pin high.
 *
 * Preconditions:
 *   - CS pin configured as a push-pull output.
 *
 * Concurrency:
 *   Not thread-safe, caller must serialize SPI access.
 *
 * Timing:
 *   GPIO write is immediate.
 *
 * Errors:
 *   None.
 *
 * Notes:
 *   SD cards release MISO and internally finalize some operations when CS goes high.
 *==============================================================================*/
static void SD_Deselect(void)
{
    HAL_GPIO_WritePin(SPI5_CS_GPIO_Port, SPI5_CS_Pin, GPIO_PIN_SET);    // Pull CS high to deselect the SD card.
}

/*==============================================================================
 * Function: SD_SPI_TransferByte
 *
 * Purpose:
 *   Transfer one byte over SPI and return the simultaneously received byte.
 *
 * Parameters:
 *   transmit_byte - The byte to clock out on MOSI.
 *
 * Returns:
 *   The byte sampled on MISO during the same 8 SCK edges.
 *
 * Side Effects:
 *   Generates SPI clock edges and toggles MOSI.
 *
 * Preconditions:
 *   - hspi5 initialized and configured for SPI mode required by SD card (Mode 0).
 *
 * Concurrency:
 *   Not thread-safe, caller must serialize SPI access.
 *
 * Timing:
 *   Blocks until transfer completes (HAL_MAX_DELAY).
 *
 * Errors:
 *   HAL status is ignored here, because this is bring-up code; production code should check return status.
 *==============================================================================*/
static uint8_t SD_SPI_TransferByte(uint8_t transmit_byte)
{
    uint8_t receive_byte = 0xFFU;                                         // Default to 0xFF so failures are obvious in debug.
    (void)HAL_SPI_TransmitReceive(&hspi5, &transmit_byte, &receive_byte, 1U, HAL_MAX_DELAY); // Clock one byte, get one byte.
    return receive_byte;                                                   // Return the byte we sampled on MISO.
}

/*==============================================================================
 * Function: SD_SendIdleClocks
 *
 * Purpose:
 *   Provide idle clocks (sending 0xFF) so the SD card can complete internal timing and respond.
 *
 * Parameters:
 *   byte_count - Number of 8-bit clock bursts to generate.
 *
 * Returns:
 *   None.
 *
 * Side Effects:
 *   Generates SPI clocks with MOSI high.
 *
 * Preconditions:
 *   - SPI initialized.
 *
 * Concurrency:
 *   Not thread-safe.
 *
 * Timing:
 *   Runs in O(byte_count).
 *
 * Errors:
 *   None.
 *
 * Notes:
 *   During SPI mode, the host often sends 0xFF while waiting for responses or allowing Ncr timing.
 *==============================================================================*/
static void SD_SendIdleClocks(uint32_t byte_count)
{
    for (uint32_t i = 0U; i < byte_count; i++)                             // Loop for the requested number of bytes.
    {
        (void)SD_SPI_TransferByte(0xFFU);                                   // Send 0xFF to keep MOSI high and provide clocks.
    }
}

/*==============================================================================
 * Function: SD_ComputeCrc7_ForCommandPacket
 *
 * Purpose:
 *   Compute CRC7 for an SD command packet over the first 5 bytes (CMD + 4 arg bytes).
 *
 * Parameters:
 *   command_packet_five_bytes - Pointer to exactly 5 bytes: [0]=0x40|cmd, [1..4]=arg.
 *
 * Returns:
 *   CRC byte with end bit set, (crc7 << 1) | 1.
 *
 * Side Effects:
 *   None.
 *
 * Preconditions:
 *   - command_packet_five_bytes is not NULL and points to at least 5 bytes.
 *
 * Concurrency:
 *   Thread-safe.
 *
 * Timing:
 *   5 * 8 iterations, deterministic.
 *
 * Errors:
 *   None.
 *
 * Notes:
 *   CRC is required for CMD0 and CMD8 in SPI mode; after that it is typically ignored unless CRC is enabled.
 *==============================================================================*/
static uint8_t SD_ComputeCrc7_ForCommandPacket(const uint8_t command_packet_five_bytes[5])
{
    uint8_t crc7 = 0U;                                                      // Start CRC accumulator at zero.

    for (uint32_t byte_index = 0U; byte_index < 5U; byte_index++)           // Process 5 bytes of the command packet.
    {
        uint8_t data = command_packet_five_bytes[byte_index];               // Copy current byte so we can shift it bit-by-bit.

        for (uint32_t bit_index = 0U; bit_index < 8U; bit_index++)          // Process each bit, MSB first.
        {
            crc7 <<= 1U;                                                    // Shift CRC left to make room for the next bit step.

            if (((data & 0x80U) ^ (crc7 & 0x80U)) != 0U)                    // If data MSB differs from CRC MSB, apply polynomial.
            {
                crc7 ^= 0x09U;                                              // Apply CRC7 polynomial (x^7 + x^3 + 1), represented as 0x09.
            }

            data <<= 1U;                                                    // Shift data to bring next bit into MSB position.
        }
    }

    return (uint8_t)((crc7 << 1U) | 0x01U);                                  // Append end bit '1' as required by SD command format.
}

/*==============================================================================
 * Function: SD_SendCommand
 *
 * Purpose:
 *   Send a standard SD SPI command (6-byte frame) and return the R1 response byte.
 *
 * Parameters:
 *   command_index      - SD command number (0..63).
 *   argument           - 32-bit command argument.
 *   crc7_with_end_bit  - CRC byte to transmit; if 0, compute CRC7 automatically.
 *
 * Returns:
 *   R1 response byte, or 0xFF if no response observed within timeout.
 *
 * Side Effects:
 *   Clocks SPI bus, transmits command, reads from MISO.
 *
 * Preconditions:
 *   - CS must already be asserted (low) before calling this function.
 *   - SPI configured for Mode 0 for SD cards.
 *
 * Concurrency:
 *   Not thread-safe.
 *
 * Timing:
 *   Wait loop limited to 100 bytes for response.
 *
 * Errors:
 *   Returns 0xFF on timeout waiting for R1.
 *
 * Notes:
 *   In SPI mode, the card responds with R1 where MSB is 0 when valid. Many implementations treat 0xFF as "no response yet".
 *==============================================================================*/
static uint8_t SD_SendCommand(uint8_t command_index, uint32_t argument, uint8_t crc7_with_end_bit)
{
    uint8_t command_packet[5];                                              // Build the 5-byte command body used for CRC and transmit.

    command_packet[0] = (uint8_t)(0x40U | (command_index & 0x3FU));          // Compose command byte with start bits and command index.
    command_packet[1] = (uint8_t)((argument >> 24) & 0xFFU);                // Argument byte 3, MSB first, per SD spec.
    command_packet[2] = (uint8_t)((argument >> 16) & 0xFFU);                // Argument byte 2.
    command_packet[3] = (uint8_t)((argument >> 8) & 0xFFU);                 // Argument byte 1.
    command_packet[4] = (uint8_t)(argument & 0xFFU);                        // Argument byte 0, LSB.

    if (crc7_with_end_bit == 0x00U)                                         // If caller requested auto CRC, compute it.
    {
        crc7_with_end_bit = SD_ComputeCrc7_ForCommandPacket(command_packet); // Compute CRC7 and set end bit for the final CRC byte.
    }

    (void)SD_SPI_TransferByte(command_packet[0]);                            // Send command byte so the card knows which command this is.
    (void)SD_SPI_TransferByte(command_packet[1]);                            // Send argument MSB so card interprets full 32-bit argument.
    (void)SD_SPI_TransferByte(command_packet[2]);                            // Send next argument byte.
    (void)SD_SPI_TransferByte(command_packet[3]);                            // Send next argument byte.
    (void)SD_SPI_TransferByte(command_packet[4]);                            // Send argument LSB to complete the 32-bit argument.
    (void)SD_SPI_TransferByte(crc7_with_end_bit);                            // Send CRC byte, required for CMD0/CMD8, ignored later normally.

    for (uint32_t attempt = 0U; attempt < 100U; attempt++)                   // Poll for a response for a bounded number of bytes.
    {
        uint8_t r1 = SD_SPI_TransferByte(0xFFU);                             // Clock one byte while sampling response on MISO.
        if (r1 != 0xFFU)                                                     // First non-0xFF indicates the card started responding.
        {
            return r1;                                                       // Return the R1 response byte to the caller.
        }
    }

    return 0xFFU;                                                            // Return timeout marker if we never saw a response byte.
}

/*==============================================================================
 * Function: SD_ReadResponseBytes
 *
 * Purpose:
 *   Read a fixed number of bytes from the card by clocking 0xFF.
 *
 * Parameters:
 *   response_buffer - Destination buffer for received bytes.
 *   response_length - Number of bytes to read into the buffer.
 *
 * Returns:
 *   None.
 *
 * Side Effects:
 *   Clocks SPI bus and reads MISO.
 *
 * Preconditions:
 *   - CS should be asserted when reading response bytes that belong to a command.
 *   - response_buffer is not NULL and has at least response_length bytes.
 *
 * Concurrency:
 *   Not thread-safe.
 *
 * Timing:
 *   O(response_length).
 *
 * Errors:
 *   None.
 *==============================================================================*/
static void SD_ReadResponseBytes(uint8_t *response_buffer, uint32_t response_length)
{
    for (uint32_t i = 0U; i < response_length; i++)                          // Read exactly the requested number of bytes.
    {
        response_buffer[i] = SD_SPI_TransferByte(0xFFU);                      // Send 0xFF to provide clocks and sample response byte.
    }
}

/*==============================================================================
 * Function: SPI_Test_Run
 *
 * Purpose:
 *   Provide the initial idle clocks with CS high, a common first step when entering SD SPI mode.
 *
 * Parameters:
 *   None.
 *
 * Returns:
 *   None.
 *
 * Side Effects:
 *   Clocks the SPI bus with MOSI high while CS is deasserted.
 *
 * Preconditions:
 *   - SPI initialized at low speed suitable for SD init.
 *
 * Concurrency:
 *   Not thread-safe.
 *
 * Timing:
 *   Small fixed delay plus fixed number of clocks.
 *
 * Errors:
 *   None.
 *
 * Notes:
 *   Some cards need a short delay after power is applied before they respond predictably.
 *==============================================================================*/
void SPI_Test_Run(void)
{
    SD_Deselect();                                                           // Ensure CS is high so the card treats clocks as idle clocks.
    HAL_Delay(10U);                                                          // Wait briefly so power and internal card state can settle.

    SD_SendIdleClocks(10U);                                                  // Provide 80 clocks (10 bytes), the spec minimum is 74.
    (void)SD_SPI_TransferByte(0xFFU);                                        // Send one extra byte for additional margin during bring-up.
}

/*==============================================================================
 * Function: SPI_Test_SendCMD0_GetR1
 *
 * Purpose:
 *   Send CMD0 (GO_IDLE_STATE) to force the card into SPI mode and IDLE state.
 *
 * Parameters:
 *   None.
 *
 * Returns:
 *   R1 response byte, expected 0x01 when card enters IDLE.
 *
 * Side Effects:
 *   Controls CS, clocks bus, and sends the CMD0 packet.
 *
 * Preconditions:
 *   - SPI configured Mode 0, low speed (<= 400 kHz).
 *   - SD card powered and wired correctly.
 *
 * Concurrency:
 *   Not thread-safe.
 *
 * Timing:
 *   Includes a small delay and bounded response wait.
 *
 * Errors:
 *   Returns 0xFF if no response observed.
 *
 * Notes:
 *   CMD0 requires a valid CRC in SPI mode, the standard CRC byte is 0x95.
 *==============================================================================*/
uint8_t SPI_Test_SendCMD0_GetR1(void)
{
    uint8_t r1 = 0xFFU;                                                      // Initialize with timeout value, will be replaced on success.

    SD_Deselect();                                                           // Keep CS high so we can provide idle clocks correctly.
    HAL_Delay(5U);                                                           // Give the card a little time before attempting CMD0.

    SD_SendIdleClocks(10U);                                                  // Provide 80 idle clocks with CS high to enter SPI mode.
    SD_Select();                                                             // Assert CS low to start the CMD0 transaction.
    SD_SendIdleClocks(1U);                                                   // Provide one byte gap (Ncr) before sending the command.

    r1 = SD_SendCommand(0U, 0U, 0x95U);                                      // Send CMD0 with the required CRC for SPI mode.

    SD_Deselect();                                                           // End the transaction so card can finalize its internal state.
    SD_SendIdleClocks(1U);                                                   // Provide trailing clocks after deselect per common practice.

    return r1;                                                               // Return the R1 byte so caller can confirm 0x01 IDLE.
}

/*==============================================================================
 * Function: SPI_Test_SendCMD8_ReadR7
 *
 * Purpose:
 *   Send CMD8 (SEND_IF_COND) to determine SD v2 capability and read the 4-byte R7 payload.
 *
 * Parameters:
 *   r7_out - Destination buffer of 4 bytes for R7 data.
 *
 * Returns:
 *   R1 response byte. Expected:
 *     - 0x01 for SD v2 cards in IDLE, with R7 payload present.
 *     - 0x05 for older cards that treat CMD8 as illegal.
 *
 * Side Effects:
 *   Controls CS, clocks bus, and reads response bytes.
 *
 * Preconditions:
 *   - Card is in IDLE (CMD0 done successfully).
 *   - SPI Mode 0 and low init speed.
 *   - r7_out is not NULL.
 *
 * Concurrency:
 *   Not thread-safe.
 *
 * Timing:
 *   Bounded response wait, plus 4-byte read on success.
 *
 * Errors:
 *   Returns 0xFF on timeout waiting for R1.
 *
 * Notes:
 *   CMD8 requires valid CRC in SPI mode, standard CRC is 0x87 for arg 0x000001AA.
 *==============================================================================*/
uint8_t SPI_Test_SendCMD8_ReadR7(uint8_t r7_out[4])
{
    uint8_t r1 = 0xFFU;                                                      // Default to timeout marker until we receive a response.

    if (r7_out == NULL)                                                      // Validate pointer so we do not crash on NULL.
    {
        return 0xFFU;                                                        // Return timeout marker for invalid input in this bring-up code.
    }

    SD_Select();                                                             // Select the card so it will accept the command.
    SD_SendIdleClocks(1U);                                                   // Provide a small gap before command bytes.

    r1 = SD_SendCommand(8U, 0x000001AAU, 0x87U);                             // Send CMD8 with arg=1AA, CRC=0x87 for SPI mode.
    if (r1 != 0xFFU)                                                         // Only try to read payload if the card responded at all.
    {
        SD_ReadResponseBytes(r7_out, 4U);                                    // Read the 32-bit R7 payload, it echoes voltage and check pattern.
    }
    else
    {
        r7_out[0] = 0xFFU;                                                   // Fill output with 0xFF to make failure visible in debug.
        r7_out[1] = 0xFFU;                                                   // Fill output with 0xFF to make failure visible in debug.
        r7_out[2] = 0xFFU;                                                   // Fill output with 0xFF to make failure visible in debug.
        r7_out[3] = 0xFFU;                                                   // Fill output with 0xFF to make failure visible in debug.
    }

    SD_Deselect();                                                           // End transaction after reading R7, or after timeout.
    SD_SendIdleClocks(2U);                                                   // Provide trailing clocks to let the card release MISO cleanly.

    return r1;                                                               // Return R1 so caller can interpret SD v2 or illegal command.
}

/*==============================================================================
 * Function: SPI_Test_SendACMD41_UntilReady
 *
 * Purpose:
 *   Initialize the card by repeatedly sending CMD55 then ACMD41 until the card leaves IDLE.
 *
 * Parameters:
 *   cmd55_r1_out - Optional pointer to store the last CMD55 R1 response.
 *
 * Returns:
 *   0x00 when the card is ready, otherwise the last ACMD41 R1 response.
 *
 * Side Effects:
 *   Controls CS, clocks bus, sends commands, delays between attempts.
 *
 * Preconditions:
 *   - CMD0 done, and for SD v2 cards CMD8 done.
 *   - SPI low init speed, Mode 0.
 *
 * Concurrency:
 *   Not thread-safe.
 *
 * Timing:
 *   Up to 100 attempts, with 10 ms delay each attempt.
 *
 * Errors:
 *   Returns non-zero R1 if card never becomes ready.
 *
 * Notes:
 *   For SD v2, use HCS bit in ACMD41 argument (0x40000000) to request SDHC/SDXC.
 *==============================================================================*/
uint8_t SPI_Test_SendACMD41_UntilReady(uint8_t *cmd55_r1_out)
{
    uint8_t r1_cmd55 = 0xFFU;                                                // Hold the most recent CMD55 response for debug purposes.
    uint8_t r1_acmd41 = 0xFFU;                                               // Hold the most recent ACMD41 response for loop exit condition.

    if (cmd55_r1_out != NULL)                                                // Only write output if the caller provided storage.
    {
        *cmd55_r1_out = 0xFFU;                                               // Initialize output to 0xFF so "never ran" is obvious.
    }

    for (uint32_t attempt = 0U; attempt < 100U; attempt++)                   // Try multiple times because SD card init is allowed to take time.
    {
        SD_Deselect();                                                       // Ensure CS high between attempts to reset SPI framing expectations.
        SD_SendIdleClocks(2U);                                               // Provide clocks with CS high to satisfy idle timing between retries.

        SD_Select();                                                         // Select the card to send CMD55 and ACMD41 as one logical sequence.
        SD_SendIdleClocks(1U);                                               // Provide a byte gap so card is ready to sample command bytes.

        r1_cmd55 = SD_SendCommand(55U, 0U, 0xFFU);                           // CMD55 indicates that the next command is an application command.
        (void)SD_SPI_TransferByte(0xFFU);                                    // Provide a small gap that helps some cards respond reliably.

        r1_acmd41 = SD_SendCommand(41U, 0x40000000U, 0xFFU);                 // ACMD41 asks card to exit IDLE, HCS=1 requests high capacity.

        SD_Deselect();                                                       // End transaction so card can progress its internal init state.
        SD_SendIdleClocks(2U);                                               // Provide trailing clocks after deselect, common SD SPI practice.

        if (cmd55_r1_out != NULL)                                            // Store CMD55 response for external debugging if requested.
        {
            *cmd55_r1_out = r1_cmd55;                                        // Write latest CMD55 response so caller can confirm APP_CMD acceptance.
        }

        if (r1_acmd41 == 0x00U)                                              // R1==0 means the card is ready for data commands.
        {
            return 0x00U;                                                    // Return success, card is initialized and out of IDLE.
        }

        HAL_Delay(10U);                                                      // Delay so we do not hammer the card, and allow internal init to continue.
    }

    return r1_acmd41;                                                        // Return last response so caller can diagnose why it never became ready.
}

/*==============================================================================
 * Function: SPI_Test_SendCMD58_ReadOCR
 *
 * Purpose:
 *   Send CMD58 (READ_OCR) and read the 4-byte OCR register in SPI mode.
 *
 * Parameters:
 *   ocr_out - Pointer to a 4-byte buffer to receive OCR bytes (MSB first).
 *
 * Returns:
 *   R1 response byte from CMD58. 0x00 indicates success. 0xFF indicates timeout.
 *
 * Side Effects:
 *   Controls CS, clocks bus, and reads OCR bytes from MISO.
 *
 * Preconditions:
 *   - Card initialized (ACMD41 ready) or at least responsive.
 *   - SPI Mode 0, low speed is fine, higher speed is also fine post-init.
 *   - ocr_out is not NULL and points to at least 4 bytes.
 *
 * Concurrency:
 *   Not thread-safe.
 *
 * Timing:
 *   Bounded wait for R1, then fixed 4-byte read.
 *
 * Errors:
 *   Returns 0xFF if no response observed.
 *
 * Notes:
 *   CRC is ignored after init unless CRC checking is enabled on the card. Use 0xFF as the standard dummy CRC byte.
 *==============================================================================*/
uint8_t SPI_Test_SendCMD58_ReadOCR(uint8_t ocr_out[4])
{
    uint8_t r1 = 0xFFU;                                                      // Default to timeout marker until we see a response.

    if (ocr_out == NULL)                                                     // Guard against NULL pointer to prevent hard fault.
    {
        return 0xFFU;                                                        // Return timeout marker for invalid input in bring-up code.
    }

    SD_Select();                                                             // Select the card so it will respond to CMD58.
    (void)SD_SPI_TransferByte(0xFFU);                                        // Provide at least one byte time before sending the command.

    r1 = SD_SendCommand(58U, 0U, 0xFFU);                                     // Send CMD58 with dummy CRC (0xFF) since CRC is normally ignored now.
    if (r1 != 0xFFU)                                                         // Only read OCR if the card responded.
    {
        SD_ReadResponseBytes(ocr_out, 4U);                                   // Read OCR bytes, MSB first, immediately after R1.
    }
    else
    {
        ocr_out[0] = 0xFFU;                                                  // Fill OCR buffer so timeout is obvious in debug.
        ocr_out[1] = 0xFFU;                                                  // Fill OCR buffer so timeout is obvious in debug.
        ocr_out[2] = 0xFFU;                                                  // Fill OCR buffer so timeout is obvious in debug.
        ocr_out[3] = 0xFFU;                                                  // Fill OCR buffer so timeout is obvious in debug.
    }

    SD_Deselect();                                                           // End the transaction so the card can release MISO.
    SD_SendIdleClocks(2U);                                                   // Provide trailing clocks after deselect to complete bus timing.

    return r1;                                                               // Return R1 so caller can confirm success (0x00) or error/timeout.
}

/*==============================================================================
 * Function: SD_SendCommandRaw_AndGetR1
 *
 * Purpose:
 *   Low-level "raw packet" sender for experiments. Sends a 6-byte command frame and returns R1.
 *
 * Parameters:
 *   cmd_index - SD command number.
 *   argument  - 32-bit argument.
 *   crc       - CRC byte to transmit (must include end bit 1).
 *
 * Returns:
 *   R1 byte, or 0xFF on timeout.
 *
 * Side Effects:
 *   Transmits bytes on MOSI and reads MISO.
 *
 * Preconditions:
 *   - CS must already be low before calling.
 *
 * Concurrency:
 *   Not thread-safe.
 *
 * Timing:
 *   Bounded wait loop for response.
 *
 * Errors:
 *   Returns 0xFF on timeout.
 *
 * Notes:
 *   Prefer SD_SendCommand for normal use, this is mainly for learning and quick test variations.
 *==============================================================================*/
static uint8_t SD_SendCommandRaw_AndGetR1(uint8_t cmd_index, uint32_t argument, uint8_t crc)
{
    uint8_t r1 = 0xFFU;                                                      // Default to no-response marker.

    (void)SD_SPI_TransferByte((uint8_t)(0x40U | (cmd_index & 0x3FU)));        // Send command byte formatted for SPI mode.
    (void)SD_SPI_TransferByte((uint8_t)((argument >> 24) & 0xFFU));          // Send argument MSB.
    (void)SD_SPI_TransferByte((uint8_t)((argument >> 16) & 0xFFU));          // Send argument byte 2.
    (void)SD_SPI_TransferByte((uint8_t)((argument >> 8) & 0xFFU));           // Send argument byte 1.
    (void)SD_SPI_TransferByte((uint8_t)(argument & 0xFFU));                  // Send argument LSB.
    (void)SD_SPI_TransferByte(crc);                                          // Send CRC byte with end bit set.

    for (uint32_t i = 0U; i < 1000U; i++)                                    // Poll for response, raw path uses a larger loop for bring-up.
    {
        r1 = SD_SPI_TransferByte(0xFFU);                                     // Clock one byte while sampling for R1.
        if (r1 != 0xFFU)                                                     // First non-0xFF is the R1 response.
        {
            break;                                                          // Exit loop once response appears.
        }
    }

    return r1;                                                               // Return the R1 byte, or 0xFF if timeout.
}


/*==============================================================================
 * Function: SPI_Test_SendCMD17_ReadSingleBlock
 *
 * Purpose:
 *   Read one 512-byte block from the SD card using CMD17 (READ_SINGLE_BLOCK) in SPI mode.
 *
 * Parameters:
 *   block_lba   - Block index (LBA). For SDHC/SDXC, this is the correct addressing unit.
 *                For SDSC cards, caller must pass (byte_address = lba * 512) instead.
 *   data_out    - Pointer to a 512-byte buffer to receive the block data.
 *
 * Returns:
 *   SD_SPI_CMD17_STATUS_OK on success, otherwise an SD_SPI_CMD17_STATUS_* error code.
 *
 * Side Effects:
 *   Drives CS, clocks SPI, reads MISO, and consumes card output bytes.
 *
 * Preconditions:
 *   - Card is initialized and ready (ACMD41 completed, R1=0x00).
 *   - SPI configured for SD card: Mode 0, 8-bit, MSB-first.
 *   - CS pin configured as output and wired correctly.
 *   - data_out points to at least 512 bytes.
 *
 * Concurrency:
 *   Not thread-safe; caller must ensure exclusive SPI access.
 *
 * Timing:
 *   Waits up to SD_SPI_CMD17_TOKEN_TIMEOUT_MS for the 0xFE token, then reads 514 bytes (512 + CRC).
 *
 * Errors:
 *   Returns non-zero status codes on failures or timeouts.
 *
 * Notes:
 *   - This ignores the 16-bit CRC that follows the data block, which is fine for bring-up.
 *   - For SDHC/SDXC, block_lba is correct. Your OCR earlier indicated SDHC (CCS=1), so this is correct for you.
 *==============================================================================*/
uint8_t SPI_Test_SendCMD17_ReadSingleBlock(uint32_t block_lba, uint8_t data_out[512])
{
    const uint32_t SD_SPI_CMD17_TOKEN_TIMEOUT_MS = 100U;                    // Use a simple 100 ms timeout for bring-up token wait.
    uint8_t r1 = 0xFFU;                                                     // Hold the command R1 response, defaulting to "no response".
    uint8_t token = 0xFFU;                                                  // Hold the data token byte, defaulting to idle fill.
    uint32_t start_tick_ms = 0U;                                            // Track start time for timeout calculations.
    uint32_t elapsed_ms = 0U;                                               // Track elapsed time for timeout calculations.

    if (data_out == NULL)                                                   // Validate output buffer pointer to avoid a hard fault.
    {
        return SD_SPI_CMD17_STATUS_NULL_BUFFER;                             // Return an explicit error code for clarity.
    }

    SD_Select();                                                            // Assert CS low so the card will accept CMD17 and output data.
    (void)SD_SPI_TransferByte(0xFFU);                                       // Provide one byte time (Ncr) before sending the command.

    r1 = SD_SendCommand(17U, block_lba, 0xFFU);                             // Send CMD17 with LBA argument, dummy CRC is fine post-init.
    if (r1 != 0x00U)                                                        // R1 must be 0x00 for "ready"; anything else means command was rejected.
    {
        SD_Deselect();                                                      // Deassert CS to end the transaction cleanly.
        SD_SendIdleClocks(2U);                                              // Provide trailing clocks so the card can release MISO properly.
        return SD_SPI_CMD17_STATUS_CMD_R1_NOT_READY;                        // Return an error to indicate CMD17 did not start a read.
    }

    start_tick_ms = HAL_GetTick();                                          // Capture start time so we can enforce a bounded wait for token.

    while (1)                                                               // Loop until we receive the start token or we time out.
    {
        token = SD_SPI_TransferByte(0xFFU);                                 // Clock one byte while sampling for the data start token.
        if (token == SD_SPI_DATA_START_TOKEN_SINGLE_BLOCK_READ)             // 0xFE indicates the card is about to stream the 512-byte block.
        {
            break;                                                          // Exit the wait loop and proceed to read the block data.
        }

        if (token != 0xFFU)                                                 // Any non-0xFF non-0xFE byte is suspicious during token wait.
        {
            SD_Deselect();                                                  // End transaction so we do not keep the card selected on error.
            SD_SendIdleClocks(2U);                                          // Provide trailing clocks to complete the bus cycle.
            return SD_SPI_CMD17_STATUS_UNEXPECTED_TOKEN;                    // Return error, we saw something other than idle or start token.
        }

        elapsed_ms = HAL_GetTick() - start_tick_ms;                         // Compute elapsed time since we started waiting for the token.
        if (elapsed_ms > SD_SPI_CMD17_TOKEN_TIMEOUT_MS)                     // If we have waited too long, bail out.
        {
            SD_Deselect();                                                  // Deassert CS so the card is not stuck selected.
            SD_SendIdleClocks(2U);                                          // Provide trailing clocks so the card can recover internally.
            return SD_SPI_CMD17_STATUS_TOKEN_TIMEOUT;                       // Return timeout error for diagnostics.
        }
    }

    for (uint32_t i = 0U; i < 512U; i++)                                    // Read exactly 512 bytes, SD block size in SPI mode.
    {
        data_out[i] = SD_SPI_TransferByte(0xFFU);                           // Send 0xFF to generate clocks while capturing one data byte.
    }

    (void)SD_SPI_TransferByte(0xFFU);                                       // Read and discard CRC byte 0, ignored for bring-up.
    (void)SD_SPI_TransferByte(0xFFU);                                       // Read and discard CRC byte 1, ignored for bring-up.

    SD_Deselect();                                                          // Deassert CS to end the read transaction.
    SD_SendIdleClocks(2U);                                                  // Provide trailing clocks to let the card release and finish.

    return SD_SPI_CMD17_STATUS_OK;                                          // Indicate successful single-block read.
}

/*==============================================================================
 * Function: SPI_Test_ReadUInt32LittleEndian
 *
 * Purpose:
 *   Read a 32-bit unsigned integer from a byte buffer in little-endian order.
 *
 * Parameters:
 *   buffer - Source byte array.
 *   offset - Start index where the 4-byte little-endian value begins.
 *
 * Returns:
 *   32-bit value assembled from buffer[offset..offset+3].
 *
 * Side Effects:
 *   None.
 *
 * Preconditions:
 *   - buffer is not NULL.
 *   - buffer has at least offset+4 bytes.
 *==============================================================================*/
static uint32_t SPI_Test_ReadUInt32LittleEndian(const uint8_t *buffer, uint32_t offset)
{
    uint32_t value = 0U;                                                     // Start with 0 so we can OR in each byte.
    value |= (uint32_t)buffer[offset + 0U];                                   // Byte 0 is least significant.
    value |= (uint32_t)buffer[offset + 1U] << 8U;                              // Byte 1 becomes bits 15..8.
    value |= (uint32_t)buffer[offset + 2U] << 16U;                             // Byte 2 becomes bits 23..16.
    value |= (uint32_t)buffer[offset + 3U] << 24U;                             // Byte 3 becomes bits 31..24.
    return value;                                                             // Return assembled integer.
}

/*==============================================================================
 * Function: SPI_Test_ReadBlock_CheckSignature
 *
 * Purpose:
 *   Read a single SD block and extract the 0x55AA signature bytes at offsets 510 and 511.
 *
 * Parameters:
 *   block_lba             - Block index to read (LBA for SDHC/SDXC).
 *   signature_byte0_out   - Optional pointer to store buffer[510] (usually 0x55).
 *   signature_byte1_out   - Optional pointer to store buffer[511] (usually 0xAA).
 *
 * Returns:
 *   0x00 on success (CMD17 read succeeded), otherwise propagates CMD17 error code.
 *
 * Side Effects:
 *   Reads block data into a file-scope buffer, overwriting prior content.
 *
 * Preconditions:
 *   - SD card initialized and ready.
 *==============================================================================*/
uint8_t SPI_Test_ReadBlock_CheckSignature(uint32_t block_lba, uint8_t *signature_byte0_out, uint8_t *signature_byte1_out)
{
    uint8_t status = 0xFFU;                                                  // Hold CMD17 status from the read.

    status = SPI_Test_SendCMD17_ReadSingleBlock(block_lba, g_sd_test_sector_buffer); // Read requested block into shared buffer.
    if (status != 0x00U)                                                     // If read failed, do not attempt to interpret buffer.
    {
        return status;                                                       // Return the read status so caller can debug.
    }

    if (signature_byte0_out != NULL)                                         // Only write output if caller provided a pointer.
    {
        *signature_byte0_out = g_sd_test_sector_buffer[510];                 // Extract signature byte 0.
    }

    if (signature_byte1_out != NULL)                                         // Only write output if caller provided a pointer.
    {
        *signature_byte1_out = g_sd_test_sector_buffer[511];                 // Extract signature byte 1.
    }

    return 0x00U;                                                            // Indicate success.
}

/*==============================================================================
 * Function: SPI_Test_FindFirstPartitionStartLba
 *
 * Purpose:
 *   Treat LBA0 as an MBR and parse partition entry 0 to obtain its start LBA and type.
 *
 * Parameters:
 *   partition_start_lba_out - Output pointer to store start LBA of partition 0.
 *   partition_type_out      - Optional output pointer to store partition type byte.
 *
 * Returns:
 *   0x00 on success, otherwise non-zero CMD17 status code.
 *
 * Side Effects:
 *   Reads LBA0 into file-scope buffer, overwriting prior content.
 *
 * Preconditions:
 *   - SD card initialized and ready.
 *
 * Notes:
 *   MBR partition table starts at offset 446, each entry is 16 bytes.
 *   Start LBA is at entry_offset+8, little-endian.
 *==============================================================================*/
uint8_t SPI_Test_FindFirstPartitionStartLba(uint32_t *partition_start_lba_out, uint8_t *partition_type_out)
{
    const uint32_t MBR_PARTITION_ENTRY0_OFFSET = 446U;                       // MBR partition table begins at byte 446.
    uint8_t status = 0xFFU;                                                  // Hold read status for LBA0.

    if (partition_start_lba_out == NULL)                                     // Must have a place to return the LBA.
    {
        return 0xFFU;                                                        // Return generic error for invalid argument in bring-up code.
    }

    status = SPI_Test_SendCMD17_ReadSingleBlock(0U, g_sd_test_sector_buffer); // Read LBA0 to parse the partition table.
    if (status != 0x00U)                                                     // Stop if we cannot read LBA0.
    {
        return status;                                                       // Return read error code.
    }

    if (partition_type_out != NULL)                                          // Store partition type if caller asked for it.
    {
        *partition_type_out = g_sd_test_sector_buffer[MBR_PARTITION_ENTRY0_OFFSET + 4U]; // Type is byte 4 of the entry.
    }

    *partition_start_lba_out = SPI_Test_ReadUInt32LittleEndian(               // Parse start LBA from entry bytes 8..11.
        g_sd_test_sector_buffer,                                             // Buffer holding LBA0 (MBR).
        MBR_PARTITION_ENTRY0_OFFSET + 8U);                                   // Offset of start LBA in partition entry.

    return 0x00U;                                                            // Indicate success.
}

/*==============================================================================
 * Function: SPI_Test_ReadVolumeBootSectorSignature
 *
 * Purpose:
 *   Determine whether LBA0 is a FAT VBR (superfloppy) or an MBR, then read the
 *   volume boot sector and return its 0x55AA signature bytes.
 *
 * Parameters:
 *   volume_boot_sector_lba_out - Output pointer to store the LBA that was treated as the VBR.
 *   signature_byte0_out        - Optional pointer to store signature byte 0 (offset 510).
 *   signature_byte1_out        - Optional pointer to store signature byte 1 (offset 511).
 *
 * Returns:
 *   0x00 on success, otherwise non-zero status code.
 *
 * Side Effects:
 *   Performs one or two CMD17 reads using a file-scope buffer.
 *
 * Preconditions:
 *   - SD card initialized and ready.
 *
 * Notes:
 *   - If LBA0 begins with 0xEB or 0xE9, it often indicates a FAT boot sector jump instruction.
 *   - If not, we assume MBR and look up partition 0 start LBA.
 *==============================================================================*/
uint8_t SPI_Test_ReadVolumeBootSectorSignature(uint32_t *volume_boot_sector_lba_out, uint8_t *signature_byte0_out, uint8_t *signature_byte1_out)
{
    uint8_t status = 0xFFU;                                                  // Hold status for reads and parsing.
    uint32_t vbr_lba = 0U;                                                   // Volume Boot Record LBA we will decide on.
    uint8_t first_byte = 0x00U;                                              // First byte of sector 0, used for quick format heuristic.

    if (volume_boot_sector_lba_out == NULL)                                  // Caller must supply storage for chosen VBR LBA.
    {
        return 0xFFU;                                                        // Return generic invalid argument error for bring-up.
    }

    status = SPI_Test_SendCMD17_ReadSingleBlock(0U, g_sd_test_sector_buffer); // Read LBA0 so we can decide MBR vs VBR.
    if (status != 0x00U)                                                     // If we cannot read LBA0, we cannot proceed.
    {
        return status;                                                       // Return read error.
    }

    first_byte = g_sd_test_sector_buffer[0];                                 // Capture first byte, FAT VBR often starts with EB/E9.
    if ((first_byte == 0xEBU) || (first_byte == 0xE9U))                      // Check for common jump opcodes in FAT boot sectors.
    {
        vbr_lba = 0U;                                                        // Superfloppy layout: VBR is at LBA0.
    }
    else
    {
        uint8_t partition_type = 0x00U;                                      // Hold partition type for debugging visibility.
        status = SPI_Test_FindFirstPartitionStartLba(&vbr_lba, &partition_type); // Parse MBR partition 0 start LBA.
        (void)partition_type;                                                // Keep visible in debugger for diagnosing layout.
        if (status != 0x00U)                                                 // If MBR parse failed, stop.
        {
            return status;                                                   // Return parse/read error.
        }
    }

    status = SPI_Test_ReadBlock_CheckSignature(vbr_lba, signature_byte0_out, signature_byte1_out); // Read VBR sector and extract signature.
    if (status != 0x00U)                                                     // If the VBR read failed, stop.
    {
        return status;                                                       // Return CMD17 error code.
    }

    *volume_boot_sector_lba_out = vbr_lba;                                   // Tell caller which LBA we treated as the boot sector.
    return 0x00U;                                                            // Indicate success.
}
