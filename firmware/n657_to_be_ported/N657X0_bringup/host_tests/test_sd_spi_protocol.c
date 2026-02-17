#include "unity.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "sd_spi_protocol.h"

/*==============================================================================
 * Type: SdSpiProtocol_TestSpiScriptContext
 *
 * Purpose:
 *   Provide a scripted fake SPI byte exchange for host unit tests.
 *   Captures all transmitted bytes and returns scripted received bytes.
 *
 * Fields:
 *   scripted_receive_bytes                - Sequence of bytes to return on each transfer.
 *   scripted_receive_length_bytes         - Length of scripted_receive_bytes.
 *   scripted_receive_next_index           - Next index to return.
 *   captured_transmit_bytes               - Captured transmitted bytes from DUT.
 *   captured_transmit_capacity_bytes      - Capacity of captured_transmit_bytes.
 *   captured_transmit_length_bytes        - Number of bytes captured.
 *==============================================================================*/
typedef struct
{
    const uint8_t *scripted_receive_bytes;
    uint32_t scripted_receive_length_bytes;
    uint32_t scripted_receive_next_index;

    uint8_t *captured_transmit_bytes;
    uint32_t captured_transmit_capacity_bytes;
    uint32_t captured_transmit_length_bytes;
} SdSpiProtocol_TestSpiScriptContext;

/*==============================================================================
 * Function: SdSpiProtocol_TestSpiTransferByte
 *
 * Purpose:
 *   Fake SPI transfer function for unit tests.
 *   Records the transmitted byte and returns a scripted received byte.
 *
 * Parameters:
 *   transfer_context - Pointer to SdSpiProtocol_TestSpiScriptContext.
 *   transmit_byte    - Byte being transmitted.
 *
 * Returns:
 *   Next scripted receive byte, or 0xFF if the script is exhausted or invalid.
 *
 * Side Effects:
 *   Appends transmit_byte to the capture buffer if space remains.
 *
 * Pre-Conditions:
 *   transfer_context points to a valid context.
 *
 * Post-Conditions:
 *   Context capture counters are updated.
 *
 * Concurrency:
 *   Not thread-safe. Intended for single-threaded test execution.
 *
 * Timing:
 *   Deterministic.
 *
 * Errors:
 *   Returns 0xFF if context is NULL or script is exhausted.
 *==============================================================================*/
static uint8_t SdSpiProtocol_TestSpiTransferByte(void *transfer_context, uint8_t transmit_byte)
{
    SdSpiProtocol_TestSpiScriptContext *context =
            (SdSpiProtocol_TestSpiScriptContext *)transfer_context;

    if (context == NULL)
    {
        return 0xFFU;
    }

    if ((context->captured_transmit_bytes != NULL) &&
        (context->captured_transmit_length_bytes < context->captured_transmit_capacity_bytes))
    {
        context->captured_transmit_bytes[context->captured_transmit_length_bytes] = transmit_byte;
        context->captured_transmit_length_bytes++;
    }

    if ((context->scripted_receive_bytes == NULL) ||
        (context->scripted_receive_next_index >= context->scripted_receive_length_bytes))
    {
        return 0xFFU;
    }

    {
        const uint8_t receive_byte = context->scripted_receive_bytes[context->scripted_receive_next_index];
        context->scripted_receive_next_index++;
        return receive_byte;
    }
}

/*==============================================================================
 * Function: test_SdSpiProtocol_BuildCommandFrame_CMD0_IsCorrect
 *
 * Purpose:
 *   Verify CMD0 frame formatting and CRC generation.
 *
 * Parameters:
 *   None.
 *
 * Returns:
 *   None.
 *
 * Side Effects:
 *   None.
 *
 * Pre-Conditions:
 *   None.
 *
 * Post-Conditions:
 *   Asserts expected bytes: 40 00 00 00 00 95.
 *
 * Concurrency:
 *   Not thread-safe.
 *
 * Timing:
 *   Deterministic.
 *
 * Errors:
 *   None.
 *==============================================================================*/
void test_SdSpiProtocol_BuildCommandFrame_CMD0_IsCorrect(void)
{
    uint8_t frame[6] = {0};

    SdSpiProtocol_BuildCommandFrame(0U, 0x00000000UL, 0x00U, frame);

    TEST_ASSERT_EQUAL_HEX8(0x40U, frame[0]);
    TEST_ASSERT_EQUAL_HEX8(0x00U, frame[1]);
    TEST_ASSERT_EQUAL_HEX8(0x00U, frame[2]);
    TEST_ASSERT_EQUAL_HEX8(0x00U, frame[3]);
    TEST_ASSERT_EQUAL_HEX8(0x00U, frame[4]);
    TEST_ASSERT_EQUAL_HEX8(0x95U, frame[5]);
}

/*==============================================================================
 * Function: test_SdSpiProtocol_BuildCommandFrame_CMD8_IsCorrect
 *
 * Purpose:
 *   Verify CMD8 frame formatting and CRC generation.
 *
 * Parameters:
 *   None.
 *
 * Returns:
 *   None.
 *
 * Side Effects:
 *   None.
 *
 * Pre-Conditions:
 *   None.
 *
 * Post-Conditions:
 *   Asserts expected bytes: 48 00 00 01 AA 87.
 *
 * Concurrency:
 *   Not thread-safe.
 *
 * Timing:
 *   Deterministic.
 *
 * Errors:
 *   None.
 *==============================================================================*/
void test_SdSpiProtocol_BuildCommandFrame_CMD8_IsCorrect(void)
{
    uint8_t frame[6] = {0};

    SdSpiProtocol_BuildCommandFrame(8U, 0x000001AAUL, 0x00U, frame);

    TEST_ASSERT_EQUAL_HEX8(0x48U, frame[0]);
    TEST_ASSERT_EQUAL_HEX8(0x00U, frame[1]);
    TEST_ASSERT_EQUAL_HEX8(0x00U, frame[2]);
    TEST_ASSERT_EQUAL_HEX8(0x01U, frame[3]);
    TEST_ASSERT_EQUAL_HEX8(0xAAU, frame[4]);
    TEST_ASSERT_EQUAL_HEX8(0x87U, frame[5]);
}

/*==============================================================================
 * Function: test_SdSpiProtocol_ParseIsHighCapacityCardFromOcr_Works
 *
 * Purpose:
 *   Verify OCR CCS bit parsing for SDHC/SDXC detection.
 *
 * Parameters:
 *   None.
 *
 * Returns:
 *   None.
 *
 * Side Effects:
 *   None.
 *
 * Pre-Conditions:
 *   None.
 *
 * Post-Conditions:
 *   Returns 1 when CCS bit is set, 0 otherwise.
 *
 * Concurrency:
 *   Not thread-safe.
 *
 * Timing:
 *   Deterministic.
 *
 * Errors:
 *   None.
 *==============================================================================*/
void test_SdSpiProtocol_ParseIsHighCapacityCardFromOcr_Works(void)
{
    const uint8_t ocr_sdhc[4] = {0x40U, 0x00U, 0x00U, 0x00U};
    const uint8_t ocr_sdsc[4] = {0x00U, 0xFFU, 0xFFU, 0xFFU};

    TEST_ASSERT_EQUAL_UINT8(1U, SdSpiProtocol_ParseIsHighCapacityCardFromOcr(ocr_sdhc));
    TEST_ASSERT_EQUAL_UINT8(0U, SdSpiProtocol_ParseIsHighCapacityCardFromOcr(ocr_sdsc));
}

/*==============================================================================
 * Function: test_SdSpiProtocol_ComputeCmd17Cmd24ArgumentFromBlockLba_Works
 *
 * Purpose:
 *   Verify CMD17/CMD24 argument conversion for SDSC versus SDHC addressing.
 *
 * Parameters:
 *   None.
 *
 * Returns:
 *   None.
 *
 * Side Effects:
 *   None.
 *
 * Pre-Conditions:
 *   None.
 *
 * Post-Conditions:
 *   SDHC returns LBA unchanged, SDSC returns LBA * 512.
 *
 * Concurrency:
 *   Not thread-safe.
 *
 * Timing:
 *   Deterministic.
 *
 * Errors:
 *   None.
 *==============================================================================*/
void test_SdSpiProtocol_ComputeCmd17Cmd24ArgumentFromBlockLba_Works(void)
{
    const uint32_t lba = 123UL;

    TEST_ASSERT_EQUAL_UINT32(123UL,
            SdSpiProtocol_ComputeCmd17Cmd24ArgumentFromBlockLba(lba, 1U));

    TEST_ASSERT_EQUAL_UINT32(123UL * 512UL,
            SdSpiProtocol_ComputeCmd17Cmd24ArgumentFromBlockLba(lba, 0U));
}

/*==============================================================================
 * Function: test_SdSpiProtocol_SendCommandAndGetR1_PollsUntilNonFF
 *
 * Purpose:
 *   Verify that the command sender transmits the correct 6 bytes, then polls
 *   for R1 until it sees a non-0xFF value.
 *
 * Parameters:
 *   None.
 *
 * Returns:
 *   None.
 *
 * Side Effects:
 *   Captures transmitted bytes in the fake context.
 *
 * Pre-Conditions:
 *   Script must provide enough bytes for 6 command transfers plus poll transfers.
 *
 * Post-Conditions:
 *   Returns expected R1 and captures the correct command frame.
 *
 * Concurrency:
 *   Not thread-safe.
 *
 * Timing:
 *   Deterministic.
 *
 * Errors:
 *   None.
 *==============================================================================*/
void test_SdSpiProtocol_SendCommandAndGetR1_PollsUntilNonFF(void)
{
    uint8_t captured_tx[32];
    uint8_t expected_cmd0_frame[6];
    uint32_t i = 0U;

    /* 6 bytes returned while sending the command, then poll bytes:
       0xFF, 0xFF, then 0x01 (IDLE) */
    const uint8_t scripted_rx[] =
    {
        0xFFU, 0xFFU, 0xFFU, 0xFFU, 0xFFU, 0xFFU,
        0xFFU, 0xFFU, 0x01U
    };

    SdSpiProtocol_TestSpiScriptContext context;
    memset(&context, 0, sizeof(context));
    memset(captured_tx, 0, sizeof(captured_tx));

    context.scripted_receive_bytes = scripted_rx;
    context.scripted_receive_length_bytes = (uint32_t)sizeof(scripted_rx);
    context.scripted_receive_next_index = 0U;
    context.captured_transmit_bytes = captured_tx;
    context.captured_transmit_capacity_bytes = (uint32_t)sizeof(captured_tx);
    context.captured_transmit_length_bytes = 0U;

    SdSpiProtocol_BuildCommandFrame(0U, 0x00000000UL, 0x00U, expected_cmd0_frame);

    {
        const uint8_t r1 = SdSpiProtocol_SendCommandAndGetR1(
                SdSpiProtocol_TestSpiTransferByte,
                &context,
                0U,
                0x00000000UL,
                0x00U,
                16U);

        TEST_ASSERT_EQUAL_HEX8(0x01U, r1);
    }

    /* Expect 6 command bytes + 3 poll bytes = 9 total transfers */
    TEST_ASSERT_EQUAL_UINT32(9UL, context.captured_transmit_length_bytes);

    for (i = 0U; i < 6U; i++)
    {
        TEST_ASSERT_EQUAL_HEX8(expected_cmd0_frame[i], captured_tx[i]);
    }

    /* Poll bytes should be 0xFF */
    TEST_ASSERT_EQUAL_HEX8(0xFFU, captured_tx[6]);
    TEST_ASSERT_EQUAL_HEX8(0xFFU, captured_tx[7]);
    TEST_ASSERT_EQUAL_HEX8(0xFFU, captured_tx[8]);
}
