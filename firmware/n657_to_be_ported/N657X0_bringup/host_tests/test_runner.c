#include "unity.h"
void test_SdSpiProtocol_BuildCommandFrame_CMD0_IsCorrect(void);
void test_SdSpiProtocol_BuildCommandFrame_CMD8_IsCorrect(void);
void test_SdSpiProtocol_ParseIsHighCapacityCardFromOcr_Works(void);
void test_SdSpiProtocol_ComputeCmd17Cmd24ArgumentFromBlockLba_Works(void);
void test_SdSpiProtocol_SendCommandAndGetR1_PollsUntilNonFF(void);
void test_rollover_occurs_when_record_would_exceed_threshold(void);
void test_open_if_needed_creates_file_if_missing(void);
void test_force_flush_and_close_closes_when_open(void);


/*==============================================================================
 * Function: setUp
 *
 * Purpose:
 *   Unity per-test setup hook. Runs before each test.
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
 *   None.
 *
 * Concurrency:
 *   Not thread-safe. Intended for single-threaded test execution.
 *
 * Timing:
 *   Deterministic.
 *
 * Errors:
 *   None.
 *
 * Notes:
 *   Keep empty unless tests require shared initialization.
 *==============================================================================*/
void setUp(void)
{
}

/*==============================================================================
 * Function: tearDown
 *
 * Purpose:
 *   Unity per-test teardown hook. Runs after each test.
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
 *   None.
 *
 * Concurrency:
 *   Not thread-safe. Intended for single-threaded test execution.
 *
 * Timing:
 *   Deterministic.
 *
 * Errors:
 *   None.
 *
 * Notes:
 *   Keep empty unless tests allocate resources that must be released.
 *==============================================================================*/
void tearDown(void)
{
}

/*==============================================================================
 * Function: test_Sanity_OnePlusOne_EqualsTwo
 *
 * Purpose:
 *   Verify that the host test runner and Unity framework are wired correctly.
 *
 * Parameters:
 *   None.
 *
 * Returns:
 *   None.
 *
 * Side Effects:
 *   Prints Unity test output to stdout.
 *
 * Pre-Conditions:
 *   Unity must be initialized by UNITY_BEGIN().
 *
 * Post-Conditions:
 *   Unity records pass/fail state for this test.
 *
 * Concurrency:
 *   Not thread-safe. Intended for single-threaded test execution.
 *
 * Timing:
 *   Deterministic.
 *
 * Errors:
 *   None.
 *
 * Notes:
 *   This is a smoke test to confirm the harness works before adding real modules.
 *==============================================================================*/
void test_Sanity_OnePlusOne_EqualsTwo(void);

int main(void)
{
    int unity_result_code = 0;

    UNITY_BEGIN();
    RUN_TEST(test_Sanity_OnePlusOne_EqualsTwo);
	RUN_TEST(test_SdSpiProtocol_BuildCommandFrame_CMD0_IsCorrect);
	RUN_TEST(test_SdSpiProtocol_BuildCommandFrame_CMD8_IsCorrect);
	RUN_TEST(test_SdSpiProtocol_ParseIsHighCapacityCardFromOcr_Works);
	RUN_TEST(test_SdSpiProtocol_ComputeCmd17Cmd24ArgumentFromBlockLba_Works);
	RUN_TEST(test_SdSpiProtocol_SendCommandAndGetR1_PollsUntilNonFF);
	RUN_TEST(test_rollover_occurs_when_record_would_exceed_threshold);
	RUN_TEST(test_open_if_needed_creates_file_if_missing);
	RUN_TEST(test_force_flush_and_close_closes_when_open);

    unity_result_code = UNITY_END();

    return unity_result_code;
}
