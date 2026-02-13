#include "unity.h"

/*==============================================================================
 * Function: test_Sanity_OnePlusOne_EqualsTwo
 *
 * Purpose:
 *   Confirm Unity assertions function correctly on the host build.
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
 *   The test passes if arithmetic and Unity asserts behave as expected.
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
 *   Replace this with real unit tests once the harness is verified.
 *==============================================================================*/
void test_Sanity_OnePlusOne_EqualsTwo(void)
{
    const int left_operand = 1;
    const int right_operand = 1;
    const int expected_sum = 2;
    const int actual_sum = left_operand + right_operand;

    TEST_ASSERT_EQUAL_INT(expected_sum, actual_sum);
}
