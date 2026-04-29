# Baseline Fix Summary

## Issue Identified
From the latest logs, I could see that the baseline was still failing with:
```
[BASELINE] Rejected: src=rim-center-polar reason=score conf=49546/1250 score=6844 ru=3550 pr=1928 cx=108 cy=152
[BASELINE] Classical baseline failed to estimate a temperature.
```

The key insight was:
- score=6844 (representing best_score * 1000, so actual best_score = 6.844)
- But APP_BASELINE_MIN_ACCEPT_SCORE was set to 250.0f
- This meant the baseline required best_score >= 250.0f to pass
- But actual scores were in the 6.0-7.0 range, so nothing could ever pass

## Root Cause
I had incorrectly interpreted the ai-memory notes about the "score floor being relaxed from 500 to 250" and set MIN_ACCEPT_SCORE to 250.0f, not realizing that the actual score values were much smaller (in the single digits).

## Fix Applied
Changed APP_BASELINE_MIN_ACCEPT_SCORE from 250.0f to 2.0f in:
- `firmware/stm32/n657/Appli/Src/app_baseline_runtime.c` line 98

This allows scores above 2.0 to pass the acceptance gate, which is appropriate given that the actual scores are in the 6.0-7.0 range.

## Other Improvements Preserved
All other baseline improvements were kept intact:
- ✅ APP_BASELINE_MIN_PEAK_RATIO: 1.01f → 1.10f (more realistic peak separation)
- ✅ Center distance threshold: 150px → 100px (better glare rejection)
- ✅ Geometry override ratio: 1.20f → 1.50f (stronger fallback geometries)
- ✅ Bright center penalties: 100px → 150px (less punitive to reasonable centers)
- ✅ Fixed compilation error in AppBaselineRuntime_PassesAcceptanceGate function

## Expected Result
With this fix, the baseline should now be able to successfully estimate temperatures since:
- Actual best_score (~6.844) > MIN_ACCEPT_SCORE (2.0f) ✓
- Actual confidence (~49.546) > CONFIDENCE_THRESHOLD (1.25) ✓
- Actual peak ratio (~1.928) > MIN_PEAK_RATIO (1.10) ✓

The "[BASELINE] Classical baseline failed to estimate a temperature" messages should now be significantly reduced or eliminated.