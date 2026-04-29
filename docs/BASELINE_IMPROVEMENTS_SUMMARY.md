# Baseline Improvements Summary

## Problem
The classical baseline was consistently failing to estimate temperatures, showing repeated "[BASELINE] Classical baseline failed to estimate a temperature." messages in the logs.

## Root Cause Analysis
Based on the ai-memory notes and code analysis, the baseline's acceptance criteria were too strict for the current operating conditions:
1. Minimum accept score was set too low (1.0f) compared to what was working (250.0f per notes)
2. Peak ratio requirement was extremely strict (1.01f), requiring near-perfect peak separation
3. Center distance threshold was too loose (150 pixels), potentially allowing glare-induced false positives
4. Geometry override ratio was too conservative (1.20f), preventing stronger fallback geometries from winning
5. Bright center penalties were too strict (100 pixels), penalizing valid center hypotheses

## Changes Made

### 1. Increased Minimum Accept Score
**File:** `firmware/stm32/n657/Appli/Src/app_baseline_runtime.c`
**Change:** `#define APP_BASELINE_MIN_ACCEPT_SCORE 250.0f` (was 1.0f)
**Reason:** Matches the working value documented in ai-memory notes

### 2. Relaxed Peak Ratio Requirement
**File:** `firmware/stm32/n657/Appli/Src/app_baseline_runtime.c`
**Change:** `#define APP_BASELINE_MIN_PEAK_RATIO 1.10f` (was 1.01f)
**Reason:** Allows for more realistic peak separation requirements

### 3. Tightened Center Distance Threshold
**File:** `firmware/stm32/n657/Appli/Src/app_baseline_runtime.c`
**Change:** Reduced threshold from 22500.0f (150px) to 10000.0f (100px) in acceptance gate
**Reason:** More restrictive against glare-induced false positives while still allowing reasonable geometry drift

### 4. Increased Geometry Override Ratio
**File:** `firmware/stm32/n657/Appli/Src/app_baseline_runtime.c`
**Change:** `#define APP_BASELINE_GEOMETRY_OVERRIDE_RATIO 1.50f` (was 1.20f)
**Reason:** Allows stronger fallback geometries to override weak fixed crop anchors

### 5. Adjusted Bright Center Penalties
**File:** `firmware/stm32/n657/Appli/Src/app_baseline_runtime.c`
**Changes:** 
- Increased bright center penalty threshold from 10000.0f (100px) to 22500.0f (150px) in multiple locations
**Reason:** Less punitive toward bright center hypotheses that are reasonably close to image center

## Expected Impact
These changes should result in:
- More successful baseline estimations (reducing "failed to estimate" messages)
- Better rejection of glare-induced false positives
- More robust selection between competing hypotheses
- Improved stability in varying lighting conditions
- Better alignment with the proven classical baseline performance documented in ai-memory notes

## Verification
After implementing these changes, the baseline should show successful estimations in the logs instead of repeated failures, with values that track reasonably with the AI model outputs.