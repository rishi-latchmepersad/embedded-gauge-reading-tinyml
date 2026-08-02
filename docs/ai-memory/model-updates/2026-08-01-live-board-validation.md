# LIVE-BOARD VALIDATION: 4 fresh captures read within ~1C — 2026-08-01

Date: 2026-08-01
Status: validated
Scope: ellipse + keypoint -> temperature pipeline on fresh STM32 board captures
Evidence: `tmp/board_2026-08-01/` (4 PNGs), pipeline run log

## Captures and predictions

Images: `capture_28-18.png`, `capture_29-24.png`, `capture_30-32.png`,
`capture_31-39.png` — 640x640 RGB frames saved from the board.  The
filename numbers are capture labels, NOT temperatures (user confirmed the
needle read 12C at capture time).

| Image | Predicted temp |
|---|---|
| capture_28-18.png | 11.5C |
| capture_29-24.png | 11.3C |
| capture_30-32.png | 11.2C |
| capture_31-39.png | 11.0C |

**Ground truth: needle at 12C.  Errors 0.5-1.0C.**

## What the pipeline did

- Ellipse: conf 0.997, center ~(0.49, 0.48), rx ~0.19, ry ~0.21 (640px
  frame) — consistent across all four frames.
- Keypoint: center at crop center (108,107)/224, tip ~straight up
  (113,51)/224 — the needle really was near the 12C position.
- Angle -> temperature via LittleGood calibration (135/270 sweep).

## Significance

This is the first validation on FRESH live captures that were never in
training and came straight from the board's camera path.  The sub-degree-C
accuracy matches the held-out sets (0.82C on 16 unseen captures) and
confirms the pipeline works end-to-end on real board imagery without any
retraining or tuning.

## Files

- Source PNGs: `tmp/board_2026-08-01/*.png`
- Pipeline results: `tmp/board_2026-08-01/board_2026-08-01_pipeline_results.csv`
- Diagnostic overlays: `tmp/board_2026-08-01/diag_*`
