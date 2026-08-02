# Labelled data inventory and leak rules — 2026-07-31

Date: 2026-07-31
Status: current
Scope: `ml/data/labelled` CVAT archives, ellipse/heatmap training data assembly
Evidence: `ml/scripts/repair_board_archive_images.py`, `ml/scripts/train_ellipse_robust_384.py`,
`ml/scripts/train_multiscale_center_proposal_384.py`, artifact reports under `ml/artifacts/ellipse_iter*`

## The archive inventory

`ml/data/labelled` holds the CVAT 1.1 zip archives. The generic pressure-gauge
corpus uses the `GaugeFace` ellipse label; the LittleGood temp-gauge family
(`initial_temp_gauge/`) uses `temp_dial` (with `temp_center`/`temp_tip` points).

| Archive | Images | Label | Role |
| --- | --- | --- | --- |
| `train_1.zip` | 7,326 | GaugeFace | generic train |
| `val_1.zip` | 916 | GaugeFace | generic validation |
| `train_2.zip` | 11 | GaugeFace | tiny-gauge train (the 11 originals behind test_2) |
| `val_2.zip` | 11 | GaugeFace | tiny-gauge validation |
| `test_1.zip` | 914 usable | GaugeFace | generic held-out |
| `test_2.zip` | 11 | GaugeFace | augmented board captures (IMG_1443 family) |
| `test_3.zip` | 22 | temp_dial | LittleGood board held-out |
| `initial_temp_gauge/board_captures_1.zip` | 201 | temp_dial | board train |
| `initial_temp_gauge/board_captures_2.zip` | 22 | temp_dial | **duplicate of test_3 — never train on it** |
| `initial_temp_gauge/board_captures_3.zip` | 10 | temp_dial | board train (repaired) |
| `initial_temp_gauge/board_captures_4.zip` | 35 | temp_dial | board train (repaired) |
| `initial_temp_gauge/gauge_1_batch_1..7.zip` | 50 each | temp_dial | LittleGood phone captures (repaired) |
| `initial_temp_gauge/gauge_1_batch_8.zip` | 2 | temp_dial | LittleGood phone captures (repaired) |

Board train pool after repair: 598 images (`board_captures_1/3/4` + `gauge_1_batch_1..8`).
Total generic pool: 8,233 (`train_1` + `val_1`); tiny pool: 22 (`train_2` + `val_2`).

## Rules that must never be broken

1. **`board_captures_2.zip` is an exact image-basename duplicate of `test_3.zip`
   (22/22 identical names).** Training on it invalidates the test_3 gate.
   Always filter it out of the board training list. The established pattern is
   `TRAIN_BOARD_ZIPS = [z for z in BOARD_TRAIN_ZIPS if z != "initial_temp_gauge/board_captures_2.zip"]`
   (see `train_multiscale_center_proposal_384.py`).

2. **Iteration 1/2 test_3 numbers were leaked.** The `ellipse_iter1`/`iter2`
   runs used `--include-labelled-board` before the constant excluded the
   duplicate, so their test_3 scores (~7.6-7.9px) are untrustworthy. Iteration 3
   (which excluded it) is the honest baseline: test_3 center 11.29px.

3. **Repeating is the standard way to rebalance small domains.** The tiny
   family has only 22 originals and the board family 598, so scripts use
   `tiny_repeats`/`board_repeats` (iter3: 100/4) before scale augmentation.
   `--generic-limit 0` means "use all generic images".

4. **Loaders skip archives that lack image bytes silently.** Several board
   archives were committed as `annotations.xml` only; `load_zips` returns
   fewer samples than the annotation count without any warning. Verify sample
   counts after any archive change.

## The repair procedure (2026-07-31)

The 10 image-less archives were rebuilt with
`ml/scripts/repair_board_archive_images.py`, which re-injects each referenced
image by basename from:

- `ml/data/raw` (352 `PXL_*.jpg` LittleGood phone captures → `gauge_1_batch_*`)
- `ml/data/captured_images/clean_board_captures` (45 `capture_*.png` board
  frames → `board_captures_3/4`)

All 397 referenced images were found; 0 missing. The repaired `gauge_1_batch_*`
captures have radius ~0.27-0.29, notably closer to test_3's ~0.31 than the old
board pool (~0.24), so they are the in-domain data most likely to move the
test_3 gate.

## Decision

For generalization experiments, always build the board train list from
`BOARD_TRAIN_ZIPS` minus `board_captures_2.zip`, and record
`excluded_test3_source` in the run report so the leak protection is
self-documenting. Re-run `repair_board_archive_images.py` after any fresh
archive import to confirm image bytes are present.
