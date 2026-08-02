# Iteration 4: universal multiscale + ALL labelled data — test_3 passes — 2026-07-31

Date: 2026-07-31
Status: current
Scope: ellipse detector iterations (target <10px center MAE on all 3 test sets)
Evidence: `ml/artifacts/ellipse_iter4_universal_all_data/report.json`,
`/tmp/iter4_universal.log`

## Results (int8, 25,650 training samples, all archives minus test_3 duplicate)

| Split | iter3 (14,008 samples) | iter4 (25,650) | Verdict |
|---|---|---|---|
| test_1 center MAE | 11.92px | 12.18px | ~flat, still misses gate |
| test_2 center MAE | 7.79px | 38.05px | **regressed hard** |
| test_3 center MAE | 11.29px | **6.23px** | **passes <10px gate** |

test_3 radius also improved (13.0px vs 19.5px). test_1 radius improved
(9.66px vs 10.39px). The recovered `gauge_1_batch_*` board captures
(radius ~0.27-0.29, close to test_3's 0.31) are confirmed to be the right
in-domain data.

## Why test_2 regressed (dilution)

- test_2 = 11 augmented captures of ONE tiny gauge (IMG_1443). Its training
  pool is train_2 (IMG_1441) + val_2 (IMG_1442) — the SAME gauge family
  photographed consecutively, each with the same 10-way augmentation
  (blur/hflip/padded/rotate/white_noise...). So test_2 is learnable: iter3
  scored 7.79px on it.
- Doubling the generic pool (4000 → 8,233) shrank the tiny family's share
  of the batch mix from ~15.7% to ~8.6%, and scale-augmentation
  (0.20-0.80×) floods the tiny-radius band with *pressure-gauge* samples
  (generic radius 0.44 × 0.20 scale = 0.088, matching test_2's 0.094).
  The model got better at synthetic tiny pressure gauges and worse at the
  actual IMG_144x family. Data-coverage-beats-arch again: test_2 needs its
  family's share restored, not more generic bulk.
- A soft-fusion decode (all 3 heads weighted by softmax confidence instead
  of argmax) only moved test_1 12.18→11.75px and test_2 stayed 35.6px, so
  the regression is in the model, not the decoder.

## Actions

1. iter5 = SimCC fine-center head (sub-pixel, 3 bins/px; removes the 24×24
   heatmap's ~4-5px decode floor) + `--tiny-repeats 200` to restore the tiny
   family's share (~14.6%, 4,400 samples). Script:
   `ml/scripts/train_ellipse_multiscale_simcc_384.py` (all memory safeguards).
2. If test_2 stays broken, add heavier on-the-fly augmentation for the tiny
   family only (rotation/scale), since the fixed 10-way augment set is
   identical across all three IMG_144x photos.
