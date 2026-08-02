# Keypoint winner: unet_wide — center excellent, tip still weak — 2026-07-31

Date: 2026-07-31
Status: current
Scope: center/tip keypoint model (56x56x2 heatmap contract, 224x224 input)
Evidence: `ml/artifacts/keypoint_unet_224g_wide/report.json`,
`ml/artifacts/keypoint_unet_224g_wide/per_split_report.json`

## Full-data results (alpha=1.5, 7,922 train images, 60+20 epochs, int8)

| Split | center MAE | center ≤8px | tip MAE | tip ≤8px |
|---|---|---|---|---|
| test_1 (913) | **3.47px** | 94.9% | 14.47px | 69.6% |
| test_2 (11) | 9.58px | 63.6% | 31.91px | 18.2% |
| test_3 (22) | 8.57px | 86.4% | 13.41px | 86.4% |
| val (report) | 3.41px | — | 6.48px | — |

int8 size: 2.24 MB (fits 2.5 MB alone; tight if combined with the 709K
ellipse model in one package).

## Findings

1. **Center is solved** (3.5px test_1, high ≤8px everywhere). The wide
   UNet learned center localization robustly across domains.
2. **Tip is the remaining weak spot** — same failure mode as every
   historical keypoint run: tip underestimates the center-to-tip distance
   (heatmaps treat keypoints independently, no geometric constraint). The
   val split (6.5px) does not predict test_1 (14.5px): generalization on
   diverse gauge geometry is the gap.
3. Width (alpha=1.5) beat depth/attention/skip variants in the screen and
   improved tip at full scale vs the deployed v6 (13.5px vs ~20px tip on
   test_1 at screen scale).
4. The deployed tip_focus v18 model (the live board contract) already
   exists; this run confirms the wide-UNet family is the strongest local
   variant of that contract on the expanded dataset.

## Candidates for the tip gap (next experiments)

- Tip-weight tuning (currently 8.0) and/or tip-focused curriculum.
- CenterNet-style offset supervision for the tip relative to center
  (literature: CenterNet, Zhou 2019) — the research doc's section 2.3-2.4.
- Explicit needle-line supervision (WACV 2024: min/max marker + center +
  tip keypoints) if more label formats become available.
