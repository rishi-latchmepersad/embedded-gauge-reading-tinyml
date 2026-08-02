# Data coverage beats architecture and augmentation — 2026-07-30

Date: 2026-07-30
Status: validated
Scope: keypoint model training strategy
Evidence: v4 vs v5 vs v5b vs v6 per-split evaluation

## The finding

Adding 201 in-domain images (temperature gauge) to the training set improved
test_3 tip error by 47% and test_2 tip error by 87%, with no regression on
test_1.  A wider model (2.4M params vs 1M) and augmentation (rotation +
brightness) both failed to match this improvement.

## Rules

1. **Data coverage is the first lever.**  If the model fails on a gauge type
   that is not in the training set, add representative images of that type
   before trying architecture or loss changes.

2. **Augmenting heatmaps is fragile.**  `scipy.ndimage.rotate` on a Gaussian
   heatmap spreads the peak and introduces artefacts.  The correct approach
   is to rotate the image and re-generate the heatmap from the rotated
   keypoint coordinates.  Even with this fix, augmentation did not help on
   this dataset.

3. **Wider models are not always better.**  The v2 wider architecture (2.4M
   params) performed worse than v4 (1M params) on the primary test set.
   For this task, data quality and coverage matter more than model capacity.

4. **Per-split evaluation is essential.**  The combined test set metric
   masked the fact that test_2 and test_3 were completely broken.  Always
   evaluate on each test set separately when the data comes from different
   domains.
