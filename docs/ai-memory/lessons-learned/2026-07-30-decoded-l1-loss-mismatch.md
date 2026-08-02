# Decoded-keypoint L1 loss needs direct GT coordinates — 2026-07-30

Date: 2026-07-30
Status: validated
Scope: keypoint heatmap loss functions
Evidence: v6 vs v9 per-split evaluation

## The finding

Adding a decoded-keypoint L1 loss on top of focal heatmap loss improved
val/test metrics but regressed per-split evaluation. The GT keypoint was
extracted via argmax from the heatmap (non-differentiable), creating a
gradient mismatch with the softargmax decoding used at inference.

## Rules

1. **Loss supervision must match inference decoding.** If the model
   decodes keypoints via softargmax, the L1 loss target must be the
   softargmax of the GT heatmap, not the argmax.  Using argmax creates
   a gradient mismatch that hurts generalisation.

2. **Val/test metrics on a subset can be misleading.** The combined loss
   model scored better on 200-image val/test subsets but worse on the
   full per-split evaluation.  Always evaluate on the full test sets.

3. **Focal heatmap loss alone is sufficient.** For this task, the focal
   loss on heatmaps already produces good keypoint localisation.  Adding
   decoded-keypoint supervision adds complexity without benefit when the
   GT extraction is noisy.
