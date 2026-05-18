# 2026-05-17 Polar-Vote Top-k Decode Note

Goal:
- Keep the geometric polar-vote CNN path.
- Reach `<5C` MAE on `hard_cases_plus_board30_valid_with_new6.csv`.

What worked:
- Use the existing geometric checkpoint:
  - `ml/artifacts/training/polar_vote_hardcases_errweighted_v1/best_weights.weights.h5`
- Keep the exact matching preprocess:
  - crop boxes: `ml/data/rectified_crop_boxes_v5_all.csv`
  - center mode: `image_center`
  - center search: `0`
  - representation/input: `vote` + `rgb_edge6_vote7`
- Decode with **top-k expectation** instead of full-distribution expectation:
  - `vote_decode_mode=topk_expectation`
  - `vote_decode_temperature=1.0`
  - `vote_decode_topk=8`

Result:
- `extra_hard_cases_plus_board30_valid_with_new6_mae=4.3569C`
- This satisfies the hard-case `<5C` target for the geometric model path.

Implementation change:
- `ml/scripts/train_polar_angle_classifier_manifest.py` now supports:
  - `--vote-decode-mode topk_expectation`
  - `--vote-decode-topk <int>`
- The decode mode is wired through validation MAE callback, test evaluation, and extra-manifest evaluation.
