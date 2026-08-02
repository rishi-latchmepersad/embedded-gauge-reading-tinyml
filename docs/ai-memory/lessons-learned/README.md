# Lessons learned

This folder is for durable principles extracted from experiments or incidents.
Move a rule here only when supported by evidence; keep model-specific numbers
in `model-updates/` and symptom-specific procedures in `troubleshooting/`.

- [`2026-07-31-from-tensor-slices-oom.md`](2026-07-31-from-tensor-slices-oom.md) — stream multi-GB training sets with `from_generator`, not `from_tensor_slices`.
- [`2026-07-31-no-wsl-restart-opencode.md`](2026-07-31-no-wsl-restart-opencode.md) — never restart WSL while opencode is running.
- [`2026-07-30-data-coverage-beats-arch.md`](2026-07-30-data-coverage-beats-arch.md) — add in-domain images before changing architecture or loss.
- [`2026-07-30-decoded-l1-loss-mismatch.md`](2026-07-30-decoded-l1-loss-mismatch.md) — loss supervision must match inference decoding.
- [`2026-07-23-qat-safe-architecture.md`](2026-07-23-qat-safe-architecture.md) — keep QAT-safe layers for int8 export.
- [`2026-07-23-tfmot-qat-layer-compat.md`](2026-07-23-tfmot-qat-layer-compat.md) — tfmot layer compatibility notes.
- [`2026-07-23-crop-pipeline-consistency.md`](2026-07-23-crop-pipeline-consistency.md) — keep the ellipse model identical between training and inference crops.
- [`2026-07-23-linear-radius-head.md`](2026-07-23-linear-radius-head.md) — linear radius head behavior.
- [`2026-07-22-heatmap-loss-weights.md`](2026-07-22-heatmap-loss-weights.md) — balanced focal heatmap loss weights beat extreme tip-weighting.
