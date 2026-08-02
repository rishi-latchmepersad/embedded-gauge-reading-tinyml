# from_tensor_slices OOMs on multi-GB training sets — stream with from_generator

Date: 2026-07-31
Status: validated
Scope: TensorFlow training data pipelines on the WSL ML box
Evidence: iteration 4 launch crashes (09:11 `Dst tensor is not initialized`;
10:20 kernel OOM kill at 53 GB anon RSS, `dmesg`: `Out of memory: Killed
process 3490 (python) total-vm:171GB, anon-rss:53GB`)

## The findings

1. `tf.data.Dataset.from_tensor_slices` stages the ENTIRE tensor set into the
   GPU allocator. Fine for 14,008 samples; at 25,650 samples (~16 GB float32
   images + heatmap targets) the copy onto the 15 GB GPU cap fails with
   `Dst tensor is not initialized`.

2. `from_generator` alone is NOT enough. The first fixed run still died:
   - `make_scale_augmented_training_set` doubles the set in RAM (~+7.6 GB)
   - a redundant `tf.image.resize` on already-384² images materializes
     another full copy (~+15 GB)
   - `.shuffle(len(images))` on a generator dataset buffers the WHOLE
     dataset in host RAM (~+17.7 GB)
   - total ~50 GB + TF runtime → kernel OOM killer → entire WSL died.

## The fix (full memory-safe stack)

```python
# 1) store images as uint8 (147 KB/sample at 384², not 589 KB float32)
images_u8 = np.clip(np.round(images * 255.0), 0, 255).astype(np.uint8)
# 2) cap the shuffle buffer so it cannot re-materialize the dataset
.shuffle(min(SHUFFLE_BUFFER, len(images_u8)), seed=SEED)  # SHUFFLE_BUFFER = 4096
# 3) convert to float32 per-sample inside the from_generator closure
yield images_u8[i].astype(np.float32) / 255.0, targets[i]
# 4) preflight estimate before allocating; SystemExit if > 40 GiB budget
```

Reference implementation: `ml/scripts/train_ellipse_multiscale_universal_384.py`
(`_augment_uint8`, `_memory_preflight`, `SHUFFLE_BUFFER`, `MEMORY_BUDGET_MB`).

## Rules

1. Every ML job must launch through `ml/scripts/run_wsl_guarded.sh` (flock +
   `MemAvailable` floor + TERM before the kernel OOM killer). Plain
   `setsid poetry run ...` bypasses the guard and can kill the whole WSL box.
2. Store training images as uint8; convert to float32 per-sample inside the
   generator closure. Never hold float32 copies of the whole set.
3. Cap the shuffle buffer at ~4096; a full-size buffer re-materializes the
   dataset in RAM.
4. Add `_memory_preflight` to every training script: estimate from real
   sample counts (images × bytes + targets × bytes + shuffle buffer) and
   `raise SystemExit` with a readable message before allocating.
5. Avoid redundant whole-set ops (e.g. no-op `tf.image.resize`); build
   export representative datasets inside the converter generator.
6. Check `free -g` and `nvidia-smi` before and during training; host RAM
   must stay under 50 GB (WSL reports ~52 GB total).
