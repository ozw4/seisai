# Training Pipeline Output Layout (Pair/PSN/Blindtrace)

This document defines the shared output layout and fixed-sample constraints for the
training pipelines in:

- `seisai_engine.pipelines.pair.train`
- `seisai_engine.pipelines.psn.train`
- `seisai_engine.pipelines.blindtrace.train`

## Output Layout

All three pipelines write outputs under `paths.out_dir` (resolved relative to the
YAML location when a relative path is provided).

- Checkpoint (best only):
  - `out_dir/ckpt/best.pt`
- Visualizations (epoch/step-based):
  - `out_dir/<vis.out_subdir>/epoch_####/step_####.png`

`vis.out_subdir` defaults to `vis` if not specified.

## Fixed Inference Samples

To keep inference samples fixed across epochs and ensure reproducibility:

- `infer.num_workers` must be **0**.
- Before each inference pass, the inference dataset RNG is reset using
  `infer.seed`.
- The inference subset is taken deterministically as:
  `Subset(ds_infer_full, range(infer.batch_size * infer.max_batches))`.

## Inference Loss Aggregation

`infer_loss` is computed as a **batch-size–weighted average** over inference
batches:

```
weighted_mean = sum(loss_i * batch_size_i) / sum(batch_size_i)
```

The best checkpoint is updated when `infer_loss` improves (minimization).
