# fbpick-coarse Global-Anchor Inference

Raw coarse inference uses the global-anchor contract:

- `coarse.input_mode: global_anchor_resize`
- model input shape `(3, 256, 2048)`
- deterministic center anchors
- `infer.batch_size: 1`
- no tiled-W inference

Legacy tiled inference keys are invalid in raw coarse inference configs:
`infer.subset_traces`, `infer.overlap_h`, `infer.tile_w`, `infer.overlap_w`,
and `infer.tiles_per_batch`.

Raw global-anchor coarse inference currently requires exactly one
`dataset.primary_keys` entry. Recommended default:

```yaml
dataset:
  primary_keys: [ffid]
```
