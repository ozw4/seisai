# FBPick Coarse Global-Anchor Resize Design

This document is the current contract for `fbpick-coarse`. It supersedes the
legacy local-crop / tiled-coarse design for the coarse stage.

## Purpose

`fbpick-coarse` gives a global first-break location estimate for each trace. It
is not the final sample-level picker. The coarse output is the large-scale
search center used by the downstream physics / robustification and fine stages.

The primary coarse evaluation question is whether the restored coarse pick falls
inside the later fine-stage search window, not whether it is already the final
sample-accurate answer.

## Input Contract

Global-anchor coarse uses a fixed model input:

- input mode: `global_anchor_resize`
- input shape: `(3, 256, 2048)`
- target shape: `(1, 256, 2048)`
- logits shape: `(B, 1, 256, 2048)`

Input channels are ordered as:

1. `waveform`
2. `offset_ch`
3. `time_ch`

`offset_ch` is built from anchor offsets and normalized by
`NormalizeOffsetByConst`. `time_ch` is built from the raw seconds grid and
normalized by `NormalizeTimeByConst`.

## Train / Validation Flow

Training and validation both start from the full gather:

1. `draw_full_gather`
2. split by offset gaps
3. choose trace anchors inside each segment
4. resample the full raw time axis to 2048 samples
5. build a fixed `3 x 256 x 2048` input and `1 x 256 x 2048` target

Training uses `trace_anchor.train_mode: random`. Validation uses
`trace_anchor.infer_mode: center`, so validation samples are deterministic.

If a gather has fewer than 256 usable anchor rows, the remaining rows are pad
rows. Pad rows are marked invalid by `trace_valid`, have ignored first-break
indices, and must not contribute to target labels or restored predictions.

## Raw Inference Flow

Raw inference does not use the legacy tiled window dataset or W-tiling helpers.
The flow is:

1. load one full raw gather
2. split by offset gaps
3. choose deterministic center anchors
4. build the fixed `3 x 256 x 2048` input
5. run the model once per gather item
6. project anchor predictions from coarse sample coordinates to raw samples
7. restore segment-wise predictions to original trace coordinates
8. write `.coarse.npz`

Current raw inference requires `infer.batch_size == 1` because metadata is
restored per full gather item.

Raw global-anchor coarse inference currently requires exactly one
`dataset.primary_keys` entry. Use `primary_keys: [ffid]` as the recommended
default unless the input SEG-Y line is keyed by another single trace header.

## Offset Gap Handling

Offset gaps split traces into independent segments. Anchor bins are allocated
inside segments, and an anchor bin must not cross a gap.

For example:

```python
offsets = [0, 10, 20, 30, 1000, 1010, 1020]
segments = [(0, 4), (4, 7)]
```

Prediction restoration interpolates only within each segment. No interpolation
or confidence propagation is allowed across offset gaps.

## Time Axis And Time Channel

The coarse time grid is endpoint-aligned between original SEG-Y samples and the
2048-sample coarse axis:

- raw sample `0` maps to coarse sample `0`
- raw sample `raw_time_len - 1` maps to coarse sample `2047`
- coarse sample `0` maps to raw sample `0`
- coarse sample `2047` maps to raw sample `raw_time_len - 1`

For `raw_time_len = 6016`, this means:

- raw `0 -> coarse 0`
- raw `6015 -> coarse 2047`
- coarse `0 -> raw 0`
- coarse `2047 -> raw 6015`

`time_view_sec` is the raw seconds grid after endpoint-aligned resampling.
`build_time_channel` / `MakeTimeChannel` only broadcasts this raw seconds grid
over trace rows; it does not normalize time.

Time normalization is performed by `NormalizeTimeByConst` in the BuildPlan:

```text
time_ch = clip(time_view_sec / time_ref_sec, 0.0, 1.5)
```

Do not normalize by `time_view_sec[-1]`; each gather's final sample must not be
forced to `1.0`.

## `.coarse.npz` Contract

Although model predictions are internal `256 x 2048` coarse-grid predictions,
saved `.coarse.npz` fields are restored to original SEG-Y coordinates.

Required coarse fields include:

- `coarse_pick_i`: original SEG-Y sample index, shape `(n_traces,)`
- `coarse_pick_t_sec`: `coarse_pick_i * dt_sec`, shape `(n_traces,)`
- `coarse_pmax`: trace-wise confidence, shape `(n_traces,)`

The payload also carries trace/sample metadata such as `dt_sec`,
`n_samples_orig`, `n_traces`, `trace_indices`, `ffid_values`, `chno_values`, and
`offsets_m`.

Physics and fine stages consume the restored original-coordinate schema and do
not need schema changes for global-anchor coarse.

## Checkpoint Compatibility

Global-anchor coarse checkpoints must carry:

- `coarse_input_mode == "global_anchor_resize"`
- `coarse_trace_len == 256`
- `coarse_time_len == 2048`
- `coarse_in_chans == 3`
- `coarse_input_channels == ["waveform", "offset_ch", "time_ch"]`

Raw inference validates the global-anchor metadata and rejects missing metadata,
legacy tiled checkpoints, and shape/channel mismatches.

## Migration From Legacy Tiled Coarse

Before:

- local crop / tiled inference
- H-window plus W-tile inference
- arbitrary input width handled by tiling
- coarse raw inference used fields such as `tile_w`, `overlap_w`, and
  `tiles_per_batch`

After:

- full gather context
- gap-aware anchor selection
- fixed `3 x 256 x 2048` input
- deterministic center anchors for validation and raw inference
- segment-wise restoration to original trace/sample coordinates

Old tiled coarse checkpoints are not compatible. Raw global-anchor inference no
longer uses `InferenceGatherWindowsDataset`, `collate_pad_w_right`,
`TiledWConfig`, `iter_infer_loader_tiled_w`, `tile_w`, `overlap_w`, or
`tiles_per_batch`.

## Evaluation Recommendations

Recommended coarse metrics:

- P50 / P90 / P95 absolute error
- coverage within the fine-window half width
- gap-neighborhood error
- trace-wise failure rate

The key coverage check is:

```text
abs(coarse_pick_i - fb_i) <= fine_window_half
```

Use this coverage as the primary coarse-stage regression metric when evaluating
whether the coarse output is good enough for physics / fine refinement.

The config-driven evaluator is:

```bash
python cli/run_fbpick_coarse_eval.py \
  --config examples/config_eval_fbpick_coarse.yaml
```

See `docs/fbpick_coarse_eval.md` for report columns, confidence-bin metrics,
gap-neighborhood metrics, and the exact gap-neighborhood inclusion rule.
