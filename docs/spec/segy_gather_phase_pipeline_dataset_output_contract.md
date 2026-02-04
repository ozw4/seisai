# SegyGatherPhasePipelineDataset: Output Contract (v0)

This document specifies the contract of a single sample returned by `SegyGatherPhasePipelineDataset.__getitem__`.

This dataset is the phase-pick CSR counterpart of `SegyGatherPipelineDataset` and is designed to be backward-compatible with FB-centric pipelines by treating **P-first** as `fb_idx`.

## Returned object
- Type: `dict[str, Any]`

## Required keys

### input
- Type: `torch.Tensor` (CPU)
- Dtype: `torch.float32`
- Shape: `(C_in, H, W)`
- Meaning: model input tensor built by `plan.input_stack`.

### target
- Type: `torch.Tensor` (CPU)
- Dtype: `torch.float32`
- Shape: `(C_tgt, H, W)`
- Meaning: training target tensor built by `plan.target_stack`.

When using `PhasePSNMap(dst="psn_map")` and `SelectStack(keys="psn_map", dst="target")`:
- `C_tgt == 3`
- Channel order is `[P, S, Noise]`

### trace_valid
- Type: `torch.Tensor` (CPU)
- Dtype: `torch.bool`
- Shape: `(H,)`
- Meaning: per-trace validity mask. `True` for real traces, `False` for padded traces.

### fb_idx
- Type: `torch.Tensor` (CPU)
- Dtype: `torch.int64`
- Shape: `(H,)`
- Meaning: legacy-compatible alias for **P-first** pick indices (raw, pre-view), aligned after padding.
  - Valid rule: `> 0` is valid.
  - No-pick (real trace): `0`.
  - Padding value: `-1`.

### p_idx
- Type: `torch.Tensor` (CPU)
- Dtype: `torch.int64`
- Shape: `(H,)`
- Meaning: same as `fb_idx` (P-first).

### s_idx
- Type: `torch.Tensor` (CPU)
- Dtype: `torch.int64`
- Shape: `(H,)`
- Meaning: **S-first** pick indices (raw, pre-view), aligned after padding.
  - Valid rule: `> 0` is valid.
  - No-pick (real trace): `0`.
  - Padding value: `-1`.

Notes:
- S invalidation rule is applied before this output is created:
  - if `s_first < p_first` on a trace, all S picks for that trace are cleared and `s_idx` becomes `0`.

### label_valid
- Type: `torch.Tensor` (CPU)
- Dtype: `torch.bool`
- Shape: `(H,)`
- Meaning: loss mask for labels.

This dataset requires the plan to populate `label_valid` (for example, `PhasePSNMap` produces it).

For `PhasePSNMap`:
- `label_valid[t]` is `True` only when:
  - `meta["trace_valid"][t]` is `True`, and
  - the trace has at least one valid P or S pick after view projection (out-of-range picks are dropped in view space)

### offsets
- Type: `torch.Tensor` (CPU)
- Dtype: `torch.float32`
- Shape: `(H,)`
- Meaning: per-trace offsets aligned after padding.
  - Padding value: `0`.

### dt_sec
- Type: `torch.Tensor` (CPU scalar)
- Dtype: `torch.float32`
- Shape: `()`
- Meaning: effective sampling interval in seconds used for this sample (`dt_eff_sec`).

### indices
- Type: `np.ndarray`
- Dtype: `np.int64`
- Shape: `(H,)`
- Meaning: original trace indices in the SEG-Y file aligned after padding.
  - Padding value: `-1`.
  - IMPORTANT: do not use `indices` directly for NumPy indexing without filtering by `trace_valid` (because `-1` would index the last element).

### meta
- Type: `dict[str, Any]`
- Presence: always returned

Minimum required fields (keys must exist):
- `time_view`: `np.ndarray` float32, shape `(W,)`
- `offsets_view`: `np.ndarray` float32, shape `(H,)`
- `fb_idx_view`: `np.ndarray` int64, shape `(H,)` (invalid rule: `<= 0` or `>= W` becomes `-1`)
- `p_idx_view`: `np.ndarray` int64, shape `(H,)` (same values as `fb_idx_view`, but stored as an independent array)
- `s_idx_view`: `np.ndarray` int64, shape `(H,)`
- `dt_eff_sec`: `float`
- `trace_valid`: `np.ndarray` bool, shape `(H,)`

Additional fields may exist depending on the transform and pipeline:
- transform-related fields (subset may exist): `hflip`, `factor`, `factor_h`, `start`
- sampling identifiers: `key_name`, `primary_unique`

### file_path
- Type: `str`
- Meaning: source SEG-Y file path.

### key_name
- Type: `str`
- Meaning: which primary key was used for sampling (for example: `ffid`, `chno`, `cmp`).

### secondary_key
- Type: `str`
- Meaning: name of the secondary sort key chosen by the sampler (typically one of `ffid`, `chno`, `offset`).

### primary_unique
- Type: `str`
- Meaning: comma-joined unique primary key values present in the sampled subset (for logging and debugging).

### did_superwindow
- Type: `bool`
- Meaning: whether the sample was drawn using a superwindow strategy.

## Optional keys

### mask_bool
- Type: `np.ndarray`
- Dtype: `bool`
- Shape: `(H, W)`
- Meaning: boolean mask produced by masking ops (for example `MaskedSignal(mask_key="mask_bool")`).

## Empty-gather behavior

The dataset has an `include_empty_gathers` option:
- `include_empty_gathers=False` (default): rejects subsets where both P and S picks are absent and resamples.
- `include_empty_gathers=True`: returns such subsets and skips FB-based quality gates for those empty samples.

When using `PhasePSNMap`, an empty sample produces:
- `label_valid` is all `False`
- `target` has `Noise == 1` for every pixel

## Invariants
- `input.shape[1] == target.shape[1] == trace_valid.shape[0] == fb_idx.shape[0] == p_idx.shape[0] == s_idx.shape[0] == label_valid.shape[0] == offsets.shape[0] == indices.shape[0] == H`
- `input.shape[2] == target.shape[2] == W`
- `meta["time_view"].shape == (W,)`
- `meta["offsets_view"].shape == (H,)`
- `meta["fb_idx_view"].shape == (H,)`
- `meta["p_idx_view"].shape == (H,)`
- `meta["s_idx_view"].shape == (H,)`

## DataLoader collation note

Default PyTorch collation will batch tensors into shapes like:
- `input`: `(B, C_in, H, W)`
- `target`: `(B, C_tgt, H, W)`
- `trace_valid`: `(B, H)`

`meta` is a nested dict and may be heavy. Use a custom `collate_fn` if you want to drop or simplify metadata.
