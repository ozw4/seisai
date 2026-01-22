# SegyGatherPipelineDataset: Output contract (v0)

This document specifies the contract of a single sample returned by `SegyGatherPipelineDataset.__getitem__`.

## Returned object
- Type: `dict[str, Any]`

## Required keys

### input
- Type: `torch.Tensor` (CPU)
- Dtype: `torch.float32`
- Shape: `(C_in, H, W)`
- Requirement: **always 3D**. Even single-channel inputs must be `(1, H, W)`.
- Meaning: model input tensor (built by `plan.input_stack`).

### target
- Type: `torch.Tensor` (CPU)
- Dtype: `torch.float32`
- Shape: `(C_tgt, H, W)`
- Requirement: **always 3D**. Even single-channel targets must be `(1, H, W)`.
- Meaning: training target tensor (built by `plan.target_stack`).

### trace_valid
- Type: `torch.Tensor` (CPU)
- Dtype: `torch.bool`
- Shape: `(H,)`
- Meaning: per-trace validity mask. `True` for real traces, `False` for padded traces.

### fb_idx
- Type: `torch.Tensor` (CPU)
- Dtype: `torch.int64`
- Shape: `(H,)`
- Meaning: raw (pre-view) first-break sample indices aligned to traces after padding.
  - Invalid rule: `fb <= 0` is invalid.
  - Padding value: `-1`.

### offsets
- Type: `torch.Tensor` (CPU)
- Dtype: `torch.float32`
- Shape: `(H,)`
- Meaning: per-trace offsets aligned to traces after padding.
  - If offsets are missing/broken in SEG-Y, they are treated as all-zero.
  - Padding value: `0`.

### dt_sec
- Type: `torch.Tensor` (CPU scalar)
- Dtype: `torch.float32`
- Shape: `()`
- Meaning: effective sampling interval in seconds used for this sample (`dt_eff_sec` if time scaling is applied).

### indices
- Type: `np.ndarray`
- Dtype: `np.int64`
- Shape: `(H,)`
- Meaning: original trace indices in the SEG-Y file (trace order indices) aligned after padding.
  - Padding value: `-1`.
  - IMPORTANT: `indices` must NOT be used directly for array indexing without filtering by `trace_valid`
    (because `-1` would silently index the last element in NumPy).

### meta
- Type: `dict[str, Any]`
- Presence: **always returned**
- Minimum required fields (keys must exist):
  - `time_view`: `np.ndarray` float32, shape `(W,)`
  - `offsets_view`: `np.ndarray` float32, shape `(H,)`
  - `fb_idx_view`: `np.ndarray` int64, shape `(H,)` (invalid rule: `<= 0 -> -1`)
  - `dt_eff_sec`: `float`
  - `trace_valid`: `np.ndarray` bool, shape `(H,)`
- Additional fields may exist depending on transform and pipeline:
  - transform-related fields (subset may exist): `hflip: bool`, `factor: float`, `factor_h: float`, `start: int`
  - sampling identifiers: `key_name`, `primary_unique`, etc.

### file_path
- Type: `str`
- Meaning: source SEG-Y file path.

### key_name
- Type: `str`
- Meaning: which primary key was used for sampling (e.g., `ffid`, `cmp`, ...).

### secondary_key
- Type: `Any` (typically `int` or `str`)
- Meaning: secondary key value used by sampler (for grouping/sub-selection).

### primary_unique
- Type: `Any` (typically `int` or `str`)
- Meaning: unique identifier of the sampled primary group.

### did_superwindow
- Type: `bool`
- Meaning: whether the sample was drawn using a superwindow strategy.

## Optional keys

### mask_bool
- Type: `np.ndarray`
- Dtype: `bool`
- Shape: `(H, W)`
- Meaning: boolean mask produced by masking ops (e.g., `MaskedSignal`).
  - `True` indicates masked (corrupted/hidden) pixels.
- Presence: only if `plan` populates it (e.g., via `MaskedSignal(mask_key="mask_bool")`).

## Invariants
- `input.shape[1] == target.shape[1] == trace_valid.shape[0] == fb_idx.shape[0] == offsets.shape[0] == indices.shape[0] == H`
- `input.shape[2] == target.shape[2] == W`
- `meta["time_view"].shape == (W,)`
- `meta["offsets_view"].shape == (H,)`
- `meta["fb_idx_view"].shape == (H,)`
- If `mask_bool` exists: `mask_bool.shape == (H, W)`

## DataLoader collation note
- Default PyTorch collation will batch tensors into shapes like:
  - `input`: `(B, C_in, H, W)`
  - `target`: `(B, C_tgt, H, W)`
  - `trace_valid`: `(B, H)`
- `meta` is a nested dict; default collation may create a dict of batched fields.
  If `meta` is too heavy, use a custom `collate_fn` to drop or simplify it.
