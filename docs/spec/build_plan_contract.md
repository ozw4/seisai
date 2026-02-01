# BuildPlan: Contract (v0)

This document specifies the contract of `BuildPlan` and the minimal expectations for operators (`ops`) and stacks.

## Overview

`BuildPlan` is a minimal pipeline executor that mutates a `sample` dict in-place:

1. run `wave_ops` in order
2. run `label_ops` in order
3. run `input_stack` (SelectStack)
4. run `target_stack` (SelectStack)

## API

### class BuildPlan
Constructor:

- `BuildPlan(wave_ops, label_ops, input_stack, target_stack)`

Method:

- `run(sample: dict[str, Any], rng=None) -> None`

### Execution order

- `for op in wave_ops: op(sample, rng)`
- `for op in label_ops: op(sample, rng)`
- `input_stack(sample, rng)`
- `target_stack(sample, rng)`

## sample dict

### Mutability
- `sample` is a mutable dict, mutated in-place.
- Ops and stacks must write outputs by setting new keys in `sample`.

### Required keys (minimal)
`BuildPlan` itself does not hardcode required keys, but **the following keys are required by the standard ops used in this repo**:

- For most pipelines:
  - `x_view`: waveform view (typically `np.ndarray` of shape `(H,W)`)

- For view-based label/channel builders:
  - `meta`: dict that must include:
    - `time_view`: `np.ndarray(float32)` shape `(W,)`
    - `offsets_view`: `np.ndarray(float32)` shape `(H,)`
    - `fb_idx_view`: `np.ndarray(int64)` shape `(H,)`
    - `trace_valid`: `np.ndarray(bool)` shape `(H,)`

> Note: Additional keys may be present (e.g., `dt_sec`, `offsets`, `fb_idx`, `indices`, etc.).
> They are not required by BuildPlan core, but can be consumed by custom ops.

## Operator (op) contract

### Signature
- Each op is a callable:
  - `op(sample: dict[str, Any], rng=None) -> None`

### Behavior
- An op may read existing keys and must write derived outputs into `sample`.
- Ops must not silently change the meaning of existing keys. (Overwriting is allowed only if explicitly intended.)
- Missing prerequisites should raise:
  - `KeyError` for missing keys
  - `ValueError` for shape/type mismatches

### RNG
- `rng` is passed through from `BuildPlan.run(...)`.
- Ops may treat `rng` as `np.random.Generator` (recommended).
- Ops may fall back to a local generator if `rng is None`.

## SelectStack contract (input_stack / target_stack)

### Purpose
`SelectStack(keys, dst, dtype, to_torch)` collects 2D/3D arrays and concatenates them into a channel-first tensor.

### Inputs
For each `k in keys`, `sample[k]` must be:
- `np.ndarray` or `torch.Tensor`
- with shape either:
  - 2D: `(H,W)`  -> automatically expanded to `(1,H,W)`
  - 3D: `(C,H,W)` -> kept as-is

Any other dimensionality raises `ValueError`.

### Shape consistency
- All stacked items must share the same `(H,W)`.
- Mismatch raises `ValueError`.

### Output
- Writes `sample[dst] = out`
- `out` shape is always `(C_total, H, W)`
- `dtype` is applied via numpy casting before concatenation (default: float32)
- If `to_torch=True`, output is `torch.Tensor` on CPU (`torch.from_numpy`)

### Integration requirement with SegyGatherPipelineDataset
When used inside `SegyGatherPipelineDataset`, the plan must produce:
- `sample['input']` and `sample['target']`
and both must be **3D `(C,H,W)`** (this is naturally satisfied by SelectStack).

## Standard ops in this repo (behavior summary)

### IdentitySignal(src='x_view', dst='x_id', copy=False)
- Requires: `sample[src]`
- Writes: `sample[dst]`
- If `copy=True`, makes a deep copy (numpy copy / torch clone).

### MaskedSignal(generator, src='x_view', dst='x_masked', mask_key='mask_bool')
- Requires: `sample[src]`
- Writes:
  - `sample[dst]`: masked waveform
  - `sample[mask_key]`: `np.ndarray(bool)` shape `(H,W)` (mask)

### MakeTimeChannel(dst='time_ch')
- Requires: `sample['x_view']` for `(H,W)`, `sample['meta']['time_view']` for `(W,)`
- Writes: `sample[dst]`: `np.ndarray(float32)` shape `(H,W)` (time channel)

### MakeOffsetChannel(dst='offset_ch', normalize=True)
- Requires:
  - `sample['x_view']` with shape `(H, W)`
  - `sample['meta']['offsets_view']` with shape `(H,)` (float32-castable)
  - `sample['meta']['trace_valid']` with shape `(H,)` (bool; `True` for real traces, `False` for padded)
- Writes:
  - `sample[dst]`: `np.ndarray(float32)` shape `(H, W)` (offset channel)
    - each row `i` is filled with the (optionally normalized) offset for that trace
- Normalization (`normalize=True`):
  - z-score is computed **using valid traces only** (`trace_valid == True`)
    - `m = mean(offsets_view[valid])`
    - `s = std(offsets_view[valid]) + 1e-6`
    - `offset_z[valid] = (offsets_view[valid] - m) / s`
  - invalid traces (`trace_valid == False`) are forced to `0`
  - if there are no valid traces, all rows become `0`
- Errors:
  - missing keys: `KeyError`
  - length mismatch between `offsets_view` and `trace_valid`: `ValueError`

### FBGaussMap(dst='fb_map', sigma=1.5, src='fb_idx_view')
- Requires:
  - `sample['x_view']` for `(H,W)`
  - `sample['meta'][src]` for `(H,)`
- Valid fb rule (current implementation):
  - valid = `fb > 0` (invalid includes `<=0` and `-1`)
- Writes: `sample[dst]`: `np.ndarray(float32)` shape `(H,W)`

## Failure modes (expected)
- Missing prerequisite keys -> `KeyError`
- Wrong shape (e.g., fb length != H) -> `ValueError`
- SelectStack receives non 2D/3D -> `ValueError`
- If the caller expects `input/target` but stacks write different `dst` -> caller may raise `KeyError`

## Extension guidance
- Custom ops may introduce additional keys, but should follow the same signature and error behavior.
- Prefer producing numpy arrays prior to SelectStack; SelectStack handles conversion to torch.
