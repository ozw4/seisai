# Mask contract (v0)

This document specifies the masking contract implemented in
`packages/seisai-transforms/src/seisai_transforms/masking.py`.

## Terminology
- `H`: number of traces (height)
- `T`: number of time samples (width). (`W` in other docs is equivalent to `T` here.)
- `mask_bool`: boolean mask of shape `(H, T)` where:
  - `True` means **masked/corrupted/hidden** pixel
  - `False` means untouched pixel

---

## mask_traces_bool(H, T, ratio, rng, width=1)

### Purpose
Generate a **trace-band mask**: selected trace rows are masked for all time samples.

### Inputs
- `H`, `T`: positive integers (`> 0`)
- `ratio`: float in `[0, 1]`
- `width`: int `>= 1`
- `rng`: `np.random.Generator`

### Output
- Type: `np.ndarray`
- Dtype: `bool`
- Shape: `(H, T)`
- Semantics: masked trace rows are `True` for all columns.

### Coverage rule (exact implementation)
- `target = int(round(ratio * H))`  **(fixed as spec)**
- If `target == 0`: returns all-`False`
- If `target >= H` OR `width >= H`: returns all-`True`
- Else:
  - `n_centers = int(ceil(target / width))`
  - `n_centers` is clamped to `[1, H]`
  - `centers = rng.choice(H, size=n_centers, replace=False)` (unique)
  - For each center `c`, mask a contiguous band of exactly `width` rows:
    - initial: `h0 = c - (width // 2)`, `h1 = h0 + width`
    - then shift into bounds so that the band stays within `[0, H)`:
      - if `h0 < 0`: set `(h0, h1) = (0, min(width, H))`
      - if `h1 > H`: set `(h1, h0) = (H, max(0, H - width))`
    - mask: `m[h0:h1, :] = True`
  - Overlap between bands is allowed.

### Notes
- For `width == 1`, the number of masked traces is exactly `target` (unique centers).
- For `width > 1`, the number of masked traces is **approximate** due to overlap.

### Errors
- `H <= 0` or `T <= 0`: `ValueError`
- `ratio` not in `[0,1]`: `ValueError`
- `width < 1`: `ValueError`

---

## CheckerJitterConfig / mask_checkerboard_jitter_bool(H, T, cfg, rng)

### Purpose
Generate a **jittered 2D block mask** on a coarse cell grid.

### CheckerJitterConfig fields
- `block_h`, `block_t`: block size (must be `> 0`)
- `cell_h`, `cell_t`: cell size (must be `> 0`)
- `jitter_h`, `jitter_t`: max jitter (if `<=0`, treated as `0`)
- `keep_prob`: probability in `[0,1]` to place a block in each cell
- `offset_h`, `offset_t`: phase offsets (normalized by modulo of cell sizes)

### Output
- Type: `np.ndarray`
- Dtype: `bool`
- Shape: `(H, T)`
- Semantics: `True` on pixels covered by placed blocks.

### Placement rule (exact implementation)
- Normalize offsets:
  - `off_h = offset_h % cell_h`
  - `off_t = offset_t % cell_t`
- Number of cells scanned (includes one extra cell to cover edges):
  - `n_cells_h = (H + cell_h - 1) // cell_h + 1`
  - `n_cells_t = (T + cell_t - 1) // cell_t + 1`
- For each cell `(gh, gt)`:
  - base position:
    - `base_h = gh * cell_h + off_h`
    - `base_t = gt * cell_t + off_t`
  - keep or skip:
    - if `rng.random() > keep_prob`: skip
  - jitter:
    - `jh = 0` if `jitter_h <= 0` else uniform integer in `[-jitter_h, +jitter_h]` (inclusive)
    - `jt = 0` if `jitter_t <= 0` else uniform integer in `[-jitter_t, +jitter_t]` (inclusive)
  - top-left is clipped to keep the block inside bounds:
    - `h0 = max(0, min(base_h + jh, H - block_h))`
    - `t0 = max(0, min(base_t + jt, T - block_t))`
    - `h1 = h0 + block_h`
    - `t1 = t0 + block_t`
  - apply:
    - `mask[h0:h1, t0:t1] = True`

### Errors
- `H <= 0` or `T <= 0`: `ValueError`
- `block_h <= 0` or `block_t <= 0`: `ValueError`
- `cell_h <= 0` or `cell_t <= 0`: `ValueError`
- `keep_prob` not in `[0,1]`: `ValueError`

---

## MaskGenerator

`MaskGenerator` combines:
- A) a mask generator function `fn(H, T, rng) -> np.ndarray(bool, (H,T))`
- B) a corruption profile (`mode`, `noise_std`) used by `apply()`

### Modes (exact)
- `mode == "replace"`:
  - if `noise_std == 0.0`: masked pixels are replaced by `0.0`
  - else: masked pixels are replaced by noise `N(0, noise_std)`
- `mode == "add"`:
  - if `noise_std == 0.0`: output is a copy of `x` (no change)
  - else: masked pixels have noise added: `x + N(0, noise_std)` on masked pixels

### Constructor
- `MaskGenerator(fn, mode="replace", noise_std=1.0)`
- Validation:
  - `mode` must be `"replace"` or `"add"` else `ValueError`
  - `noise_std >= 0` else `ValueError`

### generate(H, T, rng)
- Calls `m = fn(H, T, rng)`
- Validates:
  - `m.dtype` is boolean
  - `m.shape == (H, T)`
- Returns:
  - `m` (`np.ndarray(bool)` shape `(H, T)`)
- Violations raise `ValueError`

### apply(x, rng, mask=None, return_mask=False)

#### Inputs
- `x`: `np.ndarray` with shape:
  - `(H, T)` or
  - `(C, H, T)` (channel-first)
- `rng`: `np.random.Generator` (**required**)
- `mask` (optional):
  - if provided: must be `np.ndarray(bool)` shape `(H, T)`
  - if omitted: `generate(H, T, rng)` is used
- `return_mask`: bool

#### Shape handling (exact)
- If `x.ndim == 2`:
  - treat as `(1, H, T)` internally and return squeezed back to `(H, T)`
- If `x.ndim == 3`:
  - treat as `(C, H, T)` and return `(C, H, T)`
- Otherwise:
  - `ValueError`

#### Dtype handling (exact)
- `x` is converted to `float32` via `.astype(np.float32, copy=False)`
- Output `Y` is `float32`

#### Mask broadcasting (exact)
- `M = m[None, :, :]` is broadcast over channels to match `(C, H, T)`.

#### Output
- If `return_mask=False`: returns `Y`
- If `return_mask=True`: returns `(Y, m)` where `m` is `(H, T)` boolean

#### Errors
- unsupported `x.ndim`: `ValueError`
- invalid `mask` dtype/shape: `ValueError`
- invalid generator output dtype/shape: `ValueError`

### Factories
- `MaskGenerator.traces(ratio, width=1, mode="replace", noise_std=1.0)`
  - Validates `ratio in [0,1]` and `width >= 1`
  - Uses a closure capturing `ratio` and `width`:
    - `fn(H, T, rng) = mask_traces_bool(H, T, ratio=ratio, width=width, rng=rng)`
  - Implication: changing `ratio` at runtime is done by replacing the generator instance.
- `MaskGenerator.checker_jitter(cfg, mode="replace", noise_std=1.0)`
  - `fn(H, T, rng) = mask_checkerboard_jitter_bool(H, T, cfg=cfg, rng=rng)`

---

## Integration with MaskedSignal / BuildPlan
- `MaskedSignal` stores:
  - `sample[dst] = masked waveform`
  - `sample[mask_key] = mask_bool`
- `mask_bool` is expected to be:
  - `np.ndarray(bool)` shape `(H, T)`
  - `True` indicates masked pixels
