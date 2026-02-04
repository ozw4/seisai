# Phase Pick Files Spec (CSR `.npz`)

This document defines the CSR-based phase pick file format consumed by `seisai_dataset` (for `SegyGatherPhasePipelineDataset`).

## Overview

A phase pick file is a NumPy `.npz` archive that stores per-trace variable-length pick lists for:
- P phase (`p_*`)
- S phase (`s_*`)

The representation is a simple CSR-style layout: an `indptr` array (row pointers) plus a flat `data` array.

Pick values follow the same validity convention as existing FB picks:
- `pick <= 0` is invalid (missing)
- `pick > 0` is valid

## Required keys

The `.npz` must contain the following keys:
- `p_indptr`: `np.ndarray` of integers, shape `(n_traces + 1,)`
- `p_data`: `np.ndarray` of integers, shape `(nnz_p,)`
- `s_indptr`: `np.ndarray` of integers, shape `(n_traces + 1,)`
- `s_data`: `np.ndarray` of integers, shape `(nnz_s,)`

`n_traces` must match the number of traces in the corresponding SEG-Y file.

## CSR interpretation

For each trace index `t` in `[0, n_traces)`:
- P picks for trace `t` are: `p_data[p_indptr[t] : p_indptr[t + 1]]`
- S picks for trace `t` are: `s_data[s_indptr[t] : s_indptr[t + 1]]`

An empty trace (no picks) is represented by an empty slice:
- `indptr[t] == indptr[t + 1]`

The order of values inside each slice is not required to be sorted.

## Validation rules (as enforced by `seisai_dataset.phase_pick_io`)

For both P and S (with the same `n_traces`):
- `indptr` is 1D
- `indptr` has an integer dtype
- `len(indptr) == n_traces + 1`
- `indptr[0] == 0`
- `indptr` is monotonic non-decreasing
- `indptr[-1] == len(data)`
- `data` is 1D
- `data` has an integer dtype

In addition:
- P and S must share the same `n_traces` (same `indptr` length).

At load time, arrays are normalized to `int64` after validation.

## Pick value convention

Each pick value is an integer sample index on the raw time axis of the SEG-Y trace, using this rule:
- `pick <= 0` is invalid
- `pick > 0` is valid

Downstream utilities compute first picks robustly even if the slice order is arbitrary:
- `p_first[t] = min({picks > 0})` or `0` if none
- `s_first[t] = min({picks > 0})` or `0` if none

## Example

Example for `n_traces = 3`:
- trace 0: P picks `[10, 20]`, S picks `[]`
- trace 1: P picks `[]`, S picks `[30]`
- trace 2: P picks `[5]`, S picks `[]`

```python
import numpy as np

p_indptr = np.array([0, 2, 2, 3], dtype=np.int64)
p_data = np.array([10, 20, 5], dtype=np.int64)

s_indptr = np.array([0, 0, 1, 1], dtype=np.int64)
s_data = np.array([30], dtype=np.int64)

np.savez_compressed(
	"example_phase_picks.npz",
	p_indptr=p_indptr,
	p_data=p_data,
	s_indptr=s_indptr,
	s_data=s_data,
)
```
