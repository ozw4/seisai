# FBPick Fine Inference

Fine inference refines first-break picks from one SEG-Y gather using two upstream
artifacts:

- physics / robust output: `*.robust.npz`
- coarse output: `*.coarse.npz`

The current fine inference CLI is a single-gather entrypoint. It requires exactly
one `paths.segy_files` entry and exactly one `paths.robust_npz_files` entry.

## Window Center

Fine inference reads the local-window center from the robust npz. The
recommended config uses `fine_center_i` when present and falls back to
`robust_pick_i` for older robust artifacts:

```yaml
window_center:
  npz_key: fine_center_i
  fallback_npz_key: robust_pick_i
```

Physics output writes `fine_center_i` even when physical trend centering is not
used, so new robust artifacts should provide the fine-stage center explicitly.

## Coarse NPZ Input

Set `paths.coarse_npz_files` when the coarse and physics outputs live in
different directories:

```yaml
paths:
  segy_files:
    - /path/to/sample.sgy
  robust_npz_files:
    - /workspace/proc/fbpick/site54/fbpick_physics_valid_out/sample.robust.npz
  coarse_npz_files:
    - /workspace/proc/fbpick/site54/fbpick_coarse_infer_valid_out/sample.coarse.npz
  out_dir: /workspace/proc/fbpick/site54/fbpick_fine_infer_valid_out
```

`paths.coarse_npz_files` accepts the same forms as `paths.segy_files` and
`paths.robust_npz_files`: either a list of paths or a listfile path. When it is
provided, its length must match `paths.segy_files` and `paths.robust_npz_files`.
For the current single-gather CLI, all three lists must contain one path.

If `paths.coarse_npz_files` is omitted, fine inference keeps the legacy behavior
and infers the coarse path from the robust path:

```text
/path/to/sample.robust.npz -> /path/to/sample.coarse.npz
```

Use explicit `paths.coarse_npz_files` for workflows where:

```text
coarse inference output:
  fbpick_coarse_infer_valid_out/*.coarse.npz

physics output:
  fbpick_physics_valid_out/*.robust.npz
```

In that layout, the inferred same-directory coarse path is wrong, so fine
inference should point to both upstream outputs explicitly.

## Viewer QC

Fine inference can write gather-based QC PNGs without materializing the full
SEG-Y waveform:

```yaml
viewer:
  enabled: true
  save_overview_png: false
  save_gather_png: true
  max_gathers_per_file: 8
  skip_gather_keys:
    ffid: [0]
  max_traces_per_gather: 10000
  waveform_norm: per_trace
  dpi: 150
  clip_percentile: 99.0
```

Gather QC uses `dataset.primary_keys` such as `ffid`, skips configured keys and
oversized gathers before reading waveform traces, and writes a limited set of
per-gather PNGs under `<out_dir>/<parent>__<stem>.fine_qc/`.

The legacy `save_overview_png` path draws all traces from the full SEG-Y file.
For large production datasets, keep `save_overview_png: false` and use
gather-based QC.
