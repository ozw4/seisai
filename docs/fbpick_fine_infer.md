# FBPick Fine Inference

Fine inference refines first-break picks from one SEG-Y gather using two upstream
artifacts:

- physics / robust output: `*.robust.npz`
- coarse output: `*.coarse.npz`

The current fine inference CLI is a single-gather entrypoint. It requires exactly
one `paths.segy_files` entry and exactly one `paths.robust_npz_files` entry.

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
