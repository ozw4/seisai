# Arakawa canonical layout

This file defines the canonical `proc/arakawa` layout. New user-facing docs and
configs should use these paths.

```text
proc/arakawa/
  README.md
  README_RUNTIME_SPEEDUP.md
  MIGRATION.md
  README_LAYOUT.md

  configs/
    run_coarse_physics_export_minimal.yaml
    run_coarse_physics_export.yaml
    templates/
      coarse.yaml
      physics.yaml
      physics_qc_no_fb.yaml
      fine.yaml

  experiments/
    runtime_speedup/
      configs/
        A0_full.yaml
        A1_diagnostics_only.yaml
        A2_anchor_selection_dry_run.yaml
        A3_anchor_stride5_nearest_anchor.yaml
        A4_anchor_stride5_t0_shift.yaml
        A5_anchor_stride5_t0_shift_adaptive_refit.yaml
        A6_A5_obs_downsample256.yaml
        A0D_downsample_only.yaml
      README.md

  reference/
    README.md
    # user-provided grstat files may live here locally; ignored by git.

  scripts/
    # proc-local helpers only when a root cli/ entrypoint is not appropriate.

  outputs/
    coarse/
    robust/
    fine/
    grstat/
    qc/
    qc_no_fb/
    eval/
    fb_dummy/
    generated_configs/
    runtime_runs/
    runtime_compare/
    summaries/
```

## Contract

- `configs/run_coarse_physics_export_minimal.yaml` is the primary user-edited
  config.
- `configs/run_coarse_physics_export.yaml` is the full commented user config.
- `configs/templates/*.yaml` are runner templates, not per-run generated files.
- Legacy `configs/*_one.yaml` files are deprecated comment-only stubs.
- `experiments/runtime_speedup/configs/*.yaml` are development benchmark configs.
- `reference/` is for local user-provided grstat files. Only
  `reference/README.md` is tracked.
- `outputs/` is ignored and owns generated artifacts, including generated
  configs, runtime benchmark runs, QC images, summaries, NPZ, CRD, JSON, and CSV.
- New README examples should use only the canonical paths above.

## Legacy path mapping

| legacy path | canonical path |
|---|---|
| `proc/arakawa/configs/coarse_one.yaml` | `proc/arakawa/configs/templates/coarse.yaml` |
| `proc/arakawa/configs/physics_one.yaml` | `proc/arakawa/configs/templates/physics.yaml` |
| `proc/arakawa/configs/physics_qc_one_no_fb.yaml` | `proc/arakawa/configs/templates/physics_qc_no_fb.yaml` |
| `proc/arakawa/configs/fine_one.yaml` | `proc/arakawa/configs/templates/fine.yaml` |
| `proc/arakawa/configs/runtime_speedup/*.yaml` | `proc/arakawa/experiments/runtime_speedup/configs/*.yaml` |
| `proc/arakawa/coarse/` | `proc/arakawa/outputs/coarse/` |
| `proc/arakawa/robust/` | `proc/arakawa/outputs/robust/` |
| `proc/arakawa/fine/` | `proc/arakawa/outputs/fine/` |
| `proc/arakawa/grstat/` | `proc/arakawa/outputs/grstat/` |
| `proc/arakawa/qc/` | `proc/arakawa/outputs/qc/` |
| `proc/arakawa/qc_no_fb/` | `proc/arakawa/outputs/qc_no_fb/` |
| `proc/arakawa/eval/` | `proc/arakawa/outputs/eval/` |
| `proc/arakawa/fb_dummy/` | `proc/arakawa/outputs/fb_dummy/` |
| `proc/arakawa/generated_configs/` | `proc/arakawa/outputs/generated_configs/` |
| `proc/arakawa/runtime_runs/` | `proc/arakawa/outputs/runtime_runs/` |
| `proc/arakawa/runtime_compare/` | `proc/arakawa/outputs/runtime_compare/` |

## Template Resolution

When `paths.coarse_template` or `paths.physics_template` is omitted, the
Arakawa one-shot runner uses only these canonical template paths:

```text
coarse:  proc/arakawa/configs/templates/coarse.yaml
physics: proc/arakawa/configs/templates/physics.yaml
```

Explicit `paths.coarse_template` and `paths.physics_template` values are
resolved as written. Deprecated legacy config names are not used as implicit
fallback templates.
