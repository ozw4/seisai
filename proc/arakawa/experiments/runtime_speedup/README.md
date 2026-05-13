# Arakawa runtime speedup experiment

The source configs for this experiment live only in:

```text
proc/arakawa/experiments/runtime_speedup/configs/
```

Run commands from the repository root. Each config writes generated configs and
runtime artifacts under the ignored output tree:

```text
proc/arakawa/outputs/runtime_runs/<RUN_NAME>/
```

Run `A0_full` first. The other configs point `paths.coarse_dir` at
`A0_full/coarse` so the comparison focuses on the physics stage.

## Benchmark harness

The checked-in manifest runs the configured candidates, compares each robust
NPZ against A0, evaluates per-candidate gates, and writes combined
JSON/CSV/Markdown reports. Candidate `gates` override manifest-level scalar
and list gates; dict-valued gates are merged by key. A1/A2 use exact-match
gates against A0; anchor reuse and
observation-sampling stages use the looser tolerance gates recorded next to
those candidate entries.

```bash
python -m cli.run_fbpick_physics_runtime_benchmark \
  --manifest proc/arakawa/experiments/runtime_speedup/benchmark_manifest.yaml \
  --tag Arakawa2026__fdata_hset_ARA26_Vib \
  --out-dir proc/arakawa/outputs/runtime_benchmark/A0_manifest
```

To compare already-created artifacts without running the configs, add
`--artifacts-only`. The command exits non-zero when any gate fails; add
`--no-fail-on-gate` for exploratory report-only runs.

Report files:

```text
proc/arakawa/outputs/runtime_benchmark/<RUN>/summary.json
proc/arakawa/outputs/runtime_benchmark/<RUN>/summary.csv
proc/arakawa/outputs/runtime_benchmark/<RUN>/summary.md
proc/arakawa/outputs/runtime_benchmark/<RUN>/comparisons/*.json
proc/arakawa/outputs/runtime_benchmark/<RUN>/comparisons/*.csv
```

## Configs

| config | purpose |
|---|---|
| `A0_full` | Full physics fit baseline with runtime diagnostics enabled. |
| `A1_diagnostics_only` | Same settings as A0, used to check instrumentation and run-to-run stability. |
| `A2_anchor_selection_dry_run` | Full fitting with source-XY anchor-selection diagnostics; non-anchor gathers are not reused. |
| `A3_anchor_stride5_nearest_anchor` | Fits every 5th source group and reuses the nearest compatible anchor. |
| `A4_anchor_stride5_t0_shift` | Adds median t0-shift estimation for nearest-anchor reuse. |
| `A5_anchor_stride5_t0_shift_adaptive_refit` | Adds adaptive full refit when reuse residual or shift thresholds are large. |
| `A6_A5_obs_downsample256` | A5 plus offset-bin observation downsampling with at most 256 observations per fit. |
| `A0D_downsample_only` | Full fitting with anchor reuse disabled and observation downsampling enabled. |

## Run

```bash
python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/experiments/runtime_speedup/configs/A0_full.yaml

python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/experiments/runtime_speedup/configs/A1_diagnostics_only.yaml

python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/experiments/runtime_speedup/configs/A3_anchor_stride5_nearest_anchor.yaml

python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/experiments/runtime_speedup/configs/A5_anchor_stride5_t0_shift_adaptive_refit.yaml

python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/experiments/runtime_speedup/configs/A6_A5_obs_downsample256.yaml

python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/experiments/runtime_speedup/configs/A0D_downsample_only.yaml
```

## Compare

For the default example data, `<TAG>` is
`Arakawa2026__fdata_hset_ARA26_Vib`.

```bash
TAG=Arakawa2026__fdata_hset_ARA26_Vib
mkdir -p proc/arakawa/outputs/runtime_compare

python -m cli.compare_fbpick_physics_runtime \
  --baseline proc/arakawa/outputs/runtime_runs/A0_full/robust/${TAG}.robust.npz \
  --candidate proc/arakawa/outputs/runtime_runs/A3_anchor_stride5_nearest_anchor/robust/${TAG}.robust.npz \
  --out-json proc/arakawa/outputs/runtime_compare/A3_vs_A0.json \
  --out-csv proc/arakawa/outputs/runtime_compare/A3_vs_A0.csv
```
