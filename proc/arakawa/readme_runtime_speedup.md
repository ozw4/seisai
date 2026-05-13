# Arakawa runtime speedup benchmark

This guide runs the same Arakawa physical-center export through a full baseline
and staged speedup settings, then compares runtime diagnostics and center
differences against A0.

Run commands from the repository root:

```bash
cd /workspace
```

The example configs are under:

```text
proc/arakawa/experiments/runtime_speedup/configs/
```

Edit `paths.sgy_file` and `paths.sgy_dir` in the configs if the SEG-Y file is
not `fdata_hset_ARA26_Vib.sgy` under
`/home/dcuser/data/ActiveSeisField/Arakawa2026`.

## Stages

| stage | purpose |
|---|---|
| `A0_full` | Full physics fit baseline with runtime diagnostics enabled. |
| `A0D_downsample_only` | Full fitting with anchor reuse disabled and offset-bin observation downsampling enabled. |
| `A1_diagnostics_only` | Same settings as A0, used to check instrumentation and run-to-run stability. |
| `A2_anchor_selection_dry_run` | Still performs full fitting, but records source-XY anchor selection diagnostics. |
| `A3_anchor_stride5_nearest_anchor` | Fits every 5th source group and reuses the nearest compatible anchor for non-anchor groups. |
| `A4_anchor_stride5_t0_shift` | Adds a median t0 shift when reusing nearest anchors. |
| `A5_anchor_stride5_t0_shift_adaptive_refit` | Adds adaptive full refit for reused groups whose residual or shift checks exceed thresholds. |
| `A6_A5_obs_downsample256` | Adds offset-bin observation downsampling with at most 256 observations per fit. |

A0D and A1 through A6 point `paths.coarse_dir` at `A0_full/coarse` so the
speedup comparison focuses on the physics stage. Run A0 first so the shared
coarse NPZ is present.

## Run each stage

```bash
python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/experiments/runtime_speedup/configs/A0_full.yaml

python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/experiments/runtime_speedup/configs/A0D_downsample_only.yaml

python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/experiments/runtime_speedup/configs/A1_diagnostics_only.yaml

python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/experiments/runtime_speedup/configs/A2_anchor_selection_dry_run.yaml

python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/experiments/runtime_speedup/configs/A3_anchor_stride5_nearest_anchor.yaml

python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/experiments/runtime_speedup/configs/A4_anchor_stride5_t0_shift.yaml

python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/experiments/runtime_speedup/configs/A5_anchor_stride5_t0_shift_adaptive_refit.yaml

python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/experiments/runtime_speedup/configs/A6_A5_obs_downsample256.yaml
```

Each config writes outputs below:

```text
proc/arakawa/outputs/runtime_runs/<STAGE>/
```

The key files are:

```text
proc/arakawa/outputs/runtime_runs/<STAGE>/robust/<TAG>.robust.npz
proc/arakawa/outputs/runtime_runs/<STAGE>/robust/<TAG>.physics_runtime_summary.json
proc/arakawa/outputs/runtime_runs/<STAGE>/grstat/<TAG>.physical_center.snap_peak.ltcor2.npz
```

For the default example data, `<TAG>` is:

```text
Arakawa2026__fdata_hset_ARA26_Vib
```

## Compare against A0

The baseline-diff comparison does not require an FB file or reference grstat.
It compares robust NPZ fields and, when present, the runtime summary JSON next
to each robust NPZ.

Example for A3:

```bash
TAG=Arakawa2026__fdata_hset_ARA26_Vib
mkdir -p proc/arakawa/outputs/runtime_compare

python -m cli.compare_fbpick_physics_runtime \
  --baseline proc/arakawa/outputs/runtime_runs/A0_full/robust/${TAG}.robust.npz \
  --candidate proc/arakawa/outputs/runtime_runs/A3_anchor_stride5_nearest_anchor/robust/${TAG}.robust.npz \
  --out-json proc/arakawa/outputs/runtime_compare/A3_vs_A0.json \
  --out-csv proc/arakawa/outputs/runtime_compare/A3_vs_A0.csv
```

Compare every non-baseline stage:

```bash
TAG=Arakawa2026__fdata_hset_ARA26_Vib
mkdir -p proc/arakawa/outputs/runtime_compare

for RUN in \
  A0D_downsample_only \
  A1_diagnostics_only \
  A2_anchor_selection_dry_run \
  A3_anchor_stride5_nearest_anchor \
  A4_anchor_stride5_t0_shift \
  A5_anchor_stride5_t0_shift_adaptive_refit \
  A6_A5_obs_downsample256
do
  python -m cli.compare_fbpick_physics_runtime \
    --baseline proc/arakawa/outputs/runtime_runs/A0_full/robust/${TAG}.robust.npz \
    --candidate proc/arakawa/outputs/runtime_runs/${RUN}/robust/${TAG}.robust.npz \
    --out-json proc/arakawa/outputs/runtime_compare/${RUN}_vs_A0.json \
    --out-csv proc/arakawa/outputs/runtime_compare/${RUN}_vs_A0.csv
done
```

To include post-export snap differences, pass both export NPZs:

```bash
python -m cli.compare_fbpick_physics_runtime \
  --baseline proc/arakawa/outputs/runtime_runs/A0_full/robust/${TAG}.robust.npz \
  --candidate proc/arakawa/outputs/runtime_runs/A3_anchor_stride5_nearest_anchor/robust/${TAG}.robust.npz \
  --baseline-export proc/arakawa/outputs/runtime_runs/A0_full/grstat/${TAG}.physical_center.snap_peak.ltcor2.npz \
  --candidate-export proc/arakawa/outputs/runtime_runs/A3_anchor_stride5_nearest_anchor/grstat/${TAG}.physical_center.snap_peak.ltcor2.npz \
  --out-json proc/arakawa/outputs/runtime_compare/A3_vs_A0_with_export.json \
  --out-csv proc/arakawa/outputs/runtime_compare/A3_vs_A0_with_export.csv
```

## Interpreting metrics

Runtime fields live under the `runtime` group in the compare JSON and CSV.

- `physics_total_sec`: total physics-stage runtime. Lower is faster.
- `speedup_physics_total`: `A0 physics_total_sec / candidate physics_total_sec`.
  Values above `1.0` are faster than A0.
- `ransac_fit_total_sec`: time spent inside RANSAC fits. This should drop when
  anchor reuse or observation downsampling is effective.
- `n_fit_calls`: number of physical model fit calls.
- `fit_call_reduction_rate`: `(A0 n_fit_calls - candidate n_fit_calls) / A0 n_fit_calls`.
- `cache_hit_rate`: fraction of repeated fit contexts served from cache.
- `observation_sampling_enabled`, `max_obs_per_fit`, `obs_count_before_*`,
  and `obs_count_after_*`: confirm that A6 or A0D actually downsampled
  observations.

Center-diff fields live under `physical_center_i_diff` and `fine_center_i_diff`
in the compare CSV, and under `center_diffs` in the JSON.

- `abs_diff_p90_samples` and `abs_diff_p99_samples`: tail movement versus A0.
  Small values mean the speedup preserved the baseline center for most traces.
- `within_4_sample_rate`: fraction of valid traces within 4 samples of A0.
- `within_16_sample_rate`: broader tolerance check for large but often still
  bounded shifts.
- `bias_mean_samples`: signed mean movement. A large magnitude can indicate a
  systematic shift rather than isolated outliers.

Status counts live under `status_counts`. Check both `counts_match` and the
per-label counts. A speedup can be fast but still change model-status outcomes;
that is a quality-risk signal and should be reviewed with center diffs.

## Report template

| run_name | physics_total_sec | speedup_vs_A0 | ransac_fit_total_sec | n_fit_calls | fit_call_reduction_rate | cache_hit_rate | physical_center_diff_p90 | physical_center_diff_p99 | fine_center_diff_p90 | fine_center_diff_p99 | within_4_samples_rate | within_16_samples_rate | status_counts | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| A0_full |  | 1.00 |  |  |  |  | 0 | 0 | 0 | 0 | 1.00 | 1.00 |  | baseline |
| A0D_downsample_only |  |  |  |  |  |  |  |  |  |  |  |  |  | downsampling only |
| A1_diagnostics_only |  |  |  |  |  |  |  |  |  |  |  |  |  | stability repeat |
| A2_anchor_selection_dry_run |  |  |  |  |  |  |  |  |  |  |  |  |  | anchor diagnostics only |
| A3_anchor_stride5_nearest_anchor |  |  |  |  |  |  |  |  |  |  |  |  |  | nearest anchor reuse |
| A4_anchor_stride5_t0_shift |  |  |  |  |  |  |  |  |  |  |  |  |  | t0 shift |
| A5_anchor_stride5_t0_shift_adaptive_refit |  |  |  |  |  |  |  |  |  |  |  |  |  | adaptive refit |
| A6_A5_obs_downsample256 |  |  |  |  |  |  |  |  |  |  |  |  |  | observation downsampling |

## QC PNG inspection

If a candidate has large p90, p99, max, or low within-threshold rates, inspect
QC PNGs before treating the run as acceptable. The benchmark configs keep
`visualization.enabled: false` so runtime measurements are not mixed with PNG
generation time.

For a QC pass, enable visualization for the candidate run:

```yaml
visualization:
  enabled: true
  allow_no_fb: true
  out_dir: ../../../outputs/runtime_runs/A3_anchor_stride5_nearest_anchor/qc
  max_gathers_per_file: 10
  gather_selection: even
  first_panel_only: true
  first_panel_flatten:
    enabled: true
    reference_key: physical_center_i
    half_samples: 256
```

Then rerun the candidate config and inspect:

```text
proc/arakawa/outputs/runtime_runs/<STAGE>/qc/<TAG>/gather_*.png
proc/arakawa/outputs/runtime_runs/<STAGE>/qc/summary_global.csv
proc/arakawa/outputs/runtime_runs/<STAGE>/qc/summary_per_file.csv
```

FB is not required for this QC mode when `allow_no_fb: true`; the runner creates
a dummy FB array for visualization.
